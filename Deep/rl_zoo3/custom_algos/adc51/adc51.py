import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update

from rl_zoo3.custom_algos.adc51.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, C51Policy, CategoricalNetwork

SelfADC51 = TypeVar("SelfADC51", bound="ADC51")


class ADC51(OffPolicyAlgorithm):
    """
    Quantile Regression Deep Q-Network (QR-DQN)
    Paper: https://arxiv.org/abs/1710.10044
    Default hyperparameters are taken from the paper and are tuned for Atari games.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping (if None, no clipping)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    categorical_net: CategoricalNetwork
    categorical_net_target: CategoricalNetwork
    policy: C51Policy

    def __init__(
        self,
        policy: Union[str, Type[C51Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 5e-5,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 64,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.005,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.01,
        max_grad_norm: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        if "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.Adam
            # Proposed in the QR-DQN paper where `batch_size = 32`
            self.policy_kwargs["optimizer_kwargs"] = dict(eps=0.01 / batch_size)

        if _init_setup_model:
            self._setup_model()

        self.range_tensor = th.arange(self.n_atoms).repeat(batch_size, self.n_critics, self.action_space.n, 1).to(self.device)
        self.median_helper = th.full((batch_size, self.n_critics, self.action_space.n, 1), 0.5).to(self.device)

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
        self.batch_norm_stats = get_parameters_by_name(self.categorical_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.categorical_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

            self.target_update_interval = max(self.target_update_interval // self.n_envs, 1)

    def _create_aliases(self) -> None:
        self.categorical_net = self.policy.categorical_net
        self.categorical_net_target = self.policy.categorical_net_target
        self.n_atoms = self.policy.n_atoms
        self.support = self.policy.support
        self.n_critics = self.policy.n_critics

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(self.categorical_net.parameters(), self.categorical_net_target.parameters(), self.tau)
            # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for i in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # size: batch_size
            random_indices = th.randint(low=0, high=self.categorical_net.n_critics, size=(batch_size,))

            # Get current logits estimates
            # batch_size x n_critics x actions x n_atoms
            current_logits = self.categorical_net(replay_data.observations)

            # targets
            with th.no_grad():
                tiled_support = self.support.repeat(batch_size, 1)

                target_support = replay_data.rewards + (1 - replay_data.dones) * self.gamma * tiled_support

                # Compute the probs of next observation
                next_probs = self.categorical_net_target(replay_data.next_observations)
                next_probs = nn.functional.softmax(next_probs, dim=-1)  # maybe use log_softmax instead?

                # Compute the greedy actions which maximize the next Q values
                next_greedy_actions = (
                    (next_probs[th.arange(batch_size), random_indices] * self.support)
                    .sum(dim=-1, keepdim=True)
                    .argmax(dim=1, keepdim=True)
                    .unsqueeze(1)
                )
                # make copies and reshape
                next_greedy_actions = next_greedy_actions.expand(batch_size, self.n_critics, 1, self.n_atoms)
                # Follow greedy policy: use the one with the highest Q values
                next_probs = next_probs.gather(dim=2, index=next_greedy_actions).squeeze(2)

                current_probs = nn.functional.softmax(current_logits, dim=-1)

                current_cum_probs = th.cumsum(current_probs, dim=-1)

                median_indices = th.searchsorted(current_cum_probs, self.median_helper)

                repeated_support = self.support.repeat((batch_size, self.n_critics, self.action_space.n, 1))

                vs = (((repeated_support - repeated_support.gather(index=median_indices, dim=-1)) ** 2) * current_probs).sum(dim=-1).unsqueeze(-1) / (
                    1 - current_cum_probs.gather(dim=-1, index=median_indices - 1)
                )

                vs = vs.squeeze(-1).mean(dim=1)

                v_updated = vs.gather(dim=1, index=replay_data.actions)

                v_ratio = v_updated.squeeze(-1) / vs.mean(dim=-1)

                betas = th.where(v_ratio < 0.75, 0.75, 0.5)

                betas = th.where(v_ratio > 1.25, 0.25, betas).unsqueeze(1)

                next_probs = (
                    betas * next_probs[th.arange(batch_size), random_indices]
                    + (1 - betas) * next_probs[th.arange(batch_size), 1 - random_indices]
                )

                projected_target = self.project_distribution(target_support, next_probs, self.support)

            # Vectorized selection (select critics to be updated and actions)
            # batch_size x n_atoms
            selected_logits = current_logits[th.arange(batch_size), random_indices, replay_data.actions.squeeze()]

            selected_log_probs = nn.functional.log_softmax(selected_logits, dim=-1)

            # compute the cross-entropy loss
            loss = -(projected_target * selected_log_probs).sum(dim=-1).mean()
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def project_distribution(self, supports, weights, target_support):

        delta = target_support[1] - target_support[0]
        v_min, v_max = target_support[0], target_support[-1]
        clipped_support = th.clamp(supports, v_min, v_max).unsqueeze(1)

        tiled_support = clipped_support.repeat(1, 1, self.n_atoms, 1)

        batch_size = supports.size()[0]
        reshaped_target_support = target_support.unsqueeze(1).repeat(batch_size, 1)

        reshaped_target_support = reshaped_target_support.reshape(batch_size, self.n_atoms, 1)

        numerator = th.abs(tiled_support - reshaped_target_support)

        quotient = 1 - (numerator / delta)

        clipped_quotient = th.clamp(quotient, 0, 1)

        weights = weights.unsqueeze(1)

        inner_prod = clipped_quotient * weights

        projection = th.sum(inner_prod, dim=3)

        projection = projection.view(batch_size, self.n_atoms)

        return projection

    def learn(
        self: SelfADC51,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "ADC51",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfADC51:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["categorical_net", "categorical_net_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
