from typing import Any, Dict, List, Optional, Type

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)

from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device
from torch import nn


class CategoricalNetwork(BasePolicy):
    """
    Categorical network for C51

    :param observation_space: Observation space
    :param action_space: Action space
    :param n_atoms: Number of atoms
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        # features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        support,
        n_atoms: int = 51,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        features_extractor_class=FlattenExtractor,
    ):
        super().__init__(
            observation_space,
            action_space,
            # features_extractor=features_extractor,
            normalize_images=normalize_images,
            features_extractor_class=features_extractor_class,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.n_atoms = n_atoms
        action_dim = int(self.action_space.n)  # number of actions

        self.support = support

        features_extractor = self.make_features_extractor()
        qf_net_list = create_mlp(features_dim, action_dim * n_atoms, net_arch, activation_fn)
        qf_net = nn.Sequential(*qf_net_list)
        self.categorical_net = nn.Sequential(features_extractor, qf_net)

        # quantile_net = create_mlp(self.features_dim, action_dim * self.n_quantiles, self.net_arch, self.activation_fn)
        # self.quantile_net = nn.Sequential(*quantile_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the atoms probs.

        :param obs: Observation
        :return: The estimated probs for each action.
        """
        # features = self.extract_features(obs, self.features_extractor)

        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)

        logits = self.categorical_net(preprocessed_obs)

        return logits.view(-1, int(self.action_space.n), self.n_atoms)

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        logits = self(observation)
        probs = nn.functional.softmax(logits, dim=-1)

        q_values = (probs * self.support).sum(dim=-1)

        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                n_atoms=self.n_atoms,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class C51Policy(BasePolicy):
    """
    Policy class with quantile and target networks for QR-DQN.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param n_quantiles: Number of quantiles
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    categorical_net: CategoricalNetwork
    categorical_net_target: CategoricalNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        n_atoms: int = 51,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        v_max: float = 10,
        v_min: Optional[float] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.n_atoms = n_atoms
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        v_max = float(v_max)
        v_min = v_min if v_min else -v_max
        self.support = th.linspace(v_min, v_max, n_atoms).to(get_device("auto"))

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_atoms": self.n_atoms,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "support": self.support,
            "features_dim": 512,  # TODO: fix / make general
            "features_extractor_class": features_extractor_class,
        }
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.categorical_net = self.make_categorical_net()
        self.categorical_net_target = self.make_categorical_net()
        self.categorical_net_target.load_state_dict(self.categorical_net.state_dict())
        self.categorical_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_categorical_net(self) -> CategoricalNetwork:
        # Make sure we always have separate networks for features extractors etc
        # net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        net_args = self.net_args
        return CategoricalNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.categorical_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                n_atoms=self.net_args["n_atoms"],
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.categorical_net.set_training_mode(mode)
        self.training = mode


MlpPolicy = C51Policy


class CnnPolicy(C51Policy):
    """
    Policy class for QR-DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param n_atoms: Number of atoms
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        n_atoms: int = 51,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        v_max: float = 10,
        v_min: Optional[float] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            n_atoms,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            v_max,
            v_min,
        )


# TODO adapt
class MultiInputPolicy(C51Policy):
    """
    Policy class for QR-DQN when using dict observations as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param n_quantiles: Number of quantiles
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        n_quantiles: int = 200,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            n_quantiles,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
