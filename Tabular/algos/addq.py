from typing import Dict, Any, Union, List, Tuple
import numpy as np
from copy import deepcopy

import envs
from algos.algo import Algo
from algos.policy import Policy, BasePolicy
from utils import schedule, check_for_schedule_allowed, sample_from_dist

class ADDQ(Algo):
    def __init__(self,
                 env: envs.Env = None, # The environment you want to execute the categorical Q Learning algorithm on
                 env_kwargs: Dict = None, # The arguments to be passed on the environment
                 policy: Policy = None, # The policy to be used
                 policy_kwargs: Dict = None, # The arguments to be passed on the policy
                 num_atoms: int = 51, # Number of atoms to be used
                 range_atoms: List[Union[int,float]] = None, # Range of values where the atoms should be distributed on
                 learning_rate_kwargs: Dict[str,Any] = None, # The keyword arguments for handling the step sizes of the updates
                 learning_rate_state_action_wise: bool = True, # Should the schedule for the step sizes be applied statewise
                 gamma: Union[int,float] = 0.99, # Discount factor for the game
                 interpolation_mode: str = "constant", # Mode for the interpolation of Q and Double Q
                 interpolation_mode_kwargs: Dict[str,Any] = None, # Necessary keyword arguments for interpolation of Q and Double Q
                 atom_probs_manual_init: bool = False, # Should the Q Functions be initialized manually
                 initial_atom_probs: Dict = None, # Initial probability for atoms
                 special_logs_kwargs: Dict = None, # Keyword arguments for logging special metrics
                 rng_seed: int = 42, # Seed for random number generator
                 checks: str = "all_checks", # Which checks should be performed
                 ) -> None:
        
        """
        Initializes the Categorical Adaptive Double Q Learning algorithm. The environment and policy along with their 
        arguments are used to model the Markov decision process. The learning rate schedule mode and if it should be 
        applied statewise or not is passed. Additionally, the interpolation mode is chosen. Optionally, a manual 
        initialization of the atom probabilities is possible.

        Parameters:
        - env (Env): The environment that is used.
        - env_kwargs (Dict): The keyword arguments with which the environment should be initialized. Should not contain
          rng_seed.
        - policy (Policy): The Policy type that is used.
        - policy_kwargs (Dict): The keyword argument with which the policy should be initialized. Should not contain 
          rng_seed.
        - num_atoms (int): The number of atoms used for approximating the distributions.
        - range_atoms (List): A list containing the lower and upper limit for the atom placement.
        - learning_rate_kwargs (Dict): The keyword arguments used to specify the update schedule for the learning rates.
        - learning_rate_state_action_wise (bool): Should the learning rate be updated statewise or not?
        - gamma (int,float): The discount factor that should be applied in the value function.
        - interpolation_mode (str): The mode for interpolation to be used during the algorithm. One can choose between a 
          constant choice of interpolation factor or an adaptive one based on the distribution.
        - interpolation_mode_kwargs (dict): The keyword arguments used to execute the chosen interpolation mode.
        - atom_probs_manual_init (bool): Should the probabilities on the atoms be initialized manually or not?
        - initial_atom_probs (Dict): The initial probabilities on the atoms to be initialized in case manual initialization 
          was chosen.
        - special_logs_kwargs (Dict): The keyword arguments for logging special metrics.
        - rng_seed (int): Seed for the random number generator. Default is 42.
        - checks (str): Checking mode. "all_checks", "only_initial_checks", and "no_checks" is available.
        """

        # Default arguments if arguments are None
        if env is None:
            env = envs.GridWorld
        if env_kwargs is None:
            env_kwargs = {}
        if policy is None:
            policy = BasePolicy
        if policy_kwargs is None:
            policy_kwargs = {}
        if range_atoms is None:
            range_atoms = [-10,10]
        if learning_rate_kwargs is None:
            learning_rate_kwargs = { 
                     "initial_rate": 1,
                     "mode": "rate", 
                     "mode_kwargs": {"rate_fct": lambda n: 1 / n, "iteration_num":1, "final_rate": 0},}
        if interpolation_mode_kwargs is None:
            interpolation_mode_kwargs = {"beta": 0.5}
        if initial_atom_probs is None:
            initial_atom_probs = {}
        if special_logs_kwargs is None:
            special_logs_kwargs = {}

        # Standard initialization of arguments from the input
        self.env = deepcopy(env) # Gets modified inside the class.
        self.env_kwargs = env_kwargs # Not modified inside the class
        self.policy = deepcopy(policy) # Gets modified inside the class
        self.policy_kwargs = policy_kwargs # Not modified inside the class
        self.num_atoms = num_atoms
        self.range_atoms = range_atoms # Not modified inside the class
        self.learning_rate_kwargs = deepcopy(learning_rate_kwargs) # Gets modified inside the class
        self.learning_rate_state_action_wise = learning_rate_state_action_wise
        self.gamma = gamma
        self.interpolation_mode = interpolation_mode
        self.interpolation_mode_kwargs = interpolation_mode_kwargs # Not modified inside the class
        self.atom_probs_manual_init = atom_probs_manual_init
        self.rng_seed = rng_seed
        self.initial_atom_probs = deepcopy(initial_atom_probs) # In case it gets modified inside the class
        self.special_logs_kwargs = special_logs_kwargs # Not modified inside the class
        if isinstance(checks,str):
            if not (checks == "all_checks" or checks == "no_checks" or checks == "only_initial_checks"):
                raise ValueError("Checks needs to be either all_checks, no_checks, or only_initial_checks")
        else:
            raise TypeError("Flag for type and value checking must be a string!")
        self.checks = checks 

        # Initializations for checking input constraints
        self.allowed_interpolation_modes = ["constant", "adaptive"]
        self.allowed_interpolation_kwargs = {
            "constant": ["beta"],
            "adaptive": ["center", "left_truncated", "bounds", "betas", "which", "current_state"]
        }
        self.allowed_special_logs_kwargs_keys = ["which_sample_variances"]
                
        # Checking input constraints
        if self.checks != "no_checks":
            self.inputcheck()

        # Advanced initializations

        # Initialize the environment and policy with the provided parameters if possible
        try:
            self.env = self.env(rng_seed = self.rng_seed, **self.env_kwargs)
        except (TypeError,ValueError) as e:
            print("The environment kwargs you provided seem to be faulty. The environment was instead created with default parameters!")
            print(f"Error message: {e}")
            self.env = self.env()
        try:
            self.policy = self.policy(rng_seed = self.rng_seed, env_allowed_actions = self.env.allowed_actions, env_num_states = self.env.num_states, env_num_actions = self.env.num_actions, **self.policy_kwargs)
        except (TypeError,ValueError) as e:
            self.policy = self.policy(env_allowed_actions = self.env.allowed_actions, env_num_states = self.env.num_states, env_num_actions = self.env.num_actions)
            print("The policy kwargs you provided seem to be faulty. The policy was instead created with default parameters!")
            print(f"Error message: {e}")

        # Initialize a rng for choosing which atom net to update in each step
        self.rng = np.random.default_rng(seed = self.rng_seed)

        # Initialize locations of the atoms and distance in-between
        self.atom_delta = (self.range_atoms[1] -  self.range_atoms[0]) / (self.num_atoms - 1)
        self.atom_locs = [self.range_atoms[0] + self.atom_delta * i for i in range(self.num_atoms)]

        # Initialize the atom probabilities, check if manual initialization conforms to the shape of the environment
        if self.atom_probs_manual_init:
            if self.checks != "no_checks":
                self.length_Q = 0
                for state in range(self.env.num_states):
                    self.length_Q += len(self.env.allowed_actions[state])
                if isinstance(self.initial_atom_probs,dict):
                    if len(self.initial_atom_probs) == self.length_Q:
                        for key in self.initial_atom_probs.keys():
                            if isinstance(key,tuple):
                                if len(key) == 2:
                                    if isinstance(key[0],int) and isinstance(key[1],int):
                                        if key[1] in self.env.allowed_actions[key[0]]:
                                            if isinstance(self.initial_atom_probs[key],list):
                                                if len(self.initial_atom_probs[key]) == self.num_atoms:
                                                    for value in self.initial_atom_probs[key]:
                                                        if isinstance(value,(int,float)):
                                                            if not (0 <= value <= 1):
                                                                raise ValueError(f"The given atom probability value of {value} in state-action pair {key} is not withing the range for probabilities!")
                                                        else:
                                                            raise TypeError(f"The given atom probability values for state-action pair {key} need to be numerical!")
                                                    if sum(self.initial_atom_probs[key]) != 1:
                                                        raise ValueError(f"The given atom probability values for state-action pair {key} need to sum to one!")
                                                else: 
                                                    raise ValueError(f"The given number of atom probability values for state-action pair {key} needs to correspond to the number of atoms!")
                                            else:
                                                raise TypeError(f"The given atom probability values for state-action pair {key} need to be given in a list!")
                                        else:
                                            raise ValueError(f"Action {key[1]} not allowed in state {key[0]}!")
                                    else:
                                        raise TypeError("State and actions in the keys of the given dictionary of atom probabilities need to be integers!")
                                else:
                                    raise ValueError("Keys of the given dictionary of atom probabilities need to be state action tuples of length 2!")
                            else:
                                raise TypeError("Keys of the given dictionary of atom probabilities need to be state action tuples!")
                    else:
                        raise TypeError("The given dictionary of atom probabilities misses some entries!")
                else:
                    for i in range(2):
                        if len(self.initial_atom_probs[i]) == self.length_Q:
                            for key in self.initial_atom_probs[i].keys():
                                if isinstance(key,tuple):
                                    if len(key) == 2:
                                        if isinstance(key[0],int) and isinstance(key[1],int):
                                            if key[1] in self.env.allowed_actions[key[0]]:
                                                if isinstance(self.initial_atom_probs[i][key],list):
                                                    if len(self.initial_atom_probs[i][key]) == self.num_atoms:
                                                        for value in self.initial_atom_probs[i][key]:
                                                            if isinstance(value,(int,float)):
                                                                if not (0 <= value <= 1):
                                                                    raise ValueError(f"The given atom probability value of {value} in state-action pair {key} is not withing the range for probabilities!")
                                                            else:
                                                                raise TypeError(f"The given atom probability values for state-action pair {key} need to be numerical!")
                                                        if sum(self.initial_atom_probs[i][key]) != 1:
                                                            raise ValueError(f"The given atom probability values for state-action pair {key} need to sum to one!")
                                                    else: 
                                                        raise ValueError(f"The given number of atom probability values for state-action pair {key} needs to correspond to the number of atoms!")
                                                else:
                                                    raise TypeError(f"The given atom probability values for state-action pair {key} need to be given in a list!")
                                            else:
                                                raise ValueError(f"Action {key[1]} not allowed in state {key[0]}!")
                                        else:
                                            raise TypeError("State and actions in the keys of the given dictionary of atom probabilities need to be integers!")
                                    else:
                                        raise ValueError("Keys of the given dictionary of atom probabilities need to be state action tuples of length 2!")
                                else:
                                    raise TypeError("Keys of the given dictionary of atom probabilities need to be state action tuples!")
                        else:
                            raise TypeError("The given dictionary of atom probabilities misses some entries!")
            if isinstance(self.initial_atom_probs,dict):
                self.atom_probs = [deepcopy(self.initial_atom_probs),deepcopy(self.initial_atom_probs)]
            else:
                self.atom_probs = deepcopy(self.initial_atom_probs)
        else:
            self.atom_probs = [{(state,action): [0 for _ in range(self.num_atoms)] for state in range(self.env.num_states) for action in self.env.allowed_actions[state]} for _ in range(2)]
            for i in range(2):
                for sa in self.atom_probs[i].keys():
                    self.atom_probs[i][sa][self.num_atoms // 2] = 1

        # Start at the beginning of the game
        self.current_state = self.env.start_state_num

        # If learning rate statewise is activated, store the current learning rates
        if self.learning_rate_state_action_wise:
            self.learning_rate_schedule_list = ([{action : self.learning_rate_kwargs["initial_rate"] for action in self.env.allowed_actions[state]} for state in range(self.env.num_states)],[{action : self.learning_rate_kwargs["initial_rate"] for action in self.env.allowed_actions[state]} for state in range(self.env.num_states)])
            if self.learning_rate_kwargs["mode"] == "rate":
                self.learning_rate_iteration_num_list = ([{action: 1 for action in self.env.allowed_actions[state]} for state in range(self.env.num_states)],[{action: 1 for action in self.env.allowed_actions[state]} for state in range(self.env.num_states)])
        
        # Initialize the Q functions
        self.q_fct = [{(state,action):0 for state in range(self.env.num_states) for action in self.env.allowed_actions[state]} for _ in range(2)]

    def __str__(self):
        return "ADDQ"

    def step(self) -> Tuple[int,int,int,float,bool]:

        """
        Exercises a step of the algorithm with the baseparameters. Thereby, for exploration, the average of both
        probability nets is taken if the policy requires access to a Q Function.

        Returns:
        - state (int): The state on which the algorithm was updated.
        - chosen_action (int): The action chosen during exploration in one step of the algorithm.
        - next_state (int): The resulting next state after playing the chosen action.
        - reward (float): The reward obtained by executing one step of the algorithm.
        - restart (bool): If a terminal state was reached and the game will be restarted.
        """
        
        # Save current state
        state = self.current_state

        # Update the Q function according to the current atom probabilities
        self.update_Q_from_atoms()

        # Average the Q functions for exploration in case we need a Q function
        average_q_fct = {}
        for key in self.q_fct[0].keys():
            average_q_fct[key] = (self.q_fct[0][key]+ self.q_fct[1][key]) / 2

        # Choose next action according to policy
        chosen_action = self.policy.choose_next_action(self.current_state,average_q_fct)

        # Sample reward and next state given chosen action
        next_state, t, reward = self.env.get_next_state_and_reward(self.current_state,chosen_action)

        # Choose which probability net to update
        index = int(self.rng.integers(0,2))
        other_index = (index + 1) % 2

        # Choose next stepsize according to the schedule and update the schedule afterwards
        if self.learning_rate_state_action_wise:
            stepsize = self.learning_rate_schedule_list[index][self.current_state][chosen_action]
            if stepsize > self.learning_rate_kwargs["mode_kwargs"]["final_rate"]:
                if self.learning_rate_kwargs["mode"] == "rate":
                    iteration_num = self.learning_rate_iteration_num_list[index][self.current_state][chosen_action]
                    self.learning_rate_kwargs["mode_kwargs"]["iteration_num"] = iteration_num
                self.learning_rate_kwargs["current_rate"] = stepsize
                self.learning_rate_kwargs = schedule(**self.learning_rate_kwargs)
                self.learning_rate_schedule_list[index][self.current_state][chosen_action] = self.learning_rate_kwargs["current_rate"]
                if self.learning_rate_kwargs["mode"] == "rate":
                    self.learning_rate_iteration_num_list[index][self.current_state][chosen_action] = self.learning_rate_kwargs["mode_kwargs"]["iteration_num"]
        else:
            stepsize = self.learning_rate_kwargs["current_rate"]
            if stepsize > self.learning_rate_kwargs["mode_kwargs"]["final_rate"]:
                self.learning_rate_kwargs = schedule(**self.learning_rate_kwargs)

        # Find action maximizing the Q Function of chosen index at the next state
        next_state_q_fct = {key: value for key, value in self.q_fct[index].items() if key[0] == next_state}
        max_val = max(next_state_q_fct.values())
        arg_max = [key[1] for key, value in next_state_q_fct.items() if value == max_val]
        if len(arg_max) == 1:
            max_action = arg_max[0]
        else:
            max_action = int(sample_from_dist(self.policy.rng,"choice",1,**{"a": arg_max, "p": [1/len(arg_max) for _ in arg_max]})[0])

        # Find betas according to chosen interpolation mode
        if self.interpolation_mode == "constant":
            beta = self.interpolation_mode_kwargs["beta"]

        elif self.interpolation_mode == "adaptive":

            # Which state`s and action`s data is relevant?
            relevant_s = self.current_state
            relevant_a = chosen_action
            if not self.interpolation_mode_kwargs["current_state"]:
                relevant_s = next_state
                relevant_a = max_action

            # If which is chosen, only use the chosen index net`s probabilities
            if self.interpolation_mode_kwargs["which"] == "chosen":

                # For the adaptive interpolation mode first find out the center values of the random variables
                if self.interpolation_mode_kwargs["center"] == "variance":
                    center_values = {}
                    for a in self.env.allowed_actions[relevant_s]:
                        center_values[a] = self.q_fct[index][(relevant_s,a)]
                elif self.interpolation_mode_kwargs["center"] == "median":
                    center_values = {}
                    for a in self.env.allowed_actions[relevant_s]:
                        p = 0
                        for i, probs in enumerate(self.atom_probs[index][(relevant_s,a)]):
                            p += probs
                            if p >= 0.5:
                                center_values[a] = self.atom_locs[i]
                                break
                else:
                    raise ValueError("The chosen center mode seems to not be valid. If you tried implementing a new one, you need to specify how to choose the center values based on the mode!")
                
                # Next, find out the indices from which on the variance calculations should be done
                if self.interpolation_mode_kwargs["left_truncated"]:
                    start_indices = {}
                    for a in self.env.allowed_actions[relevant_s]:
                        for i, loc in enumerate(self.atom_locs):
                            if loc >= center_values[a]:
                                start_indices[a] = i
                                break
                else:
                    start_indices = {a: 0 for a in self.env.allowed_actions[relevant_s]}

                # Finally, calculate the chosen dispersal measure
                var_dict = {}
                for a in self.env.allowed_actions[relevant_s]:
                    disp = 0
                    for i in range(start_indices[a],len(self.atom_locs)):
                        disp += self.atom_probs[index][(relevant_s,a)][i] * ((center_values[a] - self.atom_locs[i]) ** 2)
                    var_dict[a] = disp
                
                # Compute the total dispersal
                total_var = sum(var_dict.values()) / len(var_dict)

            # If other is chosen, only use the not chosen index net`s probabilities
            elif self.interpolation_mode_kwargs["which"] == "other":
                
                # For the adaptive interpolation mode first find out the center values of the random variables
                if self.interpolation_mode_kwargs["center"] == "variance":
                    center_values = {}
                    for a in self.env.allowed_actions[relevant_s]:
                        center_values[a] = self.q_fct[other_index][(relevant_s,a)]
                elif self.interpolation_mode_kwargs["center"] == "median":
                    center_values = {}
                    for a in self.env.allowed_actions[relevant_s]:
                        p = 0
                        for i, probs in enumerate(self.atom_probs[other_index][(relevant_s,a)]):
                            p += probs
                            if p >= 0.5:
                                center_values[a] = self.atom_locs[i]
                                break
                else:
                    raise ValueError("The chosen center mode seems to not be valid. If you tried implementing a new one, you need to specify how to choose the center values based on the mode!")
                
                # Next, find out the indices from which on the variance calculations should be done
                if self.interpolation_mode_kwargs["left_truncated"]:
                    start_indices = {}
                    for a in self.env.allowed_actions[relevant_s]:
                        for i, loc in enumerate(self.atom_locs):
                            if loc >= center_values[a]:
                                start_indices[a] = i
                                break
                else:
                    start_indices = {a: 0 for a in self.env.allowed_actions[relevant_s]}

                # Finally, calculate the chosen dispersal measure
                var_dict = {}
                for a in self.env.allowed_actions[relevant_s]:
                    disp = 0
                    for i in range(start_indices[a],len(self.atom_locs)):
                        disp += self.atom_probs[other_index][(relevant_s,a)][i] * ((center_values[a] - self.atom_locs[i]) ** 2)
                    var_dict[a] = disp
                
                # Compute the total dispersal
                total_var = sum(var_dict.values()) / len(var_dict)

            # If average, use the average of both net`s probabilities
            elif self.interpolation_mode_kwargs["which"] == "average":

                # For the adaptive interpolation mode first find out the center values of the random variables
                if self.interpolation_mode_kwargs["center"] == "variance":
                    center_values = [{},{}]
                    for a in self.env.allowed_actions[relevant_s]:
                        center_values[0][a] = self.q_fct[0][(relevant_s,a)]
                        center_values[1][a] = self.q_fct[1][(relevant_s,a)]
                elif self.interpolation_mode_kwargs["center"] == "median":
                    center_values = [{},{}]
                    for a in self.env.allowed_actions[relevant_s]:
                        p0 = 0
                        p1 = 0
                        for i, probs in enumerate(self.atom_probs[0][(relevant_s,a)]):
                            p0 += probs
                            if p0 >= 0.5:
                                center_values[0][a] = self.atom_locs[i]
                                break
                        for i, probs in enumerate(self.atom_probs[1][(relevant_s,a)]):
                            p1 += probs
                            if p1 >= 0.5:
                                center_values[1][a] = self.atom_locs[i]
                                break
                else:
                    raise ValueError("The chosen center mode seems to not be valid. If you tried implementing a new one, you need to specify how to choose the center values based on the mode!")
                
                # Next, find out the indices from which on the variance calculations should be done
                if self.interpolation_mode_kwargs["left_truncated"]:
                    start_indices = [{},{}]
                    for a in self.env.allowed_actions[relevant_s]:
                        for i, loc in enumerate(self.atom_locs):
                            if loc >= center_values[0][a]:
                                start_indices[0][a] = i
                                break
                        for i, loc in enumerate(self.atom_locs):
                            if loc >= center_values[1][a]:
                                start_indices[1][a] = i
                                break
                else:
                    start_indices = [{a: 0 for a in self.env.allowed_actions[relevant_s]} for _ in range(2)]

                # Finally, calculate the chosen dispersal measure
                vars_temp = [{},{}]
                for a in self.env.allowed_actions[relevant_s]:
                    var0 = 0
                    var1 = 0
                    for i in range(start_indices[0][a],len(self.atom_locs)):
                        var0 += self.atom_probs[0][(relevant_s,a)][i] * ((center_values[0][a] - self.atom_locs[i]) ** 2)
                    for i in range(start_indices[1][a],len(self.atom_locs)):
                        var1 += self.atom_probs[1][(relevant_s,a)][i] * ((center_values[1][a] - self.atom_locs[i]) ** 2)
                    vars_temp[0][a] = var0
                    vars_temp[1][a] = var1

                # Compute the averages
                var_dict = {}
                for a in self.env.allowed_actions[relevant_s]:
                    var_dict[a] = (vars_temp[0][a] + vars_temp[1][a]) / 2
                
                # Compute the total dispersal
                total_var = sum(var_dict.values()) / len(var_dict)
            
            else:
                raise ValueError("The chosen mode for the which parameter seems to not be valid. If you tried implementing a new one, you need to specify how to get action-wise dispersal measures!")

            # Compute the beta_hat
            if total_var != 0:
                beta_hat = var_dict[relevant_a] / total_var
            else:
                beta_hat = self.interpolation_mode_kwargs["bounds"][-1][0] + 1

            # Check which beta needs to be set
            beta_index = 0
            for i, bound in enumerate(self.interpolation_mode_kwargs["bounds"]):
                if bound[1] == "strict":
                    if beta_hat < bound[0]:
                        beta_index = i
                        break
                elif beta_hat <= bound[0]:
                    beta_index = i
                    break
                beta_index += 1

            # Set the correct beta
            beta = self.interpolation_mode_kwargs["betas"][beta_index]

        else:
            raise ValueError("The chosen interpolation mode seems to not exist. If you tried to implement a new interpolation mode you need to specify how to determine the interpolation coefficient beta in the code!")

        # Initialize target atom probabilities
        target_atom_probs = [0 for _ in range(self.num_atoms)]

        # Do the algorithm key steps for all atoms, whereby the atom probabilities of the other net are used
        for atom_index in range(self.num_atoms):

            # Compute the target distribution atoms and clip them to fit the range
            target_atom = reward + (1 - t) * self.gamma * self.atom_locs[atom_index]
            target_atom = max(self.range_atoms[0],min(self.range_atoms[1],target_atom))

            # Distribute probability of target atom according to location to nearest neighboring support atoms
            target_atom_rel_index = (target_atom - self.range_atoms[0]) / self.atom_delta
            if target_atom_rel_index.is_integer():
                target_atom_probs[int(target_atom_rel_index)] += beta * self.atom_probs[index][(next_state,max_action)][atom_index] + (1 - beta) * self.atom_probs[other_index][(next_state,max_action)][atom_index]
            else:
                lower = int(np.floor(target_atom_rel_index))
                upper = int(np.ceil(target_atom_rel_index))
                target_atom_probs[lower] += (beta * self.atom_probs[index][(next_state,max_action)][atom_index] + (1 - beta) * self.atom_probs[other_index][(next_state,max_action)][atom_index]) * (upper - target_atom_rel_index)
                target_atom_probs[upper] += (beta * self.atom_probs[index][(next_state,max_action)][atom_index] + (1 - beta) * self.atom_probs[other_index][(next_state,max_action)][atom_index]) * (target_atom_rel_index - lower)

        # Refactor target_atom_probs in case of rounding errors
        sum_prob_vec = sum(target_atom_probs)
        target_atom_probs_norm = [val / sum_prob_vec for val in target_atom_probs]

        # Interpolate target probabilities with current estimates using stepsize
        self.atom_probs[index][(self.current_state,chosen_action)] = [(1 - self.learning_rate_kwargs["current_rate"]) * self.atom_probs[index][(self.current_state,chosen_action)][i] + self.learning_rate_kwargs["current_rate"] * target_atom_probs_norm[i] for i in range(self.num_atoms)]

        # Update current state
        self.current_state = next_state

        return state, chosen_action, next_state, reward, t

    def get_special_log_keys(self) -> List[Tuple[str,str]]:
        """Returns the log and plot names for all special plots and when to log them."""
        if "which_sample_variances" in self.special_logs_kwargs.keys():
            return [("at_eval",f"sample variances of state {state} and action {action}") for state_index, state in enumerate(self.special_logs_kwargs["which_sample_variances"][0]) for action in self.special_logs_kwargs["which_sample_variances"][1][state_index]]
        else:
            return []
    
    def get_special_logs_at_step(self) -> List[Tuple[str,Union[int,float]]]:
        return []
    
    def get_special_logs_at_epoch(self) -> List[Tuple[str,Union[int,float]]]:
        return []

    def get_special_logs_at_eval(self) -> List[Tuple[str,Union[int,float]]]:
        """Returns the logs of the sample variances at the chosen state action pairs."""
        # Initialize the list for returning the values to log
        to_log_list = []

        # For all state action pairs to be logged get the label and variance
        for state_index,state in enumerate(self.special_logs_kwargs["which_sample_variances"][0]):
            for action in self.special_logs_kwargs["which_sample_variances"][1][state_index]:

                # Get the label
                label = f"sample variances of state {state} and action {action}"

                # Update the q functions to get the correct means
                self.update_Q_from_atoms()

                # The mean values at the chosen state action pair are the Q functions' values
                mean0 = self.q_fct[0][(state,action)]
                mean1 = self.q_fct[1][(state,action)]

                # Get the variances
                var0 = 0
                var1 = 0
                for i in range(len(self.atom_locs)):
                    var0 += self.atom_probs[0][(state,action)][i] * ((mean0 - self.atom_locs[i]) ** 2)
                    var1 += self.atom_probs[1][(state,action)][i] * ((mean1 - self.atom_locs[i]) ** 2)

                # Average the variances
                var = (var0 + var1) / 2

                to_log_list.append((label,var))
        
        return to_log_list
            

    def get_greedy_policy(self) -> List[int]:

        """
        Takes the current atom probabilities and locations to update the Q functions, averages them, and gives out 
        a policy list corresponding to the greedy policy.

        Returns:
        - list: The list of greedy actions with respect to the current Q function.
        """

        # Update the Q functions to have current estimate
        self.update_Q_from_atoms()

        # Average the Q functions
        average_q_fct = {}
        for key in self.q_fct[0].keys():
            average_q_fct[key] = (self.q_fct[0][key] + self.q_fct[1][key]) / 2

        # Compute greedy actions
        greedy_policy = []
        for state in range(self.env.num_states):
            state_q_fct = {key: value for key, value in average_q_fct.items() if key[0] == state}
            max_val = max(state_q_fct.values())
            arg_max = [key[1] for key, value in state_q_fct.items() if value == max_val]
            if len(arg_max) == 1:
                greedy_policy.append(arg_max[0])
            else:
                greedy_policy.append(int(sample_from_dist(self.policy.rng,"choice",1,**{"a": arg_max, "p": [1/len(arg_max) for _ in arg_max]})[0]))

        return greedy_policy
    
    def get_q_fct(self) -> Dict[Tuple[int,int],Union[int,float]]:

        """ Returns the current estimate of the Q function as an average over the estimated ones. """

        # Average the Q functions and return it as the estimated Q function
        average_q_fct = {}
        for key in self.q_fct[0].keys():
            average_q_fct[key] = (self.q_fct[0][key] + self.q_fct[1][key]) / 2

        return average_q_fct

    def update_Q_from_atoms(self) -> None:

        """ Updates the current Q functions according to the current probability estimates and atom locations. """

        for i in range(2):
            for key in self.q_fct[i]:
                self.q_fct[i][key] = float(sum(np.multiply(self.atom_locs,self.atom_probs[i][key])))

    def inputcheck(self) -> int:

        """
        Validates the input parameters to ensure they follow the expected formats and constraints.

        Raises:
        - ValueError: If any of the input parameters are invalid.
        - TypeError: If any of the input types are invalid.
        """

        # env is of type Env
        if not issubclass(self.env,envs.Env):
            raise TypeError("Environment needs to be of base type Env!")
        
        # env_kwargs is dictionary and does not contain rng_seed
        if isinstance(self.env_kwargs,dict):
            if "rng_seed" in self.env_kwargs.keys():
                raise ValueError("Environment key arguments should not contain rng_seed!")
        else:
            raise TypeError("Environment key arguments need to be contained in a dictionary!")
        
        # Policy is of type policy
        if not issubclass(self.policy,Policy):
            raise TypeError("Policy needs to be of the correct type!")
        
        # policy_kwargs is dictionary and does not contain rng_seed,env_allowed_actions, env_num_states, or env_num_actions as keys
        if isinstance(self.policy_kwargs,dict):
            if "rng_seed" in self.policy_kwargs.keys():
                raise ValueError("Policy key arguments should not contain rng_seed!")
            if "env_allowed_actions" in self.policy_kwargs.keys():
                raise ValueError("Policy key arguments should not contain env_allowed_actions!")
            if "env_num_state" in self.policy_kwargs.keys():
                raise ValueError("Policy key arguments should not contain env_num_state!")
            if "env_num_actions" in self.policy_kwargs.keys():
                raise ValueError("Policy key arguments should not contain env_num_actions!")
        else:
            raise TypeError("Policy key arguments need to be contained in a dictionary!")
        
        # num_atoms needs to be a positive integer
        if isinstance(self.num_atoms,int):
            if self.num_atoms <= 0:
                raise ValueError("The number of atoms needs to be a positive integer!")
        else:
            raise TypeError("The number of atoms needs to be an integer!")
        
        # range_atoms needs to be a list containing two numerical values, where the second needs to be bigger
        if isinstance(self.range_atoms,list):
            if len(self.range_atoms) == 2:
                if isinstance(self.range_atoms[0],(int,float)) and isinstance(self.range_atoms[1],(int,float)):
                    if self.range_atoms[0] >= self.range_atoms[1]:
                        raise ValueError("The first entry of the range for the atoms needs to be smaller than the second one!")
                else:
                    raise TypeError("The entries of the range of atoms list need to be numerical!")
            else:
                raise ValueError("The range for the atoms needs to be specified only by its lower and upper value!")
        else:
            raise TypeError("The range for the atoms needs to be contained in a list!")

        # learning_rate_kwargs is dictionary and is allowed for scheduling
        if isinstance(self.learning_rate_kwargs,dict):
            if "initial_rate" in self.learning_rate_kwargs.keys():
                self.learning_rate_kwargs["current_rate"] = self.learning_rate_kwargs["initial_rate"]
                self.learning_rate_kwargs["mode_kwargs"]["iteration_num"] = 1
                check_for_schedule_allowed(**self.learning_rate_kwargs)
            else:
                raise ValueError("Initial rate is missing from the learning rate key word arguments!")
        else:
            raise TypeError("Learning rate keyword arguments need to be contained in a dictionary!")
        
        # special_logs_kwargs is dictionary and is allowed for logging
        if isinstance(self.special_logs_kwargs,dict):
            for key in self.special_logs_kwargs.keys():
                if key in self.allowed_special_logs_kwargs_keys:
                    if key == "which_sample_variances":
                        if isinstance(self.special_logs_kwargs[key],tuple):
                            if len(self.special_logs_kwargs[key]) == 2:
                                if isinstance(self.special_logs_kwargs[key][0],list) and isinstance(self.special_logs_kwargs[key][1],list):
                                    if len(self.special_logs_kwargs[key][0]) == len(self.special_logs_kwargs[key][1]):
                                        for state in self.special_logs_kwargs[key][0]:
                                            if isinstance(state,int):
                                                if state < 0:
                                                    raise ValueError("One of the states you provided for logging the sample variance is not a positive integer!")
                                            else:
                                                raise ValueError("One of the states you provided for logging the sample variance is not a positive integer!")
                                        for act_list in self.special_logs_kwargs[key][1]:
                                            if isinstance(act_list,list):
                                                for act in act_list:
                                                    if isinstance(act,int):
                                                        if act < 0:
                                                            raise ValueError("One of the actions you provided for logging the sample variance is not a positive integer!")
                                                    else:
                                                        raise ValueError("One of the actions you provided for logging the sample variance is not a positive integer!")
                                            else:
                                                raise ValueError("For one of the states you provided for logging the sample variance the actions are not passed in a list!")
                                    else:
                                        raise ValueError("The number of states and number of corresponding action lists you provided for logging the sample variance does not match!")
                                else:
                                    raise TypeError("The states and corresponding actions for logging the sample variance need to be provided in lists!")
                            else:
                                raise TypeError("The states and corresponding actions for logging the sample variance need to be provided in two lists!")
                        else:
                            raise TypeError("The states and corresponding actions for logging the sample variance need to be provided in two lists inside a tuple!")
                    else:
                        raise TypeError("You used the wrong key for logging special parameters. If you tried implementing a new one try updating the inputcheck function!")
                else:
                    raise ValueError("Invalid key for logging special parameters. If you tried implementing a new one register it in the self.allowed_special_logs_kwargs_keys list!")
        else:
            raise TypeError("The special logs keyword arguments need to be passed in a dictionary!")

        # learning_rate_state_action_wise is bool
        if not isinstance(self.learning_rate_state_action_wise,bool):
            raise TypeError("Mode for statewise learning schedule needs to be a boolean value!")
        
        # Gamma is float or int between 0 and 1, not including 1
        if isinstance(self.gamma,(int,float)):
            if 0 <= self.gamma <= 1:
                if 0 == self.gamma:
                    print("Your discount factor is set to zero! Proceed with caution, your MDP problem might not make sense!")
                elif 1 == self.gamma:
                    print("Your discount factor is set to one! Proceed with caution, your MDP problem might not make sense!")
            else:
                raise ValueError("The discount factor for your game needs to be between 0 and 1!")
        else:
            raise TypeError("The discount factor needs to be a numerical value!")
        
        # interpolation_mode is allowed string
        if isinstance(self.interpolation_mode,str):
            if (not self.interpolation_mode in self.allowed_interpolation_modes) or (not self.interpolation_mode in self.allowed_interpolation_kwargs.keys()):
                raise ValueError("The interpolation mode you selected seems to not be implemented. If you tried implementing a new one, update the allowed interpolation modes and interpolation kwargs lists in the __init__ function!")
        else:
            raise TypeError("The interpolation mode needs to be given as a string!")
        
        # interpolation_mode_kwargs need to contain the necessary keywords
        if isinstance(self.interpolation_mode_kwargs,dict):
            for item in self.allowed_interpolation_kwargs[self.interpolation_mode]:
                if not (item in self.interpolation_mode_kwargs.keys()):
                    raise ValueError(f"The keyword argument {item} is missing for the interpolation mode {self.interpolation_mode}!")
            for it in self.interpolation_mode_kwargs.keys():
                if not (it in self.allowed_interpolation_kwargs[self.interpolation_mode]):
                    raise ValueError(f"The keyword argument {it} is wrong for the interpolation mode {self.interpolation_mode}!")
        else:
            raise TypeError("The keyword arguments for the interpolation mode need to be passed in a dictionary!")
        
        # check the values of the interpolation_mode_kwargs
        if self.interpolation_mode == "constant":
            if isinstance(self.interpolation_mode_kwargs["beta"],(int,float)):
                if not (0 <= self.interpolation_mode_kwargs["beta"] <= 1):
                    raise ValueError("The beta value for constant interpolation needs to be between zero and one!")
            else:
                raise TypeError("The beta value for constant interpolation needs to be a numerical value!")
        elif self.interpolation_mode == "adaptive":
            if not isinstance(self.interpolation_mode_kwargs["left_truncated"],bool):
                raise TypeError("If left truncation is activated or not for adaptive interpolation needs to be a boolean value!")
            if not isinstance(self.interpolation_mode_kwargs["current_state"],bool):
                raise TypeError("If the current state should be used for calculating the dispersion measure needs to be a boolean value!")
            if not (self.interpolation_mode_kwargs["which"] == "chosen" or self.interpolation_mode_kwargs["which"] == "other" or self.interpolation_mode_kwargs["which"] == "average"):
                raise TypeError("If the average of both atom nets should be used to calculate the betas or just the one from the chosen")
            if not (self.interpolation_mode_kwargs["center"] == "variance" or self.interpolation_mode_kwargs["center"] == "median"):
                raise ValueError("The center for the dispersion-based adaptive interpolation must be either variance or median!")
            if isinstance(self.interpolation_mode_kwargs["bounds"],list):
                for index, item in enumerate(self.interpolation_mode_kwargs["bounds"]):
                    if isinstance(item,tuple):
                        if len(item) == 2:
                            if isinstance(item[0],(int,float)):
                                if index == 0:
                                    if not (0 < item[0]):
                                        raise ValueError("The first bound for adaptive interpolation must be positive!")
                                else:
                                    if not (self.interpolation_mode_kwargs["bounds"][index - 1][0] < item[0]):
                                        raise ValueError("The bounds for adaptive interpolation must be strictly increasing inside the list!")
                            else:
                                raise TypeError("The bounds for adaptive interpolation must be of numerical value!")
                            if not (item[1] == "strict" or item[1] == "not_strict"):
                                raise ValueError("The bounds for adaptive interpolation must either have strict or not_strict assigned to them!")
                        else:
                            raise ValueError("The bounds for adaptive interpolation must be passed as tuples of length 2, containing the bound value and if a strict or a not_strict inequality should be applied!")
                    else:
                        raise TypeError("The bounds for adaptive interpolation must be passed as tuples containing the bound value and if a strict or a not_strict inequality should be applied!")
            else:
                raise TypeError("The bounds for adaptive interpolation must be passed in a list!")
            if isinstance(self.interpolation_mode_kwargs["betas"],list):
                for item in self.interpolation_mode_kwargs["betas"]:
                    if isinstance(item,(int,float)):
                        if not (0 <= item <= 1):
                            raise ValueError("The betas for adaptive interpolation must be between zero and one!")
                    else:
                        raise TypeError("The betas for adaptive interpolation must be of numerical value!")
            else:
                raise TypeError("The betas for adaptive interpolation must be passed in a list!")
            if not (len(self.interpolation_mode_kwargs["betas"]) == len(self.interpolation_mode_kwargs["bounds"]) + 1):
                raise ValueError("The betas need to be exactly one more than the bounds, since the first default bound is zero!")
        else:
            raise ValueError("If you tried implementing a new interpolation mode, please add a check for the respective values of the interpolation mode keyword arguments!")
        
        # atom_probs_manual_init of atoms needs to be boolean
        if not isinstance(self.atom_probs_manual_init,bool):
            raise TypeError("The variable atom_probs_manual_init needs to be boolean!")
        
        # If the atom probabilities will be initialized manually, the initial probabilities need to be contained in a dictionary or a list of two dictionaries
        if self.atom_probs_manual_init:
            if not (isinstance(self.initial_atom_probs,dict) or isinstance(self.initial_atom_probs,list)):
                raise TypeError("The probabilites for the atoms passed for manual initialization need to be contained in either a list if two dictionaries or as a single dictionary!")
            elif isinstance(self.initial_atom_probs,list):
                if len(self.initial_atom_probs) == 2:
                    if not (isinstance(self.initial_atom_probs[0],dict) and isinstance(self.initial_atom_probs[1],dict)):
                        raise TypeError("In case the atom probabilities should be initialized differently from one another they need to be passed in dictionaries each contained in the list!")
                else:
                    raise TypeError("In case the atom probabilities should be initialized differently from one another they need to both be passed in a list of length two!")
            
        # Seed is in valid range:
        if isinstance(self.rng_seed, int):
            if not (0 <= self.rng_seed < 2**32):
                raise ValueError("The provided seed is not in the range of acceptable integer seeds!")
        else:
            raise TypeError("The seed needs to be an integer!")
        
        return 1