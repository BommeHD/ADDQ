from typing import Dict, Any, Union, List,Tuple
import numpy as np
from copy import deepcopy

import envs
from algos.algo import Algo
from algos.policy import Policy, BasePolicy
from utils import schedule, check_for_schedule_allowed, sample_from_dist

class WDQ(Algo):
    def __init__(self,
                 env: envs.Env = None, # The environment you want to execute Q Learning on
                 env_kwargs: Dict = None, # The arguments to be passed on the environment
                 policy: Policy = None, # The policy to be used
                 policy_kwargs: Dict = None, # The arguments to be passed on the policy
                 learning_rate_kwargs: Dict[str,Any] = None, # The keyword arguments for handling the step sizes of the updates
                 learning_rate_state_action_wise: bool = True, # Should the schedule for the step sizes be applied statewise
                 gamma: Union[int,float] = 0.99, # Discount factor for the game
                 interpolation_mode: str = "constant", # Mode for the interpolation of Q and Double Q
                 interpolation_mode_kwargs: Dict[str,Any] = None, # Necessary keyword arguments for interpolation of Q and Double Q
                 q_fct_manual_init: bool = False, # Should the Q Functions be initialized manually
                 initial_q_fct: Dict = None, # Initial Q function values
                 special_logs_kwargs: Dict = None, # Keyword arguments for logging special metrics
                 rng_seed = 42, # Seed for random number generator
                 checks: str = "all_checks", # Which checks should be performed
                 ) -> None:
        
        """
        Initializes the Weighted Double Q Learning algorithm. The environment and policy along with their arguments are 
        used to model the Markov decision process. The learning rate schedule mode and if it should be applied statewise 
        or not is passed. Additionally, the interpolation mode is chosen. Optionally, a manual initialization of the Q 
        functions is possible.

        Parameters:
        - env (Env): The environment that is used.
        - env_kwargs (Dict): The keyword arguments with which the environment should be initialized. Should not contain
          rng_seed.
        - policy (Policy): The Policy type that is used.
        - policy_kwargs (Dict): The keyword argument with which the policy should be initialized. Should not contain 
          rng_seed.
        - learning_rate_kwargs (Dict): The keyword arguments used to specify the update schedule for the learning rates.
        - learning_rate_state_action_wise (bool): Should the learning rate be updated statewise or not?
        - gamma (int,float): The discount factor that should be applied in the value function.
        - interpolation_mode (str): The mode for interpolation to be used during the algorithm. One can choose between a 
          constant choice of interpolation factor or an adaptive one.
        - interpolation_mode_kwargs (dict): The keyword arguments used to execute the chosen interpolation mode.
        - q_fcts_manual_init (bool): Should the Q Functions be initialized manually or not?
        - initial_q_fcts (Dict): The initial Q Functions to be initialized in case manual initialization was chosen.
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
        if learning_rate_kwargs is None:
            learning_rate_kwargs = { 
                     "initial_rate": 1,
                     "mode": "rate", 
                     "mode_kwargs": {"rate_fct": lambda n: 1 / n, "iteration_num":1, "final_rate": 0},}
        if interpolation_mode_kwargs is None:
            interpolation_mode_kwargs = {"beta": 0.5}
        if initial_q_fct is None:
            initial_q_fct = {}
        if special_logs_kwargs is None:
            special_logs_kwargs = {}
        
        # Standard initialization of arguments from the input
        self.env = deepcopy(env) # Gets modified inside the class.
        self.env_kwargs = env_kwargs # Not modified inside the class
        self.policy = deepcopy(policy) # Gets modified inside the class
        self.policy_kwargs = policy_kwargs # Not modified inside the class
        self.learning_rate_kwargs = deepcopy(learning_rate_kwargs) # Gets modified inside the class
        self.learning_rate_state_action_wise = learning_rate_state_action_wise
        self.gamma = gamma
        self.interpolation_mode = interpolation_mode
        self.interpolation_mode_kwargs = interpolation_mode_kwargs # Not modified inside the class
        self.q_fct_manual_init = q_fct_manual_init
        self.rng_seed = rng_seed
        self.initial_q_fct = deepcopy(initial_q_fct) # In case it gets modified inside the class
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
            "adaptive": ["c", "current_state", "which_copy", "which_reference_arm"]
        }
        self.allowed_special_logs_kwargs_keys = []
                
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
            self.env = env()
        try:
            self.policy = self.policy(rng_seed = self.rng_seed, env_allowed_actions = self.env.allowed_actions, env_num_states = self.env.num_states, env_num_actions = self.env.num_actions, **self.policy_kwargs)
        except (TypeError,ValueError) as e:
            self.policy = self.policy(env_allowed_actions = self.env.allowed_actions, env_num_states = self.env.num_states, env_num_actions = self.env.num_actions)
            print("The policy kwargs you provided seem to be faulty. The policy was instead created with default parameters!")
            print(f"Error message: {e}")
        
        # Initialize a rng for choosing which Q Function to update in each step
        self.rng = np.random.default_rng(seed = self.rng_seed)

        # Initialize Q, check if manual initialization conforms to the shape of the environment
        if self.q_fct_manual_init:
            if self.checks != "no_checks":
                self.length_Q = 0
                for state in range(self.env.num_states):
                    self.length_Q += len(self.env.allowed_actions[state])
                if isinstance(self.initial_q_fct,dict):
                    if len(self.initial_q_fct) == self.length_Q:
                        for key in self.initial_q_fct.keys():
                            if isinstance(key,tuple):
                                if len(key) == 2:
                                    if isinstance(key[0],int) and isinstance(key[1],int):
                                        if key[1] in self.env.allowed_actions[key[0]]:
                                            if not isinstance(self.initial_q_fct[key],(int,float)):
                                                raise ValueError("The given Q function values need to be numerical!")
                                        else:
                                            raise ValueError(f"Action {key[1]} not allowed in state {key[0]}!")
                                    else:
                                        raise TypeError("State and actions in the keys of the given Q function need to be integers!")
                                else:
                                    raise ValueError("Keys of the given Q function need to be state action tuples of length 2!")
                            else:
                                raise TypeError("Keys of the given Q function need to be state action tuples!")
                    else:
                        raise TypeError("The given Q function misses some entries!")
                else:
                    for i in range(2):
                        if len(self.initial_q_fct[i]) == self.length_Q:
                            for key in self.initial_q_fct[i].keys():
                                if isinstance(key,tuple):
                                    if len(key) == 2:
                                        if isinstance(key[0],int) and isinstance(key[1],int):
                                            if key[1] in self.env.allowed_actions[key[0]]:
                                                if not isinstance(self.initial_q_fct[i][key],(int,float)):
                                                    raise ValueError("The given Q function values need to be numerical!")
                                            else:
                                                raise ValueError(f"Action {key[1]} not allowed in state {key[0]}!")
                                        else:
                                            raise TypeError("State and actions in the keys of the given Q function need to be integers!")
                                    else:
                                        raise ValueError("Keys of the given Q function need to be state action tuples of length 2!")
                                else:
                                    raise TypeError("Keys of the given Q function need to be state action tuples!")
                        else:
                            raise TypeError("The given Q function misses some entries!")
            if isinstance(self.initial_q_fct,dict):
                self.q_fct = [deepcopy(self.initial_q_fct), deepcopy(self.initial_q_fct)]
            else:
                self.q_fct = deepcopy(self.initial_q_fct)
        else:
            self.q_fct = [{(state,action):0 for state in range(self.env.num_states) for action in self.env.allowed_actions[state]} for _ in range(2)]
        
        # Start at the beginning of the game
        self.current_state = self.env.start_state_num

        # If learning rate statewise is activated, store the current learning rates
        if self.learning_rate_state_action_wise:
            self.learning_rate_schedule_list = ([{action : self.learning_rate_kwargs["initial_rate"] for action in self.env.allowed_actions[state]} for state in range(self.env.num_states)],[{action : self.learning_rate_kwargs["initial_rate"] for action in self.env.allowed_actions[state]} for state in range(self.env.num_states)])
            if self.learning_rate_kwargs["mode"] == "rate":
                self.learning_rate_iteration_num_list = ([{action: 1 for action in self.env.allowed_actions[state]} for state in range(self.env.num_states)],[{action: 1 for action in self.env.allowed_actions[state]} for state in range(self.env.num_states)])

    def __str__(self):
        return "WDQ"

    def step(self) -> Tuple[int,int,int,float,bool]:

        """
        Exercises a step of the algorithm with the baseparameters. Thereby, for exploration, the average of both
        Q functions is taken if the policy requires access to a Q Function.

        Returns:
        - state (int): The state on which the algorithm was updated.
        - action (int): The action chosen during exploration in one step of the algorithm.
        - next_state (int): The resulting next state after playing the chosen action.
        - reward (float): The reward obtained by executing one step of the algorithm.
        - restart (bool): If a terminal state was reached and the game will be restarted.
        """

        # Save current state
        state = self.current_state

        # Average the Q functions for exploration in case we need a Q function
        average_q_fct = {}
        for key in self.q_fct[0].keys():
            average_q_fct[key] = (self.q_fct[0][key] + self.q_fct[1][key]) / 2

        # Choose next action according to policy
        chosen_action = self.policy.choose_next_action(self.current_state,average_q_fct)

        # Sample reward and next state given chosen action
        next_state, t, reward = self.env.get_next_state_and_reward(self.current_state,chosen_action)

        # Choose which Q Function to update
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
            max_act = arg_max[0]
        else:
            max_act = int(sample_from_dist(self.policy.rng,"choice",1,**{"a": arg_max, "p": [1/len(arg_max) for _ in arg_max]})[0])

        beta = 0

        # Find betas according to chosen interpolation mode
        if self.interpolation_mode == "constant":
            beta = self.interpolation_mode_kwargs["beta"]

        elif self.interpolation_mode == "adaptive":

            # Determine if current or future state are to be used, max action always from index arm
            if self.interpolation_mode_kwargs["current_state"]:
                s = self.current_state
                this_state_q_fct = {k: v for k, v in self.q_fct[index].items() if k[0] == s}
                m_val = max(this_state_q_fct.values())
                arg_m = [ke[1] for ke, va in this_state_q_fct.items() if va == m_val]
                if len(arg_m) == 1:
                    max_a = arg_m[0]
                else:
                    max_a = int(sample_from_dist(self.policy.rng,"choice",1,**{"a": arg_m, "p": [1/len(arg_m) for _ in arg_m]}))
            else:
                s = next_state
                max_a = max_act

            # Which of the copies needs to be used for determining the beta 
            if self.interpolation_mode_kwargs["which_copy"] == "chosen":

                reference_state_q_fct = {key: val for key, val in self.q_fct[index].items() if key[0] == s}
                if len(reference_state_q_fct) >= 2:

                    # Determine the reference value to be used
                    if self.interpolation_mode_kwargs["which_reference_arm"] == "lowest":
                        ref_val = min(reference_state_q_fct.values())

                    elif self.interpolation_mode_kwargs["which_reference_arm"] == "median":
                        ref_index = int(np.floor(len(reference_state_q_fct)/2))
                        ref_val = sorted(reference_state_q_fct.values())[ref_index]

                    elif self.interpolation_mode_kwargs["which_reference_arm"] == "second_highest":
                        ref_val = sorted(reference_state_q_fct.values(), reverse= True)[1]

                    else:
                        raise ValueError("The chosen mode of determining the reference arm seems to not exist. If you tried to implement a new interpolation mode you need to specify how to determine the reference arm used in the calculation of the interpolation coefficient!")
                    
                    # Determine the arm corresponding to the reference value
                    arg_min = [key[1] for key, val in reference_state_q_fct.items() if val == ref_val]
                    if len(arg_min) == 1:
                        ref_a = arg_min[0]
                    else:
                        ref_a = int(sample_from_dist(self.policy.rng,"choice",1,**{"a": arg_min, "p": [1/len(arg_min) for _ in arg_min]}))

                    # Determine the absolute value of the difference of Q functions
                    abs_q = abs(self.q_fct[index][(s,max_a)] - self.q_fct[index][(s,ref_a)])
                
                else:
                    abs_q = 0

            elif self.interpolation_mode_kwargs["which_copy"] == "other":
                
                reference_state_q_fct = {key: val for key, val in self.q_fct[other_index].items() if key[0] == s}
                if len(reference_state_q_fct) >= 2:

                    # Determine the reference value to be used
                    if self.interpolation_mode_kwargs["which_reference_arm"] == "lowest":
                        ref_val = min(reference_state_q_fct.values())

                    elif self.interpolation_mode_kwargs["which_reference_arm"] == "median":
                        ref_index = int(np.floor(len(reference_state_q_fct)/2))
                        ref_val = sorted(reference_state_q_fct.values())[ref_index]

                    elif self.interpolation_mode_kwargs["which_reference_arm"] == "second_highest":
                        ref_val = sorted(reference_state_q_fct.values(), reverse= True)[1]

                    else:
                        raise ValueError("The chosen mode of determining the reference arm seems to not exist. If you tried to implement a new interpolation mode you need to specify how to determine the reference arm used in the calculation of the interpolation coefficient!")
                    
                    # Determine the arm corresponding to the reference value
                    arg_min = [key[1] for key, val in reference_state_q_fct.items() if val == ref_val]
                    if len(arg_min) == 1:
                        ref_a = arg_min[0]
                    else:
                        ref_a = int(sample_from_dist(self.policy.rng,"choice",1,**{"a": arg_min, "p": [1/len(arg_min) for _ in arg_min]}))

                    # Determine the absolute value of the difference of Q functions
                    abs_q = abs(self.q_fct[other_index][(s,max_a)] - self.q_fct[other_index][(s,ref_a)])
                
                else:
                    abs_q = 0

            elif self.interpolation_mode_kwargs["which_copy"] == "average":
                
                reference_state_q_fct = {key: val for key, val in average_q_fct.items() if key[0] == s}
                if len(reference_state_q_fct) >= 2:

                    # Determine the reference value to be used
                    if self.interpolation_mode_kwargs["which_reference_arm"] == "lowest":
                        ref_val = min(reference_state_q_fct.values())

                    elif self.interpolation_mode_kwargs["which_reference_arm"] == "median":
                        ref_index = int(np.floor(len(reference_state_q_fct)/2))
                        ref_val = sorted(reference_state_q_fct.values())[ref_index]

                    elif self.interpolation_mode_kwargs["which_reference_arm"] == "second_highest":
                        ref_val = sorted(reference_state_q_fct.values(), reverse= True)[1]

                    else:
                        raise ValueError("The chosen mode of determining the reference arm seems to not exist. If you tried to implement a new interpolation mode you need to specify how to determine the reference arm used in the calculation of the interpolation coefficient!")
                    
                    # Determine the arm corresponding to the reference value
                    arg_min = [key[1] for key, val in reference_state_q_fct.items() if val == ref_val]
                    if len(arg_min) == 1:
                        ref_a = arg_min[0]
                    else:
                        ref_a = int(sample_from_dist(self.policy.rng,"choice",1,**{"a": arg_min, "p": [1/len(arg_min) for _ in arg_min]}))

                    # Determine the absolute value of the difference of Q functions
                    abs_q = abs(average_q_fct[(s,max_a)] - average_q_fct[(s,ref_a)])
                
                else:
                    abs_q = 0

            else:
                raise ValueError("The chosen mode of determining which copy to use in the calculation of beta seems to not exist. If you tried to implement a new one you need to specify how to determine the interpolation coefficient beta in the code!")

            # Determine the interpolation coefficient beta
            beta = abs_q / (self.interpolation_mode_kwargs["c"] + abs_q)

        else:
            raise ValueError("The chosen interpolation mode seems to not exist. If you tried to implement a new interpolation mode you need to specify how to determine the interpolation coefficient beta in the code!")

        # Update the Q function according to the Weighted Double Q Learning algorithm
        self.q_fct[index][(self.current_state,chosen_action)] = (1 - stepsize) * self.q_fct[index][(self.current_state,chosen_action)] + stepsize * (reward + (1 - t) * self.gamma * ( beta * self.q_fct[index][(next_state,max_act)] + (1 - beta) * self.q_fct[other_index][(next_state,max_act)]))

        # Update current state
        self.current_state = next_state

        return state, chosen_action, next_state, reward, t
    
    def get_special_log_keys(self) -> List[Tuple[str,str]]:
        """Returns the log and plot names for all special plots and when to log them"""
        return []

    def get_special_logs_at_step(self) -> List[Tuple[str,Union[int,float]]]:
        return []
    
    def get_special_logs_at_epoch(self) -> List[Tuple[str,Union[int,float]]]:
        return []
    
    def get_special_logs_at_eval(self) -> List[Tuple[str,Union[int,float]]]:
        return []

    def get_greedy_policy(self) -> List[int]:

        """
        Takes the current Q functions, averages them, and gives out a policy list corresponding to the greedy policy.

        Returns:
        - list: The list of greedy actions with respect to the current Q function.
        """

        average_q_fct = {}
        for key in self.q_fct[0].keys():
            average_q_fct[key] = (self.q_fct[0][key] + self.q_fct[1][key]) / 2

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
            if not isinstance(self.interpolation_mode_kwargs["current_state"],bool):
                raise TypeError("If the current state should be used for calculating the beta needs to be a boolean value!")
            if not (self.interpolation_mode_kwargs["which_copy"] in ["chosen", "other", "average"]):
                raise ValueError("The chosen copy to use in the calculation of beta needs to be either chosen, other, or average!")
            if not (self.interpolation_mode_kwargs["which_reference_arm"] in ["lowest", "median", "second_highest"]):
                raise ValueError("The chosen reference arm needs to be either the lowest, the median, or the second highest!")
            if isinstance(self.interpolation_mode_kwargs["c"],(int,float)):
                if not (0 < self.interpolation_mode_kwargs["c"]):
                    raise ValueError("The chosen constant in the calculation of beta needs to be positive!")
            else:
                raise TypeError("The chosen constant in the calculation of beta needs to be a numerical value!")
        else:
            raise ValueError("If you tried implementing a new interpolation mode, please add a check for the respective values of the interpolation mode keyword arguments!")

        # q_fct_manual_init needs to be boolean
        if not isinstance(self.q_fct_manual_init,bool):
            raise TypeError("The variable q_fcts_manual_init needs to be boolean!")
        
        # If the q functions will be initialized manually, the initialization needs to be contained in a dictionary or it needs to be contained in a list of two dictionaries
        if self.q_fct_manual_init:
            if not (isinstance(self.initial_q_fct,dict) or isinstance(self.initial_q_fct,list)):
                raise TypeError("The Q functions passed for manual initialization need to be contained in either a list of two dictionaries or passed as a single dictionary!")
            elif isinstance(self.initial_q_fct,list):
                if len(self.initial_q_fct) == 2:
                    if not (isinstance(self.initial_q_fct[0],dict) and isinstance(self.initial_q_fct[1],dict)):
                        raise TypeError("In case the Q functions should be initialized differently from one another they need to be passed in dictionaries each contained in the list!")
                else:
                    raise TypeError("In case the Q functions should be initialized differently from one another they need to both be passed in a list!")
            
        # Seed is in valid range:
        if isinstance(self.rng_seed, int):
            if not (0 <= self.rng_seed < 2**32):
                raise ValueError("The provided seed is not in the range of acceptable integer seeds!")
        else:
            raise TypeError("The seed needs to be an integer!")
        
        return 1