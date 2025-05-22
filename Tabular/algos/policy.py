import numpy as np
from typing import Dict, List, Tuple, Union
from copy import deepcopy

from utils import check_for_schedule_allowed, sample_from_dist, schedule

class Policy():
    def __init__(self):
        pass

class BasePolicy(Policy):
    def __init__(self,
                 policy_mode: str = "epsilon_greedy_statewise", # Default mode for update schedule is epsilon greedy
                 policy_mode_kwargs: Dict = None,
                 env_allowed_actions: Dict[int,List[int]] = None, # Dictionary of allowed actions for your game
                 env_num_states: int = 0, # Number of states that your game has
                 env_num_actions: int = 0, # Number of actions that your game has
                 rng_seed: int = 42, # Random number generator seed
                 checks: str = "all_checks", # Should all checks, only initial_checks, or no checks be performed
                 ) -> None:
        
        """
        Initializes a Policy for the Q Learning algorithm. The agent chooses the next action according to the
        specified mode with its keyword arguments. It acts on a game, whose number of states and actions, as
        well as allowed actions per states dictionary need to be passed as well.

        Parameters:
        - policy_mode (str): Mode for choosing the next action. Default is epsilon greedy. The implemented modes can
          be found in the self.policy_modes_implemented list.
        - policy_mode_kwargs (dict): A dictionary containing the necessary keyword arguments for the chosen policy mode.
          The necessary keywords for each mode can be found in the self.policy_modes_allowed_kwargs dictionary
        - env_allowed_actions (dict): A dictionary mapping state numbers to a list of allowed actions in that state.
          Needs to match to the number of states and actions specified.
        - env_num_states (int): The number of states the played game has.
        - env_num_actions (int): The number of actions the played game has.
        - rng_seed (int): Seed for the random number generator. Default is 42.
        - checks (str): Checking mode. "all_checks", "only_initial_checks", and "no_checks" is available.
        """

        # Default arguments if arguments are None
        if policy_mode_kwargs is None:
            policy_mode_kwargs = {
                     "initial_rate": 1, 
                     "mode": "rate", 
                     "mode_kwargs": {"rate_fct": lambda n: 1 / n, "iteration_num":1, "final_rate": 0}}
        if env_allowed_actions is None:
            env_allowed_actions = {}
                
        # Standard initialization of arguments from the input
        self.policy_mode = policy_mode
        self.policy_mode_kwargs = deepcopy(policy_mode_kwargs) # Gets modified inside the class
        self.env_allowed_actions = env_allowed_actions # Not modified inside the class
        self.env_num_states = env_num_states
        self.env_num_actions = env_num_actions
        self.rng_seed = rng_seed
        if isinstance(checks,str):
            if not (checks == "all_checks" or checks == "no_checks" or checks == "only_initial_checks"):
                raise ValueError("Checks needs to be either all_checks, no_checks, or only_initial_checks!")
        else:
            raise TypeError("Flag for type and value checking must be a string!")
        self.checks = checks   
        self.init_check1_done = False

        # Basic initialization for input checks
        self.policy_modes_implemented = ["offpolicy","epsilon_greedy","epsilon_greedy_statewise","greedy","softmax"] # If new mode, put it also in the self.policy_modes_allowed_kwargs dict with its necessary kwargs and modify the inputcheck-function
        self.policy_modes_allowed_kwargs = {
            "offpolicy": ["type","kwargs"],
            "epsilon_greedy": ["initial_rate","mode","mode_kwargs"],
            "epsilon_greedy_statewise": ["initial_rate", "mode", "mode_kwargs"],
            "greedy": [],
            "softmax": ["temperature"]
        }
        self.offpolicy_modes_implemented = ["uniform_random", "full_init"]
        self.offpolicy_modes_allowed_kwargs = {
            "uniform_random": [],
            "full_init": ["policy_list"]
        }
        self.length_Q = 0
        for state in range(self.env_num_states):
            self.length_Q += len(self.env_allowed_actions[state])
        self.rng = np.random.default_rng(seed = self.rng_seed) # Random number generator for stochastic draws

        # Checking input constraints
        if self.checks != "no_checks":
            self.inputcheck()

        # Advanced initializations
        if self.policy_mode == "epsilon_greedy":
            self.policy_mode_kwargs["current_rate"] = self.policy_mode_kwargs["initial_rate"]
        if self.policy_mode == "epsilon_greedy_statewise":
            self.schedule_list = [self.policy_mode_kwargs["initial_rate"] for _ in range(self.env_num_states)]
            if self.policy_mode_kwargs["mode"] == "rate":
                self.iteration_num_list = [1 for _ in range(self.env_num_states)]

        # If offpolicy was chosen, the behaviour policy needs to be set up
        if self.policy_mode == "offpolicy":
            self.policy_list = self.init_policy_dict()
    
    def __str__(self):
        return "BasePolicy"

    def choose_next_action(self,state: int, Q: Dict[Tuple[int,int],Union[int,float]]) -> int:

        """
        Returns the next action after a state and an estimated state action value function 
        were observed. Takes into account the chosen mode of the policy.

        Parameters:
        - state (int): The current state, represented as a state number.
        - original_Q (dict): The current estimate of the state action value function

        Returns:
        - int: The action number chosen.
        """

        # Check if state is valid integer and Q is a valid state action value function for the passed game
        if self.checks != "no_checks" and not self.init_check1_done:
            if isinstance(state,int):
                if not state in range (self.env_num_states):
                    raise ValueError("State is not allowed!")
            else:
                raise TypeError("State needs to be an integer!")
            if isinstance(Q,dict):
                if len(Q) == self.length_Q:
                    for key in Q.keys():
                        if isinstance(key,tuple):
                            if len(key) == 2:
                                if isinstance(key[0],int) and isinstance(key[1],int):
                                    if key[1] in self.env_allowed_actions[key[0]]:
                                        if not isinstance(Q[key],(int,float)):
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
                raise TypeError("The given Q function needs to be a dictionary!")
            if self.checks == "only_initial_check":
                self.init_check1_done = True

        # Depending on the policy mode, choose an action
        if self.policy_mode == "offpolicy":
            if isinstance(self.policy_list[state],int):
                return self.policy_list[state]
            elif isinstance(self.policy_list[state],list):
                return int(sample_from_dist(self.rng,self.policy_list[state][0],1,**self.policy_list[state][1])[0])
            else:
                raise ValueError("Something went wrong in the initialiation of the policy list!")
        elif self.policy_mode == "greedy":
            relevant_Q_values = {key: value for key, value in Q.items() if key[0] == state}
            max_value = max(relevant_Q_values.values())
            argmax_values = [key[1] for key, value in relevant_Q_values.items() if value == max_value]
            if len(argmax_values) == 1:
                return argmax_values[0]
            elif len(argmax_values) > 1:
                return int(sample_from_dist(self.rng,"choice",1,**{"a": argmax_values, "p": [1/len(argmax_values) for _ in argmax_values]})[0])
            else:
                raise ValueError("I do not know how you got here, however, something went wrong while choosing the highest Q Value!")
        elif self.policy_mode == "epsilon_greedy":
            epsilon = self.policy_mode_kwargs["current_rate"]
            if epsilon > self.policy_mode_kwargs["mode_kwargs"]["final_rate"]:
                self.policy_mode_kwargs = schedule(**self.policy_mode_kwargs)
            if self.rng.uniform(0,1) < epsilon:
                return int(sample_from_dist(self.rng, "choice",1,**{"a": [act for act in self.env_allowed_actions[state]], "p": [1/len(self.env_allowed_actions[state]) for _ in self.env_allowed_actions[state]]})[0])
            else:
                relevant_Q_values = {key: value for key, value in Q.items() if key[0] == state}
                max_value = max(relevant_Q_values.values())
                argmax_values = [key[1] for key, value in relevant_Q_values.items() if value == max_value]
                if len(argmax_values) == 1:
                    return argmax_values[0]
                elif len(argmax_values) > 1:
                    return int(sample_from_dist(self.rng,"choice",1,**{"a": argmax_values, "p": [1/len(argmax_values) for _ in argmax_values]})[0])
                else:
                    raise ValueError("I do not know how you got here, however, something went wrong while choosing the highest Q Value!")
        elif self.policy_mode == "epsilon_greedy_statewise":
            epsilon = self.schedule_list[state]
            if epsilon > self.policy_mode_kwargs["mode_kwargs"]["final_rate"]:
                if self.policy_mode_kwargs["mode"] == "rate":
                    iteration_num = self.iteration_num_list[state]
                    self.policy_mode_kwargs["mode_kwargs"]["iteration_num"] = iteration_num
                self.policy_mode_kwargs["current_rate"] = epsilon
                self.policy_mode_kwargs = schedule(**self.policy_mode_kwargs)
                self.schedule_list[state] = self.policy_mode_kwargs["current_rate"]
                if self.policy_mode_kwargs["mode"] == "rate":
                    self.iteration_num_list[state] = self.policy_mode_kwargs["mode_kwargs"]["iteration_num"]
            if self.rng.uniform(0,1) < epsilon:
                return int(sample_from_dist(self.rng, "choice",1,**{"a": [act for act in self.env_allowed_actions[state]], "p": [1/len(self.env_allowed_actions[state]) for _ in self.env_allowed_actions[state]]})[0])
            else:
                relevant_Q_values = {key: value for key, value in Q.items() if key[0] == state}
                max_value = max(relevant_Q_values.values())
                argmax_values = [key[1] for key, value in relevant_Q_values.items() if value == max_value]
                if len(argmax_values) == 1:
                    return argmax_values[0]
                elif len(argmax_values) > 1:
                    return int(sample_from_dist(self.rng,"choice",1,**{"a": argmax_values, "p": [1/len(argmax_values) for _ in argmax_values]})[0])
                else:
                    raise ValueError("I do not know how you got here, however, something went wrong while choosing the highest Q Value!")
        elif self.policy_mode == "softmax":
            relevant_Q_values_exp = {key[1]: np.exp(value/self.policy_mode_kwargs["temperature"]) for key, value in Q.items() if key[0] == state}
            sm = sum(relevant_Q_values_exp.values())
            return int(sample_from_dist(self.rng,"choice",1,**{"a":list(relevant_Q_values_exp.keys()), "p": [val/sm for val in relevant_Q_values_exp.values()]})[0])
        else:
            raise ValueError("In case you want to implement new policy modes, you need to specify how to draw the next action!")

    def init_policy_dict(self) -> List:

        """
        Return the chosen behaviour policies state-wise probability list. Takes into
        account the chosen offpolicy mode and the passed game parameters.

        Returns:
        - list: The probabilites of taking actions in certain states.
        """

        # Initialize the behaviour policy according to the offpolicy type
        if self.policy_mode_kwargs["type"] == "uniform_random":
            return [["choice",{"a": [act for act in self.env_allowed_actions[state]], "p": [1/len(self.env_allowed_actions[state]) for _ in self.env_allowed_actions[state]]}] for state in range(self.env_num_states)]
        elif self.policy_mode_kwargs["type"] == "full_init":
            return deepcopy(self.policy_mode_kwargs["kwargs"]["policy_list"])
        else:
            raise ValueError("In case you want to implement new modes for offline policies, you need to specify how to initialize the policy list!")

    def inputcheck(self) -> int:

        """
        Validates the input parameters to ensure they follow the expected formats and constraints.

        Raises:
        - ValueError: If any of the input parameters are invalid.
        - TypeError: If any of the input types are invalid.
        """

        # Policy mode is string and is implemented
        if isinstance(self.policy_mode,str):
            if not (self.policy_mode in self.policy_modes_implemented):
                raise ValueError(f"Policy mode {self.policy_mode} is not implemented!")
        else:
            raise TypeError("Policy mode must be a string!")
        
        # Number of states is a positive integer
        if isinstance(self.env_num_states,int):
            if not (self.env_num_states >= 0):
                raise ValueError("Number of states needs to be positive!")
        else:
            raise TypeError("Number of states needs to be an integer!")
        
        # Number of actions is a positive integer
        if isinstance(self.env_num_actions,int):
            if not (self.env_num_actions >= 0):
                raise ValueError("Number of actions needs to be positive!")
        else:
            raise TypeError("Number of actions needs to be an integer!")
        
        # Allowed actions is a dictionary with integers as keys that contain everything up to num_states and lists of integers in the range of action space as values
        if isinstance(self.env_allowed_actions,dict):
            if (len(self.env_allowed_actions) == self.env_num_states):
                for key in self.env_allowed_actions.keys():
                    if isinstance(key,int):
                        if key in range(self.env_num_states):
                            if isinstance(self.env_allowed_actions[key],list):
                                if len(self.env_allowed_actions[key]) == len(set(self.env_allowed_actions[key])):
                                    for item in self.env_allowed_actions[key]:
                                        if isinstance(item,int):
                                            if not item in range(self.env_num_actions):
                                                raise ValueError("Entries of allowed actions dictionary need to be lists of integers in the range of the number of actions!")
                                        else:
                                            raise ValueError("Entries of allowed actions dictionary need to be lists of integers!")
                                else:
                                    raise ValueError("There should be no doubles in the lists of allowed actions for each state!")
                            else:
                                raise ValueError("Entries of allowed actions dictionary need to be lists!")
                        else:
                            raise TypeError("All keys of the allowed actions dictionary need to be insinde the range of the number of states!")
                    else:
                        raise TypeError("Keys for the allowed actions need to be integers!")
            else:
                raise ValueError("The length of the allowed states dictionary must match the number of states!")
            
        else:
            raise TypeError("Allowed actions needs to be a dictionary!")
        
        # Policy kwargs is dict and keywords are strings and match the mode
        if isinstance(self.policy_mode_kwargs,dict):
            if all([isinstance(kw,str) for kw in self.policy_mode_kwargs.keys()]):
                for key in self.policy_mode_kwargs.keys():
                    if not (key in self.policy_modes_allowed_kwargs[self.policy_mode]):
                        if key == "current_rate":
                            break
                        raise ValueError(f"The key {key} is invalid for the policy mode {self.policy_mode}!")
                for entry in self.policy_modes_allowed_kwargs[self.policy_mode]:
                    if not (entry in self.policy_mode_kwargs.keys()):  
                        raise ValueError(f"The key {entry} is missing for the policy mode {self.policy_mode}!")
            else:
                raise TypeError("All keyword argument names for the policy must be strings!")
        else:
            raise TypeError("Keyword arguments for the policy mode must be passed as a dictionary!")
        
        # Keywords values for the policy are allowed
        if self.policy_mode == "offpolicy":
            if isinstance(self.policy_mode_kwargs["type"],str):
                if self.policy_mode_kwargs["type"] in self.offpolicy_modes_implemented:
                    if isinstance(self.policy_mode_kwargs["kwargs"],dict):
                        for offlinekey in self.policy_mode_kwargs["kwargs"].keys():
                            if not offlinekey in self.offpolicy_modes_allowed_kwargs[self.policy_mode_kwargs["type"]]:
                                raise ValueError(f"The keyword {offlinekey} is invalid for the offline policy mode {self.policy_mode_kwargs['type']}!")
                        for offlineentry in self.offpolicy_modes_allowed_kwargs[self.policy_mode_kwargs["type"]]:
                            if not (offlineentry in self.policy_mode_kwargs["kwargs"].keys()):
                                raise ValueError(f"The keyword {offlineentry} is missing for the offline policy mode {self.policy_mode_kwargs['type']}!")     
                    else:
                        raise TypeError("The keyword arguments for the offline policy type need to be passed as a dictionary!")
                else:
                    raise ValueError(f"The offpolicy initialization type {self.policy_mode_kwargs['type']} is not implemented!")
            else:
                raise TypeError("The type of an offline policy needs to be a valid string!")
            if self.policy_mode_kwargs["type"] == "full_init":
                if isinstance(self.policy_mode_kwargs["kwargs"]["policy_list"],list):
                    if len(self.policy_mode_kwargs["kwargs"]["policy_list"]) == self.env_num_states:
                        for i in range(self.env_num_states):
                            if isinstance(self.policy_mode_kwargs["kwargs"]["policy_list"][i],int):
                                if not (self.policy_mode_kwargs["kwargs"]["policy_list"][i] in self.env_allowed_actions[i]):
                                    raise ValueError(f"For deterministic action choosing in state {i} an integer in the range of allowed actions needs to be passed!")
                            elif isinstance(self.policy_mode_kwargs["kwargs"]["policy_list"][i],list):
                                if len(self.policy_mode_kwargs["kwargs"]["policy_list"][i]) == 2:
                                    if self.policy_mode_kwargs["kwargs"]["policy_list"][i][0] == "choice":
                                        if isinstance(self.policy_mode_kwargs["kwargs"]["policy_list"][i][1],dict):
                                            if "a" in self.policy_mode_kwargs["kwargs"]["policy_list"][i][1].keys() and "p" in self.policy_mode_kwargs["kwargs"]["policy_list"][i][1].keys() and len(self.policy_mode_kwargs["kwargs"]["policy_list"][i][1]) == 2:
                                                if isinstance(self.policy_mode_kwargs["kwargs"]["policy_list"][i][1]["a"],list) and isinstance(self.policy_mode_kwargs["kwargs"]["policy_list"][i][1]["p"],list):
                                                    if len(self.policy_mode_kwargs["kwargs"]["policy_list"][i][1]["a"]) == len(self.policy_mode_kwargs["kwargs"]["policy_list"][i][1]["p"]):
                                                        if len(self.policy_mode_kwargs["kwargs"]["policy_list"][i][1]["a"]) == len(set(self.policy_mode_kwargs["kwargs"]["policy_list"][i][1]["a"])):
                                                            for action in self.policy_mode_kwargs["kwargs"]["policy_list"][i][1]["a"]:
                                                                if isinstance(action,int):
                                                                    if not action in self.env_allowed_actions[i]:
                                                                        raise ValueError(f"All actions to choose from stochastically in state {i} need to be integers in the range of allowed actions!")
                                                                else:
                                                                    raise ValueError(f"All actions to choose from stochastically in state {i} need to be integers!")
                                                            for prob in self.policy_mode_kwargs["kwargs"]["policy_list"][i][1]["p"]:
                                                                if isinstance(prob,(int,float)):
                                                                    if not 0 <= prob <= 1:
                                                                        raise ValueError(f"All probabilities for stochastic choosing in state {i} need to be numbers between 0 and 1!")
                                                                else:
                                                                    raise ValueError(f"All probabilities for stochastic choosing in state {i} need to be numbers!")
                                                            if np.sum(self.policy_mode_kwargs["kwargs"]["policy_list"][i][1]["p"]) != 1:
                                                                raise ValueError(f"The sum of the probabilites for stochastic choosing in state {i} needs to be one!")
                                                        else:
                                                            raise ValueError(f"There should be no doubles in the list of actions from which to choose stochastically in state {i}!")
                                                    else:
                                                        raise TypeError(f"For stochastic action choosing in state {i} the parameters a and p of the numpy choice distribution need to be lists of equal length!")
                                                else:
                                                    raise TypeError(f"For stochastic action choosing in state {i} the parameters a and p of the numpy choice distribution need to be lists!")
                                            else:
                                                raise TypeError(f"For stochastic action choosing in state {i} the parameters a and p of the numpy choice distribution need to be specified and nothing else or less!")
                                        else:
                                            raise TypeError(f"For stochastic action choosing in state {i} the parameters for the numpy choice distribution need to be passed as a dictionary!")
                                    else:
                                        raise ValueError(f"For stochastic action choosing in state {i} only the sampling with the numpy choice distribution is allowed!")
                                else:
                                    raise ValueError("The list for the policy initialization needs to be of length 2 in the stochastic case!")
                            else:
                                raise TypeError("The list for the policy initialization needs to contain either integers for the deterministic case or lists for the stochastic case!")
                    else:
                        raise ValueError("The list for the policy initialization needs to match the number of states!")
                else:
                    raise TypeError("The policy initalization in the case of offline policies with manual initializations needs to be a list!")
            elif self.policy_mode_kwargs["type"] == "uniform_random":
                pass
            else:
                raise ValueError("If you want to implement a new initialization mode for offline policies, please also update the inputcheck function!")
        elif self.policy_mode == "epsilon_greedy":
            if "initial_rate" in self.policy_mode_kwargs.keys():
                self.policy_mode_kwargs["current_rate"] = self.policy_mode_kwargs["initial_rate"]
                check_for_schedule_allowed(**self.policy_mode_kwargs)
            else:
                raise ValueError("Initial rate is missing from the policy mode key word arguments dictionary!")
        elif self.policy_mode == "epsilon_greedy_statewise":
            self.policy_mode_kwargs["current_rate"] = self.policy_mode_kwargs["initial_rate"]
            check_for_schedule_allowed(**self.policy_mode_kwargs)
        elif self.policy_mode == "greedy":
            pass
        elif self.policy_mode == "softmax":
            if isinstance(self.policy_mode_kwargs["temperature"],(int,float)):
                if self.policy_mode_kwargs["temperature"] <= 0:
                    raise ValueError("The temperature for softmax policies needs to be positive!")
            else:
                raise ValueError("The temperature for softmax policies needs to have a numerical value!")
        else:
            raise ValueError("If you want to implement a new policy mode, please also update the inputcheck function!")
        
        # Seed is in valid range:
        if isinstance(self.rng_seed, int):
            if not (0 <= self.rng_seed < 2**32):
                raise ValueError("The provided seed is not in the range of acceptable integer seeds!")
        else:
            raise TypeError("The seed needs to be an integer!")
        
        return 1