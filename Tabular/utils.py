import numpy as np
import inspect
import random
import os
from typing import Union, Any
import matplotlib.axes
import types

# Utils for passing distributions to environments

def check_for_allowed_dist(rng: np.random.Generator, dist: str, **kwargs: dict) -> int:

    """
        Validates the input parameters for the sample_from_dist function.

        Parameters:
        - rng (numpy.random.Generator): The numpy random generator to be used in sample_from_dist.
        - kwargs (dict): The keyword arguments to be used in sample_from_dist.

        Raises:
        - ValueError: If any of the input parameters are invalid.
        - TypeError: If any of the input types are invalid.
    """

    if isinstance(rng, np.random.Generator):
        dist_func = getattr(rng, dist, None)
        if dist_func is None:
            raise ValueError(f"Distribution '{dist}' is not supported with np.random.Generator!")
        sig = inspect.signature(dist_func)
        for key in kwargs:
            if key not in sig.parameters:
                raise ValueError(f"Invalid argument {key} for distribution {dist} specified!")
        for param in sig.parameters.values():
            if param.default == inspect.Parameter.empty and param.name not in kwargs and param.name != 'size':
                raise ValueError(f"Missing required argument {param.name} for distribution {dist}!")
    else:
        raise TypeError("The only random number generator currently supported and teste is np.random.Generator!")
    return 1

def sample_from_dist(rng: np.random.Generator, dist: str, size: int, **kwargs: dict) -> np.ndarray:

    """
        Samples from a distribution with parameters to be specified using a numpy random number generator.
        Use this function only when you are sure that the keyword arguments passed match the distribution
        and the distribution is implemented in numpy random module. You can run check_for_allowed_dist if 
        you are unsure.

        Parameters:
        - rng (numpy.random.Generator): The numpy random generator to be used in sample_from_dist.
        - dist (str): The distribution to be passed to the random number generator.
        - size (int): The number of random samples to be drawn.
        - kwargs (dict): The keyword arguments for the chosen distribution.

        Returns:
        - sample (numpy ndarray): The drawn sample(s) in a numpy array.

        Raises:
        - ValueError: If no distribution is specified.
    """

    # Dynamically get the distribution method from the generator
    dist_func = getattr(rng, dist, None)
    if dist_func is None:
        raise ValueError(f"Distribution '{dist}' is not supported.")
    return dist_func(size=size, **kwargs)

# Utils for learning rate and exploration schedules

def check_for_schedule_allowed(initial_check: bool = True, **kwargs: dict) -> int: 

    """
        Validates the input parameters for the schedule function.

        Parameters:
        - initial_check (bool): True, if this is the first check and the schedule function has not
          already been applied to the passed dictionary.
        - kwargs (dict): The keyword arguments to be used in schedule.

        Raises:
        - ValueError: If any of the input parameters are invalid.
        - TypeError: If any of the input types are invalid.
    """

    # Implemented stuff
    mandatory_implemented_keys = ["initial_rate", "current_rate", "mode", "mode_kwargs"]
    implemented_modes = ["constant", "linear", "rate"]
    allowed_mode_kwargs = {
        "constant": ["final_rate"],
        "linear": ["final_rate", "num_steps", "slope"],
        "rate": ["rate_fct", "iteration_num", "final_rate"],
    }
    # Exactly the mandatory keys are contained
    if isinstance(kwargs, dict):
        for key in mandatory_implemented_keys:
            if not key in kwargs.keys():
                raise ValueError(f"Key {key} is missing!")
        for key in kwargs.keys():
            if not key in mandatory_implemented_keys:
                raise ValueError(f"Keyword {key} is appearing but not implemented!")
    else:
        raise TypeError(f"Keyword arguments need to be passed in a dictionary!")
    
    # Initial rate is a number between 0 and 1
    if isinstance(kwargs["initial_rate"],(float,int)):
        if not (0 <= kwargs["initial_rate"] <= 1):
            raise ValueError("Initial rate needs to be between 0 and 1!")
    else:
        raise TypeError("Initial rate needs to be a number!")
    
    # Current rate is a number between 0 and initial rate
    if isinstance(kwargs["current_rate"],(float,int)):
        if not (0 <= kwargs["current_rate"] <= kwargs["initial_rate"]):
            raise ValueError("Current rate needs to be between 0 and initial rate!")
    else:
        raise TypeError("Current rate needs to be a number!")
    
    # If it is the first check, the initial and current rate need to coincide
    if initial_check:
        if not (kwargs["current_rate"] == kwargs["initial_rate"]):
            raise ValueError("Current and initial rate need to coincide in the beginning!")
    
    # Mode is implemented
    if isinstance(kwargs["mode"],str):
        if not (kwargs["mode"] in implemented_modes):
            raise ValueError(f"Mode {kwargs['mode']} is not implemented!")
    else:
        raise TypeError("Mode needs to be a string!")
    
    # Keyword arguments for mode are supported
    if isinstance(kwargs["mode_kwargs"],dict):
        for key in kwargs["mode_kwargs"].keys():
            if not key in allowed_mode_kwargs[kwargs["mode"]]:
                raise ValueError(f"Keyword {key} is appearing but not implemented!")
        for key in allowed_mode_kwargs[kwargs["mode"]]:
            if not key in kwargs["mode_kwargs"].keys():
                raise ValueError(f"Keyword {key} is missing!")
    else:
        raise TypeError("Keyword arguments for the mode must be passed as a dictionary!")

    # Keyword arguments for mode take the right values
    if kwargs["mode"] == "constant":
        pass
    elif kwargs["mode"] == "linear":
        if isinstance(kwargs["mode_kwargs"]["final_rate"],(int,float)):
            if not (0 <= kwargs["mode_kwargs"]["final_rate"] <= kwargs["current_rate"]):
                raise ValueError("For the schedule mode linear, the final rate needs to be less than the initial and current rates!")
        else:
            raise TypeError("For the schedule mode linear, the final rate needs to be a numerical value!")
        if isinstance(kwargs["mode_kwargs"]["num_steps"],int):
            if not (0 < kwargs["mode_kwargs"]["num_steps"]):
                raise ValueError("For the schedule mode linear, the number of steps needs to be positive!")
        else:
            raise TypeError("For the schedule mode linear, the number of steps needs to be an integer!")
        if isinstance(kwargs["mode_kwargs"]["slope"],(int,float)):
            if not (0 <= kwargs["mode_kwargs"]["slope"] or kwargs["mode_kwargs"]["slope"] == -1):
                raise ValueError("For the schedule mode linear, the slope needs to either be non-negative, or take the value -1 for initialization via final rate and number of steps!")
        else:
            raise TypeError("For the schedule mode linear, the slope must be numerical!")
    elif kwargs["mode"] == "rate":
        if callable(kwargs["mode_kwargs"]["rate_fct"]):
            if not kwargs["mode_kwargs"]["rate_fct"].__name__ == '<lambda>':
                raise TypeError("For the schedule mode rate, the rate function needs to be passed as a lambda function!")
        else:
            raise TypeError("For the schedule mode rate, the rate function needs to be a callable!")
        if isinstance(kwargs["mode_kwargs"]["iteration_num"],int):
            if not kwargs["mode_kwargs"]["iteration_num"] > 0:
                raise ValueError("For the schedule mode rate, the iteration number needs to be positive!")
            if initial_check:
                if not kwargs["mode_kwargs"]["iteration_num"] == 1:
                    raise ValueError("In the beginning, the number of iterations done needs to be one!")
        else:
            raise TypeError("For the schedule mode rate, the iteration number needs to be numerical!")
    else:
        raise ValueError("If you want to implement a new schedule mode please specify a type check in check_for_schedule_allowed!")
    return 1

def schedule(**kwargs: dict) -> dict:

    """
        Returns the next scheduled rate for a dictionary containing the initial rate, the current 
        rate, the mode, and the necessary keyword arguments for the mode to be applied.

        Parameters:
        - kwargs (dict): The keyword arguments for the update of the rate.

        Returns:
        - kwargs (dict): The keyword arguments dictionary, but with updated current rate.

        Raises:
        - ValueError: If the schedule mode is not implemented.
    """

    # Calculate next rate
    if kwargs["current_rate"] == 0:
        return kwargs
    elif kwargs["mode"] == "constant":
        next_rate = kwargs["current_rate"]
    elif kwargs["mode"] == "linear":
        if kwargs["mode_kwargs"]["slope"] == -1:
            kwargs["mode_kwargs"]["slope"] = (kwargs["initial_rate"] - kwargs["mode_kwargs"]["final_rate"]) / (kwargs["mode_kwargs"]["num_steps"] - 1)
        next_rate = kwargs["current_rate"] - kwargs["mode_kwargs"]["slope"]
        if next_rate < kwargs["mode_kwargs"]["final_rate"]:
            kwargs["mode_kwargs"]["slope"] = 0
            next_rate = kwargs["mode_kwargs"]["final_rate"]
    elif kwargs["mode"] == "rate":
        next_rate = kwargs["initial_rate"] * kwargs["mode_kwargs"]["rate_fct"](kwargs["mode_kwargs"]["iteration_num"])
        kwargs["mode_kwargs"]["iteration_num"] += 1
        if next_rate < kwargs["mode_kwargs"]["final_rate"]:
            next_rate = kwargs["mode_kwargs"]["final_rate"]
    else:
        raise ValueError("If you want to implement a new schedule mode please modify the schedule function!")
    if next_rate <= 0:
        next_rate = 0
        print("Warning: Your algorithm has reached a point where the scheduled next epsilon or stepsize is zero! You may reconsider your choices of schedule!")
    kwargs["current_rate"] = next_rate
    return kwargs

# Utils for train function

def generate_random_seed():
    """Generates a random seed for NumPy RNG."""
    return random.randint(0, 2**32 - 1)

def check_input_for_train(**kwargs: dict) -> None:

    """
        Validates the input parameters for the train function. Does not validate algo and env!

        Parameters:
        - kwargs (dict): The keyword arguments to be used in train.

        Raises:
        - ValueError: If any of the input parameters are invalid.
        - TypeError: If any of the input types are invalid.
    """

    allowed_arguments = ["algo", "algo_kwargs", "algo_special_logs", "algo_special_logs_kwargs", "env", "env_kwargs", "training_mode", "num_steps", "max_steps_per_epoch", "training_seed_schedule", "training_reseeding", "eval_freq", "eval_steps", "eval_seed_schedule", "eval_reseeding", "bias_estimation", "focus_state_actions", "which_state_actions_focus", "correct_action_log", "correct_action_log_which", "correct_act_q_fct_mode", "correct_act_q_fct_mode_kwargs", "safe_mode", "progress", "measure_runtime"]

    # Are all keywords allowed
    if isinstance(kwargs, dict):
        for key in kwargs.keys():
            if not key in allowed_arguments:
                raise ValueError(f"The keyword {key} is not allowed for the train function!")
    else:
        raise TypeError("The keyword arguments should be passed as a dictionary!")
        
    # Algo kwargs should be dictionary and not contain environment name, environment kwargs, special_logs_kwargs, or the seed or checks parameters
    if "algo_kwargs" in kwargs.keys():
        if isinstance(kwargs["algo_kwargs"],dict):
            if "env" in kwargs["algo_kwargs"].keys():
                raise ValueError("The Environment name should not be passed as an algorithm keyword argument but instead only as the env argument!")
            elif "env_kwargs" in kwargs["algo_kwargs"].keys():
                raise ValueError("The Environment keyword arguments should not be passed as an algorithm keyword argument but instead only as the env_kwargs argument!")
            elif "rng_seed" in kwargs["algo_kwargs"].keys():
                raise ValueError("The seed for the random number generator should not be passed as an algorithm keyword argument but instead for the training and evaluation cycles seperate schedules may be provided with the option to reseed each!")
            elif "checks" in kwargs["algo_kwargs"].keys():
                raise ValueError("The checks parameter should not be passed as an algorithm keyword argument but instead you should use the safe_mode argument!")
            elif "special_logs_kwargs" in ["algo_kwargs"].keys():
                raise ValueError("The special logs keyword arguments should not be passed as an algorithm keyword argument but instead you should use the algo_special_logs_kwargs argument!")
        else:
            raise TypeError("The Algorithm keyword arguments need to be passed as a dictionary!")
        
    # algo_special_logs needs to be boolean
    if "algo_special_logs" in kwargs.keys():
        if not isinstance(kwargs["algo_special_logs"],bool):
            raise TypeError("The parameter algo_special_logs needs to be boolean!")
    
    # algo_special_logs_kwargs needs to be a dictionary
    if "algo_special_logs_kwargs" in kwargs.keys():
        if not isinstance(kwargs["algo_special_logs_kwargs"],dict):
            raise TypeError("The parameter algo_special_logs_kwargs needs to be a dictionary!")
        
    # Env kwargs should be dictionary and not contain the seed or checks parameters
    if "env_kwargs" in kwargs.keys():
        if isinstance(kwargs["env_kwargs"],dict):
            if "rng_seed" in kwargs["env_kwargs"].keys():
                raise ValueError("The seed for the random number generator should not be passed as an environment keyword argument but instead for the training and evaluation cycles seperate schedules may be provided with the option to reseed each!")
            elif "checks" in kwargs["env_kwargs"].keys():
                raise ValueError("The checks parameter should not be passed as an environment keyword argument but instead you should use the safe_mode argument!")
        else:
            raise TypeError("The Environment keyword arguments need to be passed as a dictionary!")

    # Training mode should be either steps or epoch
    if "training_mode" in kwargs.keys():
        if isinstance(kwargs["training_mode"],str):
            if not (kwargs["training_mode"] == "steps" or kwargs["training_mode"] == "epoch"):
                raise ValueError(f"It seems that the training mode {kwargs['training_mode']} is not implemented!")
        else:
            raise TypeError("The training mode needs to be a string!")
    
    # Number of steps is positive integer
    if "num_steps" in kwargs.keys():
        if isinstance(kwargs["num_steps"],int):
            if kwargs["num_steps"] <= 0:
                raise ValueError("The number of steps needs to be positive!")
        else:
            raise TypeError("The number of steps needs to be an integer!")
    
    # Number of maximum steps per epoch needs to be a positive integer or -1
    if "max_steps_per_epoch" in kwargs.keys():
        if isinstance(kwargs["max_steps_per_epoch"],int):
            if not (kwargs["max_steps_per_epoch"] > 0 or kwargs["max_steps_per_epoch"] == -1):
                raise ValueError("The maximum number of steps per epoch needs to be either a positive integer or -1 in case no maximum is applied!")
        else:
            raise TypeError("The maximum number of steps per epoch needs to be an integer!")

    # Training seed schedule is list containing at least one seed and all seeds are in the range of possible seeds or -1 for random
    if "training_seed_schedule" in kwargs.keys():
        if isinstance(kwargs["training_seed_schedule"],list):
            for s in kwargs["training_seed_schedule"]:
                if isinstance(s, int):
                    if not (0 <= s < 2**32 or s == -1):
                        raise ValueError(f"The provided seed {s} in the training seed schedule list is not in the range of acceptable integer seeds and does not take the value -1 corresponding to a random seed!")
                else:
                    raise TypeError("The seeds in the training seed schedule list need to be integers!")
        else:
            raise TypeError("The training seed schedule needs to be a list!")
    
    # Training reseeding needs to be boolean
    if "training_reseeding" in kwargs.keys():
        if not isinstance(kwargs["training_reseeding"],bool):
            raise TypeError("The parameter training_reseeding needs to be boolean!")
    
    # Evaluation frequency needs to be a positive integer
    if "eval_freq" in kwargs.keys():
        if isinstance(kwargs["eval_freq"],int):
            if kwargs["eval_freq"] <= 0:
                raise ValueError("The evaluation frequency needs to be a positive integer!")
        else:
            raise TypeError("The evaluation frequency needs to be an integer!")
    
    # Evaluation steps needs to be a positive integer
    if "eval_steps" in kwargs.keys():
        if isinstance(kwargs["eval_steps"],int):
            if kwargs["eval_steps"] <= 0:
                raise ValueError("The evaluation steps need to be a positive integer!")
        else:
            raise TypeError("The evaluation steps need to be an integer!")
    
    # Evaluation seed schedule is list containing at least one seed and all seeds are in the range of possible seeds or -1 for random
    if "eval_seed_schedule" in kwargs.keys():
        if isinstance(kwargs["eval_seed_schedule"],list):
            for s in kwargs["eval_seed_schedule"]:
                if isinstance(s, int):
                    if not (0 <= s < 2**32 or s == -1):
                        raise ValueError(f"The provided seed {s} in the evaluation seed schedule list is not in the range of acceptable integer seeds and does not take the value -1 corresponding to a random seed!")
                else:
                    raise TypeError("The seeds in the evaluation seed schedule list need to be a integers!")
        else:
            raise TypeError("The evaluation seed schedule needs to be a list!")
    
    # Eval reseeding needs to be boolean
    if "eval_reseeding" in kwargs.keys():
        if not isinstance(kwargs["eval_reseeding"],bool):
            raise TypeError("The parameter eval_reseeding needs to be boolean!")
    
    # Bias estimation needs to be boolean
    if "bias_estimation" in kwargs.keys():
        if not isinstance(kwargs["bias_estimation"],bool):
            raise TypeError("The parameter bias_estimation needs to be boolean!")
        
    # Choice of states and actions to log the bias estimation for should be a Tuple containing a list of states and a list containing lists of action numbers or "best"
    if "which_state_actions_focus" in kwargs.keys():
        if isinstance(kwargs["which_state_actions_focus"],tuple):
            if len(kwargs["which_state_actions_focus"]) == 2:
                if isinstance(kwargs["which_state_actions_focus"][0],list) and isinstance(kwargs["which_state_actions_focus"][1],list):
                    for state in kwargs["which_state_actions_focus"][0]:
                        if isinstance(state,int):
                            if state < 0:
                                raise ValueError(f"The state {state} you provided for logging individual bias estimations is not valid, since it is negative!")
                        elif state != "start":
                            raise TypeError(f"The state {state} you provided for logging individual bias estimations is not valid since it is neither an integer nor 'start'!")
                    for actlist in kwargs["which_state_actions_focus"][1]:
                        if isinstance(actlist,list):
                            for act in actlist:
                                if isinstance(act,int):
                                    if act < 0:
                                        raise ValueError(f"The action {act} you provided for logging individual bias estimations is not valid since it is negative!")
                                elif act != "best":
                                    raise ValueError(f"The action {act} you provided for logging individual bias estimations is not valid since it is neither an integer nor best!")
                        else:
                            raise TypeError(f"The list of actions {actlist} you provided for logging individual bias estimations is not valid since it is not a list!")
                else:
                    raise TypeError("The states and actions provided for logging individual bias estimations have to be provided in the form of a list!")
            else:
                raise ValueError("The states and actions provided for logging individual bias estimations need to be provided as a tuple of length two!")
        else:
            raise ValueError("The states and actions provided for logging individual bias estimations need to be provided as a tuple!")
    
    # Focus state actions needs to be boolean
    if "focus_state_actions" in kwargs.keys():
        if not isinstance(kwargs["focus_state_actions"],bool):
            raise TypeError("The parameter focus_state_actions needs to be boolean!")

    # Correct action log needs to be boolean
    if "correct_action_log" in kwargs.keys():
        if not isinstance(kwargs["correct_action_log"],bool):
            raise TypeError("The parameter correct_action_log needs to be boolean!")
    
    # Correct action log which needs to be either all or a list of positive integers without doubling
    if "correct_action_log_which" in kwargs.keys():
        if kwargs["correct_action_log_which"] != "all":
            if isinstance(kwargs["correct_action_log_which"],list):
                if len(kwargs["correct_action_log_which"]) == len(set(kwargs["correct_action_log_which"])):
                    for state in kwargs["correct_action_log_which"]:
                        if isinstance(state,(float,int)):
                            if state < 0:
                                raise ValueError("The states considered for the correct action rates need to be positive integers!")
                        else:
                            raise TypeError("The states considered for the correct action rates need to be integers!")
                else:
                    raise ValueError("The states considered for the correct action rates cannot contain duplicates!")
            else:
                raise ValueError("The states considered for the correct action rates need to be contained in a list!")
        
    # Correct action and q function mode needs to be manual or value iteration
    if "correct_act_q_fct_mode" in kwargs.keys():
        if not (kwargs["correct_act_q_fct_mode"] == "manual" or kwargs["correct_act_q_fct_mode"] == "value_iteration"):
            raise ValueError("The correct action and q function mode you passed seems to not be valid. If you tried implementing a new one make sure to update the inputcheck function!")
        
    # Correct action and q function mode keyword arguments need to match the chosen keyword
    if "correct_act_q_fct_mode_kwargs" in kwargs.keys():
        if "correct_act_q_fct_mode" in kwargs.keys():
            if isinstance(kwargs["correct_act_q_fct_mode_kwargs"],dict):
                if kwargs["correct_act_q_fct_mode"] == "manual":
                    necessary_kwargs = ["correct_actions", "correct_q_fct"]
                    for kwarg in necessary_kwargs:
                        if not kwarg in kwargs["correct_act_q_fct_mode_kwargs"].keys():
                            raise ValueError(f"The keyword {kwarg} is missing for determining the correct action and q function!")
                    for kwarg in kwargs["correct_act_q_fct_mode_kwargs"].keys():
                        if not kwarg in necessary_kwargs:
                            raise ValueError(f"The keyword {kwargs} is not allowed for determining the correct action and q function!")
                    if isinstance(kwargs["correct_act_q_fct_mode_kwargs"]["correct_actions"],list):
                        for actlist in kwargs["correct_act_q_fct_mode_kwargs"]["correct_actions"]:
                            if isinstance(actlist,list):
                                for act in actlist:
                                    if isinstance(act,int):
                                        if act < 0:
                                            raise ValueError(f"The action {act} passed in the dictionary of correct actions is wrong as it is negative!")
                                    else:
                                        raise TypeError(f"The action {act} passed in the dictionary of correct actions is wrong as it is not an interger!")
                            else:
                                raise TypeError("The correct actions for the manual initialization of correct actions need to be passed in lists!")
                    else:
                        raise TypeError("The correct actions for the manual initialization of correct actions need to be passed as a list!")
                    if isinstance(kwargs["correct_act_q_fct_mode_kwargs"]["correct_q_fct"],dict):
                        for key,val in kwargs["correct_act_q_fct_mode_kwargs"]["correct_q_fct"].items():
                            if not isinstance(val,(int,float)):
                                raise TypeError(f"The q value {val} you passed for the manual initialization of the Q value is not allowed, as it is not numerical!")
                            if isinstance(key,tuple):
                                if len(key) == 2:
                                    if isinstance(key[0],int) and isinstance(key[1],int):
                                        if key[0] < 0 or key[1] < 0:
                                            raise ValueError(f"The state action pair {key} contains a negative integer and is thus invalid!")
                                    else:
                                        raise TypeError(f"The state action pair {key} contains non-integers and is thus invalid!")
                                else:
                                    raise ValueError(f"The state action pair {key} is not a tuple of length 2 and is thus invalid!")
                            else:
                                raise TypeError(f"The state action pair {key} is not a tuple and is thus invalid!")
                elif kwargs["correct_act_q_fct_mode"] == "value_iteration":
                    necessary_kwargs = ["n_max", "tol", "env_mean_rewards", "env_mean_rewards_mc_runs"]
                    for kwarg in necessary_kwargs:
                        if not kwarg in kwargs["correct_act_q_fct_mode_kwargs"].keys():
                            raise ValueError(f"The keyword {kwarg} is missing for determining the correct action and q function!")
                    for kwarg in kwargs["correct_act_q_fct_mode_kwargs"].keys():
                        if not kwarg in necessary_kwargs:
                            raise ValueError(f"The keyword {kwargs} is not allowed for determining the correct action and q function!")
                    if isinstance(kwargs["correct_act_q_fct_mode_kwargs"]["n_max"],int):
                        if kwargs["correct_act_q_fct_mode_kwargs"]["n_max"] <= 0:
                            raise ValueError("The number of maximum iterations for the value iteration needs to be a positive number!")
                    else:
                        raise TypeError("The number of maximum iterations for the value iteration needs to be an integer!")
                    if isinstance(kwargs["correct_act_q_fct_mode_kwargs"]["tol"],(int,float)):
                        if kwargs["correct_act_q_fct_mode_kwargs"]["tol"] < 0:
                            raise ValueError("The error tolerance for the value iteration needs to be positive!")
                    else:
                        raise TypeError("The error tolerance for the value iteration needs to be a numerical value!")
                else:
                    raise ValueError("The correct action and q value determination mode you chose seems to be wrong. If you tried implementing a new one, you should also update the check input function!")
            else:
                raise TypeError("The keyword arguments for the determination of the correct actions and q function need to be passed in a dictionary!")
        else:
            raise ValueError("You cannot specify keyword argument for the determination of action and q values without passing a determination mode!")

    # Safe mode needs to be boolean
    if "safe_mode" in kwargs.keys():
        if not isinstance(kwargs["safe_mode"],bool):
            raise TypeError("The parameter safe_mode needs to be boolean!")
        
    # Progress needs to be boolean
    if "progress" in kwargs.keys():
        if not isinstance(kwargs["progress"],bool):
            raise TypeError("The parameter progress needs to be boolean!")
    
    # Measure runtime needs to be boolean
    if "measure_runtime" in kwargs.keys():
        if not isinstance(kwargs["measure_runtime"],bool):
            raise TypeError("The parameter measure_runtime needs to be boolean!")
    
    return 1

# Utils for experiment manager

def is_lambda(obj):
    return isinstance(obj, types.LambdaType) and obj.__name__ == "<lambda>"

# Utils for plot functions

def check_input_for_results_single_to_batch_for_plot(result_paths: list[str], labels: list[str], output_folder: str, project_name: str, safe_mode: bool, conditional_plots: bool) -> None:

    """
        Validates the input parameters for the results_single_to_batch_for_plot function.

        Parameters:
        - result_paths (list): A list of paths to the folder in which the results.pkl files to be used can be found.
        - labels (str): A list of labels corresponding to the runs in the reslut_paths list. If there are not enough labels in the list to match
          all results in result path, the rest of the results will be assigned their respective paths as label.
        - output_folder (str): The folder in which the aggregated results should be saved.
        - project_name (str): The project name under which the aggregated results should be saved.
        - safe_mode (bool): If True, a check will be performed on the inputs. Additionally, it will be checked if the runs that will be aggregated
          are comparable in terms of parameters given to the execute_experiment function.

        Raises:
        - ValueError: If any of the input parameters are invalid.
        - TypeError: If any of the input types are invalid.
    """

    # conditional_plots needs to be boolean
    if not isinstance(conditional_plots,bool):
        raise TypeError("The parameter conditional_plots needs to be boolean!")

    # results_paths needs to be a list containing valid paths to folders containing the necessary files
    if isinstance(result_paths,list):
        if len(result_paths) == len(set(result_paths)):
            for path in result_paths:
                if isinstance(path,str):
                    if os.path.exists(path):
                        if not os.path.exists(os.path.join(path,"results.pkl")):
                            raise ValueError(f"The path {path} you provided contains no results file!")
                        if os.path.exists(os.path.join(path,"reproduce_run")):
                            if not os.path.exists(os.path.join(os.path.join(path,"reproduce_run"),"parameters.yaml")):
                                raise ValueError(f"The path {path} you provided contains no parameters file!")
                            if conditional_plots:
                                if not os.path.exists(os.path.join(path,"correct_policy_and_q_function.txt")):
                                    raise ValueError(f"The path {path} you provided contains no file containing the correct policy and Q function even though you want to plot plots requiring conditional statements that need them!")
                        else:
                            raise ValueError(f"The path {path} you provided contains no reproduce_run folder!")
                    else:
                        raise ValueError(f"The path {path} you provided does not exist!")
                else:
                    raise TypeError(f"The path {path} you provided is no string and thus invalid!")
        else:
            raise ValueError("The list of paths you provided contains doubles!")
    else:
        raise TypeError("The paths in result_paths need to be passed in a list!")
    
    # labels needs to be a list containing strings that are unique, match none of the results_paths and the list is not longer than result_paths
    if isinstance(labels,list):
        if len(labels) == len(set(labels)):
            if len(labels) <= len(result_paths):
                for label in labels:
                    if isinstance(label,str):
                        if label in result_paths:
                            raise ValueError(f"The label {label} matches one of the result paths and is thus not allowed!")
                    else:
                        raise TypeError(f"The label {label} is not a string and thus invalid!")
            else:
                raise ValueError("The list of labels is longer than the list of given paths to results!")
        else:
            raise ValueError("The list of labels you provided contains doubles!")
    else:
        raise TypeError("The labels in labels need to be passed in a list!")
    
    # output_folder needs to be a valid path
    if isinstance(output_folder,str):
        if not os.path.exists(output_folder):
            raise ValueError(f"The path {output_folder} you provided for saving the aggregated results does not exist!")
    else:
        raise TypeError("The path to the output folder needs to be a string!")
    
    # project_name needs to be a string
    if not isinstance(project_name,str):
        raise TypeError("The project name needs to be a string!")
    
    # safe_mode needs to be boolean
    if not isinstance(safe_mode,bool):
        raise TypeError("The parameter safe_mode needs to be boolean!")
    
def check_input_for_single_plot_fct(input_path: str, plot_folder: str, project_name: str,figsize: tuple[Union[int,float]],loc: Union[str,int],grid: bool,show: bool,save: bool,mode: Any ,safe_mode: bool) -> None:
    
    """
        Validates the input parameters for the different plot functions.

        Parameters:
        - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
          constructed from the given plot folder and project name.
        - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
          results to be plotted are located.
        - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
          which the results to be plotted are located.
        - figsize (tuple): A tuple of integers or float, specifying the width and height of the plot in inches.
        - loc (str): The location of the legend.
        - grid (bool): If True, the plot will exhibit a grid.
        - show (bool): If True, the plot will be shown.
        - save (bool): If True, the plot will be saved as a .png file.
        - mode (any): The mode for the plot. Can either be 'single plot', meaning the plot will be treated as a single plot, or can be a tuple 
          consisting of 'multiple plots' and an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
          different functions to unify plots with different data or plot all metrics in one figure). In the latter case save and show must be 
          turned off.
        - safe_mode (bool): If True, a parameter check will be performed.

        Raises:
        - ValueError: If any of the input parameters are invalid.
        - TypeError: If any of the input types are invalid.
    """

    # Input path needs to be None or a valid path to a pickle file
    if input_path != None:
        if isinstance(input_path,str):
            if os.path.isfile(input_path):
                if not input_path.endswith((".pkl",".pickle")):
                    raise ValueError(f"The given input path {input_path} does not point to a pickle file!")
            else:
                raise ValueError(f"The given input path {input_path} does not point to a file!")
        else:
            raise TypeError("The given input path is no string and thus invalid!")
    
    # Plot folder needs to be a string and point to an existing directory
    if isinstance(plot_folder,str):
        if not os.path.exists(plot_folder):
            raise ValueError(f"The given plot folder {plot_folder} does not exist!")
    else:
        raise TypeError("The given plot folder is no string and thus invalid!")
    
    # Project name needs to be a string
    if not isinstance(project_name,str):
        raise TypeError("The project name needs to be a string!")
    
    # If input path is none, the combination of plot folder and project name needs to lead to a pickle file
    if input_path == None:
        if not (os.path.exists(os.path.join(plot_folder,project_name + ".pkl")) or os.path.exists(os.path.join(plot_folder,project_name + ".pickle"))):
            raise ValueError("In case the input path was not specified, the plot folder and project name need to specify the location of a pickle file!")
    
    # Fig size needs to be tuple of numerical values
    if isinstance(figsize,tuple):
        if len(figsize) == 2:
            if not (isinstance(figsize[0],(int,float)) and isinstance(figsize[1],(int,float))):
                raise ValueError("The figure width and height need to be specified by numerical values!")
        else:
            raise TypeError("The figure width and height need to be specified by a tuple of length two!")
    else:
        raise TypeError("The figure size needs to be contained in a tuple!")
    
    # Location of the legend needs to be an allowed string or int in the range
    if isinstance(loc,str):
        allowed_locs = ["best", "upper right", "upper left", "lower right", "lower left", "center", "center left", "center right", "upper center", "lower center", "right"]
        if not (loc in allowed_locs):
            raise ValueError(f"The location {loc} of the legend is not allowed!")
    elif isinstance(loc,int):
        if not (0 <= loc <= 10):
            raise ValueError("The location of the legend can only be an integer between 0 and 10!")
    else:
        raise TypeError("The location of the legend needs to either be an integer or a string!")
    
    # grid needs to be boolean
    if not isinstance(grid,bool):
        raise TypeError("Parameter grid needs to be boolean!")
    
    # show needs to be boolean
    if not isinstance(show,bool):
        raise TypeError("Parameter show needs to be boolean!")
    
    # save needs to be boolean
    if not isinstance(save,bool):
        raise TypeError("Parameter save needs to be boolean!")
    
    # mode needs to be either "single plot" or tuple of "multiple plot", matplotlib axis, and boolean. If the latter, both show and save need to be disabled
    if mode != "single plot":
        if isinstance(mode,tuple):
            if len(mode) == 3:
                if not (mode[0] == "multiple plots" and isinstance(mode[1],matplotlib.axes.Axes) and (isinstance(mode[2],str) or mode[2] == None)):
                    raise ValueError("The mode can either be 'single plot' or a tuple consisting of 'multiple plots', a matplotlib axis, and a title!")
            else:
                raise ValueError("The mode can either be 'single plot' or a tuple consisting of 'multiple plots', a matplotlib axis, and a title!")
        else:
            raise ValueError("The mode can either be 'single plot' or a tuple consisting of 'multiple plots', a matplotlib axis, and a title!")
        
    # safe_mode needs to be boolean
    if not isinstance(safe_mode,bool):
        raise TypeError("Parameter safe_mode needs to be boolean!")

def parse_correct_policy_and_q_function(file_path:str)->tuple[list,dict]:
    policy_list = []
    q_function_dict = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse the "Estimated correct Policy" section
    policy_start = lines.index("Estimated correct Policy:\n") + 2
    for line in lines[policy_start:]:
        if line.strip() == "":  # Stop at the empty line
            break
        _, values = line.split(":")
        policy_list.append(eval(values.strip()))
    
    # Parse the "Estimated correct Q Function" section
    q_function_start = lines.index("Estimated correct Q Function:\n") + 1
    for line in lines[q_function_start:]:
        if line.strip():  # Ignore empty lines
            key, value = line.split(":")
            q_function_dict[eval(key.strip())] = float(value.strip())
    
    return policy_list, q_function_dict