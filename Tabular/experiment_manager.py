from typing import Dict, List, Union, Tuple, Any
from tqdm import tqdm
import os
import pickle
import yaml
import time
from copy import deepcopy
import numpy as np
import inspect

import algos
import envs
from train import train
from utils import check_input_for_train, is_lambda
import plots

def execute_experiment(
        # Parameters for executing experiment
        base_folder: str = "results",
        num_runs: int = 1000,
        progress: bool = True,
        project_name: str = "testproject",
        runtime_estimation: bool = True,
        safe_mode: bool = True,
        verbose: bool = True,
        # Parameters for controlling the individual training steps
        algo: algos.Algo = None,
        algo_special_logs: bool = False,
        algo_special_logs_kwargs: Dict = None,
        bias_estimation: bool = True,
        correct_act_q_fct_mode: str = "value_iteration",
        correct_act_q_fct_mode_kwargs: Dict[str,Any] = None,
        correct_action_log: bool = True,
        correct_action_log_which: Union[str,List] = "all",
        env: envs.Env = None,
        eval_reseeding: bool = True,
        eval_seed_schedule: List[Union[int,float]] = None,
        eval_steps: int = 7,
        eval_freq: int = 10,
        focus_state_actions: bool = True,
        max_steps_per_epoch: int = -1,
        num_steps: int = 500,
        policy: algos.Policy = None,
        progress_single_games: bool = True,
        training_mode: str = "steps",
        training_reseeding: bool = False,
        training_seed_schedule: List[Union[int,float]] = None,
        which_state_actions_focus: Tuple[List[int], List] = None,
        # 3. Parameters to pass to the algorithm
        algo_specific_params: Dict = None,
        gamma: float = 0.99,
        learning_rate_kwargs: Dict = None,
        learning_rate_state_action_wise: bool = True,
        # 4. Parameters to pass to the environment
        env_specific_params: Dict = None,
        # 4. Parameters to pass to the policy
        policy_specific_params: Dict = None,
) -> str:
    
    """
    Executes an algorithm on an environment a specified amount of times. Saves all gathered data to a pickle file in a specified location for plotting.

    Parameters:
    All parameters are explained in detail in the experiment_manager_parameter_guide.md file.

    Returns:
    The path where the experiment's data is stored.
    The estimated correct policy
    The estimated correct Q function
    """

    # Default arguments if arguments are None
    if algo is None:
        algo = algos.Q
    if which_state_actions_focus is None:
        which_state_actions_focus = ([0],[["best"]])
    if correct_act_q_fct_mode_kwargs is None:
        correct_act_q_fct_mode_kwargs = {}
    if env is None:
        env = envs.GridWorld
    if eval_seed_schedule is None:
        eval_seed_schedule = [-1]
    if policy is None:
        policy = algos.BasePolicy
    if training_seed_schedule is None:
        training_seed_schedule = [-1]    
    if algo_specific_params is None:
        algo_specific_params = {}
    if learning_rate_kwargs is None:
        learning_rate_kwargs = {}
    if env_specific_params is None:
        env_specific_params = {}
    if policy_specific_params is None:
        policy_specific_params = {}
    if algo_special_logs_kwargs is None:
        algo_special_logs_kwargs = {}
    
    # Welcome message and start preprocessing if verbose
    if verbose:
        print("="*50)

        print(f"Training algo {algo().__str__()} on environment {env().__str__()}.")

        iterator_preprocessing = tqdm(total = 5, desc="Preprocessing", leave=False)
    
    if safe_mode:

        if verbose:

            # Start input checking
            iterator_preprocessing.write("Check input parameters ... ")

        # Allowed kwarg keys check prerequisites
        algo_specific_params_allowed = {
            algos.Q: ["q_fct_manual_init", "initial_q_fct"],
            algos.Double: ["q_fct_manual_init", "initial_q_fct"],
            algos.WDQ: ["q_fct_manual_init", "initial_q_fct", "interpolation_mode", "interpolation_mode_kwargs"],
            algos.CategoricalQ: ["num_atoms", "range_atoms", "atom_probs_manual_init", "initial_atom_probs"],
            algos.CategoricalDouble: ["num_atoms", "range_atoms", "atom_probs_manual_init", "initial_atom_probs"],
            algos.ADDQ: ["num_atoms", "range_atoms", "atom_probs_manual_init", "initial_atom_probs", "interpolation_mode", "interpolation_mode_kwargs"],
        }
        env_specific_params_allowed = {
            envs.GridWorld: ["grid_size", "state_type_loc", "rewards", "hovering", "windy", "wind_prob", "wind_dir", "slippery", "slip_prob", "random_actions", "random_prob", "random_vec"],
            envs.SuttonExtended: ["grid_size", "num_arms", "rewards"],
        }
        policy_specific_params_allowed = {
            algos.BasePolicy: ["policy_mode", "policy_mode_kwargs"],
        }

        # algo_specific_params is dict and keywords are allowed
        if isinstance(algo_specific_params,dict):
            for key in algo_specific_params.keys():
                if algo in algo_specific_params_allowed.keys():
                    if not key in algo_specific_params_allowed[algo]:
                        raise ValueError(f"Key {key} is not allowed for algorithm {algo()}!")
                else:
                    print("Warning: The algorithm you chose is not yet registered with its allowed inputs in the execute_experiment function. Please register it properly or proceed at own risk!")
        else:
            raise TypeError("Parameter algo_specific_params needs to be a dictionary!")
        
        # env_specific_params is dict and keywords are allowed
        if isinstance(env_specific_params,dict):
            for key in env_specific_params.keys():
                if env in env_specific_params_allowed.keys():
                    if not key in env_specific_params_allowed[env]:
                        raise ValueError(f"Key {key} is not allowed for environment {env()}!")
                else:
                    print("Warning: The environment you chose is not yet registered with its allowed inputs in the execute_experiment function. Please register it properly or proceed at own risk!")
        else:
            raise TypeError("Parameter env_specific_params needs to be a dictionary!")
        
        # policy_specific_params is dict and keywords are allowed
        if isinstance(policy_specific_params,dict):
            for key in policy_specific_params.keys():
                if policy in policy_specific_params_allowed.keys():
                    if not key in policy_specific_params_allowed[policy]:
                        raise ValueError(f"Key {key} is not allowed for policy {policy()}!")
                else:
                    print("Warning: The policy you chose is not yet registered with its allowed inputs in the execute_experiment function. Please register it properly or proceed at own risk!")
        else:
            raise TypeError("Parameter policy_specific_params needs to be a dictionary!")
        
        # Initialize string to be printed in case of errors
        error_string = "The following error messages were found during the input check:\n"
        error_string_print = False
        error_counter = 0

        # Aggregate kwargs and check if environment can be set up with the given parameters
        try:
            env(**env_specific_params)
        except (TypeError,ValueError) as e:
            error_counter +=1
            error_string = error_string + f"Error message {error_counter}: {e}\n"
            error_string_print = True  

        # Aggregate kwargs and check if policy can be set up with the given parameters
        try:
            policy(**policy_specific_params)
        except (TypeError,ValueError) as e:
            error_counter +=1
            error_string = error_string + f"Error message {error_counter}: {e}\n"
            error_string_print = True 

        # Aggregate kwargs and check if algo can be set up with the given parameters
        check_algo_kwargs = deepcopy(algo_specific_params)
        if learning_rate_kwargs != {}:
            check_algo_kwargs["learning_rate_kwargs"] = learning_rate_kwargs
        check_algo_kwargs["learning_rate_state_action_wise"] = learning_rate_state_action_wise
        check_algo_kwargs["gamma"] = gamma
        try:
            algo(env=env,env_kwargs=env_specific_params ,**check_algo_kwargs)
        except (TypeError,ValueError) as e:
            error_counter +=1
            error_string = error_string + f"Error message {error_counter}: {e}\n"
            error_string_print = True

        # Aggregate kwargs and check if train can be done with the given parameters 
        check_train_kwargs = {}
        check_train_kwargs["algo_special_logs"] = algo_special_logs
        check_train_kwargs["algo_special_logs_kwargs"] = algo_special_logs_kwargs
        check_train_kwargs["bias_estimation"] = bias_estimation
        check_train_kwargs["which_state_actions_focus"] = which_state_actions_focus
        check_train_kwargs["focus_state_actions"] = focus_state_actions
        check_train_kwargs["correct_act_q_fct_mode"] = correct_act_q_fct_mode
        if correct_act_q_fct_mode_kwargs != {}:
            check_train_kwargs["correct_act_q_fct_mode_kwargs"] = correct_act_q_fct_mode_kwargs
        check_train_kwargs["correct_action_log"] = correct_action_log
        check_train_kwargs["correct_action_log_which"] = correct_action_log_which
        check_train_kwargs["eval_reseeding"] = eval_reseeding
        check_train_kwargs["eval_seed_schedule"] = eval_seed_schedule.copy()
        check_train_kwargs["eval_steps"] = eval_steps
        check_train_kwargs["eval_freq"] = eval_freq
        check_train_kwargs["max_steps_per_epoch"] = max_steps_per_epoch
        check_train_kwargs["num_steps"] = num_steps
        check_train_kwargs["progress"] = progress_single_games
        check_train_kwargs["training_mode"] = training_mode
        check_train_kwargs["training_reseeding"] = training_reseeding
        check_train_kwargs["training_seed_schedule"] = training_seed_schedule.copy()
        try:
            check_input_for_train(**check_train_kwargs)
        except (TypeError,ValueError) as e:
            error_counter +=1
            error_string = error_string + f"Error message {error_counter}: {e}\n"
            error_string_print = True

        # Check the experiment_manager inputs
        if isinstance(base_folder,str):
            if not os.path.exists(base_folder):
                error_counter +=1
                error_string = error_string + f"Error message {error_counter}: The parameter base_folder needs to be a string containing a valid folder path!\n"
                error_string_print = True
        else:
            error_counter +=1
            error_string = error_string + f"Error message {error_counter}: The parameter base_folder needs to be a string!\n"
            error_string_print = True
        if isinstance(num_runs,int):
            if not num_runs > 0:
                error_counter +=1
                error_string = error_string + f"Error message {error_counter}: The parameter num_runs needs to be a positive integer!\n"
                error_string_print = True
        else:
            error_counter +=1
            error_string = error_string + f"Error message {error_counter}: The parameter num_runs needs to be an integer!\n"
            error_string_print = True
        if not isinstance(progress,bool):
            error_counter +=1
            error_string = error_string + f"Error message {error_counter}: The parameter progress needs to be boolean!\n"
            error_string_print = True
        if not isinstance(project_name,str):
            error_counter +=1
            error_string = error_string + f"Error message {error_counter}: The parameter project_name needs to be a string!\n"
            error_string_print = True
        if not isinstance(runtime_estimation,bool):
            error_counter +=1
            error_string = error_string + f"Error message {error_counter}: The parameter runtime_estimation needs to be boolean!\n"
            error_string_print = True
        if not isinstance(safe_mode,bool):
            error_counter += 1
            error_string = error_string + f"Error message {error_counter}: The parameter safe_mode needs to be boolean!\n"
            error_string_print = True
        if not isinstance(verbose,bool):
            error_counter += 1
            error_string = error_string + f"Error message {error_counter}: The parameter verbose needs to be boolean!\n"
            error_string_print = True

        # Finalize the error string and print it if needed
        error_string = error_string + "As the checks rely on initializing the algorithm, environment, and policy, if there is an error message connected to their initialization, there may be more than one error, though only the first one to be encountered will be printed in this case!"
        # If errors were found, print the error message and raise an error
        if error_string_print:
            raise ValueError(f"{error_string}")

    if verbose:

        # Input check done
        time.sleep(0.5)
        iterator_preprocessing.update(1)

        # Start running pretrial
        iterator_preprocessing.write("Run preprocessing run ... ")

    # Assemble all relevant inputs for train.py
    preprocessing_train_kwargs = check_train_kwargs
    preprocessing_train_kwargs["algo"] = algo
    algo_kwargs = check_algo_kwargs
    algo_kwargs["policy"] = policy
    algo_kwargs["policy_kwargs"] = policy_specific_params
    preprocessing_train_kwargs["algo_kwargs"] = algo_kwargs
    preprocessing_train_kwargs["env"] = env
    preprocessing_train_kwargs["env_kwargs"] = env_specific_params
    saving_train_kwargs = deepcopy(preprocessing_train_kwargs)
    preprocessing_train_kwargs["safe_mode"] = False
    preprocessing_train_kwargs["progress"] = False
    if runtime_estimation:
        preprocessing_train_kwargs["measure_runtime"] = True

    if not algo_special_logs:
        _, _, estimated_correct_policy, estimated_correct_q_fct, _, _, preprocess_runtime = train(**preprocessing_train_kwargs)
    else:
        _, _, estimated_correct_policy, estimated_correct_q_fct, results_for_algo_special_logs, _, preprocess_runtime = train(**preprocessing_train_kwargs)

    if preprocess_runtime < 0.1:
        progress_single_games = False
    
    if verbose:

        # Preruns done
        time.sleep(0.5)
        iterator_preprocessing.update(1)

        # Start setting up 
        iterator_preprocessing.write("Set up target directory ...")
    
    # Set up path and files
    if os.path.exists(base_folder):

        # Create path to project-algorithm directory if not already existing
        project_path = os.path.join(base_folder,project_name)
        os.makedirs(project_path,exist_ok=True)
        project_algo_path = os.path.join(project_path, algo().__str__())
        os.makedirs(project_algo_path,exist_ok=True)

        # Get the unique directory name for the run folder
        counter = 0
        unique_foldername = f"{env()}_{counter}"
        while os.path.exists(os.path.join(project_algo_path,unique_foldername)):
            counter += 1
            unique_foldername = f"{env()}_{counter}"
        save_path = os.path.join(project_algo_path,unique_foldername)
        os.makedirs(save_path,exist_ok=False)

        # Create the reproducability folder with its files and the results file
        with open(os.path.join(save_path,"results.pkl"), "a"):
            pass
        os.makedirs(os.path.join(save_path,"reproduce_run"),exist_ok=False)
        with open(os.path.join(os.path.join(save_path,"reproduce_run"),"seeds.pkl"), "a"):
            pass
        with open(os.path.join(os.path.join(save_path,"reproduce_run"),"parameters.yaml"), "a"):
            pass
        if not (len(estimated_correct_policy) == 0 or len(estimated_correct_q_fct) == 0):
            with open(os.path.join(save_path,"correct_policy_and_q_function.txt"), "a"):
                pass
    else:
        raise ValueError("The provided base path seems to not exists!")

    if verbose:

        # Setting up path and files done
        time.sleep(0.5)
        iterator_preprocessing.update(1)

        # Start save preprocessing run results
        iterator_preprocessing.write("Save preprocessing run result ...")


    # Estimate runtime from preprocessing run
    if runtime_estimation:
        estimated_runtime = preprocess_runtime * num_runs
        if estimated_runtime < 60:
            estimated_runtime_unit = "s"
            estimated_runtime = round(estimated_runtime,2)
        elif estimated_runtime < 3600:
            estimated_runtime_unit = "min"
            estimated_runtime = round(estimated_runtime / 60,2)
        else:
            estimated_runtime_unit = "h"
            estimated_runtime = round(estimated_runtime / 3600,2)

    # Save parameters in yaml file, produce yaml default readable types
    saving_train_kwargs["algo"] = algo().__str__()
    saving_train_kwargs["env"] = env().__str__()
    saving_train_kwargs["algo_kwargs"]["policy"] = policy().__str__()
    if saving_train_kwargs["algo_kwargs"]["learning_rate_kwargs"] != {}:
        if saving_train_kwargs["algo_kwargs"]["learning_rate_kwargs"]["mode"] == "rate":
            if is_lambda(saving_train_kwargs["algo_kwargs"]["learning_rate_kwargs"]["mode_kwargs"]["rate_fct"]):
                start_index = inspect.getsource(saving_train_kwargs["algo_kwargs"]["learning_rate_kwargs"]["mode_kwargs"]["rate_fct"]).strip().find("lambda")
                end_index = inspect.getsource(saving_train_kwargs["algo_kwargs"]["learning_rate_kwargs"]["mode_kwargs"]["rate_fct"]).strip()[start_index:].find(",")
                append_text = inspect.getsource(saving_train_kwargs["algo_kwargs"]["learning_rate_kwargs"]["mode_kwargs"]["rate_fct"]).strip()[start_index:][:end_index]
                saving_train_kwargs["algo_kwargs"]["learning_rate_kwargs"]["mode_kwargs"]["rate_fct"] = append_text
    if saving_train_kwargs["algo_kwargs"]["policy_kwargs"] != {}:
        if saving_train_kwargs["algo_kwargs"]["policy_kwargs"]["policy_mode"] == "epsilon_greedy" or saving_train_kwargs["algo_kwargs"]["policy_kwargs"]["policy_mode"] == "epsilon_greedy_statewise":
            if saving_train_kwargs["algo_kwargs"]["policy_kwargs"]["policy_mode_kwargs"] != {}:
                if saving_train_kwargs["algo_kwargs"]["policy_kwargs"]["policy_mode_kwargs"]["mode"] == "rate":
                    if is_lambda(saving_train_kwargs["algo_kwargs"]["policy_kwargs"]["policy_mode_kwargs"]["mode_kwargs"]["rate_fct"]):
                        start_index = inspect.getsource(saving_train_kwargs["algo_kwargs"]["policy_kwargs"]["policy_mode_kwargs"]["mode_kwargs"]["rate_fct"]).strip().find("lambda")
                        end_index = inspect.getsource(saving_train_kwargs["algo_kwargs"]["policy_kwargs"]["policy_mode_kwargs"]["mode_kwargs"]["rate_fct"]).strip()[start_index:].find(",")
                        append_text = inspect.getsource(saving_train_kwargs["algo_kwargs"]["policy_kwargs"]["policy_mode_kwargs"]["mode_kwargs"]["rate_fct"]).strip()[start_index:][:end_index]
                        saving_train_kwargs["algo_kwargs"]["policy_kwargs"]["policy_mode_kwargs"]["mode_kwargs"]["rate_fct"] = append_text
    del saving_train_kwargs["progress"]
    if correct_act_q_fct_mode_kwargs == {}:
        saving_train_kwargs["correct_act_q_fct_mode_kwargs"] = correct_act_q_fct_mode_kwargs
    parameters_to_save = {"config": {"num_runs": num_runs}, "args": saving_train_kwargs}
    with open(os.path.join(os.path.join(save_path,"reproduce_run"),"parameters.yaml"), "w") as file:
        yaml.dump(parameters_to_save, file)
        
    # Save estimated correct policy and correct q fct
    if not (len(estimated_correct_policy) == 0 or len(estimated_correct_q_fct) == 0):
        with open(os.path.join(save_path,"correct_policy_and_q_function.txt"), "a") as file:
            file.write("Estimated correct Policy:\n\n")
            for index, value in enumerate(estimated_correct_policy):
                file.write(f"State {index}: {value}\n")
            file.write("\nEstimated correct Q Function:\n\n")
            for key, value in estimated_correct_q_fct.items():
                file.write(f"{key}: {value}\n")

    if verbose:

        # Save preprocessing run results done
        time.sleep(0.5)
        iterator_preprocessing.update(1)

        # Start data structure initialization
        iterator_preprocessing.write("Data structure initialization ...")

    # Initialize lists, where the seeds and mean values will be saved into
    used_training_seeds = []
    used_eval_seeds = []

    seen_steps = []
    steps_num_reached = []
    mean_steps_reward = []

    seen_epoch_nums = []
    epoch_num_reached = []
    mean_epoch_durations = []
    if max_steps_per_epoch != -1:
        percent_of_capped_epochs = []
    mean_epoch_results = []
    if correct_action_log:
        mean_epoch_correct_action_rates = []
    
    seen_eval_times = []
    eval_times_reached = []
    mean_evaluation_scores = []
    mean_number_reached_terminal_states_during_eval = []
    mean_time_reached_terminal_states_during_eval = []
    if correct_action_log:
        mean_eval_correct_action_rates = []
        mean_eval_correct_action_rates_at_chosen = []
    if focus_state_actions:
        eval_q_values_at_chosen = []
        eval_biases_at_chosen = []
    if bias_estimation:
        total_evaluation_biases = []
        total_squared_evaluation_biases = []
        total_evaluation_biases_best_arms = []
        total_squared_evaluation_biases_best_arms = []
        total_normalized_evaluation_biases = []
        total_squared_normalized_evaluation_biases = []
        total_normalized_evaluation_biases_best_arms = []
        total_squared_normalized_evaluation_biases_best_arms = []
    
    if algo_special_logs:
        special = {}
        if "special" in results_for_algo_special_logs.keys():
            if "at_step" in results_for_algo_special_logs["special"].keys():
                special["at_step"] = {}
                for label in results_for_algo_special_logs["special"]["at_step"]:
                    special["at_step"][label] = []
            if "at_epoch" in results_for_algo_special_logs["special"].keys():
                special["at_epoch"] = {}
                for label in results_for_algo_special_logs["special"]["at_epoch"]:
                    special["at_epoch"][label] = []
            if "at_eval" in results_for_algo_special_logs["special"].keys():
                special["at_eval"] = {}
                for label in results_for_algo_special_logs["special"]["at_eval"]:
                    special["at_eval"][label] = []


    # Initialize dictionary to be stepwise copied for parameter input into train
    train_params = preprocessing_train_kwargs
    train_params["correct_act_q_fct_mode"] = "manual"
    train_params["correct_act_q_fct_mode_kwargs"] = {"correct_actions": estimated_correct_policy, "correct_q_fct": estimated_correct_q_fct}
    train_params["progress"] = progress_single_games
    train_params["measure_runtime"] = False

    print_replication_warning = True

    if verbose:

        # Preprocessing completed
        time.sleep(0.5)
        iterator_preprocessing.update(1)

        # Training starts
        print("Preprocessing is complete. The training evaluation for the chosen algorithm starts now.")

        if runtime_estimation:
            print(f"The estimated runtime is: {estimated_runtime} {estimated_runtime_unit}.")
        
        print("="*50)

        iterator_preprocessing.close()

    # Start measuring runtime for the experiment
    start_time = time.time()

    # If progress, show the bar
    if progress:
        iterator_execution = tqdm(total = num_runs, desc="Training algorithm", leave=False)

    # For all runs do
    for i in range(num_runs):

        if i > 0:
            # Update evaluation and training seed lists to remove all seeds that have been used.
            if len(train_params["training_seed_schedule"]) > 1 or (len(train_params["training_seed_schedule"]) == 1 and train_params["training_seed_schedule"][0] != -1):
                for t_seed in execution_results["at_epoch"]["training seeds"]:
                    if len(train_params["training_seed_schedule"]) == 1 and train_params["training_seed_schedule"] == -1:
                        break
                    if t_seed == train_params["training_seed_schedule"][0] or train_params["training_seed_schedule"][0] == -1:
                        del train_params["training_seed_schedule"][0]
                    else:
                        if print_replication_warning:
                            iterator_execution.write("Something weird happened. Replication by seed schedule can not be guaranteed!")
                            print_replication_warning = False
                        del train_params["training_seed_schedule"][0]
                    if len(train_params["training_seed_schedule"]) == 0:
                        train_params["training_seed_schedule"].append(-1)
            if len(train_params["eval_seed_schedule"]) > 1 or (len(train_params["eval_seed_schedule"]) == 1 and train_params["eval_seed_schedule"][0] != -1):
                for e_seed in execution_results["at_eval"]["evaluation seeds"]:
                    if len(train_params["eval_seed_schedule"]) == 1 and train_params["eval_seed_schedule"] == -1:
                        break
                    if e_seed == train_params["eval_seed_schedule"][0] or train_params["eval_seed_schedule"][0] == -1:
                        del train_params["eval_seed_schedule"][0]
                    else:
                        if print_replication_warning:
                            iterator_execution.write("Something weird happened. Replication by seed schedule can not be guaranteed!")
                            print_replication_warning = False
                        del train_params["eval_seed_schedule"][0]
                    if len(train_params["eval_seed_schedule"]) == 0:
                        train_params["eval_seed_schedule"].append(-1)

        # Let the train function run with the copied arguments, receive the results, final policy, and final Q function
        _, _, _,  _, execution_results, _, _ = train(**train_params)

        # Append the seeds to the seed lists
        used_training_seeds.extend(execution_results["at_epoch"]["training seeds"])
        used_eval_seeds.extend(execution_results["at_eval"]["evaluation seeds"])

        # Aggregate Results at steps
        if len(execution_results["at_step"]["timesteps"]) > len(seen_steps):
            start_appending_index = len(seen_steps)
            seen_steps.extend(execution_results["at_step"]["timesteps"][start_appending_index:])
        else:
            start_appending_index = len(execution_results["at_step"]["timesteps"])
        for index, _ in enumerate(execution_results["at_step"]["timesteps"]):
            if index < start_appending_index:
                steps_num_reached[index] += 1
                mean_steps_reward[index] += execution_results["at_step"]["rewards"][index]
            else:
                steps_num_reached.append(1)
                mean_steps_reward.append(execution_results["at_step"]["rewards"][index])

        # Aggregate special results if turned on
        if algo_special_logs:
            if special != {}:
                if "at_step" in special.keys():
                    for index, _ in enumerate(execution_results["at_step"]["timesteps"]):
                        for label in execution_results["special"]["at_step"].keys():
                            if index < start_appending_index:
                                special["at_step"][label][index] += execution_results["special"]["at_step"][label][index]
                            else:
                                special["at_step"][label].append(execution_results["special"]["at_step"][label][index])

        # Aggregate Results at epoch
        if len(execution_results["at_epoch"]["epoch numbers"]) > len(seen_epoch_nums):
            start_appending_index = len(seen_epoch_nums)
            seen_epoch_nums.extend(execution_results["at_epoch"]["epoch numbers"][start_appending_index:])
        else:
            start_appending_index = len(execution_results["at_epoch"]["epoch numbers"])
        for index, _ in enumerate(execution_results["at_epoch"]["epoch numbers"]):
            if index < start_appending_index:
                epoch_num_reached[index] += 1
                mean_epoch_durations[index] += execution_results["at_epoch"]["epoch durations"][index]
                if max_steps_per_epoch != -1:
                    percent_of_capped_epochs[index] += execution_results["at_epoch"]["capped"][index]
                mean_epoch_results[index] += execution_results["at_epoch"]["epoch results"][index]
                if correct_action_log:
                    mean_epoch_correct_action_rates[index] += execution_results["at_epoch"]["correct action rates"][index]
            else:
                epoch_num_reached.append(1)
                mean_epoch_durations.append(execution_results["at_epoch"]["epoch durations"][index])
                if max_steps_per_epoch != -1:
                    percent_of_capped_epochs.append(execution_results["at_epoch"]["capped"][index])
                mean_epoch_results.append(execution_results["at_epoch"]["epoch results"][index])
                if correct_action_log:
                    mean_epoch_correct_action_rates.append(execution_results["at_epoch"]["correct action rates"][index])

        # Aggregate special results if turned on
        if algo_special_logs:
            if special != {}:
                if "at_epoch" in special.keys():
                    for index, _ in enumerate(execution_results["at_epoch"]["epoch numbers"]):
                        for label in execution_results["special"]["at_epoch"].keys():
                            if index < start_appending_index:
                                special["at_epoch"][label][index] += execution_results["special"]["at_epoch"][label][index]
                            else:
                                special["at_epoch"][label].append(execution_results["special"]["at_epoch"][label][index])

        # Aggregate Results at eval
        if len(execution_results["at_eval"]["evaluation times"]) > len(seen_eval_times):
            start_appending_index = len(seen_eval_times)
            seen_eval_times.extend(execution_results["at_eval"]["evaluation times"][start_appending_index:])
        else:
            start_appending_index = len(execution_results["at_eval"]["evaluation times"])
        for index, _ in enumerate(execution_results["at_eval"]["evaluation times"]):
            if index < start_appending_index:
                eval_times_reached[index] += 1
                mean_evaluation_scores[index] += execution_results["at_eval"]["evaluation scores"][index][0]
                mean_number_reached_terminal_states_during_eval[index] += execution_results["at_eval"]["evaluation scores"][index][1]
                mean_time_reached_terminal_states_during_eval[index] += execution_results["at_eval"]["evaluation scores"][index][2]
                if correct_action_log:
                    mean_eval_correct_action_rates[index] += execution_results["at_eval"]["correct action rates at evaluations"][index]
                    mean_eval_correct_action_rates_at_chosen[index] += execution_results["at_eval"]["correct action rates at chosen states"][index]
                if focus_state_actions:
                    for state_index, hyperlist in enumerate(execution_results["at_eval"]["evaluation biases at chosen states and actions"][index]):
                        for arm_index, bias_value in enumerate(hyperlist):
                            eval_biases_at_chosen[index][state_index][arm_index] += bias_value
                    for state_index, hyperlist in enumerate(execution_results["at_eval"]["Q function values at chosen states and actions"][index]):
                        for arm_index, q_fct_value in enumerate(hyperlist):
                            eval_q_values_at_chosen[index][state_index][arm_index] += q_fct_value
                if bias_estimation:
                    total_evaluation_biases[index] += execution_results["at_eval"]["total evaluation biases"][index]
                    total_squared_evaluation_biases[index] += execution_results["at_eval"]["total squared evaluation biases"][index]
                    total_evaluation_biases_best_arms[index] += execution_results["at_eval"]["total evaluation biases at best arms"][index]
                    total_squared_evaluation_biases_best_arms[index] += execution_results["at_eval"]["total squared evaluation biases at best arms"][index]
                    total_normalized_evaluation_biases[index] += execution_results["at_eval"]["total normalized evaluation biases"][index]
                    total_squared_normalized_evaluation_biases[index] += execution_results["at_eval"]["total squared normalized evaluation biases"][index]
                    total_normalized_evaluation_biases_best_arms[index] += execution_results["at_eval"]["total normalized evaluation biases at best arms"][index]
                    total_squared_normalized_evaluation_biases_best_arms[index] += execution_results["at_eval"]["total squared normalized evaluation biases at best arms"][index]
            else:
                eval_times_reached.append(1)
                mean_evaluation_scores.append(execution_results["at_eval"]["evaluation scores"][index][0])
                mean_number_reached_terminal_states_during_eval.append(execution_results["at_eval"]["evaluation scores"][index][1])
                mean_time_reached_terminal_states_during_eval.append(execution_results["at_eval"]["evaluation scores"][index][2])
                if correct_action_log:
                    mean_eval_correct_action_rates.append(execution_results["at_eval"]["correct action rates at evaluations"][index])
                    mean_eval_correct_action_rates_at_chosen.append(execution_results["at_eval"]["correct action rates at chosen states"][index])
                if focus_state_actions:
                    eval_biases_at_chosen.append(execution_results["at_eval"]["evaluation biases at chosen states and actions"][index])
                    eval_q_values_at_chosen.append(execution_results["at_eval"]["Q function values at chosen states and actions"][index])
                if bias_estimation:
                    total_evaluation_biases.append(execution_results["at_eval"]["total evaluation biases"][index])
                    total_squared_evaluation_biases.append(execution_results["at_eval"]["total squared evaluation biases"][index])
                    total_evaluation_biases_best_arms.append(execution_results["at_eval"]["total evaluation biases at best arms"][index])
                    total_squared_evaluation_biases_best_arms.append(execution_results["at_eval"]["total squared evaluation biases at best arms"][index])
                    total_normalized_evaluation_biases.append(execution_results["at_eval"]["total normalized evaluation biases"][index])
                    total_squared_normalized_evaluation_biases.append(execution_results["at_eval"]["total squared normalized evaluation biases"][index])
                    total_normalized_evaluation_biases_best_arms.append(execution_results["at_eval"]["total normalized evaluation biases at best arms"][index])
                    total_squared_normalized_evaluation_biases_best_arms.append(execution_results["at_eval"]["total squared normalized evaluation biases at best arms"][index])

        # Aggregate special results if turned on
        if algo_special_logs:
            if special != {}:
                if "at_eval" in special.keys():
                    for index, _ in enumerate(execution_results["at_eval"]["evaluation times"]):
                        for label in execution_results["special"]["at_eval"].keys():
                            if index < start_appending_index:
                                special["at_eval"][label][index] += execution_results["special"]["at_eval"][label][index]
                            else:
                                special["at_eval"][label].append(execution_results["special"]["at_eval"][label][index])

        # Update the progress bar
        iterator_execution.update(1)
    
    # End measuring runtime for the experiment
    end_time = time.time()
    experiment_runtime = end_time - start_time
    if experiment_runtime < 60:
        experiment_runtime_unit = "s"
        experiment_runtime_with_unit = round(experiment_runtime,2)
    elif experiment_runtime < 3600:
        experiment_runtime_unit = "min"
        experiment_runtime_with_unit = round(experiment_runtime / 60,2)
    else:
        experiment_runtime_unit = "h"
        experiment_runtime_with_unit = round(experiment_runtime / 3600,2)

    if verbose:

        time.sleep(0.5)
        steps_left = iterator_execution.total - iterator_execution.n
        iterator_execution.update(steps_left)

        # Postprocessing starts
        print(f"Training evaluation completed in {experiment_runtime_with_unit} {experiment_runtime_unit}. Postprocessing starts now.")

        
        print("="*50)

        iterator_execution.close()

        iterator_postprocessing = tqdm(total = 3, desc="Postprocessing", leave=False)

        # Saving seeds starts
        iterator_postprocessing.write("Saving seeds ...")
    
    # Set up seeds dictionary and save it
    seeds = {"eval": used_eval_seeds, "training": used_training_seeds}
    with open(os.path.join(os.path.join(save_path,"reproduce_run"),"seeds.pkl"), "wb") as file:
        pickle.dump(seeds,file)

    if verbose:

        # Saving seeds done
        time.sleep(0.5)
        iterator_postprocessing.update(1)

        # Processing results starts 
        iterator_postprocessing.write("Processing results ...")

    # Processing step results
    for index, _ in enumerate(seen_steps):
        mean_steps_reward[index] = mean_steps_reward[index] / steps_num_reached[index]

    # Processing epoch results
    for index, _ in enumerate(seen_epoch_nums):
        mean_epoch_durations[index] = mean_epoch_durations[index] / epoch_num_reached[index]
        if max_steps_per_epoch != -1:
            percent_of_capped_epochs[index] = percent_of_capped_epochs[index] / epoch_num_reached[index]
        mean_epoch_results[index] = mean_epoch_results[index] / epoch_num_reached[index]
        if correct_action_log:
            mean_epoch_correct_action_rates[index] = mean_epoch_correct_action_rates[index] / epoch_num_reached[index]

    # Processing eval results
    for index, _ in enumerate(seen_eval_times):
        mean_evaluation_scores[index] = mean_evaluation_scores[index] / eval_times_reached[index]
        mean_number_reached_terminal_states_during_eval[index] = mean_number_reached_terminal_states_during_eval[index] / eval_times_reached[index]
        mean_time_reached_terminal_states_during_eval[index] = mean_time_reached_terminal_states_during_eval[index] / eval_times_reached[index]
        if correct_action_log:
            mean_eval_correct_action_rates[index] = mean_eval_correct_action_rates[index] / eval_times_reached[index]
            mean_eval_correct_action_rates_at_chosen[index] = mean_eval_correct_action_rates_at_chosen[index] / eval_times_reached[index]
        if focus_state_actions:
            for state_index, hyperlist in enumerate(eval_biases_at_chosen[index]):
                for arm_index, _ in enumerate(hyperlist):
                    eval_biases_at_chosen[index][state_index][arm_index] = eval_biases_at_chosen[index][state_index][arm_index] / eval_times_reached[index]
            for state_index, hyperlist in enumerate(eval_q_values_at_chosen[index]):
                for arm_index, _ in enumerate(hyperlist):
                    eval_q_values_at_chosen[index][state_index][arm_index] = eval_q_values_at_chosen[index][state_index][arm_index] / eval_times_reached[index]
        if bias_estimation:
            total_evaluation_biases[index] = total_evaluation_biases[index] / eval_times_reached[index]
            total_squared_evaluation_biases[index] = total_squared_evaluation_biases[index] / eval_times_reached[index]
            total_evaluation_biases_best_arms[index] = total_evaluation_biases_best_arms[index] / eval_times_reached[index]
            total_squared_evaluation_biases_best_arms[index] = total_squared_evaluation_biases_best_arms[index] / eval_times_reached[index]
            total_normalized_evaluation_biases[index] = total_normalized_evaluation_biases[index] / eval_times_reached[index]
            total_squared_normalized_evaluation_biases[index] = total_squared_normalized_evaluation_biases[index] / eval_times_reached[index]
            total_normalized_evaluation_biases_best_arms[index] = total_normalized_evaluation_biases_best_arms[index] / eval_times_reached[index]
            total_squared_normalized_evaluation_biases_best_arms[index] = total_squared_normalized_evaluation_biases_best_arms[index] / eval_times_reached[index]
    
    # Processing special results if turned on
    if algo_special_logs:
        if special != {}:
            if "at_step" in special.keys():
                for index, _ in enumerate(seen_steps):
                    for label in special["at_step"].keys():
                        special["at_step"][label][index] = special["at_step"][label][index] / steps_num_reached[index]
            if "at_epoch" in special.keys():
                for index, _ in enumerate(seen_epoch_nums):
                    for label in special["at_epoch"].keys():
                        special["at_epoch"][label][index] = special["at_epoch"][label][index] / epoch_num_reached[index]
            if "at_eval" in special.keys():
                for index, _ in enumerate(seen_eval_times):
                    for label in special["at_eval"].keys():
                        special["at_eval"][label][index] = special["at_eval"][label][index] / eval_times_reached[index]

    # Construct the at steps dictionary
    at_step = {}
    at_step["Timesteps"] = seen_steps
    at_step["Number of times the timesteps were reached"] = steps_num_reached
    at_step["Mean reward of steps"] = mean_steps_reward

    # Construct the at epoch dictionary
    at_epoch = {}
    at_epoch["Epoch numbers"] = seen_epoch_nums
    at_epoch["Number of times the epoch numbers were reached"] = epoch_num_reached
    at_epoch["Mean durations of epochs"] = mean_epoch_durations
    if max_steps_per_epoch != -1:
        at_epoch[f"Percent of epochs capped at {max_steps_per_epoch}"] = percent_of_capped_epochs
    at_epoch["Mean results at epochs"] = mean_epoch_results
    if correct_action_log:
        at_epoch["Mean correct action rates at epochs"] = mean_epoch_correct_action_rates
    
    # Construct the at eval dictionary
    at_eval = {}
    at_eval["Evaluation times"] = seen_eval_times
    at_eval["Number of times the evaluation times were reached"] = eval_times_reached
    at_eval["Mean scores at evaluations"] = mean_evaluation_scores
    at_eval["Mean number of terminal states reached during evaluations"] = mean_number_reached_terminal_states_during_eval
    at_eval["Mean time of reaching terminal states during evaluations"] = mean_time_reached_terminal_states_during_eval
    if correct_action_log:
        at_eval["Mean correct action rates at evaluations"] = mean_eval_correct_action_rates
        at_eval["Mean correct action rates at chosen states"] = mean_eval_correct_action_rates_at_chosen
    if focus_state_actions:
        for state_index, state in enumerate(which_state_actions_focus[0]):
            for act_index, act in enumerate(which_state_actions_focus[1][state_index]):
                if state == "start":
                    if act == "best":
                        at_eval[f"Mean biases at start state playing the best action at evaluations"] = [eval_biases_at_chosen[index][state_index][act_index] for index in range(len(seen_eval_times))]
                    else:
                        at_eval[f"Mean biases at start state playing action {act} at evaluations"] = [eval_biases_at_chosen[index][state_index][act_index] for index in range(len(seen_eval_times))]
                else:
                    if act == "best":
                        at_eval[f"Mean biases at state {state} playing the best action at evaluations"] = [eval_biases_at_chosen[index][state_index][act_index] for index in range(len(seen_eval_times))]
                    else:
                        at_eval[f"Mean biases at state {state} playing action {act} at evaluations"] = [eval_biases_at_chosen[index][state_index][act_index] for index in range(len(seen_eval_times))]
        for state_index, state in enumerate(which_state_actions_focus[0]):
            for act_index, act in enumerate(which_state_actions_focus[1][state_index]):
                if state == "start":
                    if act == "best":
                        at_eval[f"Mean Q function values at start state playing the best action at evaluations"] = [eval_q_values_at_chosen[index][state_index][act_index] for index in range(len(seen_eval_times))]
                    else:
                        at_eval[f"Mean Q function values at start state playing action {act} at evaluations"] = [eval_q_values_at_chosen[index][state_index][act_index] for index in range(len(seen_eval_times))]
                else:
                    if act == "best":
                        at_eval[f"Mean Q function values at state {state} playing the best action at evaluations"] = [eval_q_values_at_chosen[index][state_index][act_index] for index in range(len(seen_eval_times))]
                    else:
                        at_eval[f"Mean Q function values at state {state} playing action {act} at evaluations"] = [eval_q_values_at_chosen[index][state_index][act_index] for index in range(len(seen_eval_times))]
    if bias_estimation:
        at_eval["Mean total biases at evaluations"] = total_evaluation_biases
        at_eval["Mean total squared biases at evaluations"] = total_squared_evaluation_biases
        at_eval["Mean total biases at best arms at evaluations"] = total_evaluation_biases_best_arms
        at_eval["Mean total squared biases at best arms at evaluations"] = total_squared_evaluation_biases_best_arms
        at_eval["Mean total normalized biases at evaluations"] = total_normalized_evaluation_biases
        at_eval["Mean total squared normalized biases at evaluations"] = total_squared_normalized_evaluation_biases
        at_eval["Mean total normalized biases at best arms at evaluations"] = total_normalized_evaluation_biases_best_arms
        at_eval["Mean total squared normalized biases at best arms at evaluations"] = total_squared_normalized_evaluation_biases_best_arms
    
    experiment_results = {}
    experiment_results["Runtime in seconds"] = experiment_runtime
    experiment_results["Mean Q function values after training"] = estimated_correct_q_fct
    experiment_results["Start State number"] = env(**env_specific_params).start_state_num
    experiment_results["Mean Policy after training"] = estimated_correct_policy
    experiment_results["Data at steps"] = at_step
    experiment_results["Data at epochs"] = at_epoch
    experiment_results["Data at evaluations"] = at_eval
    if algo_special_logs:
        if special != {}:
            experiment_results["Special Data"] = special

    if verbose:

        # Processing results done
        time.sleep(0.5)
        iterator_postprocessing.update(1)

        # Saving results starts 
        iterator_postprocessing.write("Saving results ...")

    # Save the results
    with open(os.path.join(save_path,"results.pkl"), "wb") as file:
        pickle.dump(experiment_results,file)

    # If verbose, announce post-processing is finished and give out the path on which the results can be found.
    if verbose:

        # Postprocessing completed
        time.sleep(0.5)
        iterator_postprocessing.update(1)

        # Experiment finished
        print("Postprocessing is complete. The execution of the experiment was successful.")
        
        print("="*50)

        iterator_preprocessing.close()

        print(f"The results are located here: {save_path}")

        print("="*50+"\n")
    
    return save_path

###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################

# Configuration 1 for comparing CategoricalQ, CategoricalDouble, and ADDQ:
 
# Highlevel organizational parameters
project_name = "grid_world"; base_folder = "results"; output_folder = "plots"; subproject_labels = ["GridWorld"]
 
# Most important parameters
environments = [envs.GridWorld]
environments_specific_parameters = [{"grid_size": (4,4),"state_type_loc": {"goal":([(3,1)],True),"start":([(0,3)],False),"fake goal":([(0,0)],True),"stoch region":([(2,3),(3,3),(2,2),(3,2)],False)},"rewards":{"default": ["choice",{"a":[-0.05,0.05],"p":[0.5,0.5]}], "stoch region": ["choice",{"a":[-2.1,2],"p":[0.5,0.5]}], "goal":1, "fake goal":0.65}}]
algorithms = [[algos.CategoricalQ,algos.CategoricalDouble,algos.ADDQ]]; algo_labels = [["CategoricalQ","CategoricalDouble","ADDQ (us)"]]
algorithms_specific_parameters = [[{"range_atoms":[-3,3]},{"range_atoms":[-3,3]},{"range_atoms":[-3,3],"interpolation_mode":"adaptive","interpolation_mode_kwargs":{"center":"variance", "left_truncated":False, "which":"average", "current_state":True, "bounds":[(0.75,"strict"),(1.25,"not_strict")], "betas":[0.75,0.5,0.25]}}]]
algo_special_logs = True; algo_special_logs_kwargs = [[{"which_sample_variances":([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[[0],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0],[0,1,2,3],[0,1,2,3]])},{"which_sample_variances":([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[[0],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0],[0,1,2,3],[0,1,2,3]])},{"which_sample_variances":([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[[0],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0],[0,1,2,3],[0,1,2,3]])}]]
num_runs = [300]
num_steps = [30000]
 
# Secondary parameters. If moving up, be sure to modify the below section accordingly!
gamma = 0.9; learning_rate_kwargs = {"initial_rate":1,"mode":"rate","mode_kwargs":{"rate_fct":lambda n: 1/n,"iteration_num":1,"final_rate":0}}; learning_rate_state_action_wise = True
policy = algos.BasePolicy; policy_specific_params = {"policy_mode": "epsilon_greedy_statewise","policy_mode_kwargs": {"initial_rate":1,"mode":"linear","mode_kwargs":{"final_rate":0.1, "num_steps":10000,"slope":-1}}}
eval_steps = 6; eval_freq = 500; max_steps_per_epoch = -1
correct_act_q_fct_mode = "value_iteration"; correct_act_q_fct_mode_kwargs = {"n_max": 1000, "tol": 0.0001, "env_mean_rewards": {"default": 0, "stoch region": -0.05, "goal":1, "fake goal":0.65},"env_mean_rewards_mc_runs": 1}
bias_estimation = True
focus_state_actions = True; which_state_actions_focus = ([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[[0],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0],[0,1,2,3],[0,1,2,3]])
correct_action_log = True; correct_action_log_which = [1,4,6,7]
eval_reseeding = False; eval_seed_schedule = [-1]; training_mode = "steps"; training_reseeding = False; training_seed_schedule = [-1]
 
# Cosmetic parameters.
safe_mode = True; verbose = True
progress = True; progress_single_games = True; runtime_estimation = True

###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################

print("The experiment will be initialized ... ")

# Intialize dictionary for keeping the result paths
resultspath_dict = {subproject_label: {} for subproject_label in subproject_labels}
num_of_experiments = len(algorithms) * len(algorithms[0])

# For each experiment, get the data
print("Executing the experiments ... \n\n")
exp_num = 1
for label_index, label in enumerate(subproject_labels):
    for algo_index, algo in enumerate(algorithms[label_index]):
        # Let the function run and save the path where the results are
        print(f"Run {exp_num} of {num_of_experiments}:")
        saved_path = execute_experiment(
            base_folder= base_folder,
            num_runs = num_runs[label_index],
            progress = progress,
            project_name = project_name,
            runtime_estimation = runtime_estimation,
            safe_mode = safe_mode,
            verbose = verbose,
            algo = algo,
            algo_special_logs=algo_special_logs,
            algo_special_logs_kwargs=algo_special_logs_kwargs[label_index][algo_index],
            bias_estimation = bias_estimation,
            which_state_actions_focus= which_state_actions_focus,
            focus_state_actions = focus_state_actions,
            correct_act_q_fct_mode = correct_act_q_fct_mode,
            correct_act_q_fct_mode_kwargs = correct_act_q_fct_mode_kwargs[label_index],
            correct_action_log = correct_action_log,
            correct_action_log_which = correct_action_log_which,
            env = environments[label_index],
            eval_reseeding = eval_reseeding,
            eval_seed_schedule = eval_seed_schedule,
            eval_steps = eval_steps,
            eval_freq = eval_freq,
            max_steps_per_epoch = max_steps_per_epoch,
            num_steps = num_steps[label_index],
            policy = policy,
            progress_single_games = progress_single_games,
            training_mode = training_mode,
            training_reseeding = training_reseeding,
            training_seed_schedule = training_seed_schedule,
            algo_specific_params = algorithms_specific_parameters[label_index][algo_index],
            gamma = gamma,
            learning_rate_kwargs = learning_rate_kwargs,
            learning_rate_state_action_wise = True,
            env_specific_params = environments_specific_parameters[label_index],
            policy_specific_params = policy_specific_params,
        )
        print("\n")
        exp_num += 1
        save_label = algo().__str__()
        save_label_temp = save_label
        save_label_index = 1
        while save_label_temp in resultspath_dict[label].keys():
            save_label_temp = save_label + f"_{save_label_index}"
            save_label_index += 1
        resultspath_dict[label][save_label_temp] = saved_path

print("All experiments have been executed.\n")

# Initialize dictionary for keeping the aggregated result paths
aggregated_resultspath_dict = {subproject_label: None for subproject_label in subproject_labels}

# Aggregate the data algorithm-wise into appropriately named files
print("Aggregating data ... ")
if bias_estimation or focus_state_actions or correct_action_log:
    conditional_plots = True
else:
    conditional_plots = False
if algo_special_logs:
    special_plots = True
else:
    special_plots = False
for label_index, label in enumerate(subproject_labels):
    if algo_labels is None:
        labels = []
    else:
        labels = algo_labels[label_index]
    aggregated_saved_path = plots.results_single_to_batch_for_plot(
        result_paths = list(resultspath_dict[label].values()),
        labels=labels,
        output_folder= output_folder + "/.results_to_plot",
        project_name=project_name + "_" + label,
        safe_mode=safe_mode,
        conditional_plots=conditional_plots,
        special_plots = special_plots
    )
    aggregated_resultspath_dict[label] = aggregated_saved_path

# Print the locations of interesting stuff
for subproj in aggregated_resultspath_dict.keys():
    print(f"The aggregated results for subproject '{subproj}' can be found here:\n{aggregated_resultspath_dict[subproj]}\n\nThe single results can be found here:")
    for alg_name in resultspath_dict[subproj].keys():
        print(f"{alg_name}: {resultspath_dict[subproj][alg_name]}")

###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################

# Configuration 2 for Ablation Study:
 
# Highlevel organizational parameters
project_name = "ablation_study"; base_folder = "results"; output_folder = "plots"; subproject_labels = ["GridWorld"]
 
# Most important parameters
environments = [envs.GridWorld]
environments_specific_parameters = [{"grid_size": (4,4),"state_type_loc": {"goal":([(3,1)],True),"start":([(0,3)],False),"fake goal":([(0,0)],True),"stoch region":([(2,3),(3,3),(2,2),(3,2)],False)},"rewards":{"default": ["choice",{"a":[-0.05,0.05],"p":[0.5,0.5]}], "stoch region": ["choice",{"a":[-2.1,2],"p":[0.5,0.5]}], "goal":1, "fake goal":0.65}}]
algorithms = [[algos.CategoricalQ,algos.CategoricalDouble,algos.WDQ,algos.ADDQ,algos.ADDQ,algos.ADDQ,algos.ADDQ,algos.ADDQ,algos.ADDQ]]; algo_labels = [["CategoricalQ","CategoricalDouble","WDQ","ADDQ (neutral)","ADDQ (aggressive)","ADDQ (conservative)","ADDQ (0.7)","ADDQ (0.5)","ADDQ (0.3)"]]
algorithms_specific_parameters = [[{"range_atoms":[-3,3]},
                                   {"range_atoms":[-3,3]},
                                   {"interpolation_mode":"adaptive","interpolation_mode_kwargs":{"c":10, "current_state":False, "which_copy":"other", "which_reference_arm":"lowest"}},
                                   {"range_atoms":[-3,3],"interpolation_mode":"adaptive","interpolation_mode_kwargs":{"center":"variance", "left_truncated":False, "which":"average", "current_state":True, "bounds":[(0.75,"strict"),(1.25,"not_strict")], "betas":[0.75,0.5,0.25]}},
                                   {"range_atoms":[-3,3],"interpolation_mode":"adaptive","interpolation_mode_kwargs":{"center":"variance", "left_truncated":False, "which":"average", "current_state":True, "bounds":[(0.99,"strict"),(1.01,"not_strict")], "betas":[1,0.5,0]}},
                                   {"range_atoms":[-3,3],"interpolation_mode":"adaptive","interpolation_mode_kwargs":{"center":"variance", "left_truncated":False, "which":"average", "current_state":True, "bounds":[(0.6,"strict"),(1.4,"not_strict")], "betas":[0.6,0.5,0.4]}},
                                   {"range_atoms":[-3,3],"interpolation_mode":"constant","interpolation_mode_kwargs":{"beta":0.7}},
                                   {"range_atoms":[-3,3],"interpolation_mode":"constant","interpolation_mode_kwargs":{"beta":0.5}},
                                   {"range_atoms":[-3,3],"interpolation_mode":"constant","interpolation_mode_kwargs":{"beta":0.3}}]]
algo_special_logs = False; algo_special_logs_kwargs = [[{},{},{},{},{},{},{},{},{}]]
num_runs = [50]
num_steps = [10000]
 
# Secondary parameters. If moving up, be sure to modify the below section accordingly!
gamma = 0.9; learning_rate_kwargs = {"initial_rate":1,"mode":"rate","mode_kwargs":{"rate_fct":lambda n: 1/n,"iteration_num":1,"final_rate":0}}; learning_rate_state_action_wise = True
policy = algos.BasePolicy; policy_specific_params = {"policy_mode": "epsilon_greedy_statewise","policy_mode_kwargs": {"initial_rate":1,"mode":"linear","mode_kwargs":{"final_rate":0.1, "num_steps":10000,"slope":-1}}}
eval_steps = 6; eval_freq = 100; max_steps_per_epoch = -1
correct_act_q_fct_mode = "value_iteration"; correct_act_q_fct_mode_kwargs = [{"n_max": 1000, "tol": 0.0001, "env_mean_rewards": {"default": 0, "stoch region": -0.05, "goal":1, "fake goal":0.65},"env_mean_rewards_mc_runs": 1}]
bias_estimation = True
focus_state_actions = True; which_state_actions_focus = ([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[[0],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0],[0,1,2,3],[0,1,2,3]])
correct_action_log = True; correct_action_log_which = [1,4,6,7]
eval_reseeding = False; eval_seed_schedule = [-1]; training_mode = "steps"; training_reseeding = False; training_seed_schedule = [-1]
 
# Cosmetic parameters.
safe_mode = True; verbose = True
progress = True; progress_single_games = True; runtime_estimation = True

###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################

print("The experiment will be initialized ... ")

# Intialize dictionary for keeping the result paths
resultspath_dict = {subproject_label: {} for subproject_label in subproject_labels}
num_of_experiments = len(algorithms) * len(algorithms[0])

# For each experiment, get the data
print("Executing the experiments ... \n\n")
exp_num = 1
for label_index, label in enumerate(subproject_labels):
    for algo_index, algo in enumerate(algorithms[label_index]):
        # Let the function run and save the path where the results are
        print(f"Run {exp_num} of {num_of_experiments}:")
        saved_path = execute_experiment(
            base_folder= base_folder,
            num_runs = num_runs[label_index],
            progress = progress,
            project_name = project_name,
            runtime_estimation = runtime_estimation,
            safe_mode = safe_mode,
            verbose = verbose,
            algo = algo,
            algo_special_logs=algo_special_logs,
            algo_special_logs_kwargs=algo_special_logs_kwargs[label_index][algo_index],
            bias_estimation = bias_estimation,
            which_state_actions_focus= which_state_actions_focus,
            focus_state_actions = focus_state_actions,
            correct_act_q_fct_mode = correct_act_q_fct_mode,
            correct_act_q_fct_mode_kwargs = correct_act_q_fct_mode_kwargs[label_index],
            correct_action_log = correct_action_log,
            correct_action_log_which = correct_action_log_which,
            env = environments[label_index],
            eval_reseeding = eval_reseeding,
            eval_seed_schedule = eval_seed_schedule,
            eval_steps = eval_steps,
            eval_freq = eval_freq,
            max_steps_per_epoch = max_steps_per_epoch,
            num_steps = num_steps[label_index],
            policy = policy,
            progress_single_games = progress_single_games,
            training_mode = training_mode,
            training_reseeding = training_reseeding,
            training_seed_schedule = training_seed_schedule,
            algo_specific_params = algorithms_specific_parameters[label_index][algo_index],
            gamma = gamma,
            learning_rate_kwargs = learning_rate_kwargs,
            learning_rate_state_action_wise = True,
            env_specific_params = environments_specific_parameters[label_index],
            policy_specific_params = policy_specific_params,
        )
        print("\n")
        exp_num += 1
        save_label = algo().__str__()
        save_label_temp = save_label
        save_label_index = 1
        while save_label_temp in resultspath_dict[label].keys():
            save_label_temp = save_label + f"_{save_label_index}"
            save_label_index += 1
        resultspath_dict[label][save_label_temp] = saved_path

print("All experiments have been executed.\n")

# Initialize dictionary for keeping the aggregated result paths
aggregated_resultspath_dict = {subproject_label: None for subproject_label in subproject_labels}

# Aggregate the data algorithm-wise into appropriately named files
print("Aggregating data ... ")
if bias_estimation or focus_state_actions or correct_action_log:
    conditional_plots = True
else:
    conditional_plots = False
if algo_special_logs:
    special_plots = True
else:
    special_plots = False
for label_index, label in enumerate(subproject_labels):
    if algo_labels is None:
        labels = []
    else:
        labels = algo_labels[label_index]
    aggregated_saved_path = plots.results_single_to_batch_for_plot(
        result_paths = list(resultspath_dict[label].values()),
        labels=labels,
        output_folder= output_folder + "/.results_to_plot",
        project_name=project_name + "_" + label,
        safe_mode=safe_mode,
        conditional_plots=conditional_plots,
        special_plots = special_plots
    )
    aggregated_resultspath_dict[label] = aggregated_saved_path

# Print the locations of interesting stuff
for subproj in aggregated_resultspath_dict.keys():
    print(f"The aggregated results for subproject '{subproj}' can be found here:\n{aggregated_resultspath_dict[subproj]}\n\nThe single results can be found here:")
    for alg_name in resultspath_dict[subproj].keys():
        print(f"{alg_name}: {resultspath_dict[subproj][alg_name]}")