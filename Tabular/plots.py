from typing import List, Tuple, Union, Any, Dict
import yaml
import os
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

from utils import check_input_for_results_single_to_batch_for_plot, check_input_for_single_plot_fct, parse_correct_policy_and_q_function

# Aggregate the results from different runs, and check if they can be reasonably compared (i.e. are the significant parameters the same)

def results_single_to_batch_for_plot(
        result_paths: List[str] = None,
        labels: List[str] = None,
        output_folder: str = "plots",
        project_name: str = "testproject",
        safe_mode: bool = True,
        conditional_plots: bool = True,
        special_plots: bool = False,
        ) -> str:
    
    """
    Takes a list of paths to the folders where results.pkl files from execute_experiment runs are stored and aggregates the results into
    a bigger dictionary to be saved in the outpul folder under the project name. The given labels will be applied. If none are applied,
    the algorithm names from the parameters file are taken. If safe mode is activated, the input parameters will be checked and additionally
    a check wheter the runs in results_path are comparable in terms of plotting and in terms of run parameters will be performed.

    Parameters:
    - result_paths (list): A list of paths to the folder in which the results.pkl files to be used can be found.
    - labels (str): A list of labels corresponding to the runs in the reslut_paths list. If there are not enough labels in the list to match
      all results in result path, the rest of the results will be assigned their respective algorithm names as label.
    - output_folder (str): The folder in which the aggregated results should be saved.
    - project_name (str): The project name under which the aggregated results should be saved.
    - safe_mode (bool): If True, a check will be performed on the inputs. Additionally, it will be checked if the runs that will be aggregated
      are comparable in terms of parameters given to the execute_experiment function.
    - conditional_plots (bool): If you plan on plotting plots, for which data is not collected by default. In this case, if safe_mode is 
      activated, check for if the data was collected or not will be done.
    - special_plots (bool): If True, keys for the special plots will be assigned to aid in printing the special logs later on.

    Returns:
    - save_path (str): The path to the saved file.

    Raises:
    - ValueError: If any of the input parameter values are invalid or at least one of the runs' data is not comparable to the other runs.
    - TypeError: If any of the input parameter types are invalid if safe_mode is True.
    """

    # Input check if safe_mode
    if safe_mode:
        check_input_for_results_single_to_batch_for_plot(result_paths,labels,output_folder,project_name,safe_mode,conditional_plots)

    # Initialize aggregated results dictionary, warning printed messages
    aggregated_results = {}
    aggregated_results["Data"] = {}
    warning_num_steps_printed = False
    warning_trainin_mode_printed = False
    warning_num_runs_printed = False
    warning_conflicting_correct_policies = False
    steps_min_index = 0
    epoch_min_index = 0
    eval_min_index = 0
    if special_plots:
        special_plot_keys = {}
        special_plot_at_step_index = 0
        special_plot_at_epoch_index = 0
        special_plot_at_eval_index = 0

    # Initialize dictionary with necessary shared parameters if safe_mode
    if safe_mode:
        shared_parameters = {}

    # Initialize aggregated results
    aggregated_results = {"Data":{}}

    for index, path in enumerate(result_paths):
        
        # If safe_mode, load the yaml file and compare it to other data
        if safe_mode:
            with open(os.path.join(os.path.join(path, "reproduce_run"),"parameters.yaml"),"r") as file:
                result_data_for_path = yaml.load(file, Loader=yaml.FullLoader)
            if index > 0:
                if result_data_for_path["args"].keys() == shared_parameters["args"].keys() and result_data_for_path["config"].keys() == shared_parameters["config"].keys():
                    # Parameters to always check:
                    if not result_data_for_path["args"]["env"] == shared_parameters["args"]["env"]:
                        raise ValueError("There are inconsistencies in the chosen environment among your data!")
                    if not result_data_for_path["args"]["env_kwargs"] == shared_parameters["args"]["env_kwargs"]:
                        raise ValueError("There are inconsistencies in the chosen environment keyword arguments among your data!")
                    if not result_data_for_path["args"]["eval_freq"] == shared_parameters["args"]["eval_freq"]:
                        raise ValueError("There are inconsistencies in the chosen evaluation frequency among your data!")
                    if not result_data_for_path["args"]["eval_steps"] == shared_parameters["args"]["eval_steps"]:
                        raise ValueError("There are inconsistencies in the chosen evaluation steps among your data!")
                    if not result_data_for_path["args"]["max_steps_per_epoch"] == shared_parameters["args"]["max_steps_per_epoch"]:
                        raise ValueError("There are inconsistencies in the chosen maximal amount of steps per epoch among your data!")
                    # Parameters to check if conditional plots will be done
                    if conditional_plots:
                        if not result_data_for_path["args"]["bias_estimation"] == shared_parameters["args"]["bias_estimation"]:
                            raise ValueError("There are inconsistencies in if the bias estimation should be logged or not among your data!")
                        if not result_data_for_path["args"]["which_state_actions_focus"] == shared_parameters["args"]["which_state_actions_focus"]:
                            raise ValueError("There are inconsistencies in which state and action's bias estimation should be logged among your data!")
                        if not result_data_for_path["args"]["focus_state_actions"] == shared_parameters["args"]["focus_state_actions"]:
                            raise ValueError("There are inconsistencies in if the Q fct values and biases of selected state action pairs should be logged!")
                        if not result_data_for_path["args"]["correct_action_log"] == shared_parameters["args"]["correct_action_log"]:
                            raise ValueError("There are inconsistencies in if the correct action rates should be logged or not among your data!")
                        if not result_data_for_path["args"]["correct_action_log_which"] == shared_parameters["args"]["correct_action_log_which"]:
                            raise ValueError("There are inconsistencies in which states should count towards the correct action rate for the policy at evaluation!")
                    # Parameters to check with a warning if not equal
                    if (not result_data_for_path["args"]["num_steps"] == shared_parameters["args"]["num_steps"]) and (not warning_num_steps_printed):
                        print("Warning: There are inconsistencies in the number of played steps per run among your data, your results will be cropped according to the smallest one!")
                        warning_num_steps_printed = True
                    if (not result_data_for_path["args"]["training_mode"] == shared_parameters["args"]["training_mode"]) and (not warning_trainin_mode_printed):
                        print("Warning: There are inconsistencies in the training mode among your data, your results will be cropped accordingly!")
                        warning_trainin_mode_printed = True
                    if (not result_data_for_path["config"]["num_runs"] == shared_parameters["config"]["num_runs"]) and (not warning_num_runs_printed):
                        print("Warning: There are inconsistencies in the number of runs among your data, which makes the data less comparable!")
                        warning_num_runs_printed = True
                else:
                    raise ValueError("This is weird. Most likely something is wrong with the execute_experiment function, as there are inconsistencies in the keys of the parameters dictionaries!")
            else:
                shared_parameters = result_data_for_path.copy()
        else:
            with open(os.path.join(os.path.join(path, "reproduce_run"),"parameters.yaml"),"r") as file:
                result_data_for_path = yaml.load(file, Loader=yaml.FullLoader)
        
        # Get the label
        if len(labels) > index:
            label = labels[index]
        else:
            label = result_data_for_path["args"]["algo"]
        # Exclude double labels
        label_additional_index = 1
        label_temp = label
        while label in aggregated_results["Data"].keys():
            label_temp = label + f"_{label_additional_index}"
            label_additional_index +=1
        label = label_temp
        
        # Get the result data
        with open(os.path.join(path,"results.pkl"),"rb") as file:
            results_dict = pickle.load(file)
        
        # Update the steps_min, epoch_min and eval_min
        if index > 0:
            if steps_min_index >= len(results_dict["Data at steps"]["Timesteps"]):
                steps_min_index = len(results_dict["Data at steps"]["Timesteps"])
            if eval_min_index >= len(results_dict["Data at evaluations"]["Evaluation times"]):
                eval_min_index = len(results_dict["Data at evaluations"]["Evaluation times"])
            if epoch_min_index >= len(results_dict["Data at epochs"]["Epoch numbers"]):
                epoch_min_index = len(results_dict["Data at epochs"]["Epoch numbers"])
        else:
            steps_min_index = len(results_dict["Data at steps"]["Timesteps"])
            eval_min_index = len(results_dict["Data at evaluations"]["Evaluation times"])
            epoch_min_index = len(results_dict["Data at epochs"]["Epoch numbers"])

        # Add the results to the aggregated results dictionary
        aggregated_results["Data"][label] = deepcopy(results_dict)

        # If conditional plots need to be done, save the estimated correct policy and check if it is the same across the data points. Additionally, compute the mean of the estimated correct Q functions to get an even better estimate
        if conditional_plots:
            estimated_correct_policy, estimated_correct_q_fct = parse_correct_policy_and_q_function(os.path.join(path,"correct_policy_and_q_function.txt"))
            if index > 0:
                if safe_mode:
                    if correct_policy != estimated_correct_policy and not warning_conflicting_correct_policies:
                        print("Warning: Some of the estimated correct policies are conflicting with one another. You might want to run your experiments again using a longer value iteration!")
                        warning_conflicting_correct_policies = True
                for key in correct_q_fct:
                    correct_q_fct[key] += estimated_correct_q_fct[key]
            else:
                correct_policy = deepcopy(estimated_correct_policy)
                correct_q_fct = deepcopy(estimated_correct_q_fct)
        
        # If special plots need to be done, assign indices to keys for each type
        if special_plots:
            if "Special Data" in results_dict.keys():
                if "at_step" in results_dict["Special Data"].keys():
                    if not ("at_step" in special_plot_keys.keys()):
                        special_plot_keys["at_step"] = {}
                    for special_log_metric_key in results_dict["Special Data"]["at_step"]:
                        if special_log_metric_key in [special_plot_keys["at_step"][index][1] for index in special_plot_keys["at_step"].keys()]:
                            i = [special_plot_keys["at_step"][index][1] for index in special_plot_keys["at_step"].keys()].index(special_log_metric_key)
                            special_plot_keys["at_step"][i][0].append(label)
                        else:
                            special_plot_keys["at_step"][special_plot_at_step_index] = ([label],special_log_metric_key)
                            special_plot_at_step_index += 1
                if "at_epoch" in results_dict["Special Data"].keys():
                    if not ("at_epoch" in special_plot_keys.keys()):
                        special_plot_keys["at_epoch"] = {}
                    for special_log_metric_key in results_dict["specSpecial Dataial"]["at_epoch"]:
                        if special_log_metric_key in [special_plot_keys["at_epoch"][index][1] for index in special_plot_keys["at_epoch"].keys()]:
                            i = [special_plot_keys["at_epoch"][index][1] for index in special_plot_keys["at_epoch"].keys()].index(special_log_metric_key)
                            special_plot_keys["at_epoch"][i][0].append(label)
                        else:
                            special_plot_keys["at_epoch"][special_plot_at_epoch_index] = ([label],special_log_metric_key)
                            special_plot_at_epoch_index += 1
                if "at_eval" in results_dict["Special Data"].keys():
                    if not ("at_eval" in special_plot_keys.keys()):
                        special_plot_keys["at_eval"] = {}
                    for special_log_metric_key in results_dict["Special Data"]["at_eval"]:
                        if special_log_metric_key in [special_plot_keys["at_eval"][index][1] for index in special_plot_keys["at_eval"].keys()]:
                            i = [special_plot_keys["at_eval"][index][1] for index in special_plot_keys["at_eval"].keys()].index(special_log_metric_key)
                            special_plot_keys["at_eval"][i][0].append(label)
                        else:
                            special_plot_keys["at_eval"][special_plot_at_eval_index] = ([label],special_log_metric_key)
                            special_plot_at_eval_index += 1
    
    # Average the correct Q function
    for key in correct_q_fct:
        correct_q_fct[key] = correct_q_fct[key] / len(result_paths)

    # Add the steps_min_index, eval_min_index, and epoch_min_index as well as the correct policy and Q function to the dictionary
    aggregated_results["Maximum shared index of timesteps"] = steps_min_index
    aggregated_results["Maximum shared index of evaluations length"] = eval_min_index
    aggregated_results["Maximum shared index of epochs length"] = epoch_min_index
    aggregated_results["Correct policy"] = correct_policy
    aggregated_results["Correct Q function"] = correct_q_fct
    aggregated_results["Starting state"] = results_dict["Start State number"]
    if special_plots:
        aggregated_results["Special plot keys"] = special_plot_keys

    # Get the save path and save the aggregated results
    save_path = os.path.join(output_folder,project_name + ".pkl")
    with open(save_path, "wb") as file:
        pickle.dump(aggregated_results,file)

    return save_path

###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################

# Functions for different singular plot types at steps

###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################

# Number of times the timesteps were reached
def plot_num_times_timesteps_reached(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the number of times the respective timesteps were reached stemming from a data set either given by input_path or if 
    input_path is None given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen 
    as well as if the figure should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' 
    can be chosen and if safe_mode is activated a check on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            plt.plot(data["Data"][label]["Data at steps"]["Timesteps"][:data["Maximum shared index of timesteps"]],data["Data"][label]["Data at steps"]["Number of times the timesteps were reached"][:data["Maximum shared index of timesteps"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title("Number of times the timesteps were reached", fontsize=12)
        plt.xlabel("Timesteps",fontsize=8)
        plt.xlim(0,data["Data"][label]["Data at steps"]["Timesteps"][data["Maximum shared index of timesteps"]-1])
        plt.ylabel("Amount of times reached",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"num_times_timesteps_reached.png"))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            mode[1].plot(data["Data"][label]["Data at steps"]["Timesteps"][:data["Maximum shared index of timesteps"]],data["Data"][label]["Data at steps"]["Number of times the timesteps were reached"][:data["Maximum shared index of timesteps"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title("Number of times the timesteps were reached", fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Timesteps",fontsize=8)
        mode[1].set_xlim(0,data["Data"][label]["Data at steps"]["Timesteps"][data["Maximum shared index of timesteps"]-1])
        mode[1].set_ylabel("Amount of times reached",fontsize=8)
        mode[1].grid(grid)

# Average scores at steps
def plot_avg_rewards_at_step(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the average scores at the timesteps stemming from a data set either given by input_path or if input_path is None given by 
    combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen as well as if the figure
    should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' can be chosen and
    if safe_mode is activated a check on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            plt.plot(data["Data"][label]["Data at steps"]["Timesteps"][:data["Maximum shared index of timesteps"]],data["Data"][label]["Data at steps"]["Mean reward of steps"][:data["Maximum shared index of timesteps"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title("Average reward at timesteps during training", fontsize=12)
        plt.xlabel("Timesteps",fontsize=8)
        plt.xlim(0,data["Data"][label]["Data at steps"]["Timesteps"][data["Maximum shared index of timesteps"]-1])
        plt.ylabel("Rewards",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_rewards_at_steps.png"))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            mode[1].plot(data["Data"][label]["Data at steps"]["Timesteps"][:data["Maximum shared index of timesteps"]],data["Data"][label]["Data at steps"]["Mean reward of steps"][:data["Maximum shared index of timesteps"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title("Average reward at timesteps during training", fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Timesteps",fontsize=8)
        mode[1].set_xlim(0,data["Data"][label]["Data at steps"]["Timesteps"][data["Maximum shared index of timesteps"]-1])
        mode[1].set_ylabel("Rewards",fontsize=8)
        mode[1].grid(grid)

# Functions for different singular plot types at epochs

# Number of times the epochs were reached
def plot_num_times_epochs_reached(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the number of times the respective epoch numbers were reached stemming from a data set either given by input_path or if 
    input_path is None given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen 
    as well as if the figure should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' 
    can be chosen and if safe_mode is activated a check on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            plt.plot(data["Data"][label]["Data at epochs"]["Epoch numbers"][:data["Maximum shared index of epochs length"]],data["Data"][label]["Data at epochs"]["Number of times the epoch numbers were reached"][:data["Maximum shared index of epochs length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title("Number of times the epochs were reached", fontsize=12)
        plt.xlabel("Epochs",fontsize=8)
        plt.xlim(0,data["Data"][label]["Data at epochs"]["Epoch numbers"][data["Maximum shared index of epochs length"]-1])
        plt.ylabel("Amount of times reached",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"num_times_epochs_reached.png"))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            mode[1].plot(data["Data"][label]["Data at epochs"]["Epoch numbers"][:data["Maximum shared index of epochs length"]],data["Data"][label]["Data at epochs"]["Number of times the epoch numbers were reached"][:data["Maximum shared index of epochs length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title("Number of times the epochs were reached", fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Epochs",fontsize=8)
        mode[1].set_xlim(0,data["Data"][label]["Data at epochs"]["Epoch numbers"][data["Maximum shared index of epochs length"]-1])
        mode[1].set_ylabel("Amount of times reached",fontsize=8)
        mode[1].grid(grid)

# Average scores at epochs
def plot_avg_scores_at_epoch(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the average scores of the epochs stemming from a data set either given by input_path or if input_path is None given by 
    combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen as well as if the figure
    should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' can be chosen and
    if safe_mode is activated a check on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            plt.plot(data["Data"][label]["Data at epochs"]["Epoch numbers"][:data["Maximum shared index of epochs length"]],data["Data"][label]["Data at epochs"]["Mean results at epochs"][:data["Maximum shared index of epochs length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title("Average score of epochs during training", fontsize=12)
        plt.xlabel("Epochs",fontsize=8)
        plt.xlim(0,data["Data"][label]["Data at epochs"]["Epoch numbers"][data["Maximum shared index of epochs length"]-1])
        plt.ylabel("Scores",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_scores_at_epochs.png"))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            mode[1].plot(data["Data"][label]["Data at epochs"]["Epoch numbers"][:int((data["Maximum shared index of epochs length"])*2/3)],data["Data"][label]["Data at epochs"]["Mean results at epochs"][:int((data["Maximum shared index of epochs length"])*2/3)], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title("Average score of epochs during training", fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Epochs",fontsize=8)
        mode[1].set_xlim(0,data["Data"][label]["Data at epochs"]["Epoch numbers"][int((data["Maximum shared index of epochs length"])*2/3)-1])
        mode[1].set_ylabel("Scores",fontsize=8)
        mode[1].grid(grid)

# Average correct action rates at epochs
def plot_avg_correct_act_rates_at_epoch(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the average correct action rates at the epochs stemming from a data set either given by input_path or if input_path is 
    None given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen as well as if 
    the figure should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' can be chosen 
    and if safe_mode is activated a check on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)
    
    # If the correct actions rates were logged, plot them, else raise an error
    if "Mean correct action rates at epochs" in data["Data"][list(data["Data"].keys())[0]]["Data at epochs"].keys():

        # Depending on the mode create a plot or a subplot
        if mode == "single plot":

            # Initialize the plot
            plt.figure(figsize=figsize)

            # Plot the gathered data depending on the max shared length with corresponding labels
            for label in data["Data"].keys():
                plt.plot(data["Data"][label]["Data at epochs"]["Epoch numbers"][:data["Maximum shared index of epochs length"]],data["Data"][label]["Data at epochs"]["Mean correct action rates at epochs"][:data["Maximum shared index of epochs length"]], label=label)

            # Add title, x/y-label with fontsize, legend with location and fontsize
            plt.title("Average correct action rates\nof epochs during training", fontsize=12)
            plt.xlabel("Epochs",fontsize=8)
            plt.xlim(0,data["Data"][label]["Data at epochs"]["Epoch numbers"][data["Maximum shared index of epochs length"]-1])
            plt.ylabel("Correct action rates",fontsize=8)
            plt.legend(fontsize=8)
            plt.grid(grid)

            # If save is activated, make the save folder and save the plot
            if save:
                os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
                plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_correct_action_rates_at_epochs.png"))

            # If show is activated, show the plot
            if show:
                plt.show()
        
        else:

            # Plot the gathered data depending on the max shared length with corresponding labels
            for label in data["Data"].keys():
                mode[1].plot(data["Data"][label]["Data at epochs"]["Epoch numbers"][:data["Maximum shared index of epochs length"]],data["Data"][label]["Data at epochs"]["Mean correct action rates at epochs"][:data["Maximum shared index of epochs length"]], label=label)

            # Add title, x/y-label with fontsize, legend with location and fontsize
            if mode[2] == None:
                mode[1].set_title("Average correct action rates\n of epochs during training", fontsize=12)
            elif mode[2] != "no title":
                mode[1].set_title(mode[2], fontsize=12)
            mode[1].set_xlabel("Epochs",fontsize=8)
            mode[1].set_xlim(0,data["Data"][label]["Data at epochs"]["Epoch numbers"][data["Maximum shared index of epochs length"]-1])
            mode[1].set_ylabel("Correct action rates",fontsize=8)
            mode[1].grid(grid)
    
    else:
        raise ValueError("The correct action rates are not contained in the data you provided!")

# Mean durations of epoch
def plot_avg_durations_of_epochs(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the mean duration of the epochs stemming from a data set either given by input_path or if input_path is None given by combining the 
    path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen as well as if the figure should be shown and/or 
    saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' can be chosen and if safe_mode is activated a check 
    on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            plt.plot(data["Data"][label]["Data at epochs"]["Epoch numbers"][:data["Maximum shared index of epochs length"]],data["Data"][label]["Data at epochs"]["Mean durations of epochs"][:data["Maximum shared index of epochs length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title("Average durations of epochs during training", fontsize=12)
        plt.xlabel("Epochs",fontsize=8)
        plt.xlim(0,data["Data"][label]["Data at epochs"]["Epoch numbers"][data["Maximum shared index of epochs length"]-1])
        plt.ylabel("Durations",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_durations_of_epochs.png"))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            mode[1].plot(data["Data"][label]["Data at epochs"]["Epoch numbers"][:int((data["Maximum shared index of epochs length"])*2/3)],data["Data"][label]["Data at epochs"]["Mean durations of epochs"][:int((data["Maximum shared index of epochs length"])*2/3)], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title("Average durations of epochs during training", fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Epochs",fontsize=8)
        mode[1].set_xlim(0,data["Data"][label]["Data at epochs"]["Epoch numbers"][int((data["Maximum shared index of epochs length"])*2/3)-1])
        mode[1].set_ylabel("Durations",fontsize=8)
        mode[1].grid(grid)

# Percent of capped epochs
def plot_percent_of_capped_epochs(
        max_steps_per_epoch: int = -1,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the mean duration of the epochs stemming from a data set either given by input_path or if input_path is None given by combining the 
    path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen as well as if the figure should be shown and/or 
    saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' can be chosen and if safe_mode is activated a check 
    on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
        if isinstance(max_steps_per_epoch,int):
            if max_steps_per_epoch <= 0 and max_steps_per_epoch != -1:
                raise ValueError("max_steps_per_epoch needs to be either a positive integer or -1!")
        else:
            raise ValueError("max_steps_per_epoch needs to be an integer!")
    
    # If max_steps_per_epoch is default, there is nothing to plot.
    if max_steps_per_epoch == -1:
        return None
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            plt.plot(data["Data"][label]["Data at epochs"]["Epoch numbers"][:data["Maximum shared index of epochs length"]],data["Data"][label]["Data at epochs"][f"Percent of epochs capped at {max_steps_per_epoch}"][:data["Maximum shared index of epochs length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title("Percent of capped epochs during training", fontsize=12)
        plt.xlabel("Epochs",fontsize=8)
        plt.xlim(0,data["Data"][label]["Data at epochs"]["Epoch numbers"][data["Maximum shared index of epochs length"]-1])
        plt.ylabel("Percents",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"percent_of_capped_epochs.png"))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            mode[1].plot(data["Data"][label]["Data at epochs"]["Epoch numbers"][:data["Maximum shared index of epochs length"]],data["Data"][label]["Data at epochs"][f"Percent of epochs capped at {max_steps_per_epoch}"][:data["Maximum shared index of epochs length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title("Percent of capped epochs during training", fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Epochs",fontsize=8)
        mode[1].set_xlim(0,data["Data"][label]["Data at epochs"]["Epoch numbers"][data["Maximum shared index of epochs length"]-1])
        mode[1].set_ylabel("Percents",fontsize=8)
        mode[1].grid(grid)

# Functions for different singular plot types at evaluations

# Number of times the evaluation times were reached
def plot_num_times_evaluation_times_reached(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the number of times the respective evaluation times were reached stemming from a data set either given by input_path or if 
    input_path is None given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen 
    as well as if the figure should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' 
    can be chosen and if safe_mode is activated a check on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            plt.plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"]["Number of times the evaluation times were reached"][:data["Maximum shared index of evaluations length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title("Number of times the\nevaluation times were reached", fontsize=12)
        plt.xlabel("Timesteps",fontsize=8)
        plt.xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
        plt.ylabel("Amount of times reached",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"num_times_eval_times_reached.png"))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            mode[1].plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"]["Number of times the evaluation times were reached"][:data["Maximum shared index of evaluations length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title("Number of times the\nevaluation times were reached", fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Timesteps",fontsize=8)
        mode[1].set_xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
        mode[1].set_ylabel("Amount of times reached",fontsize=8)
        mode[1].grid(grid)

# Average scores at evaluation
def plot_avg_scores_at_eval(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the average scores at the evaluations stemming from a data set either given by input_path or if input_path is None given by 
    combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen as well as if the figure
    should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' can be chosen and
    if safe_mode is activated a check on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            plt.plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"]["Mean scores at evaluations"][:data["Maximum shared index of evaluations length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title("Average score of agent during evaluation", fontsize=12)
        plt.xlabel("Timesteps",fontsize=8)
        plt.xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
        plt.ylabel("Scores",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_scores_at_evaluations.png"))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            mode[1].plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"]["Mean scores at evaluations"][:data["Maximum shared index of evaluations length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title("Average score of agent during evaluation", fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Timesteps",fontsize=8)
        mode[1].set_xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
        mode[1].set_ylabel("Scores",fontsize=8)
        mode[1].grid(grid)

# Average correct action rates at evaluations
def plot_avg_correct_act_rates_at_eval(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the average correct action rates at the evaluations stemming from a data set either given by input_path or if input_path is 
    None given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen as well as if 
    the figure should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' can be chosen 
    and if safe_mode is activated a check on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)
    
    # If the correct actions rates were logged, plot them, else raise an error
    if "Mean correct action rates at evaluations" in data["Data"][list(data["Data"].keys())[0]]["Data at evaluations"].keys():

        # Depending on the mode create a plot or a subplot
        if mode == "single plot":

            # Initialize the plot
            plt.figure(figsize=figsize)

            # Plot the gathered data depending on the max shared length with corresponding labels
            for label in data["Data"].keys():
                plt.plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"]["Mean correct action rates at evaluations"][:data["Maximum shared index of evaluations length"]], label=label)

            # Add title, x/y-label with fontsize, legend with location and fontsize
            plt.title("Average correct action rates of\nagent during evaluations", fontsize=12)
            plt.xlabel("Timesteps",fontsize=8)
            plt.xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
            plt.ylabel("Correct action rates",fontsize=8)
            plt.legend(fontsize=8)
            plt.grid(grid)

            # If save is activated, make the save folder and save the plot
            if save:
                os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
                plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_correct_action_rates_at_evaluations.png"))

            # If show is activated, show the plot
            if show:
                plt.show()
        
        else:

            # Plot the gathered data depending on the max shared length with corresponding labels
            for label in data["Data"].keys():
                mode[1].plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"]["Mean correct action rates at evaluations"][:data["Maximum shared index of evaluations length"]], label=label)

            # Add title, x/y-label with fontsize, legend with location and fontsize
            if mode[2] == None:
                mode[1].set_title("Average correct action rates of\nagent during evaluations", fontsize=12)
            elif mode[2] != "no title":
                mode[1].set_title(mode[2], fontsize=12)
            mode[1].set_xlabel("Timesteps",fontsize=8)
            mode[1].set_xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
            mode[1].set_ylabel("Correct action rates",fontsize=8)
            mode[1].grid(grid)
    
    else:
        raise ValueError("The correct action rates are not contained in the data you provided!")

# Average correct action rates at chosen states of policy at evaluations
def plot_avg_correct_act_rates_at_chosen_at_eval(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the average correct action rates the policy of a trained agent exhibits for predetermined states at evaluation times stemming from 
    a data set either given by input_path or if input_path is None given by combining the path given by plot_folder and "project_name.pkl". 
    The figsize and the location can be chosen as well as if the figure should be shown and/or saved. If the plot should be a 'single plot' 
    or the function is part of 'multiple plots' can be chosen and if safe_mode is activated a check on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)
    
    # If the correct actions rates were logged, plot them, else raise an error
    if "Mean correct action rates at chosen states" in data["Data"][list(data["Data"].keys())[0]]["Data at evaluations"].keys():

        # Depending on the mode create a plot or a subplot
        if mode == "single plot":

            # Initialize the plot
            plt.figure(figsize=figsize)

            # Plot the gathered data depending on the max shared length with corresponding labels
            for label in data["Data"].keys():
                plt.plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"]["Mean correct action rates at chosen states"][:data["Maximum shared index of evaluations length"]], label=label)

            # Add title, x/y-label with fontsize, legend with location and fontsize
            plt.title("Average correct action rates of\nagent's policy at chosen states", fontsize=12)
            plt.xlabel("Timesteps",fontsize=8)
            plt.xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
            plt.ylabel("Correct action rates",fontsize=8)
            plt.legend(fontsize=8)
            plt.grid(grid)

            # If save is activated, make the save folder and save the plot
            if save:
                os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
                plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_correct_action_rates_at_chosen_at_evaluations.png"))

            # If show is activated, show the plot
            if show:
                plt.show()
        
        else:

            # Plot the gathered data depending on the max shared length with corresponding labels
            for label in data["Data"].keys():
                mode[1].plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"]["Mean correct action rates at chosen states"][:data["Maximum shared index of evaluations length"]], label=label)

            # Add title, x/y-label with fontsize, legend with location and fontsize
            if mode[2] == None:
                mode[1].set_title("Average correct action rates of\nagent's policy at chosen states", fontsize=12)
            elif mode[2] != "no title":
                mode[1].set_title(mode[2], fontsize=12)
            mode[1].set_xlabel("Timesteps",fontsize=8)
            mode[1].set_xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
            mode[1].set_ylabel("Correct action rates",fontsize=8)
            mode[1].grid(grid)
    
    else:
        raise ValueError("The correct action rates are not contained in the data you provided!")

# Plot the bias at one of the chosen state action pairs
def plot_avg_biases_one_chosen_state_action_at_eval(
        which: Tuple[int,Union[str,int]] = (0,"best"),
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the bias at one of the chosen state action pairs for the agent specified by which, stemming from a data set either given by 
    input_path or if input_path is None given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location 
    can be chosen as well as if the figure should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 
    'multiple plots' can be chosen and if safe_mode is activated a check on the parameters is performed.

    Parameters:
    - which (tuple): The state action pair whose bias should be plotted. It is given as a tuple where the first entry is the state given as
      an integer or the string "start" in case the start state is meant and the second is the action given as an integer or the string "best" 
      in case the best action for the state is meant.
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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
        if isinstance(which,tuple):
            if len(which) == 2:
                if not ((isinstance(which[0],int) or which[0]=="start") and (isinstance(which[1],int) or which[1] == "best")):
                    raise ValueError("The state action pair whose bias should be plotted needs to be given as a tuple of integer or 'start' and integer or 'best'!")
            else:
                raise ValueError("The state action pair whose bias should be plotted needs to be given as a tuple of integer or 'start' and integer or 'best'!")
        else:
            raise TypeError("The state action pair whose bias should be plotted needs to be given as a tuple of integer or 'start' and integer or 'best'!")
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Construct the keyword to look for, the plot title, and the file name
    if which[0] == "start":
        if which[1] == "best":
            key = f"Mean biases at start state playing the best action at evaluations"
            plt_title = f"Average biases for estimating the Q function\nat the start state playing a best action"
            file_name = f"average_biases_at_start_state_best_action_at_evaluations.png"
        else:
            key = f"Mean biases at start state playing action {which[1]} at evaluations"
            plt_title = f"Average biases for estimating the Q function\nat the start state playing action {which[1]}"
            file_name = f"average_biases_at_start_state_action_{which[1]}_at_evaluations.png"
    else:
        if which[1] == "best":
            key = f"Mean biases at state {which[0]} playing the best action at evaluations"
            plt_title = f"Average biases for estimating the Q function\nat state {which[0]} playing a best action"
            file_name = f"average_biases_at_state_{which[0]}_best_action_at_evaluations.png"
        else:
            key = f"Mean biases at state {which[0]} playing action {which[1]} at evaluations"
            plt_title = f"Average biases for estimating the Q function\n at state {which[0]} playing action {which[1]}"
            file_name = f"average_biases_at_state_{which[0]}_action_{which[1]}_at_evaluations.png"

    # If the biases were logged, plot them, else raise an error
    if key in data["Data"][list(data["Data"].keys())[0]]["Data at evaluations"].keys():

        # Depending on the mode create a plot or a subplot
        if mode == "single plot":

            # Initialize the plot
            plt.figure(figsize=figsize)

            # Plot the gathered data depending on the max shared length with corresponding labels
            for label in data["Data"].keys():
                plt.plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"][key][:data["Maximum shared index of evaluations length"]], label=label)

            # Add title, x/y-label with fontsize, legend with location and fontsize
            plt.title(plt_title, fontsize=12)
            plt.xlabel("Timesteps",fontsize=8)
            plt.xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
            plt.ylabel("Biases",fontsize=8)
            plt.legend(fontsize=8)
            plt.grid(grid)

            # If save is activated, make the save folder and save the plot
            if save:
                os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
                plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),file_name))

            # If show is activated, show the plot
            if show:
                plt.show()
        
        else:

            # Plot the gathered data depending on the max shared length with corresponding labels
            for label in data["Data"].keys():
                mode[1].plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"][key][:data["Maximum shared index of evaluations length"]], label=label)

            # Add title, x/y-label with fontsize, legend with location and fontsize
            if mode[2] == None:
                mode[1].set_title(plt_title, fontsize=12)
            elif mode[2] != "no title":
                mode[1].set_title(mode[2], fontsize=12)
            mode[1].set_xlabel("Timesteps",fontsize=8)
            mode[1].set_xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
            mode[1].set_ylabel("Biases",fontsize=8)
            mode[1].grid(grid)
    
    else:
        raise ValueError(f"The biases of the state action pair {which} are not contained in the data you provided!")

# Plot the Q value at one of the chosen state action pairs
def plot_avg_q_fct_values_one_chosen_state_action_at_eval(
        which: Tuple[int,Union[str,int]] = (0,"best"),
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the q function values at one of the chosen state action pairs for the agent specified by which, stemming from a data set either given by 
    input_path or if input_path is None given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location 
    can be chosen as well as if the figure should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 
    'multiple plots' can be chosen and if safe_mode is activated a check on the parameters is performed.

    Parameters:
    - which (tuple): The state action pair whose bias should be plotted. It is given as a tuple where the first entry is the state given as
      an integer or the string "start" in case the start state is meant and the second is the action given as an integer or the string "best" 
      in case the best action for the state is meant.
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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
        if isinstance(which,tuple):
            if len(which) == 2:
                if not ((isinstance(which[0],int) or which[0]=="start") and (isinstance(which[1],int) or which[1] == "best")):
                    raise ValueError("The state action pair whose bias should be plotted needs to be given as a tuple of integer or 'start' and integer or 'best'!")
            else:
                raise ValueError("The state action pair whose bias should be plotted needs to be given as a tuple of integer or 'start' and integer or 'best'!")
        else:
            raise TypeError("The state action pair whose bias should be plotted needs to be given as a tuple of integer or 'start' and integer or 'best'!")
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Get the correct Q function value and its label at the appropriate state action pair
    if which[0] == "start":
        rel_state = data["Starting state"]
    else:
        rel_state = which[0]
    if which[1] == "best":
        best_rel_actions = data["Correct policy"][rel_state]
        mean_rel_q_fct_values = 0
        for act in best_rel_actions:
            mean_rel_q_fct_values += data["Correct Q function"][(rel_state,act)]
        mean_rel_q_fct_values = mean_rel_q_fct_values / len(best_rel_actions)
        linelabel = f"$Q^*$"
    else:
        mean_rel_q_fct_values = data["Correct Q function"][(rel_state,which[1])]
        linelabel = f"$Q^*$"

    # Construct the keyword to look for, the plot title, and the file name
    if which[0] == "start":
        if which[1] == "best":
            key = f"Mean Q function values at start state playing the best action at evaluations"
            plt_title = f"Average Q function values\nat the start state playing a best action"
            file_name = f"average_q_fct_at_start_state_best_action_at_evaluations.png"
        else:
            key = f"Mean Q function values at start state playing action {which[1]} at evaluations"
            plt_title = f"Average Q function values\nat the start state playing action {which[1]}"
            file_name = f"average_q_fct_at_start_state_action_{which[1]}_at_evaluations.png"
    else:
        if which[1] == "best":
            key = f"Mean Q function values at state {which[0]} playing the best action at evaluations"
            plt_title = f"Average Q function values\nat state {which[0]} playing a best action"
            file_name = f"average_q_fct_at_state_{which[0]}_best_action_at_evaluations.png"
        else:
            key = f"Mean Q function values at state {which[0]} playing action {which[1]} at evaluations"
            plt_title = f"Average Q function values\nat state {which[0]} playing action {which[1]}"
            file_name = f"average_q_fct_at_state_{which[0]}_action_{which[1]}_at_evaluations.png"

    # If the biases were logged, plot them, else raise an error
    if key in data["Data"][list(data["Data"].keys())[0]]["Data at evaluations"].keys():

        # Depending on the mode create a plot or a subplot
        if mode == "single plot":

            # Initialize the plot
            plt.figure(figsize=figsize)

            # Plot the gathered data depending on the max shared length with corresponding labels
            for label in data["Data"].keys():
                plt.plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"][key][:data["Maximum shared index of evaluations length"]], label=label)

            # Plot the correct Q function
            plt.axhline(y=mean_rel_q_fct_values,color="black",linestyle="--",label=linelabel)

            # Add title, x/y-label with fontsize, legend with location and fontsize
            plt.title(plt_title, fontsize=12)
            plt.xlabel("Timesteps",fontsize=8)
            plt.xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
            plt.ylabel("Q function values",fontsize=8)
            plt.legend(fontsize=8)
            plt.grid(grid)

            # If save is activated, make the save folder and save the plot
            if save:
                os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
                plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),file_name))

            # If show is activated, show the plot
            if show:
                plt.show()
        
        else:

            # Plot the gathered data depending on the max shared length with corresponding labels
            for label in data["Data"].keys():
                mode[1].plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"][key][:data["Maximum shared index of evaluations length"]], label=label)

            # Plot the correct Q function
            mode[1].axhline(y=mean_rel_q_fct_values,color="black",linestyle="--")

            # Add title, x/y-label with fontsize, legend with location and fontsize
            if mode[2] == None:
                mode[1].set_title(plt_title, fontsize=12)
            elif mode[2] != "no title":
                mode[1].set_title(mode[2], fontsize=12)
            mode[1].set_xlabel("Timesteps",fontsize=8)
            mode[1].set_xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
            mode[1].text(x=mode[1].get_xlim()[1]+0.01*(mode[1].get_xlim()[1]-mode[1].get_xlim()[0]),y=mean_rel_q_fct_values,s=linelabel,color="k",va="center",ha="left")
            mode[1].set_ylabel("Q function values",fontsize=8)
            mode[1].grid(grid)
    
    else:
        raise ValueError(f"The Q function values of the state action pair {which} are not contained in the data you provided!")

# Plot one of the bias metrics
def plot_avg_total_biases_one_metric_at_eval(
        squared: bool = True,
        normalized: bool = True,
        best_arms: bool = True,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the chosen bias metric for the agent stemming from a data set either given by input_path or if input_path is None given by 
    combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen as well as if the figure 
    should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' can be chosen and if 
    safe_mode is activated a check on the parameters is performed. The chosen metric is specified by if the biases should be normalized
    and/or only the biases at the best arms for each state should be considered.

    Parameters:
    - squared (bool): If True, the summed up squared version of the biases will be plotted.
    - normalized (bool): If True, the normalized version of the total squared bias will be plotted.
    - best_arms (bool): If True, only the bias at the best arms will be taken into account.
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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
        if not isinstance(squared,bool):
            raise TypeError("Parameter squared needs to be boolean!")
        if not isinstance(normalized,bool):
            raise TypeError("Parameter normalized needs to be boolean!")
        if not isinstance(best_arms,bool):
            raise TypeError("Parameter best_arms needs to be boolean!")
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Construct the keyword to look for, the plot title, and the file name
    key = "Mean total "
    plt_title = "Average total "
    file_name = "average_total_"
    if squared:
        key = key + "squared "
        plt_title = plt_title + "squared "
        file_name = file_name + "squared_"
    if normalized:
        key = key + "normalized "
        plt_title = plt_title + "normalized "
        file_name = file_name + "normalized_"
    key = key + "biases "
    plt_title = plt_title + "biases\n"
    file_name = file_name + "biases_"
    if best_arms:
        key = key + "at best arms "
        plt_title = plt_title + "at best arms "
        file_name = file_name + "at_best_arms_"
    key = key + "at evaluations"
    plt_title = plt_title + "of agent"
    file_name = file_name + "at_evaluations.png"

    # If the biases were logged, plot them, else raise an error
    if key in data["Data"][list(data["Data"].keys())[0]]["Data at evaluations"].keys():

        # Depending on the mode create a plot or a subplot
        if mode == "single plot":

            # Initialize the plot
            plt.figure(figsize=figsize)

            # Plot the gathered data depending on the max shared length with corresponding labels
            for label in data["Data"].keys():
                plt.plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"][key][:data["Maximum shared index of evaluations length"]], label=label)

            # Add title, x/y-label with fontsize, legend with location and fontsize
            plt.title(plt_title, fontsize=12)
            plt.xlabel("Timesteps",fontsize=8)
            plt.xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
            plt.ylabel("Biases",fontsize=8)
            plt.legend(fontsize=8)
            plt.grid(grid)

            # If save is activated, make the save folder and save the plot
            if save:
                os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
                plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),file_name))

            # If show is activated, show the plot
            if show:
                plt.show()
        
        else:

            # Plot the gathered data depending on the max shared length with corresponding labels
            for label in data["Data"].keys():
                mode[1].plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"][key][:data["Maximum shared index of evaluations length"]], label=label)

            # Add title, x/y-label with fontsize, legend with location and fontsize
            if mode[2] == None:
                mode[1].set_title(plt_title,fontsize=12)
            elif mode[2] != "no title":
                mode[1].set_title(mode[2],fontsize=12)
            mode[1].set_xlabel("Timesteps",fontsize=8)
            mode[1].set_xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
            mode[1].set_ylabel("Biases",fontsize=8)
            mode[1].grid(grid)
    
    else:
        raise ValueError("The bias metrics are not contained in the data you provided!")

# Average number of terminal states reached at evaluation
def plot_avg_num_terminal_states_reached_at_eval(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the average number of times a terminal state was reached at the evaluations stemming from a data set either given by input_path 
    or if input_path is None given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be 
    chosen as well as if the figure should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple 
    plots' can be chosen and if safe_mode is activated a check on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            plt.plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"]["Mean number of terminal states reached during evaluations"][:data["Maximum shared index of evaluations length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title("Average number of times terminal states\nwere reached during evaluation", fontsize=12)
        plt.xlabel("Timesteps",fontsize=8)
        plt.xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
        plt.ylabel("Terminal states reached",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_num_terminal_states_at_evaluations.png"))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            mode[1].plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"]["Mean number of terminal states reached during evaluations"][:data["Maximum shared index of evaluations length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title("Average number of times terminal states were reached during evaluation", fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Timesteps",fontsize=8)
        mode[1].set_xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
        mode[1].set_ylabel("Terminal states reached",fontsize=8)
        mode[1].grid(grid)

# Average time of reaching terminal states at evaluation
def plot_avg_time_of_reaching_terminal_states_at_eval(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the average time of reaching a terminal state at the evaluations stemming from a data set either given by input_path 
    or if input_path is None given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be 
    chosen as well as if the figure should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple 
    plots' can be chosen and if safe_mode is activated a check on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            plt.plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"]["Mean time of reaching terminal states during evaluations"][:data["Maximum shared index of evaluations length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title("Average times of reaching terminal\nstates during evaluation", fontsize=12)
        plt.xlabel("Timesteps",fontsize=8)
        plt.xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
        plt.ylabel("Times",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_time_terminal_states_reached_at_evaluations.png"))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:

        # Plot the gathered data depending on the max shared length with corresponding labels
        for label in data["Data"].keys():
            mode[1].plot(data["Data"][label]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][label]["Data at evaluations"]["Mean time of reaching terminal states during evaluations"][:data["Maximum shared index of evaluations length"]], label=label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title("Average times of reaching terminal\nstates during evaluation", fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Timesteps",fontsize=8)
        mode[1].set_xlim(0,data["Data"][label]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
        mode[1].set_ylabel("Times",fontsize=8)
        mode[1].grid(grid)

# Functions for different special singular plot types

# Plot barplot of time utilized by algorithms
def plot_runtimes(
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the total runtimes in seconds for the different algorithms stemming from a data set either given by input_path or if input_path is None 
    given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen as well as if the figure 
    should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' can be chosen and if safe_mode 
    is activated a check on the parameters is performed.

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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
    
    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the gathered data depending on the max shared length with corresponding labels
        labels = list(data["Data"].keys())
        values = []
        for label in labels:
            values.append(data["Data"][label]["Runtime in seconds"])
        plt.bar(labels,values)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title("Total training and evaluation\nruntimes of algorithms", fontsize=12)
        plt.xlabel("Algorithms",fontsize=8)
        plt.ylabel("Runtimes in seconds",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"runtimes.png"))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:

        # Plot the gathered data depending on the max shared length with corresponding labels
        labels = list(data["Data"].keys())
        values = []
        for label in labels:
            values.append(data["Data"][label]["Runtime in seconds"])
        mode[1].bar(labels,values)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title("Total training and evaluation\nruntimes of algorithms", fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Algorithms",fontsize=8)
        mode[1].set_ylabel("Runtimes in seconds",fontsize=8)
        mode[1].grid(grid)

# Plot one of the special logs at timesteps
def plot_avg_special_logs_one_at_step(
        index: int = 0,
        real_value: Union[int,float] = None,
        real_value_label: str = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the special log metric at the timesteps associated with the passed index, stemming from a data set either given by input_path or if 
    input_path is None given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen as 
    well as if the figure should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' can 
    be chosen and if safe_mode is activated a check on the parameters is performed.

    Parameters:
    - index (int): The special plots index chosen for printing.
    - real_value (float): A real value to be plotted in case the special logs should be converging to some value. If None, no real_value will 
      be plotted.
    - real_value_label (str): The label to be given to the real_value passed.
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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
        if isinstance(index,int):
            if index < 0:
                raise ValueError("The index for printing a special plot needs to be a non-negative integer!")
        else:
            raise ValueError("The index for printing a special plot needs to be a non-negative integer!")
        if real_value is not None:
            if not isinstance(real_value,(int,float)):
                raise ValueError("The real value passed to be plotted with the special metrics needs to be a numerical value!")
            if not isinstance(real_value_label,str):
                raise ValueError("The label of the real value passed to be plotted with the special metrics needs to be a string!")

    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Construct the keyword to look for, the plot title, and the file name
    if "Special plot keys" in data.keys():
        if "at_step" in data["Special plot keys"].keys():
            if index in data["Special plot keys"]["at_step"]:
                algo_plotkey = data["Special plot keys"]["at_step"][index]
            else:
                ValueError("There are no special metrics at the timesteps to be plotted belonging to the chosen index contained in your data!")
        else:
            raise ValueError("There are no special metrics at the timesteps to be plotted belonging to the chosen index contained in your data!")
    else:
        raise ValueError("There are no special metrics at the timesteps to be plotted belonging to the chosen index contained in your data!")
    alg_keys = algo_plotkey[0]
    plot_label_key = algo_plotkey[1]
    plt_title = "Mean " + plot_label_key 
    file_name = plt_title.replace(" ", "_").lower() + f"_at timesteps"
    plt_title = plt_title + f"\nat timesteps"

    
    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the special metric
        for alg_key in alg_keys:
            plt.plot(data["Data"][alg_key]["Data at steps"]["Timesteps"][:data["Maximum shared index of timesteps"]],data["Data"][alg_key]["Special Data"]["at_step"][plot_label_key][:data["Maximum shared index of timesteps"]], label=alg_key)

        # Plot the correct value if necessary
        if real_value is not None:
            plt.axhline(y=real_value,color="black",linestyle="--",label=real_value_label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title(plt_title, fontsize=12)
        plt.xlabel("Timesteps",fontsize=8)
        plt.xlim(0,data["Data"][alg_key]["Data at steps"]["Timesteps"][data["Maximum shared index of timesteps"]-1])
        plt.ylabel("Values",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),file_name))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:
        # Plot the special metric
        for alg_key in alg_keys:
            mode[1].plot(data["Data"][alg_key]["Data at steps"]["Timesteps"][:data["Maximum shared index of timesteps"]],data["Data"][alg_key]["Special Data"]["at_step"][plot_label_key][:data["Maximum shared index of timesteps"]], label=alg_key)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title(plt_title,fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Timesteps",fontsize=8)
        mode[1].set_xlim(0,data["Data"][alg_key]["Data at steps"]["Timesteps"][data["Maximum shared index of timesteps"]-1])
        mode[1].set_ylabel("Values",fontsize=8)
        mode[1].grid(grid)

# Plot one of the special logs at epochs
def plot_avg_special_logs_one_at_epoch(
        index: int = 0,
        real_value: Union[int,float] = None,
        real_value_label: str = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the special log metric at the epochs associated with the passed index, stemming from a data set either given by input_path or if 
    input_path is None given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen as 
    well as if the figure should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' can 
    be chosen and if safe_mode is activated a check on the parameters is performed.

    Parameters:
    - index (int): The special plots index chosen for printing.
    - real_value (float): A real value to be plotted in case the special logs should be converging to some value. If None, no real_value will 
      be plotted.
    - real_value_label (str): The label to be given to the real_value passed.
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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
        if isinstance(index,int):
            if index < 0:
                raise ValueError("The index for printing a special plot needs to be a non-negative integer!")
        else:
            raise ValueError("The index for printing a special plot needs to be a non-negative integer!")
        if real_value is not None:
            if not isinstance(real_value,(int,float)):
                raise ValueError("The real value passed to be plotted with the special metrics needs to be a numerical value!")
            if not isinstance(real_value_label,str):
                raise ValueError("The label of the real value passed to be plotted with the special metrics needs to be a string!")

    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Construct the keyword to look for, the plot title, and the file name
    if "Special plot keys" in data.keys():
        if "at_epoch" in data["Special plot keys"].keys():
            if index in data["Special plot keys"]["at_epoch"]:
                algo_plotkey = data["Special plot keys"]["at_epoch"][index]
            else:
                ValueError("There are no special metrics at the epochs to be plotted belonging to the chosen index contained in your data!")
        else:
            raise ValueError("There are no special metrics at the epochs to be plotted belonging to the chosen index contained in your data!")
    else:
        raise ValueError("There are no special metrics at the epochs to be plotted belonging to the chosen index contained in your data!")
    alg_keys = algo_plotkey[0]
    plot_label_key = algo_plotkey[1]
    plt_title = "Mean " + plot_label_key 
    file_name = plt_title.replace(" ", "_").lower() + f"_at epochs"
    plt_title = plt_title + f"\nat epochs"

    
    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the special metric
        for alg_key in alg_keys:
            plt.plot(data["Data"][alg_key]["Data at epochs"]["Epoch numbers"][:data["Maximum shared index of epochs length"]],data["Data"][alg_key]["Special Data"]["at_epoch"][plot_label_key][:data["Maximum shared index of epochs length"]], label=alg_key)

        # Plot the correct value if necessary
        if real_value is not None:
            plt.axhline(y=real_value,color="black",linestyle="--",label=real_value_label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title(plt_title, fontsize=12)
        plt.xlabel("Epochs",fontsize=8)
        plt.xlim(0,data["Data"][alg_key]["Data at epochs"]["Epoch numbers"][data["Maximum shared index of epochs length"]-1])
        plt.ylabel("Values",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),file_name))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:
        # Plot the special metric
        for alg_key in alg_keys:
            mode[1].plot(data["Data"][alg_key]["Data at epochs"]["Epoch numbers"][:data["Maximum shared index of epochs length"]],data["Data"][alg_key]["Special Data"]["at_epoch"][plot_label_key][:data["Maximum shared index of epochs length"]], label=alg_key)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title(plt_title,fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Epochs",fontsize=8)
        mode[1].set_xlim(0,data["Data"][alg_key]["Data at epochs"]["Epoch numbers"][data["Maximum shared index of epochs length"]-1])
        mode[1].set_ylabel("Values",fontsize=8)
        mode[1].grid(grid)

# Plot one of the special logs at evaluations
def plot_avg_special_logs_one_at_eval(
        index: int = 0,
        real_value: Union[int,float] = None,
        real_value_label: str = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        figsize: Tuple[Union[int,float]] = (4,4),
        loc: Union[str,int] = "best",
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        mode: Any = "single plot",
        safe_mode: bool = True
        ) -> None:
    """
    Plots the special log metric at the evaluations associated with the passed index, stemming from a data set either given by input_path or if 
    input_path is None given by combining the path given by plot_folder and "project_name.pkl". The figsize and the location can be chosen as 
    well as if the figure should be shown and/or saved. If the plot should be a 'single plot' or the function is part of 'multiple plots' can 
    be chosen and if safe_mode is activated a check on the parameters is performed.

    Parameters:
    - index (int): The special plots index chosen for printing.
    - real_value (float): A real value to be plotted in case the special logs should be converging to some value. If None, no real_value will 
      be plotted.
    - real_value_label (str): The label to be given to the real_value passed.
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
      consisting of 'multiple plots', an axis that is passed, meaning it should be a subplot on a specified axis ax (This is used in 
      different functions to unify plots with different data or plot all metrics in one figure), and a string indicating the title, which can be 
      passed as None in case the default title should be plotted or 'no title' in case no title should be assigned. If 'multiple plots' is chosen 
      save and show must be turned off.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode is on check the parameters
    if safe_mode:
        check_input_for_single_plot_fct(input_path,plot_folder,project_name,figsize,loc,grid,show,save,mode,safe_mode)
        if isinstance(index,int):
            if index < 0:
                raise ValueError("The index for printing a special plot needs to be a non-negative integer!")
        else:
            raise ValueError("The index for printing a special plot needs to be a non-negative integer!")
        if real_value is not None:
            if not isinstance(real_value,(int,float)):
                raise ValueError("The real value passed to be plotted with the special metrics needs to be a numerical value!")
            if not isinstance(real_value_label,str):
                raise ValueError("The label of the real value passed to be plotted with the special metrics needs to be a string!")

    # Load the results file
    if input_path == None:
        if os.path.isfile(os.path.join(plot_folder,project_name) + ".pkl"):
            results_path = os.path.join(plot_folder,project_name) + ".pkl"
        else:
            results_path = os.path.join(plot_folder,project_name) + ".pickle"
    else:
        results_path = input_path
    with open(results_path,"rb") as file:
        data = pickle.load(file)

    # Construct the keyword to look for, the plot title, and the file name
    if "Special plot keys" in data.keys():
        if "at_eval" in data["Special plot keys"].keys():
            if index in data["Special plot keys"]["at_eval"]:
                algo_plotkey = data["Special plot keys"]["at_eval"][index]
            else:
                ValueError("There are no special metrics at the evaluations to be plotted belonging to the chosen index contained in your data!")
        else:
            raise ValueError("There are no special metrics at the evaluations to be plotted belonging to the chosen index contained in your data!")
    else:
        raise ValueError("There are no special metrics at the evaluations to be plotted belonging to the chosen index contained in your data!")
    alg_keys = algo_plotkey[0]
    plot_label_key = algo_plotkey[1]
    plt_title = "Mean " + plot_label_key 
    file_name = plt_title.replace(" ", "_").lower() + f"_at evaluations"
    plt_title = plt_title + f"\nat evaluations"
    
    # Depending on the mode create a plot or a subplot
    if mode == "single plot":

        # Initialize the plot
        plt.figure(figsize=figsize)

        # Plot the special metric
        for alg_key in alg_keys:
            plt.plot(data["Data"][alg_key]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][alg_key]["Special Data"]["at_eval"][plot_label_key][:data["Maximum shared index of evaluations length"]], label=alg_key)

        # Plot the correct value if necessary
        if real_value is not None:
            plt.axhline(y=real_value,color="black",linestyle="--",label=real_value_label)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        plt.title(plt_title, fontsize=12)
        plt.xlabel("Timesteps",fontsize=8)
        plt.xlim(0,data["Data"][alg_key]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
        plt.ylabel("Values",fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(grid)

        # If save is activated, make the save folder and save the plot
        if save:
            os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
            plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),file_name))

        # If show is activated, show the plot
        if show:
            plt.show()
    
    else:
        # Plot the special metric
        for alg_key in alg_keys:
            mode[1].plot(data["Data"][alg_key]["Data at evaluations"]["Evaluation times"][:data["Maximum shared index of evaluations length"]],data["Data"][alg_key]["Special Data"]["at_eval"][plot_label_key][:data["Maximum shared index of evaluations length"]], label=alg_key)

        # Add title, x/y-label with fontsize, legend with location and fontsize
        if mode[2] == None:
            mode[1].set_title(plt_title,fontsize=12)
        elif mode[2] != "no title":
            mode[1].set_title(mode[2], fontsize=12)
        mode[1].set_xlabel("Timesteps",fontsize=8)
        mode[1].set_xlim(0,data["Data"][alg_key]["Data at evaluations"]["Evaluation times"][data["Maximum shared index of evaluations length"]-1])
        mode[1].set_ylabel("Sample variances",fontsize=8)
        mode[1].grid(grid)
    
###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################

# Function for different multiple plot types from the same family

###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################

# Plot the biases at multiple chosen state action pairs
def plot_avg_biases_multiple_chosen_state_action_at_eval(
        which: List[Tuple[int,Union[str,int]]] = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        safe_mode: bool = True) -> None:
    
    """
    Plots the bias at multiple chosen state action pairs. You can choose to plot only the squared metrics, only the regular metrics or both. 
    The data is given either by input_path or if input_path is None it is given by combining the path given by plot_folder and "project_name.pkl".
    The figsize of the individual plots can be chosen as well as if the figure should be shown and/or saved. If safe_mode is activated a check on 
    the parameters is performed.

    Parameters:
    - which (list): A list containing state action pairs whose bias should be plotted.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    # If safe_mode, perform a parameter check
    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if isinstance(which,list):
            for sa in which:
                if isinstance(sa,tuple):
                    if len(sa) == 2:
                        if not (isinstance(sa[0],int) and (isinstance(sa[1],int) or sa[1] == "best")):
                            raise ValueError(f"State action pair {sa} is invalid!")
                    else:
                        raise ValueError(f"State action pair {sa} is not a tuple of length 2!")
                else:
                    raise ValueError(f"State action pair {sa} is not a tuple of length 2!")
        else:
            raise ValueError("The state action pairs whose bias should be plotted need to be given as a list!")
        if isinstance(num_rows,int):
            if num_rows <= 0:
                raise ValueError("Number of rows needs to be positive!")
        else:
            raise TypeError("Number of rows needs to be an integer!")
    
    if which is None:
        raise ValueError("You need to specify which average bias plots you want to put in your plot!")
    
    # Initialize the unified plot
    len_plot = len(which)
    num_cols = int(np.ceil(len_plot / num_rows))
    remove_index = - (num_rows - (len_plot % num_rows))
    fig, axs = plt.subplots(num_cols,num_rows,figsize=(individual_figsize[0]*num_rows,individual_figsize[1]*num_cols))
    
    # Flatten the axis
    axs = axs.flatten()

    # Plot the subplots using the dedicated function
    for index, sa in enumerate(which):
        plot_avg_biases_one_chosen_state_action_at_eval(
            which = sa,
            input_path = input_path,
            plot_folder = plot_folder,
            project_name = project_name,
            figsize = individual_figsize,
            grid = grid,
            mode = ("multiple plots", axs[index],f"State: {sa[0]}, Action: {sa[1]}"),
            safe_mode = True
            )
    
    # Collect all handles and labels
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Remove duplicate labels, adjust layout
    unique = dict(zip(labels,handles))
    fig.legend(unique.values(), unique.keys(), loc = "upper right")
    fig.suptitle("Average biases at different state action pairs",fontsize=16)
    if remove_index != -num_rows:
        for j in range(int(num_cols * num_rows + remove_index),int(num_cols * num_rows)):
            fig.delaxes(axs[j])
    fig.tight_layout()

    # If save is activated, make the save folder and save the plot
    if save:
        os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
        plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_biases_at_multiple_chosen_state_action_at_eval.png"))

    # If show is activated, show the plot
    if show:
        plt.show()

# Plot the biases at all chosen state action pairs
def plot_avg_biases_all_chosen_state_action_at_eval(
        which: Tuple[List[int],List[List[int]]] = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        safe_mode: bool = True) -> None:
    
    """
    Plots the bias at all chosen state action pairs. You can choose to plot only the squared metrics, only the regular metrics or both. 
    The data is given either by input_path or if input_path is None it is given by combining the path given by plot_folder and "project_name.pkl".
    The figsize of the individual plots can be chosen as well as if the figure should be shown and/or saved. If safe_mode is activated a check on 
    the parameters is performed.

    Parameters:
    - which (tuple): The tuple containing a list of states for which the average biases were tracked and a list of lists of corresponding actions.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if isinstance(which,tuple):
            if isinstance(which[0],list) and isinstance(which[1],list):
                for state in which[0]:
                    if isinstance(state,int):
                        if state < 0:
                            raise ValueError("All states must be positive integers!")
                    elif state != "start":
                        raise TypeError("All states must be positive integers!")
                for list_list in which[1]:
                    if isinstance(list_list,list):
                        for action in list_list:
                            if isinstance(action,int):
                                if action < 0:
                                    raise ValueError("All actions must be positive integers!")
                            elif action != "best":
                                raise TypeError("All actions must be positive integers!")
                    else:
                        raise TypeError("The actions for the corresponding states must be passed in lists!")
            else:
                raise TypeError("The state and action pairs must be passed as a tuple of lists!")
        else:
            raise TypeError("The state and action pairs must be passed as a tuple of lists!")
    
    # Fill the list of configurations
    which_list = []
    for state_index,state in enumerate(which[0]):
        for action in which[1][state_index]:
            which_list.append((state,action))

    plot_avg_biases_multiple_chosen_state_action_at_eval(
        which=which_list,
        input_path= input_path,
        plot_folder= plot_folder,
        project_name= project_name,
        individual_figsize= individual_figsize,
        num_rows= num_rows,
        grid= grid,
        show= show,
        save= save,
        safe_mode= safe_mode)

# Plot the Q function values at multiple chosen state action pairs
def plot_avg_q_fct_values_multiple_chosen_state_action_at_eval(
        which: List[Tuple[int,Union[str,int]]] = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        safe_mode: bool = True) -> None:
    
    """
    Plots the Q values at multiple chosen state action pairs. You can choose to plot only the squared metrics, only the regular metrics or both. 
    The data is given either by input_path or if input_path is None it is given by combining the path given by plot_folder and "project_name.pkl".
    The figsize of the individual plots can be chosen as well as if the figure should be shown and/or saved. If safe_mode is activated a check on 
    the parameters is performed.

    Parameters:
    - which (list): A list containing state action pairs whose Q function values should be plotted.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    # If safe_mode, perform a parameter check
    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if which is not None:
            if isinstance(which,list):
                for sa in which:
                    if isinstance(sa,tuple):
                        if len(sa) == 2:
                            if not ((isinstance(sa[0],int) or sa[0] == "start") and (isinstance(sa[1],int) or sa[1] == "best")):
                                raise ValueError(f"State action pair {sa} is invalid!")
                        else:
                            raise ValueError(f"State action pair {sa} is not a tuple of length 2!")
                    else:
                        raise ValueError(f"State action pair {sa} is not a tuple of length 2!")
            else:
                raise ValueError("The state action pairs whose Q function values should be plotted need to be given as a list!")
            if isinstance(num_rows,int):
                if num_rows <= 0:
                    raise ValueError("Number of rows needs to be positive!")
            else:
                raise TypeError("Number of rows needs to be an integer!")
    
    if which is None:
        raise ValueError("You need to specify which average Q function value plots you want to put in your plot!")
    
    # Initialize the unified plot
    len_plot = len(which)
    num_cols = int(np.ceil(len_plot / num_rows))
    remove_index = - (num_rows - (len_plot % num_rows))
    fig, axs = plt.subplots(num_cols,num_rows,figsize=(individual_figsize[0]*num_rows,individual_figsize[1]*num_cols))
    
    # Flatten the axis
    axs = axs.flatten()

    # Plot the subplots using the dedicated function
    for index, sa in enumerate(which):
        plot_avg_q_fct_values_one_chosen_state_action_at_eval(
            which = sa,
            input_path = input_path,
            plot_folder = plot_folder,
            project_name = project_name,
            figsize = individual_figsize,
            grid = grid,
            mode = ("multiple plots", axs[index],f"State: {sa[0]}, Action: {sa[1]}"),
            safe_mode = safe_mode
            )
    
    # Collect all handles and labels
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Remove duplicate labels, adjust layout
    unique = dict(zip(labels,handles))
    fig.legend(unique.values(), unique.keys(), loc = "upper right")
    fig.suptitle("Average Q function values at different state action pairs",fontsize=16)
    if remove_index != -num_rows:
        for j in range(int(num_cols * num_rows + remove_index),int(num_cols * num_rows)):
            fig.delaxes(axs[j])
    fig.tight_layout()

    # If save is activated, make the save folder and save the plot
    if save:
        os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
        plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_q_fct_values_at_multiple_chosen_state_action_at_eval"))

    # If show is activated, show the plot
    if show:
        plt.show()

# Plot the Q function values at multiple chosen state action pairs
def plot_avg_q_fct_values_all_chosen_state_action_at_eval(
        which: Tuple[List[int],List[List[int]]] = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        safe_mode: bool = True) -> None:
    
    """
    Plots the Q function values at all chosen state action pairs. You can choose to plot only the squared metrics, only the regular metrics or both. 
    The data is given either by input_path or if input_path is None it is given by combining the path given by plot_folder and "project_name.pkl".
    The figsize of the individual plots can be chosen as well as if the figure should be shown and/or saved. If safe_mode is activated a check on 
    the parameters is performed.

    Parameters:
    - which (tuple): The tuple containing a list of states for which the average biases were tracked and a list of lists of corresponding actions.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if isinstance(which,tuple):
            if isinstance(which[0],list) and isinstance(which[1],list):
                for state in which[0]:
                    if isinstance(state,int):
                        if state < 0:
                            raise ValueError("All states must be positive integers!")
                    elif state != "start":
                        raise TypeError("All states must be positive integers!")
                for list_list in which[1]:
                    if isinstance(list_list,list):
                        for action in list_list:
                            if isinstance(action,int):
                                if action < 0:
                                    raise ValueError("All actions must be positive integers!")
                            elif action != "best":
                                raise TypeError("All actions must be positive integers!")
                    else:
                        raise TypeError("The actions for the corresponding states must be passed in lists!")
            else:
                raise TypeError("The state and action pairs must be passed as a tuple of lists!")
        else:
            raise TypeError("The state and action pairs must be passed as a tuple of lists!")
    
    # Fill the list of configurations
    which_list = []
    for state_index,state in enumerate(which[0]):
        for action in which[1][state_index]:
            which_list.append((state,action))

    plot_avg_q_fct_values_multiple_chosen_state_action_at_eval(
        which=which_list,
        input_path= input_path,
        plot_folder= plot_folder,
        project_name= project_name,
        individual_figsize= individual_figsize,
        num_rows= num_rows,
        grid= grid,
        show= show,
        save= save,
        safe_mode= safe_mode)

# Plot multiple bias metrics
def plot_avg_total_biases_multiple_metrics_at_eval(
        squared: List[bool] = None,
        normalized: List[bool] = None,
        best_arms: List[bool] = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        safe_mode: bool = True) -> None:
    
    """
    Plots chosen bias metrics, which stem from a list determining squaring, normalizing, and best_arms. The data is given either by input_path or 
    if input_path is None it is given by combining the path given by plot_folder and "project_name.pkl". The figsize of the individual plots can 
    be chosen as well as if the figure should be shown and/or saved. If safe_mode is activated a check on the parameters is performed.

    Parameters:
    - squared (List): A list corresponding to the bias metrics' values to be plotted
    - normalized (List): A list corresponding to the bias metrics' values to be plotted
    - best_arms (List): A list corresponding to the bias metrics' values to be plotted
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.    
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    # If safe_mode, perform a parameter check
    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if squared is not None:
            if isinstance(squared,list) and isinstance(normalized,list) and isinstance(best_arms,list):
                if len(squared) == len(normalized) and len(normalized) == len(best_arms):
                    for val in squared:
                        if not isinstance(val,bool):
                            raise TypeError("Some of the given values for 'squared' are not boolean!")
                    for val in normalized:
                        if not isinstance(val,bool):
                            raise TypeError("Some of the given values for 'normalized' are not boolean!")
                    for val in best_arms:
                        if not isinstance(val,bool):
                            raise TypeError("Some of the given values for 'best_arms' are not boolean!")
                else:
                    raise ValueError("The length of the lists for 'squared', 'normalized', and 'best_arms' need to be the same!")
            else:
                raise TypeError("The parameters for 'squared', 'normalized', and 'best_arms' need to be passed in a list!")
        if isinstance(num_rows,int):
            if num_rows <= 0:
                raise ValueError("Number of rows needs to be positive!")
        else:
            raise TypeError("Number of rows needs to be an integer!")
    
    if squared is None:
        raise ValueError("You need to specify which bias metric plots you want to put in your plot!")

    # Initialize the unified plot
    len_plot = len(squared)
    num_cols = int(np.ceil(len_plot / num_rows))
    remove_index = - (num_rows - (len_plot % num_rows))
    fig, axs = plt.subplots(num_cols,num_rows,figsize=(individual_figsize[0]*num_rows,individual_figsize[1]*num_cols))

    # Flatten the axis
    axs = axs.flatten()

    # Plot the subplots using the dedicated function
    for i,_ in enumerate(squared):
        label = ""
        if squared[i]:
            label = label + "Squared "
            if normalized[i]:
                label = label + "normalized "
            label = label + "biases"
        elif normalized[i]:
            label = label + "Normalized biases"
        else:
            label = label + "Biases"
        if best_arms[i]:
            label = label + " at best arms"
        plot_avg_total_biases_one_metric_at_eval(
            squared = squared[i],
            normalized = normalized[i],
            best_arms = best_arms[i],
            input_path=input_path,
            plot_folder=plot_folder,
            project_name=project_name,
            loc = "best",
            grid = grid,
            mode = ("multiple plots",axs[i],label),
            safe_mode = safe_mode
        )
    
    # Collect all handles and labels
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Remove duplicate labels, adjust layout
    unique = dict(zip(labels,handles))
    fig.legend(unique.values(), unique.keys(), loc = "upper right")
    fig.suptitle("Some average bias metrics at evaluations",fontsize=16)
    if remove_index != -num_rows:
        for j in range(int(num_cols * num_rows + remove_index),int(num_cols * num_rows)):
            fig.delaxes(axs[j])
    fig.tight_layout()

    # If save is activated, make the save folder and save the plot
    if save:
        os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
        plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_total_biases_multiple_metrics_at_eval.png"))

    # If show is activated, show the plot
    if show:
        plt.show()

# Plot all bias metrics
def plot_avg_total_biases_all_metrics_at_eval(
        squared: bool = True,
        normalized: bool = True,
        best_arms: bool = True,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        safe_mode: bool = True) -> None:
    
    """
    Plots all bias metrics available. You can choose to plot only the squared metrics, only the regular metrics or both. The data is given either 
    by input_path or if input_path is None it is given by combining the path given by plot_folder and "project_name.pkl". The figsize of the 
    individual plots can be chosen as well as if the figure should be shown and/or saved. If safe_mode is activated a check on the parameters is 
    performed.

    Parameters:
    - squared (bool): If True, plots all the squared bias metrics.
    - normalized (bool): If True, plots all the normalized metrics.
    - best_arms (bool): If True, plots all the bias metrics at the best arms.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.    
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if not (isinstance(squared,bool) and isinstance(normalized,bool) and isinstance(best_arms,bool)):
            raise TypeError("The parameters squared, normalized, and best_arms need to be boolean!")
    
    # Fill the list of configurations
    squared_list = [False]
    normalized_list = [False]
    best_arms_list = [False]
    if normalized:
        squared_list.append(False)
        normalized_list.append(True)
        best_arms_list.append(False)
    if best_arms:
        squared_list.append(False)
        normalized_list.append(False)
        best_arms_list.append(True)
        if normalized:
            squared_list.append(False)
            normalized_list.append(True)
            best_arms_list.append(True)
    if squared:
        squared_list.append(True)
        normalized_list.append(False)
        best_arms_list.append(False)
        if normalized:
            squared_list.append(True)
            normalized_list.append(True)
            best_arms_list.append(False)
        if best_arms:
            squared_list.append(True)
            normalized_list.append(False)
            best_arms_list.append(True)
            if normalized:
                squared_list.append(True)
                normalized_list.append(True)
                best_arms_list.append(True)

    plot_avg_total_biases_multiple_metrics_at_eval(
        squared= squared_list,
        normalized= normalized_list,
        best_arms= best_arms_list,
        input_path= input_path,
        plot_folder= plot_folder,
        project_name= project_name,
        individual_figsize= individual_figsize,
        num_rows= num_rows,
        grid= grid,
        show= show,
        save= save,
        safe_mode= safe_mode)

# Plot multiple special log plots at steps
def plot_avg_special_logs_multiple_at_step(
        index: List[int] = None,
        real_value: Dict[int,Union[int,float]] = None,
        real_value_label: Dict[int,str] = None,
        input_path: str  = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        safe_mode: bool = True) -> None:
    
    """
    Plots multiple chosen special logs at steps into one figure. The data is given either by input_path or if input_path is None it is given 
    by combining the path given by plot_folder and "project_name.pkl". The figsize of the individual plots can be chosen as well as if the figure 
    should be shown and/or saved. If safe_mode is activated a check on the parameters is performed.

    Parameters:
    - index (list): A list containing the indices to be plotted.
    - real_value (dict): A dictionary of real values to be plotted in case the special logs should be converging to some value. If None, or some
      of the indices are not contained in the dictionary no real values will be plotted.
    - real_value_label (dict): A dictionary of labels to be given to the passed real values.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    # If safe_mode, perform a parameter check
    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if isinstance(index,list):
            for i in index:
                if isinstance(i,int):
                    if i < 0:
                        raise ValueError("All indices must be non-negative integers!")
                else:
                    raise TypeError("All indices must be non-negative integers!")
        else:
            raise ValueError("The state action pairs whose bias should be plotted need to be given as a list!")
        if isinstance(num_rows,int):
            if num_rows <= 0:
                raise ValueError("Number of rows needs to be positive!")
        else:
            raise TypeError("Number of rows needs to be an integer!")
        if real_value is not None:
            if isinstance(real_value,dict):
                for key in real_value.keys():
                    if key in index:
                        if not isinstance(real_value[key],(int,float)):
                            raise TypeError("The given real values to be plotted need to be numerical values!")
            else:
                raise TypeError("The given real values to be plotted need to be passed in a dictionary!")
            if isinstance(real_value_label,dict):
                for key in real_value_label.keys():
                    if key in index and key in real_value.keys():
                        if not isinstance(real_value_label[key],str):
                            raise TypeError("The given labels for the real values to be plotted need to be strings!")
                    else:
                        raise ValueError("Labels may only be given for indexec for which real values are provided!")
            else:
                raise TypeError("The given labels for the real values to be plotted need to be passed in a dictionary!")
    
    if index is None:
        raise ValueError("You need to specify which special logs you want to put in your plot!")
    
    # Initialize the unified plot
    len_plot = len(index)
    num_cols = int(np.ceil(len_plot / num_rows))
    remove_index = - (num_rows - (len_plot % num_rows))
    fig, axs = plt.subplots(num_cols,num_rows,figsize=(individual_figsize[0]*num_rows,individual_figsize[1]*num_cols))
    
    # Flatten the axis
    axs = axs.flatten()

    # Plot the subplots using the dedicated function
    for i, ind in enumerate(index):
        if real_value is not None:
            if ind in real_value.keys():
                rv = real_value[ind]
                rvl = real_value_label[ind]
        else:
            rv = None
            rvl = None
        plot_avg_special_logs_one_at_step(
            index = index[i],
            real_value = rv,
            real_value_label= rvl,
            input_path = input_path,
            plot_folder = plot_folder,
            project_name = project_name,
            figsize = individual_figsize,
            grid = grid,
            mode = ("multiple plots", axs[index],None),
            safe_mode = safe_mode
            )
    
    # Collect all handles and labels
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Remove duplicate labels, adjust layout
    unique = dict(zip(labels,handles))
    fig.legend(unique.values(), unique.keys(), loc = "upper right")
    fig.suptitle("Some average special logs at steps",fontsize=16)
    if remove_index != -num_rows:
        for j in range(int(num_cols * num_rows + remove_index),int(num_cols * num_rows)):
            fig.delaxes(axs[j])
    fig.tight_layout()

    # If save is activated, make the save folder and save the plot
    if save:
        os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
        plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_special_logs_multiple_at_step.png"))

    # If show is activated, show the plot
    if show:
        plt.show()

# Plot multiple special log plots at epochs
def plot_avg_special_logs_multiple_at_epoch(
        index: List[int] = None,
        real_value: Dict[int,Union[int,float]] = None,
        real_value_label: Dict[int,str] = None,
        input_path: str  = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        safe_mode: bool = True) -> None:
    
    """
    Plots multiple chosen special logs at epochs into one figure. The data is given either by input_path or if input_path is None it is given 
    by combining the path given by plot_folder and "project_name.pkl". The figsize of the individual plots can be chosen as well as if the figure 
    should be shown and/or saved. If safe_mode is activated a check on the parameters is performed.

    Parameters:
    - index (list): A list containing the indices to be plotted.
    - real_value (dict): A dictionary of real values to be plotted in case the special logs should be converging to some value. If None, or some
      of the indices are not contained in the dictionary no real values will be plotted.
    - real_value_label (dict): A dictionary of labels to be given to the passed real values.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    # If safe_mode, perform a parameter check
    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if isinstance(index,list):
            for i in index:
                if isinstance(i,int):
                    if i < 0:
                        raise ValueError("All indices must be non-negative integers!")
                else:
                    raise TypeError("All indices must be non-negative integers!")
        else:
            raise ValueError("The state action pairs whose bias should be plotted need to be given as a list!")
        if isinstance(num_rows,int):
            if num_rows <= 0:
                raise ValueError("Number of rows needs to be positive!")
        else:
            raise TypeError("Number of rows needs to be an integer!")
        if real_value is not None:
            if isinstance(real_value,dict):
                for key in real_value.keys():
                    if key in index:
                        if not isinstance(real_value[key],(int,float)):
                            raise TypeError("The given real values to be plotted need to be numerical values!")
            else:
                raise TypeError("The given real values to be plotted need to be passed in a dictionary!")
            if isinstance(real_value_label,dict):
                for key in real_value_label.keys():
                    if key in index and key in real_value.keys():
                        if not isinstance(real_value_label[key],str):
                            raise TypeError("The given labels for the real values to be plotted need to be strings!")
                    else:
                        raise ValueError("Labels may only be given for indexec for which real values are provided!")
            else:
                raise TypeError("The given labels for the real values to be plotted need to be passed in a dictionary!")
    
    if index is None:
        raise ValueError("You need to specify which special logs you want to put in your plot!")
    
    # Initialize the unified plot
    len_plot = len(index)
    num_cols = int(np.ceil(len_plot / num_rows))
    remove_index = - (num_rows - (len_plot % num_rows))
    fig, axs = plt.subplots(num_cols,num_rows,figsize=(individual_figsize[0]*num_rows,individual_figsize[1]*num_cols))
    
    # Flatten the axis
    axs = axs.flatten()

    # Plot the subplots using the dedicated function
    for i, ind in enumerate(index):
        if real_value is not None:
            if ind in real_value.keys():
                rv = real_value[ind]
                rvl = real_value_label[ind]
        else:
            rv = None
            rvl = None
        plot_avg_special_logs_one_at_epoch(
            index = index[i],
            real_value = rv,
            real_value_label= rvl,
            input_path = input_path,
            plot_folder = plot_folder,
            project_name = project_name,
            figsize = individual_figsize,
            grid = grid,
            mode = ("multiple plots", axs[index],None),
            safe_mode = safe_mode
            )
    
    # Collect all handles and labels
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Remove duplicate labels, adjust layout
    unique = dict(zip(labels,handles))
    fig.legend(unique.values(), unique.keys(), loc = "upper right")
    fig.suptitle("Some average special logs at epochs",fontsize=16)
    if remove_index != -num_rows:
        for j in range(int(num_cols * num_rows + remove_index),int(num_cols * num_rows)):
            fig.delaxes(axs[j])
    fig.tight_layout()

    # If save is activated, make the save folder and save the plot
    if save:
        os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
        plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_special_logs_multiple_at_epoch.png"))

    # If show is activated, show the plot
    if show:
        plt.show()

# Plot multiple special log plots at evals
def plot_avg_special_logs_multiple_at_eval(
        index: List[int] = None,
        real_value: Dict[int,Union[int,float]] = None,
        real_value_label: Dict[int,str] = None,
        input_path: str  = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        safe_mode: bool = True) -> None:
    
    """
    Plots multiple chosen special logs at evaluations into one figure. The data is given either by input_path or if input_path is None it is given 
    by combining the path given by plot_folder and "project_name.pkl". The figsize of the individual plots can be chosen as well as if the figure 
    should be shown and/or saved. If safe_mode is activated a check on the parameters is performed.

    Parameters:
    - index (list): A list containing the indices to be plotted.
    - real_value (dict): A dictionary of real values to be plotted in case the special logs should be converging to some value. If None, or some
      of the indices are not contained in the dictionary no real values will be plotted.
    - real_value_label (dict): A dictionary of labels to be given to the passed real values.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    # If safe_mode, perform a parameter check
    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if isinstance(index,list):
            for i in index:
                if isinstance(i,int):
                    if i < 0:
                        raise ValueError("All indices must be non-negative integers!")
                else:
                    raise TypeError("All indices must be non-negative integers!")
        else:
            raise ValueError("The state action pairs whose bias should be plotted need to be given as a list!")
        if isinstance(num_rows,int):
            if num_rows <= 0:
                raise ValueError("Number of rows needs to be positive!")
        else:
            raise TypeError("Number of rows needs to be an integer!")
        if real_value is not None:
            if isinstance(real_value,dict):
                for key in real_value.keys():
                    if key in index:
                        if not isinstance(real_value[key],(int,float)):
                            raise TypeError("The given real values to be plotted need to be numerical values!")
            else:
                raise TypeError("The given real values to be plotted need to be passed in a dictionary!")
            if isinstance(real_value_label,dict):
                for key in real_value_label.keys():
                    if key in index and key in real_value.keys():
                        if not isinstance(real_value_label[key],str):
                            raise TypeError("The given labels for the real values to be plotted need to be strings!")
                    else:
                        raise ValueError("Labels may only be given for indexec for which real values are provided!")
            else:
                raise TypeError("The given labels for the real values to be plotted need to be passed in a dictionary!")
    
    if index is None:
        raise ValueError("You need to specify which special logs you want to put in your plot!")
    
    # Initialize the unified plot
    len_plot = len(index)
    num_cols = int(np.ceil(len_plot / num_rows))
    remove_index = - (num_rows - (len_plot % num_rows))
    fig, axs = plt.subplots(num_cols,num_rows,figsize=(individual_figsize[0]*num_rows,individual_figsize[1]*num_cols))
    
    # Flatten the axis
    axs = axs.flatten()

    # Plot the subplots using the dedicated function
    for i, ind in enumerate(index):
        if real_value is not None:
            if ind in real_value.keys():
                rv = real_value[ind]
                rvl = real_value_label[ind]
        else:
            rv = None
            rvl = None
        plot_avg_special_logs_one_at_eval(
            index = ind,
            real_value = rv,
            real_value_label= rvl,
            input_path = input_path,
            plot_folder = plot_folder,
            project_name = project_name,
            figsize = individual_figsize,
            grid = grid,
            mode = ("multiple plots", axs[i],None),
            safe_mode = safe_mode
            )
    
    # Collect all handles and labels
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Remove duplicate labels, adjust layout
    unique = dict(zip(labels,handles))
    fig.legend(unique.values(), unique.keys(), loc = "upper right")
    fig.suptitle("Some average special logs at evaluations",fontsize=16)
    if remove_index != -num_rows:
        for j in range(int(num_cols * num_rows + remove_index),int(num_cols * num_rows)):
            fig.delaxes(axs[j])
    fig.tight_layout()

    # If save is activated, make the save folder and save the plot
    if save:
        os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
        plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),"average_special_logs_multiple_at_eval.png"))

    # If show is activated, show the plot
    if show:
        plt.show()

# Plot all special log plots at steps
def plot_avg_special_logs_all_at_step(
        num_index: int = 1,
        real_value: Dict[int,Union[int,float]] = None,
        real_value_label: Dict[int,str] = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        safe_mode: bool = True) -> None:
    
    """
    Plots all special logs at steps. The data is given either by input_path or if input_path is None it is given by combining the path given by 
    plot_folder and "project_name.pkl". The figsize of the individual plots can be chosen as well as if the figure should be shown and/or saved. 
    If safe_mode is activated a check on the parameters is performed.

    Parameters:
    - num_index (int): The number of indices of the special logs.
    - real_value (dict): A dictionary of real values to be plotted in case the special logs should be converging to some value. If None, or some
      of the indices are not contained in the dictionary no real values will be plotted.
    - real_value_label (dict): A dictionary of labels to be given to the passed real values.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if isinstance(num_index,int):
            if not num_index > 0:
                raise ValueError("The number of indices needs to be a positive integer!")
        else:
            raise TypeError("The number of indices needs to be a positive integer!")
        if real_value is not None:
            if isinstance(real_value,dict):
                for key in real_value.keys():
                    if key in range(num_index):
                        if not isinstance(real_value[key],(int,float)):
                            raise TypeError("The given real values to be plotted need to be numerical values!")
            else:
                raise TypeError("The given real values to be plotted need to be passed in a dictionary!")
            if isinstance(real_value_label,dict):
                for key in real_value_label.keys():
                    if key in range(num_index) and key in real_value.keys():
                        if not isinstance(real_value_label[key],str):
                            raise TypeError("The given labels for the real values to be plotted need to be strings!")
                    else:
                        raise ValueError("Labels may only be given for indexec for which real values are provided!")
            else:
                raise TypeError("The given labels for the real values to be plotted need to be passed in a dictionary!")
        
    # Fill the list of configurations
    index_list = [i for i in range(num_index)]
    

    plot_avg_special_logs_multiple_at_step(
        index=index_list,
        real_value=real_value,
        real_value_label=real_value_label,
        input_path= input_path,
        plot_folder= plot_folder,
        project_name= project_name,
        individual_figsize= individual_figsize,
        num_rows= num_rows,
        grid= grid,
        show= show,
        save= save,
        safe_mode= safe_mode)

# Plot all special log plots at epochs
def plot_avg_special_logs_all_at_epoch(
        num_index: int = 1,
        real_value: Dict[int,Union[int,float]] = None,
        real_value_label: Dict[int,str] = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        safe_mode: bool = True) -> None:
    
    """
    Plots all special logs at epochs. The data is given either by input_path or if input_path is None it is given by combining the path given by 
    plot_folder and "project_name.pkl". The figsize of the individual plots can be chosen as well as if the figure should be shown and/or saved. 
    If safe_mode is activated a check on the parameters is performed.

    Parameters:
    - num_index (int): The number of indices of the special logs.
    - real_value (dict): A dictionary of real values to be plotted in case the special logs should be converging to some value. If None, or some
      of the indices are not contained in the dictionary no real values will be plotted.
    - real_value_label (dict): A dictionary of labels to be given to the passed real values.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if isinstance(num_index,int):
            if not num_index > 0:
                raise ValueError("The number of indices needs to be a positive integer!")
        else:
            raise TypeError("The number of indices needs to be a positive integer!")
        if real_value is not None:
            if isinstance(real_value,dict):
                for key in real_value.keys():
                    if key in range(num_index):
                        if not isinstance(real_value[key],(int,float)):
                            raise TypeError("The given real values to be plotted need to be numerical values!")
            else:
                raise TypeError("The given real values to be plotted need to be passed in a dictionary!")
            if isinstance(real_value_label,dict):
                for key in real_value_label.keys():
                    if key in range(num_index) and key in real_value.keys():
                        if not isinstance(real_value_label[key],str):
                            raise TypeError("The given labels for the real values to be plotted need to be strings!")
                    else:
                        raise ValueError("Labels may only be given for indexec for which real values are provided!")
            else:
                raise TypeError("The given labels for the real values to be plotted need to be passed in a dictionary!")
        
    # Fill the list of configurations
    index_list = [i for i in range(num_index)]
    

    plot_avg_special_logs_multiple_at_epoch(
        index=index_list,
        real_value=real_value,
        real_value_label=real_value_label,
        input_path= input_path,
        plot_folder= plot_folder,
        project_name= project_name,
        individual_figsize= individual_figsize,
        num_rows= num_rows,
        grid= grid,
        show= show,
        save= save,
        safe_mode= safe_mode)

# Plot all special log plots at evaluations
def plot_avg_special_logs_all_at_eval(
        num_index: int = 1,
        real_value: Dict[int,Union[int,float]] = None,
        real_value_label: Dict[int,str] = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        safe_mode: bool = True) -> None:
    
    """
    Plots all special logs at evaluations. The data is given either by input_path or if input_path is None it is given by combining the path given by 
    plot_folder and "project_name.pkl". The figsize of the individual plots can be chosen as well as if the figure should be shown and/or saved. 
    If safe_mode is activated a check on the parameters is performed.

    Parameters:
    - num_index (int): The number of indices of the special logs.
    - real_value (dict): A dictionary of real values to be plotted in case the special logs should be converging to some value. If None, or some
      of the indices are not contained in the dictionary no real values will be plotted.
    - real_value_label (dict): A dictionary of labels to be given to the passed real values.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if isinstance(num_index,int):
            if not num_index > 0:
                raise ValueError("The number of indices needs to be a positive integer!")
        else:
            raise TypeError("The number of indices needs to be a positive integer!")
        
    # Fill the list of configurations
    index_list = [i for i in range(num_index)]
    

    plot_avg_special_logs_multiple_at_eval(
        index=index_list,
        real_value=real_value,
        real_value_label=real_value_label,
        input_path= input_path,
        plot_folder= plot_folder,
        project_name= project_name,
        individual_figsize= individual_figsize,
        num_rows= num_rows,
        grid= grid,
        show= show,
        save= save,
        safe_mode= safe_mode)

###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################

# Functions for different multiple plot types in general

###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################

# Selected plots to print
def selected_plots_one_experiment(
        plottitle: str = "Selected plots",
        subplottitles: List = None,
        plotlist: List[str] = None,
        correct_action_log: bool = True,
        focus_state_actions: bool = True,
        bias_estimation: bool = True,
        algo_special_logs: bool = True,
        squared_normalized_best_arms_list: List[List[tuple[bool]]] = None,
        which: List[List[Tuple[int,Union[str,int]]]] = None,
        max_steps_per_epoch: int = -1,
        index: List[List[List[int]]] = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        save_string: str = "selected_plots",
        safe_mode: bool = True) -> None:
    
    """
    Plots the multiple chosen training evaluation metrics given by the keys in plotlist in that order. If one of the conditional plots was chosen,
    its corresponding condition needs to be turned on and the respective parameters need to be passed. The data is given either by input_path or if 
    input_path is None it is given by combining the path given by plot_folder and "project_name.pkl". The figsize of the individual plots can be 
    chosen as well as if the figure should be shown and/or saved. If safe_mode is activated a check on the parameters is performed.

    Parameters:
    - plotlist (list): A list containing valid keys for the plot functions to be used.
    - subplottitles (list): A list of individual titles for the plots.
    - correct_action_log (bool): If True, allows for plotting the correct action rate plots.
    - focus_state_actions (bool): If True, allows for plotting the focussed states and actions.
    - bias_estimation (bool): If True, allows for plotting the bias metrics.
    - algo_special_logs (bool): If True, allows for plotting the special logs of the algorithms.
    - squared_normalized_best_arms_list (list): A list of lists of configurations for the bias metrics plots. For all plots to be plotted a list of three boolean
      values needs to be contained in the desired order as a tuple. If the first is True, the squared metrics will be accesed. If the second is True, the normalized
      metrics will be accessed. If the third is True, only the bias at the best arms will be factored in.
    - which (list): A list of lists containing lists of state action pairs whose bias should be plotted.
    - max_steps_per_epoch (int): The given maximum amount of steps an epoch is allowed to take. If -1 it means no maximum is set and the percent of
      capped epochs plot can not be plotted.
    - index (list): A list of lists of indices for the special logs at step, at epoch, and at eval.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - save_string (str): The string with which the plots should be saved.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    # If safe_mode, perform a parameter check
    if safe_mode:
        check_input_for_single_plot_fct(input_path=input_path,plot_folder=plot_folder,project_name=project_name,figsize=individual_figsize,grid=grid,show=show,save=save,safe_mode=safe_mode,loc="best",mode="single plot")
        if isinstance(plotlist,list):
            for key in plotlist:
                if not(key in ALL_SINGLE_PLOT_KEYNAMES_EPOCHS.keys() or key in ALL_SINGLE_PLOT_KEYNAMES_EVAL.keys() or key in ALL_SINGLE_PLOT_KEYNAMES_OTHER.keys() or key in ALL_SINGLE_PLOT_KEYNAMES_STEPS.keys() or key == "empty"):
                    raise ValueError(f"The key {key} is not a valid key for a plot function!")
        else: 
            raise TypeError("The keys to the plot functions need to be contained in a list!")
        if isinstance(subplottitles,list):
            for subtitle in subplottitles:
                if not isinstance(subtitle,str):
                    raise ValueError("The subplot titles need to be strings!")
        else:
            raise ValueError("The subplot titles need to be passed in a list!")
        if not isinstance(correct_action_log,bool):
            raise TypeError("Parameter correct_action_log needs to be boolean!")
        if not isinstance(focus_state_actions,bool):
            raise TypeError("Parameter initial_q_fct_estimation_log needs to be boolean!")
        if not isinstance(bias_estimation,bool):
            raise TypeError("Parameter bias_estimation needs to be boolean!")
        if not isinstance(algo_special_logs,bool):
            raise TypeError("Parameter algo_special_logs needs to be boolean!")
        if isinstance(squared_normalized_best_arms_list,list):
            for sqnlist in squared_normalized_best_arms_list:
                if isinstance(sqnlist,list):
                    for threelist in sqnlist:
                        if isinstance(threelist,tuple):
                            if len(threelist) == 3:
                                for val in threelist:
                                    if not isinstance(val,bool):
                                        raise ValueError("The values in the list of snb configurations need to be boolean!")
                            else:
                                raise TypeError("The configurations for the bias metrics to be plotted need to be passed in a list containing tuples of three boolean values")
                        else:
                            raise TypeError("The configurations for the bias metrics to be plotted need to be passed in a list containing tuples of three boolean values")
                else:
                    raise ValueError("The configurations for the bias metrics to be plotted need to be passed as a list of lists!")
        else:
            raise TypeError("The configurations for the bias metrics to be plotted need to be passed in a list containing tuples of three boolean values")
        if isinstance(which,list):
            for whichlst in which:
                if isinstance(whichlst,list):
                    for sa in whichlst:
                        if isinstance(sa,tuple):
                            if len(sa) == 2:
                                if not (isinstance(sa[0],int) and (isinstance(sa[1],int) or sa[1] == "best")):
                                    raise ValueError(f"State action pair {sa} is invalid!")
                            else:
                                raise ValueError(f"State action pair {sa} is not a tuple of length 2!")
                        else:
                            raise ValueError(f"State action pair {sa} is not a tuple of length 2!")
                else:
                    raise ValueError("The state action pairs whose bias should be plotted need to be given as a list of lists!")
        else:
            raise ValueError("The state action pairs whose bias should be plotted need to be given as a list of lists!")
        if isinstance(index,list):
            for indlst in index:
                if isinstance(indlst,list):
                    if len(indlst) == 3:
                        for ind_list in indlst:
                            if isinstance(ind_list,list):
                                for ind in ind_list:
                                    if isinstance(ind,int):
                                        if ind < 0:
                                            raise ValueError("Indices must be non-negative integers!")
                                    else:
                                        raise TypeError("Indices must be non-negative integers!")
                            else:
                                raise TypeError("The indices for the special logs need to be given as a list of three lists!")
                    else:
                        raise TypeError("The indices for the special logs need to be given as a list of three lists!")
                else:
                    raise TypeError("The indices for the special logs need to be given as a list of lists of three lists!")
        else:
            raise TypeError("The indices for the special logs need to be given as a list of lists of three lists!")
        if isinstance(max_steps_per_epoch,int):
            if not (max_steps_per_epoch > 0 or max_steps_per_epoch== -1):
                raise ValueError("The maximum amount of allowed steps per epoch needs to be either a positive integer or -1!")
        else:
            raise TypeError("The maximum amount of allowed steps per epoch needs to be an integer!")
        if isinstance(num_rows,int):
            if num_rows <= 0:
                raise ValueError("Number of rows needs to be positive!")
        else:
            raise TypeError("Number of rows needs to be an integer!")
    
    # Initialize non-mutable stuff
    if plotlist is None:
        plotlist = ["Mean eval results", "Mean eval correct action rates"]
    if squared_normalized_best_arms_list is None:
        squared_normalized_best_arms_list = [[(True,True,True)]]
    if which is None:
        which = [[(0,"best")]]
    if index is None:
        index = [[0]]

    # Allow certain plots
    allowed_plot_act = ["default_act"]
    if correct_action_log:
        allowed_plot_act.append("correct_action_log_act")
    if focus_state_actions:
        allowed_plot_act.append("focus_state_actions_act")
    if bias_estimation:
        allowed_plot_act.append("bias_estimation_act")
    if max_steps_per_epoch != -1:
        allowed_plot_act.append("max_steps_per_epoch_act")
    if algo_special_logs:
        allowed_plot_act.append("special_act")
    
    # Initialize special remove list for empty plots
    special_remove_list = []

    # Initialize the unified plot
    len_plot = len(plotlist)
    if "Mean bias metrics at evals" in plotlist:
        for sqn_list_inner in squared_normalized_best_arms_list:
            len_plot += len(sqn_list_inner) - 1
    if "Mean biases at chosen at evals" in plotlist:
        for which_inner in which:
            len_plot += len(which_inner) - 1
    elif "Mean Q function values at chosen at evals" in plotlist:
        for which_inner in which:
            len_plot += len(which_inner) - 1
    if "Mean special logs at steps" in plotlist:
        for index_inner in index:
            len_plot += len(index_inner[0]) - 1
    elif "Mean special logs at epochs" in plotlist:
        for index_inner in index:
            len_plot += len(index_inner[1]) - 1    
    elif "Mean special logs at evals" in plotlist:
        for index_inner in index:
            len_plot += len(index_inner[2]) - 1
    mbm_index = 0
    wh_index = 0
    ind_index = 0
    num_cols = int(np.ceil(len_plot / num_rows))
    remove_index = - (num_rows - (len_plot % num_rows))
    fig, axs = plt.subplots(num_cols,num_rows,figsize=(individual_figsize[0]*num_rows,individual_figsize[1]*num_cols))
    
    # Flatten the axis
    axs = axs.flatten()

    # Plot the subplots using the dedicated function
    dunked = 0
    additional = 0
    for plot_index, plotkey in enumerate(plotlist):
        keep = False
        if plotkey in ALL_SINGLE_PLOT_KEYNAMES_STEPS.keys():
            if ALL_SINGLE_PLOT_KEYNAMES_STEPS[plotkey][1] in allowed_plot_act:
                goalfunc = ALL_SINGLE_PLOT_KEYNAMES_STEPS[plotkey][0]
                keep = True
            else:
                dunked += 1
        elif plotkey in ALL_SINGLE_PLOT_KEYNAMES_EPOCHS.keys():
            if ALL_SINGLE_PLOT_KEYNAMES_EPOCHS[plotkey][1] in allowed_plot_act:
                goalfunc = ALL_SINGLE_PLOT_KEYNAMES_EPOCHS[plotkey][0]
                if subplottitles is not None:
                    if plot_index - dunked + additional < len(subplottitles):
                        title = subplottitles[plot_index - dunked + additional]
                    else:
                        title = ALL_SINGLE_PLOT_KEYNAMES_EPOCHS[plotkey][2]
                else:
                    title = ALL_SINGLE_PLOT_KEYNAMES_EPOCHS[plotkey][2]
                keep = True
            else:
                dunked += 1
        elif plotkey in ALL_SINGLE_PLOT_KEYNAMES_EVAL.keys():
            if ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][1] in allowed_plot_act:
                goalfunc = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][0]
                if subplottitles is not None:
                    if plot_index - dunked + additional < len(subplottitles):
                        title = subplottitles[plot_index - dunked + additional]
                    else:
                        title = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][2]
                else:
                    title = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][2]
                keep = True
            else:
                if plotkey == "Mean bias metrics":
                    dunked += len(squared_normalized_best_arms_list[mbm_index])
                    mbm_index+=1
                elif plotkey == "Mean biases at chosen state action":
                    dunked += len(which[wh_index])
                    wh_index += 1
                else:
                    dunked += 1
        elif plotkey in ALL_SINGLE_PLOT_KEYNAMES_OTHER.keys():
            if ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][1] in allowed_plot_act:
                goalfunc = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][0]
                if subplottitles is not None:
                    if plot_index - dunked + additional < len(subplottitles):
                        title = subplottitles[plot_index - dunked + additional]
                    else:
                        title = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][2]
                else:
                    title = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][2]
                keep = True
            else:
                if plotkey in ["Mean special logs at steps","Mean special logs at epochs","Mean special logs at evals"]:
                    dunked += len(index[ind_index])
                    ind_index += 1
                dunked += 1
        elif plotkey == "empty":
            keep = True
        if keep:
            if plotkey == "Mean bias metrics at evals":
                for i, snb in enumerate(squared_normalized_best_arms_list[mbm_index]):
                    if i > 0:
                        additional += 1
                    if subplottitles is not None:
                        if plot_index - dunked + additional < len(subplottitles):
                            title = subplottitles[plot_index - dunked + additional]
                        else:
                            title = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][2]
                    else:
                        title = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][2]
                    goalfunc(
                        squared = snb[0],
                        normalized = snb[1],
                        best_arms = snb[2],
                        input_path = input_path,
                        plot_folder = plot_folder,
                        project_name = project_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[plot_index - dunked + additional],title),
                        safe_mode = True
                        )
                mbm_index += 1
            elif plotkey == "Mean biases at chosen at evals":
                for i, sa in enumerate(which[wh_index]):
                    if i > 0:
                        additional += 1
                    if subplottitles is not None:
                        if plot_index - dunked + additional < len(subplottitles):
                            title = subplottitles[plot_index - dunked + additional]
                        else:
                            title = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][2]
                    else:
                        title = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][2]
                    goalfunc(
                        which = sa,
                        input_path = input_path,
                        plot_folder = plot_folder,
                        project_name = project_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[plot_index - dunked + additional],title),
                        safe_mode = True
                        )
                wh_index += 1
            elif plotkey == "Mean Q function values at chosen at evals":
                for i, sa in enumerate(which[wh_index]):
                    if i > 0:
                        additional += 1
                    if subplottitles is not None:
                        if plot_index - dunked + additional < len(subplottitles):
                            title = subplottitles[plot_index - dunked + additional]
                        else:
                            title = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][2]
                    else:
                        title = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][2]
                    goalfunc(
                        which = sa,
                        input_path = input_path,
                        plot_folder = plot_folder,
                        project_name = project_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[plot_index - dunked + additional],title),
                        safe_mode = True
                        )
                wh_index += 1
            elif plotkey == "Percent of capped epochs":
                if subplottitles is not None:
                    if plot_index - dunked + additional < len(subplottitles):
                        title = subplottitles[plot_index - dunked + additional]
                    else:
                        title = ALL_SINGLE_PLOT_KEYNAMES_EPOCHS[plotkey][2]
                else:
                    title = ALL_SINGLE_PLOT_KEYNAMES_EPOCHS[plotkey][2]
                goalfunc(
                    max_steps_per_epoch = max_steps_per_epoch,
                    input_path = input_path,
                    plot_folder = plot_folder,
                    project_name = project_name,
                    figsize = individual_figsize,
                    grid = grid,
                    mode = ("multiple plots", axs[plot_index - dunked + additional],title),
                    safe_mode = True
                    )
            elif plotkey == "Mean special logs at steps":
                for i, ind in enumerate(index[ind_index][0]):
                    if i > 0:
                        additional += 1
                    if subplottitles is not None:
                        if plot_index - dunked + additional < len(subplottitles):
                            title = subplottitles[plot_index - dunked + additional]
                        else:
                            title = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][2]
                    else:
                        title = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][2]
                    goalfunc(
                        index = ind,
                        input_path = input_path,
                        plot_folder = plot_folder,
                        project_name = project_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[plot_index - dunked + additional],title),
                        safe_mode = True
                        )
                ind_index += 1
            elif plotkey == "Mean special logs at epochs":
                for i, ind in enumerate(index[ind_index][1]):
                    if i > 0:
                        additional += 1
                    if subplottitles is not None:
                        if plot_index - dunked + additional < len(subplottitles):
                            title = subplottitles[plot_index - dunked + additional]
                        else:
                            title = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][2]
                    else:
                        title = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][2]
                    goalfunc(
                        index = ind,
                        input_path = input_path,
                        plot_folder = plot_folder,
                        project_name = project_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[plot_index - dunked + additional],title),
                        safe_mode = True
                        )
                ind_index += 1
            elif plotkey == "Mean special logs at evals":
                for i, ind in enumerate(index[ind_index][2]):
                    if i > 0:
                        additional += 1
                    if subplottitles is not None:
                        if plot_index - dunked + additional < len(subplottitles):
                            title = subplottitles[plot_index - dunked + additional]
                        else:
                            title = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][2]
                    else:
                        title = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][2]
                    goalfunc(
                        index = ind,
                        input_path = input_path,
                        plot_folder = plot_folder,
                        project_name = project_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[plot_index - dunked + additional],title),
                        safe_mode = True
                        )
                ind_index += 1
            elif plotkey == "empty":
                special_remove_list.append(plot_index - dunked + additional)
            else:
                if subplottitles is not None:
                    if plot_index - dunked + additional < len(subplottitles):
                        title = subplottitles[plot_index - dunked + additional]
                    else:
                        if plotkey in ALL_SINGLE_PLOT_KEYNAMES_EPOCHS.keys():
                            title = ALL_SINGLE_PLOT_KEYNAMES_EPOCHS[plotkey][2]
                        if plotkey in ALL_SINGLE_PLOT_KEYNAMES_EVAL.keys():
                            title = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][2]
                        if plotkey in ALL_SINGLE_PLOT_KEYNAMES_STEPS.keys():
                            title = ALL_SINGLE_PLOT_KEYNAMES_STEPS[plotkey][2]
                        if plotkey in ALL_SINGLE_PLOT_KEYNAMES_OTHER.keys():
                            title = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][2]
                else:
                    if plotkey in ALL_SINGLE_PLOT_KEYNAMES_EPOCHS.keys():
                        title = ALL_SINGLE_PLOT_KEYNAMES_EPOCHS[plotkey][2]
                    if plotkey in ALL_SINGLE_PLOT_KEYNAMES_EVAL.keys():
                        title = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][2]
                    if plotkey in ALL_SINGLE_PLOT_KEYNAMES_STEPS.keys():
                        title = ALL_SINGLE_PLOT_KEYNAMES_STEPS[plotkey][2]
                    if plotkey in ALL_SINGLE_PLOT_KEYNAMES_OTHER.keys():
                        title = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][2]
                goalfunc(
                    input_path = input_path,
                    plot_folder = plot_folder,
                    project_name = project_name,
                    figsize = individual_figsize,
                    grid = grid,
                    mode = ("multiple plots", axs[plot_index - dunked + additional],title),
                    safe_mode = True
                    )
        else:
            if plotkey == "Mean bias metrics at evals":
                remove_index += - len(squared_normalized_best_arms_list)
            elif plotkey == "Mean biases at chosen at evals":
                remove_index += - len(which)
            elif plotkey == "Mean Q function values at chosen at evals":
                remove_index += - len(which)
            elif plotkey == "Mean special logs at steps":
                remove_index += - len(index) 
            elif plotkey == "Mean special logs at epochs":
                remove_index += - len(index)
            elif plotkey == "Mean special logs at evals":
                remove_index += - len(index)  
            else:
                remove_index += -1
            print(f"Warning: The plot with the key {plotkey} will not be plotted as it has not been allowed to be plotted!")
    
    # Collect all handles and labels
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Remove duplicate labels, adjust layout
    unique = dict(zip(labels,handles))
    fig.legend(unique.values(), unique.keys(), loc = "right")
    if plottitle is not None:
        fig.suptitle(plottitle,fontsize=16)
    if remove_index != -num_rows:
        for j in range(int(num_cols * num_rows + remove_index),int(num_cols * num_rows)):
            fig.delaxes(axs[j])
    for i in special_remove_list:
        fig.delaxes(axs[i])
    fig.tight_layout()

    # If save is activated, make the save folder and save the plot
    if save:
        os.makedirs(os.path.join(plot_folder,project_name + "_plots"),exist_ok=True)
        plt.savefig(os.path.join(os.path.join(plot_folder, project_name + "_plots"),save_string + ".png"))

    # If show is activated, show the plot
    if show:
        plt.show()

# All plots for either evaluation or epoch metrics
def all_plots_one_experiment(
        board: str = "evaluation",
        correct_action_log: bool = True,
        focus_state_actions: bool = True,
        bias_estimation: bool = True,
        algo_special_logs: bool = True,
        algo_special_logs_where: List = None,
        which: List[Tuple[int,Union[str,int]]] = None,
        max_steps_per_epoch: int = -1,
        num_index: int = None,
        input_path: str = None,
        plot_folder: str = "plots",
        project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        save_string: str = "all_plots_evaluation",
        safe_mode: bool = True) -> None:  
    
    """
    Plots all training evaluation metrics either in terms of progress during epoch or at evaluation times, adding the runtime. If the data for the 
    conditional plots is present or not needs to be turned on and of which state action pairs there is data needs to also be passed. The data is given 
    either by input_path or if input_path is None it is given by combining the path given by plot_folder and "project_name.pkl". The figsize of the 
    individual plots can be chosen as well as if the figure should be shown and/or saved. If safe_mode is activated a check on the parameters is performed.

    Parameters:
    - board (str): If 'evaluation' or 'epoch' board should be plotted.
    - correct_action_log (bool): If True, allows for plotting the correct action rate plots.
    - initial_q_fct_estimation_log (bool): If True, allows for plotting the initial Q function estimation at the best arm.
    - bias_estimation (bool): If True, allows for plotting the bias metrics.
    - algo_special_logs (bool): If True, allows for plotting the special logs of the algorithms.
    - algo_special_logs_where (list): List containing "at_step", "at_eval", and/or "at_epoch", indicating for which of the three data should be plotted.
    - which (list): A list containing state action pairs whose bias should be plotted.
    - max_steps_per_epoch (int): The given maximum amount of steps an epoch is allowed to take. If -1 it means no maximum is set and the percent of
      capped epochs plot can not be plotted.
    - num_index (int): The number of special logs to be plotted.
    - input_path (str): The input path where the aggregated results file is located. Can be passed as None. In this case the path will be 
      constructed from the given plot folder and project name.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_name (str): The project name under which the plots should be saved. If no input path was given, simultaneously the file name in 
      which the results to be plotted are located.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - save_string (str): The string with which the plots should be saved.
    - safe_mode (bool): If True, a parameter check will be performed.
    """
    
    # If safe mode, perform initial parameter check
    if safe_mode:
        if not (board == "evaluation" or board == "epoch"):
            raise ValueError("Parameter board needs to be either 'evaluation' or 'epoch'!")
    
    # Initialize non-mutable stuff
    if which is None:
        which = ([0],[["best"]])
    if algo_special_logs_where is None:
        algo_special_logs_where = ["at_eval"]
    
    # Get all squared/normalized/best_arms list, all which list, and index list
    snb_list = [(True,True,True),(False,True,True),(True,False,True),(False,False,True),(True,True,False),(False,True,False),(True,False,False),(False,False,False)]
    which_list = []
    for state_index, state in which[0]:
        for action in which[1][state_index]:
            which_list.append((state,action))
    index_list = [i for i in range(num_index)]

    if board == "evaluation":
        plotlist = list(ALL_SINGLE_PLOT_KEYNAMES_EVAL.keys())
        extend_list = ["Runtimes"]
        if "at_eval" in algo_special_logs_where:
            extend_list.extend("Mean special logs at evals")
        extend_extend_list = list(ALL_SINGLE_PLOT_KEYNAMES_STEPS.keys())
        plotlist.extend(extend_list)
        plotlist.extend(extend_extend_list)
        if "at_step" in algo_special_logs_where:
            extend_extend_extend_list = ["Mean special logs at steps"]
            plotlist.extend(extend_extend_extend_list)
        plottitle = "Training metrics at evaluations"
    elif board == "epoch":
        plotlist = list(ALL_SINGLE_PLOT_KEYNAMES_EPOCHS.keys())
        extend_list = ["Runtimes"]
        if "at_epoch" in algo_special_logs_where:
            extend_list.extend("Mean special logs at epochs")
        extend_extend_list = list(ALL_SINGLE_PLOT_KEYNAMES_STEPS.keys())
        plotlist.extend(extend_list)
        plotlist.extend(extend_extend_list)
        if "at_step" in algo_special_logs_where:
            extend_extend_extend_list = ["Mean special logs at steps"]
            plotlist.extend(extend_extend_extend_list)
        plottitle = "Training metrics at epochs"
    
    selected_plots_one_experiment(
        plottitle= plottitle,
        plotlist=plotlist,
        correct_action_log=correct_action_log,
        focus_state_actions=focus_state_actions,
        bias_estimation=bias_estimation,
        algo_special_logs=algo_special_logs,
        squared_normalized_best_arms_list=snb_list,
        which=which_list,
        max_steps_per_epoch=max_steps_per_epoch,
        index=index_list,
        input_path=input_path,
        plot_folder=plot_folder,
        project_name=project_name,
        individual_figsize=individual_figsize,
        num_rows=num_rows,
        grid=grid,
        show=show,
        save=save,
        save_string=save_string,
        safe_mode=safe_mode
    )

def one_plot_multiple_experiments(
        plottitle: str = "Selected experiments",
        plotkey: str = "Mean eval results",
        squared: bool = True,
        normalized: bool = True,
        best_arms: bool = True,
        which: List[Tuple[int,Union[str,int]]] = (0,"best"),
        max_steps_per_epoch: int = -1,
        input_paths: str = None,
        plot_folder: str = "plots",
        project_names: str = ["testproject"],
        titles: str = ["Testproject"],
        save_project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        save_string: str = "testproject_multiple",
        safe_mode: bool = True
) -> None:
    
    """
    Plots given trainin evaluation metric in plotkey. If it is a plot needing conditional data, only the arguments are required, but not confirming that
    the conditionas are fulfilled.The data is given in order by the list of input paths or project names if input path is None. The figsize of the 
    individual plots can be chosen as well as if the figure should be shown and/or saved. If safe_mode is activated a check on the parameters is performed.

    Parameters:
    - plottitle (str): The title given to the plot.
    - plotkey (str): A valid plotkey.
    - squared (bool): If True, the squared bias metrics will be used in case the biases are plotted.
    - normalized (bool): If True, the normalized bias metrics will be used in case the biases are plotted.
    - best_arms (bool); If True, only the biases at the best arms will be taken into account for the bias metrics in case the biases are plotted.
    - which (tuple): A tuple containing state action pairs whose bias should be plotted in case this type of plot was chosen.
    - max_steps_per_epoch (int): The given maximum amount of steps an epoch is allowed to take. If -1 it means no maximum is set and the percent of
      capped epochs plot can not be plotted. Is used in case the percent of capped epochs should be plotted.
    - input_paths (list): The input paths where the aggregated results file is located. Can be passed as None. In this case the paths will be 
      constructed from the given plot folder and project names.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_names (list): If no input paths were given, simultaneously the file names in 
      which the results to be plotted are located.
    - titles (list): The list of titles to be given to the plots.
    - save_project_name (str): The project name under which the plots should be saved.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    # If safe mode, check the inputs
    if safe_mode:
        if not isinstance(plottitle,str):
            raise TypeError("Parameter plottitle needs to be a string!")
        if not (plotkey in ALL_SINGLE_PLOT_KEYNAMES_EPOCHS.keys() or plotkey in ALL_SINGLE_PLOT_KEYNAMES_EVAL.keys() or plotkey in ALL_SINGLE_PLOT_KEYNAMES_OTHER.keys() or plotkey in ALL_SINGLE_PLOT_KEYNAMES_STEPS.keys()):
            raise ValueError(f"Plotkey {plotkey} not allowed!")
        if not isinstance(project_names,list):
            raise TypeError("The project names need to be contained in a list!")
        if not isinstance(save_project_name,str):
            raise TypeError("The project name under which the plots should be saved needs to be a string!")
        if not isinstance(show,bool):
            raise TypeError("Parameter show needs to be boolean!")
        if not isinstance(save,bool):
            raise TypeError("Parameter save needs to be boolean!")
        if isinstance(titles,list):
            for val in titles:
                if not isinstance(val,str):
                    raise TypeError("The titles need to be strings!")
            if input_paths != None:
                if len(titles) != len(input_paths):
                    raise ValueError("The titles must correspond to the amount of input paths!")
            else:
                if len(titles) != len(project_names):
                    raise ValueError("The titles must correspond to the amount of project names!")
        else:
            raise TypeError("Parameter titles needs to be a list!")
        
        
    # Initialize the unified plot
    len_plot = len(titles)
    num_cols = int(np.ceil(len_plot / num_rows))
    remove_index = - (num_rows - (len_plot % num_rows))
    fig, axs = plt.subplots(num_cols,num_rows,figsize=(individual_figsize[0]*num_rows,individual_figsize[1]*num_cols))
    
    # Flatten the axis
    axs = axs.flatten()

    # Get the dedicated function
    if plotkey in ALL_SINGLE_PLOT_KEYNAMES_STEPS.keys():
        goalfunc = ALL_SINGLE_PLOT_KEYNAMES_STEPS[plotkey][0]
    elif plotkey in ALL_SINGLE_PLOT_KEYNAMES_EPOCHS.keys():
        goalfunc = ALL_SINGLE_PLOT_KEYNAMES_EPOCHS[plotkey][0]
    elif plotkey in ALL_SINGLE_PLOT_KEYNAMES_EVAL.keys():
        goalfunc = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][0]
    elif plotkey in ALL_SINGLE_PLOT_KEYNAMES_OTHER.keys():
        goalfunc = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][0]

    # Plot the subplots using the dedicated function
    if input_paths == None:
        for index, proj_name in enumerate(project_names):
            if plotkey == "Mean bias metrics":
                goalfunc(
                    squared = squared,
                    normalized = normalized,
                    best_arms = best_arms,
                    input_path = input_paths,
                    plot_folder = plot_folder,
                    project_name = proj_name,
                    figsize = individual_figsize,
                    grid = grid,
                    mode = ("multiple plots", axs[index],titles[index]),
                    safe_mode = safe_mode
                    )
            elif plotkey == "Mean biases at chosen state action":
                goalfunc(
                    which = which,
                    input_path = input_paths,
                    plot_folder = plot_folder,
                    project_name = proj_name,
                    figsize = individual_figsize,
                    grid = grid,
                    mode = ("multiple plots", axs[index],titles[index]),
                    safe_mode = safe_mode
                    )
            elif plotkey == "Percent of capped epochs":
                goalfunc(
                    max_steps_per_epoch = max_steps_per_epoch,
                    input_path = input_paths,
                    plot_folder = plot_folder,
                    project_name = proj_name,
                    figsize = individual_figsize,
                    grid = grid,
                    mode = ("multiple plots", axs[index],titles[index]),
                    safe_mode = safe_mode
                    )
            else:
                goalfunc(
                    input_path = input_paths,
                    plot_folder = plot_folder,
                    project_name = proj_name,
                    figsize = individual_figsize,
                    grid = grid,
                    mode = ("multiple plots", axs[index],titles[index]),
                    safe_mode = safe_mode
                    )
    else:
        for index, in_path in enumerate(input_paths):
            if plotkey == "Mean bias metrics":
                goalfunc(
                    squared = squared,
                    normalized = normalized,
                    best_arms = best_arms,
                    input_path = in_path,
                    plot_folder = plot_folder,
                    project_name = save_project_name,
                    figsize = individual_figsize,
                    grid = grid,
                    mode = ("multiple plots", axs[index],titles[index]),
                    safe_mode = safe_mode
                    )
            elif plotkey == "Mean biases at chosen state action":
                goalfunc(
                    which = which,
                    input_path = in_path,
                    plot_folder = plot_folder,
                    project_name = save_project_name,
                    figsize = individual_figsize,
                    grid = grid,
                    mode = ("multiple plots", axs[index],titles[index]),
                    safe_mode = safe_mode
                    )
            elif plotkey == "Percent of capped epochs":
                goalfunc(
                    max_steps_per_epoch = max_steps_per_epoch,
                    input_path = in_path,
                    plot_folder = plot_folder,
                    project_name = save_project_name,
                    figsize = individual_figsize,
                    grid = grid,
                    mode = ("multiple plots", axs[index],titles[index]),
                    safe_mode = safe_mode
                    )
            else:
                goalfunc(
                    input_path = in_path,
                    plot_folder = plot_folder,
                    project_name = save_project_name,
                    figsize = individual_figsize,
                    grid = grid,
                    mode = ("multiple plots", axs[index],titles[index]),
                    safe_mode = safe_mode
                    )

    
    # Collect all handles and labels
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Remove duplicate labels, adjust layout
    unique = dict(zip(labels,handles))
    fig.legend(unique.values(), unique.keys(), loc = "upper right")
    fig.suptitle(plottitle,fontsize=16)
    if remove_index != -num_rows:
        for j in range(int(num_cols * num_rows + remove_index),int(num_cols * num_rows)):
            fig.delaxes(axs[j])
    fig.tight_layout()

    # If save is activated, make the save folder and save the plot
    if save:
        os.makedirs(os.path.join(plot_folder,save_project_name + "_plots"),exist_ok=True)
        plt.savefig(os.path.join(os.path.join(plot_folder, save_project_name + "_plots"),save_string + ".png"))

    # If show is activated, show the plot
    if show:
        plt.show()

def selected_plots_multiple_experiments(
        plottitle: str = "Selected experiments",
        plotkeys: list = ["Mean eval results","Mean bias metrics"],
        squared: bool = True,
        normalized: bool = True,
        best_arms: bool = True,
        which: List[Tuple[int,Union[str,int]]] = (0,"best"),
        max_steps_per_epoch: int = -1,
        input_paths: str = None,
        plot_folder: str = "plots",
        project_names: str = ["testproject"],
        titles: str = ["Testproject mean eval results", "Testproject mean bias metrics"],
        save_project_name: str = "testproject",
        individual_figsize: Tuple[Union[int,float]] = (4,4),
        num_rows: int = 4,
        grid: bool = True,
        show: bool = True,
        save: bool = True,
        save_string: str = "testproject_multiple",
        safe_mode: bool = True
) -> None:
    
    """
    Plots given trainin evaluation metric in plotkey. If it is a plot needing conditional data, only the arguments are required, but not confirming that
    the conditionas are fulfilled.The data is given in order by the list of input paths or project names if input path is None. The figsize of the 
    individual plots can be chosen as well as if the figure should be shown and/or saved. If safe_mode is activated a check on the parameters is performed.

    Parameters:
    - plottitle (str): The title given to the plot.
    - plotkeys (list): A list of valid plotkeys to be used.
    - squared (bool): If True, the squared bias metrics will be used in case the biases are plotted.
    - normalized (bool): If True, the normalized bias metrics will be used in case the biases are plotted.
    - best_arms (bool); If True, only the biases at the best arms will be taken into account for the bias metrics in case the biases are plotted.
    - which (tuple): A tuple containing state action pairs whose bias should be plotted in case this type of plot was chosen.
    - max_steps_per_epoch (int): The given maximum amount of steps an epoch is allowed to take. If -1 it means no maximum is set and the percent of
      capped epochs plot can not be plotted. Is used in case the percent of capped epochs should be plotted.
    - input_paths (list): The input paths where the aggregated results file is located. Can be passed as None. In this case the paths will be 
      constructed from the given plot folder and project names.
    - plot_folder (str): The folder to which the plots should be saved. If no input path was given, simultaneously the folder in which the 
      results to be plotted are located.
    - project_names (list): If no input paths were given, simultaneously the file names in 
      which the results to be plotted are located.
    - titles (list): The list of titles to be given to the plots.
    - save_project_name (str): The project name under which the plots should be saved.
    - individual_figsize (tuple): A tuple of integers or float, specifying the width and height of the individual plots in inches.
    - num_rows (int): The number of rows in the plot.
    - grid (bool): If True, the plot will exhibit a grid.
    - show (bool): If True, the plot will be shown.
    - save (bool): If True, the plot will be saved as a .png file.
    - safe_mode (bool): If True, a parameter check will be performed.
    """

    # If safe mode, check the inputs
    if safe_mode:
        if not isinstance(plottitle,str):
            raise TypeError("Parameter plottitle needs to be a string!")
        if isinstance(plotkeys,list):
            for plotkey in plotkeys:
                if not (plotkey in ALL_SINGLE_PLOT_KEYNAMES_EPOCHS.keys() or plotkey in ALL_SINGLE_PLOT_KEYNAMES_EVAL.keys() or plotkey in ALL_SINGLE_PLOT_KEYNAMES_OTHER.keys() or plotkey in ALL_SINGLE_PLOT_KEYNAMES_STEPS.keys()):
                    raise ValueError(f"Plotkey {plotkey} not allowed!")
        else:
            raise TypeError("The plotkeys need to be contained in a list!")
        if not isinstance(project_names,list):
            raise TypeError("The project names need to be contained in a list!")
        if not isinstance(save_project_name,str):
            raise TypeError("The project name under which the plots should be saved needs to be a string!")
        if not isinstance(show,bool):
            raise TypeError("Parameter show needs to be boolean!")
        if not isinstance(save,bool):
            raise TypeError("Parameter save needs to be boolean!")
        if isinstance(titles,list):
            for val in titles:
                if not isinstance(val,str):
                    raise TypeError("The titles need to be strings!")
            if input_paths != None:
                if len(titles) != len(input_paths) * len(plotkeys):
                    raise ValueError("The titles must correspond to the amount of input paths!")
            else:
                if len(titles) != len(project_names) * len(plotkeys):
                    raise ValueError("The titles must correspond to the amount of project names!")
        else:
            raise TypeError("Parameter titles needs to be a list!")
        
        
    # Initialize the unified plot
    len_plot = len(titles)
    num_cols = int(np.ceil(len_plot / num_rows))
    remove_index = - (num_rows - (len_plot % num_rows))
    fig, axs = plt.subplots(num_cols,num_rows,figsize=(individual_figsize[0]*num_rows,individual_figsize[1]*num_cols))
    
    # Flatten the axis
    axs = axs.flatten()

    # For each plotkey iterate
    for i, plotkey in enumerate(plotkeys):

        # Get the dedicated function
        if plotkey in ALL_SINGLE_PLOT_KEYNAMES_STEPS.keys():
            goalfunc = ALL_SINGLE_PLOT_KEYNAMES_STEPS[plotkey][0]
        elif plotkey in ALL_SINGLE_PLOT_KEYNAMES_EPOCHS.keys():
            goalfunc = ALL_SINGLE_PLOT_KEYNAMES_EPOCHS[plotkey][0]
        elif plotkey in ALL_SINGLE_PLOT_KEYNAMES_EVAL.keys():
            goalfunc = ALL_SINGLE_PLOT_KEYNAMES_EVAL[plotkey][0]
        elif plotkey in ALL_SINGLE_PLOT_KEYNAMES_OTHER.keys():
            goalfunc = ALL_SINGLE_PLOT_KEYNAMES_OTHER[plotkey][0]

        # Plot the subplots using the dedicated function
        if input_paths == None:
            for index, proj_name in enumerate(project_names):
                if plotkey == "Mean bias metrics":
                    goalfunc(
                        squared = squared,
                        normalized = normalized,
                        best_arms = best_arms,
                        input_path = input_paths,
                        plot_folder = plot_folder,
                        project_name = proj_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[index + i * len(project_names) - 1],titles[index + i * len(project_names) - 1]),
                        safe_mode = safe_mode
                        )
                elif plotkey == "Mean biases at chosen state action":
                    goalfunc(
                        which = which,
                        input_path = input_paths,
                        plot_folder = plot_folder,
                        project_name = proj_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[index + i * len(project_names) - 1],titles[index + i * len(project_names) - 1]),
                        safe_mode = safe_mode
                        )
                elif plotkey == "Percent of capped epochs":
                    goalfunc(
                        max_steps_per_epoch = max_steps_per_epoch,
                        input_path = input_paths,
                        plot_folder = plot_folder,
                        project_name = proj_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[index + i * len(project_names) - 1],titles[index + i * len(project_names) - 1]),
                        safe_mode = safe_mode
                        )
                else:
                    goalfunc(
                        input_path = input_paths,
                        plot_folder = plot_folder,
                        project_name = proj_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[index + i * len(project_names) - 1],titles[index + i * len(project_names) - 1]),
                        safe_mode = safe_mode
                        )
        else:
            for index, in_path in enumerate(input_paths):
                if plotkey == "Mean bias metrics":
                    goalfunc(
                        squared = squared,
                        normalized = normalized,
                        best_arms = best_arms,
                        input_path = in_path,
                        plot_folder = plot_folder,
                        project_name = save_project_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[index + i * len(input_paths) - 1],titles[index + i * len(input_paths) - 1]),
                        safe_mode = safe_mode
                        )
                elif plotkey == "Mean biases at chosen state action":
                    goalfunc(
                        which = which,
                        input_path = in_path,
                        plot_folder = plot_folder,
                        project_name = save_project_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[index + i * len(input_paths) - 1],titles[index + i * len(input_paths) - 1]),
                        safe_mode = safe_mode
                        )
                elif plotkey == "Percent of capped epochs":
                    goalfunc(
                        max_steps_per_epoch = max_steps_per_epoch,
                        input_path = in_path,
                        plot_folder = plot_folder,
                        project_name = save_project_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[index + i * len(input_paths) - 1],titles[index + i * len(input_paths) - 1]),
                        safe_mode = safe_mode
                        )
                else:
                    goalfunc(
                        input_path = in_path,
                        plot_folder = plot_folder,
                        project_name = save_project_name,
                        figsize = individual_figsize,
                        grid = grid,
                        mode = ("multiple plots", axs[index + i * len(input_paths) - 1],titles[index + i * len(input_paths) - 1]),
                        safe_mode = safe_mode
                        )

    
    # Collect all handles and labels
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Remove duplicate labels, adjust layout
    unique = dict(zip(labels,handles))
    fig.legend(unique.values(), unique.keys(), loc = "upper right")
    fig.suptitle(plottitle,fontsize=16)
    if remove_index != -num_rows:
        for j in range(int(num_cols * num_rows + remove_index),int(num_cols * num_rows)):
            fig.delaxes(axs[j])
    fig.tight_layout()

    # If save is activated, make the save folder and save the plot
    if save:
        os.makedirs(os.path.join(plot_folder,save_project_name + "_plots"),exist_ok=True)
        plt.savefig(os.path.join(os.path.join(plot_folder, save_project_name + "_plots"),save_string + ".png"))

    # If show is activated, show the plot
    if show:
        plt.show()

###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################

# Keys for single plots

ALL_SINGLE_PLOT_KEYNAMES_STEPS = {
    "Num times timesteps reached": (plot_num_times_timesteps_reached, "default_act", "Number of times timesteps were reached"),
    "Mean rewards at steps": (plot_avg_rewards_at_step, "default_act", "Average rewards"),
}

ALL_SINGLE_PLOT_KEYNAMES_EPOCHS = {
    "Num times epochs reached": (plot_num_times_epochs_reached, "default_act", "Number of times epochs were reached"),
    "Mean scores at epochs": (plot_avg_scores_at_epoch, "default_act", "Average Scores"),
    "Mean correct action rates at epochs": (plot_avg_correct_act_rates_at_epoch, "correct_action_log_act", "Average correct action rates"),
    "Mean durations of epochs": (plot_avg_durations_of_epochs, "default_act", "Average durations"),
    "Percent of capped epochs": (plot_percent_of_capped_epochs, "max_steps_per_epoch_act", "Capped epochs") # Needs max_steps_per_epoch as input
}
ALL_SINGLE_PLOT_KEYNAMES_EVAL = {
    "Num times eval times reached": (plot_num_times_evaluation_times_reached, "default_act", "Number of times evaluation times were reached"),
    "Mean scores at evals": (plot_avg_scores_at_eval, "default_act", "Average scores"),
    "Mean correct action rates at evals": (plot_avg_correct_act_rates_at_eval, "correct_action_log_act", "Average correct action rates"),
    "Mean correct action rates at chosen at evals": (plot_avg_correct_act_rates_at_chosen_at_eval,"correct_action_log_act","Average correct action rates of policy at chosen states"),
    "Mean biases at chosen at evals": (plot_avg_biases_one_chosen_state_action_at_eval, "focus_state_actions_act", None), # Needs which as input (tuple of state,action)
    "Mean Q function values at chosen at evals": (plot_avg_q_fct_values_one_chosen_state_action_at_eval,"focus_state_actions_act", None), # Needs which as input (tuple of state,action)
    "Mean bias metrics at evals": (plot_avg_total_biases_one_metric_at_eval, "bias_estimation_act", ""), # needs squared, normalized, best_arms (bool,bool,bool)
    "Mean number of terminal states reached at evals": (plot_avg_num_terminal_states_reached_at_eval, "default_act", "Average number of terminal states reached"),
    "Mean time of reaching terminal states at evals": (plot_avg_time_of_reaching_terminal_states_at_eval, "default_act", "Average time of reaching terminal states"),
}
ALL_SINGLE_PLOT_KEYNAMES_OTHER = {
    "Runtimes": (plot_runtimes, "default_act", "Runtimes"),
    "Mean special logs at steps": (plot_avg_special_logs_one_at_step, "special_act", None), # needs index (int)
    "Mean special logs at epochs": (plot_avg_special_logs_one_at_epoch, "special_act", None), # needs index (int)
    "Mean special logs at evals": (plot_avg_special_logs_one_at_eval, "special_act", None), # needs index (int)
}

# Keys for multiple plot types same family

ALL_MULTIPLE_PLOT_KEYNAMES = {
    "Mean biases at multiple chosen at evals": (plot_avg_biases_multiple_chosen_state_action_at_eval, "focus_state_actions_act", None), # Needs which as input
    "Mean Q function values at multiple chosen at evals": (plot_avg_q_fct_values_multiple_chosen_state_action_at_eval,"focus_state_actions_act", None), # Needs which as input
    "Mean bias metrics multiple at evals": (plot_avg_total_biases_multiple_metrics_at_eval, "bias_estimation_act", None), # needs squared, normalized, best_arms
    "Mean average special logs multiple at steps": (plot_avg_special_logs_multiple_at_step, "special_act", None), # needs index, optionally real_value, real_value_label
    "Mean average special logs multiple at epochs": (plot_avg_special_logs_multiple_at_epoch, "special_act", None), # needs index, optionally real_value, real_value_label
    "Mean average special logs multiple at evals": (plot_avg_special_logs_multiple_at_eval, "special_act", None), # needs index, optionally real_value, real_value_label
}

ALL_BOARD_PLOT_KEYNAMES = {
    "Mean biases at all chosen at evals": (plot_avg_biases_all_chosen_state_action_at_eval, "focus_state_actions_act", None), # Needs which as input
    "Mean Q function values at all chosen at evals": (plot_avg_q_fct_values_all_chosen_state_action_at_eval,"focus_state_actions_act", None), # Needs which as input
    "Mean bias metrics all at evals": (plot_avg_total_biases_all_metrics_at_eval, "bias_estimation_act", None), # needs squared, normalized, best_arms
    "Mean average special logs all at steps": (plot_avg_special_logs_all_at_step, "special_act", None), # needs num_index, optionally real_value, real_value_label
    "Mean average special logs all at epochs": (plot_avg_special_logs_all_at_epoch, "special_act", None), # needs num_index, optionally real_value, real_value_label
    "Mean average special logs all at evals": (plot_avg_special_logs_all_at_eval, "special_act", None), # needs num_index, optionally real_value, real_value_label
}

### TODO: Big Comparison boards plot functions