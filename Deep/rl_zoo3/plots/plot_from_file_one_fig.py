import argparse
import itertools
import pickle
import warnings

import numpy as np
import pandas as pd
import pytablewriter
import seaborn
from matplotlib import pyplot as plt

try:
    from rliable import library as rly  # pytype: disable=import-error
    from rliable import metrics, plot_utils  # pytype: disable=import-error
except ImportError:
    rly = None

from rl_zoo3.plots.score_normalization import normalize_score


def smooth(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode="valid")


def plot_from_file_one_fig():  # noqa: C901
    parser = argparse.ArgumentParser("Gather results, plot them and create table")
    parser.add_argument("-i", "--input", help="Input filename (numpy archive)", type=str)
    parser.add_argument("-skip", "--skip-envs", help="Environments to skip", nargs="+", default=[], type=str)
    parser.add_argument("--keep-envs", help="Envs to keep", nargs="+", default=[], type=str)
    parser.add_argument("--skip-keys", help="Keys to skip", nargs="+", default=[], type=str)
    parser.add_argument("--keep-keys", help="Keys to keep", nargs="+", default=[], type=str)
    parser.add_argument("--no-million", action="store_true", default=False, help="Do not convert x-axis to million")
    parser.add_argument("--skip-timesteps", action="store_true", default=False, help="Do not display learning curves")
    parser.add_argument("-o", "--output", help="Output filename (image)", type=str)
    parser.add_argument("--format", help="Output format", type=str, default="svg")
    parser.add_argument("-loc", "--legend-loc", help="The location of the legend.", type=str, default="best")
    parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[6.4, 4.8])
    parser.add_argument("--fontsize", help="Font size", type=int, default=14)
    parser.add_argument("-l", "--labels", help="Custom labels", type=str, nargs="+")
    parser.add_argument("-b", "--boxplot", help="Enable boxplot", action="store_true", default=False)
    parser.add_argument("-r", "--rliable", help="Enable rliable plots", action="store_true", default=False)
    parser.add_argument("-vs", "--versus", help="Enable probability of improvement plot", action="store_true", default=False)
    parser.add_argument("-iqm", "--iqm", help="Enable IQM sample efficiency plot", action="store_true", default=False)
    parser.add_argument("-ci", "--ci-size", help="Confidence interval size (for rliable)", type=float, default=0.95)
    parser.add_argument("-latex", "--latex", help="Enable latex support", action="store_true", default=False)
    parser.add_argument("--merge", help="Merge with other results files", nargs="+", default=[], type=str)
    parser.add_argument("--merge-type", help="how to merge table, accepts: col or row", type=str, default="col")
    parser.add_argument("-w", "--window-size", help="Rolling window size", type=int, default=4)

    args = parser.parse_args()

    # Activate seaborn
    seaborn.set()
    # Seaborn style
    seaborn.set(style="whitegrid")

    # Enable LaTeX support
    if args.latex:
        plt.rc("text", usetex=True)

    filename = args.input

    if not filename.endswith(".pkl"):
        filename += ".pkl"

    with open(filename, "rb") as file_handler:
        results = pickle.load(file_handler)

    main_results = results["results_table"]

    # Initialize combined headers and value matrix with main results
    combined_headers = main_results["headers"]
    combined_value_matrix = main_results["value_matrix"][1:]

    # Merge additional columns from other files
    for filename in args.merge:
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "rb") as file_handler:
            results_2 = pickle.load(file_handler)
            additional_results = results_2["results_table"]

            if args.merge_type == "col":
                # Add new headers
                for header in additional_results["headers"][1:]:
                    combined_headers.append(header)

                # Add new values for each row
                for i, row in enumerate(additional_results["value_matrix"][1:]):
                    combined_value_matrix[i].extend(row[1:])

                del results_2["results_table"]
                for key in results.keys():
                    if key in results_2:
                        for new_key in results_2[key].keys():
                            results[key][new_key] = results_2[key][new_key]
            elif args.merge_type == "row":
                combined_value_matrix.extend(additional_results["value_matrix"][1:])

                del results_2["results_table"]
                for key, value in results_2.items():
                    if key not in results:
                        results[key] = value

    if args.labels is not None:
        print(f"headers: {combined_headers[1:]}")
        print("replace with")
        print(f"labels: {args.labels}")
        combined_headers[1:] = args.labels

    # Plot combined table
    writer = pytablewriter.LatexTableWriter(max_precision=3)
    writer.table_name = "results_table"
    writer.headers = combined_headers
    writer.value_matrix = combined_value_matrix
    writer.write_table()

    # # Plot table
    # writer = pytablewriter.MarkdownTableWriter(max_precision=3)
    # writer.table_name = "results_table"
    # writer.headers = results["results_table"]["headers"]
    # writer.value_matrix = results["results_table"]["value_matrix"]
    # writer.write_table()

    del results["results_table"]

    keys = [key for key in results[next(iter(results.keys()))].keys() if key not in args.skip_keys]
    print(f"keys: {keys}")
    if len(args.keep_keys) > 0:
        keys = [key for key in keys if key in args.keep_keys]
    envs = [env for env in results.keys() if env not in args.skip_envs]

    if len(args.keep_envs) > 0:
        envs = [env for env in envs if env in args.keep_envs]

    labels = {key: key for key in keys}
    if args.labels is not None:
        for key, label in zip(keys, args.labels):
            labels[key] = label

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    axs = axs.flatten()  # Flatten the 2D array to easily iterate over

    used_axs = [0, 1, 3, 4, 5]
    for i, env in enumerate(envs):
        ax = axs[used_axs[i]]
        title = f"{env}"  # BulletEnv-v0
        if "Mountain" in env:
            title = "MountainCarContinuous-v0"

        ax.set_title(title, fontsize=args.fontsize)

        x_label_suffix = "" if args.no_million else "(1e6)"
        ax.set_xlabel(f"Timesteps {x_label_suffix}", fontsize=args.fontsize)
        ax.set_ylabel("Total Reward", fontsize=args.fontsize)

        for key in keys:
            # x axis in Millions of timesteps
            divider = 1e6
            if args.no_million:
                divider = 1.0

            if args.window_size > 0:
                timesteps = results[env][key]["timesteps"][args.window_size - 1 :]
                mean_ = smooth(results[env][key]["mean"], args.window_size)
                std_error = smooth(results[env][key]["std_error"], args.window_size)
            else:
                timesteps = results[env][key]["timesteps"]
                mean_ = results[env][key]["mean"]
                std_error = results[env][key]["std_error"]

            ax.plot(timesteps / divider, mean_, label=labels[key], linewidth=1.6)
            ax.fill_between(timesteps / divider, mean_ + std_error, mean_ - std_error, alpha=0.4)

        ax.tick_params(axis="both", which="major", labelsize=13)

    # Remove the last subplot if there are only 5 envs
    fig.delaxes(axs[2])
    # Create a single legend for the entire figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=args.fontsize)
    plt.tight_layout()

    if args.output is not None:
        plt.savefig(args.output, format=args.format, dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_from_file_one_fig()
