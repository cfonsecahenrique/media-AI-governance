# system imports
import os
import sys
import multiprocessing as mp
from time import time, sleep, time_ns

# external libraries
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import pprint

# custom libraries
from simulator import Simulator
from plotting import plot_time_series, plot_heatmap


def read_args() -> tuple:
    """Read and retrieve arguments from input file.

    Raises:
        ValueError: Invalid input (no file provided on argv).
        ValueError: Invalid type of simulation. Only valid are "time_series" and "heatmap".

    Returns:
        tuple: running, simulation, parameters (and heatmap) arguments.
    """

    if len(sys.argv) == 2:
        file_name: str = "inputs/" + str(sys.argv[1])
    else:
        raise ValueError("Invalid inputs. Please run as 'python main.py <filename>'")

    with open(file_name, "r") as f:
        data = yaml.safe_load(f)

    if data["simulation"]["type"] == "time_series":
        return data["running"], data["simulation"], data["parameters"], None
    elif data["simulation"]["type"] == "heatmap":
        return data["running"], data["simulation"], data["parameters"], data["heatmap"]
    else:
        raise ValueError(
            "This type of simulation currently doesn't exist. Please choose 'time series' or 'heatmap'."
        )


def concat_output(output_dir: str, clear_data: bool = True) -> None:
    """Concatenate all output files from output dir in a single fine.

    Args:
        output_dir (str): output directory where all output files are stored.
        clear_data (bool, optional): delete all files after concatenate. Defaults to True.
    """

    path = f"./outputs/{output_dir}/"

    all_dataframes = []

    for file in os.listdir(path):
        if not file.endswith(".csv"):
            continue

        full_path = os.path.join(path, file)

        # Read with headers, all as strings first
        df = pd.read_csv(full_path, dtype=str)

        # Drop rows where 'gen' equals 'gen' (i.e., extra headers accidentally included)
        df = df[df["gen"] != "gen"]

        # Convert columns to correct types
        df = df.astype(
            {
                "gen": int,
                "acr": float,
                "acr_u": float,
                "acr_c": float,
                "AllD": float,
                "AllC": float,
                "BMedia": float,
                "GMedia": float,
                "Unsafe": float,
                "Safe": float,
            }
        )

        all_dataframes.append(df)

    # Combine all clean DataFrames
    combined = pd.concat(all_dataframes, ignore_index=True)

    # Export to CSV
    output_path = path[:-1] + ".csv"
    combined.to_csv(output_path, index=False)

    # Clean up files
    if clear_data:
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        os.rmdir(path)
    print(f"Saved cleaned data to {output_path}")


def run_simulation(args: tuple):
    """Run a simulation for given set of arguments.

    Args:
        args (tuple): tuple with arguments for simulation and parameters.
    """

    simulation, parameters = args
    sim = Simulator(simulation, parameters)
    sim.run(
        filename="./outputs/"
        + simulation["outdir"]
        + "/"
        + str(mp.current_process()._identity)
    )


def run_simulations(run_args: dict, sim_args: tuple, clear_data=True) -> str:
    """Run multiple simulations in parallel for given set of arguments.

    Args:
        run_args (dict): arguments for running conditions.
        sim_args (tuple): arguments for simulation and game parameters.
        clear_data (bool, optional): delete all files after concatenate. Defaults to True.

    Returns:
        str: file name where all data will be stored.
    """

    outdir: str = f"{round(time())}"
    os.mkdir(f"./outputs/{outdir}")
    sim_args[0]["outdir"] = outdir

    num_cores = mp.cpu_count() - 1 if run_args["cores"] == "all" else run_args["cores"]

    print(
        "============ Running experiment of",
        run_args["runs"],
        "simulations in",
        num_cores,
        "cores: ============",
    )
    if sim_args[0]["type"] == "time_series":
        pprint.pp(sim_args)

    print("Pooling processes...")
    with mp.Pool(processes=num_cores) as pool:
        list(
            tqdm(pool.imap(run, [sim_args] * run_args["runs"]), total=run_args["runs"])
        )

    print("Simulations done. Processing results...")

    concat_output(outdir, clear_data)

    return outdir


def run_heatmap(
    vars: list = ("q", "cI"),
    v1_start: float = 0.5,
    v1_end: float = 1.0,
    v1_steps: int = 3,
    v1_scale: str = "lin",
    v2_start: float = 0.0,
    v2_end: float = 0.2,
    v2_steps: int = 3,
    v2_scale: str = "lin",
    clear_data=True,
) -> tuple:
    """Run simulations for each point of the heatmap.

    Args:
        vars (list, optional): heatmap (x, y) variables. Defaults to ("q", "cI").
        v1_start (float, optional): start point for variable x. Defaults to 0.5.
        v1_end (float, optional): final point for variable x. Defaults to 1.0.
        v1_steps (int, optional): incrementing step for variable x. Defaults to 3.
        v1_scale (str, optional): scale for variable x. Defaults to "lin".
        v2_start (float, optional): start point for variable y. Defaults to 0.0.
        v2_end (float, optional): final point for variable y. Defaults to 0.2.
        v2_steps (int, optional): incrementing step for variable y. Defaults to 3.
        v2_scale (str, optional): scale for variable y. Defaults to "lin".
        clear_data (bool, optional): clear all files after concatenating. Defaults to True.

    Raises:
        ValueError: vars must include known variables.
        ValueError: var x scale can only be 'lin' of 'log'.
        ValueError: var y scale can only be 'lin' of 'log'.

    Returns:
        tuple: tuple of results (the results from the heatmap) and new_dir (the directory where data ios stored).
    """
    
    translator = {
        "q": "media quality",
        "bU": "user benefit",
        "cU": "user cost",
        "cI": "cost investigation",
        "bP": "creator benefit",
        "cP": "creator cost",
        "um": "user mutation probability",
        "cm": "creator mutation probability",
    }
    available_vars = ["q", "bU", "cU", "cI", "bP", "cP", "um", "cm"]
    if len(vars) != 2 or vars[0] not in available_vars or vars[1] not in available_vars:
        raise ValueError("Parameter <vars> must be a list of 2 known variables.")

    run_args, sim_args, payoffs, _ = read_args()

    if v1_scale == "lin":
        v1_range = np.linspace(v1_start, v1_end, v1_steps)
    elif v1_scale == "log":
        v1_range = np.logspace(v1_start, v1_end, v1_steps)
    else:
        raise ValueError("Var 1 scale can only be 'lin' of 'log'.")

    if v2_scale == "lin":
        v2_range = np.linspace(v2_start, v2_end, v2_steps)
    elif v2_scale == "log":
        v2_range = np.logspace(v2_start, v2_end, v2_steps)
    else:
        raise ValueError("Var 2 scale can only be 'lin' of 'log'.")

    # Run simulation for all sets of parameters
    n_sims = len(v1_range) * len(v2_range)
    results = []
    for i, v1 in enumerate(v1_range):
        if vars[0] in ("um", "cm"):
            sim_args[translator[vars[0]]] = v1
        else:
            payoffs[translator[vars[0]]] = v1
        for j, v2 in enumerate(v2_range):
            print(
                f"============ Running experiment {i*len(v2_range)+j+1} of {n_sims} ============"
            )
            if vars[1] in ("um", "cm"):
                sim_args[translator[vars[1]]] = v2
            else:
                payoffs[translator[vars[1]]] = v2
            results.append(run_simulation(run_args, (sim_args, payoffs)))
            sleep(0.05)

    path = f"./outputs/"
    new_dir = str(time_ns())
    os.makedirs(f"{path}{new_dir}/", exist_ok=True)
    for result in results:
        os.rename(f"{path}{result}.csv", f"{path}{new_dir}/{result}.csv")

    # Plot heatmap
    plot_heatmap(
        results,
        new_dir,
        vars,
        v1_range,
        v1_scale,
        v2_range,
        v2_scale,
        save_fig=True,
    )

    # Clean up files
    path = f"{path}{new_dir}/"
    if clear_data:
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        os.rmdir(path)

    return results, new_dir


if __name__ == "__main__":
    # get arguments from input file
    run_args, sim_args, payoffs, heatmap_args = read_args()

    if sim_args["type"] == "time_series":
        result = run_simulation(run_args, (sim_args, payoffs))
        plot_time_series(
            result,
            (sim_args, payoffs),
            run_args["runs"],
            maxg=sim_args["generations"],
            save_fig=True,
        )
    elif sim_args["type"] == "heatmap":
        run_heatmap(
            vars=heatmap_args["vars"],
            v1_start=heatmap_args["v1_start"],
            v1_end=heatmap_args["v1_end"],
            v1_steps=heatmap_args["v1_steps"],
            v1_scale=heatmap_args["v1_scale"],
            v2_start=heatmap_args["v2_start"],
            v2_end=heatmap_args["v2_end"],
            v2_steps=heatmap_args["v2_steps"],
            v2_scale=heatmap_args["v2_scale"],
        )
    else:
        raise ValueError("__main__: Oops, that type doesn't exist yet.")
