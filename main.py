# system imports
import os
import sys
import multiprocessing as mp
from time import time, sleep

# external libraries
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import pprint
# custom libraries
from simulator import Simulator
from plotting import plot_time_series, plot_heatmap


def read_args():
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
        raise ValueError("This type of simulation currently doesn't exist. Please choose 'time series' or 'heatmap'.")


def get_average_output(filename, clear_data=True):
    path = f"./outputs/{filename}/"

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


def run(args):
    simulation, parameters = args
    sim = Simulator(simulation, parameters)
    # print(sim.__str__())
    sim.run(
        filename="./outputs/"
        + simulation["outdir"]
        + "/"
        + str(mp.current_process()._identity)
    )


def run_simulation(run_args, sim_args, clear_data=True):
    outdir: str = f"{round(time())}"
    os.mkdir(f"./outputs/{outdir}")
    sim_args[0]["outdir"] = outdir

    num_cores = mp.cpu_count() - 1 if run_args["cores"] == "all" else run_args["cores"]

    print("============ Running experiment of", run_args["runs"],
          "simulations in", num_cores, "cores: ============")
    if sim_args[0]["type"] == "time_series":
        pprint.pp(sim_args)

    print("Pooling processes...")
    with mp.Pool(processes=num_cores) as pool:
        list(
            tqdm(pool.imap(run, [sim_args] * run_args["runs"]), total=run_args["runs"])
        )

    print("Simulations done. Processing results...")
    get_average_output(outdir, clear_data)

    return outdir


def run_heatmap(vars: list = ("q", "cI"), v1_start=0.5, v1_end=1.0, v1_steps=3, v2_start=0.0,
                v2_end=0.2, v2_steps=3, clear_data=True):
    translator = {
        "q": "media quality",
        "bU": "user benefit",
        "cU": "user cost",
        "cI": "cost investigation",
        "bP": "creator benefit",
        "cP": "creator cost",
    }
    available_vars = ["q", "bU", "cU", "cI", "bP", "cP"]
    if len(vars) != 2 or vars[0] not in available_vars or vars[1] not in available_vars:
        raise ValueError("Parameter <vars> must be a list of 2 known variables.")

    run_args, sim_args, payoffs, _ = read_args()

    v1_range = np.linspace(v1_start, v1_end, v1_steps)
    v2_range = np.linspace(v2_start, v2_end, v2_steps)

    # Run simulation for all sets of parameters
    n_sims = len(v1_range) * len(v2_range)
    results = []
    for i, v1 in enumerate(v1_range):
        payoffs[translator[vars[0]]] = v1
        for j, v2 in enumerate(v2_range):
            print(f"============ Running experiment {i*len(v2_range)+j+1} of {n_sims} ============")
            payoffs[translator[vars[1]]] = v2
            results.append(run_simulation(run_args, (sim_args, payoffs)))
            sleep(0.05)

    path = f"./outputs/"
    new_dir = str(time.time_ns())
    os.mkdir(f"{path}{new_dir}/", exist_ok=True)
    for result in results:
        os.rename(f"{path}{result}.csv", f"{path}{new_dir}/{result}.csv")

    # Plot heatmap
    plot_heatmap(
        results,
        new_dir,
        vars,
        v1_range,
        v2_range,
        data_len=sim_args["generations"] * run_args["runs"],
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
    # run_args: runs + cores
    run_args, sim_args, payoffs, heatmap_args = read_args()

    if sim_args["type"] == "time_series":
        # sim_args[0]: "simulation"
        # sim_args[1]: "parameters"
        result = run_simulation(run_args, (sim_args, payoffs))
        plot_time_series(result, (sim_args, payoffs), run_args["runs"], maxg=sim_args["generations"], save_fig=True)
    elif sim_args["type"] == "heatmap":
        run_heatmap(
            vars=heatmap_args["vars"],
            v1_start=heatmap_args["v1_start"],
            v1_end=heatmap_args["v1_end"],
            v1_steps=heatmap_args["v1_steps"],
            v2_start=heatmap_args["v2_start"],
            v2_end=heatmap_args["v2_end"],
            v2_steps=heatmap_args["v2_steps"]
        )
    else:
        raise ValueError("__main__: Oops, that type doesn't exist yet.")
