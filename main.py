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

    return data["running"], (data["simulation"], data["parameters"])


def get_average_output(filename, clear_data=True):
    path = f"./outputs/{filename}/"
    print("Processing results...")

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

    with mp.Pool(processes=num_cores) as pool:
        list(
            tqdm(pool.imap(run, [sim_args] * run_args["runs"]), total=run_args["runs"])
        )

    get_average_output(outdir, clear_data)

    return outdir


def run_heatmap(vars: list = ["q", "cI"], v1_start=0.5, v1_end=1.0, v1_steps=3, v2_start=0.0,
                v2_end=0.2, v2_steps=3, clear_data=True):
    print("Drawing heatmaps...")
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

    run_args, sim_args = read_args()

    v1_range = np.linspace(v1_start, v1_end, v1_steps)
    v2_range = np.linspace(v2_start, v2_end, v2_steps)

    # Run simulation for all sets of parameters
    results = []
    for v1 in reversed(v1_range):
        sim_args[1][translator[vars[0]]] = v1
        for v2 in v2_range:
            sim_args[1][translator[vars[1]]] = v2
            results.append(run_simulation(run_args, sim_args))
            sleep(0.1)
            

    path = f"./outputs/"
    new_dir = round(time())
    os.mkdir(f"{path}{new_dir}/")
    for result in results:
        os.rename(f"{path}{result}.csv", f"{path}{new_dir}/{result}.csv")

    print("Saving heatmap figure...")
    # Plot heatmap
    plot_heatmap(
        results,
        new_dir,
        vars,
        v1_range,
        v2_range,
        data_len=sim_args[0]["generations"] * run_args["runs"],
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
    # sim_args[0]: "simulation"
    # sim_args[1]: "parameters"
    # run_args, sim_args = read_args()
    # result = run_simulation(run_args, sim_args)
    # print("All simulations complete, plotting results...")
    # plot_time_series(
    #     result, sim_args, run_args["runs"], maxg=sim_args[0]["generations"]
    # )
    run_heatmap(
        vars=["q", "cI"],
        v1_start=0.5,
        v1_end=1.0,
        v1_steps=51,
        v2_start=0.0,
        v2_end=0.1,
        v2_steps=51,
    )

    # v2_range = np.linspace(0, 0.2, 10)

    # results = [1747486729,1747486731,1747486734,1747486736,1747486738,1747486740,1747486743,1747486745,1747486747,1747486749,1747486752,1747486754,1747486757,1747486760,1747486762,1747486765,1747486767,1747486770,1747486772,1747486774,1747486777,1747486779,1747486782,1747486785,1747486787, 1747486790,1747486792,1747486795,1747486797,1747486800]
    # plot_heatmap(results, 1747486802, ["q", "cI"], [0.5,0.75,1], v2_range, 1000*100, save_fig=True)
