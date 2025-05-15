# system imports
import os
import sys
import multiprocessing as mp
from time import time

# external libraries
import yaml
import pandas as pd
from tqdm import tqdm

# custom libraries
from simulator import Simulator
from plotting import plot_time_series


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
        df = df.astype({
            "gen": int,
            "acr": float,
            "AllD": float,
            "AllC": float,
            "BMedia": float,
            "GMedia": float,
            "Unsafe": float,
            "Safe": float,
        })

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
    #print(sim.__str__())
    sim.run(
        filename="./outputs/" + simulation["outdir"] + "/" + str(mp.current_process()._identity)
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


if __name__ == "__main__":
    # run_args: runs + cores
    # sim_args[0]: "simulation"
    # sim_args[1]: "parameters"
    run_args, sim_args = read_args()
    result = run_simulation(run_args, sim_args)
    print("All simulations complete, plotting results...")
    plot_time_series(result, sim_args, run_args["runs"], maxg=sim_args[0]["generations"])
