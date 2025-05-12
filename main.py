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
        file_name: str = "inputs/" + str(sys.argv[1]) + ".yaml"
    else:
        raise ValueError("Invalid inputs. Please run as 'python main.py <filename>'")

    with open(file_name, "r") as f:
        data = yaml.safe_load(f)

    return data["running"], (data["simulation"], data["parameters"])


def get_average_output(filename, clear_data=True):
    path = f"./outputs/{filename}/"

    data = pd.DataFrame()
    for file in os.listdir(path):
        data = pd.concat([data, pd.read_csv(path + file)], ignore_index=True)
    data = data[data.gen != "gen"]

    for col in data.columns[1:]:
        data[col] = data[col].astype(float)

    data.to_csv(path[:-1] + ".csv")

    if clear_data:
        for file in os.listdir(path):
            os.remove(path + file)
        os.rmdir(path)


def run(args):
    simulation, parameters = args
    sim = Simulator(simulation, parameters)
    sim.run(
        filename=f"./outputs/{simulation["outdir"]}/{mp.current_process()._identity}"
    )


def run_simulation(run_args, sim_args, clear_data=True):
    outdir = f"{round(time())}"
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
    run_args, sim_args = read_args()
    result = run_simulation(run_args, sim_args)
    plot_time_series(result, sim_args[0]["generations"], run_args["runs"], maxg=1000)
