# system imports
import os
import sys
import random as rand
from time import time
from multiprocessing import Pool, cpu_count, current_process

# external libraries
import yaml
import numpy as np
from tqdm import tqdm
import pandas as pd

# custom libraries
from user import User
from creator import Creator


def read_args():
    if len(sys.argv) >= 2:
        file_name: str = "inputs/" + str(sys.argv[1]) + ".yaml"
    else:
        raise ValueError(
            "No filename provided. Please run as 'python main.py <filename>'"
        )

    with open(file_name, "r") as f:
        data = yaml.safe_load(f)

    outdir = f"./outputs/{round(time())}"
    os.mkdir(outdir)
    data["simulation"]["outdir"] = outdir

    return data["running"], (data["simulation"], data["parameters"])


class Simulator:
    def __init__(self, simulation, parameters):
        self.num_users = simulation["user population size"]
        self.num_creators = simulation["creator population size"]
        self.user_beta = simulation["user selection strength"]
        self.user_mutation_rate = simulation["user mutation probability"]
        self.creator_beta = simulation["user selection strength"]
        self.creator_mutation_rate = simulation["creator mutation probability"]
        self.gens = simulation["generations"]

        self.user_pop, self.creator_pop = self.init_population(parameters)

    def init_population(self, parameters):
        user_population = []
        creator_population = []

        # create population of users
        for i in range(0, self.num_users):
            user_population.append(User(i, parameters))

        # Create population of Devs
        for k in range(0, self.num_creators):
            creator_population.append(Creator(k))

        return user_population, creator_population

    def user_evolution_step(self):
        if rand.random() < self.user_mutation_rate:
            random_user: User = rand.choice(self.user_pop)
            random_user.mutate()
        else:
            user_a: User
            user_b: User
            user_a, user_b = rand.sample(self.user_pop, 2)
            user_a.fitness = 0
            user_b.fitness = 0

            # user A plays Z games
            for _ in range(self.num_creators):
                creator: Creator = rand.choice(self.creator_pop)
                user_a.calculate_payoff(creator)

            # user B plays Z games
            for _ in range(self.num_creators):
                creator: Creator = rand.choice(self.creator_pop)
                user_b.calculate_payoff(creator)

            # learning step
            p_i: float = (
                1 + np.exp(self.user_beta * (user_a.fitness - user_b.fitness))
            ) ** (-1)

            if rand.random() < p_i:
                user_a.strategy = user_b.strategy

    def creator_evolution_step(self):
        if rand.random() < self.creator_mutation_rate:
            random_creator = rand.choice(self.creator_pop)
            random_creator.mutate()
        else:
            # monte carlo step stuff
            creator_a: Creator
            creator_b: Creator
            creator_a, creator_b = rand.sample(self.creator_pop, 2)
            creator_a.fitness = 0
            creator_b.fitness = 0

            # creator A plays X games
            for _ in range(self.num_users):
                user: User = rand.choice(self.user_pop)
                creator_a.calculate_payoff(user)

            # creator B plays X games
            for _ in range(self.num_users):
                user: User = rand.choice(self.user_pop)
                creator_b.calculate_payoff(user)

            # learning step
            p_i: float = (
                1 + np.exp(self.creator_beta * (creator_a.fitness - creator_b.fitness))
            ) ** (-1)

            if rand.random() < p_i:
                creator_a.strategy = creator_b.strategy

    def count_user_strategies(self):
        totals = {0: 0, 1: 0, 2: 0, 3: 0}
        for u in self.user_pop:
            totals[u.strategy] += 1
        return totals

    def count_creator_strategies(self):
        totals = {0: 0, 1: 0}
        for c in self.creator_pop:
            totals[c.strategy] += 1
        return totals

    def write_output(self, filename, n, a, o, p, cc, cd):
        path = f"{filename}.csv"
        with open(path, "a") as file:
            labels = "gen,N,A,O,P,CC,CD"
            labels += "\n"
            file.write(labels)

            for g in range(self.gens):
                output = f"{g},{n[g]},{a[g]},{o[g]},{p[g]},{cc[g]},{cd[g]}"
                output += "\n"
                file.write(output)

    def run(self, filename: str = ""):
        d, c, g, b, cc, cd = [], [], [], [], [], []

        for _ in range(self.gens):
            # 1. Evolve agents
            self.user_evolution_step()
            # 2. Evolve Creators
            self.creator_evolution_step()

            user_strats_dict: dict = self.count_user_strategies()
            creator_strats_dict: dict = self.count_creator_strategies()

            d.append(user_strats_dict[0] / self.num_users)
            c.append(user_strats_dict[1] / self.num_users)
            g.append(user_strats_dict[2] / self.num_users)
            b.append(user_strats_dict[3] / self.num_users)
            cc.append(creator_strats_dict[1] / self.num_creators)
            cd.append(creator_strats_dict[0] / self.num_creators)

        if filename:
            self.write_output(filename, d, c, g, b, cc, cd)


def get_average_output(path, gens, runs):
    # prefix = "./outputs/"
    # filename = path.removeprefix(prefix)

    data = pd.DataFrame()
    for file in os.listdir(path):
        data = pd.concat([data, pd.read_csv(path + file)], ignore_index=True)
    data = data[data.gen != "gen"]

    columns = data.columns[1:]

    for col in columns:
        data[col] = data[col].astype(float)

    average = {col: [] for col in columns}
    standard_dev = {col: [] for col in columns}

    for gen in range(gens):
        df = data.iloc[[i * gens + gen for i in range(runs)]]
        for col in columns:
            average[col].append(df[col].mean())
            standard_dev[col].append(df[col].std())


def run(args):
    simulation, parameters = args
    sim = Simulator(simulation, parameters)
    sim.run(filename=f"{simulation["outdir"]}/{current_process()._identity}")


def run_simulation(run_args, sim_args):
    num_cores = cpu_count() - 1 if run_args["cores"] == "all" else run_args["cores"]

    with Pool(processes=num_cores) as pool:
        list(
            tqdm(pool.imap(run, [sim_args] * run_args["runs"]), total=run_args["runs"])
        )


if __name__ == "__main__":
    run_args, sim_args = read_args()
    run_simulation(run_args, sim_args)
    # get_average_output("./outputs/1746743977/", 10, 10)
