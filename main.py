import os
import time
import yaml
import sys
import numpy as np
import random as rand
from tqdm import tqdm
import multiprocessing
# import plotext as plt
import matplotlib.pyplot as plt
import pandas as pd

from User import User
from Commentator import Commentator
from Creator import Creator

COOPERATE = 1
DEFECT = 0

NUMBER_USERS: int
NUMBER_COMMENTATORS: int
NUMBER_CREATORS: int
MEDIA_QUALITY: list
# Vector of public reputations
MEDIA_QUALITY_EXPECTED: list
delta_q: float = 0.0001
GENS: int
RUNS: int
USER_MUTATION_PROBABILITY: float
CREATOR_MUTATION_PROBABILITY: float
U_SELECTION_STRENGTH = 1.
C_SELECTION_STRENGTH = 1.
# Ground truth of creator reputations
REAL_CREATOR_STRATEGIES: list = []

user_population: list[User] = []
media_population: list[Commentator] = []
creator_population: list[Creator] = []

# for plotting
generations = []  # Store time steps
never_adopt = []  # Users who never adopt
always_adopt = []  # Users who always adopt
optimist = []  # Optimistic users
pessimist = []  # Pessimistic users
creator_cooperator = []
creator_defector = []
media_reputation = {}

# Benefit a user receives when adopting a safe technology
bU = 0.4
# Cost for the user adopting unsafe technology
cU = 0.8
# Benefit the media gets by users paying the (same) cost to access it
bM = 0.01
# Benefit for the creator when user uses
bP = 0.4
# Cost paid by creators to create safe AI
cP = 0.2


def read_args():
    if sys.argv[1]:
        file_name: str = "inputs/" + str(sys.argv[1]) + ".yaml" 
    else:
        raise ValueError("No filename provided. Please run as 'python main.py <filename>'")

    global NUMBER_USERS
    global NUMBER_COMMENTATORS
    global NUMBER_CREATORS
    global USER_MUTATION_PROBABILITY
    global CREATOR_MUTATION_PROBABILITY
    global MEDIA_QUALITY
    global MEDIA_QUALITY_EXPECTED
    global GENS
    global RUNS

    # Open and parse the YAML file
    with open(file_name, "r") as f:
        data = yaml.safe_load(f)

    for entry in data.get("instructions", []):
        NUMBER_USERS = int(entry["user population size"])
        NUMBER_COMMENTATORS = int(entry["commentator population size"])
        NUMBER_CREATORS = int(entry["creators population size"])
        MEDIA_QUALITY = list(entry["media quality"])
        MEDIA_QUALITY_EXPECTED = np.random.uniform(low=0.5, high=1.0, size=(NUMBER_COMMENTATORS,))
        USER_MUTATION_PROBABILITY = float(
            entry["user mutation probability"]
        )  # /NUMBER_USERS)
        CREATOR_MUTATION_PROBABILITY = float(
            entry["creator mutation probability"]
        )  # /NUMBER_CREATORS)
        GENS = int(entry["generations"])
        RUNS = int(entry["runs"])


def initialization():
    # Create population of users
    global user_population
    global media_population
    global REAL_CREATOR_STRATEGIES
    global creator_population

    user_population = []
    media_population = []
    creator_population = []

    for i in range(0, NUMBER_USERS):
        user_population.append(User(i, NUMBER_COMMENTATORS))

    # Create population of commentators
    for j in range(0, NUMBER_COMMENTATORS):
        media_population.append(Commentator(j, MEDIA_QUALITY[j]))

    # Create population of Devs
    for k in range(0, NUMBER_CREATORS):
        creator_population.append(Creator(k))

    for user in user_population:
        # initialy useless
        user.media_trust_vector = []


def update_reputation_all(media_trust_vector: list, up_or_down: int):
    # up = 1, down = -1
    # up if media suggestion was right, down if it was wrong
    global MEDIA_QUALITY_EXPECTED
    for media in media_trust_vector:
        MEDIA_QUALITY_EXPECTED[media.id] += up_or_down * delta_q
        # clamp it to [0,1]
        MEDIA_QUALITY_EXPECTED[media.id] = max(0, min(1, MEDIA_QUALITY_EXPECTED[media.id]))


def update_reputation_discriminate(media_trust_vector: list, up_or_down: list):
    # here up or down is a list of -1 or 1s
    # up if media suggestion was right, down if it was wrong
    global MEDIA_QUALITY_EXPECTED

    for i, media in enumerate(media_trust_vector):
        MEDIA_QUALITY_EXPECTED[media.id] += up_or_down[i] * delta_q
        # clamp it to [0,1]
        MEDIA_QUALITY_EXPECTED[media.id] = max(0, min(1, MEDIA_QUALITY_EXPECTED[media.id]))


def generate_media_beliefs(u: User, c: Creator):
    media_beliefs_of_creator = []
    if u.tm != 0:
        for media in u.media_trust_vector:
            if rand.random() <= media.quality:
                media_beliefs_of_creator.append(c.strategy)
            else:
                media_beliefs_of_creator.append(1-c.strategy)
    return media_beliefs_of_creator


def user_evolution_step():
    if rand.random() < USER_MUTATION_PROBABILITY:
        random_user: User = rand.choice(user_population)
        random_user.mutate(NUMBER_COMMENTATORS)
    else:
        # Monte carlo step stuff
        user_a: User
        user_b: User
        user_a, user_b = rand.sample(user_population, 2)
        user_a.fitness = 0
        user_b.fitness = 0

        # build trust media vector stochastically
        user_a.media_trust_vector = rand.choices(population=media_population, weights=MEDIA_QUALITY_EXPECTED, k=user_a.tm)
        user_b.media_trust_vector = rand.choices(population=media_population, weights=MEDIA_QUALITY_EXPECTED, k=user_b.tm)

        # user A plays Z games
        for _ in range(NUMBER_CREATORS):
            creator: Creator = rand.choice(creator_population)
            calculate_payoff_users(user_a, creator)        
            
        # user B plays Z games
        for _ in range(NUMBER_CREATORS):
            creator: Creator = rand.choice(creator_population)
            calculate_payoff_users(user_b, creator)

        # learning step
        # Calculate Probability of imitation
        p_i: float = (
            1 + np.exp(U_SELECTION_STRENGTH * (user_a.fitness - user_b.fitness))
        ) ** (-1)

        if rand.random() < p_i:
            user_a.strat = user_b.strat
            user_a.tm = user_b.tm


def creator_evolution_step():
    if rand.random() < CREATOR_MUTATION_PROBABILITY:
        random_creator = rand.choice(creator_population)
        random_creator.mutate()
    else:
        # Monte carlo step stuff
        creator_a: Creator
        creator_b: Creator
        creator_a, creator_b = rand.sample(creator_population, 2)
        creator_a.fitness = 0
        creator_b.fitness = 0

        # Creator A plays X games
        for _ in range(NUMBER_USERS):
            user: User = rand.choice(user_population)
            calculate_payoff_creators(user, creator_a)

        # Creator B
        for _ in range(NUMBER_USERS):
            user: User = rand.choice(user_population)
            calculate_payoff_creators(user, creator_b)

        # learning step
        # Calculate Probability of imitation
        p_i: float = (
            1
            + np.exp(C_SELECTION_STRENGTH * (creator_a.fitness - creator_b.fitness))
        ) ** (-1)
        if rand.random() < p_i:
            creator_a.strategy = creator_b.strategy


def theta_function(sum_media_beliefs_of_creator, threshold: int):
    return 1 if sum_media_beliefs_of_creator >= threshold else 0


def payoff_matrix(user: User, sum_media_beliefs_of_creator: int):
    theta = -1
    # x = recommended action = 0 or 1
    # tM = number of trusted sources
    if user.strat == 0:
        # Never Adopt
        pass
    elif user.strat == 1:
        # Always Adopt
        pass
    elif user.strat == 2:
        # Optimist
        theta = theta_function(sum_media_beliefs_of_creator, threshold=1)
    elif user.strat == 3:
        # Pessimist
        theta = theta_function(sum_media_beliefs_of_creator, threshold=user.tm)
    else:
        raise ValueError("User type error")

    user_payoffs = np.array(
        [
            [0, -cU, theta * (-cU) - (user.tm * bM), theta * (-cU) - (user.tm * bM)],
            [0, bU, theta * bU - user.tm * bM, theta * bU - user.tm * bM],
        ]
    )
    creator_payoffs = np.array(
        [
            [0, bP, theta * bP, theta * bP],
            [-cP, bP - cP, theta * bP - cP, theta * bP - cP],
        ]
    )
    return user_payoffs, creator_payoffs


def calculate_payoff_users(u: User, c: Creator):
    # Create a list of opinions of only trusted sources
   
    # media_beliefs_of_creators
    media_beliefs_of_creator = generate_media_beliefs(u, c)

    # compare media beliefs with creators strategies
    up_or_down = np.ones(len(media_beliefs_of_creator))
    for i in range(len(up_or_down)):
        if media_beliefs_of_creator[i] != c.strategy:
            up_or_down[i] = -1

    # Payoffs are (kinda) different depending on u.strat being 2 or 3 or more
    user_payoffs, creator_payoffs = payoff_matrix(u, sum(media_beliefs_of_creator))
    u.fitness += user_payoffs[c.strategy, u.strat]
    c.fitness += creator_payoffs[c.strategy, u.strat]

    # update reputation
    update_reputation_discriminate(u.media_trust_vector, up_or_down)


def calculate_payoff_creators(u: User, c: Creator):
    # Create a list of opinions of only trusted sources
   
    # generate media beliefs of creators
    media_beliefs_of_creator = generate_media_beliefs(u, c)

    # Payoffs are (kinda) different depending on u.strat being 2 or 3 or more
    user_payoffs, creator_payoffs = payoff_matrix(u, sum(media_beliefs_of_creator))
    u.fitness += user_payoffs[c.strategy, u.strat]
    c.fitness += creator_payoffs[c.strategy, u.strat]


def count_user_strategies():
    totals = {0: 0, 1: 0, 2: 0, 3: 0}
    for u in user_population:
        totals[u.strat] += 1
    return totals


def count_creator_strategies():
    totals = {DEFECT: 0, COOPERATE: 0}
    for c in creator_population:
        totals[c.strategy] += 1
    return totals


def export_results(users_strats_counts: dict, creators_strats_counts: dict, plotting: bool = False):
    print("USERS:", users_strats_counts)
    print("CREATORS:", creators_strats_counts)

    # Create a unique filename. Change it later to experiment name/id
    file_name: str = "outputs/" + str(round(time.time())) + ".csv"
    f = open(file_name, "a")
    # Write the time series of all relevant frequencies

    labels = "gen,N,A,O,P,CC,CD"
    for media in media_reputation:
        labels += f",M{media}"
    labels += "\n"
    f.write(labels)
    for g in range(GENS):
        output: str = (
            str(generations[g])
            + ","
            + str(never_adopt[g])
            + ","
            + str(always_adopt[g])
            + ","
            + str(optimist[g])
            + ","
            + str(pessimist[g])
            + ","
            + str(creator_cooperator[g])
            + ","
            + str(creator_defector[g])
        )
        for media, value in media_reputation.items():
            output += f",{value[g]}"
        output += "\n"
        f.write(output)
    f.close()

    if plotting:
        df = pd.read_csv(file_name).drop("gen", axis=1)

        fig, (ax1,ax2,ax3) = plt.subplots(3)

        # color=['r','b','orange','g','purple','brown']
        ls=['-','-','-', '-','-','-'] + ["dotted" for _ in media_reputation]
        labels=['N','A','O','P','CC','CD'] + [f"M{i}" for i in media_reputation]
        for i, col in enumerate(['N','A','O','P']):
            df[col].plot(ls=ls[i], label=labels[i], ax=ax1)
        for i, col in enumerate(['CC','CD']):
            df[col].plot(ls=ls[i+4], label=labels[i+4], ax=ax2)
        for i, col in enumerate([f"M{i}" for i in media_reputation]):
            df[col].plot(ls=ls[i+6], label=labels[i+6], ax=ax3)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')
        ax3.legend(loc='upper left')
        plt.show()


def run_one_generation(logging: bool = False):
    initialization()

    g, n, a, o, p, cc, cd = [], [], [], [], [], [], [] 
    r = { i: [] for i in range(len(MEDIA_QUALITY_EXPECTED)) }

    for generation in tqdm(range(GENS)):
        # 1. Evolve agents
        user_evolution_step()
        # 2. Evolve Creators
        creator_evolution_step()

        user_strats_dict: dict = count_user_strategies()
        creator_strats_dict: dict = count_creator_strategies()
        # Store data for plotting
        g.append(generation)
        n.append(user_strats_dict[0] / NUMBER_USERS)
        a.append(user_strats_dict[1] / NUMBER_USERS)
        o.append(user_strats_dict[2] / NUMBER_USERS)
        p.append(user_strats_dict[3] / NUMBER_USERS)
        cc.append(creator_strats_dict[COOPERATE] / NUMBER_CREATORS)
        cd.append(creator_strats_dict[DEFECT] / NUMBER_CREATORS)
        for media in range(NUMBER_COMMENTATORS):
            r[media].append(MEDIA_QUALITY_EXPECTED[media])

    return g, n, a, o, p, cc, cd, r


def run(logging: bool = True, plotting: bool = False, output: bool=False):
    global REAL_CREATOR_STRATEGIES
    global generations
    global never_adopt 
    global always_adopt 
    global optimist
    global pessimist 
    global creator_cooperator 
    global creator_defector
    global media_reputation

    # Have a fixed initial configuration of trustworthiness of commentators
    read_args()

    g_tmp = np.zeros(GENS)
    n_tmp = np.zeros(GENS)
    a_tmp = np.zeros(GENS)
    o_tmp = np.zeros(GENS)
    p_tmp = np.zeros(GENS)
    cc_tmp = np.zeros(GENS)
    cd_tmp = np.zeros(GENS)
    r_tmp = {i: np.zeros(GENS) for i in range(NUMBER_COMMENTATORS)}

    for run in range(1, RUNS+1):
        print(
            "Running simulation: " + "|" + run * "█" + (RUNS - run) * " " + f"|{run}/{RUNS}|"
        )

        g, n, a, o, p, cc, cd, r = run_one_generation(logging)

        g_tmp = np.array(g)
        n_tmp += np.array(n)
        a_tmp += np.array(a)
        o_tmp += np.array(o)
        p_tmp += np.array(p)
        cc_tmp += np.array(cc)
        cd_tmp += np.array(cd)
      
        for i, v in r.items():
            r_tmp[i] += v
    
    generations = g_tmp
    never_adopt = n_tmp/RUNS
    always_adopt = a_tmp/RUNS
    optimist = o_tmp/RUNS
    pessimist = p_tmp/RUNS
    creator_cooperator = cc_tmp/RUNS
    creator_defector = cd_tmp/RUNS

    for i, v in r_tmp.items():
        media_reputation[i] = v/RUNS
    
    if output:
        export_results(count_user_strategies(), count_creator_strategies(), plotting)

    # calculate average cooperation rate for creators
    nc = NUMBER_CREATORS
    count, _ = np.histogram(creator_cooperator, bins=nc+1)
    cc_stat_dist = count/GENS
    avg_cooperation_creator = sum([k/nc * cc_stat_dist[k] for k in range(nc+1)])

    return avg_cooperation_creator


def run_cp_bm(n_bits: int = 6, plotting: bool = False, output: bool=False):
    global cP
    global bM

    cps = [round(i / (n_bits-1) - 0.5, 1) for i in range(n_bits)]
    bms = [i * 0.01 for i in range(n_bits)]

    # u_heatmap = np.zeros((5,5))
    c_heatmap = np.zeros((n_bits, n_bits))

    for i, cp in enumerate(reversed(cps)):
        cP = cp
        for j, bm in enumerate(bms):
            bM = bm
            c_heatmap[i, j] = run(plotting, output)
    
    plt.title("Average Creators' Cooperation Rate")
    plt.imshow(c_heatmap, cmap='RdYlGn')
    plt.xticks(ticks=[i for i in range(n_bits)], labels=bms)
    plt.yticks(ticks=[i for i in range(n_bits)], labels=reversed(cps))
    plt.xlabel("bM")
    plt.ylabel("cP")
    plt.colorbar()
    plt.show()

    return c_heatmap   


def multiprocess():
    manager = multiprocessing.Manager()
    lock = manager.Lock()


if __name__ == "__main__":
    run_cp_bm(n_bits=11, plotting=False, output=False)
