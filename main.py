import os
import time
import yaml
import sys
import numpy as np
import random as rand
from tqdm import tqdm
import plotext as plt
import pprint
import matplotlib.pyplot as plts
import pandas as pd

from User import User
from Commentator import Commentator
from Creator import Creator

COOPERATE = 1
DEFECT = 0
NUMBER_USERS: int
NUMBER_COMMENTATORS: int
NUMBER_CREATORS: int
# MEDIA_TRUST_VECTOR: list
MEDIA_QUALITY: list
MEDIA_QUALITY_EXPECTED: list
delta_q: float
GENS: int
RUNS: int
USER_MUTATION_PROBABILITY: float
CREATOR_MUTATION_PROBABILITY: float
U_SELECTION_STRENGTH = 1
C_SELECTION_STRENGTH = 0.5
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
trust_media = []

media_image_matrix = np.zeros((0, 0))

# Benefit a user receives when adopting a safe technology
bU = 0.4
# Cost for the user adopting unsafe technology
cU = 0.8
# Benefit the media gets by users paying the (same) cost to access it
bM = 0.05
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
    global MEDIA_MUTATION_PROBABILITY
    global MEDIA_TRUST_VECTOR
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
        MEDIA_QUALITY_EXPECTED = np.random.rand(NUMBER_COMMENTATORS)
        USER_MUTATION_PROBABILITY = float(
            entry["user mutation probability"]
        )  # /NUMBER_USERS)
        CREATOR_MUTATION_PROBABILITY = float(
            entry["creator mutation probability"]
        )  # /NUMBER_CREATORS)
        MEDIA_MUTATION_PROBABILITY = float(entry["media trust mutation"])
        MEDIA_TRUST_VECTOR = list(entry["media trust vector"])
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
        user.media_trust_vector = np.zeros(NUMBER_COMMENTATORS)



# def generate_media_beliefs():
#     global media_image_matrix
#     media_image_matrix = np.zeros((NUMBER_COMMENTATORS, NUMBER_CREATORS))
#     for comm in media_population:
#         for creator in creator_population:
#             if rand.random() < comm.quality:
#                 media_image_matrix[comm.id, creator.id] = creator.strategy
#             else:
#                 media_image_matrix[comm.id, creator.id] = rand.choice(
#                     (DEFECT, COOPERATE)
#                 )
#     return media_image_matrix

def update_reputation_all(media_trust_vector: list):
    # delta_q
    pass

def update_reputation_single():
    pass

def update_reputation_discriminate():
    pass

def generate_media_beliefs():
    # stochastically provide strat of creators with quality q
    pass

def user_evolution_step():
    if rand.random() < USER_MUTATION_PROBABILITY:
        random_user: User = rand.choice(user_population)
        random_user.mutate()
    else:
        user_a: User
        user_b: User
        user_a, user_b = rand.sample(user_population, 2)
        user_a.fitness = 0
        user_b.fitness = 0

        # if rand.random() < MEDIA_MUTATION_PROBABILITY:
        #     user_a.tm = rand.random.choice(range(0, NUMBER_COMMENTATORS))
        # if rand.random() < MEDIA_MUTATION_PROBABILITY:
        #     user_b.tm = rand.random.choice(range(0, NUMBER_COMMENTATORS))

        # build trust media vector stochastically 
        user_a.media_trust_vector = rand.choices(population=media_population, weights=MEDIA_QUALITY_EXPECTED, k=user_a.tm)
        print("aaaaa", user_a.media_trust_vector)
        user_b.media_trust_vector = rand.choices(population=media_population, weights=MEDIA_QUALITY_EXPECTED, k=user_b.tm)

        # user A plays Z games
        for _ in range(NUMBER_CREATORS):
            creator: Creator = rand.choice(creator_population)
            calculate_payoff(user_a, creator)
        # print("User", user_a.id, "with strategy", user_a.strat, "accumulated", user_a.fitness, "fitness")

        # user B plays Z games
        for _ in range(NUMBER_CREATORS):
            creator: Creator = rand.choice(creator_population)
            calculate_payoff(user_b, creator)
        # print("User", user_b.id, "with strategy", user_b.strat, "accumulated", user_b.fitness, "fitness")

        # learning step
        # Calculate Probability of imitation
        p_i: float = (
            1 + np.exp(U_SELECTION_STRENGTH * (user_a.fitness - user_b.fitness))
        ) ** (-1)
        # print("\tLearning A->B probability:", p_i)
        if rand.random() < p_i:
            user_a.strat = user_b.strat
            user_a.tm = user_b.tm

        #TODO: update reputations


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
            calculate_payoff(user, creator_a)

        # Creator B
        for _ in range(NUMBER_USERS):
            user: User = rand.choice(user_population)
            calculate_payoff(user, creator_b)

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


def calculate_payoff(u: User, c: Creator):
    # Create a list of opinions of only trusted sources
    # media_beliefs_of_creat

    sum_media_beliefs_of_creator = 0
    if u.tm != 0:
        media_beliefs_of_creator = []
        for media in u.media_trust_vector:
            if rand.random() <= media.quality:
                media_beliefs_of_creator.append(c.strategy)
            else:
                media_beliefs_of_creator.append(rand.choice([DEFECT, COOPERATE]))
        sum_media_beliefs_of_creator = sum(media_beliefs_of_creator)

    # Payoffs are (kinda) different depending on u.strat being 2 or 3 or more
    user_payoffs, creator_payoffs = payoff_matrix(u, sum_media_beliefs_of_creator)
    u.fitness += user_payoffs[c.strategy, u.strat]
    c.fitness += creator_payoffs[c.strategy, u.strat]


def draw(g: int):

    # Clear plot and re-draw
    os.system("cls")
    plt.clear_data()
    plt.title("User Strategy Evolution Over Generations")
    plt.xlabel("Generations")
    plt.ylabel("Number of Users")
    # Plot each dataset
    plt.plot(generations, never_adopt, label="Never Adopt", color="red")
    plt.plot(generations, always_adopt, label="Always Adopt", color="blue")
    plt.plot(generations, optimist, label="Optimist", color="green")
    plt.plot(generations, pessimist, label="Pessimist", color="yellow")
    plt.plot(
        generations,
        creator_cooperator,
        label="Cooperative Creators",
        color="light green",
        marker="square",
    )
    plt.plot(
        generations,
        creator_defector,
        label="Defective Creators",
        color="light red",
        marker="square",
    )

    plt.ylim(0, 1)  # Adjust Y-axis limits
    plt.xlim(0.01, g)
    plt.show()
    # plt.sleep(0.01)


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


def export_results(users_strats_counts: dict, creators_strats_counts: dict):
    print("USERS:", users_strats_counts)
    print("CREATORS:", creators_strats_counts)

    # Create a unique filename. Change it later to experiment name/id
    file_name: str = "outputs/" + str(round(time.time())) + ".csv"
    f = open(file_name, "a")
    # Write the time series of all relevant frequencies
    f.write("generation,N,A,O,P,Cc,Cd\n")
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
            + "\n"
        )
        f.write(output)
    f.close()

    df = pd.read_csv(file_name).drop("generation", axis=1)
    # color=['r','b','orange','g','purple','brown']
    ls=['-','-','-', '-','-.']
    labels=['N','A','O','P','CC','CD']
    for i, col in enumerate(['N','A','O','P','Cc']):
        df[col].plot(ls=ls[i], label=labels[i])
    df['Cc']
    plts.legend(loc='upper left')
    plts.show()


def print_media_trust_avg():
    counters = np.zeros(NUMBER_COMMENTATORS)
    for user in user_population:
        counters += np.array(user.media_trust_vector)
    print(counters / NUMBER_USERS)


def run_one_generation(logging):
    global media_image_matrix

    initialization()

    g, n, a, o, p, cc, cd = [], [], [], [], [], [], [] 

    for generation in tqdm(range(GENS)):
        # 0. Generate media image matrix
        # media_image_matrix = generate_media_beliefs()
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
        # user
        if generation % 1000 == 0 and logging:
            draw(generation)
    
    return g, n, a, o, p, cc, cd 


def main(logging: bool = True):
    global media_image_matrix
    global REAL_CREATOR_STRATEGIES
    global generations
    global never_adopt 
    global always_adopt 
    global optimist
    global pessimist 
    global creator_cooperator 
    global creator_defector

    # Have a fixed initial configuration of trustworthiness of commentators
    read_args()

    g_tmp = np.zeros(GENS)
    n_tmp = np.zeros(GENS)
    a_tmp = np.zeros(GENS)
    o_tmp = np.zeros(GENS)
    p_tmp = np.zeros(GENS)
    cc_tmp = np.zeros(GENS)
    cd_tmp = np.zeros(GENS)

    for run in range(RUNS):
        g, n, a, o, p, cc, cd = run_one_generation(logging)

        g_tmp = np.array(g)
        n_tmp += np.array(n)
        a_tmp += np.array(a)
        o_tmp += np.array(o)
        p_tmp += np.array(p)
        cc_tmp += np.array(cc)
        cd_tmp += np.array(cd)
    
    generations = g_tmp
    never_adopt = n_tmp/RUNS
    always_adopt = a_tmp/RUNS
    optimist = o_tmp/RUNS
    pessimist = p_tmp/RUNS
    creator_cooperator = cc_tmp/RUNS
    creator_defector = cd_tmp/RUNS

    # print("FINAL IMAGE MATRIX:")
    # pprint.pprint(media_image_matrix)
    # print_media_trust_avg()
    
    export_results(count_user_strategies(), count_creator_strategies())


main(logging=False)
