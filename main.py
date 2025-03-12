import os
import time
import yaml
import sys
import numpy as np
import random as rand
from tqdm import tqdm
import plotext as plt
import pprint

from User import User
from Commentator import Commentator
from Creator import Creator

COOPERATE = 1
DEFECT = 0
NUMBER_USERS: int
NUMBER_COMMENTATORS: int
NUMBER_CREATORS: int
MEDIA_TRUST_VECTOR: list
MEDIA_QUALITY: list
GENS: int
USER_MUTATION_PROBABILITY: float
CREATOR_MUTATION_PROBABILITY: float
SELECTION_STRENGTH = 1
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

media_image_matrix = np.zeros((0, 0))

# Benefit a user receives when adopting a safe technology
bU = 0.6
# Cost for the user adopting unsafe technology
cU = 0.8
# Benefit the media gets by users paying the (same) cost to access it
bM = 0.01
# Benefit for the creator when user uses
bP = 0.4
# Cost paid by creators to create safe AI
cP = 0.1


def read_args():
    file_name: str = str(sys.argv[1])
    global NUMBER_USERS
    global NUMBER_COMMENTATORS
    global NUMBER_CREATORS
    global USER_MUTATION_PROBABILITY
    global CREATOR_MUTATION_PROBABILITY
    global MEDIA_TRUST_VECTOR
    global MEDIA_QUALITY
    global GENS

    # Open and parse the YAML file
    with open(file_name, "r") as f:
        data = yaml.safe_load(f)

    for entry in data.get("instructions", []):
        NUMBER_USERS = int(entry["user population size"])
        NUMBER_COMMENTATORS = int(entry["commentator population size"])
        NUMBER_CREATORS = int(entry["creators population size"])
        MEDIA_QUALITY = list(entry["media quality"])
        USER_MUTATION_PROBABILITY = float(entry["user mutation probability"]/NUMBER_USERS)
        CREATOR_MUTATION_PROBABILITY = float(entry["creator mutation probability"]/NUMBER_CREATORS)
        MEDIA_TRUST_VECTOR = list(entry["media trust vector"])
        GENS = int(entry["generations"])


def initialization():
    # Have a fixed initial configuration of trustworthiness of commentators
    read_args()
    # Create population of users
    global user_population
    global REAL_CREATOR_STRATEGIES

    for i in range(0, NUMBER_USERS):
        user_population.append(User(i))

    # Create population of commentators
    global media_population
    for j in range(0, NUMBER_COMMENTATORS):
        media_population.append(Commentator(j, MEDIA_QUALITY[j]))

    # Create population of Devs
    global creator_population
    for k in range(0, NUMBER_CREATORS):
        creator_population.append(Creator(k))

    for user in user_population:
        user.media_trust_vector = [rand.choice((0, 1)) for _ in range(NUMBER_COMMENTATORS)]

    # Create TRUE vector of creator reputations
    for creator in creator_population:
        REAL_CREATOR_STRATEGIES.append(creator.strategy)


def generate_media_beliefs():
    global media_image_matrix
    media_image_matrix = np.zeros((NUMBER_COMMENTATORS, NUMBER_CREATORS))
    for comm in media_population:
        for creator in creator_population:
            if rand.random() < comm.quality:
                media_image_matrix[comm.id, creator.id] = creator.strategy
            else:
                media_image_matrix[comm.id, creator.id] = rand.choice((DEFECT, COOPERATE))
    return media_image_matrix


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
        p_i: float = (1 + np.exp(SELECTION_STRENGTH * (user_a.fitness - user_b.fitness))) ** (-1)
        # print("\tLearning A->B probability:", p_i)
        if rand.random() < p_i:
            user_a.strat = user_b.strat


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
        p_i: float = (1 + np.exp(SELECTION_STRENGTH * (creator_a.fitness - creator_b.fitness))) ** (-1)
        if rand.random() < p_i:
            creator_a.strategy = creator_b.strategy


def theta_function(creator_id: int, thresh: int):
    # if threshold = 1, return 1 on having one+ positive trusted recommendation -> optimist
    # if threshold = tM, return 1 only if all trusted sources recommend cooperation
    media_beliefs_of_creator = media_image_matrix[:, creator_id]
    trusted_ones = [media_beliefs_of_creator[i]
                    for i in range(len(MEDIA_TRUST_VECTOR)) if MEDIA_TRUST_VECTOR[i] == 1]
    value = np.sum(trusted_ones)
    return 1 if value >= thresh else 0


def payoff_matrix(user_type: int, tM: int, creator_id: int):
    theta = -1
    # x = recommended action = 0 or 1
    # tM = number of trusted sources
    if user_type == 0:
        # Never Adopt
        pass
    elif user_type == 1:
        # Always Adopt
        pass
    elif user_type == 2:
        # Optimist
        theta = theta_function(creator_id, 1)
    elif user_type == 3:
        # Pessimist
        theta = theta_function(creator_id, tM)
    else:
        raise ValueError("User type error")

    user_payoffs = np.array([[0, -cU, theta * (-cU) - (tM * bM), theta * (-cU) - (tM * bM)],
                             [0, bU, theta * bU - tM * bM, theta * bU - tM * bM]])
    creator_payoffs = np.array([[0, bP, theta * bP, theta * bP],
                                [-cP, bP - cP, theta * bP - cP, theta * bP - cP]])
    return user_payoffs, creator_payoffs


def calculate_payoff(u: User, c: Creator):
    # Create a list of opinions of only trusted sources
    media_beliefs_of_creator = media_image_matrix[:, c.id]
    trusted_ones = [media_beliefs_of_creator[i]
                    for i in range(len(MEDIA_TRUST_VECTOR)) if MEDIA_TRUST_VECTOR[i] == 1]
    tM = len(trusted_ones)

    # Payoffs are (kinda) different depending on u.strat being 2 or 3 or more
    user_payoffs, creator_payoffs = payoff_matrix(u.strat, tM, c.id)
    u.fitness += round(user_payoffs[c.strategy, u.strat], 2)
    c.fitness += round(creator_payoffs[c.strategy, u.strat], 2)


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
    plt.plot(generations, creator_cooperator, label="Cooperative Creators", color="light green", marker="square")
    plt.plot(generations, creator_defector, label="Defective Creators", color="light red", marker="square")

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
    file_name: str = str(round(time.time())) + ".txt"
    f = open(file_name, "a")
    # Write the time series of all relevant frequencies
    f.write("generation\tN\tA\tP\tO\tCc\tCd")
    for g in range(GENS):
        output: str = str(generations[g]) + "\t" + str(never_adopt[g]) + "\t" + str(always_adopt[g]) + "\t" + \
                      str(pessimist[g]) + "\t" + str(optimist[g]) + "\t" + str(creator_cooperator[g]) + "\t" + \
                      str(creator_defector[g]) + "\n"
        f.write(output)
    f.close()


def main(logging: bool = True):
    global media_image_matrix
    global REAL_CREATOR_STRATEGIES

    initialization()
    print("INITIAL CREATOR STRATEGIES:", REAL_CREATOR_STRATEGIES)

    for g in tqdm(range(GENS)):
        # 0. Generate media image matrix
        media_image_matrix = generate_media_beliefs()
        # 1. Evolve agents
        user_evolution_step()
        # 2. Evolve Creators
        creator_evolution_step()

        user_strats_dict: dict = count_user_strategies()
        creator_strats_dict: dict = count_creator_strategies()
        # Store data for plotting
        generations.append(g)
        never_adopt.append(user_strats_dict[0] / NUMBER_USERS)
        always_adopt.append(user_strats_dict[1] / NUMBER_USERS)
        optimist.append(user_strats_dict[2] / NUMBER_USERS)
        pessimist.append(user_strats_dict[3] / NUMBER_USERS)
        creator_cooperator.append(creator_strats_dict[COOPERATE] / NUMBER_CREATORS)
        creator_defector.append(creator_strats_dict[DEFECT] / NUMBER_CREATORS)
        if g % 1000 == 0 and logging:
            draw(g)

    print("FINAL IMAGE MATRIX:")
    pprint.pprint(media_image_matrix)
    export_results(count_user_strategies(), count_creator_strategies())


main(logging=False)
