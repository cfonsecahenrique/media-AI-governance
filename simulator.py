# system imports
import random

# external libraries
import numpy as np

# custom libraries
from user import User
from creator import Creator


class Simulator:
    def __init__(self, simulation, parameters):
        self.num_users = simulation["user population size"]
        self.num_creators = simulation["creator population size"]
        self.user_beta = simulation["user selection strength"]
        self.user_mutation_rate = simulation["user mutation probability"]/self.num_users
        self.creator_beta = simulation["user selection strength"]
        self.creator_mutation_rate = simulation["creator mutation probability"]/self.num_creators
        self.gens = simulation["generations"]
        self.media_quality = parameters["media quality"]
        self.converge = simulation["convergence period"]
        self.past_convergence = False

        self.user_pop, self.creator_pop = self.init_population(parameters)
        self.user_cooperative_acts: int = 0
        self.creator_cooperative_acts: int = 0
        self.total_actions: int = 0

    def init_population(self, parameters):
        """
        Initialize population of users and creators

        Args:
            parameters: interaction parameters

        Returns:
            list[User]: population of users
            list[Creator]: population of creators
        """
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
        """
        User evolutionary step
        """
        if random.random() < self.user_mutation_rate:
            random_user: User = random.choice(self.user_pop)
            random_user.mutate()
        else:
            user_a: User
            user_b: User
            user_a, user_b = random.sample(self.user_pop, 2)
            user_a.fitness = 0
            user_b.fitness = 0

            # Sample once with replacement into a list to avoid repeated sampling
            creators_for_user_a: list[Creator] = random.choices(self.creator_pop, k=self.num_creators)
            creators_for_user_b: list[Creator] = random.choices(self.creator_pop, k=self.num_creators)

            # user A plays Z games
            for creator in creators_for_user_a:
                user_a.calculate_payoff(creator)

                if self.past_convergence:
                    # action from user + action from creator
                    self.total_actions += 2
                    if user_a.strategy == 1:
                        self.user_cooperative_acts += 1
                    elif (user_a.strategy == 2 and random.random() < 0.5) or (user_a.strategy == 3 and random.random() < self.media_quality):
                        self.user_cooperative_acts += 1
                    # 0 = def, 1 = coop
                    self.creator_cooperative_acts += creator.strategy

            # user B plays Z games
            for creator in creators_for_user_b:
                user_b.calculate_payoff(creator)

                if self.past_convergence:
                    # action from user + action from creator
                    self.total_actions += 2
                    if user_b.strategy == 1:
                        self.user_cooperative_acts += 1
                    elif (user_b.strategy == 2 and random.random() < 0.5) or (
                            user_b.strategy == 3 and random.random() < self.media_quality):
                        self.user_cooperative_acts += 1
                    # 0 = def, 1 = coop
                    self.creator_cooperative_acts += creator.strategy

            # normalize
            user_a.fitness /= self.num_creators
            user_b.fitness /= self.num_creators

            # learning step
            p_i: float = (
                1 + np.exp(self.user_beta * (user_a.fitness - user_b.fitness))
            ) ** (-1)

            if random.random() < p_i:
                user_a.strategy = user_b.strategy

    def creator_evolution_step(self):
        """
        Creator evolutionary step
        """
        if random.random() < self.creator_mutation_rate:
            random_creator = random.choice(self.creator_pop)
            random_creator.mutate()
        else:
            # monte carlo step stuff
            creator_a: Creator
            creator_b: Creator
            creator_a, creator_b = random.sample(self.creator_pop, 2)
            creator_a.fitness = 0
            creator_b.fitness = 0

            # Sample once with replacement into a list to avoid repeated sampling
            users_for_creator_a = random.choices(self.user_pop, k=self.num_users)
            users_for_creator_b = random.choices(self.user_pop, k=self.num_users)

            # creator A plays X games (X=#users)
            for user in users_for_creator_a:
                creator_a.calculate_payoff(user)

                if self.past_convergence:
                    # action from user + action from creator
                    self.total_actions += 2
                    if user.strategy == 1:
                        self.user_cooperative_acts += 1
                    elif (user.strategy == 2 and random.random() < 0.5) or (
                            user.strategy == 3 and random.random() < self.media_quality):
                        self.user_cooperative_acts += 1
                    # 0 = def, 1 = coop
                    self.creator_cooperative_acts += creator_a.strategy

            # creator B plays X games (X=#users)
            for user in users_for_creator_b:
                creator_b.calculate_payoff(user)

                if self.past_convergence:
                    # action from user + action from creator
                    self.total_actions += 2
                    if user.strategy == 1:
                        self.user_cooperative_acts += 1
                    elif (user.strategy == 2 and random.random() < 0.5) or \
                            (user.strategy == 3 and random.random() < self.media_quality):
                        self.user_cooperative_acts += 1
                    # 0 = def, 1 = coop
                    self.creator_cooperative_acts += creator_b.strategy

            # Learning step
            # Normalize
            creator_a.fitness /= len(self.user_pop)
            creator_b.fitness /= len(self.user_pop)
            # Fermi update
            p_i: float = (
                1 + np.exp(self.creator_beta * (creator_a.fitness - creator_b.fitness))
            ) ** (-1)

            if random.random() < p_i:
                creator_a.strategy = creator_b.strategy

    def count_user_strategies(self):
        """
        Count number of users for each strategy
        (Numpy optimised)
        Returns:
            dict: count for each strategy
        """
        strategies = np.fromiter((u.strategy for u in self.user_pop), dtype=np.int8, count=self.num_users)
        counts = np.bincount(strategies, minlength=4)
        return dict(enumerate(counts))

    def count_creator_strategies(self):
        """
        Count number of creators for each strategy
        (Numpy optimised)
        Returns:
            dict: count for each strategy
        """
        strategies = np.fromiter((c.strategy for c in self.creator_pop), dtype=np.int8, count=self.num_creators)
        counts = np.bincount(strategies, minlength=2)
        return dict(enumerate(counts))

    def write_output(self, filename, acr, d, c, b, g, cd, cc):
        """
        Write the output of the simulation on a csv file

        Args:
            filename (_type_): _description_
            acr (np.array): total cooperation ratio over generations (past convergence period)
            d (np.array): list of all d users over generations
            c (np.array): list of all c users over generations
            g (np.array): list of good media users over generations
            b (np.array): list of bad media users over generations
            cd (np.array): list of defective creators over generations
            cc (np.array): list of cooperative creators over generations
        """
        path = f"{filename}.csv"
        with open(path, "a") as file:
            labels = "gen,acr,AllD,AllC,BMedia,GMedia,Unsafe,Safe\n"
            file.write(labels)

            for i in range(self.gens):
                output = f"{i},{acr[i]},{d[i]},{c[i]},{b[i]},{g[i]},{cd[i]},{cc[i]}\n"
                file.write(output)

    def run(self, filename: str = ""):
        """
        Run the simulation.
        Args:
            filename (str, optional): If given, writes the results of the simulation on the filename. Defaults to "".
        """
        d, c, b, g = np.zeros(self.gens), np.zeros(self.gens), np.zeros(self.gens), np.zeros(self.gens)
        cc, cd = np.zeros(self.gens), np.zeros(self.gens)
        acr = np.zeros(self.gens)

        for gen in range(self.gens):
            if gen > self.converge:
                self.past_convergence = True
            # 1. Evolve agents
            self.user_evolution_step()
            # 2. Evolve Creators
            self.creator_evolution_step()

            user_strats_dict: dict = self.count_user_strategies()
            creator_strats_dict: dict = self.count_creator_strategies()

            d[gen] = (user_strats_dict[0] / self.num_users)
            c[gen] = (user_strats_dict[1] / self.num_users)
            b[gen] = (user_strats_dict[2] / self.num_users)
            g[gen] = (user_strats_dict[3] / self.num_users)
            cd[gen] = (creator_strats_dict[0] / self.num_creators)
            cc[gen] = (creator_strats_dict[1] / self.num_creators)
            acr[gen] = (self.creator_cooperative_acts + self.user_cooperative_acts)/self.total_actions \
                if self.past_convergence else 0

        if filename:
            self.write_output(filename, acr.tolist(), d.tolist(), c.tolist(), b.tolist(),
                              g.tolist(), cd.tolist(), cc.tolist())
