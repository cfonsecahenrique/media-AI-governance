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
        self.user_mutation_rate = (
            simulation["user mutation probability"] / self.num_users
        )
        self.creator_beta = simulation["user selection strength"]
        self.creator_mutation_rate = (
            simulation["creator mutation probability"] / self.num_creators
        )
        self.gens = simulation["generations"]
        self.media_quality = parameters["media quality"]
        self.converge = simulation["convergence period"]
        self.past_convergence = False

        self.q = parameters["media quality"]
        self.bU = parameters["user benefit"]
        self.cU = parameters["user cost"]
        self.cI = parameters["cost investigation"]
        self.bP = parameters["creator benefit"]
        self.cP = parameters["creator cost"]

        self.user_payoff_matrix, self.creator_payoff_matrix = (
            self.calculate_payoff_matrices()
        )
        self.user_pop, self.creator_pop = self.init_population(parameters)

        self.user_cooperative_acts: int = 0
        self.creator_cooperative_acts: int = 0
        self.total_actions: int = 0

    def __str__(self):
        user_pm, creator_pm = self.user_payoff_matrix, self.creator_payoff_matrix
        return (
            f"\n===== Simulation Parameters =====\n"
            f"User population size (Z):       {self.num_users}\n"
            f"Creator population size (Zc):   {self.num_creators}\n"
            f"User selection strength (βᵤ):   {self.user_beta}\n"
            f"Creator selection strength (β𝚌): {self.creator_beta}\n"
            f"User mutation rate (μᵤ):        {self.user_mutation_rate:.6f}\n"
            f"Creator mutation rate (μ𝚌):     {self.creator_mutation_rate:.6f}\n"
            f"Generations:                    {self.gens}\n"
            f"Convergence period:             {self.converge}\n"
            f"\n===== User Payoff Matrix =====\n"
            f"         | AllD     AllC     BMedia   GMedia\n"
            f"--------------------------------------------\n"
            f"Defect   | {user_pm[0, 0]:>7.2f}  {user_pm[0, 1]:>7.2f}  {user_pm[0, 2]:>7.2f}  {user_pm[0, 3]:>7.2f}\n"
            f"Cooperate| {user_pm[1, 0]:>7.2f}  {user_pm[1, 1]:>7.2f}  {user_pm[1, 2]:>7.2f}  {user_pm[1, 3]:>7.2f}\n"
            f"\n===== Creator Payoff Matrix =====\n"
            f"         | AllD     AllC     BMedia   GMedia\n"
            f"--------------------------------------------\n"
            f"Defect   | {creator_pm[0, 0]:>7.2f}  {creator_pm[0, 1]:>7.2f}  {creator_pm[0, 2]:>7.2f}  {creator_pm[0, 3]:>7.2f}\n"
            f"Cooperate| {creator_pm[1, 0]:>7.2f}  {creator_pm[1, 1]:>7.2f}  {creator_pm[1, 2]:>7.2f}  {creator_pm[1, 3]:>7.2f}\n"
        )

    def calculate_payoff_matrices(self):
        # Order for strats:
        # [D][wtv]: opponent defected
        # [D][D]: AllD; [D][C]: AllC; [D][B]: Bad Media Follower; [D][G]: Good Media follower;
        # [C][wtv]: opponent cooperated
        # [C][D]: etc...
        user_payoffs = np.array(
            [
                [0, -self.cU, -0.5 * self.cU, -(1 - self.q) * self.cU - self.cI],
                [0, self.bU, 0.5 * self.bU, self.q * self.bU - self.cI],
            ]
        )
        creator_payoffs = np.array(
            [
                [0, self.bP, 0.5 * self.bP, (1 - self.q) * self.bP],
                [
                    -self.cP,
                    self.bP - self.cP,
                    0.5 * self.bP - self.cP,
                    self.q * self.bP - self.cP,
                ],
            ]
        )
        return user_payoffs, creator_payoffs

    def calculate_payoff(self, user: User, creator: Creator):
        user.fitness += self.user_payoff_matrix[creator.strategy, user.strategy]
        creator.fitness += self.creator_payoff_matrix[creator.strategy, user.strategy]

    def init_population(self, parameters):
        """
        Initialize population of users and creators

        Args:
            parameters: interaction parameters

        Returns:
            list[User]: population of users
            list[Creator]: population of creators
        """
        user_population: list[User] = []
        creator_population: list[Creator] = []

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
            creators_for_user_a: list[Creator] = random.choices(
                self.creator_pop, k=self.num_creators
            )
            creators_for_user_b: list[Creator] = random.choices(
                self.creator_pop, k=self.num_creators
            )

            # user A plays Z games
            for creator in creators_for_user_a:
                self.calculate_payoff(user_a, creator)

                if self.past_convergence:
                    # action from user + action from creator
                    self.total_actions += 2
                    if user_a.strategy == 1:
                        self.user_cooperative_acts += 1
                    elif (user_a.strategy == 2 and random.random() < 0.5) or (
                        user_a.strategy == 3 and random.random() < self.media_quality
                    ):
                        self.user_cooperative_acts += 1
                    # 0 = def, 1 = coop
                    self.creator_cooperative_acts += creator.strategy

            # user B plays Z games
            for creator in creators_for_user_b:
                self.calculate_payoff(user_b, creator)

                if self.past_convergence:
                    # action from user + action from creator
                    self.total_actions += 2
                    if user_b.strategy == 1:
                        self.user_cooperative_acts += 1
                    elif (user_b.strategy == 2 and random.random() < 0.5) or (
                        user_b.strategy == 3 and random.random() < self.media_quality
                    ):
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
                self.calculate_payoff(user, creator_a)

                if self.past_convergence:
                    # action from user + action from creator
                    self.total_actions += 2
                    if user.strategy == 1:
                        self.user_cooperative_acts += 1
                    elif (user.strategy == 2 and random.random() < 0.5) or (
                        user.strategy == 3 and random.random() < self.media_quality
                    ):
                        self.user_cooperative_acts += 1
                    # 0 = def, 1 = coop
                    self.creator_cooperative_acts += creator_a.strategy

            # creator B plays X games (X=#users)
            for user in users_for_creator_b:
                self.calculate_payoff(user, creator_b)

                if self.past_convergence:
                    # action from user + action from creator
                    self.total_actions += 2
                    if user.strategy == 1:
                        self.user_cooperative_acts += 1
                    elif (user.strategy == 2 and random.random() < 0.5) or (
                        user.strategy == 3 and random.random() < self.media_quality
                    ):
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
        strategies = np.fromiter(
            (u.strategy for u in self.user_pop), dtype=np.int8, count=self.num_users
        )
        counts = np.bincount(strategies, minlength=4)
        return dict(enumerate(counts))

    def count_creator_strategies(self):
        """
        Count number of creators for each strategy
        (Numpy optimised)
        Returns:
            dict: count for each strategy
        """
        strategies = np.fromiter(
            (c.strategy for c in self.creator_pop),
            dtype=np.int8,
            count=self.num_creators,
        )
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
        d, c, b, g = (
            np.zeros(self.gens),
            np.zeros(self.gens),
            np.zeros(self.gens),
            np.zeros(self.gens),
        )
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

            d[gen] = user_strats_dict[0] / self.num_users
            c[gen] = user_strats_dict[1] / self.num_users
            b[gen] = user_strats_dict[2] / self.num_users
            g[gen] = user_strats_dict[3] / self.num_users
            cd[gen] = creator_strats_dict[0] / self.num_creators
            cc[gen] = creator_strats_dict[1] / self.num_creators
            acr[gen] = (
                (self.creator_cooperative_acts + self.user_cooperative_acts)
                / self.total_actions
                if (self.past_convergence and self.total_actions > 0)
                else 0
            )

        if filename:
            self.write_output(
                filename,
                acr.tolist(),
                d.tolist(),
                c.tolist(),
                b.tolist(),
                g.tolist(),
                cd.tolist(),
                cc.tolist(),
            )
