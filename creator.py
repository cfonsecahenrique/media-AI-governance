# system imports
import random


COOPERATE = 1  # Produces SAFE systems
DEFECT = 0  # Produces UNSAFE systems


class Creator:
    def __init__(self, dev_id: int):
        self.id = dev_id
        self.strategy: int = random.choice([COOPERATE, DEFECT])
        self.fitness: int = 0

    def mutate(self):
        self.strategy: int = 1 - self.strategy

    def calculate_payoff(self, user):
        user_payoffs, creator_payoffs = user.payoff_matrix()
        self.fitness += creator_payoffs[self.strategy, user.strategy]
