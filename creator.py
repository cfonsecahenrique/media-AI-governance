# system imports
import random

COOPERATE = 1  # Produces SAFE systems
DEFECT = 0  # Produces UNSAFE systems


class Creator:
    def __init__(self, dev_id: int, initial_strat=-1):
        self.id = dev_id
        self.strategy: int = random.choice([COOPERATE, DEFECT]) if initial_strat == -1 else initial_strat
        self.fitness: int = 0

    def mutate(self):
        self.strategy: int = 1 - self.strategy
