# system imports
import random

# custom libraries
from commentator import Commentator

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
        # generate media beliefs of creators
        media_beliefs_of_creator = user.generate_media_beliefs(self.strategy)

        # Payoffs are (kinda) different depending on u.strat being 2 or 3 or more
        user_payoffs, creator_payoffs = user.payoff_matrix(
            sum(media_beliefs_of_creator)
        )
        self.fitness += creator_payoffs[self.strategy, user.strategy]
