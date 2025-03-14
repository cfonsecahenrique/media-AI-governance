# CT = 1  # Follows commentator recommendations
# N = 0  # Never adopts AI System
import random

from Commentator import Commentator

# USER STRATEGIES
# never adopt systems regardless of media opinions on creators
NEV_ADOPT = 0
# always adopt systems regardless of media opinions on creators
ALL_ADOPT = 1
# needs at least 1 positive recommendation from trusted media to adopt
OPTIMIST = 2
# needs at least 1 negative recommendation from trusted media to adopt
PESSIMIST = 3
# adopts or not depending on the mode of recommendations from trusted media
# CONSENSUS = 4


class User:
    def __init__(self, user_id: int):
        self.id = user_id
        self.fitness: int = 0
        self.strat: int = random.choice((ALL_ADOPT, NEV_ADOPT, OPTIMIST, PESSIMIST))
        self.media_trust_vector: list[Commentator]

    def mutate(self):
        self.strat: str = random.choice((ALL_ADOPT, NEV_ADOPT, OPTIMIST, PESSIMIST))
