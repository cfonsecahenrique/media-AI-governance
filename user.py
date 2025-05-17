# system imports
import random

# external libraries
import numpy as np

# custom libraries
from commentator import Commentator

# USER STRATEGIES
# never adopt systems regardless of media opinions on creators
ALL_REJECT = 0
# always adopt systems regardless of media opinions on creators
ALL_ADOPT = 1
# Uses cheap (costless) low quality media
BAD_MEDIA = 2
# Uses expensive but high quality media
GOOD_MEDIA = 3


class User:
    def __init__(self, user_id: int, parameters):
        self.id = user_id
        self.fitness: int = 0
        self.strategy: int = random.choice(
            (ALL_REJECT, ALL_ADOPT, BAD_MEDIA, GOOD_MEDIA)
        )

    def mutate(self):
        self.strategy: str = random.choice(
            list({ALL_REJECT, ALL_ADOPT, BAD_MEDIA, GOOD_MEDIA} - {self.strategy})
        )
