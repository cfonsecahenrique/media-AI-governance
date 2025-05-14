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

        self.q = parameters["media quality"]
        self.bU = parameters["user benefit"]
        self.cU = parameters["user cost"]
        self.cI = parameters["cost investigation"]
        self.bP = parameters["creator benefit"]
        self.cP = parameters["creator cost"]

    def mutate(self):
        self.strategy: str = random.choice(
            list({ALL_REJECT, ALL_ADOPT, BAD_MEDIA, GOOD_MEDIA} - {self.strategy})
        )

    def calculate_payoff(self, creator):
        user_payoffs, creator_payoffs = self.payoff_matrix()
        self.fitness += user_payoffs[creator.strategy, self.strategy]

    def payoff_matrix(self):
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
