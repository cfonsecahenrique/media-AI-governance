# system imports
import random

# external libraries
import numpy as np

# custom libraries
from commentator import Commentator

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
    def __init__(self, user_id: int, number_commentators: int, parameters):
        self.id = user_id
        self.fitness: int = 0
        self.strategy: int = random.choice((ALL_ADOPT, NEV_ADOPT, OPTIMIST, PESSIMIST))
        self.media_trust_vector: list[Commentator] = []
        self.number_commentators = number_commentators
        if number_commentators == 1:
            self.tm: int = 0 if self.strategy in [ALL_ADOPT, NEV_ADOPT] else 1
        else:
            self.tm: int = (
                0
                if self.strategy in [ALL_ADOPT, NEV_ADOPT]
                else random.choice(range(1, number_commentators))
            )

        self.bU = parameters["user benefit"]
        self.cU = parameters["user cost"]
        self.bM = parameters["media benefit"]
        self.bP = parameters["creator benefit"]
        self.cP = parameters["creator cost"]

    def mutate(self):
        self.strategy: str = random.choice(
            list({ALL_ADOPT, NEV_ADOPT, OPTIMIST, PESSIMIST} - {self.strategy})
        )
        if self.number_commentators == 1:
            self.tm: int = 0 if self.strategy in [ALL_ADOPT, NEV_ADOPT] else 1
        else:
            self.tm: int = (
                0
                if self.strategy in [ALL_ADOPT, NEV_ADOPT]
                else random.choice(range(1, self.number_commentators))
            )

    def generate_media_beliefs(self, creator_strategy):
        media_beliefs_of_creator = []
        if self.tm != 0:
            for media in self.media_trust_vector:
                if random.random() <= media.quality:
                    media_beliefs_of_creator.append(creator_strategy)
                else:
                    media_beliefs_of_creator.append(1 - creator_strategy)
        return media_beliefs_of_creator

    def calculate_payoff(self, creator):
        # media_beliefs_of_creator
        media_beliefs_of_creator = self.generate_media_beliefs(creator.strategy)

        # compare media beliefs with creators strategies
        up_or_down = np.ones(len(media_beliefs_of_creator))
        for i in range(len(up_or_down)):
            if media_beliefs_of_creator[i] != creator.strategy:
                up_or_down[i] = -1

        # Payoffs are (kinda) different depending on u.strat being 2 or 3 or more
        user_payoffs, creator_payoffs = self.payoff_matrix(
            sum(media_beliefs_of_creator)
        )
        self.fitness += user_payoffs[creator.strategy, self.strategy]
        creator.fitness += creator_payoffs[creator.strategy, self.strategy]

        return up_or_down

    @staticmethod
    def theta_function(sum_media_beliefs_of_creator, threshold: int):
        return 1 if sum_media_beliefs_of_creator >= threshold else 0

    def payoff_matrix(self, sum_media_beliefs_of_creator: int):
        theta = -1
        if self.strategy == 0:
        # Never Adopt
            pass
        elif self.strategy == 1:
        # Always Adopt
            pass
        elif self.strategy == 2:
            # Optimist
            theta = self.theta_function(sum_media_beliefs_of_creator, threshold=1)
        elif self.strategy == 3:
            # Pessimist
            theta = self.theta_function(sum_media_beliefs_of_creator, threshold=self.tm)
        else:
            raise ValueError("User strategy is not valid.")

        user_payoffs = np.array(
            [
                [
                    0,
                    -self.cU,
                    theta * (-self.cU) - (self.tm * self.bM),
                    theta * (-self.cU) - (self.tm * self.bM),
                ],
                [
                    0,
                    self.bU,
                    theta * self.bU - self.tm * self.bM,
                    theta * self.bU - self.tm * self.bM,
                ],
            ]
        )
        creator_payoffs = np.array(
            [
                [0, self.bP, theta * self.bP, theta * self.bP],
                [
                    -self.cP,
                    self.bP - self.cP,
                    theta * self.bP - self.cP,
                    theta * self.bP - self.cP,
                ],
            ]
        )

        return user_payoffs, creator_payoffs
