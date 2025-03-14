import random

COOPERATE = 1  # Does diligent research
DEFECT = 0  # Does not research


class Commentator:
    def __init__(self, commentator_id: int, q: float = None):
        self.id = commentator_id
        self.quality = q
