from .QLearner import QLearner
from .LearnBehavior import LearnBehavior

REGISTRY = {}

REGISTRY["QLearner"] = QLearner
REGISTRY["LearnBehavior"] = LearnBehavior
