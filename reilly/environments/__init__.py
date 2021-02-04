from .environment import Environment
from .gym import *
from .text import TextEnvironment, TextNeighbor


def ENVIRONMENTS():
    import sys
    import inspect
    module = sys.modules[__name__]
    return [
        value
        for _, value in module.__dict__.items()
        if inspect.isclass(value)
        and issubclass(value, Environment)
        and value != Environment
    ]
