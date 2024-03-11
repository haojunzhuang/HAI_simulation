from enum import Enum

class Status(Enum):
    healthy   = 0
    colonized = 1
    infected  = 3
    recovered = 4