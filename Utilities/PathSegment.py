from enum import Enum, auto
class segmentPathType(Enum):
    active = auto()
    passive = auto()
    transit = auto()

class SearchLeg:

    def __init__(self, startPoint, endPoint, speed, pathType):
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.speed = speed
        self.pathType = pathType

    