from enum import Enum, auto
from uuid import uuid4
from Utilities.PathSegment import SearchLeg, segmentPathType
from Utilities.Position import Position
import math
from datetime import datetime
class typeSearch(Enum):
    lined = auto()

class vehicule(Enum):
    plane = auto()
    boat = auto()
VEHICLE_COUNT = vehicule.boat.value - vehicule.plane.value +1

class SearchZone:
    def __init__(self, position, timestamp = int(datetime.now().timestamp())):
        self.id = uuid4()
        self.idstr = str(self.id)
        self.position = position
        self.width = 100
        self.height = 100
        self.endeavorHour = 2
        self.searchSpeed = 200
        self.transitSpeed = 600
        self.endeavor = self.endeavorHour* self.searchSpeed
        self.rotation = 0
        self.searchType = typeSearch.lined
        self.path = []
        self.vehicule = vehicule.boat
        self.timestamp = timestamp

    def getNumberLinesAndLengthDist(self):
        effectiveEndeavor = self.endeavor - self.width
        if effectiveEndeavor % self.height == 0:
            return effectiveEndeavor / self.height, self.width
        else:
            return math.ceil(effectiveEndeavor / self.height), self.width-(effectiveEndeavor % self.height)

    def generatePath(self):
        nbLines =  max(math.ceil((self.endeavor-self.width)/self.height), 2)
        verticalDist = nbLines*self.height
        lengthDist = self.endeavor - verticalDist
        sideMargin = (self.width - lengthDist) / 2
        step = lengthDist / (nbLines - 1)

        self.path=[]
        position = Position(x=-(self.width / 2) + sideMargin, y=-(self.height / 2))
        last, new = position.move(x=0, y=self.height)
        direction = -1
        self.path.append(SearchLeg(last, new, self.searchSpeed, segmentPathType.active))
        for _ in range(math.floor(nbLines - 1)):
            last, new = new.move(x=step, y=0)
            self.path.append(SearchLeg(last, new, self.searchSpeed, segmentPathType.passive if self.vehicule == vehicule.plane else segmentPathType.active))
            last, new = new.move(x=0, y=direction * self.height)
            self.path.append(SearchLeg(last, new, self.searchSpeed, segmentPathType.active))
            direction *= -1

