from Utilities.Particle import Particle
import uuid
from enum import Enum, auto


class ObjectType(Enum):
    raft = auto()
    person = auto()
OBJECT_COUNT = ObjectType.person.value - ObjectType.raft.value +1


class Object:
    def __init__(self, numberOfScenarios, objectClassId, objectClassName, initialPosition, initialTimestamp, rad):
        self.id = uuid.uuid4() #UUID
        self.idstr = str(self.id)
        self.objectClassId = objectClassId #INT
        self.objectClassName = objectClassName #STRING
        self.initialPosition = initialPosition #POSITION
        self.initialTimestamp = initialTimestamp #STRING
        self.objectType = ObjectType.raft
        self.rad = rad #FLOAT
        self.particles = [] #ARRAY SCENARIO
        for i in range(numberOfScenarios):
            self.particles.append(Particle(i))
    
    def appendParticle(self, id, position, state, age, particle_timestamp, orientation):
        self.particles[id].appendParticle(position, state, age, particle_timestamp, orientation)
