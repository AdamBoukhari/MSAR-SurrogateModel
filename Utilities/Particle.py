from Utilities.ParticlePoint import ParticlePoint

class Particle:
    def __init__(self, id):
        self.id =id 
        self.points = []

    def appendParticle(self, position, state, age,particle_timestamp, orientation):
        self.points.append(ParticlePoint(position, state, age, particle_timestamp, orientation))
        