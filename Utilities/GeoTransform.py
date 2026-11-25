from geopy.distance import distance
from Utilities.Position import Position
import math
from geopy.point import Point

from geopy.distance import distance
from geopy.point import Point

class GeoTransform:
    def transform_deltapoint(self, pivot, delta_km):
        pivot_point = Point(pivot.y, pivot.x)  

        point_lat = distance(kilometers=delta_km.y).destination(pivot_point, bearing=0)

        point_lon = distance(kilometers=delta_km.x).destination((point_lat.latitude, point_lat.longitude), bearing=90)

        return Position(x=point_lon.longitude, y=point_lon.latitude)

    def rotation_Position(self, angle, position):
        angle_radians = math.radians(angle)
        newX = position.x * math.cos(angle_radians) - position.y * math.sin(angle_radians)
        newY = position.x * math.sin(angle_radians) + position.y * math.cos(angle_radians)
        return Position(x = newX, y = newY)
