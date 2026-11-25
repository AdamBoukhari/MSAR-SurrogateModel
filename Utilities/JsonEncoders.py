from Model.Function import Function, FunctionType
from Model.Object import Object, ObjectType
from Model.Particle import Particle
from Model.SearchZone import SearchZone, vehicule
from Model.Scenario import Scenario
from Model.PathSegment import PathSegment
import json
from enum import Enum
import uuid
from Utilities.Position import Position

class SituationEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):  # Sérialisation des Enum
            return {"__type__": "Enum", "class": obj.__class__.__name__, "name": obj.name}
        elif isinstance(obj, uuid.UUID):  # Sérialisation des UUID
            return {"__type__": "UUID", "value": str(obj)}
        if isinstance(obj, Position):
            return {
                "__type__": "Position",
                "x": obj.x,
                "y": obj.y
            }
        if isinstance(obj, Function):
            return {
                "__type__": "Function",
                "name": obj.name,
                "functionType": obj.functionType,
                "height": getattr(obj, "height", None),
                "threshold": getattr(obj, "threshold", None)
            }
        elif isinstance(obj, Object):
            return {
                "__type__": "Object",
                "idstr": obj.idstr,
                "objectClassId": obj.objectClassId,
                "objectClassName": obj.objectClassName,
                "initialPosition": obj.initialPosition,
                "initialTimestamp": obj.initialTimestamp,
                "rad": obj.rad,
                "scenarios": obj.scenarios
            }
        elif isinstance(obj, Particle):
            return {
                "__type__": "Particle",
                "position": obj.position,
                "state": obj.state,
                "age": obj.age,
                "particle_timestamp": obj.particle_timestamp,
                "orientation": obj.orientation
            }
        elif isinstance(obj, PathSegment):
            return {
                "__type__": "PathSegment",
                "startPoint": obj.startPoint,
                "endPoint": obj.endPoint,
                "speed": obj.speed,
                "pathType": obj.pathType
            }
        elif isinstance(obj, Scenario):
            return {
                "__type__": "Scenario",
                "id": obj.id,
                "particles": obj.particles
            }
        elif isinstance(obj, SearchZone):
            return {
                "__type__": "SearchZone",
                "idstr": obj.idstr,
                "position": obj.position,
                "timestamp": obj.timestamp,
                "width": obj.width,
                "height": obj.height,
                "endeavorHour": obj.endeavorHour,
                "searchSpeed": obj.searchSpeed,
                "transitSpeed": obj.transitSpeed,
                "endeavor": obj.endeavor
            }
        return super().default(obj)
    
def situation_decoder(d):
    if "__type__" in d:
        if d["__type__"] == "UUID":
            return uuid.UUID(d["value"])
        elif d["__type__"] == "Position":
            return Position(x=d["x"], y=d["y"])
        elif d["__type__"] == "Enum":
            if d["class"] == "FunctionType":
                return FunctionType[d["name"]]
            elif d["class"] == "ObjectType":
                return ObjectType[d["name"]]
            elif d["class"] == "vehicule":
                return vehicule[d["name"]]
        if d["__type__"] == "Function":
            return Function(d["name"], d["functionType"], height=d.get("height"), threshold=d.get("threshold"))
        elif d["__type__"] == "Object":
            obj = Object(0, d["objectClassId"], d["objectClassName"], d["initialPosition"], d["initialTimestamp"], d["rad"])
            obj.idstr = d["idstr"]
            obj.scenarios = [situation_decoder(s) for s in d["scenarios"]]
            return obj
        elif d["__type__"] == "Particle":
            return Particle(d["position"], d["state"], d["age"], d["particle_timestamp"], d["orientation"])
        elif d["__type__"] == "PathSegment":
            return PathSegment(d["startPoint"], d["endPoint"], d["speed"], d["pathType"])
        elif d["__type__"] == "Scenario":
            scenario = Scenario(d["id"])
            scenario.particles = [situation_decoder(p) for p in d["particles"]]
            return scenario
        elif d["__type__"] == "SearchZone":
            return SearchZone(d["position"], d["timestamp"])
    return d
