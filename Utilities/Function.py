from enum import Enum, auto

class FunctionType(Enum):
    rectangle = auto()
    linear = auto()
    gaussian = auto()

FUNCTION_PARAMETERS = {
    FunctionType.rectangle: [("height", float), ("threshold", float)],
    FunctionType.linear: [("height", float), ("threshold", float)],
    FunctionType.gaussian: [("height", float), ("threshold", float)],
}

class Function:
    def __init__(self, name, functionType, **kwargs):
        self.name = name
        self.functionType = functionType
        if 'height' in kwargs:
            self.height = kwargs['height']
        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']