from enum import Enum, auto
class SimulatorType(Enum):
    NormalPerStep = auto()
    NormalPerLeg = auto()
    NormalPerStepLog = auto()
    ParallelPerStep = auto()
    ParallelPerLeg = auto()
    GPUCalculator = auto()
    GPUCalculatorLeg = auto()
    GPUCalculatorLegTotal = auto()
