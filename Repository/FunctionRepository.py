from Utilities.Object import ObjectType, OBJECT_COUNT
from Utilities.SearchZone import VEHICLE_COUNT, vehicule
from Utilities.Function import Function, FunctionType


class FunctionRepository:
    def __init__(self):
        self.functions_indexes_py = [
            [0 for _ in range(OBJECT_COUNT)]
            for _ in range(VEHICLE_COUNT)
        ]

        self.functions = [Function("Fonction rectangle exemple", FunctionType.rectangle, height=0.2, threshold = 7)]

    def nameExist(self, name):
        for fun in self.functions:
            if name == fun.name:
                return True
        return False
    
    def getNames(self):
        names = list()
        for fun in self.functions:
            names.append(fun.name)
        return names
    
    def setFunctionIndex(self, row, col, index):
        self.functions_indexes_py[row][col] = index

    def addFunction(self, name, type, **kwargs):
        if self.nameExist(name):
            raise Exception(f"La fonction '{name}' éxiste déjà.")
        self.functions.append(Function(name,type, **kwargs))

    def updateFunction(self,originalName, name,type, **kwargs):
        if not self.nameExist(originalName):
            raise Exception(f"La fonction '{name}' n'éxiste pas.")
        for i in range(len(self.functions)):
            if self.functions[i].name == originalName:
                self.functions[i] = Function(name,type,**kwargs)

    def removeFunction(self, name):
        if len(self.functions) ==1:
            raise Exception("Impossible de supprimer la dernière fonction")
        for fun in self.functions:
            if fun.name == name:
                index = self.functions.index(fun)
                for sublist in self.functions_indexes_py:
                    for i in range(len(sublist)):
                        if sublist[i] == index:  # Vérifier si l'élément est égal à l'index
                            sublist[i] = 0  # Remplacer l'élément par 0
                        if sublist[i] > index:
                            sublist[i] = sublist[i]-1
                self.functions.remove(fun)
                return
        raise Exception(f"La fonction '{name}' n'éxiste pas")

    def changeIndex(self, objectType, vehicle, index):
        self.functions[vehicle.value][objectType.value] = index
        