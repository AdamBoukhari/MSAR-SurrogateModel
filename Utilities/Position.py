class Position:
    def __init__(self, x=0,y=0):
        self.x = x
        self.y = y

    def move(self, x=0, y=0):
        last = Position(x= self.x, y= self.y)
        new = Position(x= self.x+x, y= self.y+y)
        return last, new
    