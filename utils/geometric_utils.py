import math


def calculate_x_by_angle(x1, y1, y2, theta):
    return (y2 - y1) / math.tan(theta * math.pi / 180) + x1


class Line:
    # y = mx + b
    def __init__(self, x1, y1, x2, y2):
        self.m = None
        self.b = None
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.init_parameters()

    def init_parameters(self):
        m = (self.y2 - self.y1) / (self.x2 - self.x1)
        b = self.y1 - m * self.x1
        self.m = m
        self.b = b

    def calculate_x(self, y):
        return int((y - self.b) / self.m)

    def calculate_y(self, x):
        return int(self.m * x + self.b)

    def __str__(self):
        return f'(x1,y1)=({self.x1},{self.y1});(x2,y2)=({self.x2},{self.y2}); (m,b) = ({self.m},{self.b})) '
