def get_line(y1, y2, m, b):
    x2 = (y2 - b) / m
    x1 = (y1 - b) / m
    return Line(x1, y1, x2, y2, m, b)


class Line:
    # y = mx + b
    def __init__(self, x1, y1, x2, y2, m=None, b=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        if m is not None and b is not None:
            self.m = m
            self.b = b
        else:
            self.init_parameters()

    def init_parameters(self):
        m = (self.y2 - self.y1) / (self.x2 - self.x1)
        b = self.y1 - m * self.x1
        self.m = m
        self.b = b

    def get_x1(self):
        return int(self.x1)

    def get_x2(self):
        return int(self.x2)

    def get_y1(self):
        return int(self.y1)

    def get_y2(self):
        return int(self.y2)

    def __str__(self):
        return f'(x1,y1)=({self.x1},{self.y1});(x2,y2)=({self.x2},{self.y2}); (m,b) = ({self.m},{self.b})) '
