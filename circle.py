import numpy as np
import matplotlib.pyplot as mlt

class Circle:
    def __init__(self, x_radius, center_x = 0, center_y = 0):
        self.radius = x_radius
        self.center_x = center_x
        self.center_y = center_y
        self.Y_positive = []
        self.Y_negative = []
    def forward(self, X):
        for x in X:
            self.Y_positive.append(np.sqrt( np.square(self.radius) - np.square( x - self.center_y) ) + self.center_x)
            self.Y_negative.append(-(np.sqrt( np.square(self.radius) - np.square( x - self.center_y) ) + self.center_x))
            

        
X = np.linspace(-100, 100, 100)
print(X)

my_circle = Circle(100)
my_circle.forward(X)

mlt.style.use('bmh')
mlt.scatter(X, my_circle.Y_positive)
mlt.scatter(X, my_circle.Y_negative)
mlt.axis('equal')
mlt.grid('on')
mlt.xticks(np.arange(-200, 201, 20))
mlt.yticks(np.arange(-200, 201, 20))
mlt.show()