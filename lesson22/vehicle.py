import numpy as np
from math import pi
from matplotlib import pyplot as plt

class Vehicle:
    def __init__(self):
        self.x       = 0 # meters
        self.y       = 0
        self.heading = "E" # Can be "N", "S", "E", or "W"
        self.history = []
        
    def drive_forward(self, displacement):
        """
        Updates x and y coordinates of vehicle based on 
        heading and appends previous (x,y) position to
        history.
        """
        
        # east
        if self.heading == "E":
            delta_x = displacement 
            delta_y = 0
        
        # north
        elif self.heading == "N":
            delta_x = 0 
            delta_y = displacement
        
        # west
        elif self.heading == "W":
            delta_x = -displacement
            delta_y = 0
        
        # south
        else:
            delta_x = 0 
            delta_y = -displacement
            
        new_x = self.x + delta_x
        new_y = self.y + delta_y
        
        self.history.append((self.x, self.y))

        self.x = new_x
        self.y = new_y
        
    def turn(self, direction):
        if direction == "L":
            self.turn_left()
        elif direction == "R":
            self.turn_right()
        else:
            print("Error. Direction must be 'L' or 'R'")
            return
        
    def turn_left(self):
        next_heading = {
            "N" : "W",
            "W" : "S",
            "S" : "E",
            "E" : "N",
        }
        self.heading = next_heading[self.heading]
        
    def turn_right(self):
        next_heading = {
            "N" : "E",
            "W" : "N",
            "S" : "W",
            "E" : "S",
        }
        self.heading = next_heading[self.heading]
    
    def show_trajectory(self):
        """
        Creates a scatter plot of vehicle's trajectory.
        """
        X = [p[0] for p in self.history]
        Y = [p[1] for p in self.history]
        
        X.append(self.x)
        Y.append(self.y)
        
        plt.scatter(X,Y)
        plt.plot(X,Y)
        plt.show()
