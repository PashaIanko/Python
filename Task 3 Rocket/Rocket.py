import math
import random
from tkinter import *

win_height = 600
win_width = 600
gravity = 0.2
dt = 0.1

master = Tk()

win = Canvas(master, width=win_width, height=win_height)
win.pack()

def randomColor():
    r = math.floor(random.random() * 255)
    g = math.floor(random.random() * 255)
    b = math.floor(random.random() * 255)
    return rgb2hex(r, g, b)

def rgb2hex (r, g, b):
    return '#%02x%02x%02x'%(r, g, b)

class Particle:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.vx = random() * 2 - 1
        self.vy = random() * 2 - 1
        self.vz = random() * 2 - 1
        self.color = randomColor()

    def update(self):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        self.vy += gravity * dt

class Firework:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.vx = 0
        self.vy = 10
        self.vz = 0
        self.explodeTime = 10

    def update(self):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        self.vy += gravity * dt
        self.explodeTime -= 1/60
        
particles[]
    
        
        
