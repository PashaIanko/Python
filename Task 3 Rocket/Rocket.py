import math
import random
from tkinter import *

win_height = 600
win_width = 600
gravity = 0.2
dt = 1

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
        self.vx = random.random() * 2 - 1
        self.vy = random.random() * 2 - 1
        self.vz = random.random() * 2 - 1
        self.color = randomColor()

    def update(self):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        self.vy += gravity * dt
        

    def draw(self):
        size = 2
        win.create_rectangle(self.x-size, self.y-size, self.x+size, self.y+size,
                             fill=self.color)
        

class Firework:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.vx = 0
        self.vy = -10
        self.vz = 0
        self.explodeTime = 10
        self.remove = False

    def update(self):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        self.vy += gravity * dt
        self.explodeTime -= 1/60

        #if self.explodeTime <= 0 :
        #    self.remove = True

        if self.vy >= 0:
            self.remove = True

            
            
    def draw(self):
        win.create_rectangle(self.x-2, self.y-4, self.x+2, self.y+5,
                             fill="black")
        
particles = []
fireworks = []

def spawnFirework(event):
    fireworks.append(Firework(event.x, event.y, 0))

win.bind("<Button-1>", spawnFirework)

def engine():
    win.delete("all")

    for i in range(0, len(fireworks)):
        if i>=len(fireworks) - 1:
            break
        
        f = fireworks[i]
        if f.remove == True:
            for j in range (0, 100):
                particles.append(Particle(f.x, f.y, f.z))
            fireworks.pop(i)   
        
        f.update()
        f.draw()

    for i in range(0, len(particles)):
        if i>=len(particles) - 1:
            break
        
        p = particles[i]
        p.update()
        p.draw()
    
    master.after(20, engine)

engine()
    
        
        
