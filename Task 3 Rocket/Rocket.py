import math
import random
from tkinter import *

win_height = 600
win_width = 600
gravity = 0.9
rocket_velocity = -16
particles_numb = 50
particles_velocity = 2
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
        self.V = particles_velocity
        velocity_angle = random.uniform(0, 2*3.14)
       
        self.vx = self.V * math.cos(velocity_angle)
        self.vy = self.V * math.sin(velocity_angle)
        self.vz = 0
        self.color = randomColor()
        self.unupdatable = False

    def update(self):
        if self.unupdatable == False:
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.z += self.vz * dt
            self.vy += gravity * dt
            
            if self.y >= win_height:
                self.y = win_height
                self.unupdatable = True

                  

    def draw(self):
        size = 2
        win.create_rectangle(self.x-size, self.y-size, self.x+size, self.y+size,
                             fill=self.color)
        

class Rocket:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.vx = 0
        self.vy = rocket_velocity
        self.vz = 0
        self.explodeTime = 10
        self.ready_explode = False
        

    def update(self):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        

        self.vy += gravity * dt
        self.explodeTime -= 1/60

               

        if self.vy >= 0:
            self.ready_explode = True

            
            
    def draw(self):
        win.create_rectangle(self.x-2, self.y-4, self.x+2, self.y+5,
                             fill="black")
        
particles = []
Rockets = []

def spawnRocket(event):
    Rockets.append(Rocket(win_width/2, win_height, 0))

win.bind("<Button-1>", spawnRocket)

def engine():
    win.delete("all")

    for i in range(0, len(Rockets)):
        if i>=len(Rockets) - 1:
            break
        
        f = Rockets[i]
        if f.ready_explode == True:
            for j in range (0, particles_numb):
                particles.append(Particle(f.x, f.y, f.z))
            Rockets.pop(i)   
        
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

"""from tkinter import Tk, Canvas, Frame, BOTH

win_height = 600
win_width = 600
ground_thickness = 10
rockets_numb = 1
gravity = 0.2
dt = 0.1

class Rocket:
    #canvas_to_draw = Canvas()
    height = 20
    width = 8
    def __init__(self, x, y, z, canvas):
        self.x = x
        self.y = y
        self.z = z
        self.vx = 0
        self.vy = -10
        self.vz = 0
        self.explodeTime = 10
        self.remove = False
        self.canvas_to_draw = canvas
        self.draw()

    def update(self):
        print("update")
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        print(self.x, self.y, self.z)
        self.vy += gravity * dt
        self.explodeTime -= 1/60

        #self.canvas_to_draw.create_rectangle(self.x, self.y, self.x+self.width, self.y-self.height,
        #                     fill="red")
        #if self.vy >= 0:
        #    self.remove = True

    def draw(self):
        self.canvas_to_draw.create_rectangle(self.x-2, self.y-4, self.x+2, self.y+5,
                             fill="red")       
            
    #def draw(self, x, y):
    #    self.canvas_to_draw.create_rectangle(x, y, x+self.width, y-self.height,
    #                         fill="red")

class Graphics(Frame):

    rockets = []
    rockets_numb
    
    def __init__(self, parent, rockets_numb):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
        self.rockets_numb = rockets_numb
        

    def set_rockets(self, rockets_numb):
        for i in range(0, rockets_numb):
            self.rockets.append(Rocket(10, 0, 0, self.canvas))
            print("drawing rocket")
            #self.rockets[i].draw(10, win_height-ground_thickness)

    def initUI(self):
        self.parent.title("Rocket")
        self.pack(fill = BOTH, expand = 1)

        self.canvas = Canvas(self, width=win_width, height=win_height)
        #ground
        self.canvas.create_rectangle(0, win_height, win_width, win_height-ground_thickness, fill = "black")
        self.canvas.pack(fill = BOTH, expand = 1)

        self.set_rockets(rockets_numb)

    


    def engine(self):
        print("engine", "len=", len(self.rockets))
        self.canvas.delete("all")

        for i in range(0, len(self.rockets)):
            print(i)
            r = self.rockets[i]
            #if f.remove == True:
            #    for j in range (0, 100):
            #        particles.append(Particle(f.x, f.y, f.z))
            #    fireworks.pop(i)   
        
            r.update()
            r.draw()
        self.parent.after(20, self.engine)
        
def main():
        
    root = Tk()
    GUI = Graphics(root, rockets_numb)
    GUI.bind("<Button-1>", GUI.engine())
    root.mainloop()


if __name__ == '__main__':
    main()"""
