import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from math import pi as PI
import math
import cmath
import numpy as np
from scipy import signal
import random as rand
import tkinter as tk
import time



from tkinter import*

def sin(x):
    for i in range(0, len(x)):
        x[i]=math.sin(x[i])
    return x

class Example(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        self.N = 100 #number of sample points
        self.W = 100 #omega
        self.From = -10
        self.To = 10
        x = []
        y = []
        #self.button = tk.Button(self, text="Press me!")
        #self.text = tk.Text(self, width=40, height=6)
        #self.vsb = tk.Scrollbar(self, command=self.text.yview)
        #self.text.configure(yscrollcommand=self.vsb.set)

        #self.button.pack(side="top")
        #self.vsb.pack(side="right", fill="y")
        #self.text.pack(side="bottom", fill="x")

        self.Nscale = tk.Scale(self, label='Sample points', orient=HORIZONTAL ,length=400,from_=0,to=1000, resolution=10)
        self.Nscale.bind("<ButtonRelease>", self.assign_N)
        self.Nscale.pack()
        self.Nscale.set(self.N)

        self.Wscale = tk.Scale(self, label='Omega', orient=HORIZONTAL ,length=400,from_=0,to=100, resolution=2)
        self.Wscale.bind("<ButtonRelease>", self.assign_W)
        self.Wscale.pack()
        self.Wscale.set(self.W)

        self.Fromscale = tk.Scale(self, label='From', orient=HORIZONTAL ,length=400,from_=-100,to=0, resolution=1.5)
        self.Fromscale.bind("<ButtonRelease>", self.assign_From)
        self.Fromscale.pack()
        self.Fromscale.set(self.From)

        self.Toscale = tk.Scale(self, label='To', orient=HORIZONTAL ,length=400,from_=0,to=100, resolution=1.5)
        self.Toscale.bind("<ButtonRelease>", self.assign_To)
        self.Toscale.pack()
        self.Toscale.set(self.To)

        self.figure = plt.figure()
        plt.ion()
        
        
        self.func = sin
        
        #self.button.bind("<ButtonPress>", self.on_press)
        #self.button.bind("<ButtonRelease>", self.on_release)

    def update(self):
        
        self.figure.clf()
        print("from", self.From, "to", self.To, "N", self.N)
        x = np.linspace(self.From, self.To, self.N)
                    
        y = np.sin(self.W*x)
        
        subplot = self.figure.add_subplot(111)
        subplot.set_xlim(self.From, self.To)
        subplot.plot(x, y, 'b-')
        plt.show()
        

    def assign_N(self, event):
        self.N = int(self.Nscale.get())
        print('n=', self.N, 'w=', self.W)
        self.update()
        
    def assign_W(self, event):
        self.W = int(self.Wscale.get())
        print('n=', self.N, 'w=', self.W)
        self.update()

    def assign_From(self, event):
        self.From = float(self.Fromscale.get())
        print('from=', self.From, 'to=', self.To)
        self.update()

    def assign_To(self, event):
        self.To = float(self.Toscale.get())
        print('from=', self.From, 'to=', self.To)
        self.update()
    
    def calc(self, function):
        x = np.linspace(-np.pi, np.pi, 10)
        y = function(x)
        print("x=\n",x, "y=\n", y)

    

root = tk.Tk()
Example(root).pack(side="top", fill="both", expand=True)
root.mainloop()
  
'''def get_val_command(s1):
    print("Command Number",str(s1))
  
def get_val_bind(event):
    s1 = scal.get()
    print("Get Number",str(s1))
  
def get_val_motion(event):
    s1 = scal.get()
    print("Motion Number",str(s1))
  
root = Tk()'''
  
'''# 1 способ
scal = Scale(root,orient=VERTICAL ,length=300,from_=0,to=100,tickinterval=10,resolution=10)
scal.bind("<B1-Motion>",get_val_motion)
scal.pack()'''
  
'''# 2 способ
but1 = Button(text="Get through bind")
but1.bind("<Button-1>",get_val_bind)
but1.pack()
  
# 3 способ
but2 = Button(text="Get through command", command=lambda:get_val_command(scal.get()))
but2.pack()
  
root.mainloop()'''


