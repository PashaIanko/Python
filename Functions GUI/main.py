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

class Function :
    N = 100 #sample points
    W = 2 #omega
    From = 0
    Amplitude = 1
    To = 2
    x = []
    y = []
    
    def __init__(self):#, N=0, W=2, From=0, To=10, func=np.sin):
       self.func=np.sin



functions_numb=4
class GUI(tk.Frame):

    functions = []
    check_buttons_func = []
    check_buttons_states = []
    
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        for i in range(0, functions_numb):
            self.functions.append(Function())
        
        self.N = 100 #number of sample points
        self.W = 100 #omega
        self.From = -10
        self.To = 10
        x = []
        y = []
        
        self.Nscale = tk.Scale(self, label='Sample points', orient=HORIZONTAL ,length=400,from_=0,to=1000, resolution=10)
        self.Nscale.bind("<ButtonRelease>", self.assign_N)
        self.Nscale.pack()
        self.Nscale.set(self.N)

        self.Wscale = tk.Scale(self, label='Omega 1', orient=HORIZONTAL ,length=400,from_=0,to=100, resolution=2)
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

        self.Check = tk.Checkbutton(self, text="sin")
        self.Check.pack()

        self.figure = plt.figure()

        for i in range(0, functions_numb):
            var = IntVar()
            print(var)
            self.check_buttons_func.append(tk.Checkbutton(self, text=str(i), variable=var, command=lambda: self.ChangeFunc(i)))#command = self.ChangeFunc))
            self.check_buttons_func[i].pack()
            self.check_buttons_states.append(var)
            
        #plt.ion()
        
        
        self.func = sin


    def disable_other_checkboxes(self):
        v=5
        #for button in self.check_buttons_func:
            #button.config(state=DISABLED)
        
    def ChangeFunc(self, index):
        print("checkbox caught", index)
        print(type(index))
        print(type(self))
        #print(self_.event.var.get())
        self.disable_other_checkboxes()
        
        

    def update(self):
        
        self.figure.clf()
        x = np.linspace(self.From, self.To, self.N)
                    
        y = self.func(self.W*x)
        
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
GUI(root).pack(side="top", fill="both", expand=True)
root.mainloop()
  



'''from tkinter import *      
states = []
def onPress(i):                        
    states[i] = not states[i]          
     
root = Tk()
for i in range(10):
    chk = Checkbutton(root, text=str(i), command=(lambda i=i: onPress(i)) )
    chk.pack(side=LEFT)
    states.append(0)
root.mainloop()
print (states)'''


