import turtle
import random as rand
from random import seed
from random import random

turtle_speed = 3;
turtles_numb = 10;

screen_width = 500;
screen_height = 500;
screen = turtle.Screen();
screen.setup(screen_width, screen_height);



turtles = [];
for _ in range(turtles_numb):
    turtles.append(turtle.Turtle());

offset = 10;
y_from = -screen_height/2;
y_to = -y_from;
x_from = -screen_width/2;
x_to = -x_from;

for i in range(turtles_numb):
    x_rand = rand.randrange(x_from, x_to);
    y_rand = rand.randrange(y_from, y_to);
    print(x_rand, y_rand);
    turtles[i].penup();
    turtles[i].setpos(x_rand, y_rand);

