import turtle
import random as rand
from random import seed
from random import random
from random import randint

turtle_speed = 50;
turtles_numb = 3;
dt = 1;

screen_width = 500;
screen_height = 500;
screen = turtle.Screen();
screen.setup(screen_width, screen_height);
screen.colormode(255);



turtles = [];
for _ in range(turtles_numb):
    turtles.append(turtle.Turtle());

y_from = -screen_height/2;
y_to = -y_from;
x_from = -screen_width/2;
x_to = -x_from;

for i in range(turtles_numb):
    x_rand = rand.randrange(x_from, x_to);
    y_rand = rand.randrange(y_from, y_to);
    turtles[i].color((randint(0,255), randint(0,255), randint(0,255)));
    turtles[i].penup();
    turtles[i].setpos(x_rand, y_rand);
    turtles[i].pendown();

#turning the first turtle at an arbitrary angle
turtles[0].right(randint(0, 180));

t = 0;
turtles[0].forward(turtle_speed * dt);


angle = turtles[1].towards(turtles[0]);
turtles[1].setheading(angle);


