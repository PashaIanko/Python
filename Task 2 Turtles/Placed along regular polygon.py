import turtle
import random as rand
from random import seed
from random import random
from random import randint

turtle_speed = 45;
turtles_numb = 10;
dt = 1;

screen_width = 500;
screen_height = 500;
screen = turtle.Screen();
screen.setup(screen_width, screen_height);
screen.colormode(255);

def calc_angle(turtles, index_from, index_to):
    angle = turtles[index_from].towards(turtles[index_to]);
    #print(index_from, index_to, angle);
    return angle;

def set_directions(turtles, size):
    print("setting directions");
    for i in range(0, size):
        if(i!= size-1):
            angle = calc_angle(turtles, i, i+1);
        else:
            angle = calc_angle(turtles, i, 0);
        turtles[i].setheading(angle);    

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


#turning each turtle towards the next
set_directions(turtles, turtles_numb);


i = 0;
while 1:
    #define angle from i towards i+1
    if(i == turtles_numb - 1):
        angle = calc_angle(turtles, turtles_numb-1, 0);
        turtles[i].setheading(angle);
        turtles[i].forward(turtle_speed * dt);
        i=0;
    else:
        angle = calc_angle(turtles, i, i+1);
        turtles[i].setheading(angle);
        turtles[i].forward(turtle_speed * dt);
        i+=1;






