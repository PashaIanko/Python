import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import re
from mpl_toolkits.mplot3d import Axes3D


def add_coordinates(coord_arr, parsed_str):
    
    res_coordinates = []
    for i in range(len(parsed_str)):
        item = parsed_str[i]
       
        if len (item) and item != '\n':
            number = float(item)
            res_coordinates.append(number)
    coord_arr.append(res_coordinates)         
            
            
            
def parse_energies(res_arr, lines_arr):
    for i in range(len(lines_arr)):
        res_arr[0].append(i)
        res_arr[1].append(float(lines_arr[i]))
    return
    

def main():
    file_coord = open("Distribution.txt", "r")
    lines = file_coord.readlines()
    file_coord.close()

   
    res_coordinates = []
    res_energies = [[], []]

    for line in lines:
        splitS = re.split(' ',line)
        add_coordinates(res_coordinates, splitS)

    file_energies = open("Energies.txt", "r")
    lines_energy = file_energies.readlines()
    
    file_energies.close()
    parse_energies(res_energies, lines_energy)
    
    fig = plt.figure()
    ax = fig.add_subplot(121, projection = '3d')
    for XYZ_set in res_coordinates:
        x_coord = XYZ_set[0]
        y_coord = XYZ_set[1]
        z_coord = XYZ_set[2]
        ax.scatter(x_coord, y_coord, z_coord, c = 'r')#, сolor = 'r', marker = 'o')
        ax.scatter
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    sp_energies = fig.add_subplot(122)
    sp_energies.set_ylim([-5, 0])
    sp_energies.plot(res_energies[0], res_energies[1], marker = 'o')
    plt.show()
main()
