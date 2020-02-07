from math import pi as PI
import math
import cmath



def dft (data):
    data_size = len(data)
    result = []
    if data_size:
        
        pi_div_N_mult_two = 2*PI/data_size #optimization
        for k in range (0, data_size):
            x_k_real = 0; #result
            x_k_im = 0; #result
            
            for n in range (0, data_size):
                x_n = data[n]
                arg = k*n*pi_div_N_mult_two
                x_k_real += x_n * math.cos(arg)
                x_k_im -= x_n * math.sin(arg)
            
            result.append(complex(x_k_real, x_k_im))
    
    return result
