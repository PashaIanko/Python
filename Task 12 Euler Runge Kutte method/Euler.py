
def calc_next(y, x, dx, func_arr):
    len_y = len(y)
    len_f_arr = len(func_arr)
    if len_y == len_f_arr:
        for i in range (0, len_y):
            y[i] = y[i]+dx*func_arr[i](x,y)
    
    return y

def Euler(y_init_cond, x_from, x_to, dx, functions_arr):
    x = x_from
    y = list(y_init_cond)
    res_x = [x_from]
    res_y = [[y_init_cond[0]],[y_init_cond[1]],[y_init_cond[2]],[y_init_cond[3]]]

    while x<x_to:
    #print(y)
        y = calc_next(y, x, dx, functions_arr)

        for i in range (0, len(y)):
            res_y[i].append(y[i])
        x+=dx
        res_x.append(x)
    return [res_x, res_y]
