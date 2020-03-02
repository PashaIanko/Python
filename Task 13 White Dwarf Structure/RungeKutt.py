

def calc_next_RK(y, x, dx, function_arr):
    y_const = list(y)
    res_y = []

    for i in range (0, len(y)):
        k1 = dx*function_arr[i](x, y_const)

        #увеличиваем каждое значение массива y (значения функций в СДУ) на k1/2
        
        y_const[i] = y_const[i]+k1/2

        k2 = dx*function_arr[i](x+dx/2,y_const)
        y_const = list(y)
        
        
        y_const[i] = y_const[i] + k2/2

        k3 = dx*function_arr[i](x+dx/2,y_const)
        y_const = list(y)

        y_const[i] = y_const[i] + k3
        
        k4 = dx*function_arr[i](x+dx,y_const)
        y_const = list(y)
        
        res = y_const[i] + (k1+2*(k2+k3)+k4)/6
        res_y.append(res)

    return res_y#y

def RungeKutt(y_init_cond, x_from, x_to, dx, func_arr):
    x = x_from
    res_x = [x_from]
    res_y = [[y_init_cond[0]], [y_init_cond[1]]]#, [y_init_cond[2]], [y_init_cond[3]]]
    y = list(y_init_cond)

    y_len = len(y)
    while x+dx<x_to:

        y = calc_next_RK(y, x, dx, func_arr)

        for i in range(0, y_len):
            res_y[i].append(y[i])
        x+=dx
        res_x.append(x)

    return [res_x, res_y]
