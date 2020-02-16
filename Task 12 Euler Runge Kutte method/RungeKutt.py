

def calc_next_RK(y, x, dx, function_arr):
   
    for i in range (0, len(y)):
        k1 = dx*function_arr[i](x, y)
        y_k1 = [y_+k1/2 for y_ in y]
        k2 = dx*function_arr[i](x+dx/2,y_k1)
        y_k2 = [y_+k2/2 for y_ in y]
        k3 = dx*function_arr[i](x+dx/2,y_k2)
        y_k3 = [y_+k3 for y_ in y]
        k4 = dx*function_arr[i](x+dx,y_k3)
        y[i] += (k1+2*(k2+k3)+k4)/6
        
    return y

def RungeKutt(y_init_cond, x_from, x_to, dx, func_arr):
    x = x_from
    res_x = [x_from]
    res_y = [[y_init_cond[0]], [y_init_cond[1]], [y_init_cond[2]], [y_init_cond[3]]]
    y = list(y_init_cond)

    y_len = len(y)
    while x<x_to:
        
        y = calc_next_RK(y, x, dx, func_arr)

        for i in range(0, y_len):
            res_y[i].append(y[i])
        x+=dx
        res_x.append(x)

    return [res_x, res_y]
