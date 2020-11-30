import math
def accelerated_gradient_descent(linOp, im, distance_function, reg_function, x0, NIter, acceleration = 0):
    """Function that solve in few iterations :
            argmin_x ( distance_function(linOp(x),im) + reg_function(x))

    Args:
        linOp (Function): A linear operator
        im (Array): target output.
        distance_function (Function): a function that implement the euclidean distance. Its gradient must be identity.
                                      For numpy or cupy array, distance_function = lambda x : xp.linalg.norm(x)**2/2 
        reg_function (dict): Regularization function. the field "eval" is the function evaluation. the field "prox_operator",
                             is its proximal operator defined by :
                             prox(x0, tau) = argmin_x = 1/2 ||x - xo|| + tau * reg_function(x)
        x0 (same type as input of linOP) : initial value of the iterative algorithm. Its type must implement + , -, /, * by a scalar and by an
                                           other element of the same type.
        NIter (Integer) : number of iteration
        acceleration (float, optional): for change the default gradient step. Defaults to 0.

    """
    L = linOp["norm"] * ( 1 - acceleration )
    linOp_T = linOp["transpose_operator"]
    linOp   = linOp["operator"]
    gradient = lambda x: linOp_T(linOp(x) - im) 
    cost_function_distance = lambda x : distance_function(linOp(x) - im)
    cost_function_reg      = reg_function["eval"]
    prox_tau_psi = reg_function["prox_operator"]
    t = 2/L
    x = x0
    g = x0 * 0
    A = 0
    for iter in range(NIter):
        a = ( t + math.sqrt(t**2 + 4*t * A))/2
        v = prox_tau_psi(x0 - g, A)
        v = (A * x + a * v) / (A+a)
        x = prox_tau_psi(v - gradient(v)/L, 1/L)
        g = g + a * gradient(v)
        A = A + a 
        if iter%10 == 0:
            cfr = cost_function_reg(x)
            cfd = cost_function_distance(x)
            cf  = cfr + cfd
            spaces = " " * len("     ==> CF = " + str(cf) +" _")
            print( 
                "iter = ", iter, "\n",
                "     ==> CF = ",  cf ,
                "_|| sparsity = ", cfr , "\n", spaces,
                "|| distance = ", cfd , "\n"
            )
    return x