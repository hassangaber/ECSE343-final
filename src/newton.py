"""
Johan Trippitelli
Hassan Gaber 260891600
~/ECSE343-final/src/newton.py
    This file includes all three variations of the newton iterarative optimization method
    Libraries used: Sympy,
                    Numpy,
                    timit (performance metrics).
"""
from math_tools import *

import numpy as np
import sympy
import timeit

testing:bool=True

# Newton's method for optimization implemented with no line search nor damping
# Need to find hessian and gradient using sympy
def Newton_1(func,N,xinit,gamma:float=1.0,max_iter:int=10e3,epsilon:float=10e-6):
    #define initial guess given by the x parameter
    x_now = xinit
    
    # define the gradient and hessian of the given function
    G=ND_Gradient(func,N)
    H=ND_Hessian(func,N)
    
    #perform the iterative portion of Newton Optimization
    for _ in range(int(max_iter)):

        x_next=x_now-(np.dot(gamma*np.linalg.inv(H(*x_now)),G(*x_now)))
        
        if (x_next - x_now).all() < epsilon:
            return x_now
        else:
            x_now = x_next
            
    return x_now

# Newton's method with damping
# Use Newton 1 function with non-trivial gamma value
def Newton_2(func,N,xinit,gamma,max_iter:int=10e3,epsilon:float=10e-6):
    #assert gamma < 1, "Use Newton_1 instead"
    _x = Newton_1(func,N,xinit,gamma,max_iter,epsilon)
    return _x

def lsearch(func,N,xinit):

  # step direction
  G=ND_Gradient(func,N)
  p=np.array(G(*xinit))

  alpha=0.1     # 0 < alpha < 0.5
  beta=0.75      # 0 < beta  < 1.0
  t=1.0         # step-size

  # create a variable list to lambdify function
  var_list = []
  for i in range(1,N+1):
      var_list.append("x"+str(i))
  var_list = sympy.symbols(var_list)

  f=sympy.lambdify(var_list,func)

  while (f(*xinit)-(f(*xinit-t*p)+np.dot(t*alpha,np.dot(p,p)))) < 0:
    t*=beta

  return t

# Newton's method: Line Search
# Implement using a backtracking line search
# use the alpha value found as gamma in the Newton Method
def Newton_3(func,N,xinit,max_iter:int=10e3,epsilon:float=10e-6):    
    alpha=lsearch(func,N,xinit)
    _x = Newton_2(func,N,xinit,alpha,max_iter,epsilon)
    return _x

if testing:
    # Testing Newton 1
    print("NEWTON 1 TESTING: \n")
    # vector with elements {9,9} as a starting point
    guess=9*np.ones(2)
    print("Himmelblau Global Minima: ",Newton_1(himmelblau,2,guess))
    guess=7*np.ones(2)
    print("2D Rosenbrock Global Minima: ",Newton_1(rosenbrock_2d,2,guess))


    # Testing Newton_2
    print("NEWTON 2 TESTING: \n")
    guess=9*np.ones(2)
    gammas=[0.1,0.3,0.5,0.75,0.8,0.95]
    for g in gammas:
        print("gamma ",g)
        start=timeit.default_timer()
        print("Himmelblau Global Minima: ",Newton_2(himmelblau,2,guess,g))
        end=timeit.default_timer()
        print("Function runtime ", str(end-start))

        start=timeit.default_timer()
        print("2D Rosenbrock Global Minima: ",Newton_2(rosenbrock_2d,2,guess,g))
        end=timeit.default_timer()
        print("Function runtime ", str(end-start),"\n")
    
    print("NEWTON 3 TESTING: \n")

    g_form=lambda x: float(x) * np.ones(2)

    guesses=[g_form(0.1),g_form(0.25),g_form(1.0),g_form(1.9),g_form(2.1),g_form(3),g_form(5),g_form(10)]

    for guess in guesses:
        xnow=guess

        print(f'Guess {xnow}, Norm of guess {np.linalg.norm(xnow)}')

        print("Rosenrock:")
        start=timeit.default_timer()
        print(Newton_3(rosenbrock_2d,2,xnow))
        end=timeit.default_timer()
        print(str(end-start),"\n")

        print("Himmelblau:")
        start=timeit.default_timer()
        print(Newton_3(himmelblau,2,xnow))
        end=timeit.default_timer()
        print(str(end-start),"\n")

    