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
from gd import *
import sympy
import timeit

testing:bool=True

# Newton's method for optimization implemented with no line search nor damping
# Need to find hessian and gradient using sympy
def Newton_1(func,N,xinit,gamma:float=1.0,max_iter:int=10e5,epsilon:float=10e-5):
    #define initial guess given by the x parameter
    x_now = xinit
    
    # define the gradient and hessian of the given function
    G=ND_Gradient(func,N)
    H=ND_Hessian(func,N)
    
    I=0
    #perform the iterative portion of Newton Optimization
    for i in range(int(max_iter)):
        I=i
        x_next=x_now-(np.dot(gamma*np.linalg.inv(H(*x_now)),G(*x_now)))
        
        if np.linalg.norm(x_next - x_now) < epsilon:
            print("Convergence iteration: ",I)
            return x_now
        else:
            x_now = x_next

            
    return x_now

# Newton's method with damping
# Use Newton 1 function with non-trivial gamma value
def Newton_2(func,N,xinit,gamma,max_iter:int=10e5,epsilon:float=10e-5):
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
def Newton_3(func,N,xinit,max_iter:int=10e5,epsilon:float=10e-5):    
    alpha=lsearch(func,N,xinit)
    _x = Newton_2(func,N,xinit,alpha,max_iter,epsilon)
    return _x

def test():
    #times_1=[]
    #times_2=[]
    g=0.95
    epsilons=[10e-6,5e-6,1e-5,1e-4,5e-4]
    for e in epsilons:
        print("########### Epsilon: " , e)
        if TESTING:
            # Testing Newton 1
            print("NEWTON 1: \n")
            
            # vector with elements {9,9} as a starting point
            guess=9*np.ones(2)
            print("Himmelblau Global Minima: ",Newton_1(himmelblau,2,guess,epsilon=e))
            guess=7*np.ones(2)
            print("2D Rosenbrock Global Minima: ",Newton_1(rosenbrock_2d,2,guess,epsilon=e))
            

            # Testing Newton_2
            print("NEWTON 2: \n")
            guess=1*np.ones(2)
            #gammas=[0.1,0.3,0.5,0.75,0.8,0.95]
            g=0.95
            #for g in gammas:
            #print("gamma ",g)
            start=timeit.default_timer()
            print("Himmelblau Global Minima: ",Newton_2(himmelblau,2,guess,g,epsilon=e))
            end=timeit.default_timer()
            print("runtime:", str(end-start),"\n")

            start=timeit.default_timer()
            print("2D Rosenbrock Global Minima: ",gradient_descent_1(rosenbrock_2d,2,guess,epsilon=e))
            end=timeit.default_timer()
            print("runtime B", str(end-start),"\n")
            
            print("NEWTON 3 TESTING: \n")

            #g_form=lambda x: float(x) * np.ones(2)

            #guesses=[g_form(0.1),g_form(0.25),g_form(1.0),g_form(1.9),g_form(2.1),g_form(3),g_form(5),g_form(10)]
            guess1=0.9*np.ones(2)
            guess2=1.1*np.ones(2)
            #for guess in guesses:
            #    xnow=guess

            #print(f'Guess {xnow}, Norm of guess {np.linalg.norm(xnow)}')

            print("Rosenrock:")
            start=timeit.default_timer()
            print(Newton_3(rosenbrock_2d,2,guess1,epsilon=e))
            end=timeit.default_timer()
            print("Time:",str(end-start),"\n")

            print("Himmelblau:")
            start=timeit.default_timer()
            print(Newton_3(himmelblau,2,guess2,epsilon=e))
            end=timeit.default_timer()
            print("Time:" ,str(end-start),"\n")


def final_newton():

    guess_r=0.7*np.ones(2)
    guess_h=5*np.ones(2)
    guess_rn=0.9*np.ones(6)

    print("NEWTON 1: \n")
    max_iter:int=10e5
    epsilon:float=10e-5

    print(f'*General hyperparameters: Maximum iterations {max_iter}, tolerance {epsilon}')     
    

    start=timeit.default_timer()
    print("Himmelblau Global Minima: ",Newton_1(himmelblau,2,guess_h))
    end=timeit.default_timer()
    print("runtime B", str(end-start),"\n")

    start=timeit.default_timer()
    print("2D Rosenbrock Global Minima: ",Newton_1(rosenbrock_2d,2,guess_r))
    end=timeit.default_timer()
    print("runtime B", str(end-start),"\n")

    start=timeit.default_timer()
    print("6D Rosenbrock Global Minima: ",Newton_1(rosenbrock_6d,6,guess_rn))
    end=timeit.default_timer()
    print("runtime C", str(end-start),"\n")


    # Testing Newton_2
    print("NEWTON 2: \n")
    print("*Hyperparameters: ")
    g=0.95
    print("Gamma: ",g,"\n")

    start=timeit.default_timer()
    print("Himmelblau Global Minima: ",Newton_2(himmelblau,2,guess_h,gamma=g))
    end=timeit.default_timer()
    print("runtime A:", str(end-start),"\n")

    start=timeit.default_timer()
    print("2D Rosenbrock Global Minima: ",Newton_2(rosenbrock_2d,2,guess_r,gamma=g))
    end=timeit.default_timer()
    print("runtime B", str(end-start),"\n")

    start=timeit.default_timer()
    print("6D Rosenbrock Global Minima: ",Newton_2(rosenbrock_6d,6,guess_rn,gamma=g))
    end=timeit.default_timer()
    print("runtime C", str(end-start),"\n")


    print("NEWTON 3: \n")    
    

    start=timeit.default_timer()
    print("Himmelblau Global Minima: ",Newton_3(himmelblau,2,guess_h))
    end=timeit.default_timer()
    print("runtime B", str(end-start),"\n")

    start=timeit.default_timer()
    print("2D Rosenbrock Global Minima: ",Newton_3(rosenbrock_2d,2,guess_r))
    end=timeit.default_timer()
    print("runtime B", str(end-start),"\n")

    start=timeit.default_timer()
    print("6D Rosenbrock Global Minima: ",Newton_3(rosenbrock_6d,6,guess_rn))
    end=timeit.default_timer()
    print("runtime C", str(end-start),"\n")


if __name__=='__main__':
    final_newton()
    
