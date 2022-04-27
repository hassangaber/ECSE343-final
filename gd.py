"""
Johan Trippitelli
Hassan Gaber 260891600
~/ECSE343-final/src/gd.py
    This file includes gradient descent with and without momentum coefficients
    
    *Libraries used: Numpy
"""
from math_tools import *

import numpy as np
import timeit

def gradient_descent_1(func,N,x_init,alpha=1e-3,epsilon=1e-8,max_iter=10e4):
  """
  param 1: objective function
  param 2: dimension
  param 3: initial guess (starting point)
  param 4: learning rate
  param 5: tolerance
  param 6: maximum iterations

  returns: Function global minima coordinates
  """

  # lambda expression for function gradinet
  G=ND_Gradient(func,N)
  x_now=x_init
  t:int=0

  while True:

    x_next=x_now-alpha*np.transpose(G(*x_now))

    if abs(list(np.array(x_next)-np.array(x_now))[0])<epsilon or t>max_iter:
      print("GD iterations: ",t)
      return x_now

    x_now = x_next
    t+=1


def gradient_descent_2(func,N,x_init,beta,alpha=1e-3,epsilon=1e-8,max_iter=10e4):
  """
  param 1: objective function
  param 2: dimension
  param 3: initial guess (starting point)
  param 4: momentum
  param 5: learning rate
  param 6: tolerance
  param 7: maximum iterations

  returns: Function global minima coordinates
  """

  # lambda expression for function gradinet
  G=ND_Gradient(func,N)
  x_now=x_init
  g=np.inf
  t:int=0
  grad_=[]

  while True:

    if t<2: 
      g=np.array(G(*x_now))

    else:
      g=(beta*np.array(grad_[t-1]))+((1-beta)*np.array(G(*x_now)))
      
    x_next=x_now-alpha*g
    grad_.append(g)

    #print(f'g {g}.   xk+1 {x_next}')

    if abs(list(np.array(x_next)-np.array(x_now))[0])<epsilon or t>max_iter:
      print("GD iterations: ",t)
      return x_now

    x_now = x_next
    t+=1


def test():
  
  if TESTING:
    '''print("No momentum GD: \n")

    guess=1.7*np.ones(2)
    print(gradient_descent_1(rosenbrock_2d,2,guess))
    print(gradient_descent_1(himmelblau,2,guess))

    guess_=2*np.ones(6)
    print(gradient_descent_1(rosenbrock_6d,6,guess_))'''

    print("GD with momentum: \n")

    
    rates=[5e-4,1e-4,1e-3,5e-3,0.002]
    momentum=[0.3,0.5,0.7,0.99]

    times_1=[]
    times_2=[]

    for r in rates:
      for m in momentum:
        print(f'momentum {m} lr {r}')

        guess=0.9*np.ones(2)

        start=timeit.default_timer()
        print(gradient_descent_2(rosenbrock_2d,2,guess,m,alpha=r))
        end=timeit.default_timer()
        print("runtime A: ", str(end-start))
        times_1.append(str(end-start))

        guess=1.1*np.ones(2)

        start=timeit.default_timer()
        print(gradient_descent_2(himmelblau,2,guess,m,alpha=r))
        end=timeit.default_timer()
        print("runtime B: ", str(end-start))
        times_2.append(str(end-start))
      
    print(min(times_1))
    print(min(times_2))

def final_test():
  print("GRADIENT DESCENT, NO MOMENTUM: \n")

  lr=[0.0001,0.005]
  m=[0.99,0.30]
  print(f'Rosenbrock hyperparameters: learning rate {lr[0]} momentum {m[0]}, same convergence parameters as Newton.')
  print(f'Himmelblau hyperparameters: learning rate {lr[1]} momentum {m[1]}.') 

  guess_r=0.7*np.ones(2)
  guess_h=2*np.ones(2)
  guess_rn=1.8*np.ones(6)

  start=timeit.default_timer()
  print("2D Rosenbrock Global Minima: ",gradient_descent_1(rosenbrock_2d,2,guess_r,alpha=lr[0]))
  end=timeit.default_timer()
  print("runtime A: ", str(end-start),"\n")

  start=timeit.default_timer()
  print("Himmelblau Global Minima: ", gradient_descent_1(himmelblau,2,guess_h,alpha=lr[1]))
  end=timeit.default_timer()
  print("runtime B: ", str(end-start))

  start=timeit.default_timer()
  print("6D Rosenbrock Global Minima: ",gradient_descent_1(rosenbrock_6d,6,guess_rn,alpha=lr[0]))
  end=timeit.default_timer()
  print("runtime C: ", str(end-start))


  print("GRADIENT DESCENT, WITH MOMENTUM: \n")

  start=timeit.default_timer()
  print("2D Rosenbrock Global Minima: ",gradient_descent_2(rosenbrock_2d,2,guess_r,alpha=lr[0],beta=m[0]))
  end=timeit.default_timer()
  print("runtime A: ", str(end-start),"\n")

  start=timeit.default_timer()
  print("Himmelblau Global Minima: ", gradient_descent_2(himmelblau,2,guess_h,alpha=lr[1],beta=m[1]))
  end=timeit.default_timer()
  print("runtime B: ", str(end-start))

  start=timeit.default_timer()
  print("6D Rosenbrock Global Minima: ",gradient_descent_2(rosenbrock_6d,6,guess_rn,alpha=lr[0],beta=m[0]))
  end=timeit.default_timer()
  print("runtime C: ", str(end-start))





if __name__=='__main__':
    final_test()