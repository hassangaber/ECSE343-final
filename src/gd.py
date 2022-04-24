"""
Johan Trippitelli
Hassan Gaber 260891600
~/ECSE343-final/src/gd.py
    This file includes gradient descent with and without momentum coefficients
    Libraries used: Numpy,
"""
from src.math_tools import *

import numpy as np

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
      return x_now

    x_now = x_next
    t+=1


def test():
  
  if TESTING:
    print("No momentum GD: \n")

    guess=1.7*np.ones(2)
    print(gradient_descent_1(rosenbrock_2d,2,guess))
    print(gradient_descent_1(himmelblau,2,guess))

    guess_=2*np.ones(6)
    print(gradient_descent_1(rosenbrock_6d,6,guess_))

    print("GD with momentum: \n")

    guess=[0.0072,0.1]
    print(gradient_descent_2(rosenbrock_2d,2,guess,0.001))
    print(gradient_descent_2(rosenbrock_2d,2,guess,0.3))
    print(gradient_descent_2(rosenbrock_2d,2,guess,0.5))
    print(gradient_descent_2(rosenbrock_2d,2,guess,0.7))
    print(gradient_descent_2(rosenbrock_2d,2,guess,0.99))


    guess=[1,0.3]
    print(gradient_descent_2(himmelblau,2,guess,0.001))
    print(gradient_descent_2(himmelblau,2,guess,0.5))
    print(gradient_descent_2(himmelblau,2,guess,0.85))

if __name__=='__main__':
    test()