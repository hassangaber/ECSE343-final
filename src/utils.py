"""
Johan Trippitelli
Hassan Gaber 260891600
~/ECSE343-final/src/utils.py
    Utility methods for other files
    Libraries used: Sympy,
                    Numpy,
                    timit (performance metrics).
"""
from newton import *
from gd import *

import numpy as np


display_basic_experiments:bool=True

# automatic optimization
def optimize(func,guess,N,optim):
  momentum=[0.0,0.3,0.5,0.75,0.8,0.9,0.99]
  gamma=0.95
  if optim=='gradient_descent_2':
    optim=eval(optim)
    return optim(func,N,guess,momentum[6])
  elif optim=='Newton_2':
    optim=eval(optim)
    return optim(func,N,guess,gamma)
  else:
    optim=eval(optim)
    return optim(func,N,guess)

if display_basic_experiments:
    optim_funcs=['Newton_1','Newton_2','Newton_3','gradient_descent_1','gradient_descent_2']
    funcs=[rosenbrock_2d,rosenbrock_4d,rosenbrock_6d,himmelblau]
    N=[2,4,6,2]
    guesses=[0.5*np.ones(2),1.7*np.ones(4),0.9*np.ones(6),7.0*np.ones(2)]

    for optims in optim_funcs:
        for f,n,g in zip(funcs,N,guesses):
            print(f'Guess {g};  Dimension {n};   Objective function {f};    Optimization method {optims}')
            print("Solution: ",optimize(f,g,n,optims),"\n")