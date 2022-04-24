"""
Johan Trippitelli
Hassan Gaber 260891600
~/ECSE343-final/src/utils.py
    Utility methods for other files
    Libraries used: Sympy,
                    Numpy,
                    timit (performance metrics).
"""
from src.newton import *
from src.gd import *
import numpy as np

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
