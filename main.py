import math_tools
import newton  
import gd 
import utils

import numpy as np


def baseline_tests()->None:

   
    print("Global Minima Problem Experiments... \n")

    print("#### 1 Objective function graphs: \n")


    try:
        math_tools.test()
        
    except AttributeError:
        print("Math utilities graphed correctly.")

    print("#### 2 Newton method optimization example experiments... \n")

    try:
        newton.test()
    except AttributeError:
        print("Newton methods converged correctly.")

    print("#### 3 Gradient descent method optimization example experiments... \n")

    try:
        gd.test()
    except AttributeError:
        print("GD methods converged correctly.")

    print("#### 4 Hyperparamter variation... \n")

    try:
        utils.test()
    except AttributeError:
        print("Hyperparameter testing finished... \n")

if __name__=='__main__':

    newton.final_newton()
    gd.final_test()

    print("J.Trippitelli, H.Gaber Winter 2022, Numerical Methods Final Project; Problem #3")
