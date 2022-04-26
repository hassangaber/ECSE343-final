import src.math_tools
import src.newton  
import src.gd 
import src.utils
import src.experiments

def baseline_tests()->None:

    if not src.math_tools.TESTING:
        print('Error: Please set the TESTING global variable inside ~/src/math_tools.py to True.')
        exit()
   
    print("Global Minima Problem Experiments... \n")

    print("#### 1 Objective function graphs: \n")


    try:
        src.math_tools.test()
        
    except AttributeError:
        print("Math utilities graphed correctly.")

    print("#### 2 Newton method optimization example experiments... \n")

    try:
        src.newton.test()
    except AttributeError:
        print("Newton methods converged correctly.")

    print("#### 3 Gradient descent method optimization example experiments... \n")

    try:
        src.gd.test()
    except AttributeError:
        print("GD methods converged correctly.")

    print("#### 4 Hyperparamter variation... \n")

    try:
        src.utils.test()
    except AttributeError:
        print("Hyperparameter testing finished... \n")

if __name__=='__main__':

    src.experiments.test_zero()
    
    print("J.Trippitelli, H.Gaber Winter 2022, Numerical Methods Final Project; Problem #3")
