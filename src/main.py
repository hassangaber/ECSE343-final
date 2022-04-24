import math_tools
import newton
import gd
import utils


if __name__=='__main__':

    print("Global Minima Problem Experiments... \n")

    print("#### 1 Objective function graphs: \n")

    try:
        math_tools.main()
    except AttributeError:
        print("Math utilities graphed correctly. Exiting.")

    print("#### 2 Newton method optimization example experiments... \n")

    try:
        newton.main()
    except AttributeError:
        print("Newton methods converged correctly Exiting.")

    print("#### 3 Gradient descent method optimization example experiments... \n")

    try:
        gd.main()
    except AttributeError:
        print("GD methods converged correctly Exiting.")

    print("#### 4 Hyperparamter variation... \n")

    try:
        utils.main()
    except AttributeError:
        print("Testing finished... \n")

    print("J.Trippitelli, H.Gaber Winter 2022, Numerical Methods Final Project; Problem #3")
    print("Thank you.")
