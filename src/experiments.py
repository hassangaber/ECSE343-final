from src.math_tools import *
from src.utils import optimize
import sys
import timeit

file_paths = ['/Users/hassan/Desktop/ECSE343_final/results/test_0.txt',
              '/Users/hassan/Desktop/ECSE343_final/results/test_1.txt'
            ]



# Iterating through all our optimization methods, all of our functions, and some random guesses
# to find a good starting point to start exploring hyperparameter and hyperalgorithmic properties.
def test_zero(cond:bool=True)->None:
    if cond:

        print("+++++++++++ TEST 0 +++++++++++")
        print('This is a basic algorithm and parameter variations to find optimial preformance points of our methods.')
        print("Saved to a .txt file 'test_0.txt' inside ~/ECSE343_final/results/ \n")

        optim_funcs=['Newton_1','Newton_2','Newton_3','gradient_descent_1','gradient_descent_2']
        funcs=[rosenbrock_2d,rosenbrock_4d,rosenbrock_6d,himmelblau]
        N=[2,4,6,2]
        guesses=[0.5*np.ones(2),1.7*np.ones(4),0.9*np.ones(6),7.0*np.ones(2)]

        sys.stdout=open(file_paths[0],"w")

        for optims in optim_funcs:
            for f,n,g in zip(funcs,N,guesses):
                print(f'Guess {g};  Dimension {n};   Objective function {f};    Optimization method {optims}')
                print("Solution: ",optimize(f,g,n,optims),"\n")
    
def test_one(cond:bool=True)->None:
    print("+++++++++++ TEST 1 Gradient/Hessian Convergence Observations & Conditions +++++++++++")
    print('These experiments help us reach conclusions about convergence properties of functions/methods/dimensions.')
    print("results will be saved to a .txt file 'test_1.txt' inside ~/ECSE343_final/results/\n")
    pass

if __name__=='__main__':
    test_zero()