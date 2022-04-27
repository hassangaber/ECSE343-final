from math_tools import *
from utils import optimize
from newton import *

import sys

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

        # FINAL INPUT PARAMETERS
        optim_funcs=['Newton_1','Newton_2','Newton_3','gradient_descent_1','gradient_descent_2']
        funcs=[rosenbrock_2d,rosenbrock_4d,rosenbrock_6d,himmelblau,himmelblau]
        N=[2,4,6,2,2]
        guesses=[0.9*np.ones(2),0.9*np.ones(4),0.9*np.ones(6),7*np.ones(2),1.1*np.ones(2)]

        sys.stdout=open(file_paths[0],"w")
        for optims in optim_funcs:
            for f,n,g in zip(funcs,N,guesses):
                print(f'Guess {g};  Dimension {n};   Objective function {f};    Optimization method {optims}')
                print("Solution: ",optimize(f,g,n,optims),"\n")
    
def hyper_test(cond:bool=True):
    optim_funcs=['Newton_1','Newton_2','Newton_3','gradient_descent_1','gradient_descent_2']
    funcs=[rosenbrock_2d,rosenbrock_4d,rosenbrock_6d,himmelblau,himmelblau]
    N=[2,4,6,2,2]
    guesses=[0.9*np.ones(2),0.9*np.ones(4),0.9*np.ones(6),7*np.ones(2),1.1*np.ones(2)]

    max_iters=[10e3,8e3,5e3,15e3]
    epsilons=[10e-6,5e-6,1e-5]
    for e in epsilons:
        print("########### Epsilon: " , e)
        if TESTING:
            # Testing Newton 1
            print("NEWTON 1 TESTING: \n")
            
            # vector with elements {9,9} as a starting point
            guess=9*np.ones(2)
            print("Himmelblau Global Minima: ",Newton_1(himmelblau,2,guess,epsilon=e))
            guess=7*np.ones(2)
            print("2D Rosenbrock Global Minima: ",Newton_1(rosenbrock_2d,2,guess,epsilon=e))
            

            # Testing Newton_2
            print("NEWTON 2 TESTING: \n")
            guess=9*np.ones(2)
            gammas=[0.1,0.3,0.5,0.75,0.8,0.95]
            g=0.95
            #for g in gammas:
            print("gamma ",g)
            start=timeit.default_timer()
            print("Himmelblau Global Minima: ",Newton_2(himmelblau,2,guess,g,epsilon=e))
            end=timeit.default_timer()
            print("runtime A", str(end-start))

            start=timeit.default_timer()
            print("2D Rosenbrock Global Minima: ",Newton_2(rosenbrock_2d,2,guess,g,epsilon=e))
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
    
    

def test_one(cond:bool=True)->None:
    print("+++++++++++ TEST 1 Gradient/Hessian Convergence Observations & Conditions +++++++++++")
    print('These experiments help us reach conclusions about convergence properties of functions/methods/dimensions.')
    print("results will be saved to a .txt file 'test_1.txt' inside ~/ECSE343_final/results/\n")
    pass

if __name__=='__main__':
    test_zero()