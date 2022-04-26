'''
Johan Trippitelli
Hassan Gaber 260891600
~/ECSE343-final/src/mathtools.py
    This python script organizes all vector computations.
    
    *Libraries used: Sympy (math expressions and computation),
                    Numpy,
                    Matplotlib
'''
import sympy
import numpy as np
from matplotlib import pyplot as plt

global TESTING
TESTING=True

# The Himmelblau Function
def Himmelblau(x0:float,x1:float)->float:
    return (x0**2+x1-11)**2 + ((x0+x1**2-7)**2)

# The 2D Rosenbrock function
def Rosenbrock2D(x0:float,x1:float,b:int=10)->float:
    return ((x0-1)**2 + b*(x1-x0**2)**2)

# 2-dimensional case
x1, x2 = sympy.symbols('x1 x2')

# rosenbrock 2D
rosenbrock_2d = (1-x1)**2 + 100*(x2-x1**2)**2

# himmelblau
himmelblau = (x1**2+x2-11)**2 + (x1+x2**2-7)**2

# n-dimensional case
x1, x2, x3, x4, x5, x6 = sympy.symbols('x1 x2 x3 x4 x5 x6')

# N=4
rosenbrock_4d=((100*(x1**2-x2)**2+(x1-1)**2)+(100*(x3**2-x4)**2+(x3-1)**2))

# N=6
rosenbrock_6d=((100*(x1**2-x2)**2+(x1-1)**2)+(100*(x3**2-x4)**2+(x3-1)**2)+(100*(x5**2-x6)**2+(x5-1)**2))

# Generate the correct number of symbols as variables xn
def gen_symbols(N):
    var_list = []
    for i in range(1,N+1): var_list.append("x"+str(i))
    var_list = sympy.symbols(var_list)
    return var_list

# N-dimensional gradient vector operator
def ND_Gradient(func,N:int):
    assert N%2==0, "Invalid N"
    var_list = gen_symbols(N)
    return sympy.lambdify(var_list,sympy.derive_by_array(func,var_list))

# N-dimensional hessian matrix
def ND_Hessian(func,N):
    assert N%2==0, "Invalid N"
    var_list = gen_symbols(N)
    return sympy.lambdify(var_list,sympy.hessian(func,var_list))

def test():
    if TESTING:
        # https://www.indusmic.com/post/himmelblau-function
        X=np.linspace(-5,5)
        Y=np.linspace(-5,5)
        x,y=np.meshgrid(X,Y)
        F=Himmelblau(x,y)

        fig =plt.figure(figsize=(9,9))
        ax=plt.axes(projection='3d')
        ax.contour3D(x,y,F,450)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('F(X,Y)')
        ax.set_title('Himmelblau Function')
        ax.view_init(50,50)

        plt.show()

        X=np.linspace(-5,5)
        Y=np.linspace(-5,5)
        x,y=np.meshgrid(X,Y)
        F=Rosenbrock2D(x,y)

        fig =plt.figure(figsize=(9,9))
        ax=plt.axes(projection='3d')
        ax.contour3D(x,y,F,450)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('F(X,Y)')
        ax.set_title('Rosenbrock Function')
        ax.view_init(50,50)

        plt.show()

if __name__=='__main__':
    test()
