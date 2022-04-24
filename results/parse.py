import fileinput

filenames=['test_0.txt','test_1.txt']
j=0
string_to_be_replaced=str
string_in_place=str

with fileinput.FileInput(filenames[j], inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace(string_to_be_replaced,string_in_place),end='')


"""
print(line.replace('(x1 + x2**2 - 7)**2 + (x1**2 + x2 - 11)**2','Himmelblau'),end='')
print(line.replace('(1 - x1)**2 + 100*(-x1**2 + x2)**2', '2D Rosenbrock'), end='')
print(line.replace('(x1 - 1)**2 + 100*(x1**2 - x2)**2 + (x3 - 1)**2 + 100*(x3**2 - x4)**2', '4D Rosenbrock'), end='')
print(line.replace('(x1 - 1)**2 + 100*(x1**2 - x2)**2 + (x3 - 1)**2 + 100*(x3**2 - x4)**2 + (x5 - 1)**2 + 100*(x5**2 - x6)**2', '6D Rosenbrock'), end='')
print(line.replace('4D Rosenbrock + (x5 - 1)**2 + 100*(x5**2 - x6)**2', '6D Rosenbrock'), end='')
"""