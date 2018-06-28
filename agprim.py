from autograd import numpy as np
from autograd import elementwise_grad as egrad
from autograd.extend import primitive, defvjp

@primitive
def f(x):
    return x**2 + 3*x + 5

def f_vjp(ans,x):
    return lambda g: g * (2*x + 3)

defvjp(f,f_vjp)

df = egrad(f)

x = np.float_([[3,2],[4,5]])

fx = f(x)
dfx = df(x)
