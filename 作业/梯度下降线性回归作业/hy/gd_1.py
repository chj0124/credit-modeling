import numpy as np
def f(x):
	return x**2
def gradient(x):
	return 2 * x 
x0 = 10
cnt = 0
beta = 0.7
while gradient(x0)**2 > 0.0001:
	x0 -= beta * gradient(x0) 
	cnt += 1
	print(x0, f(x0))
print(x0, f(x0))