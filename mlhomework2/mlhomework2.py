import numpy as np
import matplotlib.pyplot as plt

# part a
z = 0
slopes = []
intercepts = []
g = []
points = []
fig, (fig1, fig2) = plt.subplots(1, 2)
e_out = []

while z<100:
    
    x1 = np.random.uniform(low=-1.0)
    x2 = np.random.uniform(low=-1.0)

    y1 = np.sin(np.pi*x1)
    y2 = np.sin(np.pi*x2)

    slope = (y1 - y2)/(x1 - x2)
    slopes.append(slope)

    intercept = ((x1*y2)-(x2*y1))/(x1-x2)
    intercepts.append(intercept)
    
    fig2.plot(x1, y1, 'bo')
    fig2.plot(x2, y2, 'bo')
    x = np.arange(-1., 1., .01)
    g_Dx = (y1-y2)/(x1-x2)*x + (x1*y2-x2*y1)/(x1-x2)
    g.append(g_Dx)
    e_out.append(np.square(np.sin(np.pi * x) - g_Dx))
    fig1.plot(x, g_Dx, "b")
    z+=1
      
fig1.plot(x, np.sin(np.pi*x), "r")
fig2.plot(x, g_Dx, label = "g_D(x)")

print("average e_out = " , np.mean(e_out))
# scalar values
# g_bar  = avg slopes * x + avg intercepts
a_bar = np.mean(slopes)
b_bar = np.mean(intercepts)

g_bar = np.mean(g) # expected value of hypothesis
print("g_bar = " , g_bar)
fig1.plot(x, a_bar*x+b_bar, "g", label = "g_bar")
fig2.plot(x, a_bar*x+b_bar, "g", label = "g_bar")

bias_x = np.square((a_bar * x - b_bar - np.sin(np.pi*x)))
bias = np.mean(bias_x)
print("bias = ", bias)
variance = np.square(g - g_bar)
var = np.mean(variance)
print("variance = ", var)

e_out = bias + var
print("e_out = ", e_out)
plt.show()
