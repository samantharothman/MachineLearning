import numpy as np
import matplotlib.pyplot as plt
# part c
z = 0
slopes = []
intercepts = []
g = []
e_out = []
fig, (fig1, fig2) = plt.subplots(1, 2)
while z<100:
    
    x1 = np.random.uniform(low=-1.0)
    x2 = np.random.uniform(low=-1.0)
    y1 = np.sin(np.pi*x1)
    y2 = np.sin(np.pi*x2)
    intercept = (y1+y2)/2
    intercepts.append(intercept)
 
    fig2.plot(x1, y1, 'bo')
    fig2.plot(x2, y2, 'bo')
    x = np.arange(-1., 1., .01)
    g_Dx = (y1+y2)/2
    g.append(g_Dx)
    e_out.append(np.square(np.sin(np.pi * x) - g_Dx))
    fig1.axhline(g_Dx)
    z+=1

fig1.plot(x, np.sin(np.pi*x), "r")
print("average e_out = ", np.mean(e_out))
b_bar = np.mean(intercepts)
g_bar = np.mean(g) # expected value of hypothesis
print("g_bar = " , g_bar)
fig1.axhline(g_bar, color = "green", label = "g_bar")
fig2.axhline(g_bar, color = "green", label = "g_bar")
bias_x = np.square((b_bar - np.sin(np.pi*x)))
bias = np.mean(bias_x)
print("bias =  " , bias)
variance = np.square(g - g_bar)
var = np.mean(variance)
print("Variance = ", var)
e_out = bias + var
print("e_out =  ", e_out)
plt.show()
