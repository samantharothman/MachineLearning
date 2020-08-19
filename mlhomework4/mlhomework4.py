import numpy
import matplotlib.pyplot as plt
D = numpy.random.uniform(low=-1, size=(5, 2))
for point in D:
    xi = point[0]
    point[1] = (xi**2) + 0.1
y = D[:, 1:2]
print(y)
X = D[:,0:1]
Z = numpy.empty((5,5))   
for index in range(5):
    x = D[index][0]
    z1 = x
    z2 = (1/2)*(((3*x)**2)-1)
    z3 = (1/2)*(((5*x)**3)-(3*x))
    z4 = (1/8)*(((35*x)**4)-((30*x)**2)+3)
    z = [1, z1, z2, z3, z4]
    Z[index] = z
n, m = Z.shape
print(n)
print(m)
I = numpy.identity(m)
L = [0, numpy.exp(-5), numpy.exp(-2), numpy.exp(0)]
plt.figure(0)
plots = []
val = numpy.arange(-1,1, 0.01)
for i in range(4):
    lam = L[i]
    ax = plt.subplot2grid((1,4), (0,i))
    plt.plot(val, val**2)
    plt.scatter(X, y, color = 'r')
    
    w = numpy.matmul((numpy.linalg.inv(numpy.matmul(Z.transpose(), Z))+ lam * I), numpy.matmul(Z.transpose(), y))
    #plt.scatter(val, w[0] + w[1]*(val) + w[2]*(val**2) + w[3]*(val**3) + w[4]*(val**4))
    c = []
    for a in val:
        #a = Z[i][1]
        b = w[0] + w[1]*a + w[2]*(1/2)*(((3*a)**2)-1) + w[3]*(1/2)*(((5*a)**3)-(3*a)) +w[4]*(1/8)*(((35*a)**4)-((30*a)**2)+3)
        b = b/100000000
        c.append(b)
    plt.plot(val, c)
    error = []
    for j in range(5):
        yi = numpy.square(D[j][0])
        zi = Z[j]
        err = numpy.square(yi - numpy.matmul(w.transpose(), zi))  # how to calculate error
        error.append(err)
    final = numpy.mean(error) + lam*(numpy.matmul(w.transpose(), w))
    print("total error = ", final, "for lambda = ", lam)
    plt.ylim([0,1])

plt.show()
