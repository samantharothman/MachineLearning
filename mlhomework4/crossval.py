import numpy
import matplotlib.pyplot as plt

D = numpy.random.uniform(low=-1, size=(5, 2))
for point in D:
    xi = point[0]
    point[1] = (xi**2) + 0.1

y = D[:, 1:2]

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
I = numpy.identity(m)
L = [0, numpy.exp(-5), numpy.exp(-2), numpy.exp(0)]
val = numpy.arange(-1,1, 0.01)
plt.figure(0)
plots = []
for i in range(4):
    lam = L[i]
    error = []
    for j in range(5):
        tempd = D
        tempZ = Z
        yj = D[j][1]
        tempd = numpy.delete(tempd,j , axis = 0)
        tempX = tempd[:, 0:1]
        tempy = tempd[:, 1:2]
        zj = Z[j]
        tempZ = numpy.delete(tempZ, j, axis=0)
        w = numpy.matmul((numpy.linalg.inv(numpy.matmul(tempZ.transpose(), tempZ))+ lam * I), numpy.matmul(tempZ.transpose(), tempy))
        err = numpy.square(yj - numpy.matmul(w.transpose(), zj))
        error.append(err)
        ax = plt.subplot2grid((5,4), (j,i))
        plt.plot(val, val**2)
        #plt.scatter(val, w[0] + w[1]*(val) + w[2]*(val**2) + w[3]*(val**3) + w[4]*(val**4))
        c = []
        for a in val:
            #a = Z[i][1]
            b = w[0] + w[1]*a + w[2]*(1/2)*(((3*a)**2)-1) + w[3]*(1/2)*(((5*a)**3)-(3*a)) +w[4]*(1/8)*(((35*a)**4)-((30*a)**2)+3)
            b = b/100000000
            c.append(b)
        plt.plot(val, c)
        plt.scatter(tempX, tempy, color = 'r')
        plt.ylim([0,1])
    print("Validation error: ", numpy.mean(error), "for lambda ", lam)
        #print(err)

finalw = numpy.matmul((numpy.linalg.inv(numpy.matmul(Z.transpose(), Z))+ 0 * I), numpy.matmul(Z.transpose(), y))
print("final w: " , finalw)
plt.show()
