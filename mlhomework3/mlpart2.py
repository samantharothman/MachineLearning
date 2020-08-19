import matplotlib.pyplot as plt
import numpy

class Perceptron(object):

   def __init__(self, dimension,threshold = 1000, learning_rate = 0.01):
       self.dimension = dimension
       self.thresh = threshold
       self.learn = learning_rate
       self.ws = numpy.zeros(dimension + 1)
       self.count = 0
       self.e_in = []
    
       
   def predictor(self, point):
       w12 = self.ws[1:]
       xy = point[:self.dimension]
       net = self.ws[0] + numpy.dot(w12, xy)
       if net >= 0:
           output = 1
       else:
           output = -1
       return output

   # function to find e_in -- number of misclassified points
   def misclassify(self, inputs, num = 0):
       for point in inputs:
           label = point[self.dimension]
           xy = point[:self.dimension]
           pred = self.predictor(xy)
           if pred != point[self.dimension]:
               num += 1
       return num
   
   # stop if you dont find a misclassified point
   def training(self, inputs):
       misclassified = 100 # make this size
       while misclassified != 0:
           if self.count == self.thresh:
               break
           index = numpy.random.randint(low = 0, high = 100)
           point = inputs[index]
           label = point[self.dimension]
           xy = point[:self.dimension]
           pred = self.predictor(xy)
           if pred == point[self.dimension]:
               misclassified -= 1
           else:
               wrong = self.misclassify(inputs)
               self.e_in.append(wrong/100)
               self.ws[1:] += self.learn * (label - pred) * xy
               self.ws[0] += self.learn * (label - pred)
               self.count += 1
               misclassified = 100

# set up 2 dimensional data
X = numpy.random.uniform(size =(100, 3)) #change for dimension

# map data to 5 dimensions
Z = numpy.empty((100,6))

for index in range(100):
    x1 = X[index][0]
    x2 = X[index][1]
    s = [x1, x2, numpy.square(x1), numpy.square(x2), x1*x2, 0]
    Z[index] = s
    
# assign labels
for i in range(100):
    x = X[i][0]
    y = X[i][1]
    z = Z[i][4]
    label = 1 + 3*x + 4*y + 4*(x**2) + 2*(y**2)
    flag = numpy.random.binomial(1,0.5)
    if  flag == 0 and label > z :
        X[i][-1] = 1
        Z[i][-1] = 1
    else:
        X[i][-1] = -1
        Z[i][-1] = -1


#run perceptron on the data
perceptron =  Perceptron(5) #dimension here
perceptron.training(Z)
print("final weights:", perceptron.ws)
print("number of updates:", perceptron.count)

# e_in for perceptron
iterations = numpy.arange(0,perceptron.count)
plt.plot(iterations, perceptron.e_in)

"""
# plot perceptron
fig, (fig1, fig2) = plt.subplots(1, 2)
t = numpy.arange(0,1.5)

for i in X:
    if i[2] == -1:
        fig1.scatter(i[0], i[1], marker='_', color = 'r')
    else:
        fig1.scatter(i[0], i[1], marker='+', color = 'b')

y = -(perceptron.ws[0]/ perceptron.ws[2]) - ((perceptron.ws[1]/perceptron.ws[2]) * t)

fig1.plot(t, y, color = 'm', label = 'g')
fig1.set_ylim([0, 1])

#compute lin reg
matrix = X[:,0:2]
yvector = X[:, 2:3]

Xdag = numpy.matmul( numpy.linalg.pinv(numpy.matmul(matrix.transpose(), matrix)), matrix.transpose() )
w = numpy.matmul(Xdag, yvector)

#calculate error for lin reg
e_lin = numpy.matmul(matrix, w) - yvector
print("Error for linear regression: ", numpy.mean(e_lin))

iterations = numpy.arange(0,perceptron.count)
fig2.plot(iterations, perceptron.e_in)
"""


plt.show()
