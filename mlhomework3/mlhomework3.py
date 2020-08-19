import matplotlib.pyplot as plt
import numpy

class Perceptron(object):

    def __init__(self, dimension,threshold = 10000, learning_rate = 0.01):
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
    def misclassified(self, inputs, num = 0):
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
                wrong = self.misclassified(inputs)
                self.e_in.append(wrong/100)
                self.ws[1:] += self.learn * (label - pred) * xy
                self.ws[0] += self.learn * (label - pred)
                self.count += 1
                misclassified = 100

t = numpy.arange(0,100)
fig, (fig1, fig2) = plt.subplots(1, 2)

# random target line and random data
target = 0.5 * t + 3   
data = numpy.random.randint(low = 0, high = 101, size =(100, 3))

# assign labels to data
for i in data:
    x = i[0]
    if i[1] >= (0.5*x+3):
        i[2] = 1
    else:
        i[2] = -1
        
# generate noise
for x in range(int(len(data)*.05)):
	i = numpy.random.randint(len(data)-1)
	data[i][2] = (data[i][2] * -1)

# compute lin reg
matrix = data[:,0:2]
yvector = data[:, 2:3]

Xdag = numpy.matmul( numpy.linalg.pinv(numpy.matmul(matrix.transpose(), matrix)), matrix.transpose() )
w = numpy.matmul(Xdag, yvector)
#fig1.plot(t , -w[0]*t -w[1])

# calculate in sample error for lin reg
e_lin = numpy.matmul(matrix, w) - yvector
print("E_in for linear regression: ", numpy.mean(e_lin))

# run perceptron on the data
perceptron =  Perceptron(2) #dimension here
perceptron.training(data)
print("final weights:", perceptron.ws)
print("number of updates:", perceptron.count)
    
y = -(perceptron.ws[0]/ perceptron.ws[2]) - ((perceptron.ws[1]/perceptron.ws[2]) * t)

# plot the perceptron line and target
fig1.plot(t,target, color = 'g', label = 'f')
fig1.plot(t, y, color = 'm', label = 'g')
fig1.set_ylim([0, 100])

# plot the data points
for i in data:
    if i[2] == -1:
        fig1.scatter(i[0], i[1], marker='_', color = 'r')
    else:
        fig1.scatter(i[0], i[1], marker='+', color = 'b')

# plot the perceptron e_in as a function of iteration
iterations = numpy.arange(0,perceptron.count)
fig2.plot(iterations, perceptron.e_in)

plt.show()

