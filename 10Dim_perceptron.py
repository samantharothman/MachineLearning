import numpy
import matplotlib.pyplot as plt

class Perceptron(object):

    def __init__(self, dimension,threshold = 10000, learning_rate = 0.01):
        self.dimension = dimension
        self.thresh = threshold
        self.learn = learning_rate
        self.ws = numpy.zeros(dimension + 1)
        self.count = 0
     
        
    def predictor(self, point):
        w12 = self.ws[1:]
        xy = point[:self.dimension]
        net = self.ws[0] + numpy.dot(w12, xy)
        if net >= 0:
            output = 1
        else:
            output = -1
        return output
    
     # stop if you dont find a misclassified point
    def training(self, inputs):
        misclassified = 1000 # make this size
        while misclassified != 0:
            if self.count == self.thresh:
                break
            index = numpy.random.randint(low = 0, high = 1000)
            point = inputs[index]
            label = point[self.dimension]
            xy = point[:self.dimension]
            pred = self.predictor(xy)
            if pred == point[self.dimension]:
                misclassified -= 1
            else:
                self.ws[1:] += self.learn * (label - pred) * xy
                self.ws[0] += self.learn * (label - pred)
                self.count += 1
                misclassified = 1000 # unclear
"""               
# stop if you dont find a misclassified point
    def training(self, inputs):
        misclassified = 1000 # make this size
        while misclassified != 0:
            if self.count == self.thresh:
                break
            index = numpy.random.randint(low = 0, high = 1000)
            point = inputs[index]
            label = point[self.dimension]
            xy = point[:self.dimension]
            pred = self.predictor(xy)
            if pred == point[self.dimension]:
                misclassified -= 1
            else:
                self.ws[1:] += self.learn * (label - pred) * xy
                self.ws[0] += self.learn * (label - pred)
                self.count += 1
                misclassified += 1000 # unclear
                
"""

# create random 10d data              
ten = numpy.random.multivariate_normal(10*numpy.ones(11), 2*numpy.eye(11), size=500)
two = numpy.random.multivariate_normal(numpy.zeros(11), numpy.eye(11), size=500)
inputs = numpy.concatenate((ten, two), axis = 0)

# label data
count = 0
for i in inputs:
    if count <500:
        i[-1] =  1
    else:
        i[-1] = -1
    count += 1

# run perceptron with 10 dimensions
perceptron =  Perceptron(10)
perceptron.training(inputs)
print("final weights:", perceptron.ws)
print("number of updates:", perceptron.count)

# run perceptron 100 times with x(t) random
used = []
train = []
updates = []
for j in range(100):
    while len(used) != 1000:
        index = numpy.random.randint(low = 0, high = 1000)
        if index not in used:
            used.append(index)
            point = inputs[index]
            train.append(point)
    
    perceptron =  Perceptron(10)
    perceptron.training(train)
    updates.append(perceptron.count)
        
# create histogram with number of updates until convergence
plt.hist(updates)
plt.title("Perceptron Algorithm Convergence in R10")
plt.xlabel('Number of Updates')
plt.ylabel('Frequency')
plt.show()

