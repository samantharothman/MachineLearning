import matplotlib.pyplot as plt
import numpy 

class Perceptron(object):
    """Class to run the perceptron learning algorithm"""

    def __init__(self, dimension,threshold = 1000, learning_rate = 0.01):
        """initialize variables for the algorithm"""
        self.dimension = dimension
        self.thresh = threshold
        self.learn = learning_rate
        self.ws = numpy.zeros(dimension + 1)
        self.count = 0
     
        
    def predictor(self, point):
        """Predict label for each data point
           input = data point (has d dimensions and a label
           returns predicted label (int) """
        w12 = self.ws[1:]
        xy = point[:self.dimension]
        net = self.ws[0] + numpy.dot(w12, xy)
        if net >= 0:
            output = 1
        else:
            output = -1
        return output
 


    def training(self, inputs):
        """perceptron learning algorithm - looks for a misclassified
           point and and updates the weights until all points are correct"""
        misclassified = 1000 
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
                misclassified = 1000 
                
""" 
    def training(self, inputs):
    #Another version of the algorithm that goes through every
    #point and stops when the weights do not change between runs
    #did not use this but it has similar outcomes
        before = []
        for i in range(self.dimension + 1):
            before.append(0)
        for i in range(self.thresh):
            for point in inputs:
                label = point[self.dimension]
                xy = point[:self.dimension]
                pred = self.predictor(xy)
                self.ws[1:] += self.learn * (label - pred) * xy
                self.ws[0] += self.learn * (label - pred)
            if numpy.array_equal(before,self.ws) == False:
                self.count += 1
                for i in range(self.dimension + 1):
                    before[i] = self.ws[i]
                continue
            else:
                 break
"""

# x axis
t = numpy.arange(0,1000)

# create target and data
target = 0.5 * t + 3   
data = numpy.random.randint(low = 0, high = 1001, size =(1000, 3)) 

# label data according to target
for i in data:
    x = i[0]
    if i[1] >= (0.5*x+3):
        i[2] = 1
    else:
        i[2] = -1

# run perceptron for 2 dimensions
perceptron =  Perceptron(2) 
perceptron.training(data)
print("final weights:", perceptron.ws)
print("number of updates:", perceptron.count)

# equation of line created by perceptron    
y = -(perceptron.ws[0]/ perceptron.ws[2]) - ((perceptron.ws[1]/perceptron.ws[2]) * t)

# graph points and lines
plt.plot(t,target, color = 'g', label = 'f')
plt.plot(t, y, color = 'm', label = 'g')
plt.title("Perceptron Algorithm")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
for i in data:
    if i[2] == -1:
        plt.scatter(i[0], i[1], marker='_', color = 'r')
    else:
        plt.scatter(i[0], i[1], marker='+', color = 'b')

plt.show()

