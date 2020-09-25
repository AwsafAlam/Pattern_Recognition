import numpy as np

class Perceptron(object):

    def __init__(self, training_size, features, threshold=500, learning_rate=0.01):
        np.random.seed(23)
        self.training_size = training_size
        self.features = features
        self.threshold = threshold
        self.learning_rate = float(learning_rate)
        self.bias = 0
        self.weights = np.random.uniform(-1,1,(features + 1))
        # self.weights = np.zeros(features + 1)

    def print_weight_vec(self):
        print("Weight Vec => ")
        print(self.weights)
    
    def getWeight(self):
      return self.weights

    def predict(self, inputs):
        print(inputs)
        summation = np.dot(self.weights[1:],inputs) + self.weights[0]
        if summation >= 0: # Belongs to W1
          activation = 1
        else: # Belongs to W2
          activation = 2            
        return activation

    def train_basic_perceptron(self, training_inputs, labels, dataset):
        for i in range(self.threshold):
          Y = []
          arr_dx = []
          for j in range(self.training_size):
              x = np.array(dataset[j])
              group = x[self.features]
              x[self.features] = 1
              x = x.reshape(self.features+1,1)
              dot_product = np.dot(self.weights,x)[0]
              if(group == 2 and dot_product>0):
                  Y.append(x)
                  arr_dx.append(1)
              elif(group ==1 and dot_product<0):
                  Y.append(x)
                  arr_dx.append(-1)
              else:
                  pass
          
          sum = np.zeros(self.features+1)
          
          for j in range(len(Y)):
              sum += arr_dx[j]*Y[j].transpose()[0]
          
          
          self.weights = self.weights - self.learning_rate *sum
          print("Iter {} => {}".format(i,"---"))
          if len(Y) == 0:
              break
              