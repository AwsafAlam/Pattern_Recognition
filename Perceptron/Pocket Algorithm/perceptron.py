import numpy as np

class Perceptron(object):

    def __init__(self, features, threshold=100, learning_rate=0.01):
        np.random.seed(23)
        self.threshold = threshold
        self.learning_rate = float(learning_rate)
        self.bias = 0
        self.weights = np.random.uniform(-1,1,(features + 1))
        self.pocket = np.copy(self.weights)
        # self.weights = np.zeros(features + 1)

    def print_weight_vec(self):
        print(self.weights)

    def print_pocket_vec(self):
        print(self.pocket)

    def predict(self, inputs):
        summation = np.dot(inputs, self.pocket[1:]) + self.pocket[0]
        if summation > 0: # Belongs to W1
          activation = 1
        else: # Belongs to W2
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        for i in range(self.threshold):
          print("Iteration {} ==========".format(i))
          misclassified = 0
          hs = len(training_inputs)
          for inputs, label in zip(training_inputs, labels):
            prediction = self.predict(inputs) # output of unit activation function
            # print(inputs, label, prediction)
            ## Reward punish -- ?
            if label == 1 and prediction == 0: # class1 misclassified.
              # Wi = Wi + n*d*input -> d = 1 or -1.
              misclassified += 1
              self.weights[1:] += self.learning_rate * inputs * 1
              self.weights[0] += self.learning_rate # changing the bias

            elif label == 2 and prediction == 1: # class2 misclassified
              misclassified += 1
              self.weights[1:] += self.learning_rate * inputs * -1
              self.weights[0] += self.learning_rate * -1
            else:
              # print("Converging --")
              pass # do nothing
          
          print("{} {}".format(misclassified,hs))
          if (misclassified < hs) :
            print("Better classify--")
            hs = misclassified
            self.pocket = np.copy(self.weights)

          if misclassified == 0:
            print("\n ---- CONVERGED ---- \n\n")
            break