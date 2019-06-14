import numpy as np
import time
import re
import sys
import math
from tempfile import TemporaryFile

np.set_printoptions(threshold=sys.maxsize, suppress=True)
np.random.seed(0)

def inputXFromFile(filename, sampleSize):  # SampleSize given for performace enhancement
    inputArray = np.zeros(shape=(sampleSize, 784))  # 784 = pixels of each image
    with open(filename, 'r') as file:
        for i in range(sampleSize):
            inputList = []
            for _ in range(44):  # 44 lines of each input entry in file
                line = file.readline()
                if not line:
                    return inputArray
                line = line.strip("[")
                line = line.replace("]", "")
                inputList += line.split()
            inputArray[i] = inputList
    # print("X Input Size:", inputArray.shape)
    return np.divide(inputArray, 255)

def inputYFromFile(filename, sampleSize):
    inputArray = np.zeros(shape=(sampleSize, 10))  # 784 = pixels of each image
    with open(filename, 'r') as file:
        for i in range(sampleSize):
            value = file.readline()
            if not value:
                break
            if value == '\n':
                continue
            inputArray[i][int(value)] = 1
    # print("Y input size:", inputArray.shape)
    return inputArray

class NeuralNetwork:
    def __init__(self, inputArray, resultArray):
        self.XSize = 784            # USe these values from inputData
        self.HSize = 300
        self.OSize = 10
        self.X = inputArray
        self.C = self.X[:self.HSize]
        self.Y = resultArray
        self.W = np.random.uniform(-1, 1, (self.HSize, self.OSize))
        #self.B = np.random.uniform(-1, 1, (self.OSize))
        self.learnRate = 0.5

    def train(self):
        print("Training...")
        for _ in range(1):      # no. of epoques
            # Take each data sample from the inputData
            for i, x in enumerate(self.X):
                HLayer = rbf(x, self.C)
                # Multiply the weights to get output for each data
                output = np.dot(HLayer, self.W)# + self.B
                error = (output - self.Y[i])
                self.W = self.W - (self.learnRate * np.outer(HLayer, error))
                #self.B = self.B - (self.learnRate * error)
        print("Training done")
        # Savinf weights in a file
        np.save("resultWeight", self.W)
        # print(self.W)

    def predict(self, xData, yData):
        print("Prediciting...")
        totalAvg = totalCount = correctCount = 0.0
        # Take each data sample from the inputData
        for i, x in enumerate(xData):
            HLayer = rbf(x, self.C)
            output = np.dot(HLayer, self.W)
            o = np.argmax(output)
            y = np.argmax(yData[i])
            if o == y:
                correctCount += 1
            totalCount += 1
            totalAvg += (correctCount*100.0)/totalCount
            # print((correctCount*100.0)/totalCount)
        print("Total Avg. Accuracy", totalAvg / yData.shape[0])

def rbf(x, C, beta=0.05):
    HList = []
    for c in C:     # For each neuron in H layer
        HList.append(math.exp((-1 * beta) * np.dot(x-c, x-c)))
    return np.array(HList)

#######     MAIN    ######
start = time.time()
trainDataSize = 60000
# MENU
while True:
    userInput = input("1. Train the RBF Neural Network\n2. Predict using neural network:\n")
    if userInput == "1":
        print("Importing Data for training...")
        inputDataX = inputXFromFile("train.txt", trainDataSize)
        inputDataY = inputYFromFile("train-labels.txt", trainDataSize)
        print(trainDataSize, "Data samples imported in", time.time() - start, "sec")
        myNetwork = NeuralNetwork(inputDataX, inputDataY)
        tempTime = time.time()
        myNetwork.train()
        print("Training took:", time.time() - tempTime, "sec")
    elif userInput == "2":
        myNetwork = NeuralNetwork(inputXFromFile("train.txt", 500), inputYFromFile("train-labels.txt", 500))
        filename = input("Enter file name containing weights: ")
        myNetwork.W = np.load(filename)
        # print(myNetwork.W)
        print("Importing Data for testing...")
        myNetwork.predict(inputXFromFile("test.txt", 10000), inputYFromFile("test-labels.txt", 10000))
    else:
        break
print("Entire program took:", time.time()-start, "sec")
