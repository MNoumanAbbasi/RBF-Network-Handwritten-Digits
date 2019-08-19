import numpy as np
import time
import re
import sys
import math

np.set_printoptions(threshold=sys.maxsize, suppress=True)
np.random.seed(0)

def inputXFromFile(filename, sampleSize):  # SampleSize given for performace enhancement
    """Inputs the training examples X"""
    inputArray = np.zeros(shape=(sampleSize, 784))  # 784 = pixels of each image
    with open(filename, 'r') as file:
        for i in range(sampleSize):
            inputList = []
            for _ in range(44):  # 44 lines of each example in file
                line = file.readline().strip("[").replace("]", "")
                inputList += line.split()
            inputArray[i] = inputList
    # print("X Input Size:", inputArray.shape)
    return np.divide(inputArray, 255)

def inputYFromFile(filename, sampleSize):
    """Inputs the training examples Y"""
    inputArray = np.zeros(shape=(sampleSize, 10))   # for each row, we want a column like [0 0 1 0 ...]
    with open(filename, 'r') as file:
        for i in range(sampleSize):
            value = file.readline()
            if not value: break
            inputArray[i][int(value)] = 1
    # print("Y input size:", inputArray.shape)
    return inputArray

class Network:
    def __init__(self, inputArray=None, resultArray=None):
        # self.XSize = np.size(inputArray,1)
        self.HSize = 300
        self.OSize = 10
        self.X = []
        self.C = []
        self.Y = []
        self.W = np.random.uniform(-1, 1, (self.HSize, self.OSize))
        #self.B = np.random.uniform(-1, 1, (self.OSize))

    def loadData(self, filenameX, filenameY, sampleSize):
        """Loads training/test data

        Parameters:\n
        filenameX: filename for X features\n
        filenameY: filename for Y (labels)\n
        sampleSize: number of examples in dataset
        """
        self.X = inputXFromFile(filenameX, sampleSize)
        self.Y = inputYFromFile(filenameY, sampleSize)
        
    def initializeCenters(self):
        self.C = self.X[:self.HSize]

    def train(self, numOfEpochs=1, learnRate=0.5):
        self.initializeCenters()
        errorList = []
        print("Training...")
        for _ in range(numOfEpochs):      # no. of epoques
            # Take each data sample from the inputData
            for i, x in enumerate(self.X):
                HLayer = rbf(x, self.C)
                # Multiply the weights to get output for each data
                output = np.dot(HLayer, self.W)# + self.B
                error = (output - self.Y[i])
                self.W = self.W - (learnRate * np.outer(HLayer, error))
                #self.B = self.B - (learnRate * error)
                errorList.append(error)
        print("Training done")
        # Savinf weights in a file
        np.save("resultWeight", self.W)
        # print(self.W)

    def predict(self, xData, yData):
        print("Prediciting...")
        totalAvg = totalCount = correctCount = 0.0
        self.initializeCenters()
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
start = time.time()                 # TODO Input data should be functions of neural network class
trainDataSize = 60000
# MENU
myNetwork = Network()
while True:
    userInput = input("1. Train the RBF Neural Network\n2. Predict using neural network:\n")
    if userInput == "1":
        print("Importing Data for training...")
        myNetwork.loadData("train.txt", "train-labels.txt", trainDataSize)
        print(f"{trainDataSize} training examples imported in {time.time()-start:.2f} sec")
        startTrainingTime = time.time()
        myNetwork.train(learnRate=0.5)
        print(f"Training took: {time.time()-startTrainingTime:.2f} sec")
    elif userInput == "2":
        filename = input("Enter file name containing weights: ")
        myNetwork.W = np.load(filename)
        # print(myNetwork.W)
        print("Importing Data for testing...")
        myNetwork.predict(inputXFromFile("test.txt", 10000), inputYFromFile("test-labels.txt", 10000))
    else:
        break
print("Entire program took:", time.time()-start, "sec")
