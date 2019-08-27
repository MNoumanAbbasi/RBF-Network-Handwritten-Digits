import numpy as np
import time
import re
import sys
import math
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize, suppress=True)
np.random.seed(0)


def inputXFromFile(filename, sampleSize):  # SampleSize given for performace enhancement
    """Inputs the training examples X"""
    inputArray = np.zeros(shape=(sampleSize, 784))  # 784 = pixels of each image
    with open(filename, "r") as file:
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
    # for each row, we want a column like [0 0 1 0 ...]
    inputArray = np.zeros(shape=(sampleSize, 10))
    with open(filename, "r") as file:
        for i in range(sampleSize):
            value = file.readline()
            if not value:
                break
            inputArray[i][int(value)] = 1
    # print("Y input size:", inputArray.shape)
    return inputArray


class Network:
    def __init__(self):
        self.XSize = 0
        self.HSize = 300
        self.OSize = 10
        self.X = []
        self.C = []
        self.Y = []
        self.W = np.random.uniform(-1, 1, (self.HSize, self.OSize))
        # self.B = np.random.uniform(-1, 1, (self.OSize))
        self.trainErrors = []
        self.testErrors = []

    def loadData(self, filenameX, filenameY, sampleSize):
        """Loads training/test data\n
        Parameters:\n
        filenameX: filename for X features\n
        filenameY: filename for Y (labels)\n
        sampleSize: number of examples in dataset
        """
        self.X = inputXFromFile(filenameX, sampleSize)
        self.Y = inputYFromFile(filenameY, sampleSize)
        self.XSize = sampleSize

    def initializeCenters(self):
        """Initializes Centers (for RBF neurons in hidden layer)
        """
        self.C = self.X[: self.HSize]

    def train(self, numOfEpochs=1, learnRate=0.5):
        self.initializeCenters()
        self.trainErrors = np.zeros(shape=self.XSize)  # Preallocating numpy array
        print("Training...")
        for _ in range(numOfEpochs):
            # Take each data sample from the inputData
            for i, x in enumerate(self.X):
                HLayer = rbf(x, self.C)
                # Multiply the weights to get output for each data
                output = np.dot(HLayer, self.W)  # + self.B
                error = output - self.Y[i]
                self.W = self.W - (learnRate * np.outer(HLayer, error))
                # self.B = self.B - (learnRate * error)
                self.trainErrors[i] = 0.5 * sum(error ** 2)
        print("Training done")
        # Saving weights and centers in a file
        np.save("weights", self.W)
        np.save("centers", self.C)

    def predict(self):
        self.testErrors = np.zeros(shape=self.XSize)  # Preallocating numpy array
        print("Prediciting...")
        totalAvg = count = correctCount = 0.0
        # Take each data sample from the inputData
        for count, x in enumerate(self.X):
            HLayer = rbf(x, self.C)
            output = np.dot(HLayer, self.W)  # + self.B
            o = np.argmax(output)
            y = np.argmax(self.Y[count])
            if o == y:
                correctCount += 1

            error = output - self.Y[count]
            self.testErrors[count] = 0.5 * sum(error ** 2)

        totalAvg = (correctCount * 100.0) / (count + 1)
        print("Total Avg. Accuracy:", totalAvg)


def rbf(x, C, beta=0.05):
    """Radial Basis Function\n
    Parameters:\n
    x: a training example
    C: centers of used for the hidden layer
    """
    H = np.zeros(shape=(np.shape(C)[0]))
    for i, c in enumerate(C):  # For each neuron in H layer
        H[i] = math.exp((-1 * beta) * np.dot(x - c, x - c))
    return H


def plotLearningCurves(trainErrors, testErrors):
    """Plots the learning curves of both training cost and test cost
    """
    # Averaging over the first {avgSize} examples
    avgSize = 100
    if type(trainErrors) is np.ndarray:     # if trainError data is available
        Jtrain = trainErrors.reshape(-1, avgSize).mean(axis=1)
        plt.plot(Jtrain)
    Jtest = testErrors.reshape(-1, avgSize).mean(axis=1)
    plt.plot(Jtest)
    plt.xlabel(f"Data examples in {avgSize}s")
    plt.ylabel("Cost")
    plt.show()


#######     MAIN    ######
start = time.time()  # TODO Input data should be functions of neural network class
trainDataSize = 60000
testDataSize = 10000
# MENU
myNetwork = Network()
while True:
    print("1. Train the RBF Network\n2. Predict using the RBF Network")
    userInput = input("Choose your option: ")
    if userInput == "1":
        print("Importing data for training...")
        startTime = time.time()
        myNetwork.loadData("train.txt", "train-labels.txt", trainDataSize)
        print(
            f"{trainDataSize} training examples imported in {time.time()-startTime:.2f} sec"
        )
        startTrainingTime = time.time()
        myNetwork.train(numOfEpochs=1, learnRate=0.3)
        print(f"Training took: {time.time()-startTrainingTime:.2f} sec")
    elif userInput == "2":
        # Loading centers and weights from save file
        filename = input("Enter file name containing weights (default: weights.npy): ")
        myNetwork.W = np.load(filename)
        myNetwork.C = np.load("centers.npy")
        # print(myNetwork.W)
        print("Importing data for testing...")
        myNetwork.loadData("test.txt", "test-labels.txt", testDataSize)
        myNetwork.predict()
        # Uncomment line below to plot learning curves for first 1000 examples
        plotLearningCurves(myNetwork.trainErrors[:10000], myNetwork.testErrors[:10000])
    else:
        break
print(f"Entire program took: {time.time()-start:.2f} sec")
