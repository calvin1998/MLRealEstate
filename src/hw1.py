import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def main():
    realEstateDict = csv.DictReader(open("real_estate.csv", encoding='utf-8-sig'))
    realEstateSamples = []
    prices = []
    features = ["age", "nearestMRT", "nConvenience"]

    minMax = np.array([[float("inf"), float("-inf")] for i in range(0, len(features))])
    ave = np.zeros(1, len(features))

    for row in realEstateDict:
        if validRow(row):
            # add row
            realEstateSamples.append(extractFeatures(row, features))
            prices.append(extractValue(row, "price"))
            #update min/max
            for i in range(0, len(features)):
                sample = realEstateSamples[-1][0, i]
                if (sample < minMax[i][0]):
                    minMax[i][0] = sample
                if (sample > minMax[i][1]):
                    minMax[i][1] = sample

    #compute average
    for i in range(0, len(features)):
        total = 0.0
        for sample in realEstateSamples:
            total += sample[0, i]
        ave[i] = total/len(realEstateSamples)
                
    # normalise all features
    for sample in realEstateSamples:
        for i in range(0, len(features)):
            sample[0, i] = (sample[0, i] - minMax[i][0])/(minMax[i][1] - minMax[i][0])

    trainSet = realEstateSamples[0:int(len(realEstateSamples)/2)]
    testSet = realEstateSamples[int(len(realEstateSamples)/2):len(realEstateSamples)]

    trainResults = prices[0:int(len(prices)/2)]
    testResults = prices[0:int(len(prices)/2)]

    #show first and last elements of training and test set
    # print("trainSet[0]: " + str(trainSet[0]))
    # print("trainSet[-1]: " + str(trainSet[-1]))
    # print("testSet[0]: " + str(testSet[0]))
    # print("testSet[-1]: " + str(testSet[-1]))

    
    w = np.ones((4, 1))#initial weight vector
    steps = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01]#step sizes
    iters = 400# iterations

    gradientDescent(w, steps, iters, trainSet, trainResults)



def gradientDescent(w, steps, iters, trainSet, responses):
    fig, ax = plt.subplots(3,3, figsize=(10,10))
    losses = []
    for i, step in enumerate(steps):
        print(f"doing steps {i}")
        for j in range(0, iters):
            w = w - step*meanFunc(responses, w, trainSet, diffLoss)
            losses[i][j] = meanFunc(responses, w, trainSet, loss)
    for i, ax in enumerate(ax.flat):
        ax.plot(losses[i])
        ax.set_title(f"step size {i}")
    plt.tight_layout()
    plt.show()


def meanFunc(responses, w, predictors, func):
    sum = np.zeros((4, 1))
    for i, response in enumerate(responses):
        sum += func(responses, w, predictors[i])
    return sum/len(responses)

def loss(y, w, x):
    xFull = np.concatenate(([[1]], x), axis=1)
    yHat = xFull @ w
    return sqrt(((y - yHat)**2)/4 + 1) - 1

def diffLoss(y, w, x):
    xFull = np.concatenate(([[1]], x), axis=1)
    yHat = xFull @ w
    return np.transpose(xFull)*(yHat - y)/(2*math.sqrt((yHat - y)**2 + 4))
    
def validRow(row):
    for _, value in row.items():
        if value == "NA":
            return False
    return True

def extractFeatures(row, features):
    return np.array([[float(row[key]) for key in features]])

def extractValue(row, key):
    return float(row[key])

if __name__ == "__main__":
    main()