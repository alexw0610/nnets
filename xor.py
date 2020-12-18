
import numpy as np
from numpy import asarray
from numpy import save
from numpy import load
import random
import math
import time
from keras.datasets import mnist
from matplotlib import pyplot
from array import array

class Layer:
    def __init__(this, dimension, bias):
        this.dimension = dimension
        this.bias = bias
        this.create()
        if(bias):
            this.initBias()

    def create(this):
        this.nodes = np.zeros(this.dimension)

    def initBias(this):
        this.biasValues = np.array(np.random.random_sample(this.dimension)*2-1)

    def getBias(this):
        if this.bias:
            return this.biasValues

    def setBias(this,bias):
        this.biasValues = bias

    def dim(this):
        return this.dimension

    def hasBias(this):
        return this.bias

    def getNodes(this):
        return this.nodes

    def setNodes(this, nodes):
        this.nodes = nodes





class ANN:

    def __init__(this, layers):
        this.layers = layers
        this.inputLayer = layers[0]
        this.outputLayer = layers[len(this.layers)-1]
        this.layerLength = len(this.layers)
        this.create()

    def create(this):
        this.weights  = []
        for l in range(this.layerLength-1):
            temp = np.array(np.random.random_sample((len(this.layers[l].getNodes()),len(this.layers[l+1].getNodes())))*2-1)
            this.weights.append(temp)

    def printLayers(this):
        print('-- Layer Neuron Values --')
        for l in range(this.layerLength):
            layer = this.layers[l]
            print(layer.getNodes())
            print('----')
            #if(layer.hasBias()):
            #    print('-- Bias --')
            #    print(layer.getBias())
            #    print('----')

    def printWeights(this):
        print('-- Layer Weight Values --')
        for l in range(this.layerLength-1):
            weightSet = this.weights[l]
            print(weightSet)
            print('----')

    def setInputLayerData(this,layer):
        this.inputLayer.setNodes(layer)
    
    def setLayerData(this, layer, nmbr):
        this.layers[nmbr].setNodes(layer)

    def getLayerData(this, nmbr):
        return this.layers[nmbr].getNodes()

    def getOutputData(this):
        return this.outputLayer.getNodes()

    def predict(this,start):
        for l in range(start,this.layerLength-1):
            nodes = this.layers[l+1].getNodes() #get neurons of first layer after input
            nodes = np.zeros(len(nodes)) #zero the neurons
            inNodes = this.layers[l].getNodes()
            tempWeights = this.weights[l] #get first weight set of net
            for y in range(tempWeights.shape[1]):
                for x in range(tempWeights.shape[0]):
                    nodes[y] += inNodes[x]*tempWeights[x][y]

            if(this.layers[l+1].hasBias()):
                tempBias = this.layers[l+1].getBias()
                for node in range(this.layers[l+1].dim()):
                    nodes[node] += tempBias[node]

            nodes = this.activate(nodes)
            this.layers[l+1].setNodes(nodes)

    def activate(this, nodes):
        for x in range(nodes.size):
            nodes[x] = (1/(1+pow(math.e,-nodes[x])))
        return nodes

    def derivActivate(this, nodes):
        for x in range(nodes.size):
            nodes[x] = nodes[x]*(1-nodes[x])
        return nodes

    #forwardStart marks the layer where the predicition starts
    #start marks the last layer that should be updated with backpropagation
    #end marks the first layer that should be updated with backpropagation
    #layer count starts from 0
    def train(this, input, solution, learningRate, epoch, forwardStart, start, end):
        this.epoch = epoch
        mSQErrBand = [0]*epoch
        this.learningRate = learningRate
        for ep in range(epoch):

            # Choose sample to learn
            ind = math.floor(np.random.random()*(input.shape[0]))

            # predict sample
            this.setLayerData(input[ind],forwardStart)
            this.predict(forwardStart)
            result = this.getOutputData()

            #calculate the mean squared error
            mSqErr = this.loss(solution[ind],result)
            mSQErrBand[ep] = mSqErr

            #calculate difference between label and prediction
            loss = np.subtract(result,solution[ind])

            print("Epoch: ",ep," Error: ",np.round(mSqErr,4)," LR: ",learningRate)
            #print("Epoch: \t",ep," MSqErr: \t",np.round(mSqErr,4)," \n Loss: \t",np.round(loss,3))

            for l in range(this.layerLength-1,0,-1): # Go throught all layers back to front (l is *TO* layer)

                activeWeightSet = np.array(this.weights[l-1],copy=True) # The weight set connecting the previous and next Layer
                toNodes = this.layers[l].getNodes() # The node values of the next Layer
                fromNodes = this.layers[l-1].getNodes() # The node values of the previous Layer
                fromSize = activeWeightSet.shape[0] # The dimension of the previous Layer
                toSize = activeWeightSet.shape[1] # The dimension of the next Layer
                propagatedLoss = np.zeros(fromSize)
                derivToNodes = this.derivActivate(toNodes)
                biasWeight = np.zeros(toSize)
                # Change weights between from/to layer
                if(this.layers[l].hasBias()): # Layer has bias weights attached
                    biasWeight = np.array(this.layers[l].getBias(),copy=True)
                    for f in range(fromSize):
                        for t in range(toSize):
                            propagatedLoss[f] += activeWeightSet[f][t] * loss[t]
                            activeWeightSet[f][t] -= learningRate * fromNodes[f] * loss[t] * derivToNodes[t]
                            if(f==0): # Only update the bias weights once while going through TO Nodes
                                biasWeight[t] -= learningRate * loss[t] * derivToNodes[t]

                else: # Layer has NO bias weights attached
                    for f in range(fromSize):
                        for t in range(toSize):
                            propagatedLoss[f] += activeWeightSet[f][t] * loss[t]
                            activeWeightSet[f][t] -= learningRate * fromNodes[f] * loss[t] * derivToNodes[t]

                loss = this.normalize(propagatedLoss)
                if l-1 >=start and l-1 <end:
                    this.weights[l-1] = activeWeightSet
                    if(this.layers[l].hasBias()): # Layer has bias weights attached
                        tempWeight = this.layers[l].getBias()
                        tempWeight = biasWeight

        return mSQErrBand

    def loss(this, solution ,output):
        loss = np.sum(0.5*(pow(np.subtract(output,solution),2)))
        return loss

    def normalize(this, array):
        biggestElement = 0
        for x in range(len(array)):
            biggestElement = abs(array[x]) if abs(array[x])>biggestElement else biggestElement

        for x in range(len(array)):
            array[x] = array[x]/biggestElement

        return array

    def persistToDisk(this,path):
            data = asarray(this.weights)
            save(path+".weights",data,allow_pickle=True)
            biaslist = []
            for layer in range(len(this.layers)):
                if this.layers[layer].hasBias():
                    biaslist.append(this.layers[layer].getBias())
            bias = asarray(biaslist)
            save(path+".bias",bias)
            print("Saved weights to ",path)

    def loadFromDisk(this,path):
            dataW = load(path+".weights.npy",allow_pickle=True)
            this.weights = dataW
            dataB = load(path+".bias.npy",allow_pickle=True)
            count = 0
            for layer in range(len(this.layers)):
                if this.layers[layer].hasBias() == True:
                    this.layers[layer].setBias(dataB[count])
                    count += 1
            print("Loaded weights from ",path)





def oneHot(input):
    output = np.array(input)
    outputOneHot = np.zeros((output.shape[0],10))
    for x in range(output.shape[0]):
        outputOneHot[x][output[x]] = 1
    print("Labels loaded: ",output.shape[0])
    return outputOneHot

def preprocMnist(input):
    input = np.array(input)
    print("Inputs loaded: ",input.shape[0])
    input = input.reshape(1,input.shape[0],784)
    input = np.true_divide(input, 255.0)
    return input[0]

if __name__ == "__main__":

    #Example AutoEncoder
    (train_X, train_y), (test_X, test_y) = mnist.load_data(path="./mnist.npz")

    print("Mnist data loaded")

    layersDisc = []
    layersDisc.append(Layer(784,False))
    layersDisc.append(Layer(32,True))
    layersDisc.append(Layer(10,True))
    layersDisc.append(Layer(32,True))
    layersDisc.append(Layer(784,True))

    net = ANN(layersDisc)
    #net.loadFromDisk("./exampleNet")

    input = preprocMnist(train_X)
    output = oneHot(train_y)
    
    net.train(input,input,0.15,10,0,0,4) 
    #net.persistToDisk("./exampleNet")

    fig = pyplot.figure(figsize=(10,8)) 
    for cycle in range(6):
        net.setLayerData(input[cycle],0)
        net.predict(0)

        img = net.getOutputData()
        fig.add_subplot(2,3,cycle+1)
        pyplot.imshow(img.reshape(28,28))
    pyplot.show()
