
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
    def __init__(this, dimension, bias, identifier):
        this.dimension = dimension
        this.bias = bias
        this.identifier = identifier
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

    def train(this, input, solution, learningRate, epoch, forwardStart, start, end):
        this.epoch = epoch
        mSQErrBand = [0]*epoch
        this.learningRate = learningRate
        decay = 0.0001
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
                #if(l>1): # Only propagate the loss for the next layer when we are not at the last layer
                propagatedLoss = np.zeros(fromSize)
                derivToNodes = this.derivActivate(toNodes)
                biasWeight = np.zeros(toSize)
                # Change weights between from/to layer
                if(this.layers[l].hasBias()): # Layer has bias weights attached
                    biasWeight = np.array(this.layers[l].getBias(),copy=True)
                    for f in range(fromSize):
                        for t in range(toSize):
                 #           if(l>1): # Only propagate the loss for the next layer when we are not at the last layer
                            propagatedLoss[f] += activeWeightSet[f][t] * loss[t]
                            activeWeightSet[f][t] -= learningRate * fromNodes[f] * loss[t] * derivToNodes[t]
                            if(f==0): # Only update the bias weights once while going through TO Nodes
                                biasWeight[t] -= learningRate * loss[t] * derivToNodes[t]

                else: # Layer has NO bias weights attached
                    for f in range(fromSize):
                        for t in range(toSize):
                            #if(l>1): # Only propagate the loss for the next layer when we are not at the last layer
                            propagatedLoss[f] += activeWeightSet[f][t] * loss[t]
                            activeWeightSet[f][t] -= learningRate * fromNodes[f] * loss[t] * derivToNodes[t]

                #if(l>1): # Only propagate the loss for the next layer when we are not at the last layer
                loss = this.normalize(propagatedLoss)
                if l-1 >=start and l-1 <end:
                    #print("updated weight",l-1)
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

    (train_X, train_y), (test_X, test_y) = mnist.load_data(path="./mnist.npz")

    print("Mnist data loaded")

    layersDisc = []
    layersDisc.append(Layer(10,False,"gen"))
    layersDisc.append(Layer(32,True,"gen"))
    layersDisc.append(Layer(32,True,"gen"))
    layersDisc.append(Layer(784,True,"disc"))
    layersDisc.append(Layer(32,True,"disc"))
    layersDisc.append(Layer(32,True,"disc"))
    layersDisc.append(Layer(1,True,"disc"))
    
    net = ANN(layersDisc)
    #net.loadFromDisk("./net")

    input = preprocMnist(train_X)
    output = oneHot(train_y)
    
    #net.setLayerData(input[0],2)
    #net.predict(2)
    #net.printWeights()
    #net.train(np.array(np.random.random_sample((10,1))),np.array(np.random.random_sample((10,1))),0.2,5,0,2)
    #net.train(input,np.zeros(60000),22,5,0,4)
    #net.printWeights()
    #net.persistToDisk("./net")
    #print("Training starting..")
    net.loadFromDisk("./netBatch")

    for cycle in range(5):
        #net.train(input[120*cycle:120*(cycle+1)],np.ones(120),0.015,120,3,3,6)
        #net.train(np.array(np.random.random_sample((120,10))),np.zeros(120),0.015,120,0,3,6)
        net.train(np.array(np.random.random_sample((240,10))),np.ones(240),0.02,240,0,0,3) 
        if cycle%10 == 0:
            #net.persistToDisk("./netBatch")
            print("Saved net to disk.")
    
    #net.persistToDisk("./netBatch")
    print("Training finished..")

    fig=pyplot.figure(figsize=(10, 8))
    
    for cycle in range(6):
        input = np.array(np.random.random_sample(10))
        net.setLayerData(input,0)
        net.predict(0)

        label = net.getOutputData()
        img = net.getLayerData(3)
        print(label)
        fig.add_subplot(2,3,cycle+1)
        pyplot.imshow(img.reshape(28,28))

    print("Showing results")
    pyplot.show()

    # input = preprocMnist(test_X)
    # output = oneHot(test_y)
    # solved = 0
    # print("Testing ...")
    # for i in range(len(input)):
    #     net.setInputLayerData(input[i])
    #     net.predict()
    #     result = net.getOutputData()
    #     ind = 0
    #     for t in range(len(result)):
    #         ind = t if result[t]>result[ind] else ind
    #     #print("Truth: ",test_y[i]," Predicted: ",ind)
    #     if(test_y[i] == ind):
    #         solved += 1
    #
    # print("Precision: ",((float(solved)/float(len(input)))*100.0),"%")
