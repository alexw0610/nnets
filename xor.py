
import numpy as np
from numpy import asarray
from numpy import save
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

    def printWeights(this):
        print('-- Layer Weight Values --')
        for l in range(this.layerLength-1):
            weightSet = this.weights[l]
            print(weightSet)
            print('----')

    def setInputLayerData(this,layer):
        this.inputLayer.setNodes(layer)

    def getOutputData(this):
        return this.outputLayer.getNodes()

    def predict(this):
        for l in range(this.layerLength-1):
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

    def train(this, input, solution, learningRate, epoch):
        this.epoch = epoch
        mSQErrBand = [0]*epoch
        this.learningRate = learningRate
        decay = 0.0001
        for ep in range(epoch):

            # Choose sample to learn
            ind = math.floor(np.random.random()*(input.shape[0]))

            # predict sample
            this.setInputLayerData(input[ind])
            this.predict()
            result = this.outputLayer.getNodes()

            #calculate the learning Rate
            #learningRate = (1/(1+decay*ep))*learningRate
            #learningRate = pow(decay,epoch)*learningRate
            #learningRate = learningRate*((epoch-ep)/epoch)
            #learningRate = (learningRate * (1/(1+decay*ep)))+0.001


            #calculate the mean squared error
            mSqErr = this.loss(solution[ind],result)
            mSQErrBand[ep] = mSqErr

            #calculate difference between label and prediction
            loss = np.subtract(result,solution[ind])

            print("Epoch: ",ep," Error: ",np.round(mSqErr,4)," LR: ",learningRate)
            #print("Epoch: \t",ep," MSqErr: \t",np.round(mSqErr,4)," \n Loss: \t",np.round(loss,3))

            for l in range(this.layerLength-1,0,-1): # Go throught all layers back to front (l is *TO* layer)

                activeWeightSet = this.weights[l-1] # The weight set connecting the previous and next Layer
                toNodes = this.layers[l].getNodes() # The node values of the next Layer
                fromNodes = this.layers[l-1].getNodes() # The node values of the previous Layer
                fromSize = activeWeightSet.shape[0] # The dimension of the previous Layer
                toSize = activeWeightSet.shape[1] # The dimension of the next Layer
                if(l>1): # Only propagate the loss for the next layer when we are not at the last layer
                    propagatedLoss = np.zeros(fromSize)
                derivToNodes = this.derivActivate(toNodes)

                # Change weights between from/to layer
                if(this.layers[l].hasBias()): # Layer has bias weights attached
                    biasWeight = this.layers[l].getBias()
                    for f in range(fromSize):
                        for t in range(toSize):
                            activeWeightSet[f][t] -= learningRate * fromNodes[f] * loss[t] * derivToNodes[t]
                            if(l>1): # Only propagate the loss for the next layer when we are not at the last layer
                                propagatedLoss[f] += activeWeightSet[f][t] * loss[t]
                            if(f==0): # Only update the bias weights once while going through TO Nodes
                                biasWeight[t] -= learningRate * loss[t] * derivToNodes[t]

                else: # Layer has NO bias weights attached
                    for f in range(fromSize):
                        for t in range(toSize):
                            activeWeightSet[f][t] -= learningRate * fromNodes[f] * loss[t] * derivToNodes[t]
                            if(l>1): # Only propagate the loss for the next layer when we are not at the last layer
                                propagatedLoss[f] += activeWeightSet[f][t] * loss[t]

                if(l>1): # Only propagate the loss for the next layer when we are not at the last layer
                    loss = this.normalize(propagatedLoss)

                this.weights[l-1] = activeWeightSet

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
        with open(path,"wb") as output:
            weights = this.weights
            layers = this.layers
            count = 0
            for layer in range(len(weights)):
                activeWeightSet = weights[layer]
                fromSize = activeWeightSet.shape[0] # The dimension of the previous Layer
                toSize = activeWeightSet.shape[1] # The dimension of the next Layer
                for f in range(fromSize):
                        float_array = array('d', activeWeightSet[f])
                        float_array.tofile(output)
                        count += activeWeightSet[f].shape[0]
            for layer in range(len(layers)):
                if layers[layer].hasBias():
                    biasWeights = layers[layer].getBias()
                    float_array = array('d', biasWeights)
                    float_array.tofile(output)
                    count += biasWeights.shape[0]


            output.close()
            print("Saved ",count,"weights to ",path)

    def loadFromDisk(this,path):
        with open(path,"rb") as input:
            weights = this.weights
            layers = this.layers
            count = 0
            for layer in range(len(weights)):
                activeWeightSet = weights[layer]
                fromSize = activeWeightSet.shape[0] # The dimension of the previous Layer
                toSize = activeWeightSet.shape[1] # The dimension of the next Layer
                for f in range(fromSize):
                        float_array = array('d')
                        float_array.fromfile(input,activeWeightSet[f].shape[0])
                        activeWeightSet[f] = np.array(float_array.tolist())
                        count += activeWeightSet[f].shape[0]
            for layer in range(len(layers)):
                if layers[layer].hasBias():
                    biasWeights = layers[layer].getBias()
                    float_array = array('d')
                    float_array.fromfile(input,biasWeights.shape[0])
                    biasWeights = np.array(float_array.tolist())
                    count += biasWeights.shape[0]

            input.close()
            print("Loaded ",count,"weights from ",path)





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

    #(train_X, train_y), (test_X, test_y) = mnist.load_data(path="./mnist.npz")

    #print("Mnist data loaded")
    layers = []
    layers.append(Layer(2,False))
    layers.append(Layer(16,True))
    layers.append(Layer(1,True))

    net = ANN(layers)


    #input = preprocMnist(train_X)
    #output = oneHot(train_y)

    input = np.array([[0,0],[0,1],[1,0],[1,1]])
    output = np.array([[0],[1],[1],[0]])

    #print("Mnist train preprocessed")

    net.loadFromDisk("./xor.weights")

    print("Training starting...")
    start = time.process_time()

    #mSqErrBand = net.train(input,output,0.03,20000)

    end = time.process_time()
    print("Training finished...")
    print("Runtime: \t",(end-start)," s")

    #input = preprocMnist(test_X)

    #pyplot.plot(mSqErrBand)
    #pyplot.show()

    #net.persistToDisk("./xor.weights")

    net.setInputLayerData(input[0])
    net.predict()
    print(net.getOutputData())

    net.setInputLayerData(input[1])
    net.predict()
    print(net.getOutputData())

    net.setInputLayerData(input[2])
    net.predict()
    print(net.getOutputData())

    net.setInputLayerData(input[3])
    net.predict()
    print(net.getOutputData())

    # fig=pyplot.figure(figsize=(14, 10))
    #
    # for p in range(4*4):
    #     net.setInputLayerData(input[p])
    #     net.predict()
    #     result = net.getOutputData()
    #
    #     fig.add_subplot(4,8,p*2+1)
    #     pyplot.imshow(input[p].reshape(28,28))
    #
    #     fig.add_subplot(4,8,p*2+2)
    #     pyplot.imshow(result.reshape(28,28))
    #
    # print("Showing results")
    # pyplot.show()

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
