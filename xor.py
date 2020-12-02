import numpy as np
import math

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

    def derivActivate(this, node):
        node = node*(1-node)
        return node

    def train(this, input, solution, learningRate, epoch):
        this.learningRate = learningRate
        this.epoch = epoch

        for ep in range(epoch):
            # Choose sample to learn
            ind = math.floor(np.random.random()*(input.shape[0]))

            # predict sample
            this.setInputLayerData(input[ind])
            this.predict()
            result = this.outputLayer.getNodes()
            mSqErr = this.loss(solution[ind],result)

            #calculate difference between label and prediction
            loss = np.subtract(result,solution[ind])

            print("Epoch: ",ep," Input: ",input[ind]," Output: ",np.round(result,2)," Solution: ",solution[ind]," Error: ",np.round(mSqErr,4))

            for l in range(this.layerLength-1,0,-1): #go throught all layers back to front (l is *TO* layer)
                weightLayerIndex = l-1 # index of the weightlayer containing from - to weights
                activeWeightSet = this.weights[weightLayerIndex]
                toNodes = this.layers[l].getNodes()
                fromNodes = this.layers[l-1].getNodes()

                # Change weights between from/to layer
                for f in range(activeWeightSet.shape[0]):
                    for t in range(activeWeightSet.shape[1]):
                        delta_weight = fromNodes[f] * loss[t] * this.derivActivate(toNodes[t])
                        activeWeightSet[f][t] -= learningRate * delta_weight

                # update the bias weights if the layer has any
                if(this.layers[l].hasBias()):
                    biasWeight = this.layers[l].getBias()
                    for t in range(activeWeightSet.shape[1]):
                            delta_weight = 1 * loss[t] * this.derivActivate(toNodes[t])
                            biasWeight[t] -= learningRate * delta_weight

                # Update the loss for the next iteration
                propagatedLoss = [0.0]*activeWeightSet.shape[0]
                for f in range(activeWeightSet.shape[0]):
                    for t in range(activeWeightSet.shape[1]):
                        propagatedLoss[f] += activeWeightSet[f][t] * loss[t]

                loss = this.normalize(propagatedLoss)

    def loss(this, solution ,output):
        loss = 0.5*(pow(np.subtract(output,solution),2))
        return loss

    def normalize(this, array):
        biggestElement = 0
        for x in range(len(array)):
            biggestElement = abs(array[x]) if abs(array[x])>biggestElement else biggestElement

        for x in range(len(array)):
            array[x] = array[x]/biggestElement

        return array



if __name__ == "__main__":

    l1 = Layer(2,False)
    l2 = Layer(10,True)
    l3 = Layer(10,True)
    l4 = Layer(1,False)

    layers = []
    layers.append(l1)
    layers.append(l2)
    layers.append(l3)
    layers.append(l4)

    net = ANN(layers)
    #net.printWeights()

    input = np.array([[0,0],[0,1],[1,0],[1,1]])
    output = np.array([[0],[1],[1],[0]])
    net.train(input,output,0.2,10000)
    net.printWeights()

    #net.predict(input)
    #net.train(input,output,0.2,10000)
    net.setInputLayerData([0,0])
    net.predict()
    print(net.getOutputData())

    net.setInputLayerData([0,1])
    net.predict()
    print(net.getOutputData())

    net.setInputLayerData([1,0])
    net.predict()
    print(net.getOutputData())

    net.setInputLayerData([1,1])
    net.predict()
    print(net.getOutputData())
