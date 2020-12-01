#TODO:
#Create class containing weights as matricies
#declare methods for forward pass and backpropagation

import numpy as np
import math

class ANN:

    def __init__(this, inputD, outputD,  hiddenD, hiddenL):
        this.inputD = inputD
        this.outputD = outputD
        this.hiddenD = hiddenD
        this.hiddenL = hiddenL

        this.create()

    def create(this):

        this.weightsIH = np.array(np.random.random_sample((this.inputD,this.hiddenD))*2-1)
        this.weightsHO = np.array(np.random.random_sample((this.hiddenD,this.outputD))*2-1)
        this.bias = np.array(np.random.random_sample(this.hiddenD)*2-1)
        this.input = np.zeros(this.inputD)
        this.hidden = np.zeros(this.hiddenD)
        this.output = np.zeros(this.outputD)

    def printWeights(this):
        print("Input to hidden: \n",this.weightsIH)
        print("Hidden to output: \n",this.weightsHO)
        print("Hidden bias: \n",this.bias)

    def printNodes(this):
        print("Input: \n",this.input)
        print("Hidden: \n",this.hidden)
        print("Output: \n",this.output)

    def printOutput(this):
        print("Output: \n",this.output)

    def predict(this, input):
        this.input = input
        this.forward()

    def forward(this):
        this.hidden = np.zeros(this.hiddenD)
        for x in range(this.hiddenD):
            for y in range(this.inputD):
                this.hidden[x] += this.input[y]*this.weightsIH[y][x]

        this.hidden = this.addBias(this.hidden)
        this.hidden = this.activate(this.hidden)
        this.output = np.zeros(this.outputD)
        for x in range(this.outputD):
            for y in range(this.hiddenD):
                this.output[x] += this.hidden[y]*this.weightsHO[y][x]
        this.output = this.activate(this.output)

    def addBias(this, nodes):
        for x in range(nodes.size):
            nodes[x] += this.bias[x]
        return nodes

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
        while(this.epoch > 0):
            this.epoch -= 1
            ind = math.floor(np.random.random()*(input.shape[0]))
            count[ind] +=1
            this.predict(input[ind])
            loss = this.output - solution[ind]
            print("Epoch: ",this.epoch," Input: ",input[ind]," Output: ",this.output," Solution: ",solution[ind]," Error: ",loss)

            # Now for the hidden layer.
            hidden_deltas = [0.0]*this.hiddenD
            for j in range(this.hiddenD):
                error=0.0
                for k in range(this.outputD):
                    error+=this.weightsHO[j][k] * loss * this.derivActivate(this.output)
                hidden_deltas[j] = error * this.derivActivate(this.hidden[j])

            # Change weights of hidden to output layer accordingly.
            for j in range(this.hiddenD):
                for k in range(this.outputD):
                    delta_weight = this.hidden[j] * loss * this.derivActivate(this.output)
                    this.weightsHO[j][k] -= learningRate * delta_weight



            for i in range(this.inputD):
                for j in range(this.hiddenD):
                    delta_weight = hidden_deltas[j] * this.input[i]
                    this.weightsIH[i][j] -= learningRate*delta_weight


            for i in range(this.hiddenD):
                    delta_weight = hidden_deltas[i] * 1
                    this.bias[i] -= learningRate * delta_weight


    def loss(this, solution):
        loss = 0.5*(pow((solution - this.output),2))
        return loss




if __name__ == "__main__":
    net = ANN(2,1,6,1)
    input = np.array([[0,0],[0,1],[1,0],[1,1]])
    output = np.array([[0],[1],[1],[0]])
    #net.predict(input)
    net.train(input,output,0.2,10000)
    net.predict([0,0])
    net.printOutput()

    net.predict([0,1])
    net.printOutput()

    net.predict([1,0])
    net.printOutput()

    net.predict([1,1])
    net.printOutput()
