import pygame
import numpy as np
import random
import time
import math



input_dim = 7      # 5 inputs - angle, distance, left, middle, right surroundings
hidden_dim = 100     # hidden units
output_dim = 3      # 0: Left, 1: Straight, 2: Right

class NeuralNet:

    def sigmoidFunction(self, x, derivative=False):
        #definition of sigmoid function
        func = 1 / (1.0 + np.exp(-x))
        if (derivative == False):
            return func
        # derivative of sigmoid function
        else:
            return func * (1.0 - func)

    def __init__(self):
        self.l0 = np.ones((1,input_dim))
        self.l0_bias = self.l0

        self.l1 = np.ones((1, hidden_dim))
        self.l1_bias = self.l1

        self.l2 =np.ones((1, hidden_dim))
        self.l2_bias = self.l2

        self.l3 =np.ones((1, output_dim))

        self.syn0 = self.syn0 = 2 * np.random.random((input_dim+1,hidden_dim)) - 1
        self.syn1 = 2 * np.random.random((hidden_dim + 1, hidden_dim)) - 1
        self.syn2 = 2 * np.random.random((hidden_dim + 1, output_dim)) - 1

        self.fitness = 0
    def forwardFeed(self, data):
        # convert l1 and l2 into matrices (currently they are vectors)
        self.l0 = np.reshape(data, (-1,input_dim))
        # normalize input
        self.l0 = (self.l0-np.mean(self.l0))/(np.std(self.l0)+1)

        self.l0_bias = np.insert(self.l0, 0, values=0, axis=1)

        self.l1 = self.sigmoidFunction(self.l0_bias.dot(self.syn0))
        self.l1_bias = np.append(self.l1,np.ones((1,1)), axis=1)


        self.l2 = self.l2.reshape(self.l2.shape[0], -1)
        self.l2 = self.sigmoidFunction(self.l1_bias.dot(self.syn1))
        self.l2_bias = np.append(self.l2,np.ones((1,1)), axis=1)

        self.l3 = self.l3.reshape(self.l2.shape[0], -1)
        self.l3 = self.sigmoidFunction(self.l2_bias.dot(self.syn2))

        #print(self.l2)
        h = np.argmax(self.l3)
        return h
        ## # print("new guess is " + str(h))
