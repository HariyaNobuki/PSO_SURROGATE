import math  as mt
import numpy as np

class Function:

    """ constructor """
    # initialize method
    def __init__(self, PROB_NAME, PROB_DIMEINTION,):

        """ instance variable """
        self.PROB_DIMEINTION = PROB_DIMEINTION
        self.axis_range     = [0, 0]

        # choice of functions
        if PROB_NAME == "Rosenbrock":
            self.evaluate = self.Rosenbrock
            self.axis_range  = [-50, 50]
        elif PROB_NAME == "Ackley":
            self.evaluate = self.Ackley
            self.axis_range  = [-50, 50]
        elif PROB_NAME == "Rastrigin":
            self.evaluate = self.Rastrigin
            self.axis_range  = [-50, 50]

    """ instance method """
    # evaluation
    def doEvaluate(self, x):
        if not len(x) == self.PROB_DIMEINTION:
            print("Error: Solution X is not a {}-d vector".format(self.PROB_DIMEINTION))
            return None

        return self.evaluate(x)

    # Rosenbrock
    def Rosenbrock(self, x):
        ret = 0.
        for i in range(self.PROB_DIMEINTION - 1):
            ret += 100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2
        return ret

    # Ackley
    def Ackley(self, x):
        sum_1 = 0.
        sum_2 = 0.
        for i in range(self.PROB_DIMEINTION):
            sum_1 += x[i] * x[i]
            sum_2 += np.cos(2 * np.pi * x[i])
        ret = -20 * np.exp(-0.2 * np.sqrt(sum_1 / self.PROB_DIMEINTION)) - np.exp(sum_2 / self.PROB_DIMEINTION) + 20 + np.e
        return ret

    # Rastrigin
    def Rastrigin(self, x):
        ret = 0.
        for i in range(self.PROB_DIMEINTION):
            ret += x[i]**2 - 10 *mt.cos( 2 * mt.pi * x[i] ) + 10
        return ret