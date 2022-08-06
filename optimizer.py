import numpy as np
import function
import os , sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.spatial import distance
from pyDOE2 import lhs
from scipy.spatial import distance
from scipy.stats import rankdata
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor  # randf

class RandomSearch:

    def __init__(self, N, Xmax, MAX_EVALUATIONS, PROB_DIMEINTION):
        #RS settings
        self.N    = N
        self.Xmax = Xmax

        #Problem settings
        self.MAX_EVALUATIONS = MAX_EVALUATIONS
        self.PROB_DIMEINTION = PROB_DIMEINTION
        
        #Private variables
        self.Xs = [None for _ in range(self.N)]
        self.Fs = [None for _ in range(self.N)]
        self.BestX = None
        self.BestFX = None

    def initialization(self):
        for i in range(self.N):            
            self.Xs[i] = np.random.rand(self.PROB_DIMEINTION) *  (2* self.Xmax) - self.Xmax

    def evaluate(self, tsp):
        for i in range(self.N):
            self.Fs[i] = tsp.evaluate(self.Xs[i])

    def update(self):
        for i in range(self.N):
            if self.BestFX == None or self.Fs[i] < self.BestFX:
                self.BestX  = self.Xs[i]
                self.BestFX = self.Fs[i]

    def generation(self):
        self.initialization()


class ParticleSwarmOptimization:

    def __init__(self, N, W, C1, C2, Xmax, Vmax, MAX_EVALUATIONS, PROB_DIMEINTION):
        # PSO settings
        self.N    = N     # Number of particles
        self.W    = W     # Inertia Weight
        self.C1   = C1    # Cognitive Learning factor
        self.C2   = C2    # Social Learning factor
        self.Vmax = Vmax  # Maximum velocity
        self.Xmax = Xmax  # Defined range of position variables (search space)

        #Problem settings
        self.MAX_EVALUATIONS = MAX_EVALUATIONS 
        self.PROB_DIMEINTION = PROB_DIMEINTION 

        #Particle variables
        self.Xs        = [None for _ in range(self.N)] # Position
        self.Fs        = [None for _ in range(self.N)] # Fitness (personal best's fitness)
        self.Ps        = [None for _ in range(self.N)] # Personal best's position
        self.Vs        = [None for _ in range(self.N)] # Velocity

        self.BestX     = None # Global best
        self.BestFX    = None # Best fitness (of global best)

        ARCHIVE_X = np.array([])
        ARCHIVE_F = np.array([])
        ARCHIVE_POP = []
        ARCHIVE_DIV = []
        fit_list = []
        fit_dict = pd.DataFrame()

        aVar = ARCHIVE_X
        aObj = ARCHIVE_F
        data = np.vstack((aVar.T, aObj.T))
        rbf = Rbf(*data, function='cubic')

        
    def initialization(self):
        for i in range(self.N):            
            self.Xs[i] = np.random.rand(self.PROB_DIMEINTION) *  (2* self.Xmax) - self.Xmax
            self.Vs[i] = np.random.rand(self.PROB_DIMEINTION) *  (2* self.Vmax) - self.Vmax

    def evaluate(self, prob):
        for i in range(self.N):
            # ind evaluate
            tmp = prob.evaluate(self.Xs[i])
            # ind update
            if self.Fs[i] == None or tmp <= self.Fs[i]:
                self.Fs[i] = tmp
                self.Ps[i] = [self.Xs[i][j] for j in range(self.PROB_DIMEINTION)]
            # pop update
            if self.BestFX == None or tmp <= self.BestFX:
                self.BestFX = tmp
                self.BestX = [self.Xs[i][j] for j in range(self.PROB_DIMEINTION)]

    def generation(self):
         for i in range(self.N):
             for j in range(self.PROB_DIMEINTION):
                self.Xs[i][j] = self.Xs[i][j] + self.Vs[i][j]
                # bounding 
                if self.Xs[i][j] > self.Xmax:
                    self.Xs[i][j] = self.Xmax
                if self.Xs[i][j] < -self.Xmax:
                    self.Xs[i][j] = -self.Xmax

    def update(self):   # velocity update
        for i in range(self.N):
            for j in range(self.PROB_DIMEINTION):
                self.Vs[i][j] = self.W * self.Vs[i][j] + self.C1 * np.random.rand()*(self.Ps[i][j] - self.Xs[i][j]) + self.C2 * np.random.rand()*(self.BestX[j] - self.Xs[i][j])
                
                if self.Vs[i][j] > self.Vmax:
                    self.Vs[i][j] = self.Vmax
                if self.Vs[i][j] < -self.Vmax:
                    self.Vs[i][j] = -self.Vmax


# Do not change
def run(problem, optimizer, MAX_EVALUATIONS, opt,filename , trial):
    evals = 0
    log   = []

    if opt == "RS" or opt == "PSO":
        optimizer.initialization()
    else:
        optimizer.initialization(opt)
    optimizer.evaluate(problem)

    while evals < MAX_EVALUATIONS:
        optimizer.generation()
        optimizer.evaluate(problem)
        optimizer.update()
        evals += optimizer.N

        #logging
        print(evals, optimizer.BestFX)
        log.append([evals, optimizer.BestFX])
    np.savetxt('{}/_out_{}_{}.csv'.format(opt,filename,trial), log, delimiter=',') 

def makefiles():
    for opt in OPTIMIZER:
        os.makedirs(opt , exist_ok=True)

def makegraph():
    df_all = pd.DataFrame()

    for pro in PROBLEM_LIST:
        for opt in OPTIMIZER:
            df_fit = pd.DataFrame()
            t = 0
            for trial in range(NUM_TRIAL):
                l = glob.glob('{}/*{}_{}_{}.csv'.format(opt,opt,pro,trial))
                df = pd.read_csv(l[0] , names=('evals', 'fit'))
                if t == 0:
                    df_all["evals"] = df["evals"]
                    t += 1
                df_fit["fit_{}".format(trial)] = df["fit"]
            q1 = df_fit.quantile(0.25,axis=1)
            q2 = df_fit.quantile(0.5,axis=1)
            q3 = df_fit.quantile(0.75,axis=1)
            df_all["{}_q1".format(opt)] = q1
            df_all["{}_q2".format(opt)] = q2
            df_all["{}_q3".format(opt)] = q3

        fig, ax = plt.subplots()
        # RS  plot
        ax.plot(df_all["evals"], df_all["RS_q2"],label="RS")
        ax.fill_between(df_all["evals"], df_all["RS_q1"],  df_all["RS_q3"],alpha=0.2)
        # PSO  plot
        ax.plot(df_all["evals"], df_all["PSO_q2"],label="PSO")
        ax.fill_between(df_all["evals"], df_all["PSO_q1"],  df_all["PSO_q3"],alpha=0.2)

        plt.xlabel("evals")
        plt.ylabel("fit")
        plt.yscale('log')
        plt.legend()
        fig.savefig("{}.png".format(pro))
        plt.clf()
        plt.close()

if __name__ == "__main__":
    #Basic setting (Do NOT change)
    N, MAX_EVALUATIONS, PROB_DIMEINTION, Xmax = 50, 5000, 20, 50
    PROBLEM_LIST = ["Rosenbrock", "Ackley", "Rastrigin"]
    OPTIMIZER = ["RS","PSO"]
    NUM_TRIAL = 5

    # make files
    makefiles()

    #Random search setting
    for i in range(len(PROBLEM_LIST)):
        for trial in range(NUM_TRIAL):
            np.random.seed(trial)
            RS = RandomSearch(N, Xmax, MAX_EVALUATIONS, PROB_DIMEINTION)
            fnc = function.Function(PROBLEM_LIST[i], PROB_DIMEINTION)
            run(fnc, RS, MAX_EVALUATIONS, "RS","RS_{}".format(PROBLEM_LIST[i]),trial)


    #PSO setting
    W, C1, C2, Vmax = 0.4, 2.0, 2.0, 20
    for i in range(len(PROBLEM_LIST)):
        for trial in range(NUM_TRIAL):
            np.random.seed(trial)
            PSO = ParticleSwarmOptimization(N, W, C1, C2, Xmax, Vmax, MAX_EVALUATIONS, PROB_DIMEINTION)
            fnc = function.Function(PROBLEM_LIST[i], PROB_DIMEINTION)
            run(fnc, PSO, MAX_EVALUATIONS, "PSO","PSO_{}".format(PROBLEM_LIST[i]),trial)

    makegraph()