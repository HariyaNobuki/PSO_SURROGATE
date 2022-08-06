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
        
    def initialization(self):
        for i in range(self.N):            
            self.Xs[i] = np.random.rand(self.PROB_DIMEINTION) *  (2* self.Xmax) - self.Xmax
            self.Vs[i] = np.random.rand(self.PROB_DIMEINTION) *  (2* self.Vmax) - self.Vmax

    def init_evaluate(self, prob):
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


class RBF_ParticleSwarmOptimization:

    def __init__(self, N, W, C1, C2, Xmax, Vmax, MAX_EVALUATIONS, PROB_DIMEINTION):
        # PSO settings
        self.N    = N     # Number of particles
        self.W    = W     # Inertia Weight
        self.C1   = C1    # Cognitive Learning factor
        self.C2   = C2    # Social Learning factor
        self.Vmax = Vmax  # Maximum velocity
        self.Xmax = Xmax  # Defined range of position variables (search space)
        self.SUR = 200

        #Problem settings
        self.MAX_EVALUATIONS = MAX_EVALUATIONS 
        self.PROB_DIMEINTION = PROB_DIMEINTION 

        #Particle variables
        self.Xs        = [None for _ in range(self.N)] # Position
        self.Fs        = [None for _ in range(self.N)] # Fitness (personal best's fitness)
        self.Ps        = [None for _ in range(self.N)] # Personal best's position
        self.Vs        = [None for _ in range(self.N)] # Velocity
        self.CAND      = [None for _ in range(self.N)]

        self.BestX     = None # Global best
        self.BestFX    = None # Best fitness (of global best)

        self.ARCHIVE_X = np.array([])
        self.ARCHIVE_F = np.array([])
        self.ARCHIVE_POP = []
        self.ARCHIVE_DIV = []
        self.fit_list = []
        self.fit_dict = pd.DataFrame()
        
    def initialization(self):
        for i in range(self.N):            
            self.Xs[i] = np.random.rand(self.PROB_DIMEINTION) *  (2* self.Xmax) - self.Xmax
            self.Vs[i] = np.random.rand(self.PROB_DIMEINTION) *  (2* self.Vmax) - self.Vmax
            self.CAND[i] = np.random.rand(self.PROB_DIMEINTION) *  (2* self.Vmax) - self.Vmax

    def init_evaluate(self, prob):
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
            
        for i in range(self.N):
            if self.ARCHIVE_X.shape == (0,):
                self.ARCHIVE_X = np.expand_dims(self.Xs[i],0)
                self.ARCHIVE_F = np.append(self.ARCHIVE_F ,self.Fs[i])
                self.ARCHIVE_DIV.append([np.quantile(self.ARCHIVE_F,0),np.quantile(self.ARCHIVE_F,0.25),np.quantile(self.ARCHIVE_F,0.50),np.quantile(self.ARCHIVE_F,0.75),np.quantile(self.ARCHIVE_F,1,00)])
            else:
                self.ARCHIVE_X = np.append(self.ARCHIVE_X, np.array([self.Xs[i]]),axis=0)
                self.ARCHIVE_F = np.append(self.ARCHIVE_F , self.Fs[i])
                self.ARCHIVE_DIV.append([np.quantile(self.ARCHIVE_F,0),np.quantile(self.ARCHIVE_F,0.25),np.quantile(self.ARCHIVE_F,0.50),np.quantile(self.ARCHIVE_F,0.75),np.quantile(self.ARCHIVE_F,1,00)])

        aVar = self.ARCHIVE_X
        aObj = self.ARCHIVE_F
        data = np.vstack((aVar.T, aObj.T))
        self.rbf = Rbf(*data, function='cubic')
    
    def localrainforcements(self):
        # Velocity screening
        for i in range(self.N): # candidate pop
            candV = np.array([])
            candX = np.array([])
            for sur in range(self.SUR): # screening
                surX = []
                for j in range(self.PROB_DIMEINTION):
                    surres = self.W * self.Vs[i][j] + self.C1 * np.random.rand()*(self.Ps[i][j] - self.Xs[i][j]) + self.C2 * np.random.rand()*(self.BestX[j] - self.Xs[i][j])
                    
                    if surres > self.Vmax:
                        surres = self.Vmax
                    if surres < -self.Vmax:
                        surres = -self.Vmax
                    surX.append(surres)

                if candV.shape == (0,):
                    candV = np.expand_dims(surX,0)
                else:
                    candV = np.append(candV, np.array([surX]),axis=0)

                candX_pre = self.Xs[i] + surX
                candX_pre = np.clip(candX_pre, -self.Xmax, self.Xmax)
                if candX.shape == (0,):
                    candX = np.expand_dims(candX_pre,0)
                else:
                    candX = np.append(candX, np.array([candX_pre]),axis=0)

            rbf_fitness =  self.rbf(*(candX.T))
            min_idx = np.argmin(rbf_fitness)

            self.Xs[i] = candX[min_idx]
            self.Vs[i] = candV[min_idx]

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

            if (np.sum(self.ARCHIVE_X == self.Xs[i] , axis=1) == 20).any():
                print("BUT")
                continue
            else:   # add ARCHIVE
                if (np.sum(self.ARCHIVE_X == self.Xs[i] , axis=1) == 20).any():
                    print("BUT")
                    continue
                else:
                    self.ARCHIVE_X = np.append(self.ARCHIVE_X, np.array([self.Xs[i]]),axis=0)
                    self.ARCHIVE_F = np.append(self.ARCHIVE_F , tmp)
            
        aVar = self.ARCHIVE_X
        aObj = self.ARCHIVE_F
        data = np.vstack((aVar.T, aObj.T))
        self.rbf = Rbf(*data, function='cubic')


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

    optimizer.initialization()
    optimizer.init_evaluate(problem) # init evaluate

    while evals < MAX_EVALUATIONS:
        optimizer.update()          # update V
        optimizer.generation()      # update X
        if opt == "RBF_PSO":
            for i in range(1):
                optimizer.localrainforcements()

        optimizer.evaluate(problem)

        evals += optimizer.N    # 外回りで書くから微妙やな

        #logging
        print("EVALS : ",evals ,end='')
        print("BEST : ", optimizer.BestFX)
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
        ax.plot(df_all["evals"], df_all["RBF_PSO_q2"],label="RBF_PSO")
        ax.fill_between(df_all["evals"], df_all["RBF_PSO_q1"],  df_all["RBF_PSO_q3"],alpha=0.2)
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
    N, MAX_EVALUATIONS, PROB_DIMEINTION, Xmax = 10, 500, 20, 50
    PROBLEM_LIST = ["Rosenbrock", "Ackley", "Rastrigin"]
    OPTIMIZER = ["PSO","RBF_PSO"]
    NUM_TRIAL = 11

    # make files
    makefiles()

    # RBF_PSO setting
    W, C1, C2, Vmax = 0.4, 2.0, 2.0, 20
    for i in range(len(PROBLEM_LIST)):
        for trial in range(NUM_TRIAL):
            np.random.seed(trial)
            PSO = RBF_ParticleSwarmOptimization(N, W, C1, C2, Xmax, Vmax, MAX_EVALUATIONS, PROB_DIMEINTION)
            fnc = function.Function(PROBLEM_LIST[i], PROB_DIMEINTION)
            run(fnc, PSO, MAX_EVALUATIONS, "RBF_PSO","RBF_PSO_{}".format(PROBLEM_LIST[i]),trial)

    #PSO setting
    W, C1, C2, Vmax = 0.4, 2.0, 2.0, 20
    for i in range(len(PROBLEM_LIST)):
        for trial in range(NUM_TRIAL):
            np.random.seed(trial)
            PSO = ParticleSwarmOptimization(N, W, C1, C2, Xmax, Vmax, MAX_EVALUATIONS, PROB_DIMEINTION)
            fnc = function.Function(PROBLEM_LIST[i], PROB_DIMEINTION)
            run(fnc, PSO, MAX_EVALUATIONS, "PSO","PSO_{}".format(PROBLEM_LIST[i]),trial)

    makegraph()