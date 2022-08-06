import numpy as np
import function
import os , sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

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


class DifferentialEvolution:
    def __init__(self, N, F, CR, Xmax, MAX_EVALUATIONS, PROB_DIMEINTION):
        # DE settings
        self.N    = N     # Number of particles
        self.F    = F     # Scaling factor
        self.CR   = CR    # #Crossover rate
        self.Xmax = Xmax  # Defined range of position variables (search space)

        #Problem settings
        self.MAX_EVALUATIONS = MAX_EVALUATIONS 
        self.PROB_DIMEINTION = PROB_DIMEINTION 

        #Variables
        self.Xs        = [None for _ in range(self.N)] # Population
        self.Us        = [None for _ in range(self.N)] # Offspring
        self.Fs        = [None for _ in range(self.N)] # Fitness (personal best's fitness)

        self.BestX     = None # Global best
        self.BestFX    = None # Best fitness (of global best)


    def initialization(self,m_method):
        self.m_method = m_method
        for i in range(self.N):            
            self.Xs[i] = np.random.rand(self.PROB_DIMEINTION) *  (2* self.Xmax) - self.Xmax

    def evaluate(self, prob):
        #For "initialization"
        if self.Us[0] == None:
            for i in range(self.N):
                self.Fs[i] = prob.evaluate(self.Xs[i])
        #For "generation"
        else:
            for i in range(self.N):
                _eval = prob.evaluate(self.Us[i])
                if _eval <= self.Fs[i]: #update population only if offspring is better than the corresponding solution
                    self.Fs[i] = _eval
                    self.Xs[i] = self.Us[i]
    
    def update(self):
        #identify the best solution discovered so far
        for i in range(self.N):
            if self.BestFX == None or self.Fs[i] < self.BestFX:
                self.BestX  = self.Xs[i]
                self.BestFX = self.Fs[i]

    def generation(self):
        for i in range(self.N):
            #generate mutant solution with arbitrary strategies (rand1, best1, current to best1)
            if self.m_method == "DE_rand1":
                v = self.mutation_rand1()
            elif self.m_method == "DE_best1":
                v = self.mutation_best1()
            elif self.m_method == "DE_ctobest1":
                v = self.mutation_current_to_best1(i)
            #apply binominal crossover: generate offspring based on the mutant solution and the corresponding solution Xs[i]
            u = self.crossover_binominal(self.Xs[i], v)
            #update the offspring set
            self.Us[i] = u

    def mutation_rand1(self):
        #randomly pick up three ID's solutions in Population
        choice = np.random.randint(0,self.N, size =3)
        #identify the solutions selected randomly
        x1, x2, x3 = self.Xs[choice[0]], self.Xs[choice[1]], self.Xs[choice[2]]
        
        #prepare the output array (mutant solution)
        v = [0 for _ in range(self.PROB_DIMEINTION)]
        
        #ADD rand1's operator to produce the mutant vector
        # Fの分の差分を動作させる
        for i in range(self.PROB_DIMEINTION):
            v[i] = x1[i] + self.F * (x2[i] - x3[i])
        return v

    def mutation_best1(self):
        # Determine current best solution
        self.update()
        #randomly pick up two ID's solutions in Population
        choice = np.random.randint(0,self.N, size =2)
        #identify the solutions selected randomly
        x1, x2 = self.Xs[choice[0]], self.Xs[choice[1]]
        
        #prepare the output array (mutant solution)
        v = [0 for _ in range(self.PROB_DIMEINTION)]
        
        #ADD best1's operator to produce the mutant vector
        for i in range(self.PROB_DIMEINTION):
            v[i] = self.BestX[i] + self.F * (x1[i] - x2[i])
        return v

    def mutation_current_to_best1(self, i):
        # Determine current best solution
        self.update()
        #randomly pick up two ID's solutions in Population
        choice = np.random.randint(0,self.N, size =2)
        #identify the solutions selected randomly
        x1, x2 = self.Xs[choice[0]], self.Xs[choice[1]]
        
        #prepare the output array (mutant solution)
        v = [0 for _ in range(self.PROB_DIMEINTION)]
        
        for j in range(self.PROB_DIMEINTION):
            v[j] = self.Xs[i][j] + self.F * (self.BestX[j] - self.Xs[i][j]) + self.F * (x1[j] - x2[j])
        return v

    def crossover_binominal(self, x, v):
        # one element should be forcedly changed 
        choice = np.random.randint(0, self.PROB_DIMEINTION)

        #prepare the output array (mutant solution)
        u = [x[i] for i in range(self.PROB_DIMEINTION)]

        #Add binomial crossover to produce offspring
        for i in range(self.PROB_DIMEINTION):
            if np.random.rand() < self.CR or i == choice:
                u[i] = v[i]
        return u


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

    def evaluate(self, prob):
        for i in range(self.N):
            tmp = prob.evaluate(self.Xs[i])

            if self.Fs[i] == None or tmp <= self.Fs[i]:
                self.Fs[i] = tmp
                self.Ps[i] = [self.Xs[i][j] for j in range(self.PROB_DIMEINTION)]

            if self.BestFX == None or tmp <= self.BestFX:
                self.BestFX = tmp
                self.BestX = [self.Xs[i][j] for j in range(self.PROB_DIMEINTION)]

    def generation(self):
         for i in range(self.N):
             for j in range(self.PROB_DIMEINTION):
                self.Xs[i][j] = self.Xs[i][j] + self.Vs[i][j]
                
                if self.Xs[i][j] > self.Xmax:
                    self.Xs[i][j] = self.Xmax
                if self.Xs[i][j] < -self.Xmax:
                    self.Xs[i][j] = -self.Xmax
    def update(self):
        for i in range(self.N):
            for j in range(self.PROB_DIMEINTION):
                self.Vs[i][j] = self.W * self.Vs[i][j] + self.C1 * np.random.rand()*(self.Ps[i][j] - self.Xs[i][j]) + self.C2 * np.random.rand()*(self.BestX[j] - self.Xs[i][j])
                
                if self.Vs[i][j] > self.Vmax:
                    self.Vs[i][j] = self.Vmax
                if self.Vs[i][j] < -self.Vmax:
                    self.Vs[i][j] = -self.Vmax


# Do not change
#def run(problem, optimizer, MAX_EVALUATIONS, filename):
#    print("run {}".format(filename))
#
#    evals = 0
#    log   = []
#
#    optimizer.initialization()
#    optimizer.evaluate(problem)
#
#    while evals < MAX_EVALUATIONS:
#        optimizer.generation()
#        optimizer.evaluate(problem)
#        optimizer.update()
#        evals += optimizer.N
#
#        #logging
#        print(evals, optimizer.BestFX)
#        log.append([evals, optimizer.BestFX])
#    np.savetxt('_out_{}.csv'.format(filename), log, delimiter=',') 

# Do not change
def run(problem, optimizer, MAX_EVALUATIONS, opt,filename , trial):
    print("run {}".format(filename))

    
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
        ax.plot(df_all["evals"], df_all["RS_q2"],label="RS")
        ax.fill_between(df_all["evals"], df_all["RS_q1"],  df_all["RS_q3"],alpha=0.2)
        ax.plot(df_all["evals"], df_all["DE_ctobest1_q2"],label="DE")
        ax.fill_between(df_all["evals"], df_all["DE_ctobest1_q1"],  df_all["DE_ctobest1_q3"],alpha=0.2)
        ax.plot(df_all["evals"], df_all["PSO_q2"],label="PSO")
        ax.fill_between(df_all["evals"], df_all["PSO_q1"],  df_all["PSO_q3"],alpha=0.2)
        #ax.plot(df_all["evals"], df_all["DE_ctobest1_q2"],label="ctobest1")
        #ax.fill_between(df_all["evals"], df_all["DE_ctobest1_q1"],  df_all["DE_ctobest1_q3"],alpha=0.2)
        plt.xlabel("evals")
        plt.ylabel("fit")
        plt.yscale('log')
        plt.legend()
        fig.savefig("{}.png".format(pro))
        plt.clf()
        plt.close()

if __name__ == "__main__":
    #Basic setting (Do NOT change)
    N, MAX_EVALUATIONS, PROB_DIMEINTION, Xmax = 50, 50000, 20, 50
    PROBLEM_LIST = ["Rosenbrock", "Ackley", "Rastrigin"]
    OPTIMIZER = ["RS","DE_ctobest1","PSO"]
    NUM_TRIAL = 11

    # make files
    makefiles()

    #Random search setting
    for i in range(len(PROBLEM_LIST)):
        for trial in range(NUM_TRIAL):
            np.random.seed(trial)
            RS = RandomSearch(N, Xmax, MAX_EVALUATIONS, PROB_DIMEINTION)
            fnc = function.Function(PROBLEM_LIST[i], PROB_DIMEINTION)
            run(fnc, RS, MAX_EVALUATIONS, "RS","RS_{}".format(PROBLEM_LIST[i]),trial)

    #DE setting
    F, CR = 0.9, 0.9
    for i in range(len(PROBLEM_LIST)):
        for trial in range(NUM_TRIAL):
            np.random.seed(trial)
            DE = DifferentialEvolution(N, F, CR, Xmax, MAX_EVALUATIONS, PROB_DIMEINTION)
            fnc = function.Function(PROBLEM_LIST[i], PROB_DIMEINTION)
            run(fnc, DE, MAX_EVALUATIONS, "DE_ctobest1","DE_ctobest1_{}".format(PROBLEM_LIST[i]),trial)


    #PSO setting
    W, C1, C2, Vmax = 0.4, 2.0, 2.0, 20
    for i in range(len(PROBLEM_LIST)):
        for trial in range(NUM_TRIAL):
            np.random.seed(trial)
            PSO = ParticleSwarmOptimization(N, W, C1, C2, Xmax, Vmax, MAX_EVALUATIONS, PROB_DIMEINTION)
            fnc = function.Function(PROBLEM_LIST[i], PROB_DIMEINTION)
            run(fnc, PSO, MAX_EVALUATIONS, "PSO","PSO_{}".format(PROBLEM_LIST[i]),trial)

    makegraph()