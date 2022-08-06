import numpy as np
import function

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

            #personal bst, fitness, and global best (Best Fx) are updated in "evaluate" function
            
    def evaluate(self, prob):
        for i in range(self.N):
            tmp = prob.evaluate(self.Xs[i])

            #update personal best
            if self.Fs[i] == None or tmp <= self.Fs[i]:
                self.Fs[i] = tmp
                self.Ps[i] = [self.Xs[i][j] for j in range(self.PROB_DIMEINTION)] # deep copy

            #update global best
            if self.BestFX == None or tmp <= self.BestFX:
                self.BestFX = tmp
                self.BestX = [self.Xs[i][j] for j in range(self.PROB_DIMEINTION)] # deep copy

    def generation(self):
         for i in range(self.N):
             for j in range(self.PROB_DIMEINTION):
                
                #ADD position update equation here  
                
                #Boundary treatment
                if self.Xs[i][j] > self.Xmax:
                    self.Xs[i][j] = self.Xmax
                if self.Xs[i][j] < -self.Xmax:
                    self.Xs[i][j] = -self.Xmax

    def update(self):
        # update velocity
        for i in range(self.N):
            for j in range(self.PROB_DIMEINTION):
                
                #ADD velocity update equation here 

                #Boundary treatment
                if self.Vs[i][j] > self.Vmax:
                    self.Vs[i][j] = self.Vmax
                if self.Vs[i][j] < -self.Vmax:
                    self.Vs[i][j] = -self.Vmax


# Do not change
def run(problem, optimizer, MAX_EVALUATIONS, filename):
    print("run {}".format(filename))

    evals = 0
    log   = []

    optimizer.initialization()
    optimizer.evaluate(problem)

    while evals < MAX_EVALUATIONS:
        optimizer.generation()
        optimizer.evaluate(problem)
        optimizer.update()
        evals += optimizer.N

        #logging
        print(evals, optimizer.BestFX)
        log.append([evals, optimizer.BestFX])
    np.savetxt('_out_{}.csv'.format(filename), log, delimiter=',') 


if __name__ == "__main__":
    #Basic setting (Do NOT change)
    N, MAX_EVALUATIONS, PROB_DIMEINTION, Xmax = 50, 50000, 20, 50
    PROBLEM_LIST = ["Rosenbrock", "Ackley", "Rastrigin"]
    
    #Random search setting
    RS = RandomSearch(N, Xmax, MAX_EVALUATIONS, PROB_DIMEINTION)
    for i in range(len(PROBLEM_LIST)):
        fnc = function.Function(PROBLEM_LIST[i], PROB_DIMEINTION)
        run(fnc, RS, MAX_EVALUATIONS, "RS_{}".format(PROBLEM_LIST[i]))

    #PSO setting
    W, C1, C2, Vmax = 0.4, 2.0, 2.0, 20
    PSO = ParticleSwarmOptimization(N, W, C1, C2, Xmax, Vmax, MAX_EVALUATIONS, PROB_DIMEINTION)

    for i in range(len(PROBLEM_LIST)):
        fnc = function.Function(PROBLEM_LIST[i], PROB_DIMEINTION)
        run(fnc, PSO, MAX_EVALUATIONS, "PSO_{}".format(PROBLEM_LIST[i]))
     