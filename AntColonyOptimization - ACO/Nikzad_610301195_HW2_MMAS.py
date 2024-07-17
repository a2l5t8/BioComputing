import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def ProcessInput(inp) : 
    f = open(inp, "r")
    ptr = 0
    data = f.read()
    data = list(map(int, data.split()))
    
    n, m = data[0], data[1]
    ptr += 2

    cost = [0]
    sets = []
    for i in range(m) : 
        cost.append(data[ptr])    
        sets.append([])
        ptr += 1

    sets.append([])

    cover = []
    for i in range(n) : 
        sz = data[ptr]
        ptr += 1
        cover.append([])
        for j in range(sz) : 
            cover[i].append(data[ptr])
            sets[data[ptr]].append(i)
            ptr += 1


    cards = []
    for i in range(len(sets)) : 
        cards.append(len(sets[i]))

    return n, m, cost, cover, sets, cards


class Ant() :

    def __init__(self, n, m, alpha, beta) : 
        self.alpha = alpha
        self.beta = beta

        self.n = n
        self.m = m

        self.sol = []
        self.coverage = set()
        self.fitness = 1e9

    def cal_fitness(self, cost) :
        res = 0
        for i in range(len(self.sol)) : 
            res += cost[self.sol[i]]

        self.fitness = res

    def H(self, i) :             
        return len(set(sets[i]).difference(self.coverage))

    def transition(self, phermone, cost, sets) :
        prob = [0]
        sm = 0
        for i in range(1, self.m + 1) : 
            prob.append(0)
            if(not i in self.sol) : 
                val = (phermone[i]**self.alpha) * ((self.H(i)/cost[i])**self.beta)
                sm += val
                prob[i] = val
        
        for i in range(1, self.m + 1) : 
            prob[i] /= sm

        nxt = random.choices([i for i in range(self.m + 1)], weights = prob, k = 1)[0]
        self.add_sol(nxt)
        
    def add_sol(self, nxt) :
        self.sol.append(nxt)
        self.coverage.update(sets[nxt])

    def update_phermone(self, phermone) : 
        for i in self.sol : 
            phermone[i] += 2/self.fitness

    def find_sol(self, phermone, cost, sets) : 
        while(not self.check_sol()) : 
            self.transition(phermone, cost, sets)
        self.cal_fitness(cost)

    def check_sol(self) : 
        return (len(self.coverage) == n)

    def max_cost(self) :
        res = 0
        for i in range(len(self.sol)) : 
            res = max(res, cost[self.sol[i]])
        return res

    def local_search(self, cost, p1 = 0.15, p2 = 1.1) : 
        N_s = len(self.sol)
        Q_s = self.max_cost()
        # W_i = cover[i]

        D = int(p1 * N_s)
        E = p2 * Q_s

        prev_sol = self.sol.copy()
        cands = [i for i in self.sol if i not in (random.sample(self.sol, D))]

        self.sol = []
        self.coverage = set()
        for i in cands : 
            self.add_sol(i)
        self.cal_fitness(cost)

        while(not self.check_sol()) : 
            recs = [i for i in range(1, len(cost)) if (cost[i] <= E and i not in self.sol)]
            bst_rec = 0
            for i in recs :
                rate = self.H(i)/cost[i]
                bst_rec = max(bst_rec, rate)
            
            ops = [i for i in recs if self.H(i)/cost[i] == bst_rec]
            k = random.sample(ops, 1)[0]
            self.add_sol(k)
        
        self.cal_fitness(cost)

    def diversification(self, NC_MAX = 0.001) : 
        base = random.sample([i for i in range(1, m + 1)], int(NC_MAX * m))
        for i in base : 
            self.add_sol(i)


def MMAS(sz, iteration, n, m, cost, sets, alpha, beta, phi, MIN = 0.0002, MAX = 0.3, log = True) :
    phermone = [0]
    for i in range(1, m + 1) : 
        phermone.append(random.random()/10)

    bst = []

    for _ in range(1, iteration + 1) : 
        local_bst = 1e9
        bst_idx = -1

        ants = [Ant(n, m, alpha, beta) for i in range(sz)]
        for ant in ants : 
            ant.diversification()
            ant.find_sol(phermone, cost, sets)

        
        ################################################################################
        ## Local Search
        ################################################################################
        
        if(_ <= iteration) :
            for ant in ants : 
                ant.local_search(cost)
        
        for i in range(1, m + 1) : 
            phermone[i] *= (1 - phi)
            
        
        
        for i in range(len(ants)) :
            ants[i].cal_fitness(cost)
            if(local_bst > ants[i].fitness) : 
                local_bst = ants[i].fitness
                bst_idx = i
            
        ################################################################################
        ## ANT COLONY 
        ################################################################################
        
        # for ant in ants : 
        #     ant.update_phermone(phermone)

        ################################################################################
        ## MMAS
        ################################################################################

        ants[bst_idx].update_phermone(phermone)

        for i in range(1, m + 1) : 
            phermone[i] = min(phermone[i], MAX)
            phermone[i] = max(phermone[i], MIN)
        
        if(log) : 
            print("iteration {} : {}".format(_, local_bst))

        bst.append(local_bst)

        
    if(log) : 
        print(max(phermone[1:]), min(phermone[1:]))
        
    return bst




################################################################################
## MAIN 
################################################################################

TEST_CASES = ["scp41.txt", "scp51.txt", "scp54.txt", "scpa2.txt", "scpb1.txt"]

for TEST in TEST_CASES : 
    n, m, cost, cover, sets, cards = ProcessInput(TEST)

    ave = 0
    bst = 1e9
    for _ in range(10) : 
        rec = MMAS(sz = 15,iteration = 15, n = n, m = m, cost = cost, sets = sets, alpha = 1, beta = 6, phi = 0.6, log = False)
        ave += min(rec)
        bst = min(bst, min(rec))

    print("Average score : {}, Best Score : {}".format(ave, bst))