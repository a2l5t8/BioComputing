import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import copy

class Particle() : 

    def __init__(self, dim, MIN, MAX, fitness) : 
        self.dim = dim
        self.MIN = MIN
        self.MAX = MAX
        self.fit_function = fitness

        self.pos = [random.uniform(self.MIN, self.MAX) for i in range(self.dim)]
        self.vel = [random.uniform(self.MIN, self.MAX) for i in range(self.dim)]

        self.fitness = self.fit_function(self.pos[0], self.pos[1])

        self.bst_pos = self.pos.copy()
        self.bst_fitness = self.fitness


def PSO(target, intertia_coef = 0.3, cog_coef = 1, soci_coef = 0.9 , iterations = 1000, population_size = 500, MIN = -10, MAX = +10, dim = 2) : 
    swarm = [Particle(dim, MIN, MAX, target) for i in range(population_size)]

    bst_particle = copy.copy(swarm[0])
    for i in range(population_size) :
        if(swarm[i].fitness < bst_particle.fitness) : 
            bst_particle = copy.copy(swarm[i])

    for _ in range(iterations) : 
        for i in range(population_size) :

            for d in range(dim) : 
                r1 = random.random()
                r2 = random.random()

                swarm[i].vel[d] = ( 
                                 (intertia_coef * swarm[i].vel[d]) +
                                 (cog_coef * r1 * (swarm[i].bst_pos[d] - swarm[i].pos[d])) + 
                                 (soci_coef * r2 * (bst_particle.pos[d] - swarm[i].pos[d])) 
                               )  
            
            for d in range(dim) : 
                swarm[i].vel[d] += swarm[i].vel[d]

                swarm[i].pos[d] = max(swarm[i].pos[d], MIN)
                swarm[i].pos[d] = min(swarm[i].pos[d], MAX)

            swarm[i].fitness = swarm[i].fit_function(swarm[i].pos[0], swarm[i].pos[1])

            if(swarm[i].fitness < swarm[i].bst_fitness) : 
                swarm[i].bst_fitness = swarm[i].fitness
                swarm[i].bst_pos = swarm[i].pos.copy()

            if(swarm[i].fitness < bst_particle.fitness) : 
                bst_particle = copy.copy(swarm[i])
    
    return bst_particle


##############################################################################################################
##############################################################################################################
##############################################################################################################
####################################################   TEST   ################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################



##############################################################################################################
##################################################   CASE (1)   ##############################################
##############################################################################################################
def f(x, y) : 
    return -abs(np.sin(x) * np.cos(y) * np.exp(abs(1 - np.sqrt(x**2 + y**2)/np.pi)))

sol = None
while(True) : 
    res = PSO(target = f)
    if(abs(res.fitness) > 19.2) :
        sol = res
        break
    

print(abs(sol.fitness))



##############################################################################################################
##################################################   CASE (2)   ##############################################
##############################################################################################################

def g(x, y) : 
    return (x*sin(np.pi * np.cos(x) * np.tan(y)) * np.sin(y/x))/(1 + np.cos(y/x))

sol = None
while(True) : 
    res = PSO(target = f, MIN = -100, MAX = +100)
    if(res.fitness < -1.7*1e6) :
        sol = res
        break
    

print(sol.fitness)
print(sol.pos)