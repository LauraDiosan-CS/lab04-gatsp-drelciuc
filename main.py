import os
from random import randint

import networkx as nx
import tsplib95 as tsp
import numpy as np

from Chromosome import Chromosome


def read():
    f = open("D:\FACULTATE\AI\LAB4\data\\100p_fricker26.txt", "r")
    lines = f.readlines()
    net={}

    n=int(lines[0])
    net['noNodes']=n
    mat=[]
    for i in range(1, n + 1):
        mat.append([int(j.rstrip()) for j in lines[i].split(',')])

    net['mat']=mat

    return net

def readTSP(file_name_input):
    tsp_problem = tsp.load_problem(file_name_input)
    G = tsp_problem.get_graph()
    n = len(G.nodes())
    net = {}
    net['noNodes'] = n
    matrix = nx.to_numpy_matrix(G)
    net['mat'] = matrix
    return net


def fcEvalTSP(communities,param):
    noNodes = param['noNodes']
    mat = param['mat']
    distanta = 0
    firsNode = communities[0]
    nextNode = communities[0]
    for i in range(0, noNodes):
        distanta += mat.item((nextNode, communities[i]))
        nextNode = communities[i]
    distanta += mat.item((noNodes - 1, firsNode))
    return distanta

def fcEval(communities,param):

    noNodes = param['noNodes']
    mat = param['mat']
    distanta = 0
    firsNode = communities[0]
    nextNode = communities[0]
    for i in range(0, noNodes):
        distanta += mat[nextNode][communities[i]]
        nextNode = communities[i]
    distanta += mat[communities[noNodes - 1]][firsNode]
    return distanta



class GA:
    def __init__(self, param=None, problParam=None):
        self.__param = param
        self.__problParam = problParam
        self.__population = []

    @property
    def population(self):
        return self.__population

    def initialisation(self):
        for _ in range(0, self.__param['popSize']):
            c = Chromosome(self.__problParam)

            self.__population.append(c)

    def evaluation(self):
        for c in self.__population:
            c.fitness = self.__problParam['function'](c.repres, net)

    def worstChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if (c.fitness > best.fitness):
                best = c
        return best

    def bestChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if (c.fitness < best.fitness):
                best = c
        return best

    def selection(self):
        pos1 = randint(0, self.__param['popSize'] - 1)
        pos2 = randint(0, self.__param['popSize'] - 1)
        if (self.__population[pos1].fitness < self.__population[pos2].fitness):
            return pos1
        else:
            return pos2



    def oneGenerationElitism(self):
        newPop = [self.bestChromosome()]
        for _ in range(self.__param['popSize'] - 1):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            newPop.append(off)
        self.__population = newPop
        self.evaluation()







#net=read()
net=readTSP("D:\FACULTATE\AI\LAB4\data\\150p_eil51.txt")

MIN=1

MAX=net["noNodes"]
noDim=net["noNodes"]

gaParam = {'popSize' : 100, 'noGen' : 500}

problParam = {'min' : MIN, 'max' : MAX, 'function' : fcEvalTSP, 'noDim' : noDim, 'noBits' : 8,'noNodes': noDim}



allBestFitnesses = []
allAvgFitnesses = []
generations = []
ga = GA(gaParam, problParam)
ga.initialisation()
ga.evaluation()
maximFitness = 999999
bestRepres = []
fileName_output = "out.txt"
f = open(fileName_output, 'w')


for g in range(gaParam['noGen']):

    bestSolX = ga.bestChromosome().repres
    bestSolY = ga.bestChromosome().fitness
    if bestSolY < maximFitness:
        maximFitness = bestSolY
        bestRepres = bestSolX
    allBestFitnesses.append(bestSolY)
    ga.oneGenerationElitism()

    bestChromo = ga.bestChromosome()
    f.write('Best solution in generation ' + str(g) + ' is: x = ' + str(bestChromo.repres) + ' f(x) = ' + str(
        bestChromo.fitness))
    f.write('\n')
f.write("Best repres&fitness: " + str(bestRepres) + " " + str(maximFitness))
f.write('\n')
f.close()


"""""
crtDir = os.getcwd()
filePath = os.path.join(crtDir, 'data', 'test.gml')
network = readGML(filePath)
print(network)
"""
