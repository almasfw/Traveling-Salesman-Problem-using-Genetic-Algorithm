# -*- coding: utf-8 -*-
"""
Created on Sun May  3 19:25:43 2020

@author: Almas Fauzia
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

#function to create initial population with the first data as initial point
def createPopulation(cities, populationSize):
    population = []
    initialPoint = np.array([0])
    for i in range(populationSize):
        temp = np.array(random.sample(cities, len(cities)))
        temp = np.delete(temp, np.where(temp == 0))
        temp = np.concatenate([initialPoint, temp])
        population.append(np.concatenate([temp, initialPoint]))
    
    return np.array(population)

#function to calculate distance, fitness, and sort population based on them
def calculateDistanceFitness(dMatrix, population):
    distance = []
    fitness = []
    nCity = dMatrix.shape[0]
    d = 0
    for route in population:
        for i in range(nCity-1):
            city1 = route[i]
            city2 = route[i+1]
            d += dMatrix[int(city1)][int(city2)]
        distance.append(d)
        fitness.append(1/d)
        d = 0
    
    distance = np.array(distance)
    fitness = np.array(fitness)
    
    sortedDistance = np.sort(distance)
    sortedFitness = -(np.sort(-1*fitness))
    sortIndex = distance.argsort()
    sortedPopulation = population[sortIndex]
    
    return sortedPopulation, sortedDistance, sortedFitness

#function to create mating pool (group of individu that is potential to be parents)
def matingPool(population, fitness, eliteSize):
    mates = []
    nPopulation = population.shape[0]
    cum_sum = np.cumsum(fitness)/np.sum(fitness)
    for i in range(eliteSize):
        mates.append(population[i])
    for i in range(nPopulation - eliteSize):
        pick = random.random()
        for j in range(nPopulation):
            if pick <= cum_sum[j]:
                mates.append(population[i])
                break
    
    return np.array(mates)

#function to breed between 2 parents
def breeding(parents1, parents2):
    nGen = parents1.shape[0]
    index1 = random.randint(1, nGen-1)
    index2 = random.randint(1, nGen-1)
    child = -1 * np.ones(nGen)
    
    start = min(index1, index2)
    end = max(index1, index2)
    
    for i in range(start, end+1):
        child[i] = parents1[i]
        
    child[0] = 0
    child[nGen-1] = 0
    
    for gen in parents2:
        for i in range(0, start):
            if not np.isin(gen, child):
                if child[i] == -1:
                    child[i] = gen
        for i in range(end+1, nGen):
            if not np.isin(gen, child):
                if child[i] == -1:
                    child[i] = gen
    
    return np.array(child)

#function to fill new population with elite and children
def breedingPopulation(parents, eliteSize, crossoverRate):
    children = []
    nPopulation = len(parents)
    nChildren = nPopulation - eliteSize
    
    for i in range(eliteSize):
        children.append(parents[i])
    
    for i in range(nChildren):
        parentsIndex = np.random.randint(low = len(parents), size = 2)
        parents1 = parents[parentsIndex[0]]
        parents2 = parents[parentsIndex[1]]
        if(random.random()<crossoverRate):
            child = breeding(parents1, parents2)
        else:
            child = parents1
        children.append(child)
    
    return np.array(children)

#function to mutate an individu
def mutate(individu):
    nGen = len(individu)
    indexSwap = np.random.randint(1, nGen-1, size = 2)
    swap1 = indexSwap[0]
    swap2 = indexSwap[1]
    temp = individu[swap1]
    individu[swap1] = individu[swap2]
    individu[swap2] = temp
    
    return individu

#function to demonstrate mutations that is happening in population
def mutatePopulation(population, mutationRate):
    mutated = []
    
    for ind in population:
        if(random.random()<mutationRate):
            mutated.append(mutate(ind))
        else:
            mutated.append(ind)
    
    return np.array(mutated)

#function to wrap all steps in building new generation
def newGeneration(dMatrix, currentGeneration, eliteSize, mutationRate, crossoverRate):
    population, distance, fitness = calculateDistanceFitness(dMatrix, currentGeneration)
    bestFitness = fitness[0]
    
    parents = matingPool(population, fitness, eliteSize)
    children = breedingPopulation(parents, eliteSize, crossoverRate)
    nextGeneration = mutatePopulation(children, mutationRate)
    
    return nextGeneration, bestFitness

#function to wrap all steps in doing genetic algorithm
def geneticAlgorithm(dMatrix, cities, populationSize, eliteRate, mutationRate, crossoverRate, generation):
    #initialize population
    population = createPopulation(cities, populationSize)
    progress = []
    eliteSize = int(eliteRate*populationSize)
    
    for i in range(generation):
        population, bestFitness = newGeneration(dMatrix, population, eliteSize, mutationRate, crossoverRate)
        progress.append(bestFitness)
    
    sortedPopulation, distance, fitness = calculateDistanceFitness(dMatrix, population)
    bestRoute = sortedPopulation[0]
    
    return bestRoute, progress

#plot the best route on map
def plotRoute(cities, bestRoute):
    x = cities[1][bestRoute]
    y = cities[2][bestRoute]
    #print(x[1:])
    
    fig = plt.figure(figsize = (5,5))
    fig1 = fig.add_axes([0,0,1,1])
    plt.plot(x, y, 'ro-', ms=5, c='black')
    fig1.plot(x[0], y[0], c='red', marker='s')
    plt.title('Best Route'), plt.xlabel('x'), plt.ylabel('y')
    plt.show()
    
def import_data_city(filepath):
    city = pd.read_excel(r'{}'.format(filepath), sheet_name='Data', header=None)
    nCity = len(city)
    for i in range(nCity):
        city[0][i] = city[0][i] - 1
    
    return city

"""
CASE 1
"""
#import data case 1
dMatrix1 = pd.read_excel (r'E:/yk11.xls', sheet_name='DistanceMatrix', header=None)
dMatrix1 = np.array(dMatrix1)

cities1 = import_data_city('E:/yk11.xls')
cityName1 = set(cities1[0])

#initiate value
populationSize = 10
eliteRate = 0.2
mutationRate = 0.2
crossoverRate = 0.9
generation = 50

#searching for the best route from the first data
bestRoute, progress = geneticAlgorithm(dMatrix1, cityName1, populationSize, eliteRate, mutationRate, crossoverRate, generation)
print("Best route for case 1: ", end = "")
print(bestRoute)

#plot best distance in every iteration
iteration = [i+1 for i in range(generation)]

plt.plot(iteration, progress, 'ro-', ms=1, mec='k')
plt.title('Progress Case 1'), plt.xlabel('Generation'), plt.ylabel('Progress')
plt.show()

#plot the map
plotRoute(cities1, bestRoute)

"""
CASE 2
"""
#import data case 2
dMatrix2 = pd.read_excel (r'E:/berlin52.xls', sheet_name='DistanceMatrix', header=None)
dMatrix2 = np.array(dMatrix2)

cities2 = import_data_city('E:/berlin52.xls')
cityName2 = set(cities2[0])

#with the same value as case 1
#searching for the best route from the first data
bestRoute2, progress2 = geneticAlgorithm(dMatrix2, cityName2, populationSize, eliteRate, mutationRate, crossoverRate, generation)
print("Best route for case 2: ", end = "")
print(bestRoute2)

#plot best distance in every iteration
iteration = [i+1 for i in range(generation)]

plt.plot(iteration, progress, 'ro-', ms=1, mec='k')
plt.title('Progress Case 2'), plt.xlabel('Generation'), plt.ylabel('Progress')
plt.show()

#plot the map
plotRoute(cities2, bestRoute2)