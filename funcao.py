from math import sqrt
import math
import pandas as pd
import numpy as np
from numpy import random
import random


# Cria solução aleatória

def solucao_aleatoria(Intervalos, N):
    V = list(range(Intervalos[0], Intervalos[1]+1))
    solucao = []


    for _ in range(0,N):
        #print(_, rainhas, solucao)
        elem = random.choice(V)

        solucao.append(elem)
        V.remove(elem)

    return solucao


# A partir de uma dada solução, gera diversas variações (vizinhos)
def gera_vizinhos(solucao):

    N = len(solucao)
    for i in range(1, N):       # deixa o primeiro fixo
        for j in range(i + 1, N):
            vizinho = solucao.copy()
            vizinho[i] = solucao[j]
            vizinho[j] = solucao[i]

            yield(vizinho)
            
def calcula_custo(x):
    '''
    Função objetivo, calculo o f(x) para encontrar o Y
    '''
    fx = (0.01 * (x**2)) + (10 * math.sin((2 * x - math.pi)/ 2))

    return fx
            

def gera_tuplas_custos(LVT):
    '''
    Gera tuplas com os custos de todos os individuos da populacao.
    '''
    TuplasCustos = []
    for individuo in LVT:
        custo = calcula_custo(individuo)
        TuplasCustos += [(custo, individuo)]

    return TuplasCustos

def _selecao(Candidato1, Candidato2):
    a1 = calcula_custo(Candidato1)
    a2 = calcula_custo(Candidato2)
    #print(a1,a2)

    # eleito o candidato com menor custo
    eleito = Candidato1 if a1<=a2 else Candidato2

    return eleito

def mutacao(Media, p_mutacao=0.30, sigma=0.1):

    dx = Media.copy()

    p = np.random.rand()

    if p < p_mutacao:
        dx = Media + sigma * np.random.randn()

        #print(col+1, linha)

    return dx

def crossover(Parent1, Parent2):

    N = len(Parent1)

    # ponto de corte
    Alpha = np.random.uniform(0, 1)

    # crossover no ponto de corte
    # gerando dois filhos
    
    child1 = (Parent1 * Alpha) + (Parent2 * (1 - Alpha))
    child2 = (Parent2 * Alpha) + (Parent1 * (1 - Alpha))

    return child1, child2

def gera_populacao_inicial(Intervalos, N_population):
    populacao = []
    for i in range(N_population):
        populacao.append(solucao_aleatoria(Intervalos))
    return populacao


def algoritmo_genetico(Intervalos, N_generations, N_population):
    # pseudo-código:

    # START
    # Generate the initial population
    population = gera_populacao_inicial(Intervalos, N_population) # Tabuleiro inicial
    print(population)
    # Compute fitness
    fitness_population = gera_tuplas_custos(population)
    print(fitness_population)
    # REPEAT
    for i in range (0, N_generations):
        new_population = []
        for i in range(0, 10):
            #     Selection
            rand_idx_parent1 = random.randrange(len(fitness_population))
            rand_idx_parent2 = random.randrange(len(fitness_population))
            #     Crossover
            child1, child2 = crossover(fitness_population[rand_idx_parent1][1], fitness_population[rand_idx_parent2][1])
            #     Mutation
            child1 = mutacao(child1)
            child2 = mutacao(child2)
            # Tournament - seleciona dois candidatos aleatoriamente e retorna o melhor
            best_child = _selecao(child1, child2)
            #     ADD to new population
            new_population.append(best_child)
            
        #     SET new population
        population = new_population 
        #     Compute fitness
        fitness_population = gera_tuplas_custos(population)
        
    # Tournament - seleciona dois candidatos aleatoriamente e retorna o melhor
    rand_idx_candidate1 = random.randrange(len(population))
    rand_idx_candidate2 = random.randrange(len(population))
    solucao_final = _selecao(population[rand_idx_candidate1], population[rand_idx_candidate2])
    #print(pop_old)
    #print(population)
    print("\n", solucao_final)
    print("\n")
    # UNTIL population has converged
    # STOP
    custo = calcula_custo(solucao_final)
    return solucao_final, custo

def main():
    #solucao, ataques = algoritmo_genetico(8, 50)
    Intervalos = [-100,100]
    t = solucao_aleatoria(Intervalos, 20)
    return 0

if __name__ == "__main__":
    main()