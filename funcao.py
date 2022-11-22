from math import sqrt
import math
import pandas as pd
import numpy as np
from numpy import random
import random


# Cria solução aleatória para o algoritmo genetico

def solucao_aleatoria(Intervalos, N):
    V = list(range(Intervalos[0], Intervalos[1]+1))
    solucao = []

    for _ in range(0,N):
        elem = random.choice(V)

        solucao.append(elem)
        V.remove(elem)

    return solucao

# Cria solução aleatória para o algoritmo hill-climbing

def solucao_aleatoria_hc(Intervalos):
    V = list(range(Intervalos[0], Intervalos[1]+1))

    elem = random.choice(V)

    return elem


# A partir de uma dada solução, gera diversas variações (vizinhos)
def gera_vizinhos_tst(solucao):

    N = len(solucao)
    for i in range(1, N):       # deixa o primeiro fixo
        for j in range(i + 1, N):
            vizinho = solucao.copy()
            vizinho[i] = solucao[j]
            vizinho[j] = solucao[i]

            yield(vizinho)

# A partir de uma dada solução, gera diversas variações (vizinhos)
def gera_vizinhos(solucao, sigma=0.1):

    for i in range(0, 2): 
        dx = solucao + sigma * np.random.randn()
        yield(dx)
        
        
def gera_vizinhos_sa(solucao, sigma=0.1):


    dx = solucao + sigma * np.random.randn()
    yield(dx)
    
def escolheMelhorVizinho_sa(Tuplas):
    custo = Tuplas[0][0]
    vizinho = Tuplas[0][1]
    
    for c, v in Tuplas:
        if c <= custo:
            custo = c
            vizinho = v
    
    return custo, vizinho
            
def calcula_custo(x):
    '''
    Função objetivo, calculo o f(x) para encontrar o Y
    '''
    fx = (0.01 * (x**2)) + (10 * math.sin((2 * x - math.pi)/ 2))

    return fx

# Escolhe o melhor vizinho

def escolheMelhorVizinho(Tuplas):
    custo = Tuplas[0][0]
    vizinho = Tuplas[0][1]
    
    for c, v in Tuplas:
        if c < custo:
            custo = c
            vizinho = v
    
    return custo, vizinho
            

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

    dx = Media

    p = np.random.rand()

    if p < p_mutacao:
        dx = Media + sigma * np.random.randn()

    return dx

def crossover(Parent1, Parent2):
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
        populacao.append(solucao_aleatoria(Intervalos, N_population))
    return populacao

def seleciona_melhor_elem(fitness_population):
    best_elem = ()
    custo = 99999999
    
    for elem in fitness_population:
        if elem[0] < custo:
            custo = elem[0]
            best_elem = elem
    return best_elem

def seleciona_pior_elem(new_population):
    worst_elem = ()
    custo = -99999999
    
    for elem in new_population:
        if elem[0] > custo:
            custo = elem[0]
            worst_elem = elem
    return worst_elem

def manter_best_elem_lastGeneation(fitness_population, best_elem_lastGenaration):
    worst_elem_actualGeneration = seleciona_pior_elem(fitness_population)
    population = fitness_population.copy()
    
    if worst_elem_actualGeneration[0] > best_elem_lastGenaration[0]:
        idx = fitness_population.index(worst_elem_actualGeneration)
        population[idx] = best_elem_lastGenaration
    return population


def hillClimbing(Intervalos):
    solucao = solucao_aleatoria_hc(Intervalos)
    custo = calcula_custo(solucao)
    cont = 0 #
    #print(str((custo, solucao)) + ' <~ Solução aleatória') #

    while True:
        vizinhos = list(gera_vizinhos(solucao))
        Tuplas = gera_tuplas_custos(vizinhos)
        custoVizinho, vizinho = escolheMelhorVizinho(Tuplas)
        cont += 1
        #print(str((custoVizinho, vizinho)) + ' <~ ' + str(cont) + 'º vizinho') #

        if custoVizinho >= custo:
            break
        else:
            solucao = vizinho
            custo = custoVizinho

    return custo, solucao

# Hill-Climbing com Random Restart

def hillClimbingRandomRestart(Intervalos, restarts):
    solucao = solucao_aleatoria_hc(Intervalos)
    custo = calcula_custo(solucao)
    l = []

    for _ in range(restarts):
        custoVizinho, vizinho = hillClimbing(Intervalos)
        if custoVizinho < custo:
            solucao = vizinho
            custo = custoVizinho
        l.append((custoVizinho, vizinho))
    
    return custo, solucao, l


# Simulated Annealing

def simulatedAnnealing(Intervalos):
    T = 900
    solucao = -91#solucao_aleatoria_hc(Intervalos)
    custo = calcula_custo(solucao)

    while T > 0:
        vizinhos = list(gera_vizinhos(solucao))
        Tuplas = gera_tuplas_custos(vizinhos)
        custoVizinho, vizinho = escolheMelhorVizinho(Tuplas)

        if custoVizinho < custo:
            solucao = vizinho
            custo = custoVizinho
            #print('T = ' + str(T) + ' ~> Encontrou solução melhor')
        else:
            p = np.random.rand()
            prob = T/900
            #print('p = ' + str(p) + ' - prob = ' + str(prob))

            if p <= prob:
                solucao = vizinho
                custo = custoVizinho
                #print('T = ' + str(T) + ' ~> Aceitou solução pior/igual')
            '''else:
                print('T = ' + str(T) + ' ~> Manteve solução')'''
            T -= 1

    return custo, solucao

def algoritmo_genetico(Intervalos, N_generations, N_population):
    # pseudo-código:

    # START
    # Generate the initial population
    population = solucao_aleatoria(Intervalos, N_population) # Tabuleiro inicial
    # Compute fitness
    fitness_population = gera_tuplas_custos(population)
    # REPEAT
    for i in range (0, N_generations):
        new_population = []
        best_elem_population = seleciona_melhor_elem(fitness_population)
        for i in range(0, N_population):
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
        fitness_population = manter_best_elem_lastGeneation(fitness_population, best_elem_population)
        
    # Tournament - seleciona dois candidatos aleatoriamente e retorna o melhor
    rand_idx_candidate1 = random.randrange(len(population))
    rand_idx_candidate2 = random.randrange(len(population))
    solucao_final = _selecao(population[rand_idx_candidate1], population[rand_idx_candidate2])
    #print(pop_old)
    #print(population)
    # UNTIL population has converged
    # STOP
    custo = calcula_custo(solucao_final)
    return solucao_final, custo

def main():
    #solucao, ataques = algoritmo_genetico(8, 50)
    Intervalos = [-100,100]
    solucao, custo = algoritmo_genetico(Intervalos, 50, 20)
    print(solucao, custo)
    custo, solucao = hillClimbing(Intervalos)
    print(solucao, custo)
    custo, solucao, l = hillClimbingRandomRestart(Intervalos, 20)
    print(solucao, custo)
    custo, solucao = simulatedAnnealing(Intervalos) 
    print(solucao, custo)
    return 0

if __name__ == "__main__":
    main()