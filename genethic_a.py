# Cria uma solucao inicial com as cidades em um ordem aleatoria
import numpy as np
from numpy import random
import random

# Cria solução aleatória

def solucao_aleatoria(N):
    rainhas = list(range(1, N+1))
    solucao = []

    # as 3 linhas abaixo não são estritamente necessarias, servem
    # apenas para fixar a primeira cidade da lista na solução
    rainha = rainhas[0]
    solucao.append(rainha)
    rainhas.remove(rainha)

    for _ in range(1,len(rainhas)+1):
        #print(_, rainhas, solucao)
        rainha = random.choice(rainhas)

        solucao.append(rainha)
        rainhas.remove(rainha)

    return solucao

def converte_vetor_tabuleiro(VT):
    '''
    Recebe um vetor representando um tabuleiro
    com N rainhas, uma por coluna e retorna 
    uma lista de lista de 0 e 1 representando
    um tabuleiro com as rainhas.
    
    '''
    N = len(VT)

    L = [0]*N
    T = []
    for i in range(N):
        T += [L.copy()]

    for lin in range(N):
        for col in range(N):
            if lin+1 == VT[col]:
                T[lin][col] = 1

    return T

def __conta_ataques_linhas(VT):
    '''
    Função que recebe um Vetor-Tabuleiro e
    retorna o número de pares de rainhas se
    atacando mutuamente nas linhas.
    '''
    ataques = 0
    N = len(VT)
    for col1 in range(N):
        lin1 = VT[col1]
        for col2 in range(col1+1, N):
            lin2 = VT[col2]
            if lin1==lin2:
                ataques +=1
    
    return ataques

def __conta_ataques_diagonais(VT):
    '''
    Função que recebe um Vetor-Tabuleiro e
    retorna o número de pares de rainhas se
    atacando mutuamente nas diagonais.
    '''
    ataques = 0
    N = len(VT)

    for col1 in range(N):
        lin1 = VT[col1]
        for col2 in range(col1+1, N):
            lin2 = VT[col2]

            # diferenças entre as linhas e colunas
            d1 = lin1-col1
            d2 = lin2-col2

            # somas das linhas e colunas
            s1 = lin1+col1
            s2 = lin2+col2

            # condições para ataques nas diagonais
            if d1==d2 or s1==s2:
                ataques +=1
                #print(f'({lin1},{col1+1}) ({lin2},{col2+1}) -->', ataques,
                #      '<--', f'  -({d1:2},{d2:2})  +({s1:2},{s2:2})')
    
    return ataques

def conta_ataques(VT):
    '''
    Função que recebe um Vetor-Tabuleiro e
    retorna o número de pares de rainhas se
    atacando mutuamente nas linhas e diagonais.
    '''
    ataques = __conta_ataques_linhas(VT)

    ataques += __conta_ataques_diagonais(VT)

    return ataques

def gera_vizinhos(VT):

    N = len(VT)
    for col in range(N):
        for lin in range(N):
            # se nao existe rainha naquela linha,
            # entao gera estado vizinho.
            linha = lin+1
            if linha != VT[col]:
                vizinho   = VT.copy()
                vizinho[col] = linha

                yield vizinho

def gera_tuplas_custos(LVT):
    '''
    Gera tuplas com os custos de todos os individuos da populacao.
    '''
    TuplasCustos = []
    for individuo in LVT:
        ataques = conta_ataques(individuo)
        TuplasCustos += [(ataques, individuo)]

    return TuplasCustos


def mutacao(VT, p_mutacao=0.30):

    VT_mutated = VT.copy()

    N = len(VT)
    p = np.random.rand()

    if p < p_mutacao:
        col   = np.random.randint(0,N)    # indice da coluna (base-0)
        linha = np.random.randint(1,N+1)  # valor da linha   (base-1)

        VT_mutated[col] = linha
        #print(col+1, linha)

    return VT_mutated

def crossover(Parent1, Parent2):

    N = len(Parent1)

    # ponto de corte
    c = np.random.randint(1, N-1)

    # crossover no ponto de corte
    # gerando dois filhos
    child1 = Parent1[:c] + Parent2[c:]
    child2 = Parent2[:c] + Parent1[c:]

    return child1, child2

def _selecao(Candidato1, Candidato2):
    a1 = conta_ataques(Candidato1)
    a2 = conta_ataques(Candidato2)
    #print(a1,a2)

    # eleito o candidato com menor custo
    eleito = Candidato1 if a1<=a2 else Candidato2

    return eleito

def imprime_tabuleiro(T):
    for line in T:
        for elem in line:
            print(elem, end = ' ')
        print("")

def gera_populacao_inicial(N, N_population):
    populacao = []
    for i in range(N_population):
        populacao.append(solucao_aleatoria(N))
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
    
    

def algoritmo_genetico(N, N_generations, N_population):
    # pseudo-código:
    # START
    # Generate the initial population
    population = gera_populacao_inicial(N, N_population)
    #print(population)
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
    #print("\n", best_candidate)
    print("\n")
    print("--- Tabuleiro final --- \n")
    imprime_tabuleiro(converte_vetor_tabuleiro(solucao_final))
    # UNTIL population has converged
    # STOP
    ataques = conta_ataques(solucao_final)
    return solucao_final, ataques

def main():
    solucao_inicial = [1,2,3,4,5,6,7,8] # Tabuleiro inicial
    T = converte_vetor_tabuleiro(solucao_inicial)
    print("--- Tabuleiro inicial --- \n")
    imprime_tabuleiro(T)
    print(solucao_inicial, conta_ataques(solucao_inicial))
    solucao, ataques = algoritmo_genetico(8,50, 20)
    print(solucao, ataques)
    return 0

if __name__ == "__main__":
    main()