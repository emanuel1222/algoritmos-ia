# Cria uma solucao inicial com as cidades em um ordem aleatoria
import numpy as np
from numpy import random

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
    ataques  = __conta_ataques_linhas(VT)

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

def gera_tuplas_custos(VT):
    '''
    Gera tuplas com os custos de todos os individuos da populacao.
    '''
    TuplasCustos = []
    for individuo in gera_vizinhos(VT):
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

def algoritmo_genetico():
    # pseudo-código:

    # START
    VT = [1,2,3,4,5,6,7,8] # Tabuleiro inicial
    T = converte_vetor_tabuleiro(VT)
    print(T)
    # Generate the initial population
    # Compute fitness
    # REPEAT
    #     Selection
    #     Crossover
    #     Mutation
    #     Compute fitness
    # UNTIL population has converged
    # STOP

    # coloque seu código aqui

def main():
    algoritmo_genetico()
    return 0