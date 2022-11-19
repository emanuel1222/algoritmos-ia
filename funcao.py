from math import sqrt
import math
import pandas as pd
import numpy as np
from numpy import random
import random



def calcula_custo(tsp, solucao):

    N = len(solucao)
    custo = 0

    for i in range(N):

        # Quando chegar na última cidade, será necessário
        # voltar para o início para adicionar o 
        # comprimento da rota da última cidade 
        # até a primeira cidade, fechando o ciclo.
        #
        # Por isso, a linha abaixo:
        k = (i+1) % N
        cidadeA = solucao[i]
        cidadeB = solucao[k]

        custo += tsp.loc[cidadeA, cidadeB]

        #print(tsp.loc[cidadeA, cidadeB], cidadeA,cidadeB)

    return custo

# A partir de uma dada solução, gera diversas variações (vizinhos)
def gera_vizinhos(solucao):

    N = len(solucao)
    for i in range(1, N):       # deixa o primeiro fixo
        for j in range(i + 1, N):
            vizinho = solucao.copy()
            vizinho[i] = solucao[j]
            vizinho[j] = solucao[i]

            yield(vizinho)
            

def gera_tuplas_custos(solucao, tsp):
    '''
    Gera tuplas com os custos de todos os individuos da populacao.
    '''
    TuplasCustos = []
    for individuo in gera_vizinhos(solucao):
        custo = calcula_custo(tsp, solucao)
        TuplasCustos += [(custo, individuo)]

    return TuplasCustos

def _selecao(Candidato1, Candidato2, tsp):
    a1 = calcula_custo(tsp,Candidato1)
    a2 = calcula_custo(tsp,Candidato2)
    #print(a1,a2)

    # eleito o candidato com menor custo
    eleito = Candidato1 if a1<=a2 else Candidato2

    return eleito