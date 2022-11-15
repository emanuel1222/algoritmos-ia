from math import sqrt
import pandas as pd
import numpy as np
from numpy import random
import random
from collections import deque
import plotly.express as px
import plotly.graph_objects as go

# Calcula matriz de distancias.
#
# OBS:  Não é estritamente necessario calculá-las a priori.
#       Foi feito assim apenas para fins didáticos.
#       Ao invés, as distâncias podem ser calculadas sob demanda.
def gera_matriz_distancias(Coordenadas):

    n_cidades = len(Coordenadas)
    dist = np.zeros((n_cidades,n_cidades), dtype=float)

    for i in range(0, n_cidades):
        for j in range(i+1, n_cidades):
            x1,y1 = Coordenadas.iloc[i]
            x2,y2 = Coordenadas.iloc[j]
            
            dist[i,j] = distancia(x1,y1,x2,y2)
            dist[j,i] = dist[i,j]
    
    return dist

# Recebe uma lista com as coordenadas reais de uma cidade e
# gera uma matriz de distancias entre as cidades.
# Obs: a matriz é simetrica e com diagonal nula
def gera_problema_tsp(df_cidades):
    # nomes ficticios das cidades
    cidades = df_cidades.index

    # calcula matriz de distancias
    distancias = gera_matriz_distancias(df_cidades)

    # cria estrutura para armazena as distâncias entre todas as cidades
    tsp = pd.DataFrame(distancias, columns=cidades, index=cidades)

    return tsp


# distancia Euclidiana entre dois pontos
def distancia(x1,y1,x2,y2):
    dx = x2 - x1
    dy = y2 - y1
    return sqrt(dx**2 + dy**2)

# Função Objetivo: calcula custo de uma dada solução.
# Obs: Neste caso do problema do caixeiro viajante (TSP problem),
# o custo é o comprimento da rota entre todas as cidades.
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
 
# Plota a solução do roteamento das cidades
# usando a biblioteca PLOTLY
def plota_rotas(df_cidades, ordem_cidades):
    df_solucao = df_cidades.copy()
    df_solucao = df_solucao.reindex(ordem_cidades)

    X = df_solucao['X']
    Y = df_solucao['Y']
    cidades = list(df_solucao.index)

    # cria objeto gráfico
    fig = go.Figure()

    fig.update_layout(autosize=False, width=600, height=400, showlegend=False)

    # gera linhas com as rotas da primeira ate a ultima cidade
    fig.add_trace(go.Scatter(x=X, y=Y,
                             text=cidades, textposition='bottom center',
                             mode='lines+markers+text',
                             name=''))

    # acrescenta linha da última para a primeira para fechar o ciclo
    fig.add_trace(go.Scatter(x=X.iloc[[-1,0]], y=Y.iloc[[-1,0]],
                             mode='lines+markers', name=''))

    fig.show()
    
    
def _selecao(Candidato1, Candidato2, tsp):
    a1 = calcula_custo(tsp,Candidato1)
    a2 = calcula_custo(tsp,Candidato2)
    #print(a1,a2)

    # eleito o candidato com menor custo
    eleito = Candidato1 if a1<=a2 else Candidato2

    return eleito

def elimina_cidades_repetidas(child, Parent):
    A = set(Parent)
    B = set(child)
    child_aux = child.copy()
    cities_child_donthave = A - B
    
    for P in cities_child_donthave:
        if P not in child_aux:
            for elem in child_aux:
                if child_aux.count(elem) > 1:
                    idx = child_aux.index(elem)
                    child_aux[idx] = P
                    break
    return child_aux
            
    

def crossoverOX(Parent1, Parent2):

    N = len(Parent1)

    # ponto de corte
    corte1 = np.random.randint(1, N-1)
    corte2 = np.random.randint(corte1, N-1)

    # crossover no ponto de corte
    # gerando dois filhos
    child1 = Parent1[:corte1] + Parent2[corte1:corte2] + Parent1[corte2:]
    child2 = Parent2[:corte1] + Parent1[corte1:corte2] + Parent2[corte2:]
    
    c1 = elimina_cidades_repetidas(child1,Parent1)
    c2 = elimina_cidades_repetidas(child2,Parent2)

    return c1, c2
       
def mutacao(VT, p_mutacao=0.30):

    VT_mutated = VT.copy()

    N = len(VT)
    p = np.random.rand()

    if p < p_mutacao:
        cidade1   = np.random.randint(0,N-1)    # indice da coluna (base-0)
        cidade2 = np.random.randint(0,N-1)  # valor da linha   (base-1)
        aux = VT_mutated[cidade1]
        VT_mutated[cidade1] = VT_mutated[cidade2]
        VT_mutated[cidade2] = aux

        #print(col+1, linha)

    return VT_mutated
                
def algoritmo_genetico(tsp):
    # pseudo-código:

    # START
    solucao = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    # Generate the initial population # Compute fitness
    generations = 50
    population = gera_tuplas_custos(solucao,tsp)
    #print(population)
    # REPEAT
    for i in range (0, generations):
        new_population = []
        for i in range(0, 10):
            #     Selection
            rand_idx_parent1 = random.randrange(len(population))
            rand_idx_parent2 = random.randrange(len(population))
            #     Crossover
            child1, child2 = crossoverOX(population[rand_idx_parent1][1], population[rand_idx_parent2][1])
            #     Mutation
            child1 = mutacao(child1)
            child2 = mutacao(child2)
            #     ADD to new population
            new_population.append(child1)
            new_population.append(child2)

        rand_idx_candidate1 = random.randrange(len(new_population))
        rand_idx_candidate2 = random.randrange(len(new_population))
        best_candidate = _selecao(new_population[rand_idx_candidate1], new_population[rand_idx_candidate2])
    #     Compute fitness
        population = gera_tuplas_custos(best_candidate,tsp)
    # UNTIL population has converged
    # STOP
    return best_candidate
    # coloque seu código aqui
    pass
     
def main():
    url_coordenadas_cidade = 'http://www.math.uwaterloo.ca/tsp/world/wi29.tsp'

    df_coordenadas = pd.read_table(
                    url_coordenadas_cidade,
                    skiprows=7,           # ignora as 7 primeiras linhas com informações
                    names=['X', 'Y'],     # nomes das colunas
                    sep=' ',              # separador das colunas
                    index_col=0,          # usar col=0 como index (nome das cidades)
                    skipfooter=1,         # ignora a última linha (EOF)
                    engine='python'       # para o parser usar skipfooter sem warning
              )
    solucao = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    tsp = gera_problema_tsp(df_coordenadas)
    #plota_rotas(df_coordenadas, solucao)
    custo = calcula_custo(tsp,solucao)
    print("--- Custo da solução inicial: ", custo)
    solucao_final = algoritmo_genetico(tsp)
    custo_final = calcula_custo(tsp,solucao_final)
    print("--- Custo da solução final: ", custo_final)
    plota_rotas(df_coordenadas, solucao_final)
    return 0

if __name__ == "__main__":
    main()