from math import sqrt
import pandas as pd
import numpy as np
from numpy import random
import random
import plotly.express as px
import plotly.graph_objects as go


# Cria uma solucao inicial com as cidades em um ordem aleatoria
def solucao_aleatoria(tsp):
    cidades = list(tsp.keys())
    solucao = []

    # as 3 linhas abaixo não são estritamente necessarias, servem
    # apenas para fixar a primeira cidade da lista na solução
    cidade = cidades[0]
    solucao.append(cidade)
    cidades.remove(cidade)

    for _ in range(0,len(cidades)):
        #print(_, cidades, solucao)
        cidade = random.choice(cidades)

        solucao.append(cidade)
        cidades.remove(cidade)

    return solucao

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
            

def gera_tuplas_custos(LVT, tsp):
    '''
    Gera tuplas com os custos de todos os individuos da populacao.
    '''
    TuplasCustos = []
    for individuo in LVT:
        custo = calcula_custo(tsp, individuo)
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
        cidade1   = np.random.randint(1,N-1)    # indice da coluna (base-0)
        cidade2 = np.random.randint(1,N-1)  # valor da linha   (base-1)
        aux = VT_mutated[cidade1]
        VT_mutated[cidade1] = VT_mutated[cidade2]
        VT_mutated[cidade2] = aux

        #print(col+1, linha)

    return VT_mutated

def gera_populacao_inicial(tsp, N_population):
    populacao = []
    for i in range(N_population):
        populacao.append(solucao_aleatoria(tsp))
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
                
def algoritmo_genetico(tsp, N_generations, N_population):
    # pseudo-código:

    # START
    # Generate the initial population 
    population = gera_populacao_inicial(tsp,N_population)
    #print(population)
    # Compute fitness
    fitness_population = gera_tuplas_custos(population, tsp)
    # REPEAT
    for i in range (0, N_generations):
        new_population = []
        best_elem_population = seleciona_melhor_elem(fitness_population)
        for i in range(0, N_population):
            #     Selection
            rand_idx_parent1 = random.randrange(len(fitness_population))
            rand_idx_parent2 = random.randrange(len(fitness_population))
            #     Crossover
            child1, child2 = crossoverOX(fitness_population[rand_idx_parent1][1], fitness_population[rand_idx_parent2][1])
            #     Mutation
            child1 = mutacao(child1)
            child2 = mutacao(child2)
            # Tournament - seleciona dois candidatos aleatoriamente e retorna o melhor
            best_child = _selecao(child1, child2, tsp)
            #     ADD to new population
            new_population.append(best_child)

        #     SET new population
        population = new_population 
        #     Compute fitness
        fitness_population = gera_tuplas_custos(population, tsp)
        fitness_population = manter_best_elem_lastGeneation(fitness_population, best_elem_population)
    # UNTIL population has converged
    # Tournament - seleciona dois candidatos aleatoriamente e retorna o melhor
    rand_idx_candidate1 = random.randrange(len(population))
    rand_idx_candidate2 = random.randrange(len(population))
    solucao_final = _selecao(population[rand_idx_candidate1], population[rand_idx_candidate2], tsp)
    # STOP
    custo = calcula_custo(tsp,solucao_final)
    return solucao_final, custo
     
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
    solucao_inicial = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    tsp = gera_problema_tsp(df_coordenadas)
    plota_rotas(df_coordenadas, solucao_inicial)
    custo = calcula_custo(tsp,solucao_inicial)
    print("--- Custo da solução inicial: ", custo)
    solucao_final, custo_final = algoritmo_genetico(tsp,50,20)
    print("--- Custo da solução final: ", custo_final)
    plota_rotas(df_coordenadas, solucao_final)
    print(solucao_final)
    print(custo - custo_final)
    return 0

if __name__ == "__main__":
    main()