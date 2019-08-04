#----------------------------------------------------------------------------#
# Introducao as Redes Neurais Artificias - Prof. Dr. Marcos Quiles           #
# Projeto 2: Implementacao Rede SOM - Self Organizing Maps                   #
# Nome: Willian Dihanster Gomes de Oliveira RA: 112269                       #
#----------------------------------------------------------------------------#
import math 
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from matplotlib import patches as patches

def plota_rede(rede):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, rede.shape[0]+1))
    ax.set_ylim((0, rede.shape[1]+1))
     
    for i in range(1, rede.shape[0] + 1):
        for j in range(1, rede.shape[1] + 1):
            ax.add_patch(patches.Rectangle((i-0.5, j-0.5), 1, 1,
                        facecolor = rede[i-1,j-1,:],
                        edgecolor = 'none'))
    plt.show()
    
#Calcula a Umatrix da rede
def calcula_umatrix(rede):
	umatrix = []
    
	for i in range(rede.shape[0]):
		umatrix.append([])
		for j in range(rede.shape[0]):
			cont = 0
			soma = 0
            #Checa os limites
			for k in range(i - 1, i + 2):
				if(k < 0 or k >= rede.shape[0]):
					continue
				for l in range(j - 1, j + 2):
					if(l < 0 or l >= rede.shape[1]):
						continue
					if(i == k and j == l):
						continue
                    #Entao cont e vai somando a distancia em uma variavel 
					cont += 1
					soma += distance.euclidean(rede[i, j, :], rede[k, l, :])
            
            #Pega a soma e divida pelo numero de posicoes 
			umatrix[i].append(soma/float(cont))
    
    #Retorna a Umatrix
	return umatrix

#Acha o neuronio vencedor dado um exemplo
def acha_vencedor(rede, exemplo):
    min_dist = 10000000
    
    #Calcula a distancia do exemplo para todos os outros
    for i in range(rede.shape[0]):
        for j in range(rede.shape[1]):
            w = rede[i, j, :]
            dist = np.sqrt(sum((w - exemplo) ** 2))
            #Acha o mais proximo
            if dist < min_dist:
                min_dist = dist
                k, l = i, j
    
    #Retorna os indices do neuronio vencedor
    return k, l

#Treinamento com um numero maximo de epocas
def treinamento(data, rede, n, m, max_epocas, eta0, sigma0, tau):
    
    #Faz o treinamento com um numero determinado de epocas
    for epoca in range(max_epocas):
        
        #Para cada exemplo do conjunto de dados
        for d in data:
            
            #Acha o neuronio vencedor
            i_venc, j_venc = acha_vencedor(rede, d)
            
            #Calcula o valor do decaimento de eta e sigma
            eta = eta0 * (math.exp(-(epoca/max_epocas)))
            sigma = sigma0 * (math.exp(-epoca/tau))
            
            #Atualiza os pesos da rede
            for i in range(rede.shape[0]):
                for j in range(rede.shape[1]):
                    peso = rede[i, j, :]
                    peso_dist = np.sqrt(sum(
                    (np.array([i, j]) - np.array([i_venc, j_venc])) ** 2))
                
                    if peso_dist <= sigma:
                        h =  math.exp(-peso_dist / (2 * (sigma**2)))
                        novo_peso = peso + (eta * h * (d - peso))
                        rede[i, j, :] = novo_peso
    
    #Retorna a rede treinada
    return rede

#---------------------------#
# Main                      #
#---------------------------#
#Fixa um semente
random.seed(1)

#Leitura do Dataset
#Dataset Iris (txt ja sem a classe)
#data = np.loadtxt('iris.txt', delimiter = ',')

#Dataset Cores RGB
data = np.random.randint(0, 255, (100, 3))
    
#Dataset Jains Toys
#data = np.loadtxt('jain.txt', delimiter = '\t')
#data = data[:, 0:2]

#Normaliza os dados
max_col = data.max(axis=0)
data = data / max_col[np.newaxis, :]

#Definicao dos parametros e ctes
m = data.shape[0]
n = data.shape[1]
max_epocas = 100
eta0 = 0.3
sigma0 = 10
tau = max_epocas/np.log(sigma0)    

#Cria um Grid 10x10xn
rede = np.random.random((15, 15, n))

#Plota a rede antes do  treinamento
plota_rede(rede)

#Faz o Treinamento
rede = treinamento(data, rede, n, m, max_epocas, eta0, sigma0, tau)

#Plota a rede antes do  treinamento
plota_rede(rede)

#Faz os calculos para a Umatrix da Rede        
umatrix = calcula_umatrix(rede)
        
#Faz o plot da Umatrix gerada
fig = plt.figure()
plt.imshow(umatrix, origin='lower')
plt.show(block=True)