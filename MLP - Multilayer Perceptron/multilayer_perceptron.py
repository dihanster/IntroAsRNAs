#-----------------------------------------------------------------------#
# Introducao as Redes Neurais Artificiais - Prof. Dr. Marcos Quiles     #
# Projeto 1: Implementacao Multilayer Perceptron                        #
# Nome: Willian Dihanster Gomes de Oliveira RA: 112269                  #
#-----------------------------------------------------------------------#
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Inicia uma semente para fixar resultados
random.seed(1)

# Retorna o valor da funcao sigmoid para um dado valor
def sigmoid(s):
    return 1.0 / (1.0 + math.exp(-s))

# Retorna o valor da derivada da funcao sigmoid
def sigmoid_derivada(s):
    return s * (1.0 - s)

# Calcula a ativacao dos neuronios para a entrada
def ativa(pesos, entradas):
    ativacao = pesos[-1]
    for i in range(len(pesos)-1):
        ativacao += pesos[i] * entradas[i]
    return ativacao

# Passo de Forward da Rede
def forward_propaga(rede_neural, exemplo):
    entradas = exemplo
    for camada in rede_neural:
        novas_entradas = []
        for neuronio in camada:
            ativacao = ativa(neuronio['pesos'], entradas)
            neuronio['saida'] = sigmoid(ativacao)
            novas_entradas.append(neuronio['saida'])
        entradas = novas_entradas
    return entradas

# Fase de Backward da Rede
def backward_propaga_erro(rede_neural, esperado):
    for i in reversed(range(len(rede_neural))):
        camada = rede_neural[i]
        erros = list()
        if i != len(rede_neural)-1:
            for j in range(len(camada)):
                erro = 0.0
                for neuronio in rede_neural[i + 1]:
                    erro += (neuronio['pesos'][j] * neuronio['delta'])
                erros.append(erro)
        else:
            for j in range(len(camada)):
                neuronio = camada[j]
                erros.append(esperado[j] - neuronio['saida'])
        for j in range(len(camada)):
            neuronio = camada[j]
            neuronio['delta'] = erros[j] * sigmoid_derivada(neuronio['saida'])

# Atualiza os pesos da Rede de acordo com o erro
def atualiza_peso(rede_neural, exemplo, eta, alfa):
    aux2 = [0] * 1000

    for i in range(len(rede_neural)):
        entradas = exemplo[:-1]
        if i != 0:
            entradas = [neuronio['saida'] for neuronio in rede_neural[i - 1]]
        k = -1
        aux1 = [0.0] * 1000
        for neuronio in rede_neural[i]:
            k += 1
            for j in range(len(entradas)):
                troca = neuronio['delta'] * entradas[j]
                neuronio['pesos'][j] += eta * troca + alfa * aux1[j]
                aux1[j] = troca
            troca = neuronio['delta']
            neuronio['pesos'][-1] += eta * troca + alfa * aux2[k]
            aux2[k] = troca

# Treina a Rede Neural
def treinamento(rede_neural, treino, eta, momentum, n_epocas, n_saidas):
        erros = []
        
        for epocas in range(n_epocas):
                soma_erro = 0
                for exemplo in treino:
                        saidas = forward_propaga(rede_neural, exemplo)
                        esperado = [0 for i in range(n_saidas)]
                        esperado[exemplo[-1]] = 1
                        soma_erro += sum([(esperado[i]-saidas[i])**2 for i in range(len(esperado))])
                        backward_propaga_erro(rede_neural, esperado)
                        atualiza_peso(rede_neural, exemplo, eta, momentum)
                print('Epoca=%d, Erro=%.3f' % (epocas, soma_erro))
                erros.append(soma_erro)

        plt.plot(erros)
        plt.title('Erro x Epocas')
        plt.show()

#----------------------#
# Main                 #
#----------------------#

# Leitura do dataset
dataset_original = np.loadtxt('data_banknote_authentication.csv', delimiter = ',')
n = len(dataset_original) * 0.8
n = int(n)
dataset1 = dataset_original[0:n, 0:len(dataset_original[0])]
dataset2 = dataset_original[n:, 0:len(dataset_original[0])]
dataset = []
dataset_teste = []

# Parametros da Rede Neural
n_neuronios = 10
epocas = 100
eta = 0.3
momentum = 0.9

# Adapta o vetor de exemplos
for i in range(len(dataset1)):
        dataset.append(dataset1[i].tolist())
        dataset[i][-1] = int(dataset[i][-1])

for i in range(len(dataset2)):
        dataset_teste.append(dataset2[i].tolist())
        dataset_teste[i][-1] = int(dataset2[i][-1])

# Calcula o valor de entradas e de saidas de acordo com o dataset
n_entradas = len(dataset[0]) - 1
n_saidas = len(set([exemplo[-1] for exemplo in dataset]))

# Inicia Rede Neural
rede_neural = list()
camada_oculta = [{'pesos':[random.random() for i in range(n_entradas + 1)]} for i in range(n_neuronios)]
rede_neural.append(camada_oculta)
camada_saida = [{'pesos':[random.random() for i in range(n_neuronios + 1)]} for i in range(n_saidas)]
rede_neural.append(camada_saida)

# Treina a Rede Neural
treinamento(rede_neural, dataset, eta, momentum, epocas, n_saidas)

# Classifica os exemplos selecionados
correto = 0
for exemplo in dataset_teste:
    saida = forward_propaga(rede_neural, exemplo)
    predicao = saida.index(max(saida))
    if (exemplo[-1] == predicao): correto += 1
    print('esperado=%d, previsto=%d' % (exemplo[-1], predicao))

print('Acuracia = ', correto/len(dataset_teste) * 100, '%')