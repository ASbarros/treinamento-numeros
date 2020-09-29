# coding=utf-8
# importacao das bibliotecas necessarias

# pybrain
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

# graficos
import matplotlib.pyplot as plt
import numpy as np


# funcao para carregar os dados de treinamento
def getData(path):
    # Open file
    file = open(path, "r")

    data = []

    for linha in file:  # obtem cada linha do arquivo
        linha = linha.rstrip()  # remove caracteres de controle, \n
        digitos = linha.split(" ")  # pega os dígitos
        for numero in digitos:  # para cada número da linha
            data.append(numero)  # add ao vetor de dados

    file.close()
    return data


# configurando a rede neural artificial e o dataSet de treinamento
network = buildNetwork(45, 400, 500, 650, 500, 400, 1)  # define network
dataSet = SupervisedDataSet(45, 1)  # define dataSet

arquivos = [
    '1/1.txt', '1/1b.txt', '1/1c.txt', '1/1d.txt', '1/1e.txt',
    '2/2.txt', '2/2b.txt', '2/2c.txt', '2/2d.txt', '2/2e.txt',
    '3/3.txt', '3/3b.txt', '3/3c.txt', '3/3d.txt', '3/3e.txt',
    '4/4.txt', '4/4b.txt', '4/4c.txt', '4/4d.txt', '4/4e.txt',
    '5/5.txt', '5/5b.txt', '5/5c.txt', '5/5d.txt', '5/5e.txt',
    '6/6.txt', '6/6b.txt', '6/6c.txt', '6/6d.txt', '6/6e.txt',
    '7/7.txt', '7/7b.txt', '7/7c.txt', '7/7d.txt', '7/7e.txt',
    '8/8.txt', '8/8b.txt', '8/8c.txt', '8/8d.txt', '8/8e.txt',
    '9/9.txt', '9/9b.txt', '9/9c.txt', '9/9d.txt', '9/9e.txt',
    '0/0.txt', '0/0b.txt', '0/0c.txt', '0/0d.txt', '0/0e.txt',
            ]
# a resposta do número
resposta = [
    [1], [1], [1], [1], [1],
    [2], [2], [2], [2], [2],
    [3], [3], [3], [3], [3],
    [4], [4], [4], [4], [4],
    [5], [5], [5], [5], [5],
    [6], [6], [6], [6], [6],
    [7], [7], [7], [7], [7],
    [8], [8], [8], [8], [8],
    [9], [9], [9], [9], [9],
    [0], [0], [0], [0], [0]
    ]

i = 0
for arquivo in arquivos:  # para cada arquivo de treinamento
    data = getData(arquivo)  # pegue os dados do arquivo
    dataSet.addSample(data, resposta[i])  # add dados no dataSet
    i = i + 1

# trainer
trainer = BackpropTrainer(network, dataSet)
error = 1
iteration = 0
outputs = []
file = open("outputs.txt", "w")  # arquivo para guardar os resultados

while error > 0.001:  # 10 ^ -3
    error = trainer.train()
    outputs.append(error)
    iteration += 1
    print (iteration, error)
    file.write(str(error) + "\n")

file.close()

# Fase de teste
arquivos = [
    '1/1-test.txt', '2/2-test.txt', '3/3-test.txt', '4/4-test.txt', '5/5-test.txt',
    '6/6-test.txt', '7/7-test.txt', '8/8-test.txt', '9/9-test.txt', '0/0-test.txt'
]
for arquivo in arquivos:
    data = getData(arquivo)
    print (arquivo, ": ", network.activate(data))

# plot graph
plt.ioff()
plt.plot(outputs)
plt.xlabel('Iteracoes')
plt.ylabel('Erro Quadratico')
plt.show()
