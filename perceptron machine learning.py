#mlp sigmoide
import math
import numpy as np
import random
import sys

#Função de ativação e sua derivada

def relud(xi):

        return np.where(xi <= 0, 0, 1)

def relu(xi):
    return np.greater(xi, 0).astype(int)


def sg(xi):
    for y in range(len(xi)):
        try:
            (1 / (1 + (math.exp(-xi[y]))))
        except:
            xi[y]=0
    return (np.array(xi))

def sgd(x):
    #print(x)
    return sg(x)*(1-sg(x))


#def sigmoid(x):
#    return 1 / (1 + np.exp(-x))
#
#def sigmoid_derivative(x):
#    return x * (1 - x)

# carregar o dataset

f= open("iris.txt","r") #elemento 0 a 15 dados 0 a 2 dado 1,4 a 7 dado 2,9 a 11 dado 3,12 a 15 dado 4
f1=f.readlines()

p1=np.zeros(150)
p2=np.zeros(150)
p3=np.zeros(150)
dado=np.zeros((50,4))
dado1=np.zeros((50,4))
dado2=np.zeros((50,4))
dados=np.zeros((150,4))

for x in range(len(f1)):
    if x<=49:
        dado[x][0] = float("".join(f1[x][0:3:1]))
        dado[x][1] = float("".join(f1[x][4:7:1]))
        dado[x][2] = float("".join(f1[x][8:11:1]))
        dado[x][3]= float("".join(f1[x][12:15:1]))
        p1[x] = 1
    elif x >= 50 and x <= 99:
        dado1[x-50][0] = float("".join(f1[x][0:3:1]))
        dado1[x-50][1] = float("".join(f1[x][4:7:1]))
        dado1[x-50][2] = float("".join(f1[x][8:11:1]))
        dado1[x-50][3] = float("".join(f1[x][12:15:1]))
        p2[x]=1
    elif x >= 100 and x <= 149:
        dado2[x-100][0] = float("".join(f1[x][0:3:1]))
        dado2[x-100][1] = float("".join(f1[x][4:7:1]))
        dado2[x-100][2] = float("".join(f1[x][8:11:1]))
        dado2[x-100][3] = float("".join(f1[x][12:15:1]))
        p3[x]=1

for x in range(len(f1)):
    if x<=149 :
        dados[x][0] = float("".join(f1[x][0:3:1]))
        dados[x][1] = float("".join(f1[x][4:7:1]))
        dados[x][2] = float("".join(f1[x][8:11:1]))
        dados[x][3] = float("".join(f1[x][12:15:1]))


#Taxa de aprendizagem

mi = 0.5

#Quantidade de épocas

EPOCHS = 1000

#Vetor da função de custo
cost = np.array([])

#Arquitetura da rede

n_neurons_input_layer = 4
n_neurons_hidden_layer_1 = 4
n_neurons_hidden_layer_2 = 4
n_neurons_output_layer = 3


#Pesos


w_hidden_layer_1 = np.random.rand(n_neurons_input_layer, n_neurons_hidden_layer_1)
w_hidden_layer_1

#baias

b_hidden_layer_1 = np.zeros(n_neurons_hidden_layer_1)
#b_hidden_layer_1

b_hidden_layer_2 = np.zeros(n_neurons_hidden_layer_2)
#b_hidden_layer_2

#Treino da rede

for epoch in range(EPOCHS):
    activation_hidden_layer_1 = sigmoid(np.dot(X, w_hidden_layer_1) + b_hidden_layer_1)
    activation_hidden_layer_2 = sigmoid(np.dot(activation_hidden_layer_1, w_hidden_layer_2) + b_hidden_layer_2)
    activation_output_layer = sigmoid(np.dot(activation_hidden_layer_2, w_output_layer) + b_output_layer)
    
    cost = np.append(cost, MSE(Y, activation_output_layer))
    
    delta_output_layer = (Y - activation_output_layer) * sigmoid_derivative(activation_output_layer)
    delta_hidden_layer_2 = np.dot(delta_output_layer, w_output_layer.T) * sigmoid_derivative(activation_hidden_layer_2)
    delta_hidden_layer_1 = np.dot(delta_hidden_layer_2, w_hidden_layer_2.T) * sigmoid_derivative(activation_hidden_layer_1)
    
    w_output_layer += N * np.dot(activation_hidden_layer_2.T, delta_output_layer)
    w_hidden_layer_2 += N * np.dot(activation_hidden_layer_1.T, delta_hidden_layer_2)
    w_hidden_layer_1 += N * np.dot(X.T, delta_hidden_layer_1)

#Gráfico da função de custo

plt.plot(cost)
plt.title('Função de custo da rede')
plt.xlabel('Épocas')
plt.ylabel('Custo')
plt.show()
