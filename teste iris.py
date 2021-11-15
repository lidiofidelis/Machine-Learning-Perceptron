import numpy as np
import matplotlib.pyplot as plt

#Função de ativação e sua derivada

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#Função custo
def MSE(Y_target, Y_pred):
    return np.mean((Y_target - Y_pred) ** 2)

# carregar o dataset
# implementar isso



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
