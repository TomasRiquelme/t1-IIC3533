import numpy as np
import pandas as pd

# Definimos las valores establecidos por la tarea

total_observations = 100000
entry_variables = 300
resamples = 48

# Defino los parámetros

np.random.seed(42)

real_coeficients = np.random.randn(entry_variables + 1)

data_matrix = np.random.randn(total_observations, entry_variables)

ones_column = np.ones((total_observations, 1))

data_matrix = np.concatenate((ones_column, data_matrix), axis=1)

y_vector = np.dot(data_matrix, real_coeficients) + np.random.randn(total_observations)

#Usando la fórmula del enunciado de la T1
minimumSquareSolution = np.dot((np.dot((np.linalg.inv(np.dot(data_matrix.T, data_matrix))), data_matrix.T)), y_vector)


# Definimos funciones que permitan exportar estos datos, para que los archivos usen la misma 
# referencia
def returnCoeficients():
    return real_coeficients
def returnDataMatrix():
    return data_matrix
def returnYVector():
    return y_vector
def returnMinimumSquareSolution():
    return minimumSquareSolution