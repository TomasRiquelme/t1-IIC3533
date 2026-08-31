import numpy as np
from joblib import Parallel, delayed
from variables_config import returnYVector, returnCoeficients, returnDataMatrix, returnMinimumSquareSolution

dataMatrix = returnDataMatrix()
Yvector = returnYVector()
coeficients = returnCoeficients()
minimumSquareSolution = returnMinimumSquareSolution()

def bootstrap_regression(X, y, seed):
    rng = np.random.default_rng(seed)

    N = X.shape[0]

    # Elegir N índices con reemplazo
    indices = rng.choice(N, size=N, replace=True)

    X_bootstrap_idx = X[indices]
    y_bootstrap_idx = y[indices]


    x_train_rows = np.array(X_bootstrap_idx)
    y_train_rows = np.array(y_bootstrap_idx)

    return np.dot((np.dot((np.linalg.inv(np.dot(x_train_rows.T, x_train_rows))), x_train_rows.T)), y_train_rows)

paralleled_results = Parallel(n_jobs=5)(
    delayed(bootstrap_regression)(dataMatrix, Yvector, b)
    for b in range(48)
)
#Ordenamos los valores para sacar los percentiles
ordered_regressions = np.sort(np.array(paralleled_results), axis=0)

#Obtenemos los percentiles
lower_bound = np.percentile(ordered_regressions, 2.5, axis=0)
upper_bound = np.percentile(ordered_regressions, 97.5, axis=0)

inside_interval = (coeficients >= lower_bound) & (coeficients <= upper_bound)

#Vemos si los resultados están en los intervalos de confianza
#Obtenemos el porcentaje de parámetros que sí estuvieron dentro del intervalo
coverage = np.mean(inside_interval)

print(coverage)
