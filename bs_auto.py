import sys
import time
import numpy as np
from variables_config import returnYVector, returnCoeficients, returnDataMatrix, returnMinimumSquareSolution
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

if len(sys.argv) > 1:
    p_argument = int(sys.argv[1])
else:
    p_argument = 1

# Importamos los valores a usar
dataMatrix = returnDataMatrix()
Yvector = returnYVector()
coeficients = returnCoeficients()
minimumSquareSolution = returnMinimumSquareSolution()

start_time = time.time()
print(f'Ejecutando bs_auto con p = {p_argument} procesos...')

bs_regressor = BaggingRegressor(estimator=LinearRegression(fit_intercept=False),
                                n_estimators=48, bootstrap=True, n_jobs=p_argument)

bs_regressor.fit(dataMatrix, Yvector)

end_time = time.time()
print(f'Tiempo de ejecución: {end_time - start_time} segundos')

#Ahora se extraen las 48 regresiones
regressions_values = np.array(list(map(lambda x: x.coef_, bs_regressor.estimators_)))
#Ordenamos verticalmente cada columna
ordered_regressions = np.sort(regressions_values, axis=0)
#Calculamos los percentiles 2.5 y 97.5, que serán los vectores extremos que serán
# los intervalos de confianza
lower_bound = np.percentile(regressions_values, 2.5, axis=0)
upper_bound = np.percentile(regressions_values, 97.5, axis=0)

# Se crea una lista booleana que indica si cada parámetro real está dentro de su
# respectivo intervalo
inside_interval = (coeficients >= lower_bound) & (coeficients <= upper_bound)

# Porcentaje de parámetros que sí estuvieron dentro del intervalo
coverage = np.mean(inside_interval)
# Parámetros que no quedaron dentro del intervalo
outside_indices = np.where(~inside_interval)[0]

print(f'Coverage: {coverage}')