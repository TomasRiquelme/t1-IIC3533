import numpy as np
from variables_config import returnYVector, returnCoeficients, returnDataMatrix, returnMinimumSquareSolution
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

# Importamos los valores a usar
dataMatrix = returnDataMatrix()
Yvector = returnYVector()
coeficients = returnCoeficients()
minimumSquareSolution = returnMinimumSquareSolution()

#Creamos el modelo regreso (con 5 jobs al principio)
#Aumentaré a 10 para que tome menos tiempo
bs_regressor = BaggingRegressor(estimator=LinearRegression(fit_intercept=False),
                                n_estimators=48, bootstrap=True, n_jobs=10)

bs_regressor.fit(dataMatrix, Yvector)

#Ahora se extraen las 48 regresiones
regressions_values = np.array(list(map(lambda x: x.coef_, bs_regressor.estimators_)))
#Ordenamos verticalmente cada columna
ordered_regressions = np.sort(regressions_values, axis=0)
#Calculamos los percentiles 2.5 y 97.5, que serán los vectores extremos que serán
# los intervalos de confianza
lower_bound = np.percentile(regressions_values, 2.5, axis=0)
upper_bound = np.percentile(regressions_values, 97.5, axis=0)

#Ahora, creo una lista booleana que me indica si cada parámetro real está dentro de su
# respectivo intervalo
inside_interval = (coeficients >= lower_bound) & (coeficients <= upper_bound)

#Obtenemos el porcentaje de parámetros que sí estuvieron dentro del intervalo
coverage = np.mean(inside_interval)
#Obtenemos los parámetros que no quedaron dentro del intervalo
outside_indices = np.where(~inside_interval)[0]

print(coverage)
#Ahora vamos con la experimentación de la cantidad de workers para ir optimizando el tiempo
#5 workers: 13s
#10 workers: 11s
#20 workers: 22s
#15 workers: 17s
#12 workers: 15s