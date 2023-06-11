
###############
### Imports ###
###############
# Imports
from selenium import webdriver
import requests
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
import pandas as pd
import numpy as np
import random
import csv
import datetime as dt
import os
from bs4 import BeautifulSoup
from configuration.key.encrypt import username, fernet
import importlib.resources
from cryptography.fernet import Fernet
import requests
import sys
from riotwatcher import LolWatcher, ApiError
import time
import json
import whois
from datetime import datetime
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, roc_curve, auc, make_scorer
from project_files.variables import *
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import ssl
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from kneed import KneeLocator
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.utils as keras_utils
import subprocess
import pydot
from sklearn.tree import export_graphviz
import graphviz
from contextlib import redirect_stdout
#from adjustText import adjust_text
import sys
from mpl_toolkits.mplot3d import Axes3D
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance

path_pca = "tables/Preprocesed/X_pca.csv"
path_y = "tables/Preprocesed/data_predictiva.csv"
path_X = "tables/Preprocesed/data_explicativas.csv"
path_loadings = "tables/Preprocesed/loadings.csv"
# Si el archivo existe, cargar el DataFrame desde el archivo
with open(path_pca, mode='rb') as archivo_pca:
    print("Accediendo a la aruta del df de metadata")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas
    X_pca = pd.read_csv(archivo_pca, sep=";", encoding='utf-8')

# Si el archivo existe, cargar el DataFrame desde el archivo
with open(path_y, mode='rb') as archivo_y:
    print("Accediendo a la aruta del df de metadata")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas

    data_predictiva = pd.read_csv(archivo_y, sep=";", encoding='utf-8')
# Si el archivo existe, cargar el DataFrame desde el archivo
with open(path_X, mode='r') as archivo_X:
    print("Accediendo a la aruta del df de metadata")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas
    data_explicativas = pd.read_csv(archivo_X, sep=";", encoding='utf-8')

# Si el archivo existe, cargar el DataFrame desde el archivo
with open(path_loadings, mode='rb') as archivo_loadings:
    print("Accediendo a la aruta del df de metadata")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas
    loadings = pd.read_csv(archivo_loadings, sep=";", encoding='utf-8')


#############################
#### CREACION DEL MODELO ####
#############################

#variables_explicativas = ['dif_kills', 'dif_inhibitor', 'dif_tower', 'dif_dmg_dealt_to_objectives', 'dif_gold_earned', 'dif_magic_dmg_dealt', 'dif_physical_dmg_dealt', 'dif_true_dmg_dealt', 'dif_gold_spent']
#X = data_explicativas[variables_explicativas]
# Si se usa el criterio del PCA:



path_pca = "tables/Preprocesed/X_pca.csv"
path_y = "tables/Preprocesed/data_predictiva.csv"
path_X = "tables/Preprocesed/data_explicativas.csv"
path_loadings = "tables/Preprocesed/loadings.csv"
# Si el archivo existe, cargar el DataFrame desde el archivo
with open(path_pca, mode='rb') as archivo_pca:
    print("Accediendo a la aruta del df de metadata")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas
    X_pca = pd.read_csv(archivo_pca, sep=";", encoding='utf-8')

# Si el archivo existe, cargar el DataFrame desde el archivo
with open(path_y, mode='rb') as archivo_y:
    print("Accediendo a la aruta del df de metadata")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas

    data_predictiva = pd.read_csv(archivo_y, sep=";", encoding='utf-8')
# Si el archivo existe, cargar el DataFrame desde el archivo
with open(path_X, mode='r') as archivo_X:
    print("Accediendo a la aruta del df de metadata")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas
    data_explicativas = pd.read_csv(archivo_X, sep=";", encoding='utf-8')

# Si el archivo existe, cargar el DataFrame desde el archivo
with open(path_loadings, mode='rb') as archivo_loadings:
    print("Accediendo a la aruta del df de metadata")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas
    loadings = pd.read_csv(archivo_loadings, sep=";", encoding='utf-8')


data_explicativas = data_explicativas[['dif_kills', 'baron_first_red', 'inhibitor_first_blue', 'dif_dragons', 'dif_total_minions_killed', 'dif_physical_dmg_dealt', 'dif_magic_dmg_dealt']]


######################
#### Red Neuronal ####
######################

# Es importante binarizar las variables categóricas, así que para la región se crearán 3 columnas: region_KR, region_EUW1, region_NA1 donde, por ejemplo
# para la region_K, contendrá un 1 si es de esa región y un 0 si no


#data_explicativas['region_KR']=(data_explicativas['region'] == 'KR').astype(int)
#data_explicativas['region_EUW1']=(data_explicativas['region'] == 'EUW1').astype(int)
#data_explicativas['region_NA1']=(data_explicativas['region'] == 'NA1').astype(int)

# Carga del conjunto de datos de entrenamiento y test

X = data_explicativas
y = data_predictiva
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Normalización de los datos

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### Es importante normalizar los datos para mejorar el rendimiento de la red neuronal

##Red neuronal sencilla con 3 capas:
## Capa de entrada : tantas neuronas como variables tenga mi conjunto de datos, es decir, 35. 
## Capa oculta : Técnica de validación cruzada para buscar el mejor valor para el número de neuronas en la capa oculta de una red neuronal

# Definir la lista de posibles valores para el número de neuronas en la capa oculta
neuronas_ocultas = [7, 14, 30, 5, 100, 200]

# Realizar validación cruzada para encontrar el mejor número de neuronas
mejor_neuronas = None
mejor_precision = 0

for num_neuronas in neuronas_ocultas:
    precision_promedio = 0
    
    # Usar un valor de k igual a 5 se considera una buena práctica
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, val_index in kf.split(X_train):
        # Dividir los datos en entrenamiento y validación
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # Construir el modelo de red neuronal
        model = Sequential()
        model.add(Dense(num_neuronas, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Entrenar el modelo
        model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)
        
        # Evaluar la precisión en los datos de validación
        _, accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        precision_promedio += accuracy
    
    precision_promedio /= 5  # Calcular el promedio de precisión para los 5 folds
    
    # Actualizar el mejor número de neuronas si se obtiene una mejor precisión
    if precision_promedio > mejor_precision:
        mejor_neuronas = num_neuronas
        mejor_precision = precision_promedio

print("El mejor número de neuronas en la capa oculta es:", mejor_neuronas)
tf.random.set_seed(1)

## Capa de salida, es la variable binaria de respuesta, por lo que se utilizará una única neurona en esta capa.

# Crear el modelo de red neuronal
modelRN = tf.keras.Sequential([
    tf.keras.layers.Dense(mejor_neuronas, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
modelRN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
grafico = modelRN.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
# Calcular la importancia de características mediante la permutación
perm = permutation_importance(modelRN, X_test, y_test, scoring='r2', random_state=1)

# Mostrar las importancias de características
eli5.show_weights(perm, feature_names=X.columns.tolist())
# Crear un gráfico de barras para mostrar la importancia de las variables
plt.figure(figsize=(10, 6))
plt.barh(range(len(perm.importances_mean)), perm.importances_mean, align='center')
plt.yticks(range(len(perm.importances_mean)), X.columns.tolist())
plt.xlabel('Importancia')
plt.ylabel('Variables')
plt.title('Importancia de las variables - Permutación en Red Neuronal')

# Añadir los valores de importancia en las barras para que sea más legible
for i, coef in enumerate(perm.importances_mean):
    plt.text(coef, i, f"{coef:.2f}", ha='left', va='center')

plt.tight_layout()
plt.savefig('images/Modelos/Grafico_Importancia_Permutacion_Red_Neuronal.png')
plt.close()



plt.xlabel("Número de pruebas")
plt.ylabel("Magnitud de pérdida")
plt.title("Función de coste")
plt.plot(grafico.history["loss"])

ruta = 'images\Modelos\Entrenamiento Red neuronal.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()

# Evaluar el modelo en el conjunto de prueba
lossRN, accuracyRN = modelRN.evaluate(X_test, y_test)
ypredRN=modelRN.predict(X_test)
ypred=ypredRN
ypredRN=[1 if x >= 0.5 else 0 for x in ypredRN]
# Calcular el recall
recallRN = recall_score(y_test, ypredRN)
print("Loss: ", lossRN)
print("Accuracy: ", accuracyRN)
print("Recall: ", recallRN)
# Redirigir la salida de la consola a un archivo de texto
with open('tables\model_summary\summary_model_red_neuronal.txt', 'w') as f:
    with redirect_stdout(f):
        modelRN.summary()

# Leer el archivo de texto
with open('tables\model_summary\summary_model_red_neuronal.txt', 'r') as f2:
    summary_model_red_neuronal = f2.read()

# Crear una figura y un eje de matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Mostrar el texto en el eje
ax.text(0.1, 0.1, summary_model_red_neuronal, fontsize=12)

# Guardar la figura como imagen
plt.savefig('images\Modelos\summary_model_red_neuronal.png', bbox_inches='tight')

# Crear una representación gráfica del modelo
keras_utils.plot_model(modelRN, to_file='Grafico_Red_Neuronal.png', show_shapes=True, show_layer_names=True)

# Mostrar la imagen del modelo
img = plt.imread('Grafico_Red_Neuronal.png')
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')     
plt.tight_layout()
plt.savefig('images\Modelos\Grafico_Red_Neuronal.png')
plt.close()





#######################
#### Random Forest ####
#######################

X = data_explicativas
y = data_predictiva
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Normalización de los datos

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Para iterar sobre el posible número de estimadores:
estimadores = [5,10,50, 100, 500, 1000, 5000, 10000]
# Los estimadores especifican el número de árboles que se crean. 
# Un mayor número de árboles puede mejorar la precisión del modelo, pero también aumenta el tiempo de entrenamiento. 
## A partir de los 500 arboles, el proceso computacionalmente empieza a consumir más tiempo y no mejora la precisión
# Por otro lado, 
"""lista_errores = []
oob_error_list = []
for val in estimadores:
    model_rf = RandomForestClassifier(n_estimators=val, random_state=1, oob_score=True, min_samples_leaf=5, max_features='sqrt')
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred)
    print("Acurracy/Precisión con Random Forest, con número de estimadores a", val,":", accuracy_score(y_test, y_pred))
    oob_error = 1 - model_rf.oob_score_
    oob_error_list.append(oob_error)
    lista_errores.append(oob_error)
print(lista_errores)"""

# Para iterar sobre el v alor de max_features
# Valores evaluados
max_features_range = range(1, X_train.shape[1] + 1, 1)
lista_errores = []
oob_error_list = []
for val in max_features_range:
    model_rf = RandomForestClassifier(n_estimators=500, random_state=1, max_depth=val, oob_score=True, min_samples_leaf=5, max_features='sqrt')
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred)
    print("Acurracy/Precisión con Random Forest, con max_features_range a", val,":", accuracy_score(y_test, y_pred))
    oob_error = 1 - model_rf.oob_score_
    oob_error_list.append(oob_error)
    lista_errores.append(oob_error)
print(lista_errores)

##Parece que con 9 de profundiad ya deja de mejorar en gran cantidad

# Crear el modelo de Random Forest
modelRF = RandomForestClassifier(n_estimators=500, max_depth=6, oob_score=False, n_jobs=-1, random_state=42)
# Por defecto, el criterio del error es mse

# Entrenar el modelo
modelRF.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelRF.predict(X_test)
rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = y_pred,
        squared = False
       )
print(f"El error (rmse) de test es: {rmse}")

# Calcular la precisión del modelo
accuracyRF = accuracy_score(y_test, y_pred)
ypredRF=modelRF.predict(X_test)
ypred=ypredRF
ypred=[1 if x >= 0.5 else 0 for x in ypred]
# Calcular el recall
recallRF = recall_score(y_test, ypred)
print("Accuracy: ", accuracyRF)
print("Recall: ", recallRF)

# Obtener el primer árbol del modelo
tree = modelRF.estimators_[0]

# Exportar el árbol a formato DOT
dot_data = export_graphviz(tree, out_file = 'arbol.dot', feature_names=list(X.columns),  filled=True, rounded = True, precision = 1)

# Crear un gráfico a partir del archivo DOT
graph = graphviz.Source(dot_data)

(graph, ) = pydot.graph_from_dot_file('arbol.dot')
# Write graph to a png file
graph.write_png('images\Modelos\grafico_random_forest.png')

##########
# Para entender un poco más el gráfico, vamos a  reducir la profundidad dle árbol:
########


# Crear el modelo de Random Forest
model_pequeño = RandomForestClassifier(n_estimators=100, max_depth=3, oob_score=False, n_jobs=-1, random_state=42)
# Por defecto, el criterio del error es mse

# Entrenar el modelo
model_pequeño.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model_pequeño.predict(X_test)
rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = y_pred,
        squared = False
       )
print(f"El error (rmse) de test es: {rmse}")

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Obtener el primer árbol del modelo
tree = model_pequeño.estimators_[0]

# Exportar el árbol a formato DOT
dot_data = export_graphviz(tree, out_file = 'arbol2.dot', feature_names=list(X.columns),  filled=True, rounded = True, precision = 1)

# Crear un gráfico a partir del archivo DOT
graph_pequeño = graphviz.Source(dot_data)

(graph_pequeño, ) = pydot.graph_from_dot_file('arbol2.dot')
# Write graph to a png file
graph_pequeño.write_png('images\Modelos\grafico_random_forest2.png')


# Importancia de las variables incida en qué medida se mejora la predicción cuando se añade alguna de estas variables al modelo
importancia = list(modelRF.feature_importances_)
import_variables = [(feature, round(importance, 5)) for feature, importance in zip(list(X.columns), importancia)]
import_variables = sorted(import_variables, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('La importancia para la variable {:20} es: {}'.format(*pair)) for pair in import_variables]



feature_imp = modelRF.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(X.columns, feature_imp)
plt.xticks(rotation=90)
plt.xlabel('Variables')
plt.ylabel('Importancia')
plt.title('Importancia de variables - Random Forest')

# Añadir los valores de los coeficientes en las barras para que sea más legible
for i, coef in enumerate(feature_imp):
    plt.text(i, coef, round(coef, 2), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('images/Modelos/Grafico_Feature_Importante_Random_Forest.png')
plt.close()






#########################
# Support Vector Machines
#########################

X = data_explicativas
y = data_predictiva
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Normalizar los datos de entrada
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## GRID SEARCH: permite encontrar el mejor valor para el parámetro de regularización (C)
# Definir los valores posibles para el parámetro de regularización C
grid =  {'C': [0.1, 1, 5, 10, 100]}

# Crear el modelo de Support Vector Machine
modelSV = SVC(kernel='rbf', gamma='scale', random_state=42)

# Realizar la búsqueda de cuadrícula
grid_search = GridSearchCV(estimator=modelSV, param_grid=grid, cv=5)
grid_search.fit(X_train, y_train)

# Obtener el mejor valor de C encontrado
best_C = grid_search.best_params_['C']
print("Mejor valor de C:", best_C)

# Crear un nuevo modelo con el mejor valor de C
best_modelSV = SVC(kernel='rbf', C=best_C, gamma='scale', random_state=42, probability=True)
#kernel='rbf': RBF (Radial Basis Function) es una función de kernel comúnmente utilizada en SVM. Es útil cuando los datos no son linealmente separables en el #espacio de características original. El kernel RBF mapea los datos a un espacio de mayor dimensionalidad donde es más probable que los datos sean linealmente #separables. Esta función de kernel permite la creación de fronteras de decisión no lineales en el espacio original.

#gamma='scale': El parámetro gamma controla el alcance de influencia de cada ejemplo de entrenamiento en la formación de la frontera de decisión. Un valor bajo de #gamma significa una influencia más amplia y una frontera de decisión más suave, mientras que un valor alto de gamma significa una influencia más localizada y una #frontera de decisión más ajustada a los datos de entrenamiento. En este caso, gamma='scale' indica que se utilizará el valor 1 / (n_features * X.var()) como valor #de gamma, donde n_features es el número de características y X.var() es la varianza de los datos de entrada. Esto es útil para normalizar la influencia de gamma #en función de la escala de los datos.

# Entrenar el modelo con el mejor valor de C
best_modelSV.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
ypredSV = best_modelSV.predict(X_test)

# Calcular la precisión del modelo
accuracySV = accuracy_score(y_test, ypredSV)
ypred=ypredSV
ypred=[1 if x >= 0.5 else 0 for x in ypred]
# Calcular el recall
recallSV = recall_score(y_test, ypred)
print("Accuracy: ", accuracySV)
print("Recall: ", recallSV)



# Aplicar PCA para reducir la dimensionalidad a 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Obtener las predicciones para el área de decisión
Z = best_modelSV.predict(X_train)

# gráfico de dispersión en 2D
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Z, cmap=plt.cm.Paired)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Hiperplano resultante del SVM con PCA')
plt.tight_layout()
plt.savefig('images/Modelos/SVM_2D.png')
plt.close()



# Calcular la importancia de características mediante la permutación
perm = permutation_importance(best_modelSV, X_test, y_test, scoring='r2', random_state=1)

# Mostrar las importancias de características
eli5.show_weights(perm, feature_names=X.columns.tolist())
# Crear un gráfico de barras para mostrar la importancia de las variables
plt.figure(figsize=(10, 6))
plt.barh(range(len(perm.importances_mean)), perm.importances_mean, align='center')
plt.yticks(range(len(perm.importances_mean)), X.columns.tolist())
plt.xlabel('Importancia')
plt.ylabel('Variables')
plt.title('Importancia de las variables - Permutación en Red Neuronal')

# Añadir los valores de importancia en las barras para que sea más legible
for i, coef in enumerate(perm.importances_mean):
    plt.text(coef, i, f"{coef:.2f}", ha='left', va='center')

plt.tight_layout()
plt.savefig('images/Modelos/Grafico_Importancia_Permutacion_SVM.png')
plt.close()


#########################
# Regressión Logística
#########################

X = data_explicativas
y = data_predictiva
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Normalizar los datos de entrada
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


modelLR = LogisticRegression()
modelLR.fit(X_train, y_train)

componentes_coeficientes = modelLR.coef_
# Obtener las variables más influyentes para cada componente
variables_influyentes = []
for i, componentes in enumerate(componentes_coeficientes):
    indices_top_variables = np.argsort(np.abs(componentes))[::-1][:3]  # Obtener los índices de las 3 variables más influyentes
    variables_influyentes.append(X.columns[indices_top_variables])

# Imprimir las variables más influyentes para cada componente
for i, variables in enumerate(variables_influyentes):
    print(f"Componente {i+1}: {variables}")

# Prediccióm
ypredLR = modelLR.predict(X_test)

# Calcular la precisión del modelo
accuracyLR = accuracy_score(y_test, ypredLR)
ypred=ypredLR
ypred=[1 if x >= 0.5 else 0 for x in ypred]
# Calcular el recall
recallLR = recall_score(y_test, ypred)
print("Accuracy: ", accuracyLR)
print("Recall: ", recallLR)

# Obtener los coeficientes
coeficientes = modelLR.coef_

# Imprimir los coeficientes
for i, coef in enumerate(coeficientes[0]):
    print(f"Coeficiente {i+1}: {coef}")

# Obtener los nombres de las variables explicativas
variables_explicativas = X.columns

# Obtener los coeficientes del modelo
coeficientes = modelLR.coef_[0]
print("Intercept:", modelLR.intercept_)
# Crear un gráfico de barras para visualizar los coeficientes
plt.figure(figsize=(10, 6))
plt.bar(variables_explicativas, coeficientes)
plt.xticks(rotation=90)
plt.xlabel('Variables Explicativas')
plt.ylabel('Coeficientes')
plt.title('Coeficientes de Regresión Logística')

# Añadir los valores de los coeficientes en las barras para que sea más legible
for i, coef in enumerate(coeficientes):
    plt.text(i, coef, round(coef, 2), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('images/Modelos/Grafico_Regresion_Logistica.png')
plt.close()


## CURVA DE ROC PARA COMPARAR TODOS LOS MODELOS

##Nos quedamos solo con los positivos:

ypredRN_prob = modelRN.predict(X_test)[:, 0]
ypredRF_prob = modelRF.predict_proba(X_test)[:, 1]
ypredSV_prob = best_modelSV.predict_proba(X_test)[:, 1]
ypredLR_prob = modelLR.predict_proba(X_test)[:, 1]

#  tasas de verdaderos positivos y falsos
ypredRN_false, ypredRN_true, _ = roc_curve(y_test, ypredRN_prob)
ypredRF_false, ypredRF_true, _ = roc_curve(y_test, ypredRF_prob)
ypredSV_false, ypredSV_true, _ = roc_curve(y_test, ypredSV_prob)
ypredLR_false, ypredLR_true, _ = roc_curve(y_test, ypredLR_prob)

# área bajo la curva ROC
auc_ypredRN = auc(ypredRN_false, ypredRN_true)
auc_ypredRF = auc(ypredRF_false, ypredRF_true)
auc_ypredSV = auc(ypredSV_false, ypredSV_true)
auc_ypredLR = auc(ypredLR_false, ypredLR_true)

# Crear el gráfico ROC
plt.figure(figsize=(8, 6))
plt.plot(ypredRN_false, ypredRN_true, label='Modelo 1 (AUC = {:.2f})'.format(auc_ypredRN))
plt.plot(ypredRF_false, ypredRF_true, label='Modelo 2 (AUC = {:.2f})'.format(auc_ypredRF))
plt.plot(ypredSV_false, ypredSV_true, label='Modelo 3 (AUC = {:.2f})'.format(auc_ypredSV))
plt.plot(ypredLR_false, ypredLR_true, label='Modelo 4 (AUC = {:.2f})'.format(auc_ypredLR))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Curva de Roc')

# Configurar el gráfico
plt.title('Curva ROC')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('images/Modelos/curvas_de_roc.png')
plt.close()