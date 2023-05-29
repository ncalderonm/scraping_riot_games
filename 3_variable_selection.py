
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
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
import subprocess
#from adjustText import adjust_text
import sys



#################
### Carga de datos 
#################

path_X = "tables/Preprocesed/data_explicativas.csv"
path_y = "tables/Preprocesed/data_predictiva.csv"

# Si el archivo existe, cargar el DataFrame desde el archivo
with open(path_X, mode='rb') as archivo_X:
    print("Accediendo a la aruta del df de metadata")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas
    data_explicativas = pd.read_csv(archivo_X, sep=";", encoding='utf-8')

# Si el archivo existe, cargar el DataFrame desde el archivo
with open(path_y, mode='rb') as archivo_y:
    print("Accediendo a la aruta del df de metadata")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas
    data_predictiva = pd.read_csv(archivo_y, sep=";", encoding='utf-8')



#########################################
## Selección de variables por correlación
#########################################
#data_corr = teams_clean.drop('region', axis=1)
corr = data_explicativas.corr()
# Generar el heatmap
plt.figure(figsize=(14, 12))
sns.set(font_scale=0.8)
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')
plt.tight_layout()
plt.savefig('images/preprocessed_images/correlation.png')
plt.close()
# Selección de variables con correlación alta
high_corr_vars = set()
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i,j]) > 0.7:
            colname = corr.columns[i]
            high_corr_vars.add(colname)
            
# Eliminación de variables con correlación alta
variables_seleccionadas = [var for var in list(data_explicativas.columns) if var not in high_corr_vars]
print("Variables seleccionadas por correlación: ", variables_seleccionadas)

corr2 = data_explicativas[variables_seleccionadas].corr()
# Generar el heatmap
plt.figure(figsize=(14, 12))
sns.set(font_scale=0.8)
sns.heatmap(corr2, cmap='coolwarm', annot=True, fmt='.2f')
plt.tight_layout()
plt.savefig('images/preprocessed_images/correlation2.png')
plt.close()

data_explicativas = data_explicativas[variables_seleccionadas]
############################################################
## Selección de variables por importancia de características
############################################################
# División en variables independientes y dependiente
columnas = list(data_explicativas.columns)
X = data_explicativas
y = data_predictiva

# Creación de modelo Random Forest
rfc = RandomForestClassifier()

# Entrenamiento del modelo
rfc.fit(X, y)

# Obtención de la importancia de las características
feature_importances = rfc.feature_importances_

# Selección de las variables con importancia alta
variables_seleccionadas = []
for i in range(len(columnas)-1):
    if feature_importances[i] > 0.1:
        variables_seleccionadas.append(columnas[i])
variables_seleccionadas.append('win_blue')

print("Variables seleccionadas por importancia de características: ", variables_seleccionadas)


nombres_variables = variables_seleccionadas[:-1]
importancia_variables = importancia_variables = [feature_importances[columnas.index(var)] for var in nombres_variables]

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
plt.barh(nombres_variables, importancia_variables)
plt.xlabel('Importancia')
plt.ylabel('Variables')
plt.title('Importancia de las características')
plt.tight_layout()
plt.savefig('images/preprocessed_images/Importancia_Caracteristicas.png')
plt.close()


################################################
## Selección de variables por regresión de Lasso
#################################################
# Normalizar los datos
scaler = StandardScaler()

X = data_explicativas
y = data_predictiva

scaler.fit(X)
X_scaled = scaler.transform(X)
# Crea un modelo de regresión Lasso y ajusta los datos
# Lista de valores alpha a probar
alphas = [0.005, 0.01, 0.1, 1.0, 10.0]
# Inicializar el modelo de regresión LassoCV
lasso_cv = LassoCV(alphas=alphas, cv=5)
# Ajustar el modelo utilizando la validación cruzada
lasso_cv.fit(X_scaled, y)
# Obtener el mejor valor de alpha
best_alpha = lasso_cv.alpha_
# Obtener los coeficientes del modelo con el mejor alpha
lasso_coef = lasso_cv.coef_
# Obtener las posiciones de los coeficientes no nulos
nonzero_positions = [i for i, x in enumerate(list(lasso_coef)) if x != 0]
# Obtener los nombres de las variables seleccionadas por Lasso
variables_seleccionadas = list(X_scaled[nonzero_positions])
print("Mejor valor de alpha:", best_alpha)


# Crea un modelo de regresión Lasso y ajusta los datos
lasso = Lasso(alpha=best_alpha) # 0.00005
lasso.fit(X_scaled, y)
lasso_coef = lasso.coef_
nonzero_positions = [i for i, x in enumerate(list(lasso_coef)) if x != 0]
col_data = data_explicativas.columns
# imprime las posiciones obtenidas
list_var_lasso = list(col_data[nonzero_positions])
print("Variables seleccionadas por regresión Lasso: ", list(col_data[nonzero_positions]))


# Obtener las posiciones y nombres de las variables seleccionadas por Lasso
nonzero_positions = [i for i, x in enumerate(list(lasso_coef)) if x != 0]
variables_seleccionadas = col_data[nonzero_positions]

# Obtener los coeficientes correspondientes a las variables seleccionadas
coeficientes_seleccionados = lasso_coef[nonzero_positions]

# Crear un gráfico de barras horizontales
plt.figure(figsize=(10, 6))
plt.barh(variables_seleccionadas, coeficientes_seleccionados)
plt.xlabel('Coeficiente')
plt.ylabel('Variable')
plt.title('Importancia de las variables seleccionadas por Lasso')
plt.tight_layout()
plt.savefig('images/preprocessed_images/Importancia_Lasso.png')
plt.close()

#####################################################################
## Selección de variables por eliminacón recursiva de características
#####################################################################
X = data_explicativas
y = data_predictiva

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled=pd.DataFrame(X_scaled, columns=X.columns)
# División en train y test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.2, random_state=1)

selected_features = []
best_accuracy = 0.0

contador = []
accuracies = []

# Iteración hasta alcanzar el criterio de parada
while len(selected_features) < X_scaled.shape[1]:
    best_feature = None
    best_feature_accuracy = 0.0
    
    # Iteración sobre las características no seleccionadas
    for feature in X_scaled.columns:
        if feature not in selected_features:
            temp_features = selected_features + [feature]
            
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train[temp_features], y_train)
            
            # Predicción en el conjunto de prueba y cálculo del rendimiento
            y_pred = model.predict(X_test[temp_features])
            accuracy = accuracy_score(y_test, y_pred)
            print("El accuracy con las variables", selected_features, "es de:", accuracy)
            
            # Comprobación de si el rendimiento mejora
            if accuracy > best_feature_accuracy:
                best_feature = feature
                best_feature_accuracy = accuracy
    
    # Comprobación si el rendimiento mejora en comparación con el mejor rendimiento calculado anteriormente
    if best_feature_accuracy > best_accuracy:
        selected_features.append(best_feature)
        best_accuracy = best_feature_accuracy
    else:
        # Se detiene la iteración si no hay mejora en el rendimiento
        break
    
    contador.append(len(selected_features))
    accuracies.append(best_feature_accuracy)
# Resultado final de las características seleccionadas
print("Variables seleccionadas por eliminación recursiva de variables es: ", selected_features)

# Graficar los resultados
plt.plot(contador, accuracies)
plt.xlabel('Número de características')
plt.ylabel('Precisión')
plt.title('Accuracy sobre las variables seleccionadas')
plt.tight_layout()
plt.savefig('images/preprocessed_images/Precisión y variables seleccionadas.png')
plt.close()

# Gráfrico de valoresp or varibles
variable_values = []
for feature in selected_features:
    variable_values.append((feature, X[feature].mean()))

variable_values = sorted(variable_values, key=lambda x: x[1], reverse=True)

# Extraer los nombres de variables y los valores
variables = [x[0] for x in variable_values]
values = [x[1] for x in variable_values]

# gráfico de barras
plt.barh(range(len(variables)), values, align='center')
plt.yticks(range(len(variables)), variables)
plt.xlabel('Valor promedio')
plt.ylabel('Variables')
plt.title('Valores promedio de las variables seleccionadas')
plt.tight_layout()
plt.savefig('images/preprocessed_images/Valores promedio de las variables seleccionada.png')
plt.close()


#####################################################################
## Selección de variables por componentes principales (PCA)
#####################################################################
# Pasos:
# 1) Normalizar
# 2) Calcular autovectores y autovalores a partir de la matriz de covarianzas
# 3) Seleccionar autovectores correspondientes a las componentes principales
# 4) Nuevo dataset sobre el nuevo espacio vectorial

#path_pca_number_of_components = 'images/preprocessed_images/PCA_number_of_components.png'
#path_pca_acumulated_variance = 'images/preprocessed_images/PCA - Varianza explicada acumulada.png'
#path_pca_PC1_PC4 = 'images/preprocessed_images/PCA componentes de la PC1 a la PC4.png'
#path_pca_PC5_PC8 = 'images/preprocessed_images/PCA componentes de la PC5 a la PC8.png'
#if os.path.isfile(tabla_players):
# 1) Normalizar

scaler = StandardScaler()
X = data_explicativas
X_scaled = scaler.fit_transform(X)
y = data_predictiva

# 2) Calculamos la matriz de covarianza

print('NumPy covariance matrix: \n%s' %np.cov(X_scaled.T))

# Calculamos los autovalores y autovectores de la matriz y los mostramos

cov_mat = np.cov(X_scaled.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# 3
#  Hacemos una lista de parejas (autovector, autovalor) 
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Ordenamos estas parejas den orden descendiente con la función sort
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visualizamos la lista de autovalores en orden desdenciente
print('Autovalores en orden descendiente:')
for i in eig_pairs:
    print(i[0])


# Método o técnica del codo para obtener el número de componentes principales idoneo:

# Representación gráfica de la varianza explicada por cada uno de los components principales. Se busca el punto (codo) de inflexión en la curva
# Esto indicará el nº de componentes principales que explican la mayoría de la varianza

pca = PCA().fit(X_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Número de componentes')
plt.ylabel('Varianza explicada acumulada')
plt.tight_layout()
plt.savefig('images/preprocessed_images/PCA_number_of_components.png')
plt.close()

# A partir de los autovalores, calculamos la varianza explicada
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Representamos en un diagrama de barras la varianza explicada por cada autovalor, y la acumulada
#with plt.style.context('seaborn-pastel'):
plt.figure(figsize=(6, 4))

plt.bar(range(32), var_exp, alpha=0.5, align='center',
        label='Varianza individual explicada', color='g')
plt.step(range(32), cum_var_exp, where='mid', linestyle='--', label='Varianza explicada acumulada')
plt.ylabel('Ratio de Varianza Explicada')
plt.xlabel('Componentes Principales')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('images/preprocessed_images/PCA - Varianza explicada acumulada.png')
plt.close()

# Entorno al 45% de la varianza total se explica a partir de la primera componente principal.  A partir de la segunda y la tercera se explica entorno a un 10% de la varianza total respectivamente. Entonces, según lo que vemos en la gráfica necesitaríamos 4 componentes principales para explicar entorno al 80% de la varianza total. 

# Transformar las 32 variables en un espacio de dimensión 4D


pca = PCA(n_components=8)
#pca.fit_transform(X_scaled)
X_pca = pca.fit_transform(X_scaled)
components = pca.transform(np.eye(X_scaled.shape[1]))
components = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'], index=data_explicativas.columns)

path_loadings = "tables/Preprocesed/loadings.csv"
loadings.to_csv(path_loadings, index=False, encoding='utf-8', sep=";")

pca_results = {'components': components, 'loadings': loadings, 'explained_variance_ratio': pca.explained_variance_ratio_}

df_pca = pd.DataFrame(pca_results['components'], columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])

# Crear un DataFrame con las componentes principales
components_df = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])

# Obtener las variables más importantes de cada componente
top_vars_df = pd.DataFrame()
for i, component in enumerate(pca.components_):
    sorted_idx = np.abs(component).argsort()[::-1]
    top_vars = X.columns[sorted_idx][:3]  # Seleccionar las 3 variables más importantes
    top_vars_df[f'PC{i+1}'] = top_vars

# Gráfico 1: Comparar todas las componentes entre PC1 y PC4
sns.set(style='ticks')
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

for i, ax in enumerate(axes.flatten()):
    sns.scatterplot(x=f'PC{i+1}', y='PC1', data=components_df, ax=ax)
    ax.set_xlabel(f'PC{i+1}', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC1', fontsize=12, fontweight='bold')
    
    # Añadir anotaciones para las variables más importantes
    for j, var in enumerate(top_vars_df[f'PC{i+1}']):
        ax.annotate(var, (components_df.loc[j, f'PC{i+1}'], components_df.loc[j, 'PC1']), fontsize=10)
        
plt.tight_layout()
plt.savefig('images/preprocessed_images/PCA componentes de la PC1 a la PC4.png')
plt.close()

# Gráfico 2: Comparar las componentes de PC5 a PC8
sns.set(style='ticks')
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

for i, ax in enumerate(axes.flatten()):
    sns.scatterplot(x=f'PC{i+5}', y='PC1', data=components_df, ax=ax, color='red')
    ax.set_xlabel(f'PC{i+5}', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC1', fontsize=12, fontweight='bold')
    
    # Añadir anotaciones para las variables más importantes
    for j, var in enumerate(top_vars_df[f'PC{i+5}']):
        ax.annotate(var, (components_df.loc[j, f'PC{i+5}'], components_df.loc[j, 'PC1']), fontsize=10)
        
plt.tight_layout()
plt.savefig('images/preprocessed_images/PCA componentes de la PC5 a la PC8.png')
plt.close()

######
# Dataframe para utilizar en la creación del modelo de predicción
component_names = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']
X_pca = pd.DataFrame(X_pca, columns=component_names)

path_pca = "tables/Preprocesed/X_pca.csv"
X_pca.to_csv(path_pca, index=False, encoding='utf-8', sep=";")
