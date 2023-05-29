
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
import importlib.resources
import requests
import sys
import time
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression, Lasso
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
from adjustText import adjust_text
timestamp = datetime.now()
hora_inicio_ejecucion = timestamp.strftime('%Y-%m-%d %H-%M-%S')
print("La hora de inicio de la ejecución es:", hora_inicio_ejecucion)
###########
### URL ###
###########


# Hacemos un If para saber si está creada la tabla del CSV, si lo está, seguimos con esa, sino, la creamos.
tabla_players = 'tables/csv/players.csv'
print(os.path.isfile(tabla_players))

matches_md = 'tables/csv/match_md.csv'
matches_info = 'tables/csv/match_info.csv'
matches_info_teams = 'tables/csv/match_info_team.csv'
matches_participants = 'tables/csv/match_participant.csv'

# Si el archivo existe, cargar el DataFrame desde el archivo
with open(matches_md, mode='rb') as archivo_matches_md:
    print("Accediendo a la aruta del df de metadata")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas matches_md
    df_matches_md = pd.read_csv(archivo_matches_md, sep=";", encoding='utf-8')

# Si el archivo existe, cargar el DataFrame desde el archivo
with open(matches_info, mode='rb') as archivo_matches_info:
    print("Accediendo a la aruta del df de información")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas matches_info
    df_matches_info = pd.read_csv(archivo_matches_info, sep=";", encoding='utf-8')

# Si el archivo existe, cargar el DataFrame desde el archivo
with open(matches_info_teams, mode='rb') as archivo_matches_info_teams:
    print("Accediendo a la aruta del df de equipos")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas matches_info_teams
    df_matches_info_teams = pd.read_csv(archivo_matches_info_teams, sep=";", encoding='utf-8')

# Si el archivo existe, cargar el DataFrame desde el archivo
with open(matches_participants, mode='rb') as archivo_matches_participants:
    print("Accediendo a la aruta del df de participantes")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas matches_info
    df_matches_participant = pd.read_csv(archivo_matches_participants, header=0, sep=";", encoding='utf-8')


#########################
# PREPROCESSADO
#########################

## 1) Eliminación de nulls de los csv (si existe)

# verificar si hay valores nulos
print("Existen nuloss?")
print(df_matches_participant.isnull())

""# eliminar filas con valores nulos
df_sin_nas = df.dropna()

# imprimir dataframe resultante
print(df_sin_nas)
""
# 2) Creacion de un dataframe para los equipos

df_teams = pd.DataFrame()

path_teams = "tables/Preprocesed/TeamsCleaned.csv"
if os.path.isfile(path_teams):
    # Si el archivo existe, cargar el DataFrame desde el archivo
    with open(path_teams, mode='rb') as team_field:
        print("Accediendo a la aruta del df de TeamsCleaned")
        # Lee el contenido del archivo CSV utilizando la biblioteca pandas matches_md
        teams_clean = pd.read_csv(team_field, sep=";", encoding='utf-8')
else:
    # Filtramos participantes por id de partida

    for i in range(len(df_matches_info['matchId'])):
        partida_i = df_matches_info.loc[i,'matchId']
        df_i_participant = df_matches_participant.loc[df_matches_participant['matchId']==partida_i]
        df_i_team = df_matches_info_teams.loc[df_matches_info_teams['matchId']==partida_i]
        df_i_info = df_matches_info.loc[df_matches_info['matchId']==partida_i]
        # Región
        df_teams.loc[i, 'region'] = df_i_info['platformId'].iloc[0]
        # Filtrando por equipo azul
        blue_team = df_i_team.loc[df_i_team['teamId'].astype(int)==100]
        df_teams.loc[i, 'baron_first_blue'] = int(blue_team['baron_first'].iloc[0])
        df_teams.loc[i, 'champion_first_blue'] = int(blue_team['champion_first'].iloc[0])
        df_teams.loc[i, 'dragon_first_blue'] = int(blue_team['dragon_first'].iloc[0])
        df_teams.loc[i, 'inhibitor_first_blue'] = int(blue_team['inhibitor_first'].iloc[0])
        df_teams.loc[i, 'herald_first_blue'] = int(blue_team['riftHerald_first'].iloc[0])
        df_teams.loc[i, 'tower_first_blue'] = int(blue_team['tower_first'].iloc[0])


        # Filtrando por equipo rojo

        red_team = df_i_team[df_i_team['teamId'].astype(int)==200]
        df_teams.loc[i, 'baron_first_red'] = int(red_team['baron_first'].iloc[0])
        df_teams.loc[i, 'champion_first_red'] = int(red_team['champion_first'].iloc[0])
        df_teams.loc[i, 'dragon_first_red'] = int(red_team['dragon_first'].iloc[0])
        df_teams.loc[i, 'inhibitor_first_red'] = int(red_team['inhibitor_first'].iloc[0])
        df_teams.loc[i, 'herald_first_red'] = int(red_team['riftHerald_first'].iloc[0])
        df_teams.loc[i, 'tower_first_red'] = int(red_team['tower_first'].iloc[0])

        # Filtrando los participantes por equipo azul
        df_i_participant_blue = df_i_participant.loc[df_i_participant['teamId'].astype(int) == 100]
            # Filtrando los participantes por equipo azul
        df_i_participant_red = df_i_participant.loc[df_i_participant['teamId'].astype(int) == 200]
        # Variables de diferencia del equipo azul respecto el equipo rojo
        df_teams.loc[i, 'dif_kills'] = int(blue_team['champion_kills'].iloc[0] - red_team['champion_kills'].iloc[0])
        df_teams.loc[i, 'dif_barons'] = int(blue_team['baron_kills'].iloc[0] - red_team['baron_kills'].iloc[0])
        df_teams.loc[i, 'dif_dragons'] = int(blue_team['dragon_kills'].iloc[0] - red_team['dragon_kills'].iloc[0])
        df_teams.loc[i, 'dif_inhibitor'] = int(blue_team['inhibitor_kills'].iloc[0]- red_team['baron_kills'].iloc[0])
        df_teams.loc[i, 'dif_herald'] = int(blue_team['riftHerald_kills'].iloc[0] - red_team['riftHerald_kills'].iloc[0])
        df_teams.loc[i, 'dif_tower'] = int(blue_team['tower_kills'].iloc[0] - red_team['tower_kills'].iloc[0])
        
        df_teams.loc[i, 'dif_dmg_dealt_to_buildings'] = int(sum(df_i_participant_blue.loc[:, 'damageDealtToBuildings']) - sum(df_i_participant_red.loc[:, 'damageDealtToBuildings']))
        df_teams.loc[i, 'dif_dmg_dealt_to_objectives'] = int(sum(df_i_participant_blue.loc[:, 'damageDealtToObjectives']) - sum(df_i_participant_red.loc[:, 'damageDealtToObjectives']))
        df_teams.loc[i, 'dif_dmg_dealt_to_turrets'] = int(sum(df_i_participant_blue.loc[:, 'damageDealtToTurrets']) - sum(df_i_participant_red.loc[:, 'damageDealtToTurrets']))
        df_teams.loc[i, 'dif_dmg_self_mitigated'] = int(sum(df_i_participant_blue.loc[:, 'damageSelfMitigated']) - sum(df_i_participant_red.loc[:, 'damageSelfMitigated']))
        df_teams.loc[i, 'dif_gold_earned'] = int(sum(df_i_participant_blue.loc[:, 'goldEarned']) - sum(df_i_participant_red.loc[:, 'goldEarned']))
        df_teams.loc[i, 'dif_gold_spent'] = int(sum(df_i_participant_blue.loc[:, 'goldSpent']) - sum(df_i_participant_red.loc[:, 'goldSpent']))
        df_teams.loc[i, 'dif_magic_dmg_dealt'] = int(sum(df_i_participant_blue.loc[:, 'magicDamageDealt']) - sum(df_i_participant_red.loc[:, 'magicDamageDealt']))
        #df_teams.loc[i, 'dif_magic_dmg_to_champions'] = sum(df_i_participant_blue.loc[:, 'magicDamageDealtToChampions']) - sum(df_i_participant_red.loc[:, 'magicDamageDealtToChampions'])
        #df_teams.loc[i, 'dif_magic_dmg_taken'] = sum(df_i_participant_blue.loc[:, 'magicDamageTaken']) - sum(df_i_participant_red.loc[:, 'magicDamageTaken'])
        df_teams.loc[i, 'dif_physical_dmg_dealt'] = int(sum(df_i_participant_blue.loc[:, 'physicalDamageDealt']) - sum(df_i_participant_red.loc[:, 'physicalDamageDealt']))
        #df_teams.loc[i, 'dif_physical_dmg_to_champions'] = sum(df_i_participant_blue.loc[:, 'physicalDamageDealtToChampions']) - sum(df_i_participant_red.loc[:, 'physicalDamageDealtToChampions'])
        #df_teams.loc[i, 'dif_physical_dmg_taken'] = sum(df_i_participant_blue.loc[:, 'physicalDamageTaken']) - sum(df_i_participant_red.loc[:, 'physicalDamageTaken']) 
        #df_teams.loc[i, 'dif_total_dmg_dealt'] = sum(df_i_participant_blue.loc[:, 'totalDamageDealt']) - sum(df_i_participant_red.loc[:, 'totalDamageDealt'])
        #df_teams.loc[i, 'dif_total_dmg_to_champions'] = sum(df_i_participant_blue.loc[:, 'totalDamageDealtToChampions']) - sum(df_i_participant_red.loc[:, 'totalDamageDealtToChampions'])
        #df_teams.loc[i, 'dif_total_dmg_taken'] = sum(df_i_participant_blue.loc[:, 'totalDamageTaken']) - sum(df_i_participant_red.loc[:, 'totalDamageTaken'])
        df_teams.loc[i, 'dif_true_dmg_dealt'] = int(sum(df_i_participant_blue.loc[:, 'trueDamageDealt']) - sum(df_i_participant_red.loc[:, 'trueDamageDealt']))
        #df_teams.loc[i, 'dif_true_dmg_to_champions'] = int(sum(df_i_participant_blue.loc[:, 'trueDamageDealtToChampions']) - sum(df_i_participant_red.loc[:, 'trueDamageDealtToChampions'])
        #df_teams.loc[i), 'dif_true_dmg_taken'] = int(sum(df_i_participant_blue.loc[:, 'trueDamageTaken']) - sum(df_i_participant_red.loc[:, 'trueDamageTaken']) 
        df_teams.loc[i, 'dif_total_minions_killed'] = int(sum(df_i_participant_blue.loc[:, 'totalMinionsKilled']) - sum(df_i_participant_red.loc[:, 'totalMinionsKilled']))
        df_teams.loc[i, 'dif_neutral_minions_killed'] = int(sum(df_i_participant_blue.loc[:, 'neutralMinionsKilled']) - sum(df_i_participant_red.loc[:, 'neutralMinionsKilled']))
        df_teams.loc[i, 'dif_total_heal'] = int(sum(df_i_participant_blue.loc[:, 'totalHeal']) - sum(df_i_participant_red.loc[:, 'totalHeal']))
        df_teams.loc[i, 'dif_vision_score'] = int(sum(df_i_participant_blue.loc[:, 'visionScore']) - sum(df_i_participant_red.loc[:, 'visionScore']))
        df_teams.loc[i, 'win_blue'] = 1 if sum(df_i_participant_blue.loc[:, 'win'].astype(int)) > 0 else 0

    path_teams = "tables/Preprocesed/TeamsCleaned.csv"
    df_teams.to_csv(path_teams, index=False, encoding='utf-8', sep=";")
    teams_clean = df_teams



# Preprocesado de los datos
data_explicativas = teams_clean.drop('win_blue', axis=1)
data_predictiva = teams_clean['win_blue']


# aplica el mapeo a la columna 'region'
data_explicativas['region'] = data_explicativas['region'].replace(mapping_region)
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
        if abs(corr.iloc[i,j]) > 0.8:
            colname = corr.columns[i]
            high_corr_vars.add(colname)
            
# Eliminación de variables con correlación alta
variables_seleccionadas = [var for var in list(data_explicativas.columns) if var not in high_corr_vars]
print("Variables seleccionadas por correlación: ", variables_seleccionadas)


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
lasso = Lasso(alpha=0.1) # 0.00005
lasso.fit(X_scaled, y)
lasso_coef = lasso.coef_
nonzero_positions = [i for i, x in enumerate(list(lasso_coef)) if x != 0]
col_data = data_explicativas.columns
# imprime las posiciones obtenidas
list_var_lasso = list(col_data[nonzero_positions])
print("Variables seleccionadas por regresión Lasso: ", list(col_data[nonzero_positions]))


#####################################################################
## Selección de variables por eliminacón recursiva de características
#####################################################################

X = data_explicativas
y = data_predictiva

lr = LogisticRegression()
rfe = RFE(estimator=lr, n_features_to_select=5)
rfe.fit(X,y)
# imprimir los resultados
print("Número óptimo de características: %d" % rfe.n_features_)
print("Características seleccionadas:")
print(rfe.support_)
print("Ranking de características:")
print(rfe.ranking_)
lista_trues = rfe.support_
nonfalse_positions = [i for i, x in enumerate(list(lista_trues)) if x != False]
col_data = data_explicativas.columns
# imprime las posiciones obtenidas
print(nonfalse_positions)
print("Variables seleccionadas por por eliminacón recursiva de características: ", list(col_data[nonfalse_positions]))


#####################################################################
## Selección de variables por componentes principales (PCA)
#####################################################################

# Pasos:
# 1) Normalizar
# 2) Calcular autovectores y autovalores a partir de la matriz de covarianzas
# 3) Seleccionar autovectores correspondientes a las componentes principales
# 4) Nuevo dataset sobre el nuevo espacio vectorial



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

###
# Se usan 4 componentes
###

pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=4))
pca_pipe.fit(X_scaled)

# Se extrae el modelo entrenado del pipeline
model_pca = pca_pipe.named_steps['pca']
pca_df = pd.DataFrame(
    data    = model_pca.components_,
    columns = X.columns,
    index   = ['PC1', 'PC2', 'PC3', 'PC4']
)

print("Primer df que no se que es")
print(pca_df.head(5))

pca = PCA(n_components=4)
pca.fit_transform(X_scaled)
components = pca.transform(np.eye(X_scaled.shape[1]))
components = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3', 'PC4'])

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=data_explicativas.columns)
print("loadings")
print(loadings.head(5))
pca_results = {'components': components, 'loadings': loadings, 'explained_variance_ratio': pca.explained_variance_ratio_}

df_pca = pd.DataFrame(pca_results['components'].T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=X.columns)

df_pca['PC1_abs'] = np.abs(df_pca['PC1'])
df_pca['PC2_abs'] = np.abs(df_pca['PC2'])
df_pca['PC3_abs'] = np.abs(df_pca['PC3'])
df_pca['PC4_abs'] = np.abs(df_pca['PC4'])

# Ordenar las variables en función de su peso en cada componente
var_df = df_pca.sort_values(['PC1_abs', 'PC2_abs', 'PC3_abs', 'PC4_abs'], ascending=False)

# Seleccionar las primeras 3-4 variables más importantes para cada componente
var_sel_1 = var_df.index[:4]
var_sel_2 = var_df.index[4:8]
print("Ultimo df que no se que es")
print(df_pca.head(5))

#### COMPONENTES 1 y 2
# Creamos un scatter plot para visualizar las variables en las 4 componentes principales
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Configuramos los ejes y los títulos
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_xlabel("Componente Principal 1")
ax.set_ylabel("Componente Principal 2")
ax.set_title("Biplot para las 4 Componentes Principales")
for i, ax in enumerate(ax.flatten()):
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel(f"Componente Principal {i*2+1}")
    ax.set_ylabel(f"Componente Principal {i*2+2}")
    ax.set_title(f"Biplot para las Componentes {i*2+1} y {i*2+2}")
    
    texts = []
    for j, var in enumerate(X.columns):
        if var in var_sel_1 and i == 0:
            ax.arrow(0, 0, pca_results['components'][i*2, j], pca_results['components'][i*2+1, j], color='r', alpha=0.5,
                     head_width=0.05, head_length=0.1)
            text = ax.text(pca_results['components'][i*2, j] * 1.15, pca_results['components'][i*2+1, j] * 1.15, var,
                           color='b', ha='center', va='center', fontsize=8)
            texts.append(text)
            
        if var in var_sel_2 and i == 1:
            ax.arrow(0, 0, pca_results['components'][i*2, j], pca_results['components'][i*2+1, j], color='r', alpha=0.5,
                     head_width=0.05, head_length=0.1)
            text = ax.text(pca_results['components'][i*2, j] * 1.15, pca_results['components'][i*2+1, j] * 1.15, var,
                           color='b', ha='center', va='center', fontsize=8)
            texts.append(text)
        
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey', lw=0.5))
    
plt.tight_layout()
plt.savefig('images/preprocessed_images/PCA - Variables por cada componente principal.png')
plt.show()

"""#Generamos la matríz a partir de los pares autovalor-autovector
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

print('Matriz W:\n', matrix_w)

Y = X.dot(matrix_w)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('0', '1'),
                        ('magenta', 'cyan')):
        plt.scatter(Y[y==lab, 0],
                    Y[y==lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()  
"""
"""


# Las componentes principales estan ordenadas por la cantidad de varianza que explican en los datos originales. 
# De esta manera, PC1 explicará la mayor cantidad de varianza, PC2 la segunda mayor cantidad y PC3 la tercera. 
# El pca_df indica las coordenadas de la observación original, dentro del espacio de tres dimensiones de las componentes

# Influencia de la varianza en cada una de las componentes:
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
components = model_pca.components_
plt.imshow(components.T, cmap='viridis', aspect='auto')
plt.yticks(range(len(X.columns)), X.columns)
plt.xticks(range(len(X.columns)), np.arange(model_pca.n_components_) + 1)
plt.grid(False)
plt.colorbar()
plt.title('Influencia de la varianza en cada componente principal')
plt.savefig('images/preprocessed_images/Influencia de la varianza en cada componente principal.png')
plt.close()

# Una vez calculadas las componentes principales, se puede conocer la varianza explicada por cada una de ellas, la proporción respecto al total y la proporción de varianza acumulada.

# Porcentaje de varianza explicada por cada componente
# ==============================================================================

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.bar(
    x      = np.arange(model_pca.n_components_) + 1,
    height = model_pca.explained_variance_ratio_
)

for x, y in zip(np.arange(len(X.columns)) + 1, X.columns.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

ax.set_xticks(np.arange(model_pca.n_components_) + 1)
ax.set_ylim(0, 1.1)
ax.set_title('Porcentaje de varianza explicada por cada componente')
ax.set_xlabel('Componente principal')
ax.set_ylabel('% de varianza explicada')
ax.savefig('images/preprocessed_images/Porcentaje de varianza explicada por cada componente.png')
ax.close()


# Porcentaje de varianza explicada acumulada
prop_varianza_acum = model_pca.explained_variance_ratio_.cumsum()
print(prop_varianza_acum)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.plot(
    np.arange(len(X.columns)) + 1,
    prop_varianza_acum,
    marker = 'o'
)

for x, y in zip(np.arange(len(X.columns)) + 1, prop_varianza_acum):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )
    
ax.set_ylim(0, 1.1)
ax.set_xticks(np.arange(model_pca.n_components_) + 1)
ax.set_title('Porcentaje de varianza explicada acumulada')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza acumulada')
ax.savefig('images/preprocessed_images/orcentaje de varianza explicada acumulada.png')
ax.close()
"""
#############################
#### CREACION DEL MODELO ####
#############################

variables_explicativas = ['dif_kills', 'dif_inhibitor', 'dif_tower', 'dif_dmg_dealt_to_objectives', 'dif_gold_earned', 'dif_magic_dmg_dealt', 'dif_physical_dmg_dealt', 'dif_true_dmg_dealt', 'dif_gold_spent']

X = data_explicativas[variables_explicativas]
y = data_predictiva
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


#####################
# Regresión Logística
#####################


model = LogisticRegression()
model.fit(X_train, y_train)

# Prediccióm
y_pred = model.predict(X_test)
print("Acurracy/Precisión: con Regresión Logística", accuracy_score(y_test, y_pred))
print('Exhaustividad: con Regresión Logística', recall_score(y_test, y_pred))
print('Puntuación F1: con Regresión Logística', f1_score(y_test, y_pred))

# Variables más significativas
rfe = RFE(model, n_features_to_select=3)
rfe.fit(X_train, y_train)
print('Variables seleccionadas:', list(X.columns[rfe.support_]))



#####################
# Random Forest
#####################

## Para saber el número óptimo de estimadores, se utiliza la técnica "out-of-bag error".

# Para ello, iteraremos sobre diferentes valores de estimadores:

estimadores = [10, 50, 100, 500]
lista_errores = []
for val in estimadores:

    model_rf = RandomForestClassifier(n_estimators=val, random_state=1, oob_score=True)
    model_rf.fit(X_train, y_train)

    y_pred= model_rf.predict(X_test)

    # Precisión
    accuracy_rf = accuracy_score(y_test, y_pred)
    print("Acurracy/Precisión con Random Forest, con número de estimadores a", val,":", accuracy_score(y_test, y_pred))

    # Cálculo out-of-bag erro
    oob_error = 1-model_rf.oob_score
    lista_errores.append(oob_error)
print(lista_errores)
#########################
# Support Vector Machines
#########################

model_svm = SVC()
model_svm.fit(X_train, y_train)

y_pred = model_svm.predict(X_test)

# Calcular la precisión del modelo
accuracy_svm = accuracy_score(y_test, y_pred)
print("Acurracy/Precisión con SVM:", accuracy_score(y_test, y_pred))



#########################
# Redes Neuronales
#########################

model_rn = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# Entrenar el modelo

model_rn.fit(X_train, y_train)
y_pred = model_rn.predict(X_test)

# Calcular la precisión del modelo
accuracy_rn = accuracy_score(y_test, y_pred)
print("Acurracy/Precisión con Redes Neuronales:", accuracy_score(y_test, y_pred))
