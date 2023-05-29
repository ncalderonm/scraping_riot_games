
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
from ggplot import *

#################
### Carga de datos 
#################

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



## 1) Eliminación de nulls de los csv (si existe)

# verificar si hay valores nulos
# Existen tres partidas con todo a nulo, las eliminamos de la bbdd

games_to_delete = ['KR_5971954182', 'KR_6161284361', 'EUW1_5940733882']

# Eliminar filas con valores nulos
df_matches_participant = df_matches_participant[~df_matches_participant['matchId'].isin(games_to_delete)]

# Los servidores de Latino América "LA1" los incluimos en los de Norte América "NA1"
df_matches_info['platformId'] = df_matches_info['platformId'].replace('LA1', 'NA1')

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


path_X = "tables/Preprocesed/data_explicativas.csv"
data_explicativas.to_csv(path_X, index=False, encoding='utf-8', sep=";")

path_y = "tables/Preprocesed/data_predictiva.csv"
data_predictiva.to_csv(path_y, index=False, encoding='utf-8', sep=";")



data_explicativas_numeric = data_explicativas[['dif_kills','dif_barons','dif_dragons', 'dif_inhibitor','dif_herald','dif_tower', 'dif_dmg_dealt_to_buildings', 'dif_dmg_dealt_to_objectives', 'dif_dmg_dealt_to_turrets', 'dif_dmg_self_mitigated', 'dif_gold_earned', 'dif_gold_spent', 'dif_magic_dmg_dealt', 'dif_physical_dmg_dealt', 'dif_true_dmg_dealt', 'dif_total_minions_killed', 'dif_neutral_minions_killed', 'dif_total_heal', 'dif_vision_score']]

data_explicativas_categ = data_explicativas[['baron_first_blue','champion_first_blue','dragon_first_blue','inhibitor_first_blue', 'herald_first_blue', 'tower_first_blue', 'baron_first_red', 'champion_first_red', 'dragon_first_red', 'inhibitor_first_red', 'herald_first_red', 'tower_first_red']]

resumen_estadistico = data_explicativas_numeric.describe()

# Guardar el dataframe en un archivo CSV
resumen_estadistico.to_csv('tables/Preprocesed/resumen_estadistico.csv', index=True)


# Dividir el resumen estadístico en cinco partes aproximadamente iguales
quinto = len(resumen_estadistico.columns) // 5
parte1 = resumen_estadistico.iloc[:, :quinto]
parte2 = resumen_estadistico.iloc[:, quinto:2*quinto]
parte3 = resumen_estadistico.iloc[:, 2*quinto:3*quinto]
parte4 = resumen_estadistico.iloc[:, 3*quinto:4*quinto]
parte5 = resumen_estadistico.iloc[:, 4*quinto:]

# Agregar la columna de nombres de las estadísticas
parte1.insert(0, 'Estadística', parte1.index)
parte2.insert(0, 'Estadística', parte2.index)
parte3.insert(0, 'Estadística', parte3.index)
parte4.insert(0, 'Estadística', parte4.index)
parte5.insert(0, 'Estadística', parte5.index)

# Redondear los valores a 3 decimales
parte1 = parte1.round(3)
parte2 = parte2.round(3)
parte3 = parte3.round(3)
parte4 = parte4.round(3)
parte5 = parte5.round(3)


max_width_5 = max(len(str(name)) for name in parte5.columns)
colWidths_5 = [max_width_5] * len(parte5.columns)

# Configurar el ancho de las columnas
col_widths_1 = [0.25] + [0.25] * (len(parte1.columns))
col_widths_2 = [0.25] + [0.25] * (len(parte2.columns))
col_widths_3 = [0.25] + [0.25] * (len(parte3.columns))
col_widths_4 = [0.25] + [0.25] * (len(parte4.columns))
col_widths_5 = [0.25] + [0.25] * (len(parte5.columns))


# Convertir la primera parte en una imagen
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.axis('tight')
ax1.axis('off')
tabla1 = ax1.table(cellText=parte1.values,
                   colLabels=parte1.columns,
                   cellLoc='center',
                   loc='center',
                   colWidths=col_widths_1)
tabla1.auto_set_font_size(False)
tabla1.set_fontsize(10)
tabla1.scale(1.2, 1.2)
fig1.tight_layout()
plt.savefig('images/Analisis_descriptivo/resumen_estadistico_1.png', bbox_inches='tight')
plt.close()

# Convertir la segunda parte en una imagen
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.axis('tight')
ax2.axis('off')
tabla2 = ax2.table(cellText=parte2.values,
                   colLabels=parte2.columns,
                   cellLoc='center',
                   loc='center',
                   colWidths=col_widths_2)
tabla2.auto_set_font_size(False)
tabla2.set_fontsize(10)
tabla2.scale(1.2, 1.2)
fig2.tight_layout()
plt.savefig('images/Analisis_descriptivo/resumen_estadistico_2.png', bbox_inches='tight')
plt.close()

# Convertir la tercera parte en una imagen
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.axis('tight')
ax3.axis('off')
tabla3 = ax3.table(cellText=parte3.values,
                   colLabels=parte3.columns,
                   cellLoc='center',
                   loc='center',
                   colWidths=col_widths_3)
tabla3.auto_set_font_size(False)
tabla3.set_fontsize(10)
tabla3.scale(1.2, 1.2)
fig3.tight_layout()
plt.savefig('images/Analisis_descriptivo/resumen_estadistico_3.png', bbox_inches='tight')
plt.close()

# Convertir la cuarta parte en una imagen
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.axis('tight')
ax4.axis('off')
tabla4 = ax4.table(cellText=parte4.values,
                   colLabels=parte4.columns,
                   cellLoc='center',
                   loc='center',
                   colWidths=col_widths_4)
tabla4.auto_set_font_size(False)
tabla4.set_fontsize(10)
tabla4.scale(1.2, 1.2)
fig4.tight_layout()
plt.savefig('images/Analisis_descriptivo/resumen_estadistico_4.png', bbox_inches='tight')
plt.close()


# Convertir la quinta parte en una imagen
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.axis('tight')
ax5.axis('off')
tabla5 = ax5.table(cellText=parte5.values,
                   colLabels=parte5.columns,
                   cellLoc='center',
                   loc='center',
                   colWidths=col_widths_5)
tabla5.auto_set_font_size(False)
tabla5.set_fontsize(10)
tabla5.scale(1.2, 1.2)
fig5.tight_layout()
plt.savefig('images/Analisis_descriptivo/resumen_estadistico_5.png', bbox_inches='tight')
plt.close()




"""
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

#Analizamos cada una de las variables
## DIF_KILLS
axes[0,0].hist(data_explicativas_numeric['dif_kills'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0,0].set_xlabel('Valores')
axes[0,0].set_ylabel('Frecuencia')
axes[0,0].set_title('Histograma de Kills')

## DIF_BARONS
axes[0,1].hist(data_explicativas_numeric['dif_barons'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
axes[0,1].set_xlabel('Valores')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].set_title('Histograma de Barons')

## DIF_INHIBITORS
axes[1,0].hist(data_explicativas_numeric['dif_inhibitor'], bins=14, color='skyblue', edgecolor='black', alpha=0.7)
axes[1,0].set_xlabel('Valores')
axes[1,0].set_ylabel('Frecuencia')
axes[1,0].set_title('Histograma de Inhibitors')

## dif_dragons
axes[1,1].hist(data_explicativas_numeric['dif_dragons'], bins=14, color='skyblue', edgecolor='black', alpha=0.7)
axes[1,1].set_xlabel('Valores')
axes[1,1].set_ylabel('Frecuencia')
axes[1,1].set_title('Histograma de Inhibitors')

# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_1a4.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()

## Imagenes de la 5 a la 8
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

## dif_herald
axes[0,0].hist(data_explicativas_numeric['dif_herald'], bins=5, color='skyblue', edgecolor='black', alpha=0.7)
axes[0,0].set_xlabel('Valores')
axes[0,0].set_ylabel('Frecuencia')
axes[0,0].set_title('Histograma de Herald')

## dif_tower
axes[0,1].hist(data_explicativas_numeric['dif_tower'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
axes[0,1].set_xlabel('Valores')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].set_title('Histograma de Towers')

## dif_dmg_dealt_to_buildings
axes[1,0].hist(data_explicativas_numeric['dif_dmg_dealt_to_buildings'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
axes[1,0].set_xlabel('Valores')
axes[1,0].set_ylabel('Frecuencia')
axes[1,0].set_title('Histograma de DMG dealt to Buildings')

## dif_dmg_dealt_to_objectives
axes[1,1].hist(data_explicativas_numeric['dif_dmg_dealt_to_objectives'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
axes[1,1].set_xlabel('Valores')
axes[1,1].set_ylabel('Frecuencia')
axes[1,1].set_title('Histograma de DMG dealt to Objectives')

# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_5a8.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()



## dif_dmg_dealt_to_turrets
axes[0,0].hist(data_explicativas_numeric['dif_dmg_dealt_to_turrets'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
axes[0,0].set_xlabel('Valores')
axes[0,0].set_ylabel('Frecuencia')
axes[0,0].set_title('Histograma de DMG dealt to Turrets')

"""
"""# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_dif_dmg_dealt_to_turrets.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()""""""


## dif_dmg_self_mitigated
axes[0,1].hist(data_explicativas_numeric['dif_dmg_self_mitigated'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
axes[0,1].set_xlabel('Valores')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].set_title('Histograma de DMG self mitigated')

""""""# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_dif_dmg_self_mitigated.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()""""""


## dif_gold_earned
plt.hist(data_explicativas_numeric['dif_gold_earned'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de Gold earned')

# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_dif_gold_earned.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()

## dif_gold_spent
plt.hist(data_explicativas_numeric['dif_gold_spent'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de Gold spent')

# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_dif_gold_spent.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()

## dif_magic_dmg_dealt
plt.hist(data_explicativas_numeric['dif_magic_dmg_dealt'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de Magic DMG dealt')

# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_dif_magic_dmg_dealt.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()

""""""p1 = ggplot(data_explicativas_numeric, aes(x='dif_magic_dmg_dealt')) + \
    geom_histogram(binwidth=0.2, color='white', fill='skyblue') + \
    xlab('Valor') + \
    ylab('Frecuencia') + \
    ggtitle('Histograma')

# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
# Guardar la imagen en una ruta específica
ruta_guardado = 'images\Analisis_descriptivo\AAAAAA.png'
p1.save(filename=ruta_guardado)""""""


## dif_physical_dmg_dealt
plt.hist(data_explicativas_numeric['dif_physical_dmg_dealt'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de Physical DMG dealt')

# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_dif_physical_dmg_dealt.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()


## dif_true_dmg_dealt
plt.hist(data_explicativas_numeric['dif_true_dmg_dealt'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de True DMG dealt')

# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_dif_true_dmg_dealt.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()

"""
"""## dif_total_minions_killed
plt.hist(data_explicativas_numeric['dif_total_minions_killed'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de Minions Killed')

# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_dif_total_minions_killed.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()


## dif_neutral_minions_killed
plt.hist(data_explicativas_numeric['dif_neutral_minions_killed'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de Neutral Minions Killed')

# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_dif_neutral_minions_killed.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()"""
"""

## dif_total_heal
plt.hist(data_explicativas_numeric['dif_total_heal'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de Total Heal')

# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_dif_total_heal.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()


## dif_vision_score
plt.hist(data_explicativas_numeric['dif_vision_score'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de Vision Score')

# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_dif_vision_score.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()
"""




#BLOQUE HISTOGRAMAS DE 4 EN 4

#Diferencia Kills, Heal y Vision Score

fig, axes = plt.subplots(1,3, figsize=(12, 4))

## dif_kills
axes[0].hist(data_explicativas_numeric['dif_kills'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Valores')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Histograma de Kills')

## dif_total_heal
axes[1].hist(data_explicativas_numeric['dif_total_heal'], bins=23, color='green', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Valores')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Histograma de Total Heal')

## dif_vision_score
axes[2].hist(data_explicativas_numeric['dif_vision_score'], bins=23, color='red', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Valores')
axes[2].set_ylabel('Frecuencia')
axes[2].set_title('Histograma de Vision Score')


# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_1-3.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()


#Diferencia DMG Dealt objetivos

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))


## dif_dmg_dealt_to_buildings
axes[0].hist(data_explicativas_numeric['dif_dmg_dealt_to_buildings'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Valores')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Histograma de DMG dealt to Buildings')

## dif_dmg_dealt_to_objectives
axes[1].hist(data_explicativas_numeric['dif_dmg_dealt_to_objectives'], bins=23, color='green', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Valores')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Histograma de DMG dealt to Objectives')

## dif_dmg_dealt_to_turrets
axes[2].hist(data_explicativas_numeric['dif_dmg_dealt_to_turrets'], bins=23, color='red', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Valores')
axes[2].set_ylabel('Frecuencia')
axes[2].set_title('Histograma de DMG dealt to Turrets')


# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_4-6.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()




#Diferencia diferentes tipos de Daño realizado
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

## dif_dmg_self_mitigated
axes[0,0].hist(data_explicativas_numeric['dif_dmg_self_mitigated'], bins=23, color='red', edgecolor='black', alpha=0.7)
axes[0,0].set_xlabel('Valores')
axes[0,0].set_ylabel('Frecuencia')
axes[0,0].set_title('Histograma de DMG Mitigated')


## dif_magic_dmg_dealt
axes[0,1].hist(data_explicativas_numeric['dif_magic_dmg_dealt'], bins=23, color='green', edgecolor='black', alpha=0.7)
axes[0,1].set_xlabel('Valores')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].set_title('Histograma de Magic DMG Dealt')


## dif_physical_dmg_dealt
axes[1,0].hist(data_explicativas_numeric['dif_physical_dmg_dealt'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
axes[1,0].set_xlabel('Valores')
axes[1,0].set_ylabel('Frecuencia')
axes[1,0].set_title('Histograma de Physical DMG Dealt')


## dif_true_dmg_dealt
axes[1,1].hist(data_explicativas_numeric['dif_true_dmg_dealt'], bins=23, color='blue', edgecolor='black', alpha=0.7)
axes[1,1].set_xlabel('Valores')
axes[1,1].set_ylabel('Frecuencia')
axes[1,1].set_title('Histograma de True DMG Dealt')


# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_7-10.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()



#Diferencia diferentes tipos de Daño realizado
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

#Diferencia Oro conseguido/gastado y diferencia minions totales/minions neutrales 

## dif_gold_earned
axes[0,0].hist(data_explicativas_numeric['dif_gold_earned'], bins=23, color='red', edgecolor='black', alpha=0.7)
axes[0,0].set_xlabel('Valores')
axes[0,0].set_ylabel('Frecuencia')
axes[0,0].set_title('Histograma de Gold earned')


## dif_gold_spent
axes[0,1].hist(data_explicativas_numeric['dif_gold_spent'], bins=23, color='green', edgecolor='black', alpha=0.7)
axes[0,1].set_xlabel('Valores')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].set_title('Histograma de Gold spent')


## dif_total_minions_killed
axes[1,0].hist(data_explicativas_numeric['dif_total_minions_killed'], bins=23, color='skyblue', edgecolor='black', alpha=0.7)
axes[1,0].set_xlabel('Valores')
axes[1,0].set_ylabel('Frecuencia')
axes[1,0].set_title('Histograma de Minions Killed')


## dif_neutral_minions_killed
axes[1,1].hist(data_explicativas_numeric['dif_neutral_minions_killed'], bins=23, color='blue', edgecolor='black', alpha=0.7)
axes[1,1].set_xlabel('Valores')
axes[1,1].set_ylabel('Frecuencia')
axes[1,1].set_title('Histograma de Neutral Minions Killed')


# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_11-14.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()



#Diferencia Principal Objetivos

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16,12))


## dif_barons
axes[0,0].hist(data_explicativas_numeric['dif_barons'], bins=8, color='skyblue', edgecolor='black', alpha=0.7)
axes[0,0].set_xlabel('Valores')
axes[0,0].set_ylabel('Frecuencia')
axes[0,0].set_title('Histograma de Barons')

## dif_dragons
axes[0,1].hist(data_explicativas_numeric['dif_dragons'], bins=12, color='green', edgecolor='black', alpha=0.7)
axes[0,1].set_xlabel('Valores')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].set_title('Histograma de Dragons')

## dif_inhibitor
axes[0,2].hist(data_explicativas_numeric['dif_inhibitor'], bins=10, color='red', edgecolor='black', alpha=0.7)
axes[0,2].set_xlabel('Valores')
axes[0,2].set_ylabel('Frecuencia')
axes[0,2].set_title('Histograma de DMG dealt to Inhibitors')

## dif_herald
axes[1,0].hist(data_explicativas_numeric['dif_herald'], bins=4, color='pink', edgecolor='black', alpha=0.7)
axes[1,0].set_xlabel('Valores')
axes[1,0].set_ylabel('Frecuencia')
axes[1,0].set_title('Histograma de Herald')

## dif_tower
axes[1,1].hist(data_explicativas_numeric['dif_tower'], bins=23, color='orange', edgecolor='black', alpha=0.7)
axes[1,1].set_xlabel('Valores')
axes[1,1].set_ylabel('Frecuencia')
axes[1,1].set_title('Histograma de Towers')


# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\histograma_15-19.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()

###############
## BOXPLOT ####
###############

#Diferencia Kills, Heal y Vision Score

fig, axes = plt.subplots(1,3, figsize=(12, 4))

## dif_kills
axes[0].boxplot(data_explicativas_numeric['dif_kills'])
axes[0].set_xlabel('Valores')
axes[0].set_ylabel('Kills')
axes[0].set_title('Boxplot de Kills')

## dif_total_heal
axes[1].boxplot(data_explicativas_numeric['dif_total_heal'])
axes[1].set_xlabel('Valores')
axes[1].set_ylabel('Total Heal')
axes[1].set_title('Boxplot de Total Heal')

## dif_vision_score
axes[2].boxplot(data_explicativas_numeric['dif_vision_score'])
axes[2].set_xlabel('Valores')
axes[2].set_ylabel('Vision Score')
axes[2].set_title('Boxplot de Vision Score')


# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\BoxPlot_1-3.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()



#Diferencia DMG Dealt objetivos

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))


## dif_dmg_dealt_to_buildings
axes[0].boxplot(data_explicativas_numeric['dif_dmg_dealt_to_buildings'])
axes[0].set_xlabel('Valores')
axes[0].set_ylabel('DMG dealt to Buildings')
axes[0].set_title('Boxplot de DMG dealt to Buildings')

## dif_dmg_dealt_to_objectives
axes[1].boxplot(data_explicativas_numeric['dif_dmg_dealt_to_objectives'])
axes[1].set_xlabel('Valores')
axes[1].set_ylabel('DMG dealt to Objectives')
axes[1].set_title('Boxplot de DMG dealt to Objectives')

## dif_dmg_dealt_to_turrets
axes[2].boxplot(data_explicativas_numeric['dif_dmg_dealt_to_turrets'])
axes[2].set_xlabel('Valores')
axes[2].set_ylabel('DMG dealt to Turrets')
axes[2].set_title('Boxplot de DMG dealt to Turrets')


# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\Boxplot_4-6.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()




#Diferencia diferentes tipos de Daño realizado
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

## dif_dmg_self_mitigated
axes[0,0].boxplot(data_explicativas_numeric['dif_dmg_self_mitigated'])
axes[0,0].set_xlabel('Valores')
axes[0,0].set_ylabel('DMG self Mitigated')
axes[0,0].set_title('Boxplot de DMG Mitigated')


## dif_magic_dmg_dealt
axes[0,1].boxplot(data_explicativas_numeric['dif_magic_dmg_dealt'])
axes[0,1].set_xlabel('Valores')
axes[0,1].set_ylabel('Magic DMG dealt')
axes[0,1].set_title('Boxplot de Magic DMG Dealt')


## dif_physical_dmg_dealt
axes[1,0].boxplot(data_explicativas_numeric['dif_physical_dmg_dealt'])
axes[1,0].set_xlabel('Valores')
axes[1,0].set_ylabel('Physical DMG dealt')
axes[1,0].set_title('Boxplot de Physical DMG Dealt')


## dif_true_dmg_dealt
axes[1,1].boxplot(data_explicativas_numeric['dif_true_dmg_dealt'])
axes[1,1].set_xlabel('Valores')
axes[1,1].set_ylabel('True DMG dealt')
axes[1,1].set_title('Boxplot de True DMG Dealt')


# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\Boxplot_7-10.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()



#Diferencia diferentes tipos de Daño realizado
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

#Diferencia Oro conseguido/gastado y diferencia minions totales/minions neutrales 

## dif_gold_earned
axes[0,0].boxplot(data_explicativas_numeric['dif_gold_earned'])
axes[0,0].set_xlabel('Valores')
axes[0,0].set_ylabel('Gold earned')
axes[0,0].set_title('Boxplot de Gold earned')


## dif_gold_spent
axes[0,1].boxplot(data_explicativas_numeric['dif_gold_spent'])
axes[0,1].set_xlabel('Gold spent')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].set_title('Boxplot de Gold spent')


## dif_total_minions_killed
axes[1,0].boxplot(data_explicativas_numeric['dif_total_minions_killed'])
axes[1,0].set_xlabel('Valores')
axes[1,0].set_ylabel('Total Minions Killed')
axes[1,0].set_title('Boxplot de Minions Killed')


## dif_neutral_minions_killed
axes[1,1].boxplot(data_explicativas_numeric['dif_neutral_minions_killed'])
axes[1,1].set_xlabel('Valores')
axes[1,1].set_ylabel('Neutral Minions Killed')
axes[1,1].set_title('Boxplot de Neutral Minions Killed')


# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\Boxplot_11-14.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()



#Diferencia Principal Objetivos

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16,12))


## dif_barons
axes[0,0].boxplot(data_explicativas_numeric['dif_barons'])
axes[0,0].set_xlabel('Valores')
axes[0,0].set_ylabel('Barons')
axes[0,0].set_title('Boxplot de Barons')

## dif_dragons
axes[0,1].boxplot(data_explicativas_numeric['dif_dragons'])
axes[0,1].set_xlabel('Valores')
axes[0,1].set_ylabel('Dragons')
axes[0,1].set_title('Boxplot de Dragons')

## dif_inhibitor
axes[0,2].boxplot(data_explicativas_numeric['dif_inhibitor'])
axes[0,2].set_xlabel('Valores')
axes[0,2].set_ylabel('Inhibitor')
axes[0,2].set_title('Boxplot de Inhibitors')

## dif_herald
axes[1,0].boxplot(data_explicativas_numeric['dif_herald'])
axes[1,0].set_xlabel('Valores')
axes[1,0].set_ylabel('Herald')
axes[1,0].set_title('Boxplot de Herald')

## dif_tower
axes[1,1].boxplot(data_explicativas_numeric['dif_tower'])
axes[1,1].set_xlabel('Valores')
axes[1,1].set_ylabel('Towers')
axes[1,1].set_title('Boxplot de Towers')


# Guardar la imagen en un archivo (por ejemplo, en formato PNG)
ruta = 'images\Analisis_descriptivo\Boxplot_15-19.png'
plt.tight_layout()
plt.savefig(ruta)
plt.close()



##########################################
#### Análisis descriptivo Categoricas ####
##########################################

print(data_explicativas_categ.columns)
frecuencia = data_explicativas_categ.value_counts()


# Crear una visualización de la tabla
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')  # Ocultar los ejes

tabla = ax.table(cellText=frecuencia.values,
                 colLabels=data_explicativas_categ.columns,
                 cellLoc='center',
                 loc='center')

tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1.2, 1.2)

# Guardar la imagen
plt.savefig('images\Analisis_descriptivo\Tabla_Frecuencias.png', bbox_inches='tight')
plt.close()


# Obtener las columnas del DataFrame
columnas = data_explicativas_categ.columns

# Crear los gráficos de barras para cada variable categórica
for columna in columnas:
    # Calcular la frecuencia de cada categoría
    frecuencia = data_explicativas_categ[columna].value_counts()
    
    # Crear el gráfico de barras
    frecuencia.plot.bar(legend=False)
    
    # Configurar los ejes y título
    plt.xlabel(columna)
    plt.ylabel('Frecuencia')
    plt.title('Gráfico de barras para {}'.format(columna))
    
    # Mostrar el gráfico
    plt.savefig(f'images\Analisis_descriptivo\Bar_plot_{columna}.png')
    plt.close()

###### TABLA CONTINGENCIA

"""# Calcular la tabla de contingencia
tabla_contingencia = pd.crosstab([data_explicativas_categ['baron_first_blue'], data_explicativas_categ['champion_first_blue'], data_explicativas_categ['dragon_first_blue'], data_explicativas_categ['inhibitor_first_blue'],data_explicativas_categ['herald_first_blue'],data_explicativas_categ['tower_first_blue'],data_explicativas_categ['baron_first_red'],data_explicativas_categ['champion_first_red'],data_explicativas_categ['dragon_first_red'], data_explicativas_categ['inhibitor_first_red'], data_explicativas_categ['herald_first_red']], data_explicativas_categ['tower_first_red'])

# Mostrar la tabla de contingencia
print(tabla_contingencia)"""