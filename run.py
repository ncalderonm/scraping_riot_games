
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

timestamp = datetime.now()
hora_inicio_ejecucion = timestamp.strftime('%Y-%m-%d %H-%M-%S')
print("La hora de inicio de la ejecución es:", hora_inicio_ejecucion)


###########
### URL ###
###########


print("Entramos en la URL")
# URL
URL = 'https://developer.riotgames.com/'

# Fichero robots.txt:
def robots(url):
    if url[-1] == "/":
        r_link = url
    else:
        r_link = os.path.join(url, "/")
    rst = requests.get(r_link + 'configuration/url_info/robots.txt')
    return rst.text


# Tecnología usada:
import builtwith
tech = builtwith.builtwith(URL)

# Guardamos la información de las tecnologías de la página en un txt
with open('configuration/url_info/tech_used.txt', 'w') as file:
    # Escribir la información obtenida en el archivo
    file.write(str(tech))

# Propietario de la página:
whoisURL=whois.whois(URL)

# Guardamos la información del propietario de la página
with open('configuration/url_info/whoisURL.txt', 'w') as file:
    # Escribir la información obtenida en el archivo
    file.write(str(whoisURL))



###########################
#### 1) API EXTRACCIÓN ####
###########################

print("\n")
print("Inicio de la extracción de datos de  la API de Riot Games.")
print("\n")

# Ruta al archivo .py que deseas ejecutar
ruta_archivo_1 = os.path.abspath('1_api_extraction.py')

# Ejecutar el archivo .py utilizando el intérprete de Python
subprocess.run(['py', ruta_archivo_1], shell= True)

print("\n")
print("Fin de la extracción de datos de la api de Riot Games. ")
print("\n")


#########################
#### 2) PREPROCESADO ####
#########################

print("\n")
print("Inicio del proprocesado de los datos.")
print("\n")

# Ruta al archivo .py que deseas ejecutar
ruta_archivo_2 = '2_preprocess.py'

# Ejecutar el archivo .py utilizando el intérprete de Python
subprocess.run(['python', ruta_archivo_2])

print("\n")
print("Fin del preprocesado de los datos. ")
print("\n")


###############################
#### 3) Variable Selection ####
###############################

print("\n")
print("Inicio del proceso de selección de variables.")
print("\n")

# Ruta al archivo .py que deseas ejecutar
ruta_archivo_3 = '3_variable_selection.py'

# Ejecutar el archivo .py utilizando el intérprete de Python
subprocess.run(['python', ruta_archivo_3])

print("\n")
print("Fin del proceso de selección de variabless. ")
print("\n")


##################################
#### 4) Modelos de predicción ####
##################################

print("\n")
print("Inicio de la creación de modelos de predicción.")
print("\n")

# Ruta al archivo .py que deseas ejecutar
ruta_archivo_4 = '4_predictive_models.py'

# Ejecutar el archivo .py utilizando el intérprete de Python
subprocess.run(['python', ruta_archivo_4])

print("\n")
print("Fin de la creación de modelos de predicción. ")
print("\n")



####################################
#### 4) Interacción con PowerBi ####
####################################

print("\n")
print("Inicio de la creación de modelos de predicción.")
print("\n")

# Ruta al archivo .py que deseas ejecutar
ruta_archivo_4 = '4_predictive_models.py'

# Ejecutar el archivo .py utilizando el intérprete de Python
subprocess.run(['python', ruta_archivo_4])

print("\n")
print("Fin de la creación de modelos de predicción. ")
print("\n")
