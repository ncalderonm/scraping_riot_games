
###############
### Imports ###
###############

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
from configuration.PostgreSQL.conection_postgre import conn




###########
### URL ###
###########


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



###################
### CREDENTIALS ###
###################


# Obtenemos la password encriptada
with importlib.resources.open_text('configuration.key', 'encrypted_password') as f:
    encrypted_password = f.read()

encrypted_password = encrypted_password.encode()
decrypt_password = fernet.decrypt(encrypted_password).decode('utf-8')

# Credentials:
user = username
password = decrypt_password
summoner_name = sys.argv[1]

# Obtención password
with open('configuration/key/api.txt', 'r') as f:
    api = f.readline()

api_key = api
# Especifica el nombre del servidor y el nombre de invocador del jugador que deseas buscar
region = 'euw1'
summoner_name = 'Adelphos1313'



######################
### API RIOT GAMES ###
######################


# Obtiene información sobre las partidas en curso destacadas
url = f'https://{region}.api.riotgames.com/lol/spectator/v4/featured-games?api_key={api_key}'

# Realiza 100 solicitudes cada 2 minutos
for i in range(1):
    print("Entra al for")
    response = requests.get(url)
    ("Hace el response")   
    print(response)
    # Si la respuesta es exitosa
    if response.status_code == 200:
        featured_games = response.json()['gameList']
        print(featured_games)

        # Imprime información sobre las partidas destacadas
        print(f'Partidas destacadas ({i+1}/100):')
        for game in featured_games:
            print(f'- {game["gameMode"]} ({game["gameType"]})')
            for participant in game['participants']:
                print(f'  - {participant["summonerName"]} ({participant["championId"]})')
    else:
        # maneja el error
        print(f"Error: {response.status_code}")
        
    # espera 1.2 segundos entre cada solicitud
    time.sleep(1.2)
    
    # cada 20 solicitudes, espera 2 segundos
    if i % 20 == 0 and i > 0:
        print(f"Esperando 2 segundos para continuar... ({i+1}/100)")
        time.sleep(2)

with open("tables/last_matches_real_time.json", "w") as f:
    json.dump(featured_games, f)

# Cargar el archivo JSON
with open('tables/last_matches_real_time.json') as f:
    data = json.load(f)

# Convertir el JSON a un DataFrame de pandas
df = pd.json_normalize(data)

# Guardar el DataFrame en un archivo CSV
df.to_csv('tables/csv/games.csv', index=False)



###################
### POSTGRE SQL ###
###################


# Crear un cursor
cur = conn.cursor()


# Cerrar la conexión
cur.close()
conn.close()