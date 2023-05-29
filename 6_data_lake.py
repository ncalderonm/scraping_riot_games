
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
from configuration.PostgreSQL.conection_postgre import conn, engine
from datetime import datetime
from sqlalchemy import create_engine

###################
### POSTGRE SQL ###
###################


# Crear un cursor
cur = conn.cursor()

variable = "a"

if variable != "a":
    ##########
    ## TABLA EQUIPOS
    ##########

    path_teams= "tables/Preprocesed/TeamsCleaned.csv"

    # Si el archivo existe, cargar el DataFrame desde el archivo
    with open(path_teams, mode='rb') as archivo_X:
        print("Accediendo a la aruta del df de metadata")
        # Lee el contenido del archivo CSV utilizando la biblioteca pandas
        data_teams = pd.read_csv(archivo_X, sep=";", encoding='utf-8')

    table_name_teams = 'teamscleaned'

    # Insertar los datos del DataFrame en la tabla
    data_teams.to_sql(table_name_teams, engine, if_exists='append', index=False)
    # Confirmar los cambios realizados en la base de datos
    conn.commit()



    ##########
    ## TABLA Players
    ##########

    path_players= "tables/csv/players.csv"

    # Si el archivo existe, cargar el DataFrame desde el archivo
    with open(path_players, mode='rb') as archivo_X:
        print("Accediendo a la aruta del df de metadata")
        # Lee el contenido del archivo CSV utilizando la biblioteca pandas
        data_players = pd.read_csv(archivo_X, sep=";", encoding='utf-8')

    table_name_players = 'players'

    # Insertar los datos del DataFrame en la tabla
    data_players.to_sql(table_name_players, engine, if_exists='append', index=False)
    # Confirmar los cambios realizados en la base de datos
    conn.commit()


    ##########
    ## TABLA match_info
    ##########

    path_match_info= "tables/csv/match_info.csv"

    # Si el archivo existe, cargar el DataFrame desde el archivo
    with open(path_match_info, mode='rb') as archivo_X:
        print("Accediendo a la aruta del df de metadata")
        # Lee el contenido del archivo CSV utilizando la biblioteca pandas
        data_match_info = pd.read_csv(archivo_X, sep=";", encoding='utf-8')

    table_name_match_info = 'matchinfo'

    # Insertar los datos del DataFrame en la tabla
    data_match_info.to_sql(table_name_match_info, engine, if_exists='append', index=False)
    # Confirmar los cambios realizados en la base de datos
    conn.commit()


    ##########
    ## TABLA match_md
    ##########

    path_match_md= "tables/csv/match_md.csv"

    # Si el archivo existe, cargar el DataFrame desde el archivo
    with open(path_match_md, mode='rb') as archivo_X:
        print("Accediendo a la aruta del df de metadata")
        # Lee el contenido del archivo CSV utilizando la biblioteca pandas
        data_match_md = pd.read_csv(archivo_X, sep=";", encoding='utf-8')

    table_name_match_md = 'matchmd'

    # Insertar los datos del DataFrame en la tabla
    data_match_md.to_sql(table_name_match_md, engine, if_exists='append', index=False)
    # Confirmar los cambios realizados en la base de datos
    conn.commit()


    ##########
    ## TABLA match_info_team
    ##########

    path_match_info_team= "tables/csv/match_info_team.csv"

    # Si el archivo existe, cargar el DataFrame desde el archivo
    with open(path_match_info_team, mode='rb') as archivo_X:
        print("Accediendo a la aruta del df de metadata")
        # Lee el contenido del archivo CSV utilizando la biblioteca pandas
        data_match_info_team = pd.read_csv(archivo_X, sep=";", encoding='utf-8')

    table_name_match_info_team = 'matchinfoteam'

    # Insertar los datos del DataFrame en la tabla
    data_match_info_team.to_sql(table_name_match_info_team, engine, if_exists='append', index=False)
    # Confirmar los cambios realizados en la base de datos
    conn.commit()

    ##########
    ## TABLA data_explicativa

    path_data_explicativa= "tables/Preprocesed/data_explicativas.csv"

    # Si el archivo existe, cargar el DataFrame desde el archivo
    with open(path_data_explicativa, mode='rb') as archivo_X:
        print("Accediendo a la aruta del df de metadata")
        # Lee el contenido del archivo CSV utilizando la biblioteca pandas
        data_data_explicativa = pd.read_csv(archivo_X, sep=";", encoding='utf-8')

    table_name_data_explicativa = 'data_explicativas'

    # Insertar los datos del DataFrame en la tabla
    data_data_explicativa.to_sql(table_name_data_explicativa, engine, if_exists='append', index=False)
    # Confirmar los cambios realizados en la base de datos
    conn.commit()



    ##########
    ## TABLA data_explicativa

    path_data_predictiva= "tables/Preprocesed/data_predictiva.csv"

    # Si el archivo existe, cargar el DataFrame desde el archivo
    with open(path_data_predictiva, mode='rb') as archivo_X:
        print("Accediendo a la aruta del df de metadata")
        # Lee el contenido del archivo CSV utilizando la biblioteca pandas
        data_data_predictiva = pd.read_csv(archivo_X, sep=";", encoding='utf-8')

    table_name_data_predictiva = 'data_predictiva'

    # Insertar los datos del DataFrame en la tabla
    data_data_predictiva.to_sql(table_name_data_predictiva, engine, if_exists='append', index=False)
    # Confirmar los cambios realizados en la base de datos
    conn.commit()
else:

    ##########
    ## TABLA PARTICIPANTES
    ##########

    path_participants= "tables/csv/match_participant.csv"

    # Si el archivo existe, cargar el DataFrame desde el archivo
    with open(path_participants, mode='rb') as archivo_X:
        print("Accediendo a la aruta del df de metadata")
        # Lee el contenido del archivo CSV utilizando la biblioteca pandas
        data_participants = pd.read_csv(archivo_X, sep=";", encoding='utf-8')

    table_name_participants = 'participants'

    # Insertar los datos del DataFrame en la tabla
    data_participants.to_sql(table_name_participants, engine, if_exists='append', index=False)
    # Confirmar los cambios realizados en la base de datos
    conn.commit()


# Cerrar la conexión
cur.close()
conn.close()
