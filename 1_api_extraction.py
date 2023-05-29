
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

print(sys.path)

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
#summoner_name = sys.argv[1]

# Obtención password
with open('configuration/key/api.txt', 'r') as f:
    api = f.readline()

api_key = api
# Especifica el nombre del servidor y el nombre de invocador del jugador que deseas buscar
region = 'EUW1'
summoner_name = 'Adelphos1313'



################
### SCRAPING ###
################

URL_SCRAP="https://developer.riotgames.com/apis"

# Abrir el navegador con la URL
#s = Service(ChromeDriverManager().install())
#driver = webdriver.Chrome(service=s)

#driver.get(URL_SCRAP)
#sleep(1)
#driver.maximize_window()
#sleep(2)


# Obtener el nombre de jugadores challenger
queue = "RANKED_SOLO_5x5"
page_number = range(1, 3)
region = ["KR", "EUW1", "NA1"]
division = ["I", "II", "III", "IV"]
tier = ["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON"]
solicitudes = 0

##Creación de todos los DataFrames utilizados:

#url = f'https://{region}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/{queue}?api_key={api_key}'
df_players = pd.DataFrame({'summonerId': [], 'summonerName': [], 'region':[], 'leaguePoints': [], 'rank': [], 'wins': [], 'losses': [], 'veteran': [], 'inactive': [], 'freshBlood': [], 'hotStreak': [], 'tier': [], 'leagueId': [], 'queue': []})

df_matches_md = pd.DataFrame({'dataVersion': [], 'matchId': [], 'puuid': []})

df_matches_info = pd.DataFrame({'matchId': [], 'gameDuration': [], 'gameId': [], 'platformId': []})

df_matches_info_teams= pd.DataFrame({'matchId': [], 'teamId': [],'bans_championId_1': [],'bans_championId_2': [], 'bans_championId_3': [],'bans_championId_4': [], 'bans_championId_5': [], 'baron_first': [], 'baron_kills': [], 'champion_first': [], 'champion_kills': [], 'dragon_first': [], 'dragon_kills': [], 'inhibitor_first': [], 'inhibitor_kills': [], 'riftHerald_first': [], 'riftHerald_kills': [], 'tower_first': [], 'tower_kills': []})

df_matches_participant = pd.DataFrame({'matchId': [],'puuid': [],'teamId': [], 'allInPings': [], 'assistMePings': [], 'assists': [], 'baitPings': [], 'baronKills': [], 'basicPings': [], 'bountyLevel': [], 
'champExperience': [], 'champLevel': [], 'championId': [], 'championName': [], 'championTransform': [], 'commandPings': [], 'consumablesPurchased': [], 'damageDealtToBuildings': [], 'damageDealtToObjectives': [], 'damageDealtToTurrets': [], 'damageSelfMitigated': [], 'dangerPings': [], 'deaths': [], 'detectorWardsPlaced': [], 'doubleKills': [], 'dragonKills': [], 'eligibleForProgression': [], 'enemyMissingPings': [], 'enemyVisionPings': [], 'firstBloodAssist': [], 'firstBloodKill': [], 'firstTowerAssist': [], 'firstTowerKill': [], 'gameEndedInEarlySurrender': [], 'gameEndedInSurrender': [], 'getBackPings': [], 'goldEarned': [], 'goldSpent': [], 'holdPings': [], 'individualPosition': [], 'inhibitorKills': [], 'inhibitorTakedowns': [], 'inhibitorsLost': [], 'item0': [], 'item1': [], 'item2': [], 'item3': [], 'item4': [], 'item5': [], 'item6': [], 'itemsPurchased': [], 'killingSprees': [], 'kills': [], 'lane': [], 'largestCriticalStrike': [], 'largestKillingSpree': [], 'largestMultiKill': [], 'longestTimeSpentLiving': [], 'magicDamageDealt': [], 'magicDamageDealtToChampions': [], 'magicDamageTaken': [], 'needVisionPings': [], 'neutralMinionsKilled': [], 'nexusKills': [], 'nexusLost': [], 'nexusTakedowns': [], 'objectivesStolen': [], 'objectivesStolenAssists': [], 'onMyWayPings': [], 'participantId': [], 'pentaKills': [], 'defense': [], 'flex': [], 'offense': [], 'selection_1_description': [], 'selections1_1_perk': [], 'selections1_1_var1': [], 'selections1_1_var2': [], 'selections1_1_var3': [], 'selection1_2_perk': [], 'selection1_2_var1': [], 'selection1_2_var2': [], 'selection1_2_var3': [], 'selection1_3_perk': [], 'selection1_3_var1': [], 'selection1_3_var2': [], 'selection1_3_var3': [], 'selection1_4_perk': [], 'selection1_4_var1': [], 'selection1_4_var2': [], 'selection1_4_var3': [],'selection_1_style': [], 'selection_2_description': [], 'selection2_1_perk': [], 'selection2_1_var1': [], 'selection2_1_var2': [], 'selection2_1_var3': [], 'selection2_2_perk': [], 'selection2_2_var1': [], 'selection2_2_var2': [], 'selection2_2_var3': [],'selection_2_style': [], 'physicalDamageDealt': [], 'physicalDamageDealtToChampions': [], 'physicalDamageTaken': [], 'profileIcon': [], 'pushPings': [],  'quadraKills': [], 'riotIdName': [], 'riotIdTagline': [], 'role': [], 'sightWardsBoughtInGame': [], 'spell1Casts': [], 'spell2Casts': [], 'spell3Casts': [], 'spell4Casts': [], 'summoner1Casts': [], 'summoner1Id': [], 'summoner2Casts': [], 'summoner2Id': [], 'summonerId': [], 'summonerLevel': [], 'summonerName': [], 'teamEarlySurrendered': [], 'teamPosition': [], 'timeCCingOthers': [], 'timePlayed': [], 'totalDamageDealt': [], 'totalDamageDealtToChampions': [], 'totalDamageShieldedOnTeammates': [], 'totalDamageTaken': [], 'totalHeal': [], 'totalHealsOnTeammates': [], 'totalMinionsKilled': [], 'totalTimeCCDealt': [], 'totalTimeSpentDead': [], 'totalUnitsHealed': [], 'tripleKills': [], 'trueDamageDealt': [], 'trueDamageDealtToChampions': [], 'trueDamageTaken': [], 'turretKills': [], 'turretsLost': [], 'unrealKills': [], 'visionClearedPings': [], 'visionScore': [], 'visionWardsBoughtInGame': [], 'wardsKilled': [], 'wardsPlaced': [], 'win': [] })




# Guardar el timestamp como string
timestamp2 = datetime.now()
timestamp_str = timestamp2.strftime('%Y-%m-%d %H-%M-%S')
print(timestamp_str)

# Recoger nombres de jugadores:
# Hacemos un If para saber si está creada la tabla del CSV, si lo está, seguimos con esa, sino, la creamos.
tabla_players = 'tables/csv/players.csv'
print(os.path.isfile(tabla_players))

if os.path.isfile(tabla_players):
 # Si el archivo existe, cargar el DataFrame desde el archivo
    with open(tabla_players, mode='rb') as archivo:
        print("Ha entrado al with open")
    # Lee el contenido del archivo CSV utilizando la biblioteca pandas
        df_players = pd.read_csv(archivo, sep=";", encoding='utf-8')
else:
    # Si el archivo no existe, lo creamos
    # Por liga
    #for a in range(len(tier)):
    for r in range(len(region)):
        
        for a in range(len(tier)):      
            # Por division
            rango = range(len(division))
            if tier[a] in ("CHALLENGER", "GRANDMASTER", "MASTER"):
                rango = range(0,1)

            # Número de página
            for b in rango:
                #url = f'https://{region}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/{queue}?api_key={api_key}&page={i}'
                url = f'https://{region[r]}.api.riotgames.com/lol/league-exp/v4/entries/{queue}/{tier[a]}/{division[b]}?api_key={api_key}'
                response = requests.get(url)
                if response.status_code == 200:
                    players = response.json()

                    data_entries = players
                    print("La liga es", tier[a], "la division es", division[b], "la region es", region[r], "y el numero de personas es", len(data_entries))
                    # Guardar el nombre de los jugadores    
                    for j in range(len(data_entries)):
                        if len(df_players)==0:
                            pos = 0
                        else:
                            pos = 1
                        position = pos+len(df_players)
                        df_players.loc[position, 'summonerId'] = data_entries[j]['summonerId']
                        df_players.loc[position, 'summonerName']  = data_entries[j]['summonerName']
                        df_players.loc[position, 'region']  = region[r]
                        df_players.loc[position, 'leaguePoints']  = format(data_entries[j]['leaguePoints'], 'd')
                        df_players.loc[position, 'rank']  = data_entries[j]['rank']
                        df_players.loc[position, 'wins']  = format(data_entries[j]['wins'], 'd')
                        df_players.loc[position, 'losses']  = format(data_entries[j]['losses'], 'd')
                        df_players.loc[position, 'veteran']  = data_entries[j]['veteran']
                        df_players.loc[position, 'inactive']  = data_entries[j]['inactive']
                        df_players.loc[position, 'freshBlood']  = data_entries[j]['freshBlood']
                        df_players.loc[position, 'hotStreak']  = data_entries[j]['hotStreak']
                        df_players.loc[position, 'tier']  = data_entries[j]['tier']
                        df_players.loc[position, 'leagueId']  = data_entries[j]['leagueId']
                        df_players.loc[position, 'queue']  = data_entries[j]['queueType']
                        
                else:
                    print('No se pudo obtener la información')
                    print(response.status_code)
                solicitudes += 1
                # espera 1.2 segundos entre cada solicitud
                time.sleep(1.2)
                
                # cada 20 solicitudes, espera 2 segundos
                if solicitudes % 20 == 0 and solicitudes > 0:
                    print(f"Esperando 2 segundos para continuar... ({solicitudes+1}/100)")
                    time.sleep(2)
                # Guardamos los jugadores en un csv
    players_csv = "tables/csv/players.csv"
    df_players.to_csv(players_csv, index=False, encoding='utf-8', sep=";")
    


print("La longitud del df_players es:", len(df_players))
print(df_players)



solicitudes = 0

matches_md = 'tables/csv/match_md.csv'
matches_info = 'tables/csv/match_info.csv'
matches_info_teams = 'tables/csv/match_info_team.csv'
matches_participants = 'tables/csv/match_participant.csv'

if os.path.isfile(matches_md) and os.path.isfile(matches_info) and os.path.isfile(matches_info_teams) and os.path.isfile(matches_participants):
    print("\n")
    print ("Los ficheros ya existen, fin de la extracción de datos.")
    print("\n")
else:
        # Si el archivo no existe, lo creamos
    
    #for i in range(len(c)): ## hacemos bucle para que vaya a recoger las ultimas X partidas del summonerName de la tabla df_players

    #df_players=df_players.loc[df_players['inactive']==False]

    for i in range(len(df_players)): ## hacemos bucle para que vaya a recoger las ultimas X partidas del summonerName de la tabla df_players
        try: 
            summoner_name = df_players.loc[i, 'summonerName'] ## recogemos en sumoner_name directamente el summoner_name del jugador i
            region_player= df_players.loc[i, 'region'] ## recogemos la region del jugador i
            # Configurar la URL de la solicitud
            #summoner_name = 'Adelphos1313' #df_players.loc[0, "summoner_name"]
            print("Inicio player numero", i, "con nombre", summoner_name, "y region", region_player)
            url = f'https://{region_player}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{summoner_name}?api_key={api_key}'
            # Realizar la solicitud
            response = requests.get(url)
            solicitudes += 1
            # cada 20 solicitudes, espera 2 segundos
            if solicitudes % 100 == 0 and solicitudes > 0:
                print(f"${solicitudes} solicitudes, esperando 0.5 segundos para continuar... ")
                time_now = datetime.now()
                hora_actual = timestamp.strftime('%Y-%m-%d %H-%M-%S')
                print("Hora inicio ejecucion:", hora_inicio_ejecucion, "VS hora actual de ejecución:",hora_actual)
                time.sleep(0.5)
            elif solicitudes % 20 == 0 and solicitudes > 0:
                print(f"${solicitudes} solicitudes, esperando 1.2 segundos para continuar...")
                time_now = datetime.now()
                hora_actual = time_now.strftime('%Y-%m-%d %H-%M-%S')
                print("Hora inicio ejecucion:", hora_inicio_ejecucion, "VS hora actual de ejecución:",hora_actual)
                time.sleep(1.2)
            else:
                print(f"${solicitudes} solicitudes, esperando 1.2 segundos para continuar... ")
                time_now = datetime.now()
                hora_actual = time_now.strftime('%Y-%m-%d %H-%M-%S')
                print("Hora inicio ejecucion:", hora_inicio_ejecucion, "VS hora actual de ejecución:",hora_actual)
                time.sleep(1.2)

        
            puuidJ = []
            if response.status_code == 200:
                puuidJ = response.json()
            else:
                print('No se pudo obtener el puuid')

            if len(puuidJ)!=0:

                puuid = puuidJ["puuid"]
                print("El puuid es:", puuid)
                
                if region_player == "NA1":
                    continente = "americas"
                elif region_player == "KR":
                    continente = "asia"
                else:
                    continente = "europe"
                
                url = f'https://{continente}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&api_key={api_key}'
                # Realizar la solicitud
                response = requests.get(url)

                solicitudes += 1
                # cada 20 solicitudes, espera 2 segundos
                if solicitudes % 100 == 0 and solicitudes > 0:
                    print(f"${solicitudes} solicitudes, esperando 0.5 segundos para continuar... ")
                    time_now = datetime.now()
                    hora_actual = timestamp.strftime('%Y-%m-%d %H-%M-%S')
                    print("Hora inicio ejecucion:", hora_inicio_ejecucion, "VS hora actual de ejecución:",hora_actual)
                    time.sleep(0.5)
                elif solicitudes % 20 == 0 and solicitudes > 0:
                    print(f"${solicitudes} solicitudes, esperando 1.2 segundos para continuar...")
                    time_now = datetime.now()
                    hora_actual = time_now.strftime('%Y-%m-%d %H-%M-%S')
                    print("Hora inicio ejecucion:", hora_inicio_ejecucion, "VS hora actual de ejecución:",hora_actual)
                    time.sleep(1.2)
                else:
                    print(f"${solicitudes} solicitudes, esperando 1.2 segundos para continuar... ")
                    time_now = datetime.now()
                    hora_actual = time_now.strftime('%Y-%m-%d %H-%M-%S')
                    print("Hora inicio ejecucion:", hora_inicio_ejecucion, "VS hora actual de ejecución:",hora_actual)
                    time.sleep(1.2)

                    
                if response.status_code == 200:
                    matchidJ = response.json()
                else:
                    print('No se pudo obtener el match')
                    print(response.status_code)
                
                
                
                with open("tables/json/match_id.json", "w") as f:
                    json.dump(matchidJ, f)
                
                if len(matchidJ) != 0:
                    #########SACAMOS TODA LA INFORMACION DE LA PARTIDA POR TODOS LOS JUGADORES QUE JUEGAN EN ELLA:
                    url = f'https://{continente}.api.riotgames.com/lol/match/v5/matches/{matchidJ[0]}?api_key={api_key}'
                    
                    print("El match id que hemos usado para estudiar es:", matchidJ[0])
                    # Realizar la solicitud
                    response = requests.get(url)

                    ##limitación tiempo
                    solicitudes += 1
                    # cada 20 solicitudes, espera 2 segundos
                    if solicitudes % 100 == 0 and solicitudes > 0:
                        print(f"${solicitudes} solicitudes, esperando 0.5 segundos para continuar... ")
                        time_now = datetime.now()
                        hora_actual = timestamp.strftime('%Y-%m-%d %H-%M-%S')
                        print("Hora inicio ejecucion:", hora_inicio_ejecucion, "VS hora actual de ejecución:",hora_actual)
                        time.sleep(0.5)
                    elif solicitudes % 20 == 0 and solicitudes > 0:
                        print(f"${solicitudes} solicitudes, esperando 1.2 segundos para continuar...")
                        time_now = datetime.now()
                        hora_actual = time_now.strftime('%Y-%m-%d %H-%M-%S')
                        print("Hora inicio ejecucion:", hora_inicio_ejecucion, "VS hora actual de ejecución:",hora_actual)
                        time.sleep(1.2)
                    else:
                        print(f"${solicitudes} solicitudes, esperando 1.2 segundos para continuar... ")
                        time_now = datetime.now()
                        hora_actual = time_now.strftime('%Y-%m-%d %H-%M-%S')
                        print("Hora inicio ejecucion:", hora_inicio_ejecucion, "VS hora actual de ejecución:",hora_actual)
                        time.sleep(1.2)


                    if response.status_code == 200:
                        game = response.json()
                        #print(game)
                    else:
                        print('No se pudo obtener el game')
                        print(response.status_code)
                    
                    
                    with open("tables/json/game.json", "w") as f:
                        json.dump(game, f)

                    #########################################
                    #########################################
                    ## NECESITA REVISION LAS NUEVAS TABLAS ##
                    #########################################
                    #########################################

                    ## Carga de la tabla TeamsCleaned


                    #Creamos la tabla df_matches_info, esta tabla recoge datos genericos de la partida, tendrá 1 registro por partida.
                    df_matches_info.loc[i, 'matchId']  = game["metadata"]["matchId"]
                    #df_matches_info.loc[i, 'gameCreation']  = game["info"]["gameCreation"]
                    df_matches_info.loc[i, 'gameDuration']  = game["info"]['gameDuration']
                    #df_matches_info.loc[i, 'gameEndTimestamp']  = game["info"]['gameEndTimestamp']
                    df_matches_info.loc[i, 'gameId']  = game["info"]['gameId']
                    #df_matches_info.loc[i, 'gameMode']  = game["info"]['gameMode']
                    #df_matches_info.loc[i, 'gameName']  = game["info"]['gameName']
                    #df_matches_info.loc[i, 'gameStartTimestamp']  = game["info"]['gameStartTimestamp']
                    #df_matches_info.loc[i, 'gameType']  = game["info"]['gameType']
                    #df_matches_info.loc[i, 'gameVersion']  = game["info"]['gameVersion']
                    #df_matches_info.loc[i, 'mapId']  = game["info"]['mapId']
                    df_matches_info.loc[i, 'platformId']  = game["info"]['platformId'] #Region
                    #df_matches_info.loc[i, 'queueId']  = game["info"]['queueId'] #debería de ser siempre 420
                    #df_matches_info.loc[i, 'tournamentCode'] = game["info"]['tournamentCode'] #No se si debería de informarse puesto que en queueId es siempre 420 y nose si incluye torneos en este codigo.
                    
                
                    #match_info_csv = "tables/csv/match_info.csv"
                    #df_matches_info.to_csv(match_info_csv, index=False, encoding='utf-8', sep=";")
                    
                    for a in range(0,2):
                        if len(df_matches_info_teams)==0:
                            pos = 0
                        else:
                            pos = 1
                        position = pos+len(df_matches_info_teams)

                        info_game = game['info']['teams'][a]
                        df_matches_info_teams.loc[position, 'matchId'] = game['metadata']['matchId']
                        df_matches_info_teams.loc[position, 'teamId'] = info_game['teamId']
                        df_matches_info_teams.loc[position, 'bans_championId_1'] = info_game['bans'][0]['championId']
                        df_matches_info_teams.loc[position, 'bans_championId_2'] = info_game['bans'][1]['championId']
                        df_matches_info_teams.loc[position, 'bans_championId_3'] = info_game['bans'][2]['championId']
                        df_matches_info_teams.loc[position, 'bans_championId_4'] = info_game['bans'][3]['championId']
                        df_matches_info_teams.loc[position, 'bans_championId_5'] = info_game['bans'][4]['championId']
                        df_matches_info_teams.loc[position, 'baron_first'] = info_game['objectives']['baron']['first']
                        df_matches_info_teams.loc[position, 'baron_kills'] = info_game['objectives']['baron']['kills']
                        df_matches_info_teams.loc[position, 'champion_first'] = info_game['objectives']['champion']['first']
                        df_matches_info_teams.loc[position, 'champion_kills'] = info_game['objectives']['champion']['kills']
                        df_matches_info_teams.loc[position, 'dragon_first'] = info_game['objectives']['dragon']['first']
                        df_matches_info_teams.loc[position, 'dragon_kills'] = info_game['objectives']['dragon']['kills']
                        df_matches_info_teams.loc[position, 'inhibitor_first'] = info_game['objectives']['inhibitor']['first']
                        df_matches_info_teams.loc[position, 'inhibitor_kills'] = info_game['objectives']['inhibitor']['kills']
                        df_matches_info_teams.loc[position, 'riftHerald_first'] = info_game['objectives']['riftHerald']['first']
                        df_matches_info_teams.loc[position, 'riftHerald_kills'] = info_game['objectives']['riftHerald']['kills']
                        df_matches_info_teams.loc[position, 'tower_first'] = info_game['objectives']['tower']['first']
                        df_matches_info_teams.loc[position, 'tower_kills'] = info_game['objectives']['tower']['kills']

                    #matches_info_teams_csv = "tables/csv/matches_info_teams.csv"
                    #df_matches_info_teams.to_csv(matches_info_teams_csv, index=False, encoding='utf-8', sep=";")         
                            
                            
                        
                    # j jugadores que hay dentro de la partida
                    for j in range(len(game['metadata']['participants'])):
                        if len(df_matches_participant)==0:
                            pos = 0
                        else:
                            pos = 1
                        position = pos+len(df_matches_participant)

                        #Creamos la tabla df_matches_metadata, tendrá 1 registro por jugador de la partida, 10 registros por partida
                        df_matches_md.loc[position, 'dataVersion'] = game['metadata']['dataVersion']
                        df_matches_md.loc[position, 'matchId'] = game['metadata']['matchId']
                        df_matches_md.loc[position, 'puuid'] = game['metadata']['participants'][j]
                        


                        #Creamos la tabla df_matches_participant, tendrá 1 registro por jugador de la partida, 10 registros por partida
                            
                        info_participants = game['info']['participants'][j] 

                        df_matches_participant.loc[position, 'matchId'] = game['metadata']['matchId']
                        df_matches_participant.loc[position, 'puuid']  = info_participants['puuid']
                        df_matches_participant.loc[position, 'teamId']  = info_participants['teamId']
                        df_matches_participant.loc[position, 'allInPings']  = info_participants['allInPings']
                        df_matches_participant.loc[position, 'assistMePings']  = info_participants['assistMePings']
                        df_matches_participant.loc[position, 'assists']  = info_participants['assists']
                        df_matches_participant.loc[position, 'baitPings']  = info_participants['baitPings']
                        df_matches_participant.loc[position, 'baronKills']  = info_participants['baronKills']
                        df_matches_participant.loc[position, 'basicPings']  = info_participants['basicPings']
                        df_matches_participant.loc[position, 'bountyLevel']  = info_participants['bountyLevel']
                        df_matches_participant.loc[position, 'champExperience']  = info_participants['champExperience']
                        df_matches_participant.loc[position, 'champLevel']  = info_participants['champLevel']
                        df_matches_participant.loc[position, 'championId']  = info_participants['championId']
                        df_matches_participant.loc[position, 'championName']  = info_participants['championName']
                        df_matches_participant.loc[position, 'championTransform']  = info_participants['championTransform']
                        df_matches_participant.loc[position, 'commandPings']  = info_participants['commandPings']
                        df_matches_participant.loc[position, 'consumablesPurchased']  = info_participants['consumablesPurchased']
                        df_matches_participant.loc[position, 'damageDealtToBuildings']  = info_participants['damageDealtToBuildings']
                        df_matches_participant.loc[position, 'damageDealtToObjectives']  = info_participants['damageDealtToObjectives']
                        df_matches_participant.loc[position, 'damageDealtToTurrets']  = info_participants['damageDealtToTurrets']
                        df_matches_participant.loc[position, 'damageSelfMitigated']  = info_participants['damageSelfMitigated']
                        df_matches_participant.loc[position, 'dangerPings']  = info_participants['dangerPings']
                        df_matches_participant.loc[position, 'deaths']  = info_participants['deaths']
                        df_matches_participant.loc[position, 'detectorWardsPlaced']  = info_participants['detectorWardsPlaced']
                        df_matches_participant.loc[position, 'doubleKills']  = info_participants['doubleKills']
                        df_matches_participant.loc[position, 'dragonKills']  = info_participants['dragonKills']
                        df_matches_participant.loc[position, 'eligibleForProgression']  = info_participants['eligibleForProgression']
                        df_matches_participant.loc[position, 'enemyMissingPings']  = info_participants['enemyMissingPings']
                        df_matches_participant.loc[position, 'enemyVisionPings']  = info_participants['enemyVisionPings']
                        df_matches_participant.loc[position, 'firstBloodAssist']  = info_participants['firstBloodAssist']
                        df_matches_participant.loc[position, 'firstBloodKill']  = info_participants['firstBloodKill']
                        df_matches_participant.loc[position, 'firstTowerAssist']  = info_participants['firstTowerAssist']
                        df_matches_participant.loc[position, 'firstTowerKill']  = info_participants['firstTowerKill']
                        df_matches_participant.loc[position, 'gameEndedInEarlySurrender']  = info_participants['gameEndedInEarlySurrender']
                        df_matches_participant.loc[position, 'gameEndedInSurrender']  = info_participants['gameEndedInSurrender']
                        df_matches_participant.loc[position, 'getBackPings']  = info_participants['getBackPings']
                        df_matches_participant.loc[position, 'goldEarned']  = info_participants['goldEarned']
                        df_matches_participant.loc[position, 'goldSpent']  = info_participants['goldSpent']
                        df_matches_participant.loc[position, 'holdPings']  = info_participants['holdPings']
                        df_matches_participant.loc[position, 'individualPosition']  = info_participants['individualPosition']
                        df_matches_participant.loc[position, 'inhibitorKills']  = info_participants['inhibitorKills']
                        df_matches_participant.loc[position, 'inhibitorTakedowns']  = info_participants['inhibitorTakedowns']
                        df_matches_participant.loc[position, 'inhibitorsLost']  = info_participants['inhibitorsLost']
                        df_matches_participant.loc[position, 'item0']  = info_participants['item0']
                        df_matches_participant.loc[position, 'item1']  = info_participants['item1']
                        df_matches_participant.loc[position, 'item2']  = info_participants['item2']
                        df_matches_participant.loc[position, 'item3']  = info_participants['item3']
                        df_matches_participant.loc[position, 'item4']  = info_participants['item4']
                        df_matches_participant.loc[position, 'item5']  = info_participants['item5']
                        df_matches_participant.loc[position, 'item6']  = info_participants['item6']
                        df_matches_participant.loc[position, 'itemsPurchased']  = info_participants['itemsPurchased']
                        df_matches_participant.loc[position, 'killingSprees']  = info_participants['killingSprees']
                        df_matches_participant.loc[position, 'kills']  = info_participants['kills']
                        df_matches_participant.loc[position, 'lane']  = info_participants['lane']
                        df_matches_participant.loc[position, 'largestCriticalStrike']  = info_participants['largestCriticalStrike']
                        df_matches_participant.loc[position, 'largestKillingSpree']  = info_participants['largestKillingSpree']
                        df_matches_participant.loc[position, 'largestMultiKill']  = info_participants['largestMultiKill']
                        df_matches_participant.loc[position, 'longestTimeSpentLiving']  = info_participants['longestTimeSpentLiving']
                        df_matches_participant.loc[position, 'magicDamageDealt']  = info_participants['magicDamageDealt']
                        df_matches_participant.loc[position, 'magicDamageDealtToChampions']  = info_participants['magicDamageDealtToChampions']
                        df_matches_participant.loc[position, 'magicDamageTaken']  = info_participants['magicDamageTaken']
                        df_matches_participant.loc[position, 'needVisionPings']  = info_participants['needVisionPings']
                        df_matches_participant.loc[position, 'neutralMinionsKilled']  = info_participants['neutralMinionsKilled']
                        df_matches_participant.loc[position, 'nexusKills']  = info_participants['nexusKills']
                        df_matches_participant.loc[position, 'nexusLost']  = info_participants['nexusLost']
                        df_matches_participant.loc[position, 'nexusTakedowns']  = info_participants['nexusTakedowns']
                        df_matches_participant.loc[position, 'objectivesStolen']  = info_participants['objectivesStolen']
                        df_matches_participant.loc[position, 'objectivesStolenAssists']  = info_participants['objectivesStolenAssists']
                        df_matches_participant.loc[position, 'onMyWayPings']  = info_participants['onMyWayPings']
                        df_matches_participant.loc[position, 'participantId']  = info_participants['participantId']
                        df_matches_participant.loc[position, 'pentaKills']  = info_participants['pentaKills']

                        # Runas      
                        df_matches_participant.loc[position, 'perks_defense']  = info_participants['perks']['statPerks']['defense']
                        df_matches_participant.loc[position, 'perks_flex']  = info_participants['perks']['statPerks']['flex']
                        df_matches_participant.loc[position, 'perks_offense']  = info_participants['perks']['statPerks']['offense']
                        df_matches_participant.loc[position, 'selection_1_description']  = info_participants['perks']['styles'][0]['description']
                        df_matches_participant.loc[position, 'selections1_1_perk']  = info_participants['perks']['styles'][0]['selections'][0]['perk']
                        df_matches_participant.loc[position, 'selections1_1_var1']  = info_participants['perks']['styles'][0]['selections'][0]['var1']
                        df_matches_participant.loc[position, 'selections1_1_var2']  = info_participants['perks']['styles'][0]['selections'][0]['var2']
                        df_matches_participant.loc[position, 'selections1_1_var3']  = info_participants['perks']['styles'][0]['selections'][0]['var3']
                        df_matches_participant.loc[position, 'selection1_2_perk']  = info_participants['perks']['styles'][0]['selections'][1]['perk']
                        df_matches_participant.loc[position, 'selection1_2_var1']  = info_participants['perks']['styles'][0]['selections'][1]['var1']
                        df_matches_participant.loc[position, 'selection1_2_var2']  = info_participants['perks']['styles'][0]['selections'][1]['var2']
                        df_matches_participant.loc[position, 'selection1_2_var3']  = info_participants['perks']['styles'][0]['selections'][1]['var3']
                        df_matches_participant.loc[position, 'selection1_3_perk']  = info_participants['perks']['styles'][0]['selections'][2]['perk']
                        df_matches_participant.loc[position, 'selection1_3_var1']  = info_participants['perks']['styles'][0]['selections'][2]['var1']
                        df_matches_participant.loc[position, 'selection1_3_var2']  = info_participants['perks']['styles'][0]['selections'][2]['var2']
                        df_matches_participant.loc[position, 'selection1_3_var3']  = info_participants['perks']['styles'][0]['selections'][2]['var3']
                        df_matches_participant.loc[position, 'selection1_4_perk']  = info_participants['perks']['styles'][0]['selections'][3]['perk']
                        df_matches_participant.loc[position, 'selection1_4_var1']  = info_participants['perks']['styles'][0]['selections'][3]['var1']
                        df_matches_participant.loc[position, 'selection1_4_var2']  = info_participants['perks']['styles'][0]['selections'][3]['var2']
                        df_matches_participant.loc[position, 'selection1_4_var3']  = info_participants['perks']['styles'][0]['selections'][3]['var3']
                        df_matches_participant.loc[position, 'selection_1_style']  = info_participants['perks']['styles'][0]['style']
                        df_matches_participant.loc[position, 'selection_2_description']  = info_participants['perks']['styles'][1]['description']
                        df_matches_participant.loc[position, 'selection2_1_perk']  = info_participants['perks']['styles'][1]['selections'][0]['perk']
                        df_matches_participant.loc[position, 'selection2_1_var1']  = info_participants['perks']['styles'][1]['selections'][0]['var1']
                        df_matches_participant.loc[position, 'selection2_1_var2']  = info_participants['perks']['styles'][1]['selections'][0]['var2']
                        df_matches_participant.loc[position, 'selection2_1_var3']  = info_participants['perks']['styles'][1]['selections'][0]['var3']
                        df_matches_participant.loc[position, 'selection2_2_perk']  = info_participants['perks']['styles'][1]['selections'][1]['perk']
                        df_matches_participant.loc[position, 'selection2_2_var1']  = info_participants['perks']['styles'][1]['selections'][1]['var1']
                        df_matches_participant.loc[position, 'selection2_2_var2']  = info_participants['perks']['styles'][1]['selections'][1]['var2']
                        df_matches_participant.loc[position, 'selection2_2_var3']  = info_participants['perks']['styles'][1]['selections'][1]['var3']
                        df_matches_participant.loc[position, 'selection_2_style']  = info_participants['perks']['styles'][1]['style']
                        df_matches_participant.loc[position, 'physicalDamageDealt']  = info_participants['physicalDamageDealt']
                        df_matches_participant.loc[position, 'physicalDamageDealtToChampions']  = info_participants['physicalDamageDealtToChampions']
                        df_matches_participant.loc[position, 'physicalDamageTaken']  = info_participants['physicalDamageTaken']
                        df_matches_participant.loc[position, 'profileIcon']  = info_participants['profileIcon']
                        df_matches_participant.loc[position, 'pushPings']  = info_participants['pushPings']
                        df_matches_participant.loc[position, 'quadraKills']  = info_participants['quadraKills']
                        df_matches_participant.loc[position, 'riotIdName']  = info_participants['riotIdName']
                        df_matches_participant.loc[position, 'riotIdTagline']  = info_participants['riotIdTagline']
                        df_matches_participant.loc[position, 'role']  = info_participants['role']
                        df_matches_participant.loc[position, 'sightWardsBoughtInGame']  = info_participants['sightWardsBoughtInGame']
                        df_matches_participant.loc[position, 'spell1Casts']  = info_participants['spell1Casts']
                        df_matches_participant.loc[position, 'spell2Casts']  = info_participants['spell2Casts']
                        df_matches_participant.loc[position, 'spell3Casts']  = info_participants['spell3Casts']
                        df_matches_participant.loc[position, 'spell4Casts']  = info_participants['spell4Casts']
                        df_matches_participant.loc[position, 'summoner1Casts']  = info_participants['summoner1Casts']
                        df_matches_participant.loc[position, 'summoner1Id']  = info_participants['summoner1Id']
                        df_matches_participant.loc[position, 'summoner2Casts']  = info_participants['summoner2Casts']
                        df_matches_participant.loc[position, 'summoner2Id']  = info_participants['summoner2Id']
                        df_matches_participant.loc[position, 'summonerId']  = info_participants['summonerId']
                        df_matches_participant.loc[position, 'summonerLevel']  = info_participants['summonerLevel']
                        df_matches_participant.loc[position, 'summonerName']  = info_participants['summonerName']
                        df_matches_participant.loc[position, 'teamEarlySurrendered']  = info_participants['teamEarlySurrendered']
                        df_matches_participant.loc[position, 'teamPosition']  = info_participants['teamPosition']
                        df_matches_participant.loc[position, 'timeCCingOthers']  = info_participants['timeCCingOthers']
                        df_matches_participant.loc[position, 'timePlayed']  = info_participants['timePlayed']
                        df_matches_participant.loc[position, 'totalDamageDealt']  = info_participants['totalDamageDealt']
                        df_matches_participant.loc[position, 'totalDamageDealtToChampions']  = info_participants['totalDamageDealtToChampions']
                        df_matches_participant.loc[position, 'totalDamageShieldedOnTeammates']  = info_participants['totalDamageShieldedOnTeammates']
                        df_matches_participant.loc[position, 'totalDamageTaken']  = info_participants['totalDamageTaken']
                        df_matches_participant.loc[position, 'totalHeal']  = info_participants['totalHeal']
                        df_matches_participant.loc[position, 'totalHealsOnTeammates']  = info_participants['totalHealsOnTeammates']
                        df_matches_participant.loc[position, 'totalMinionsKilled']  = info_participants['totalMinionsKilled']
                        df_matches_participant.loc[position, 'totalTimeCCDealt']  = info_participants['totalTimeCCDealt']
                        df_matches_participant.loc[position, 'totalTimeSpentDead']  = info_participants['totalTimeSpentDead']
                        df_matches_participant.loc[position, 'totalUnitsHealed']  = info_participants['totalUnitsHealed']
                        df_matches_participant.loc[position, 'tripleKills']  = info_participants['tripleKills']
                        df_matches_participant.loc[position, 'trueDamageDealt']  = info_participants['trueDamageDealt']
                        df_matches_participant.loc[position, 'trueDamageDealtToChampions']  = info_participants['trueDamageDealtToChampions']
                        df_matches_participant.loc[position, 'trueDamageTaken']  = info_participants['trueDamageTaken']
                        df_matches_participant.loc[position, 'turretKills']  = info_participants['turretKills']
                        df_matches_participant.loc[position, 'turretsLost']  = info_participants['turretsLost']
                        df_matches_participant.loc[position, 'unrealKills']  = info_participants['unrealKills']
                        df_matches_participant.loc[position, 'visionClearedPings']  = info_participants['visionClearedPings']
                        df_matches_participant.loc[position, 'visionScore']  = info_participants['visionScore']
                        df_matches_participant.loc[position, 'visionWardsBoughtInGame']  = info_participants['visionWardsBoughtInGame']
                        df_matches_participant.loc[position, 'wardsKilled']  = info_participants['wardsKilled']
                        df_matches_participant.loc[position, 'wardsPlaced']  = info_participants['wardsPlaced']
                        df_matches_participant.loc[position, 'win']  = info_participants['win']

                        # Challenges
                        #challenges = info_participants["challenges"]
                        #df_challenges = pd.DataFrame(challenges, index=[position])
                        #df_matches_participant = pd.concat([df_matches_participant, df_challenges], axis=1)
                        """challenges = info_participants["challenges"]
                        lista_challenges = list(challenges.keys())
                        for col in lista_challenges:
                            if col not in df_matches_participant.columns:
                                df_matches_participant[col] = np.nan
                            df_matches_participant.loc[position, col] = challenges[col]"""

                        challenges = info_participants["challenges"]
                        lista_challenges = list(challenges.keys())
                        for campo in lista_challenges:
                            col_df=df_matches_participant.columns
                            new_cols={}
                            if campo not in col_df:
                                new_cols[campo] = challenges[campo]
                                df_new_cols = pd.DataFrame(new_cols, index=[position])
                                df_matches_participant = pd.concat([df_matches_participant, df_new_cols], axis=1)
                            else:
                                df_matches_participant.loc[position, campo] = challenges[campo]

                        #match_md_csv = "tables/csv/matches_md.csv"
                        #df_matches_md.to_csv(match_md_csv, index=False, encoding='utf-8', sep=";")

                        #match_participant_csv = "tables/csv/match_participant.csv"
                        #df_matches_participant.to_csv(match_participant_csv, index=False, encoding='utf-8', sep=";")
        except Exception as e:
            print(f"Error procesando el jugador", i)
            continue #Se pasa al siguiente jugador
                # Aquí 
    
    matches_info_teams_csv = "tables/csv/match_info_team.csv"
    df_matches_info_teams.to_csv(matches_info_teams_csv, index=False, encoding='utf-8', sep=";")  

    match_md_csv = "tables/csv/match_md.csv"
    df_matches_md.to_csv(match_md_csv, index=False, encoding='utf-8', sep=";")

    match_participant_csv = "tables/csv/match_participant.csv"
    df_matches_participant.to_csv(match_participant_csv, index=False, encoding='utf-8', sep=";")

    match_info_csv = "tables/csv/match_info.csv"
    df_matches_info.to_csv(match_info_csv, index=False, encoding='utf-8', sep=";")
