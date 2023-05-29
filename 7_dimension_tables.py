
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



path_champions= "tables/api_dimension_tables/champion.json"

# Leer el archivo JSON
with open(path_champions, encoding='utf-8') as f:
    data_champions = json.load(f)


df_champions = pd.DataFrame()

df_champ = data_champions['data']
champ_name=list(df_champ.keys())
print(champ_name[0])
for i in df_champ:
    df_champions.loc[i, "ChampionName"]=df_champ[champ_name[i]]
    df_champions.loc[i, "ChampionAttack"]=df_champ[champ_name[i]]['info']['attack']
    df_champions.loc[i, "ChampionDefense"]=df_champ[champ_name[i]]['info']['defense']
    df_champions.loc[i, "ChampionMagic"]=df_champ[champ_name[i]]['info']['magic']
    df_champions.loc[i, "ChampionDifficulty"]=df_champ[champ_name[i]]['info']['difficulty']