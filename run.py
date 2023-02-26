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
import random
import csv
import datetime as dt
import os
from bs4 import BeautifulSoup
from configuration.key.encrypt import username, f, fernet
import importlib.resources
from cryptography.fernet import Fernet

# URL
URL = 'https://developer.riotgames.com/'


# Fichero robots.txt:
def robots(url):
    if url[-1] == "/":
        r_link = url
    else:
        r_link = os.path.join(url, "/")
    rst = requests.get(r_link + 'robots.txt')
    return rst.text

# Lo imprimimos:
#print(robots(URL))


# Tecnología usada:
import builtwith

tech = builtwith.builtwith(URL)
print(tech)


# Propietario de la página:
import whois
print(whois.whois(URL))

# Obtenemos la password encriptada
with importlib.resources.open_text('configuration.key', 'encrypted_password') as f:
    encrypted_password = f.read()

encrypted_password = encrypted_password.encode()
decrypt_password = fernet.decrypt(encrypted_password).decode('utf-8')

# Credentials:
user = username
password = decrypt_password

# Abrir el navegador con la URL
s = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=s)

driver.get(URL)
sleep(1)
driver.maximize_window()
sleep(2)


# Click en el botón de login:
driver.find_element_by_xpath("""//*[@id="site-navbar-collapse"]/ul[2]""").click()
sleep(2)


# Inserción del nombre de usuario:
input_user = driver.find_element(By.XPATH, "/html/body/div[2]/div/div/div[2]/div/div/div[2]/div/div/div/div[1]/div/input")
input_user.send_keys(user)
sleep(3)

# Insertamos el password:
input_pass = driver.find_element(By.XPATH, "/html/body/div[2]/div/div/div[2]/div/div/div[2]/div/div/div/div[2]/div/input")
input_pass.send_keys(password)
sleep(3)

# Click en el botón de Siguiente:
next_button = driver.find_element_by_xpath("/html/body/div[2]/div/div/div[2]/div/div/button")
next_button.click()
sleep(5)


