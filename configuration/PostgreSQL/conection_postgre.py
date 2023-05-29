import psycopg2
from sqlalchemy import create_engine

# Obtención credenciales de PostgreSQL
with open('configuration/key/secret_postgre.txt', 'r') as f:
    host = f.readline().strip()
    database = f.readline().strip()
    port = f.readline().strip()
    user = f.readline().strip()
    schema = f.readline().strip()
    password = f.readline().strip()

# Conectarse a la base de datos
conn = psycopg2.connect(
    host = host,
    database= database,
    port= port,
    user= user,
    password= password
)

# Crear el motor de SQLAlchemy
engine = create_engine('postgresql+psycopg2://', creator=lambda: conn)
