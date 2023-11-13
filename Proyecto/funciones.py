import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

datos_games = pd.read_csv('/content/games.csv', sep = ';')

# Convirtiendo en una lista los elementos de las columnas "Team y Genre"
datos_games['Team'] = datos_games['Team'].apply(eval)
datos_games['Genres'] = datos_games['Genres'].apply(eval)

# Imprimir la tabla
datos_games

# Función No. 1

# Retorna un df con las filas donde encontró coincidencia en la columna 'Team'
def busca_retorna_por_equipo(df, equipo):
  # Filas encontradas
  filas = []
  # Recorriendo lista por lista
  # "enumerate" ayuda a darle seguimiento al índice actual a la vez que se recorre la lista
  for indice, i in enumerate(df['Team']):
    # Recorriendo los elementos de cada lista (todos los equipos)
    for j in i:
      if(j == equipo):
        filas.append(df.iloc[indice])

  # Crear un nuevo DataFrame con los registros encontrados
  resultado = pd.DataFrame(filas)
  # Restablecer los índices del nuevo DataFrame
  resultado.reset_index(drop = True, inplace = True)
  return(resultado)

datos_games_capcom = busca_retorna_por_equipo(datos_games, 'Capcom')

# Imprimiendo los cinco primeros títulos encontrados
datos_games_capcom.head(5)

# Función No. 2

# Retorna un df con las filas donde encontró coincidencia en la columna 'Genres'
def busca_retorna_por_genero(df, genero):
  # Filas encontradas
  filas = []
  # Recorriendo lista por lista
  for indice, i in enumerate(df['Genres']):
    # Recorriendo los elementos de cada lista (todos los generos)
    for j in i:
      if(j == genero):
        filas.append(df.iloc[indice])

  # Crear un nuevo DataFrame con los registros encontrados
  resultado = pd.DataFrame(filas)
  # Restablecer los índices del nuevo DataFrame
  resultado.reset_index(drop = True, inplace = True)
  return(resultado)

datos_games_genero = busca_retorna_por_genero(datos_games, 'RPG')

# Imprimiendo los cinco primeros títulos encontrados
datos_games_genero.head(5)

# Función No. 3

# Declarado como una tupla en lugar de una lista (estructura inmutable)
games_of_the_year = (
    "Dragon Age: Inquisition",
    "The Witcher 3: Wild Hunt",
    "Overwatch",
    "The Legend of Zelda: Breath of the Wild",
    "God of War",
    "Sekiro: Shadows Die Twice",
    "The Last of Us Part II",
    "It Takes Two",
    "Elden Ring"
)

# Retorna un df con la información de los títulos ganadores del premio "Game of the Year".
def obtener_titulos_premiados(df):
  df = df.loc[df['Title'].isin(games_of_the_year)]
  df.reset_index(drop = True, inplace = True)
  return(df)

datos_goty = obtener_titulos_premiados(datos_games)
datos_goty

# Función No. 4

# Determina la matriz de correlaciones para las columnas con valores numéricos en el df
def obtener_matriz_correlacion(df):
  # Selecciona solo las columnas con valores numéricos
  df = df.select_dtypes(include = 'number')
  return(df.corr())

# Función para convertir valores de correlación en strings
def clasificar_matriz_correlacion(valor):
    # El criterio para la clasifiación puede variar
    if valor > 0.1:
        return "Positiva"
    elif valor < 0:
        return "Negativa"
    else:
        return "Nula"

# Para términos estéticos la diagonal pasa a tener un "-" para aportar legibilidad
def limpiar_diagonal(df):
  # Obtener el tamaño del df
  filas, columnas = df.shape

  for i in range(min(filas, columnas)):
    # iat es una función que permite el acceso por valores enteros y no por etiquetas
    df.iat[i, i] = "-"

  return(df)

# Obteniendo la matriz de correlaciones
matriz_correlacion = obtener_matriz_correlacion(datos_games)
matriz_correlacion

plt.title('Mapa de calor de acuerdo a la matriz de correlación')
sns.heatmap(matriz_correlacion, annot=True, cmap="YlGnBu")

# Función No. 5

# Retorna los equipos con más de 2000 jugadores que poseen el juego pero no lo han jugado
def juegos_inactivos_por_equipo(df, equipo):
  df = busca_retorna_por_equipo(df, equipo)
  df = df.loc[(df['Backlogs'] >= 2000)]
  df.reset_index(drop = True, inplace = True)
  return(df)

datos_games_capcom = juegos_inactivos_por_equipo(datos_games, 'Sony Interactive Entertainment')
datos_games_capcom

# Función No. 6

# Obtiene los juegos que son activamente jugados por al menos 1500 jugadores
def juegos_activos_por_genero_del_juego(df, genero):
  df = busca_retorna_por_genero(df, genero)
  df = df.loc[(df['Playing'] >= 1500)]
  return(df)

datos_games_genero = juegos_activos_por_genero_del_juego(datos_games, 'RPG')
datos_games_genero

# Función No. 7

# Retorna los juegos según género con Rating mayor a 4.0, con mayor cantidad de reviews, con
# fecha de salida en los últimos 23 años (2000 en adelante) y ordenados ascendentemente por año
def juegos_rating_superior(df, genero):
    df = busca_retorna_por_genero(df, genero)
    df = df.loc[(df['Rating'] >= 4.0)]
    df['Release Date'] = pd.to_datetime(df['Release Date'], format='%d/%m/%Y')

    # Extrae el año de la fecha de lanzamiento y lo coloca como una columna más al df
    df['Year'] = df['Release Date'].dt.year

    # Agrupa por año de lanzamiento
    grupos = df.groupby('Year')

    # Para tener solo un juego por año, se toma en cuenta el juego con más reviews
    filas_seleccionadas = []
    for year, grupo in grupos:
        fila_max_reviews = grupo[grupo['Reviews'] == grupo['Reviews'].max()]
        filas_seleccionadas.append(fila_max_reviews)

    # Dataframe con el juego por año según género con mayor rating y puntuado por más usuarios
    df = pd.concat(filas_seleccionadas)

    # Filtra juegos lanzados desde el 2000 en adelante
    fecha_actual = pd.Timestamp.now()
    fecha_limite = fecha_actual - pd.DateOffset(years=23)
    df = df[df['Release Date'] >= fecha_limite]

    # Ordena por fecha de lanzamiento ascendente
    df = df.sort_values(by='Release Date', ascending=True)
    df.reset_index(drop = True, inplace = True)
    return df

datos_juegos_rating_superior = juegos_rating_superior(datos_games, 'RPG')
datos_juegos_rating_superior

# Función No. 8

# Encuentra el juego con el mayor rating para cada equipo desarrollador
def juego_mayor_rating_por_equipo(df):

    # Filtrando el df para solo conservar las columnas necesarias
    df = df[['Title', 'Team', 'Rating', 'Reviews']]

    # Crea una lista vacía para almacenar los juegos con mayor rating por equipo
    juegos_con_mayor_rating = []

    # Obtiene una lista con el nombre de cada equipo
    equipos = df['Team'].explode().unique()

    # Itera a través de cada equipo en el conjunto de datos y encuentra su juego con mayor rating
    for e in equipos:
        # Itera dentro del array de la columna "Team" para localizar al equipo en particular
        juegos_de_equipo = df[df['Team'].apply(lambda x: e in x)]
        juego_mayor_rating = juegos_de_equipo[juegos_de_equipo['Rating'] == juegos_de_equipo['Rating'].max()]
        # Dejando solamente el nombre del equipo en específico en lugar de todaos los equipos
        juego_mayor_rating.loc[juego_mayor_rating.index, 'Team'] = e
        juegos_con_mayor_rating.append(juego_mayor_rating)

    # Concatena todos los DataFrames individuales en uno solo
    df = pd.concat(juegos_con_mayor_rating)

    # Agrupa por equipo desarrollador
    grupos = df.groupby('Team')

    # Para tener solo un juego por equipo, se toma en cuenta el juego con más reviews
    filas_seleccionadas = []
    for team, grupo in grupos:
        fila_equipo = grupo[grupo['Reviews'] == grupo['Reviews'].max()]
        filas_seleccionadas.append(fila_equipo)

    # Dataframe con el juego por año según género con mayor rating y puntuado por más usuarios
    df = pd.concat(filas_seleccionadas)
    df.reset_index(drop = True, inplace = True)
    return df

# Imprimir los juegos con mayor rating por equipo desarrollador
juegos_con_mayor_rating_por_equipo = juego_mayor_rating_por_equipo(datos_games)
juegos_con_mayor_rating_por_equipo

# Función No. 9

# Busca al juego con el rating más alto de cada año y retorna la lista de juegos dependiendo de la cantidad de años que reciba por parámetro
def busca_retorna_por_rating(df, anio):

  # Se convierte la columna de la fecha de lanzamiento a formato de fecha
  df['Release Date'] = pd.to_datetime(df['Release Date'], format='%d/%m/%Y')

  # Se crea una variable con la fecha actual y se filtran los juegos de los últimos 10 años. También se crea una copia del dataFrame
  hoy = pd.Timestamp.now()
  df_filtrado = df[df['Release Date'] >= hoy - pd.DateOffset(years=10)].copy()

  # Se crea una columna año para agrupar a los juegos con mayor calificación un mismo año
  df_filtrado['Year'] = df_filtrado['Release Date'].dt.year

  # Encontrar el juego con el rating más alto de cada año
  resultados = df_filtrado.loc[df_filtrado.groupby('Year')['Rating'].idxmax()]
  resultados.reset_index(drop = True, inplace = True)

  # Mostrar los resultado
  return resultados[['Year','Title', 'Rating']]

busca_retorna_por_rating(datos_games, 20)

# Función No. 10

# Retorna un DF que muestra los juegos en los cuales la cantidad Backlogs (jugadores inactivos) es mayor a la cantidad de jugadores totales y muestra su diferencia
def comparacion_backlogs_plays(df, cantidadJuegos):

    # Se calcula la diferencia entre Backlogs y Plays
    df['Diferencia'] = df['Backlogs'] - df['Plays']

    # Se filtran los juegos donde la cantidad de Backlogs es mayor a la de Plays
    juegos_con_diferencia_positiva = df[df['Diferencia'] > 0]

    # Se muestran los juegos y la diferencia
    if not juegos_con_diferencia_positiva.empty:
        print("Juegos que se han comprado y no jugado a comparación de la cantidad de veces jugado:")
        return(juegos_con_diferencia_positiva[['Title', 'Backlogs', 'Plays', 'Genres', 'Rating', 'Diferencia']]).head(cantidadJuegos)
    else:
        return("No hay juegos con más Backlogs que Plays en el archivo.")

comparacion_backlogs_plays(datos_games, 10)

# Otro gráfico

# Muestra la cantidad de videojuegos con una calificación desde 3.5 hasta 4.6 de los géneros más influyentes
def crea_df_generos(df):
  # Filtrando el df para solo conservar las columnas necesarias
  df = df[['Genres', 'Rating']]
  generos = ['Adventure', 'RPG', 'Shooter', 'Horror', 'Indie', 'Racing', 'Strategy', 'Fighting']
  df_generos = []

  for genero in generos:
    df_aux = df[df['Genres'].apply(lambda x: genero in x)]
    df_aux.loc[df_aux.index, 'Genres'] = genero
    df_generos.append(df_aux)

  df = pd.concat(df_generos)
  return(df)

df_generos = crea_df_generos(datos_games)

df_generos = df_generos.loc[df_generos["Rating"] >= 3.5]

# Hacer un sort antes del crosstab
mapa_calor_genero_rating = pd.crosstab(df_generos['Rating'], df_generos['Genres'])

plt.title('Mapa de calor de acuerdo a la puntuación mayor a 3.5 de cada género por juego')

sns.heatmap(mapa_calor_genero_rating, annot = True, cmap = "YlGnBu")