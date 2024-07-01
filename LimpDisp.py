import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive/')

# Cargar los datos de películas y calificaciones
movies = pd.read_csv("/content/drive/MyDrive/RDI/PP/movies.csv")
ratings = pd.read_csv("/content/drive/MyDrive/RDI/PP/ratings.csv")

# Visualización de la dispersión de los datos
def visualize_data(ratings):
    # Distribución de las calificaciones
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings['rating'], bins=10, kde=False)
    plt.title('Distribución de las calificaciones')
    plt.xlabel('Calificación')
    plt.ylabel('Frecuencia')
    plt.show()

    # Número de calificaciones por usuario
    ratings_per_user = ratings.groupby('userId')['rating'].count()
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings_per_user, bins=30, kde=False)
    plt.title('Número de calificaciones por usuario')
    plt.xlabel('Número de calificaciones')
    plt.ylabel('Frecuencia')
    plt.show()

    # Número de calificaciones por película
    ratings_per_movie = ratings.groupby('movieId')['rating'].count()
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings_per_movie, bins=30, kde=False)
    plt.title('Número de calificaciones por película')
    plt.xlabel('Número de calificaciones')
    plt.ylabel('Frecuencia')
    plt.show()

# Limpieza de datos
def clean_data(ratings, min_ratings_per_user=10, min_ratings_per_movie=10):
    # Eliminar registros duplicados
    ratings = ratings.drop_duplicates()

    # Eliminar registros con valores nulos
    ratings = ratings.dropna()

    # Filtrar usuarios con menos de min_ratings_per_user calificaciones
    user_counts = ratings['userId'].value_counts()
    ratings = ratings[ratings['userId'].isin(user_counts[user_counts >= min_ratings_per_user].index)]

    # Filtrar películas con menos de min_ratings_per_movie calificaciones
    movie_counts = ratings['movieId'].value_counts()
    ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_ratings_per_movie].index)]

    return ratings

# Visualizar la dispersión de los datos originales
print("Visualización de la dispersión de los datos originales")
visualize_data(ratings)

# Limpiar los datos
cleaned_ratings = clean_data(ratings)

# Visualizar la dispersión de los datos limpiados
print("Visualización de la dispersión de los datos limpiados")
visualize_data(cleaned_ratings)

# Mostrar algunas estadísticas de los datos limpiados
print("Estadísticas de los datos limpiados:")
print(f"Número de usuarios: {cleaned_ratings['userId'].nunique()}")
print(f"Número de películas: {cleaned_ratings['movieId'].nunique()}")
print(f"Número de calificaciones: {cleaned_ratings.shape[0]}")