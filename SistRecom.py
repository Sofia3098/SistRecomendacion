import pandas as pd
import numpy as np
import time
import psutil
import platform
from sklearn.metrics.pairwise import cosine_similarity

from google.colab import drive
drive.mount('/content/drive/')

# Cargar los datos de películas y calificaciones
movies = pd.read_csv("/content/drive/MyDrive/RDI/PP/movies.csv")
ratings = pd.read_csv("/content/drive/MyDrive/RDI/PP/ratings.csv")

# Crear una tabla de usuario-película
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

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

# Limpiar los datos
cleaned_ratings = clean_data(ratings)

# Crear una tabla de usuario-película con los datos limpios
user_movie_matrix = cleaned_ratings.pivot(index='userId', columns='movieId', values='rating')

# Definir las funciones de similitud/distancia
def manhattan_distance(user1, user2):
    common_ratings = ~user1.isnull() & ~user2.isnull()
    if not common_ratings.any():
        return np.inf
    return np.sum(np.abs(user1[common_ratings] - user2[common_ratings]))

def euclidean_distance(user1, user2):
    common_ratings = ~user1.isnull() & ~user2.isnull()
    if not common_ratings.any():
        return np.inf
    return np.sqrt(np.sum((user1[common_ratings] - user2[common_ratings]) ** 2))

def pearson_correlation(user1, user2):
    common_ratings = ~user1.isnull() & ~user2.isnull()
    if not common_ratings.any():
        return 0
    return user1[common_ratings].corr(user2[common_ratings])

def cosine_similarity_user(user1, user2):
    user1_filled = user1.fillna(0)
    user2_filled = user2.fillna(0)
    dot_product = np.dot(user1_filled, user2_filled)
    norm_user1 = np.linalg.norm(user1_filled)
    norm_user2 = np.linalg.norm(user2_filled)
    if norm_user1 == 0 or norm_user2 == 0:
        return 0
    return dot_product / (norm_user1 * norm_user2)

# Función para encontrar los k vecinos más cercanos
def get_k_nearest_neighbors(user_id, k, similarity_func):
    distances = []
    user_ratings = user_movie_matrix.loc[user_id]
    for other_id in user_movie_matrix.index:
        if other_id == user_id:
            continue
        other_ratings = user_movie_matrix.loc[other_id]
        distance = similarity_func(user_ratings, other_ratings)
        distances.append((other_id, distance))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    return distances[:k]

# Función para predecir el rating basado en los k vecinos más cercanos
def predict_rating_knn(user_id, movie_id, k, similarity_func):
    neighbors = get_k_nearest_neighbors(user_id, k, similarity_func)
    ratings = []
    similarities = []
    for neighbor_id, similarity in neighbors:
        if pd.notna(user_movie_matrix.loc[neighbor_id, movie_id]):
            ratings.append(user_movie_matrix.loc[neighbor_id, movie_id])
            similarities.append(similarity)
    if not ratings:
        return np.nan
    weighted_sum = sum(r * s for r, s in zip(ratings, similarities))
    sum_of_similarities = sum(similarities)
    return weighted_sum / sum_of_similarities if sum_of_similarities != 0 else np.nan

# Solicitar al usuario el ID del usuario y el número de recomendaciones
user_id = int(input("Ingrese el ID del usuario: "))
num_recommendations = int(input("Ingrese el número de recomendaciones: "))

# Verificar que el ID del usuario y el número de recomendaciones sean válidos
if user_id not in user_movie_matrix.index:
    raise ValueError(f"El ID del usuario {user_id} no existe.")
if num_recommendations <= 0:
    raise ValueError("El número de recomendaciones debe ser mayor que 0.")

# Realizar las predicciones
similarity_funcs = {
    'Manhattan': manhattan_distance,
    'Euclidean': euclidean_distance,
    'Pearson': pearson_correlation,
    'Cosine': cosine_similarity_user
}

# Variables para medir tiempos
import time
start_total_time = time.time()

# Predicciones para cada tipo de similitud
for similarity_name, similarity_func in similarity_funcs.items():
    start_time = time.time()
    predictions = {}
    for movie_id in user_movie_matrix.columns:
        if pd.isna(user_movie_matrix.loc[user_id, movie_id]):
            predictions[movie_id] = predict_rating_knn(user_id, movie_id, 5, similarity_func)
    # Ordenar las predicciones y seleccionar las mejores
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    elapsed_time = time.time() - start_time
    print(f"\nRecomendaciones para el usuario {user_id} basadas en {similarity_name} (Tiempo: {elapsed_time:.2f} segundos):")
    for movie_id, predicted_rating in sorted_predictions:
        movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
        print(f"  {movie_title}: {predicted_rating:.2f}")

# Medir el tiempo total
total_time = time.time() - start_total_time
print(f"\nTiempo total para dar las recomendaciones: {total_time:.2f} segundos")

# Mostrar información de volumen de datos y memoria
import os, psutil
process = psutil.Process(os.getpid())
print(f"Volumen de datos analizados: {len(ratings)} calificaciones de {user_movie_matrix.index.nunique()} usuarios y {user_movie_matrix.columns.nunique()} películas")
print(f"Uso de memoria: {process.memory_info().rss / (1024 ** 2):.2f} MB")