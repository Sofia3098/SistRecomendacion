import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, find
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, ConstantInputWarning
import time
import psutil
import warnings

from google.colab import drive
drive.mount('/content/drive/')

# Cargar los datos de películas y calificaciones
movies = pd.read_csv("/content/drive/MyDrive/RDI/PP/movies.csv")
ratings = pd.read_csv("/content/drive/MyDrive/RDI/PP/ratings.csv")

# Limpiar datos
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

# Limpiar datos
cleaned_ratings = clean_data(ratings)

# Ajustar los índices de usuario y película para que comiencen en 0
user_mapper = {user_id: i for i, user_id in enumerate(cleaned_ratings['userId'].unique())}
movie_mapper = {movie_id: i for i, movie_id in enumerate(cleaned_ratings['movieId'].unique())}
cleaned_ratings['user_index'] = cleaned_ratings['userId'].map(user_mapper)
cleaned_ratings['movie_index'] = cleaned_ratings['movieId'].map(movie_mapper)

# Crear matriz dispersa usuario-película
def create_sparse_matrix(ratings):
    row = ratings['user_index']
    col = ratings['movie_index']
    data = ratings['rating']
    sparse_matrix = csr_matrix((data, (row, col)))
    return sparse_matrix

user_movie_sparse_matrix = create_sparse_matrix(cleaned_ratings)

# Funciones de similitud/distancia con matrices dispersas
def manhattan_distance_sparse(user1, user2):
    return cdist(user1.toarray().reshape(1, -1), user2.toarray().reshape(1, -1), metric='cityblock')[0][0]

def euclidean_distance_sparse(user1, user2):
    return cdist(user1.toarray().reshape(1, -1), user2.toarray().reshape(1, -1), metric='euclidean')[0][0]

def pearson_correlation_sparse(user1, user2):
    user1 = user1.toarray().flatten()
    user2 = user2.toarray().flatten()
    mask = (user1 != 0) & (user2 != 0)
    if np.sum(mask) < 2:
        return 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConstantInputWarning)
            return pearsonr(user1[mask], user2[mask])[0]
    except ConstantInputWarning:
        return 0

def cosine_similarity_sparse(user1, user2):
    return cosine_similarity(user1.toarray().reshape(1, -1), user2.toarray().reshape(1, -1))[0][0]

# Función para encontrar los k vecinos más cercanos
def get_k_nearest_neighbors_sparse(user_id, k, similarity_func):
    distances = []
    if user_id >= user_movie_sparse_matrix.shape[0]:
        return distances
    user_ratings = user_movie_sparse_matrix[user_id]
    for other_id in range(user_movie_sparse_matrix.shape[0]):
        if other_id == user_id:
            continue
        other_ratings = user_movie_sparse_matrix[other_id]
        distance = similarity_func(user_ratings, other_ratings)
        distances.append((other_id, distance))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    return distances[:k]

# Función para predecir el rating basado en los k vecinos más cercanos
def predict_rating_knn_sparse(user_id, movie_id, k, similarity_func):
    predictions = {}
    if user_id >= user_movie_sparse_matrix.shape[0] or movie_id >= user_movie_sparse_matrix.shape[1]:
        return predictions
    neighbors = get_k_nearest_neighbors_sparse(user_id, k, similarity_func)
    ratings = []
    similarities = []
    for neighbor_id, similarity in neighbors:
        rating = user_movie_sparse_matrix[neighbor_id, movie_id]
        if rating != 0:
            ratings.append(rating)
            similarities.append(similarity)
    if not ratings:
        return predictions
    weighted_sum = sum(r * s for r, s in zip(ratings, similarities))
    sum_of_similarities = sum(similarities)
    predicted_rating = weighted_sum / sum_of_similarities if sum_of_similarities != 0 else np.nan
    predictions[movie_id] = predicted_rating
    return predictions

# Solicitar al usuario el ID del usuario y el número de recomendaciones
user_id = int(input("Ingrese el ID del usuario: "))
num_recommendations = int(input("Ingrese el número de recomendaciones: "))

# Convertir el ID del usuario al índice del usuario
user_index = user_mapper.get(user_id, None)
if user_index is None:
    raise ValueError(f"El ID del usuario {user_id} no existe.")

# Verificar que el número de recomendaciones sea válido
if num_recommendations <= 0:
    raise ValueError("El número de recomendaciones debe ser mayor que 0.")

# Realizar las predicciones
similarity_funcs_sparse = {
    'Manhattan': manhattan_distance_sparse,
    'Euclidean': euclidean_distance_sparse,
    'Pearson': pearson_correlation_sparse,
    'Cosine': cosine_similarity_sparse
}

# Variables para medir tiempos
start_total_time = time.time()

# Predicciones para cada tipo de similitud dispersa
for similarity_name, similarity_func in similarity_funcs_sparse.items():
    start_time = time.time()
    predictions = {}
    for movie_index in range(user_movie_sparse_matrix.shape[1]):
        if movie_index not in user_movie_sparse_matrix[user_index].indices:
            predictions.update(predict_rating_knn_sparse(user_index, movie_index, 5, similarity_func))
    # Ordenar las predicciones y seleccionar las mejores
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    elapsed_time = time.time() - start_time
    print(f"\nRecomendaciones para el usuario {user_id} basadas en {similarity_name} (Tiempo: {elapsed_time:.2f} segundos):")
    for movie_index, predicted_rating in sorted_predictions:
        movie_id = list(movie_mapper.keys())[list(movie_mapper.values()).index(movie_index)]
        movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
        print(f"  {movie_title}: {predicted_rating:.2f}")

# Medir el tiempo total
total_time = time.time() - start_total_time
print(f"\nTiempo total para dar las recomendaciones: {total_time:.2f} segundos")

# Mostrar información de volumen de datos y memoria
process = psutil.Process(os.getpid())
print(f"Volumen de datos analizados: {len(cleaned_ratings)} calificaciones de {user_movie_sparse_matrix.shape[0]} usuarios y {user_movie_sparse_matrix.shape[1]} películas")
print(f"Uso de memoria: {process.memory_info().rss / (1024 ** 2):.2f} MB")
