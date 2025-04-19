import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

def fonction_partitions(partition_data, K, initial_centroids=None):
    df_fake = pd.DataFrame(partition_data)
    
    if initial_centroids is not None:
        centroids, labels = kmeans(df_fake, K=K, initial_centroids=initial_centroids, return_centroids_only=True)
    else:
        centroids, labels = kmeans(df_fake, K=K, return_centroids_only=True)

    medoids = []
    for i in range(K):
        cluster = partition_data[labels == i]
        if len(cluster) == 0:
            continue
        centroid = centroids[i]
        dists = np.linalg.norm(cluster - centroid, axis=1)
        medoid = cluster[np.argmin(dists)]
        medoids.append(medoid)

    return medoids

def hybride_distribue(df, K, n_partitions, initial_centers=None):
    data = df.select_dtypes(include=[float, int]).values
    partitions = np.array_split(data, n_partitions)

    results = [fonction_partitions(part, K, initial_centroids=initial_centers) for part in partitions]
    all_medoids = np.vstack(results)

    reduced_medoids_df = pd.DataFrame(all_medoids)

    if initial_centers is not None:
        reduced_centroids, reduced_labels = kmeans(reduced_medoids_df, K=K, initial_centroids=initial_centers, return_centroids_only=True)
    else:
        reduced_centroids, reduced_labels = kmeans(reduced_medoids_df, K=K, return_centroids_only=True)

    initial_medoids = []
    for i in range(K):
        cluster = all_medoids[reduced_labels == i]
        if len(cluster) == 0:
            continue
        centroid = reduced_centroids[i]
        dists = np.linalg.norm(cluster - centroid, axis=1)
        medoid = cluster[np.argmin(dists)]
        initial_medoids.append(medoid)
    initial_medoids = np.array(initial_medoids)

    distances = np.linalg.norm(data[:, np.newaxis] - initial_medoids, axis=2)
    medoid_indices = np.argmin(distances, axis=0)

    df_result = kmedoids(df.copy(), K=K, initial_medoids_indices=medoid_indices)
    df_result = df_result.rename(columns={'KMedoids_Labels': 'Hybrid_Distributed_Labels'})

    return df_result

def kmedoids(df, K, max_iter=100, initial_medoids_indices=None):
    data = df.select_dtypes(include=[float, int]).values
    N = len(data)

    if initial_medoids_indices is None or len(initial_medoids_indices) != K:
        raise ValueError("initial_medoids_indices must be a list of K valid indices.")

    medoid_indices = np.array(initial_medoids_indices)
    labels = np.zeros(N)

    for iteration in range(max_iter):
        distances = np.linalg.norm(data[:, np.newaxis] - data[medoid_indices], axis=2)
        labels = np.argmin(distances, axis=1)

        best_medoids = medoid_indices.copy()
        best_cost = np.sum([np.linalg.norm(data[i] - data[medoid_indices[labels[i]]]) for i in range(N)])
        improved = False

        for i in range(N):
            if i in medoid_indices:
                continue
            for j in range(K):
                new_medoids = medoid_indices.copy()
                new_medoids[j] = i
                new_distances = np.linalg.norm(data[:, np.newaxis] - data[new_medoids], axis=2)
                new_labels = np.argmin(new_distances, axis=1)
                new_cost = np.sum([np.linalg.norm(data[m] - data[new_medoids[new_labels[m]]]) for m in range(N)])
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_medoids = new_medoids
                    improved = True

        if not improved:
            break
        else:
            medoid_indices = best_medoids

    distances = np.linalg.norm(data[:, np.newaxis] - data[medoid_indices], axis=2)
    labels = np.argmin(distances, axis=1)

    df_result = df.copy()
    df_result['KMedoids_Labels'] = labels
    return df_result

def kmeans(df, K, return_centroids_only=False, initial_centroids=None):
    data = df.select_dtypes(include=[float, int]).values

    if initial_centroids is None or len(initial_centroids) != K:
        raise ValueError("initial_centroids must be a list or array of K centroid vectors.")

    centroids = np.array(initial_centroids)
    labels = -1 * np.ones(len(data))
    prev_labels = np.zeros(len(data))

    while not np.array_equal(labels, prev_labels):
        prev_labels = labels.copy()

        for i, point in enumerate(data):
            distances = np.linalg.norm(point - centroids, axis=1)
            labels[i] = np.argmin(distances)

        for i in range(K):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)

    if return_centroids_only:
        return centroids, labels
    else:
        df_result = df.copy()
        df_result['KMeans_Labels'] = labels.astype(int)
        return df_result

def accuracy(y_true, y_pred):
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)

    labels_matched = np.zeros_like(y_pred)
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        true_label_mode = mode(y_true_encoded[mask], keepdims=True).mode[0]
        labels_matched[mask] = true_label_mode

    return accuracy_score(y_true_encoded, labels_matched)