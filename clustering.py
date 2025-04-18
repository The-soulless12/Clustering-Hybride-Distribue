import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

def accuracy(y_true, y_pred):
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)

    labels_matched = np.zeros_like(y_pred)
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        true_label_mode = mode(y_true_encoded[mask], keepdims=True).mode[0]
        labels_matched[mask] = true_label_mode

    return accuracy_score(y_true_encoded, labels_matched)

def kmedoids(df, K=3, max_iter=100):
    data = df.select_dtypes(include=[float, int]).values
    N = len(data)

    medoid_indices = np.random.choice(N, K, replace=False)

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

    df['KMedoids_Labels'] = labels
    return df

def kmeans(df, K=3):
    data = df.select_dtypes(include=[float, int]).values

    random_indices = np.random.choice(len(data), size=K, replace=False)
    centroids = data[random_indices]

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

    df['KMeans_Labels'] = labels.astype(int)
    return df