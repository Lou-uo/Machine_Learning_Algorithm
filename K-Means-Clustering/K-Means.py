import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_dataset(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, file_path)
    dataset = pd.read_csv(full_path)
    X = dataset.iloc[:, [2, 3]].values
    return X


def preprocess_data(X):
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    return X_scaled, sc


def train_kmeans(X_scaled, n_clusters=5, random_state=0):
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=random_state
    )
    kmeans.fit(X_scaled)
    return kmeans


def plot_clusters(X_scaled, kmeans, title):
    from matplotlib.colors import ListedColormap
    
    plt.figure(figsize=(10, 7))
    
    clusters = kmeans.predict(X_scaled)
    centroids = kmeans.cluster_centers_
    
    colors = ListedColormap(['red', 'blue', 'green', 'cyan', 'magenta'])
    
    for i in range(kmeans.n_clusters):
        plt.scatter(
            X_scaled[clusters == i, 0],
            X_scaled[clusters == i, 1],
            s=100,
            c=colors(i),
            label=f'Cluster {i+1}'
        )
    
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=300,
        c='yellow',
        marker='*',
        label='Centroids'
    )
    
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_elbow_method(X_scaled, max_clusters=10):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linewidth=2, markersize=8)
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    X = load_dataset('Social_Network_Ads.csv')
    
    X_scaled, sc = preprocess_data(X)
    
    print("=== Finding Optimal Number of Clusters (Elbow Method) ===")
    plot_elbow_method(X_scaled, max_clusters=10)
    
    print("\n=== Training K-Means with n_clusters=5 ===")
    kmeans = train_kmeans(X_scaled, n_clusters=5, random_state=0)
    
    print(f"Number of clusters: {kmeans.n_clusters}")
    print(f"WCSS (Inertia): {kmeans.inertia_:.2f}")
    
    print("\n=== Cluster Centers (Scaled Features) ===")
    print(kmeans.cluster_centers_)
    
    print("\n=== Visualizing Clusters ===")
    plot_clusters(X_scaled, kmeans, 'K-Means Clustering (Social Network Ads Dataset)')