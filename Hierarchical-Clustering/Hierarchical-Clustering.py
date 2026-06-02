import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


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


def plot_dendrogram(X_scaled, title='Dendrogram'):
    plt.figure(figsize=(14, 8))

    linked = linkage(X_scaled, method='ward')

    dendrogram(
        linked,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        leaf_rotation=90,
        leaf_font_size=8
    )

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index / (Cluster Size)')
    plt.ylabel('Distance (Ward)')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_hierarchical_clustering(X_scaled, n_clusters=5, linkage='ward'):
    hc = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    hc.fit(X_scaled)
    return hc


def plot_clusters(X_scaled, hc, title):
    plt.figure(figsize=(10, 7))

    clusters = hc.labels_

    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'purple', 'orange']

    for i in range(hc.n_clusters):
        plt.scatter(
            X_scaled[clusters == i, 0],
            X_scaled[clusters == i, 1],
            s=100,
            c=colors[i % len(colors)],
            label=f'Cluster {i+1}'
        )

    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    X = load_dataset('Social_Network_Ads.csv')

    X_scaled, sc = preprocess_data(X)

    print("=== Hierarchical Clustering Dendrogram ===")
    print("Use this to determine the optimal number of clusters")
    plot_dendrogram(X_scaled)

    print("\n=== Training Hierarchical Clustering with n_clusters=5 ===")
    hc = train_hierarchical_clustering(X_scaled, n_clusters=5)

    print(f"Number of clusters: {hc.n_clusters}")

    print("\n=== Visualizing Clusters ===")
    plot_clusters(X_scaled, hc, 'Hierarchical Clustering (Social Network Ads Dataset)')