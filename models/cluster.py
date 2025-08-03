# cluster.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os

def cluster_jobs(job_tfidf, preprocessed_jobs, feature_names, output_dir, num_clusters=7, n_components=20):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    job_tfidf_reduced = pca.fit_transform(job_tfidf.toarray())
    print(f"PCA reduced dimensions to {n_components} components")
    
    wcss = []
    max_clusters = min(10, job_tfidf_reduced.shape[0])
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(job_tfidf_reduced)
        wcss.append(kmeans.inertia_)
    
    elbow_output = os.path.join(output_dir, 'elbow_results.csv')
    elbow_data = pd.DataFrame({'Clusters': range(1, max_clusters + 1), 'WCSS': wcss})
    elbow_data.to_csv(elbow_output, index=False)
    print(f"Elbow method results saved to {elbow_output}")
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(job_tfidf_reduced)
    print(f"Using k={num_clusters} clusters based on elbow plot analysis.")
    
    print("\nCluster Sizes:")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} jobs")
    
    job_df = preprocessed_jobs.copy()
    job_df['Cluster'] = cluster_labels
    job_df['Skills_Text_Display'] = job_df['Skills_Text'].apply(
        lambda x: (x[:100] + '...') if isinstance(x, str) and len(x) > 100 else x
    )
    
    cluster_names = []
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) == 0:
            cluster_names.append(f"Cluster {i} (empty)")
            continue
        cluster_tfidf = job_tfidf[cluster_indices].mean(axis=0).A1  # Convert to 1D array
        top_indices = np.argsort(cluster_tfidf)[::-1][:5]
        top_skills = [str(feature_names[idx]) for idx in top_indices]
        cluster_names.append(f"Cluster {i} ({', '.join(top_skills)})")
    
    job_df['Role'] = [cluster_names[label] for label in cluster_labels]
    
    clustered_output = os.path.join(output_dir, 'clustered_jobs.csv')
    job_df.to_csv(clustered_output, index=False)
    print(f"Clustered jobs saved to {clustered_output}")
    
    silhouette = None
    db_index = None
    if num_clusters > 1 and len(np.unique(cluster_labels)) > 1:
        try:
            silhouette = silhouette_score(job_tfidf_reduced, cluster_labels)
            db_index = davies_bouldin_score(job_tfidf_reduced, cluster_labels)
        except Exception as e:
            print(f"Error computing clustering metrics: {e}")
    
    return job_df, kmeans.cluster_centers_, cluster_names, silhouette, db_index, pca