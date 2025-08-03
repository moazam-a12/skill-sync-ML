# match.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tabulate import tabulate
import os

def match_interns(intern_tfidf, job_centroids, course_tfidf, preprocessed_interns, preprocessed_courses, feature_names, job_tfidf, cluster_names, output_dir, pca):
    intern_tfidf_reduced = pca.transform(intern_tfidf.toarray())
    course_tfidf_reduced = pca.transform(course_tfidf.toarray())
    
    job_similarities = cosine_similarity(intern_tfidf_reduced, job_centroids)
    top_clusters = np.argmax(job_similarities, axis=1)
    job_scores = np.max(job_similarities, axis=1)
    
    course_similarities = cosine_similarity(intern_tfidf_reduced, course_tfidf_reduced)
    top_courses = np.argmax(course_similarities, axis=1)
    course_scores = np.max(course_similarities, axis=1)
    
    matches = pd.DataFrame({
        'Intern ID': preprocessed_interns['Intern ID'],
        'Top Cluster': top_clusters,
        'Top Cluster Role': [cluster_names[i] for i in top_clusters],
        'Job Similarity Score': job_scores,
        'Top Course': preprocessed_courses['course'].iloc[top_courses].values,
        'Course Similarity Score': course_scores
    })
    
    matches_output = os.path.join(output_dir, 'intern_matches.csv')
    matches.to_csv(matches_output, index=False)
    print(f'Intern matches saved to {matches_output}')
    print('Sample Intern Matches (first 5 rows):')
    print(tabulate(matches.head(), 
                   headers=['Intern ID', 'Top Cluster', 'Top Cluster Role', 'Job Similarity Score', 'Top Course', 'Course Similarity Score'], 
                   tablefmt='grid', stralign='left', maxcolwidths=[15, 10, 50, 15, 35, 15]))
    
    return matches