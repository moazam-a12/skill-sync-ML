# gap_analysis.py
import pandas as pd
import numpy as np
from tabulate import tabulate
import os

def analyze_skill_gaps(intern_matches, preprocessed_interns, job_df, preprocessed_courses, feature_names, job_tfidf, output_dir, num_rows=5):
    gap_results = pd.DataFrame({
        'Intern ID': intern_matches['Intern ID'],
        'Top Cluster': intern_matches['Top Cluster'],
        'Top Cluster Role': intern_matches['Top Cluster Role']
    })
    
    def extract_skills(text):
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
            return set()
        return set(skill.strip() for skill in text.split(','))
    
    def truncate_list(items, max_items=3):
        if not items:
            return ''
        items = sorted(items)
        if len(items) > max_items:
            return ', '.join(items[:max_items]) + '...'
        return ', '.join(items)
    
    num_clusters = intern_matches['Top Cluster'].max() + 1
    cluster_skills = {i: set() for i in range(num_clusters)}
    
    for cluster in range(num_clusters):
        cluster_indices = job_df[job_df['Cluster'] == cluster].index
        if len(cluster_indices) == 0:
            continue
        cluster_tfidf = job_tfidf[cluster_indices].mean(axis=0).A1  # Convert to 1D array
        top_indices = np.argsort(cluster_tfidf)[::-1][:3]  # Limit to 3 skills
        top_skills = [str(feature_names[idx]) for idx in top_indices]
        cluster_skills[cluster] = set(top_skills)
    
    missing_skills_list = []
    for idx, row in intern_matches.iterrows():
        intern_id = row['Intern ID']
        cluster = row['Top Cluster']
        intern_skills = extract_skills(preprocessed_interns[preprocessed_interns['Intern ID'] == intern_id]['Skills_Text'].iloc[0])
        job_skills = cluster_skills[cluster]
        missing_skills = job_skills - intern_skills
        missing_skills_list.append(', '.join(sorted(missing_skills)) if missing_skills else '')
    
    gap_results['Missing Skills'] = missing_skills_list
    
    recommended_courses_list = []
    for idx, row in gap_results.iterrows():
        intern_id = row['Intern ID']
        missing_skills = extract_skills(row['Missing Skills'])
        if not missing_skills:
            recommended_courses_list.append('')
            continue
        course_scores = []
        for _, course_row in preprocessed_courses.iterrows():
            course_name = course_row['course']
            if 'Google Cybersecurity' in course_name:  # Filter out generic course
                continue
            course_skills = extract_skills(course_row['Skills_Text'])
            overlap = len(course_skills & missing_skills)
            if overlap > 0:
                similarity = intern_matches[(intern_matches['Intern ID'] == intern_id) & 
                                          (intern_matches['Top Course'] == course_name)]['Course Similarity Score']
                similarity = similarity.iloc[0] if not similarity.empty else 0.0
                course_scores.append((course_name, overlap, similarity))
        course_scores.sort(key=lambda x: (-x[1], -x[2]))
        relevant_courses = [course[0] for course in course_scores[:3]]
        recommended_courses_list.append(', '.join(relevant_courses) if relevant_courses else 'No relevant courses found')
    
    gap_results['Recommended Courses'] = recommended_courses_list
    
    gap_output = os.path.join(output_dir, 'intern_skill_gaps.csv')
    gap_results.to_csv(gap_output, index=False)
    print(f'Skill gap analysis saved to {gap_output}')
    
    display_results = gap_results.copy()
    display_results['Missing Skills'] = display_results['Missing Skills'].apply(lambda x: truncate_list(x.split(', ') if x else []))
    display_results['Recommended Courses'] = display_results['Recommended Courses'].apply(lambda x: truncate_list(x.split(', ') if x else []))
    
    num_rows = min(num_rows, len(display_results)) if num_rows > 0 else len(display_results)
    print(f'Sample Skill Gap Analysis (first {num_rows} rows):')
    print(tabulate(display_results.head(num_rows), 
                   headers=['Intern ID', 'Top Cluster', 'Top Cluster Role', 'Missing Skills', 'Recommended Courses'], 
                   tablefmt='grid', stralign='left', maxcolwidths=[15, 10, 50, 50, 50]))
    
    return gap_results