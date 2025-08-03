# preprocess.py
import pandas as pd
import re
import os

def preprocess_data(intern_file, job_file, course_file, output_dir):
    # Define a comprehensive skills list
    skills_list = [
        'python', 'r', 'sql', 'java', 'javascript', 'c++', 'c', 'scala', 'matlab',
        'machine learning', 'deep learning', 'nlp', 'computer vision', 'data science',
        'data analytics', 'data visualization', 'big data analytics', 'statistical modeling',
        'aws', 'azure', 'gcp', 'cloud computing', 'docker', 'kubernetes', 'ml-ops',
        'pandas', 'numpy', 'scikit learn', 'tensorflow', 'pytorch', 'spark', 'hadoop',
        'tableau', 'power bi', 'ms-excel', 'ms-word', 'ms-powerpoint', 'mysql', 'mongodb',
        'databases', 'git', 'github', 'agile software development', 'software engineering',
        'web development', 'angularjs', 'node.js', 'reactjs', 'typescript', 'css', 'html',
        'rest api', 'selenium', 'postman', 'linux', 'feature engineering', 'data wrangling',
        'databricks', 'diffusion models', 'large language models', 'transformers',
        'vision language models', 'distributed training algorithms', 'distributed frameworks',
        'neural networks', 'agentic frameworks', 'adobe photoshop', 'adobe illustrator',
        'adobe after effects', 'adobe xd', 'canva', 'figma', 'firebase', 'database testing'
    ]
    
    # Map proficiency levels for interns
    proficiency_map = {0: 'none', 1: 'low', 2: 'medium', 3: 'high'}
    
    # Process intern data
    interns = pd.read_csv(intern_file)
    
    # Combine scored skills with proficiency, handle NaN
    interns['Python_Skill'] = interns['Python (out of 3)'].fillna(0).map(proficiency_map).apply(lambda x: f'python {x}' if x != 'none' else '')
    interns['ML_Skill'] = interns['Machine Learning (out of 3)'].fillna(0).map(proficiency_map).apply(lambda x: f'machine learning {x}' if x != 'none' else '')
    interns['NLP_Skill'] = interns['Natural Language Processing (NLP) (out of 3)'].fillna(0).map(proficiency_map).apply(lambda x: f'nlp {x}' if x != 'none' else '')
    interns['DL_Skill'] = interns['Deep Learning (out of 3)'].fillna(0).map(proficiency_map).apply(lambda x: f'deep learning {x}' if x != 'none' else '')
    
    # Handle NaN in Other skills
    interns['Other skills'] = interns['Other skills'].fillna('')
    
    # Combine and deduplicate skills, prioritizing proficiency-based skills
    scored_skills = {'python', 'machine learning', 'nlp', 'deep learning'}
    def combine_skills(row):
        # Get proficiency-based skills
        prof_skills = [s for s in [row['Python_Skill'], row['ML_Skill'], row['NLP_Skill'], row['DL_Skill']] if s and s != 'none']
        # Get base skill names from proficiency-based skills (e.g., 'python' from 'python medium')
        prof_base_skills = {s.split()[0] for s in prof_skills}
        # Get other skills, excluding any that match base proficiency skills entirely
        other_skills = [s.strip() for s in row['Other skills'].split(',') if s.strip() and s.lower() not in prof_base_skills]
        # Combine, deduplicate, and sort
        all_skills = sorted(set(prof_skills + other_skills) - {''})
        return ', '.join(all_skills).lower()
    
    interns['Skills_Text'] = interns.apply(combine_skills, axis=1)
    
    # Select relevant columns
    interns_output = pd.DataFrame({
        'Intern ID': interns['Intern ID'],
        'Skills_Text': interns['Skills_Text']
    })
    intern_output_path = os.path.join(output_dir, 'preprocessed_interns.csv')
    interns_output.to_csv(intern_output_path, index=False)
    
    # Process job data
    def extract_skills(text, skill_list):
        if pd.isna(text):
            return ''
        text = str(text).lower()
        extracted_skills = []
        for skill in skill_list:
            if skill.lower() in text:
                extracted_skills.append(skill)
        return ', '.join(sorted(set(extracted_skills))) if extracted_skills else ''
    
    jobs = pd.read_csv(job_file)
    jobs['Skills_Text'] = jobs['job_description_text'].apply(lambda x: extract_skills(x, skills_list))
    jobs['Skills_Text'] = jobs['Skills_Text'].fillna('')  # Ensure no NaN
    jobs_output = pd.DataFrame({
        'job_title': jobs['job_title'],
        'Skills_Text': jobs['Skills_Text']
    })
    jobs_output_path = os.path.join(output_dir, 'preprocessed_jobs.csv')
    jobs_output.to_csv(jobs_output_path, index=False)
    
    # Process course data
    courses = pd.read_csv(course_file)
    courses['Skills_Text'] = courses['skills'].apply(lambda x: extract_skills(x, skills_list))
    courses['Skills_Text'] = courses['Skills_Text'].fillna('')  # Ensure no NaN
    courses_output = pd.DataFrame({
        'course': courses['course'],
        'Skills_Text': courses['Skills_Text']
    })
    courses_output_path = os.path.join(output_dir, 'preprocessed_courses.csv')
    courses_output.to_csv(courses_output_path, index=False)
    
    return intern_output_path, jobs_output_path, courses_output_path