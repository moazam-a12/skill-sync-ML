# vectorize.py
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def custom_tokenizer(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    return [skill.strip() for skill in text.split(',')]

def vectorize_skills(preprocessed_interns, preprocessed_jobs, preprocessed_courses):
    all_texts = pd.concat([
        preprocessed_interns['Skills_Text'].fillna(''),
        preprocessed_jobs['Skills_Text'].fillna(''),
        preprocessed_courses['Skills_Text'].fillna('')
    ], ignore_index=True)
    
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False, token_pattern=None)
    vectorizer.fit(all_texts)
    
    intern_tfidf = vectorizer.transform(preprocessed_interns['Skills_Text'].fillna(''))
    job_tfidf = vectorizer.transform(preprocessed_jobs['Skills_Text'].fillna(''))
    course_tfidf = vectorizer.transform(preprocessed_courses['Skills_Text'].fillna(''))
    
    feature_names = list(vectorizer.get_feature_names_out())
    
    return intern_tfidf, job_tfidf, course_tfidf, feature_names