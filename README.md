# Skill Gap Analysis Tool

## Overview
The **Skill Gap Analysis Tool** is a sophisticated Python-based machine learning pipeline designed to empower organizations by matching internship applicants to job roles, identifying skill deficiencies, and recommending tailored Coursera courses to bridge those gaps. Leveraging natural language processing, clustering, and similarity metrics, this tool delivers precise, actionable insights for workforce development with an elegant and efficient design.

## Purpose
The tool achieves three core objectives:
1. **Match Interns to Jobs**: Aligns interns with job roles that best fit their skill profiles.
2. **Identify Skill Gaps**: Highlights skills interns need to acquire for their assigned roles.
3. **Recommend Courses**: Suggests relevant Coursera courses to address skill deficiencies.

## Data Inputs
The tool processes three datasets:
- **ML_internship_Applicants.csv**: Intern profiles with skills (e.g., `python, java, sql, python low`).
- **1000_ml_jobs_us.csv**: Job postings with required skills (e.g., `data science, r, scala`).
- **Coursera.csv**: Coursera courses with associated skills (e.g., `IBM Data Science: python, data science`).

## Project Structure
The project is modular and streamlined, orchestrated through a Jupyter Notebook with supporting Python modules:
- **main.ipynb**: Drives the pipeline across four cells:
  - **Cell 1**: Installs dependencies and imports modules.
  - **Cell 2**: Preprocesses datasets and displays samples.
  - **Cell 3**: Vectorizes skills, clusters jobs, and matches interns.
  - **Cell 4**: Analyzes skill gaps and recommends courses.
- **models/**:
  - `preprocess.py`: Standardizes skills data.
  - `vectorize.py`: Converts skills to TF-IDF vectors.
  - `cluster.py`: Groups jobs using K-Means and PCA.
  - `match.py`: Matches interns to job clusters and courses.
  - `gap_analysis.py`: Identifies missing skills and suggests courses.
- **utils/**: Stores output files (CSVs, PCA model).
- **data/**: Contains input CSVs.
- **requirements.txt**: Lists dependencies (`pandas`, `numpy`, `scikit-learn`, `tabulate`, `matplotlib`, `pickle`).

## Workflow
The pipeline operates in four cohesive stages via `main.ipynb`:

### 1. Setup and Dependencies (Cell 1)
- Installs required packages and imports modules from `models` and `utils`.
- Configures the environment with the parent directory `/Users/moazam_a12/Skill Gap Analysis Tool`.

### 2. Data Preprocessing (Cell 2, `preprocess.py`)
- Loads and cleans input CSVs, creating `Skills_Text` columns (comma-separated skills).
- Handles missing values and standardizes skill formats.
- Saves preprocessed data to `utils` (e.g., `preprocessed_interns.csv`).
- Displays sample data with truncated skills (up to 100 characters) for clarity.
- **Output Example**:
  ```
  Preprocessed Intern Data: 100 rows, 2 columns
  Sample Intern Data (first 5 rows):
  | Intern ID   | Skills_Text                                    |
  |-------------|------------------------------------------------|
  | intern_0001 | deep learning, ms-excel, ms-word, mysql, ...   |
  ```

### 3. Vectorization, Clustering, and Matching (Cell 3, `vectorize.py`, `cluster.py`, `match.py`)
- **Vectorization (`vectorize.py`)**:
  - Converts `Skills_Text` into TF-IDF vectors using `TfidfVectorizer` with a custom tokenizer (splits on commas).
  - Combines skills from all datasets for a consistent vocabulary.
  - Outputs: `intern_tfidf`, `job_tfidf`, `course_tfidf` (e.g., 997 jobs × 367 skills), and `feature_names` (367 skills).
- **Clustering (`cluster.py`)**:
  - Applies PCA to reduce `job_tfidf` to 20 components.
  - Uses K-Means to group jobs into 7 clusters based on skill similarity.
  - Names clusters using the top 3 skills by TF-IDF score (e.g., `Cluster 1 (data science, r, scala)`).
  - Saves `clustered_jobs.csv` with job titles, skills, clusters, and roles.
  - Generates `elbow_results.csv` for cluster evaluation.
- **Matching (`match.py`)**:
  - Transforms intern and course TF-IDF vectors to PCA space.
  - Computes cosine similarity to match interns to job clusters and courses.
  - Saves `intern_matches.csv` with `Intern ID`, `Top Cluster`, `Top Cluster Role`, `Job Similarity Score`, `Top Course`, and `Course Similarity Score`.
  - Displays a formatted table of matches.

### 4. Skill Gap Analysis (Cell 4, `gap_analysis.py`)
- Selects the top 3 skills per cluster using TF-IDF scores.
- Compares intern skills to cluster skills to identify `Missing Skills`.
- Recommends up to 3 courses based on skill overlap and similarity scores, excluding generic courses like `Google Cybersecurity`.
- Displays a table for `num_rows` (default 5) interns, truncating `Missing Skills` and `Recommended Courses` to 3 items.
- Saves full results to `intern_skill_gaps.csv`.
- **Output Example**:
  ```
  Sample Skill Gap Analysis (first 5 rows):
  | Intern ID   | Top Cluster | Top Cluster Role       | Missing Skills         | Recommended Courses                     |
  |-------------|-------------|------------------------|------------------------|----------------------------------------|
  | intern_0001 | 1           | Cluster 1 (data science, r, scala) | data science, r, scala | IBM Data Science, Tools for Data Science, Python for Data Science |
  ```

## Key Features
- **Customizable Display**: `num_rows` parameter in Cell 4 controls the number of interns displayed (e.g., 5, 10, or 0 for all).
- **Clean Output**: Terminal tables limit `Missing Skills` and `Recommended Courses` to 3 items, with full data in CSVs.
- **Relevant Recommendations**: Filters out generic courses and prioritizes those with high skill overlap.
- **Efficient Clustering**: Uses PCA (20 components) and K-Means (7 clusters) for robust job grouping.
- **Scalable Architecture**: Modular design supports easy integration of new datasets or parameters.

## Current State
As of August 3, 2025, the tool is fully operational:
- **Performance**: Processes 997 jobs, multiple interns, and courses, producing balanced clusters (e.g., Cluster 1: 266 jobs, Cluster 0: 83 jobs).
- **Outputs**:
  - `clustered_jobs.csv`: Jobs with clusters and roles.
  - `intern_matches.csv`: Intern-job and intern-course matches with similarity scores.
  - `intern_skill_gaps.csv`: Skill gaps and course recommendations.
  - `elbow_results.csv`: Clustering evaluation data.
  - `pca_model.pkl`: Saved PCA model for consistent transformations.
- **Customization**: Adjustable `num_rows`, `num_clusters` (default 7), and `n_components` (default 20) in Cell 3.
- **Output Quality**: Terminal displays are concise, with truncated fields and targeted course recommendations.

## Dependencies
- `pandas`: Data manipulation.
- `numpy`: Numerical operations.
- `scikit-learn`: TF-IDF, PCA, K-Means, cosine similarity.
- `tabulate`: Formatted table output.
- `matplotlib`: Elbow plot visualization.
- `pickle`: PCA model serialization.

## Author's Note
Dishing out ML magic with sizzling skill matches! 
