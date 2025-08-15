# RateMyProfessor Data Science Capstone

**Author:** Beckett Newton  
**Course:** Principles of Data Science, NYU  
**Semester:** Fall 2024  
**Instructor:** Prof. Pascal Wallisch

## Overview
This project analyzes a large RateMyProfessor dataset (89,000+ professors) to investigate patterns, biases, and predictors in student evaluations.  
It was completed as the final capstone for NYU’s *Principles of Data Science* course, combining statistical analysis, machine learning, and data visualization.

Key objectives:
- Assess claims of gender bias in ratings.
- Examine the impact of teaching experience on ratings.
- Explore the relationship between course difficulty and evaluations.
- Evaluate the role of online teaching.
- Investigate “hotness” bias in ratings.
- Build predictive models for both ratings and “hotness.”

## Data Cleaning & Preprocessing
- Addressed **86% missing data** using targeted removal of rows with >1 NaN.
- Retained professors with any number of ratings to avoid bias toward more experienced faculty.
- Seeded random number generation with my NYU N-number for reproducibility.

## Analyses Performed
1. **Gender Bias in Ratings:**  
   Z-test revealed a statistically significant but small pro-male bias (3.88 vs. 3.82 average ratings).

2. **Experience vs. Teaching Quality:**  
   ANOVA across quartiles showed a modest upward trend in ratings with more ratings (proxy for experience).

3. **Rating vs. Difficulty:**  
   Pearson correlation = −0.54 (moderate negative relationship).  
   Students rate harder courses lower.

4. **Online Teaching:**  
   Bootstrapped comparison showed slightly lower ratings for professors with 5+ online classes, but not statistically significant at α=0.005.

5. **Rating vs. Retake Proportion:**  
   Very strong positive correlation (r = 0.88).  
   Retake proportion emerged as a major predictor in later models.

6. **Hotness Bias:**  
   Professors with “pepper” averaged 4.38 vs. 3.58 for others — a large and significant difference.

7. **Predictive Models:**
   - **Rating from Difficulty Only:** R² ≈ 0.29, slope = −0.61.
   - **Full Rating Model (All Factors):** R² ≈ 0.81 — strongest predictors were *proportion retake* and *hotness*.
   - **Hotness Classification:** Logistic regression with all features achieved 68% accuracy and AUROC = 0.76.

8. **Extra Credit — Hotness by Major:**  
   Significant variation in “pepper” rates across majors, with English and Psychology highest.

## Methods & Tools
- **Languages:** Python
- **Libraries:** Pandas, NumPy, SciPy, scikit-learn, Matplotlib
- **Stats Methods:** Z-tests, ANOVA, Pearson correlation, Linear Regression, Logistic Regression, Bootstrapping
- **ML Models:** Simple & multiple linear regression, classification models
- **Validation:** Train/test split, R², RMSE, AUROC, precision/recall/F1

## Repository Contents
- `captstone.py` – Python scripts for cleaning, analysis, and modeling.
- `rmpCapstoneNum/Qual.csv` – Input datasets.
- `RateMyProfessor_Report.pdf` – Full written report.
- `README.md` – Project summary.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/username/ratemyprofessor-capstone.git
   cd ratemyprofessor-capstone
