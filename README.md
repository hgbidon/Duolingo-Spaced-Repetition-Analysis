# 🦉 Duolingo Spaced Repetition Analysis (Databricks)

## 📘 Overview
Explored Duolingo’s 13M+ spaced repetition dataset to uncover retention patterns, recall decay, and memory survival. Executed fully in **Databricks** using **Pandas workflows** and **KaggleHub** for dataset integration.

---

## ⚙️ Tech Stack
- **Environment:** Databricks (Pandas)  
- **Data Source:** Kaggle — *Duolingo Spaced Repetition Dataset*  
- **Libraries:** Pandas, Seaborn, Matplotlib, Lifelines, Scikit-Learn
- **Modeling:** Logistic Regression, RandomForest, Cox Proportional Hazards  
- **Visualization:** Seaborn, Plotly, Databricks inline display  

---

## 🧠 Key Steps

### 1️⃣ Data Loading
- Pulled dataset directly from Kaggle using **KaggleHub API**.  
- Cleaned and loaded over **12.8M learning traces** efficiently with Spark.

### 2️⃣ Exploratory Data Analysis
- Computed recall accuracy by token and lexeme.  
- Identified hardest words and tokens.  
- Correlated elapsed time with recall probability.

### 3️⃣ Modeling
- Built **Logistic Regression** and **RandomForest** models for correctness prediction.  
- Extracted feature importances — key predictors: `p_recall`, `session_correct`, `session_seen`.

### 4️⃣ Time-Based Recall Decay
- Computed **recall decay rates** over time gaps.  
- Visualized the Ebbinghaus-style forgetting curve.  
- Found rapid decay beyond 24 hours of inactivity.

### 5️⃣ Survival Analysis
- Used **Kaplan-Meier** and **Cox Proportional Hazards** models.  
- Quantified how time gaps, engagement, and past recall affect forgetting risk.

### 6️⃣ Retention Clustering
- Grouped users by **median retention time** (fast vs. slow forgetters).  
- Derived adaptive review insights for personalized learning.

---

## 🔍 Results
| Analysis | Insight |
|-----------|----------|
| Average Recall | >0.75 (strong retention) |
| Fast Forgetters | High risk after 6–12h without review |
| Slow Forgetters | Retain recall for 1–2+ days |
| Median Survival | ~24 hours before significant decay |

---

## 🧩 Project Value
- Scales to 13M+ rows using Databricks and Spark.  
- Demonstrates **survival analysis**, **clustering**, and **cognitive modeling**.  
- Bridges data science with applied learning research and spaced repetition theory.  

---

### 🏷️ Badges
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Databricks](https://img.shields.io/badge/Platform-Databricks-orange)
![Spark](https://img.shields.io/badge/BigData-Spark-red)
![Kaggle](https://img.shields.io/badge/Data-Kaggle-blueviolet)
![ScikitLearn](https://img.shields.io/badge/ML-ScikitLearn-yellow)
![Lifelines](https://img.shields.io/badge/Analysis-Survival-green)
