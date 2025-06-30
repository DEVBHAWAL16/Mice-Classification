#  Mice Classification Based on Protein Expression Levels


## Project Overview

This project focuses on using **machine learning models** to classify mice based on their **protein expression levels** from brain cortex nucleus experiments. The main goal is to distinguish between **Control** and **Treated (Ts65Dn)** mice by analyzing multiple protein biomarkers.  

The project includes:  
- Exploratory Data Analysis (EDA)  
- Data Preprocessing  
- Model Building and Evaluation  
- Feature Importance Analysis  
- Visualization of Results  


## Dataset Description

- **File:** `Data_Cortex_Nuclear.csv`
- **Records:** 1080 mouse samples
- **Features:**
  - 77+ columns for **protein expression measurements**
  - Metadata columns:
    - `MouseID`
    - `Genotype` (Target: Control or Ts65Dn)
    - `Treatment`
    - `Behavior`
    - `class`

- **Target Variable:**  
  - **Genotype:**  
    - `"Control"` (Healthy mice)  
    - `"Ts65Dn"` (Treated/Experimental mice)


## Project Workflow

### 1. Data Loading & Inspection
- Loaded the CSV dataset into a pandas DataFrame.
- Checked dataset shape, column types, and missing values.
- Reviewed target class distribution.

### 2. Exploratory Data Analysis (EDA)
- Plotted class distribution (Control vs. Treated mice).
- Visualized protein expression distributions.
- Created correlation heatmaps to find relationships between proteins.
- Boxplots comparing protein levels between groups.

### 3. Data Preprocessing
- Removed or imputed missing values.
- Dropped unnecessary metadata columns (e.g., MouseID).
- Encoded target labels (Control = 0, Ts65Dn = 1).
- Applied **StandardScaler** for feature scaling.
- Optional PCA (Principal Component Analysis) for dimensionality reduction.

### 4. Model Building
Machine learning models used:

- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

Process:
- Split dataset into **train** and **test** sets (e.g., 80/20 split).
- Trained models on training data.
- Predicted on test data.

### 5. Model Evaluation
For each model, calculated:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **ROC-AUC Score**
- Plotted **ROC curves**

### 6. Feature Importance
- Used Random Forest’s `feature_importances_` to identify key proteins contributing most to classification.
- Ranked top proteins based on importance scores.
- Plotted feature importance bar charts.

### 7. Results Summary
- Compared model performances side by side.
- Highlighted which model gave the highest accuracy and AUC.
- Observed that certain proteins consistently appeared as top predictors.

## Libraries Used

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

### Install all with:
pip install pandas numpy matplotlib seaborn scikit-learn


### How to Run the Project
git clone https://github.com/DEVBHAWAL16/Mice-Classification.git
cd Mice-Classification


### Install Dependencies:
pip install -r requirements.txt
##### OR manually:
pip install pandas numpy matplotlib seaborn scikit-learn

### Launch Jupyter Notebook:
jupyter notebook

### Run the Notebook:
Open and run the file:
Classification of Mice Based on Protein Expression Levels.ipynb

### Expected Outputs:

- Data overview and cleaning steps.

- Visualizations showing class distributions and feature relationships.

- Model performance metrics.

- Confusion matrices.

- Feature importance plots.

- ROC Curves for model comparison.

### Future Work / Improvements

- Perform hyperparameter tuning using GridSearchCV.

- Implement Cross-Validation for better generalization.

- Explore advanced classifiers like XGBoost or LightGBM.



- Apply SMOTE for handling class imbalance if needed.

- Create an interactive dashboard (using Streamlit) to visualize results.


