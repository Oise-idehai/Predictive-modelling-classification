# Wine Cultivar Classification

An end-to-end machine learning classification project that predicts **wine cultivar** from chemical measurements.

## Project Objective

This project follows:

**Analysis before prediction → preprocessing → model building → hyperparameter tuning → prediction → analysis after prediction**

The notebook includes exploratory analysis, a Scikit-learn Pipeline, a parameter grid, `GridSearchCV`, the explicit `grid.fit(X_train, y_train)` step, prediction, evaluation metrics, confusion-matrix analysis and feature-importance analysis.

## Dataset

- **File:** `wine_classification.csv`
- **Problem:** Multi-class classification
- **Target:** `target`

## 1. Analysis Before Prediction

Before modelling, the project checks dataset shape, missing values, duplicates, class distribution, descriptive statistics and feature behaviour.

### Class distribution

![Class distribution](screenshots/charts/01_class_distribution.png)

### Feature ranges

![Feature ranges](screenshots/charts/02_feature_ranges.png)

The chemical variables have different numerical ranges. The exploratory analysis therefore informs the preprocessing/model choices.

### Feature association

![Feature association](screenshots/charts/03_feature_association.png)

The analysis shows that several chemical measurements contain useful information for distinguishing cultivars.

## 2. Train/Test Split

The data is split using:

```python
test_size=0.20
random_state=42
stratify=y
```

Stratification helps preserve the class proportions in the training and test sets.

## 3. Classification Pipeline

![Classification Pipeline](screenshots/code/02_pipeline.png)

The model is placed inside a Pipeline so the workflow remains reproducible and preprocessing/model fitting are handled consistently.

## 4. Parameter Grid and GridSearchCV

![Parameter Grid](screenshots/code/03_param_grid.png)

The Random Forest parameter grid tests:

```python
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5]
}
```

The search uses 5-fold stratified cross-validation and accuracy as the scoring metric.

## 5. The Important `grid.fit()` Step

![GridSearchCV and grid.fit](screenshots/code/04_gridsearch_fit.png)

The GridSearchCV object must be fitted before its best parameters or predictions can be used:

```python
grid.fit(X_train, y_train)
```

After fitting, the project obtains:

```python
grid.best_params_
grid.best_score_
grid.best_estimator_
```

## 6. Prediction

![Prediction](screenshots/code/05_prediction.png)

The best fitted model is selected and used on the unseen test set:

```python
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
```

## 7. Analysis After Prediction

The classifier is evaluated with:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification report
- Confusion matrix

![Evaluation code](screenshots/code/06_evaluation.png)

### Model performance

![Model performance](screenshots/charts/04_model_performance.png)



## 8. Confusion Matrix

The confusion matrix in the notebook examines correct and incorrect predictions for each cultivar. This gives more detail than accuracy alone and helps identify which classes are being confused.

## 9. Feature Importance

The fitted Random Forest model is also analysed using feature importance. This helps identify the chemical variables that contributed most to the model's predictive decisions.

**Important:** feature importance indicates predictive usefulness; it does not prove causation.

## 10. Key Insights

1. Exploratory analysis is performed before prediction.
2. Class distribution is checked before modelling.
3. A stratified 80/20 train/test split is used.
4. A Pipeline keeps the modelling workflow reproducible.
5. `GridSearchCV` tests multiple Random Forest configurations.
6. `grid.fit(X_train, y_train)` explicitly fits the hyperparameter search.
7. The best estimator is used for prediction on unseen test data.
8. Accuracy, precision, recall and F1 provide complementary evaluation.
9. Confusion-matrix analysis reveals class-specific errors.
10. Feature importance helps interpret the classifier.

## 11. Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 12. Repository Structure

```text
wine-classification/
├── wine_classification.csv
├── Wine_Classification_Project.ipynb
├── Wine_Classification_Results.csv
├── README.md
└── screenshots/
    ├── code/
    │   ├── 01_load_data.png
    │   ├── 02_pipeline.png
    │   ├── 03_param_grid.png
    │   ├── 04_gridsearch_fit.png
    │   ├── 05_prediction.png
    │   └── 06_evaluation.png
    └── charts/
        ├── 01_class_distribution.png
        ├── 02_feature_ranges.png
        ├── 03_feature_association.png
        └── 04_model_performance.png
```

## 13. How to Run

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook
```

Open `Wine_Classification_Project.ipynb` and run the cells from top to bottom.

## Conclusion

This project demonstrates a complete classification workflow: understanding the data before prediction, building a reproducible Pipeline, tuning the Random Forest with GridSearchCV, explicitly fitting the search with `grid.fit()`, generating predictions, and analysing model performance after prediction.

## Author

**Oisesenagha Blessing Idehai**

First Class Petroleum Engineering Graduate | Data Science & Machine Learning Enthusiast
