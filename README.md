# Experiment 3: Logistic Regression for Classification

## Aim

To compute descriptive statistics and probabilities using NumPy and
Pandas.

## Algorithm

1.  **Start.**

2.  Load the dataset and extract features `X` and target `y`.

3.  Split the data into training and test sets (e.g., 80:20).

4.  Standardize features using `StandardScaler`.

5.  Train `LogisticRegression` on the training data.

    The logistic (sigmoid) function is:

    \[ `\sigma`{=tex}(z) = `\frac{1}{1 + e^{-z}}`{=tex} \]

    where

    \[ z = `\beta`{=tex}\_0 + `\beta`{=tex}\_1 x_1 + `\cdots `{=tex}+
    `\beta`{=tex}\_p x_p \]

6.  Predict labels on the test data.

7.  Evaluate performance using accuracy, confusion matrix, and
    classification report.

8.  Print the evaluation metrics.

9.  **Stop.**




## Program
    import numpy as np
    import pandas as pd
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.metrics import classification_report
    
    # Create dataset
    data = {
        'Age': [22, 25, 28, 30, 35, 40, 45, 50, 55, 60],
        'Salary': [15000, 18000, 22000, 25000, 30000,
                   35000, 40000, 45000, 50000, 55000],
        'Purchased': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }
    
    df = pd.DataFrame(data)
    
    # Descriptive statistics
    print("Descriptive Statistics:")
    print(df.describe())
    
    # Extract features and target
    X = df[['Age', 'Salary']]
    y = df['Purchased']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Predicted probabilities
    print("Predicted Probabilities:")
    print(model.predict_proba(X_test))


## output

    Descriptive Statistics:
                Age        Salary  Purchased
    count  10.00000     10.000000  10.000000
    mean   39.00000  33500.000000   0.600000
    std    13.15717  13786.063817   0.516398
    min    22.00000  15000.000000   0.000000
    25%    28.50000  22750.000000   0.000000
    50%    37.50000  32500.000000   1.000000
    75%    48.75000  43750.000000   1.000000
    max    60.00000  55000.000000   1.000000
    Accuracy: 1.0
    Confusion Matrix:
    [[1 0]
     [0 1]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00         1
               1       1.00      1.00      1.00         1
    
        accuracy                           1.00         2
       macro avg       1.00      1.00      1.00         2
    weighted avg       1.00      1.00      1.00         2
    
    Predicted Probabilities:
    [[0.03460423 0.96539577]
     [0.77986333 0.22013667]]

## Result
Thus, descriptive statistics and probability-based classification were performed using NumPy, Pandas and Logistic Regression. The model was evaluated using accuracy, confusion matrix, classification report and predicted probabilities.
