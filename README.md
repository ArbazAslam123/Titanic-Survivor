Titanic Survival Prediction Project

Overview

This project focuses on predicting the survival chances of passengers aboard the Titanic using machine learning techniques. The dataset used is the classic Titanic dataset, which includes passenger information such as age, gender, class, fare, cabin, and port of embarkation. The objective of this project is to preprocess the data, build predictive models, and evaluate their performance in order to understand the factors that influenced survival rates.

Dataset

The dataset contains the following important features:

Survived: Target variable (0 = Not Survived, 1 = Survived)

Pclass: Passenger class (1st, 2nd, 3rd)

Sex: Gender of the passenger

Age: Age in years

SibSp: Number of siblings or spouses aboard

Parch: Number of parents or children aboard

Fare: Ticket fare paid

Cabin: Cabin assigned (simplified in preprocessing)

Embarked: Port of embarkation (C, Q, S)

Columns such as Name, Ticket and PassengerId were excluded from the analysis since they do not contribute significantly to the predictive model.

Preprocessing Steps

Handling Missing Values

Filled missing numerical values with the median.

Encoded missing categorical values appropriately.

Encoding Categorical Variables

Converted the Sex column into binary values (0 for male, 1 for female).

Encoded Embarked and simplified Cabin using Label Encoding.

Scaling

Applied MinMaxScaler to normalize numerical features for consistent model training.

Feature Selection

Dropped irrelevant columns like Name, Ticket and PassengerId.

Used remaining features for survival prediction.

Model Training

The primary model used in this project was Logistic Regression, which is suitable for binary classification problems such as predicting survival. The dataset was split into training and testing sets (80% train, 20% test).

Additional preprocessing ensured that the model could effectively handle both categorical and numerical data.

Model Evaluation

The Logistic Regression model achieved the following performance on the test data:

Accuracy: 82%

Confusion Matrix:

True Negatives: 91

False Positives: 14

False Negatives: 18

True Positives: 56

Classification Report:

Precision (Not Survived): 83%

Recall (Not Survived): 87%

Precision (Survived): 80%

Recall (Survived): 76%

The results indicate that the model is effective at predicting survival, though it performs slightly better at identifying non-survivors than survivors. This is expected due to class imbalance in the dataset.

Key Learnings

Logistic Regression provides a strong baseline for binary classification tasks.

Feature preprocessing, especially encoding and scaling, is essential for accurate predictions.

The Titanic dataset highlights the importance of factors such as gender, passenger class, and fare in survival chances.

Evaluation metrics beyond accuracy, such as precision, recall, and the confusion matrix, provide deeper insights into model performance.

Future Improvements

Experiment with more advanced models such as Random Forest, XGBoost, or Support Vector Machines.

Perform hyperparameter tuning to improve prediction accuracy.

Address class imbalance by using techniques like oversampling or assigning class weights.

Conduct feature engineering to extract more meaningful features (for example, family size from SibSp and Parch).
