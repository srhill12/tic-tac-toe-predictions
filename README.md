# Tic-Tac-Toe Endgame Database

This project involves analyzing and predicting outcomes of Tic-Tac-Toe endgames using various machine learning models. The dataset contains all possible board configurations at the end of Tic-Tac-Toe games, with the goal of determining if 'X' wins.

## Dataset Information

- **Title**: Tic-Tac-Toe Endgame database
- **Creator**: David W. Aha
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame)
- **Date**: 19 August 1991
- **Instances**: 958
- **Attributes**: 9 board positions (x, o, b) and class (positive, negative)

## Attributes

- **top-left-square**: {x, o, b}
- **top-middle-square**: {x, o, b}
- **top-right-square**: {x, o, b}
- **middle-left-square**: {x, o, b}
- **middle-middle-square**: {x, o, b}
- **middle-right-square**: {x, o, b}
- **bottom-left-square**: {x, o, b}
- **bottom-middle-square**: {x, o, b}
- **bottom-right-square**: {x, o, b}
- **Class**: {positive, negative} (65.3% positive)

## Project Steps

### 1. Data Import and Preprocessing

# Import required dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Import data
file_path = "https://static.bc-edx.com/ai/ail-v-1-0/m13/lesson_3/datasets/tic-tac-toe.csv"
df = pd.read_csv(file_path)

# Get the target variable and encode it
le = LabelEncoder()
y = le.fit_transform(df["Class"])

# Get the features and encode them
X = df.drop(columns="Class")
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype='int')
X_encoded = pd.DataFrame(data=ohe.fit_transform(X), columns=ohe.get_feature_names_out())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)

### 2. Model Training and Evaluation
# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Train the logistic regression model
lr_model = LogisticRegression(random_state=1)
lr_model.fit(X_train, y_train)

# Evaluate the model
print('Train Accuracy: %.3f' % lr_model.score(X_train, y_train))
print('Test Accuracy: %.3f' % lr_model.score(X_test, y_test))

# Support Vector Machine
from sklearn.svm import SVC

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluate the model
print('Train Accuracy: %.3f' % svm_model.score(X_train, y_train))
print('Test Accuracy: %.3f' % svm_model.score(X_test, y_test))

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Evaluate the model
print('Train Accuracy: %.3f' % knn_model.score(X_train, y_train))
print('Test Accuracy: %.3f' % knn_model.score(X_test, y_test))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Train the decision tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Evaluate the model
print('Train Accuracy: %.3f' % dt_model.score(X_train, y_train))
print('Test Accuracy: %.3f' % dt_model.score(X_test, y_test))

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Train the random forest model
rf_model = RandomForestClassifier(n_estimators=128, random_state=1)
rf_model.fit(X_train, y_train)

# Evaluate the model
print('Train Accuracy: %.3f' % rf_model.score(X_train, y_train))
print('Test Accuracy: %.3f' % rf_model.score(X_test, y_test))

# Results
Logistic Regression: Train Accuracy: 0.986, Test Accuracy: 0.975
Support Vector Machine: Train Accuracy: 0.986, Test Accuracy: 0.975
K-Nearest Neighbors: Train Accuracy: 0.947, Test Accuracy: 0.942
Decision Tree: Train Accuracy: 1.000, Test Accuracy: 0.929
Random Forest: Train Accuracy: 1.000, Test Accuracy: 0.992

# Conclusion
The Random Forest classifier provided the best performance with the highest test accuracy, followed by Logistic Regression and Support Vector Machine.
