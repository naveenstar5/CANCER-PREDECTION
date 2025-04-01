import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('path_to_your_file/cancer_dataset.csv')

print(data.head())

print("\nData Info:")
print(data.info())

print("\nDescriptive Statistics:")
print(data.describe())

print("\nMissing values:")
print(data.isnull().sum())

sns.countplot(data['target'])
plt.title("Target Variable Distribution")
plt.show()

correlation = data.drop(columns=['target']).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

data.drop(columns=['target']).hist(bins=20, figsize=(14, 10))
plt.suptitle("Histograms of Features")
plt.show()

sns.pairplot(data, hue="target", vars=data.columns[:-1], palette="coolwarm")
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

X = data.drop(columns=['target'])
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importance, y=features)
plt.title("Feature Importance from Random Forest Classifier")
plt.show()

train_accuracy = model.score(X_train, y_train)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")

y_train_pred = model.predict(X_train)
train_cm = confusion_matrix(y_train, y_train_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title("Confusion Matrix for Training Data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()