import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("aggregated_dataset.csv")

# Features and labels
X = df.drop("label", axis=1)

y = df["label"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train model
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective="multi:softprob",
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nClassification Report\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

print("\nConfusion Matrix\n")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# Feature importance
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance\n")
print(importance)