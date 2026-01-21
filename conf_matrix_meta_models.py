import os
import joblib
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# Suppress warnings for cleaner output
warnings.simplefilter("ignore")

# Ensure Models folder exists
os.makedirs("Models", exist_ok=True)

# Load the dataset
df_comb = pd.read_csv("./Dataset/disease_symptom_combinations.csv")

# Feature matrix and label
X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0]

# Remove classes with fewer than 2 samples
class_counts = Y.value_counts()
valid_classes = class_counts[class_counts >= 2].index
X = X[Y.isin(valid_classes)]
Y = Y[Y.isin(valid_classes)]

print(f"Remaining classes after filtering: {len(valid_classes)}")

# Train-test split (now safe to stratify)
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.10, random_state=42, stratify=Y
)

# Base estimators
base_estimators = [
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Meta-models to evaluate
meta_models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Stratified cross-validation
cv = StratifiedKFold(n_splits=5)

# Create directory for confusion matrices
os.makedirs("Confusion_Matrices", exist_ok=True)

# Train and evaluate each stacking model
for name, meta_model in meta_models.items():
    print(f"\n🔹 Training StackingClassifier with {name} as final estimator...")

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model
    )

    # Train model
    stack.fit(x_train, y_train.to_numpy())

    # Predict
    predictions = stack.predict(x_test)

    # Compute metrics
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
    rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    print(f"\n📊 Performance Metrics for {name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Cross-validation accuracy
    cv_scores = cross_val_score(stack, X, Y, cv=cv)
    print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=y_test.unique())
    cm_df = pd.DataFrame(cm, index=y_test.unique(), columns=y_test.unique())

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save confusion matrix plot
    cm_filename = f"Confusion_Matrices/confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(cm_filename)
    plt.close()
    print(f"Confusion matrix saved as: {cm_filename}")

    # Classification report (optional detailed output)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Save trained model
    model_filename = f"Models/stacking_model_{name.replace(' ', '_').lower()}.joblib"
    joblib.dump(stack, model_filename)
    print(f"✅ Model saved as: {model_filename}")
