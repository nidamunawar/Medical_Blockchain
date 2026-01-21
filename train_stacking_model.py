import os
import joblib
import warnings
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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

#  Remove classes with fewer than 2 samples (required for stratified splitting)
class_counts = Y.value_counts()
valid_classes = class_counts[class_counts >= 2].index
X = X[Y.isin(valid_classes)]
Y = Y[Y.isin(valid_classes)]

print(f" Remaining classes after filtering: {len(valid_classes)}")

# Train-test split (now safe to stratify)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42, stratify=Y)

# Base estimators
base_estimators = [
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Meta-models to evaluate
meta_models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Stratified cross-validation
cv = StratifiedKFold(n_splits=5)

# Train and evaluate each stacking model
for name, meta_model in meta_models.items():
    print(f"\n Training StackingClassifier with {name} as final estimator...")

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model
    )

    stack.fit(x_train, y_train.to_numpy())

    # Evaluate on test set
    predictions = stack.predict(x_test)
    test_accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy (Stacking with {name}): {test_accuracy:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(stack, X, Y, cv=cv)
    print(f"Cross-Validation Accuracy (Stacking with {name}): {cv_scores.mean() * 100:.2f}%")

    # Save model
    model_filename = f"Models/stacking_model_{name.replace(' ', '_').lower()}.joblib"
    joblib.dump(stack, model_filename)
    print(f"Model saved as: {model_filename}")