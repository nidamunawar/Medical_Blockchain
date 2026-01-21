import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import warnings
import joblib

warnings.filterwarnings("ignore")

# ✅ Load dataset
df_norm = pd.read_csv("Dataset/disease_symptom_matrix.csv")

# The first column is 'label_dis' (disease names)
X = df_norm.iloc[:, 1:]
y = df_norm.iloc[:, 0]

# ✅ Step 1: Remove rare diseases (< 2 samples)
class_counts = y.value_counts()
rare_classes = class_counts[class_counts < 2].index
df_norm = df_norm[~df_norm['label_dis'].isin(rare_classes)]
print(f"🩺 Removed {len(rare_classes)} rare diseases with <2 samples.")

# ✅ Update X and y after cleaning
X = df_norm.iloc[:, 1:]
y = df_norm.iloc[:, 0]

# ✅ Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ✅ Step 2: Split data (stratified, now safe)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
)

# ✅ Step 3: Define Gradient Boosting and parameter grid
gb_model = GradientBoostingClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("🚀 Starting Gradient Boosting optimization...")
grid_search.fit(X_train, y_train)

# ✅ Best model
best_gb = grid_search.best_estimator_
print("\n🌟 Best Parameters:", grid_search.best_params_)

# ✅ Evaluate model
y_pred = best_gb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# ✅ Cross-validation accuracy
cv_acc = cross_val_score(best_gb, X, y_encoded, cv=cv, scoring='accuracy').mean()

# ✅ Display results
print("\n📊 Optimized Gradient Boosting Performance")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Cross-Validation Accuracy: {cv_acc * 100:.2f}%")

# ✅ Save the optimized model for reuse in your ensemble
joblib.dump(best_gb, "Models/optimized_gradient_boosting.joblib")
print("✅ Optimized Gradient Boosting model saved successfully!")
