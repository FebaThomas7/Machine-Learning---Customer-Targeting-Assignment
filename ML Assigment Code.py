import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from google.colab import files
uploaded = files.upload()

# Load CSV
data = pd.read_csv("wallacecommunications.csv")

# Drop 'ID' if it exists
if 'ID' in data.columns:
    data = data.drop(columns=['ID'])

print(data.head())
print("Columns:", data.columns)
print("Missing values per column:\n", data.isnull().sum())
if data['new_contract_this_campaign'].dtype == object:
    data['new_contract_this_campaign'] = data['new_contract_this_campaign'].map({'yes': 1, 'no': 0})

y = data['new_contract_this_campaign']
X = data.drop(columns=['new_contract_this_campaign'])

print("Target distribution:\n", y.value_counts())
X = pd.get_dummies(X, drop_first=False)

# fill missing values
X = X.fillna(X.mean())

# Scale features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("Sample of scaled features:\n", X_scaled.head())

# Split temp/test
X_temp, X_test, y_temp, y_test = train_test_split (X_scaled, y, test_size=0.20, random_state=42, stratify=y)

# Split train/val
X_train, X_val, y_train, y_val = train_test_split (X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)
print("Test size:", X_test.shape)

# Define models
models = {"GradientBoosting": GradientBoostingClassifier(random_state=42),"NaiveBayes": GaussianNB(),"NeuralNetwork": MLPClassifier(hidden_layer_sizes=(64, 32),activation='relu', max_iter=500, random_state=42)}
print("Models defined:", list(models.keys()))
for name, model in models.items():
# Train model
    model.fit(X_train, y_train)
    
# Predict on validation set
    y_pred_val = model.predict(X_val)
    y_prob_val = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    
# Validation AUC
    auc_val = roc_auc_score(y_val, y_prob_val) if y_prob_val is not None else None
    print(f"{name} Validation AUC:", round(auc_val, 4) if auc_val is not None else "N/A")
    
# Confusion matrix
    cm = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {name}')
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks([0,1], ['No','Yes'])
    plt.yticks([0,1], ['No','Yes'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center', color='red', fontsize=14)
    plt.tight_layout()
    plt.show()
    
# Feature importance for Gradient Boosting
    if name == "GradientBoosting":
        feat_imp = pd.Series(model.feature_importances_, index=X_scaled.columns)
        feat_imp = feat_imp.sort_values(ascending=False).head(10)
        feat_imp[::-1].plot(kind='barh', title=f'Top 10 Features - {name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
# Select best model based on validation AUC
val_scores = {}
for name, model in models.items():
    y_prob_val = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    auc_val = roc_auc_score(y_val, y_prob_val) if y_prob_val is not None else 0
    val_scores[name] = (auc_val, model)

best_name, (best_auc, best_model) = max(val_scores.items(), key=lambda k: k[1][0])

# Evaluate on test set
y_prob_test = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
y_pred_test = best_model.predict(X_test)

test_auc = roc_auc_score(y_test, y_prob_test) if y_prob_test is not None else None
test_acc = accuracy_score(y_test, y_pred_test)

print("\nBest model on validation:", best_name)
print("Test AUC:", round(test_auc, 4) if test_auc is not None else "N/A")
print("Test Accuracy:", round(test_acc, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=['No', 'Yes']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - Test Set ({best_name})')
plt.colorbar()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks([0,1], ['No','Yes'])
plt.yticks([0,1], ['No','Yes'])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i,j]), ha='center', va='center', color='red', fontsize=14)
plt.tight_layout()
plt.show()

# Initialize a list to store results
results = []
for name, model in models.items():
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Append to results
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4)})

# Create a DataFrame from the model evaluation results and display it
results_df = pd.DataFrame(results)
print("\nModel Performance Table:")
print(results_df)




















        
















