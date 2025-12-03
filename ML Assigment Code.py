import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from google.colab import files
uploaded = files.upload()

# Load CSV
data = pd.read_csv("wallacecommunications.csv")

# Drop 'customer_id' if it exists
if 'customer_id' in data.columns:
    data = data.drop(columns=['customer_id'])

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














