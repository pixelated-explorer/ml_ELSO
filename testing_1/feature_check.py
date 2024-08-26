# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from sklearn.metrics import make_scorer
from xgboost import XGBClassifier
import shap 

# %%

# look into hiearchical clustering and k-medoids 

# for later, just use .isin() method for col list that you act want 

# Load dataset
# NOTE: changing between a different csv
# df = pd.read_csv('ELSOcleaner.csv')
df = pd.read_csv('elso_clusters.csv')
# %% 
df['HypoTension'] = (df['normalSBP'] - df['SBP']) / df['normalSBP']

# %%

# for later, just use .isin() method for col list that you act want 

# Load dataset
df = pd.read_csv('ELSOcleaner.csv')

# Casting and Encoding
non_binary_or_numeric = ['Race', 'Mode', 'SupType', 'PreRRT', 'PreECMOAKI', 'ID', 'PrimaryDiagnosis', 'Sex', 'Discontinuation', 'SurvECMO', 'SurvHosp', 'VADetc']
for col in non_binary_or_numeric:
    df[col] = df[col].astype('category')

# One-Hot Encoding for categorical features
df_encoded = pd.get_dummies(df, columns=['Race', 'Mode', 'SupType'], drop_first=True)

# Label Encoding for binary features
label_columns = ['CDH', 'PreRRT', 'PreECMOAKI']
for col in label_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])

# Define features and target
features = df_encoded.drop(columns=['RRTduringECMO', 'Sex', 'Discontinuation', 'SurvECMO', 'SurvHosp', 'VADetc', 'ID', 'PrimaryDiagnosis', 'RRTtimePostCann', 'Pressors', 'YearECLS', 'NMB', 'OI', 'LOSdays', 'HoursECMO', 'VentDur', 'HypoTerm', 'iNO', 'CPR', 'normalSBP', 'CultPos', 'Hem', 'AgeDays', 'normalSBP', 'SBP', 'Wt']) # put Hem back at some point
X_data = features
y_data = df['RRTduringECMO']
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# %%

# Define hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.5, 0.7, 1]
}

# Define custom scoring function
def xgb_accuracy_scorer(y_true, y_pred_proba):
    y_pred = (y_pred_proba > 0.5).astype(int)
    return accuracy_score(y_true, y_pred)

# Initialize XGBClassifier
xgb_model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

# Initialize GridSearchCV
# split data into 10 folds, and train model on 9 folds. test on remaining 1 fold. 
# repeat process 10 times
grid_search = GridSearchCV(xgb_model, param_grid, cv=10, scoring=make_scorer(xgb_accuracy_scorer), verbose=10)

# Fit GridSearchCV to the training data
grid_search.fit(X_train_data, y_train_data)

# Print the best parameters and best score
print("Best set of hyperparameters:", grid_search.best_params_)
print("Best score for data:", grid_search.best_score_)

# %%
# Convert to DMatrix
dtrain = xgb.DMatrix(X_train_data, label=y_train_data, enable_categorical=True)
dtest = xgb.DMatrix(X_test_data, label=y_test_data, enable_categorical=True)

# %%
# Extract best hyperparameters from GridSearchCV
best_params = grid_search.best_params_
params = {
    'objective': 'binary:logistic',
    'max_depth': best_params['max_depth'],
    'learning_rate': best_params['learning_rate'],
    'subsample': best_params['subsample'],
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

# Number of boosting rounds
num_round = 1000
early_stopping_rounds = 10

# Train the model
bst = xgb.train(params, dtrain, num_round, evals=[(dtest, 'eval')], early_stopping_rounds = early_stopping_rounds, verbose_eval=10)

# Predict and evaluate
y_pred = bst.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test_data, y_pred_binary)
print(f"Final accuracy: {accuracy:.4f}")

# %%

# %%

# attempt to implement SHAP feature importance plots 

def plot_feature_importances(model, X):
    if model:
        booster = model  # Booster object returned by xgb.train
        feature_names = X.columns
        importance = booster.get_score(importance_type='weight')
        importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Weight'])
        importance_df = importance_df.sort_values(by='Weight', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Weight'])
        plt.xlabel('Weight')
        plt.title('Feature Importances')
        plt.gca().invert_yaxis()
        plt.show()
    else:
        print("Model does not have feature importances attribute.")

print("Plotting feature importances for final model:")
plot_feature_importances(bst, X_data)

#%%
import shap
explainer = shap.TreeExplainer(bst)
shap_value = explainer.shap_values(X_test_data)
shap_values = explainer.shap_values(X_test_data)
shap.summary_plot(shap_values, X_test_data, plot_type="bar")

