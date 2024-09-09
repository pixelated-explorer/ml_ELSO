# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# identify factors associated with need for renal replacement therapy (RRT) in 
# patients requiring ECMO


# NOTE
    # accuracy
    # precision
    # f1 score
        # considering the set is imbalanced. 
    # recall 
    # specificity 
    # and roc-AUC statistics 
        # putting together specificity and recall (also known as sensitivity)

    # most of the error should be for the regression style of analysis. 

    # create confusion matrix and you can calculate all of these. 

# %%

df = pd.read_csv('elsoOI_tempML.csv')

# %%

# Define features and target
features = df.drop(columns=['RRTduringECMO', 'Unnamed: 0', 'Mode_Other', 'Race_Other', 'Sex_Unknown']) 
X_data = features
y_data = df['RRTduringECMO']
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# %%
# XGBoost Hyperparameter Tuning (with GPU)
param_grid_xgb = {
    'max_depth': [3, 5, 7],
    # 'learning_rate': [0.1, 0.01, 0.001],
    'learning_rate' : [0.2, 0.1, 0.01],
    # 'subsample': [0.5, 0.7, 1],
    'subsample' : [0.8, 1.0],
    'n_estimators' : [100, 300],
    'tree_method': ['gpu_hist']  
}

xgb_model = xgb.XGBClassifier()

# Grid Search 
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train_data, y_train_data)

best_params_xgb = grid_search.best_params_


# Printing best parameters and the best score. 
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)


# grid_xgb = GridSearchCV(XGBClassifier(objective='binary:logistic', eval_metric='logloss', verbose_eval=10), 
#                         param_grid_xgb, cv=3, scoring='accuracy')
# grid_xgb.fit(X_train_data, y_train_data)
# best_params_xgb = grid_xgb.best_params_
# print("Best parameters for XGBoost:", best_params_xgb)

# %%
# Random Forest Hyperparameter Tuning (CPU)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=10, scoring='accuracy')
grid_rf.fit(X_train_data, y_train_data)
best_params_rf = grid_rf.best_params_
print("Best parameters for Random Forest:", best_params_rf)

# Logistic Regression Hyperparameter Tuning (CPU)
param_grid_lr = {
    'C': [0.1, 1.0, 10],  # Regularization strength
    'solver': ['liblinear', 'lbfgs']  # Solvers
}
grid_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=10, scoring='accuracy')
grid_lr.fit(X_train_data, y_train_data)
best_params_lr = grid_lr.best_params_
print("Best parameters for Logistic Regression:", best_params_lr)


# %%

# Initialize base models, including Naive Bayes
base_models = [
    ('xgb', XGBClassifier(objective='binary:logistic', eval_metric='logloss', **best_params_xgb)),
    ('rf', RandomForestClassifier(**best_params_rf, random_state=42)),
    ('lr', LogisticRegression(**best_params_lr, random_state=42)),  # Logistic Regression
    ('nb', GaussianNB())  # Naive Bayes
]

# Use XGBoost as the meta-model
meta_model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', tree_method='gpu_hist')

# Create stacking classifier with XGBoost as the meta-model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Fit the stacking model
stacking_model.fit(X_train_data, y_train_data)

# Predict and evaluate
y_pred_stack = stacking_model.predict(X_test_data)
accuracy_stack = accuracy_score(y_test_data, y_pred_stack)
print(f"Stacking Model Accuracy: {accuracy_stack:.4f}")

# %%

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, confusion_matrix

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# list to store each model
metrics_data = []

# Evaluation of individual models
for name, model in base_models:
    model.fit(X_train_data, y_train_data)
    y_pred = model.predict(X_test_data)


    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred)
    f1 = f1_score(y_test_data, y_pred)
    recall = recall_score(y_test_data, y_pred)
    specifity = calculate_specificity(y_test_data, y_pred)
    roc_auc = roc_auc_score(y_test_data, model.predict_proba(X_test_data)[:, 1])

    metrics_data.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'F1 Score': f1,
        'Recall': recall,
        'Specificity': specifity,
        'ROC-AUC': roc_auc
    })

# Evaluate the stacked model
y_pred_stack = stacking_model.predict(X_test_data)
accuracy_stack = accuracy_score(y_test_data, y_pred_stack)
precision_stack = precision_score(y_test_data, y_pred_stack)
f1_stack = f1_score(y_test_data, y_pred_stack)
recall_stack = recall_score(y_test_data, y_pred_stack)
specificity_stack = calculate_specificity(y_test_data, y_pred_stack)
roc_auc_stack = roc_auc_score(y_test_data, stacking_model.predict_proba(X_test_data)[:, 1])

# Append stacked model metrics
metrics_data.append({
    'Model': 'Stacked',
    'Accuracy': accuracy_stack,
    'Precision': precision_stack,
    'F1 Score': f1_stack,
    'Recall': recall_stack,
    'Specificity': specificity_stack,
    'ROC-AUC': roc_auc_stack
})


metrics_df = pd.DataFrame(metrics_data)

metrics_df.head()


# %%

lr_model = LogisticRegression(**best_params_lr, random_state=42)
lr_model.fit(X_train_data, y_train_data)

# Extract coefficients
coefficients = lr_model.coef_[0]

# Create a DataFrame to view feature importances
feature_importances = pd.DataFrame({
    'Feature': X_train_data.columns,
    'Coefficient': coefficients
})

# Sort features by absolute value of coefficients
feature_importances['AbsCoefficient'] = feature_importances['Coefficient'].abs()
feature_importances = feature_importances.sort_values(by='AbsCoefficient', ascending=False)

print("Feature Importances (Logistic Regression):")
print(feature_importances)

# %%

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(feature_importances['Feature'], feature_importances['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Feature Importances (Logistic Regression)')
plt.gca().invert_yaxis()
plt.show()

# %%

# # Base XGBoost model feature importances
# def plot_feature_importances(model, X):
#     booster = model.get_booster()  
#     importance = booster.get_score(importance_type='weight')
#     importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
#     importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
#     plt.figure(figsize=(10, 6))
#     plt.barh(importance_df['Feature'], importance_df['Importance'])
#     plt.xlabel('Importance')
#     plt.title('XGBoost Feature Importances')
#     plt.gca().invert_yaxis()
#     plt.show()

# Plotting for the base XGBoost model
print("Base XGBoost Model Feature Importances:")
plot_feature_importances(stacking_model.named_estimators_['xgb'], X_train_data)

# # Plotting for the XGBoost meta-model
# print("Meta XGBoost Model Feature Importances:")
# plot_feature_importances(stacking_model.final_estimator_, X_train_data)

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Fit the Naive Bayes model
nb_model = GaussianNB()  # or MultinomialNB()
nb_model.fit(X_train_data, y_train_data)

# Predict on test data
y_pred = nb_model.predict(X_test_data)

# Compute confusion matrix
cm = confusion_matrix(y_test_data, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# %%


from sklearn.metrics import roc_curve, roc_auc_score


for name, model in base_models:
    # Compute ROC curve
    print(f"Name of model is {name}")
    y_prob = model.predict_proba(X_test_data)[:, 1]  # Probability estimates for positive class
    fpr, tpr, thresholds = roc_curve(y_test_data, y_prob)
    roc_auc = roc_auc_score(y_test_data, y_prob)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# explicitly printing roc curve for stacked model

print("Stacked model:")
y_prob = stacking_model.predict_proba(X_test_data)[:, 1]  # Probability estimates for positive class
fpr, tpr, thresholds = roc_curve(y_test_data, y_prob)
roc_auc = roc_auc_score(y_test_data, y_prob)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# %%

from sklearn.metrics import precision_recall_curve, average_precision_score

# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test_data, y_prob)
average_precision = average_precision_score(y_test_data, y_prob)

# Plot Precision-Recall curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (average precision = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()


# %%

import seaborn as sns

# Plot feature distributions for each class
for feature in X_train_data.columns:
    plt.figure(figsize=(12, 6))
    sns.histplot(X_train_data.loc[y_train_data == 0, feature], color='blue', label='Class 0', kde=True)
    sns.histplot(X_train_data.loc[y_train_data == 1, feature], color='red', label='Class 1', kde=True)
    plt.title(f'Feature Distribution for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


