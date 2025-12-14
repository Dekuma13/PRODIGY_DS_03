import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import joblib
sns.set(style="whitegrid")

df = pd.read_csv("bank.csv", sep=';')

y_raw = df['y'].copy()
le_y = LabelEncoder()
y = le_y.fit_transform(y_raw)

X = df.drop('y', axis=1)
obj_cols = X.select_dtypes(include=['object']).columns.tolist()
if len(obj_cols) > 0:
    X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

base_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(base_model, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
cv_scores_acc = cross_val_score(best, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
cv_scores_roc = cross_val_score(best, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

y_pred = best.predict(X_test)
if hasattr(best, "predict_proba"):
    y_proba = best.predict_proba(X_test)[:, 1]
else:
    y_proba = best.predict(X_test)

acc = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)

print("Best params:", grid.best_params_)
print("CV accuracy (train fold): mean {:.4f} std {:.4f}".format(cv_scores_acc.mean(), cv_scores_acc.std()))
print("CV ROC-AUC (train fold): mean {:.4f} std {:.4f}".format(cv_scores_roc.mean(), cv_scores_roc.std()))
print("Test Accuracy: {:.4f}".format(acc))
print("Test ROC AUC: {:.4f}".format(auc_score))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_y.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_y.classes_, yticklabels=le_y.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png", bbox_inches='tight', dpi=200)
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png", bbox_inches='tight', dpi=200)
plt.show()

feat_imp = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 20 feature importances:\n", feat_imp.head(20))
feat_imp.head(20).plot(kind='bar', figsize=(10,5))
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=200)
plt.show()

plt.figure(figsize=(20,12))
plot_tree(best, feature_names=X.columns, class_names=le_y.classes_, filled=True, rounded=True, fontsize=10)
plt.savefig("decision_tree.png", dpi=200, bbox_inches='tight')
plt.show()

joblib.dump(best, "decision_tree_bank_model.joblib")
X_train.to_csv("X_train_processed.csv", index=False)
X_test.to_csv("X_test_processed.csv", index=False)
pd.Series(y_train).to_csv("y_train.csv", index=False)
pd.Series(y_test).to_csv("y_test.csv", index=False)
print("All outputs saved: model, plots, processed splits")
