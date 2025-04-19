import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import multiprocessing as mp
mp.set_start_method("fork", force=True)


df= pd.read_csv('dataset_phishing.csv')
df

sns.countplot(data=df,x='domain_with_copyright',hue='status')

sns.histplot(data=df,x='page_rank',hue='status',multiple='stack') 

for col in ["url","length_url","length_hostname","ip","nb_dots","nb_hyphens","nb_at","nb_qm","nb_and","nb_or","nb_eq","nb_underscore","nb_tilde","nb_percent","nb_slash","nb_star","nb_colon","nb_comma","nb_semicolumn","nb_dollar","nb_space","nb_www","nb_com","nb_dslash","http_in_path","https_token","ratio_digits_url","ratio_digits_host","punycode","port","tld_in_path","tld_in_subdomain","abnormal_subdomain","nb_subdomains","prefix_suffix","random_domain","shortening_service","path_extension","nb_redirection","nb_external_redirection","length_words_raw","char_repeat","shortest_words_raw","shortest_word_host","shortest_word_path","longest_words_raw","longest_word_host","longest_word_path","avg_words_raw","avg_word_host","avg_word_path","phish_hints","domain_in_brand","brand_in_subdomain","brand_in_path","suspecious_tld","statistical_report","nb_hyperlinks","ratio_intHyperlinks","ratio_extHyperlinks","ratio_nullHyperlinks","nb_extCSS","ratio_intRedirection","ratio_extRedirection","ratio_intErrors","ratio_extErrors","login_form","external_favicon","links_in_tags","submit_email","ratio_intMedia","ratio_extMedia","sfh","iframe","popup_window","safe_anchor","onmouseover","right_clic","empty_title","domain_in_title","domain_with_copyright","whois_registered_domain","domain_registration_length","domain_age","web_traffic","dns_record","google_index","page_rank","status"]:
    unique_vals = list(df[col].unique())
    for idx in range(len(unique_vals)):
        df[col]=df[col].replace([unique_vals[idx]],idx) #remember to put [uni..[],idx]
df.shape

df.dtypes

df.isnull().sum()

sns.countplot(df.status)
print(df.status.value_counts())

z = np.abs(stats.zscore(df))
data_clean = df[(z<3).all(axis = 1)]
data_clean.shape

#sns.heatmap(data_clean.corr(), fmt='.2g')

x=df.drop('status',axis=1)
y=df.status

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) # ensures same set of data is used for splitting

def ModelEval(model, x_train, x_test, y_train, y_test):
    print(f"\nEvaluating model: {model.__class__.__name__}\n")
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    FEATURE_NAMES = list(x_train.columns)
    print(FEATURE_NAMES)


    print("Accuracy Score: ", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("F1-Score: ", f1_score(y_test, y_pred))
    print("Precision Score: ", precision_score(y_test, y_pred))
    print("Recall Score: ", recall_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(data=cm, linewidths=.5, annot=True, square=True, cmap='Blues')
    all_sample_title = "Accuracy Score: {0}".format(model.score(x_test, y_test) * 100)
    plt.title(all_sample_title, fontsize=12)
    plt.xlabel('Actual label')
    plt.ylabel('Predicted label')
    plt.savefig(f"{model.__class__.__name__}_confusion_matrix.png")
    plt.close()

    print("Calculating MAE, MSE, R2, RMSE...")

    y_pred_sample = model.predict(x_test)
    mae = metrics.mean_absolute_error(y_test, y_pred_sample)
    mse = metrics.mean_squared_error(y_test, y_pred_sample)
    r2 = metrics.r2_score(y_test, y_pred_sample)
    rmse = math.sqrt(mse)

    print(f"MAE: {mae}, MSE: {mse}, R2: {r2}, RMSE: {rmse}")



dtree = DecisionTreeClassifier(random_state=0, min_samples_split=20, min_samples_leaf=10)
ModelEval(dtree, x_train, x_test, y_train, y_test)


rfc=RandomForestClassifier(random_state=0)
ModelEval(rfc,x_train,x_test,y_train,y_test)

ada=AdaBoostClassifier(random_state=0)
ModelEval(ada,x_train,x_test,y_train,y_test)


#Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create Random Forest Classifier
rfc = RandomForestClassifier(random_state=0)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, verbose=2, n_jobs=1)
grid_search.fit(x_train, y_train)

# Get best parameters
print("Best Parameters: ", grid_search.best_params_)

# Use the best estimator found by grid search
best_rfc = grid_search.best_estimator_

# Evaluate the best model
ModelEval(best_rfc, x_train, x_test, y_train, y_test)


#Cross-Validation

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(best_rfc, x_train, y_train, cv=5)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")



#Feature Selection

importances = best_rfc.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
plt.title("Top 10 Most Important Features for Phishing Detection")
plt.barh(range(10), importances[indices[:10]], align="center")
plt.yticks(range(10), x_train.columns[indices[:10]])
plt.xlabel("Feature Importance")
plt.savefig('Feature_Importance.png')
#plt.show()
plt.close()  # Close the plot window immediately

#Model Evaluation
from sklearn.metrics import roc_curve, auc

# Predict probabilities for ROC
y_prob = best_rfc.predict_proba(x_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
#plt.show()
plt.close()  

#Precision-Recall Curve

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# Predict probabilities for precision-recall curve
y_prob_pr = best_rfc.predict_proba(x_test)[:, 1]

# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_prob_pr)
avg_precision = average_precision_score(y_test, y_prob_pr)

# Plot Precision-Recall curve
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig('precision_recall_curve.png')
#plt.show()
plt.close()

#Model Interpretation
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(best_rfc, x_test, [0, 1])  # replace 0, 1 with feature indices


# Plot Partial Dependence for top 5 important features
plt.figure(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(best_rfc, x_train, features=indices[:5])
plt.savefig('partial_dependence_plots.png')
#plt.show()
plt.close()


#Deployment

import joblib
joblib.dump(best_rfc, 'best_rfc_model.pkl')
print("\n\nDone")

