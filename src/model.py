from operator import index
from os import name
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, roc_auc_score,  
                             f1_score, accuracy_score, classification_report)
import sys
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Add project root to system path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from config import PROCESSED_DATA_DIR, MODELS_DIR

def train_model(X_train, y_train, model_type = "linear", save_path = MODELS_DIR, random_state = 42):
    if model_type == 'linear':
        model = LogisticRegression(penalty= 'l2', max_iter= 1000, multi_class='auto', 
                                       solver= 'lbfgs', warm_start= True, n_jobs= 1, random_state= random_state)
    elif model_type == 'sgd':
         model = SGDClassifier(loss= 'log_loss', max_iter= 1000, warm_start= True, random_state= random_state, n_jobs= 1)
    elif model_type== 'svc':
         model = SVC(C= 1, kernel= 'rbf', probability= True, random_state= random_state, max_iter= -1)
    else:
        raise ValueError("Invalid model input")
    
    #Train the model
    model.fit(X_train, y_train)
    
    #Save the model in the given path
    if save_path:
        joblib.dump(model, save_path)
        print(f"✅Model successfully to{save_path}")
    
    return model


#-------------- MODEL EVALUATION -------------#
def evaluation(model, X_test, y_test):
    """Evaluate model performance and return metrics."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
    y_pred = model.predict(X_test)
    metrics_dict = {
        "Accuracy Score": accuracy_score(y_true= y_test, y_pred= y_pred),
        "Classification Report": classification_report(y_test, y_pred, zero_division= 0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "F1 Score": f1_score(y_true= y_test, y_pred= y_pred, average= 'macro', zero_division=0),
        "Recall Score": recall_score(y_test, y_pred, average= 'macro', zero_division= 0),
        "Precision Score": precision_score(y_test, y_pred, average= 'macro', zero_division= 0)
    }

    ## Try computing ROC AUC (only works if model supports predict_proba)
    try:
        y_proba = model.predict_proba(X_test)
        roc_auc  = roc_auc_score(y_true= y_test, y_score= y_proba)
        metrics_dict["ROC"] = roc_auc
    except:
        metrics_dict["ROC"] = None
    
    print("📊Model Evaluation Summary")
    print(f"Accuracy {metrics_dict['Accuracy Score']:.4f}")
    print(f"F1 Score(macro) {metrics_dict['F1 Score']:.4f}")
    print(f"Presicion Score(macro) {metrics_dict['Precision Score']:.4f}")
    print(f"Recall Score(macro) {metrics_dict['Recall Score']:.4f}")
    if metrics_dict["ROC"] is not None:
        print(f"ROC AUC SCORE(OVR) {metrics_dict['ROC']:.4f}")
    print("Confusion Matrix\n", metrics_dict['Confusion Matrix'])
    print("Detailed Classification Report\n", {metrics_dict['Classification Report']})
    
    return metrics_dict

#================================#
#       MAIN EXECUTION SCRIPT
#================================#
# Load Preprocessed Data
X_train = pd.read_csv(PROCESSED_DATA_DIR/"X_train.csv")
y_train = pd.read_csv(PROCESSED_DATA_DIR/"y_train.csv").values.ravel()
X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
y_test = pd.read_csv(PROCESSED_DATA_DIR/ "y_test.csv").values.ravel()

#Train the model
models={
"Logistic Regression": train_model(X_train= X_train, y_train= y_train, model_type= "linear", save_path= MODELS_DIR / "log_model.joblib"),
"SGD":train_model(X_train= X_train, y_train= y_train, model_type= "sgd", save_path= MODELS_DIR / "sgd_model.joblib"),
"SVC":train_model(X_train= X_train, y_train= y_train, model_type= "svc", save_path= MODELS_DIR / "svc_model.joblib"),
}

#Evaluate and Compare results
results = {"Model ":[], "Accuracy ":[], "Precision ":[],"Recall ":[], "F1 Score ":[], "ROC AUC ":[]}

for name, model in models.items():
    print(f"📈Evaluating{name} model....")
    metrics_dict = evaluation(model, X_test, y_test)
    results["Model "].append(name)  
    results["Accuracy "].append(metrics_dict["Accuracy Score"])
    results["F1 Score "].append(metrics_dict["F1 Score"])
    results["Precision "].append(metrics_dict["Precision Score"])
    results["Recall "].append(metrics_dict["Recall Score"])
    results["ROC AUC "].append(metrics_dict["ROC"])

#SAVE SUMMARY
df_results = pd.DataFrame(results)
df_results.to_csv(PROCESSED_DATA_DIR/"Model_Evaluation_results.csv", index = False)
print("\n✅ Model comparison saved to model_evaluation_results.csv\n")
print(df_results)  