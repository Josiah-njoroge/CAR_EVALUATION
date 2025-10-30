import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score, accuracy_score, classification_report
import sys
import warnings

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
    metrics = {
        "Accuracy Score": accuracy_score(y_true= y_test, y_pred= y_pred),
        "Classification Report": classification_report(y_test, y_pred, zero_division= 0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "F1 score": f1_score(y_true= y_test, y_pred= y_pred, average= 'macro', zero_division=0),
        "Recall score": recall_score(y_test, y_pred, average= 'macro', zero_division= 0),
        "Precision Score": precision_score(y_test, y_pred, average= 'macro', zero_division= 0)
    }

    #Try computing 
X_train = pd.read_csv(PROCESSED_DATA_DIR/"X_train.csv")
y_train = pd.read_csv(PROCESSED_DATA_DIR/"y_train.csv").values.ravel()
#train_model(X_train= X_train, y_train= y_train, model_type= "linear", save_path= MODELS_DIR / "log_model")