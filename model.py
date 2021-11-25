import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Load model and data
model = joblib.load('poly_svm_model.joblib')
data = pd.read_csv('death_risk_predict.csv')

# Normalization value
min_dict = {}
max_min_dict = {}
for column in data.columns:
    min_dict[column] = min(data[column])
    max_min_dict[column] = max(data[column])


def death_risk_predict(Base_Excess, Anion_Gap, Chloride, Creatinine, Potassium, Sodium, Urea_Nitrogen, RDW,
                       White_Blood_Cells, icu_los):
    """
    Predict function
    """

    # Normalization
    x1 = (Base_Excess - min_dict['Base Excess']) / max_min_dict['Base Excess']
    x2 = (Anion_Gap - min_dict['Anion Gap']) / max_min_dict['Anion Gap']
    x3 = (Chloride - min_dict['Chloride']) / max_min_dict['Chloride']
    x4 = (Creatinine - min_dict['Creatinine']) / max_min_dict['Creatinine']
    x5 = (Potassium - min_dict['Potassium']) / max_min_dict['Potassium']
    x6 = (Sodium - min_dict['Sodium']) / max_min_dict['Sodium']
    x7 = (Urea_Nitrogen - min_dict['Urea Nitrogen']) / max_min_dict['Urea Nitrogen']
    x8 = (RDW - min_dict['RDW']) / max_min_dict['RDW']
    x9 = (White_Blood_Cells - min_dict['White Blood Cells']) / max_min_dict['White Blood Cells']
    x10 = (icu_los - min_dict['icu_los']) / max_min_dict['icu_los']

    # Predict
    death_risk = model.predict_proba(np.array([x1, x2, x3, x4, x5, x6,
                                               x7, x8, x9, x10]).reshape(1, -1))[0][1]

    return round(death_risk, 4)


ROC_logis_svm = pd.read_csv('ROC_logis_svm.csv')
AUC_dict = {'Logistic': 0.77,
            'Linear SVM': 0.77,
            'Poly SVM': 0.79,
            'RBF SVM': 0.75}


def ROC_plot(model_name):
    y_test = ROC_logis_svm['y_test']
    if model_name == 'Logistic':
        y_predict_proba = ROC_logis_svm['y_logis_predict_proba']
    elif model_name == 'Linear SVM':
        y_predict_proba = ROC_logis_svm['y_linear_predict_proba']
    elif model_name == 'Poly SVM':
        y_predict_proba = ROC_logis_svm['y_poly_predict_proba']
    elif model_name == 'RBF SVM':
        y_predict_proba = ROC_logis_svm['y_rbf_predict_proba']

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_proba)

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % AUC_dict[model_name])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title('ROC for Radial Basis Function SVM')
    plt.show()
