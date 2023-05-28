import pickle
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance

with open('mycode.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the data from the request
    data = request.json

    # Perform any necessary preprocessing on the data

    # Make predictions using the loaded model
    predictions = model.predict(data)

    # Return the predicted results
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run()

