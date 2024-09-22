from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
import pickle
from pycaret.regression import *

app = Flask(__name__)

@app.route("/",methods=["GET"])
def home():
    return 'Predicción precio'

file_name = 'predicciones2.json'
path = "C:/Users/MCS/OneDrive - Universidad Santo Tomás/Inteligencia Artificial/Codigos Propios/TAREA/Segundo Punto/"
dataset = pd.read_csv(path + "dataset_APP.csv",header = 0,sep=";",decimal=",") 
covariables = ['Avg. Session Length', 'Time on App', 'Time on Website',
               'Length of Membership', 'dominio', 'Tec']

with open(path + 'best_model.pkl', 'rb') as model_file:
    dt2 = pickle.load(model_file)

def save_prediction(prediction_data):
    try:
        with open(file_name, 'r') as file:
            predictions = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        predictions = []

    predictions.append(prediction_data)

    with open(file_name, 'w') as file:
        json.dump(predictions, file, indent=4)

@app.route('/predict', methods=['POST'])
def predict():
    base = dataset.get(covariables)
    data = request.json

    usuario = pd.DataFrame([data])
    usuario.drop(columns=["email"], inplace=True)

    usuario.columns = base.columns
    base = pd.concat([usuario, base], axis = 0, ignore_index=True)
    
    prediccion = predict_model(dt2, data=base)
    prediccion = prediccion['prediction_label'].head(1).values[0]
    prediccion = np.round(float(prediccion),2)
    resultado = {"email": data["email"], "Prediccion": prediccion}
    save_prediction(resultado)

    return jsonify(resultado)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)