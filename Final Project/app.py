from flask import Flask, render_template, request, send_from_directory
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)

with open('decision_tree_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/members')
def teamMembers():
    return render_template('team.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        el = int(request.form.get('elev'))
        te = int(request.form.get('temp'))
        wi = int(request.form.get('wind'))
        hu = int(request.form.get('humi'))

        if loaded_model.predict([[el, te, wi, hu]]) == 1:
            return "Predition is you should LEAVE the area."
        else:
            return "Predition is you are SAFE to reside."
    except ValueError as e:
        return "ERROR : Please enter data."

if __name__=="__main__":
    app.run(debug=True)