# Imports

from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

## Below command will install all the libraries from below file:
## pip install -r requirements.txt
 
# For running the 'app.py' file in terminal use command 'python app.py'

application = Flask(__name__)
app = application

# Ridge Regressor Model and StandardScaler Pickle
model = pickle.load(open('models/modelForPrediction.pkl','rb'))
standard_scaler = pickle.load(open("models/standardscaler.pkl","rb"))


# Route for HomePage
@app.route('/')
def index():
    return render_template('index.html')

# Route for HomePage
# Handles both 'GET' and 'POST'
@app.route('/predictdata', methods=['GET','POST']) 
def predict_datapoint():
    # Will read values from 'home.html' and 
    # Will send values from model to 'Home.html'
    
    result = ""
    
    if request.method == 'POST':
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        
        # Use data in the order for model training
        new_data_scaled = standard_scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict = model.predict(new_data_scaled)  
        
        if predict[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic' 

        return render_template('home.html', result=result)
   
    else:
        # For Get Request
        return render_template('home.html')


if __name__ == "__main__":
    # Host
    app.run(host="0.0.0.0")
    
    
## For Running the program :
## 1. pip  install -r requirements.txt
## 2. python application.py