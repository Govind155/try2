import numpy as np
from flask import Flask, request, render_template, url_for
import pickle
import joblib

app = Flask(__name__)
model = joblib.load("Students_mark_predictor_model.pkl")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    value = np.array(input_features)
    
     if value<0 or value>24:
       return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth &#127757; .')
    
    output = model.predict([value])[0][0].round(2)
    return render_template('index.html', Prediction_text = f"you will get {output}% marks, when you do study {input_features} hours per day")
if __name__ == "__main__":
    app.run(debug=True)
