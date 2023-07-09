import pandas as pd
from flask import Flask, render_template, request
import json
import pickle
import numpy as np

pipe = pickle.load(open("templates/dataset.pickle","rb"))
pipe1 = pickle.load(open("templates/data1.pickle","rb"))
app = Flask(__name__)
with open('templates/columns.json', 'r') as json_file:
    data = json.load(json_file)

@app.route("/")
def index():
    locations = sorted(data['data_columns'])
    return render_template("index.html",locations=locations)

@app.route("/predict",methods=["POST"])
def predict():
    location = request.form.get("location")
    bhk = request.form.get("bhk")
    bath = request.form.get("bath")
    sqft = request.form.get("total_sqft")
    # print(pipe1.columns)
    Xx=np.zeros(len(pipe1.columns))
    # print("1st block jayanagar" in pipe1.columns)
    loc_index = np.where(pipe1.columns == location)[0][0]
    # print(loc_index)
    if sqft and bath and bhk:
        Xx[0] = sqft
        Xx[1] = bath
        Xx[2] = bhk
        if loc_index >= 0:
            Xx[loc_index] = 1
        input=pd.DataFrame([Xx], columns=[*list(pipe1.columns)])
        prediction = pipe.predict(input)[0]
        return "Estimated Price: ₹"+str(round(prediction,2))+" Lakhs"
    else:
        return "Enter all values for proper functioning."
if __name__=="__main__":
    app.run(debug=True,port = 5001)