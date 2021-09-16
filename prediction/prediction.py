import yaml
import os
import json
import joblib
import numpy as np
from  src.get_data import read_params
from pickle import load

params_path = "params.yaml"
schema_path = os.path.join("prediction", "schema_in.json")
scaler = load(open('scaler.pkl', 'rb'))

class NotInRange(Exception):
    def __init__(self, col,message="Values entered are not in expected range"):
        self.message = col+" "+message
        super().__init__(self.message)


def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(scaler.transform(data))
    try:
        if 0 <= prediction <= 10:
            return prediction
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"


def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema

def validate_input(dict_request):

    def _validate_values(col, val):
        schema = get_schema()

        if not (schema[col]["min"] <= float(val) <= schema[col]["max"]) :
            raise NotInRange(col)
            

    for col, val in dict_request.items():
        _validate_values(col, val)
    
    return True


def form_response(dict_request):
   if validate_input(dict_request):
        print("HI")
        data = dict_request.values()
        data = [list(map(float, data))]
        response = predict(data)
        return response

    
def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data = np.array([list(dict_request.values())])
            response = predict(data)
            response = {"response": response}
            return response
            
    except NotInRange as e:
        response = {"the_exected_range": get_schema(), "response": str(e) }
        return response

    except Exception as e:
        response = {"response": str(e) }
        return response