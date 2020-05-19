import json
import pickle
import numpy as np


class PythonPredictor:
    def __init__(self, config):
        self.config = config
        self.model = self.download_model()

    def download_model(self):
        extratrees_model = pickle.load(open('extratrees-clicks-1584724398.pkl', 'rb'))
        return extratrees_model

    def predict(self, payload):
        print(payload['data'])
        prediction = self.model.predict(np.array(payload['data']))
        return json.dumps({'prediction': prediction.tolist()})
