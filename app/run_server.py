# USAGE
# Start the server:
# 	python run_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

import numpy as np
import pandas as pd
import dill
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSum(BaseEstimator, TransformerMixin):
    '''
    Transform class for sum of several columns

    column - list of columns to sum

    Return: DF with one column - sum of given columns
    '''
    counter = 0
    
    def __init__(self, column):
        self.column = column
        

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        FeatureSum.counter += 1
        Xt = X[self.column].copy()
        Xt[f'sum{FeatureSum.counter}'] = Xt.sum(axis=1, skipna=True)
        return Xt[[f'sum{FeatureSum.counter}']]


dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask
# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)

with open('app/models/features.txt', 'r') as f:  
    features = [feature.rstrip() for feature in  f.readlines()]

# data = dict(zip(features, [np.nan for i in range(len(features))]))


@app.route("/", methods=["GET"])
def general():
	return "will he satisfied?"


@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		request_json = flask.request.get_json()
		
		# делаем список значений np.nan для отсутствующих фьючей в запросе.
		if request_json:
			features_from_json = list(set(request_json.keys()).intersection(set(features)))
			if features_from_json:
				json_lement = request_json[features_from_json[-1]]
				if isinstance(json_lement, list):
					flag = 'list'
					nan_list_len =len(request_json[features_from_json[-1]])
					nan_list = [np.nan for i in range(nan_list_len)]
				else:
					flag = 'not list'
					nan_list = [np.nan]


		# делаем удобоваримый для dataframe словарь с полученными данными для предсказания.
		for feature in features:
			if request_json and flag == 'not list':
				if feature in request_json:
					data[feature] = [request_json[feature]]
				else:
					data[feature] = nan_list
			elif request_json and flag == 'list':
				if feature in request_json:
					data[feature] = request_json[feature]
				else:
					data[feature] = nan_list
			else:
				data[feature] = [np.nan]


		# делаем предсказание и отправляем ответ
		if isinstance(data[features[-1]], list):
			preds = model.predict(pd.DataFrame(data))
			preds = preds.tolist()
			data["predictions"] = preds

		data["success"] = True
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	modelpath = "app/models/gb_pipeline.dill"
	load_model(modelpath)
	app.run()