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

		for feature in features:
			if feature in request_json:
				data[feature] = request_json[feature]
			else:
				data[feature] = np.nan

		if isinstance(data[features[-1]], list):
			preds = model.predict(pd.DataFrame(data))
			preds = preds.tolist()
			data["predictions"] = preds
		else: 
			# print(data)
			preds = model.predict(pd.DataFrame(data, index=[0]))
			data["predictions"] = preds.tolist()
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