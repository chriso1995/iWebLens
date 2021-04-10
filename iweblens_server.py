import base64
from io import BytesIO
import json
import object_detection as od
import numpy as np
import cv2
import sys
from flask import Flask,jsonify, request, Response

app = Flask(__name__)

@app.route('/')
def hello_world():
	return "Hello World!"

@app.route('/api/object_detection', methods=['POST'])
def image_validation():
	error = None
	if request.method == "POST":
		data = json.loads(request.json)
		id = data['id']
		resp_arr = {}
		image_data = data['image']
		processim = np.fromstring(base64.b64decode(image_data), dtype=np.uint8)
		img = cv2.imdecode(processim, 1)
		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		nets = od.load_model(od.get_config("yolov3-tiny.cfg"),od.get_weights("yolov3-tiny.weights"))
		pred = od.do_prediction(image, nets, od.get_labels("coco.names"))
		
		resp_arr = {
			"id": data['id'],
			"objects": 
				pred
			
		}
		return json.dumps(resp_arr, indent=9)
	else:
		error = "Invalid Request"
		return error

#def yolopathsCFG():
#	cfgpath = "yolov3-tiny.cfg"
#	
#	CFG = object_detection.get_config(cfgpath)
#	return CFG
#
#def yolopathsW():
#	wpath = "yolov3-tiny.weights"
#	Weights=object_detection.get_weights(wpath)
#	return Weights
#
#def yolopathL():
#	labelsPath = "coco.names"
#	Labels = object_detection.get_labels(labelsPath)
#	return Labels
#		

if __name__ == '__main__':
	#ypaths = sys.argv[1]
	app.debug = 1
	app.run(host='0.0.0.0', port=8080, threaded = True)
