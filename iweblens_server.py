import base64
import json
import object_detection as od
import numpy as np
import cv2
from flask import Flask,jsonify, request

app = Flask(__name__)

@app.route('/api/object_detection', methods=['POST'])
def image_validation():
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

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, threaded = True)
