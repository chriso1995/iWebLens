import base64
import json
import object_detection as od
import numpy as np
import cv2
from flask import Flask,jsonify, request

app = Flask(__name__)

@app.route('/api/object_detection', methods=['POST'])
def image_validation():
	data = json.loads(request.json) # assigning the request data into data variable
	id = data['id']
	resp_arr = {}
	image_data = data['image'] # assigning the encoded image information to variable image_data
	processim = np.fromstring(base64.b64decode(image_data), dtype=np.uint8) # performing the decoding of the image using the base64 package
	img = cv2.imdecode(processim, 1) # using cv2 to decode the image data and decodes it to an image format
	image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # performing colour conversino from BGR to RGB
	nets = od.load_model(od.get_config("yolov3-tiny.cfg"),od.get_weights("yolov3-tiny.weights"))
	pred = od.do_prediction(image, nets, od.get_labels("coco.names")) # using the object detection script to perform the load_model and do prediction methods
	resp_arr = {
		"id": data['id'],
		"objects": 
			pred
	}
	return json.dumps(resp_arr, indent=9) # returning the pred variable which is then  converting python object into json string with proper indentation

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, threaded = True) # Threaded should be set to true by default with flask versions greater than 1.0: added for visibility
