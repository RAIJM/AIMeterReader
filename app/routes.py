from flask import Flask, request, make_response,Response,render_template,json,redirect, url_for
import os
import io
import numpy as np
import cv2
import jsonpickle
import cv2
import os
import zipfile
import sys

from collections import defaultdict
from PIL import Image

from models import *




app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



meter_class_model = MeterClassModel('meter_model','/frozen_meter_model.pb')
analog_model = AnalogMeterModel('analog_meter_model','/frozen_inference_graph.pb','analog_label_map.pbtxt')
digital_model = DigitalMeterModel('digital_meter_model','/frozen_inference_graph.pb','digitalmeter_label_map.pbtxt')
    

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        #	pass
        print("Converting..")
        file = request.files['file']
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)

        meter_type = meter_class_model.predict(img)

        reading = ''

        if(meter_type == 'digital'):
            
            img,reading = digital_model.predict_reading(img)

        else:
            
            reading = analog_model.predict_reading(img)


        print("Saving..")
        im = Image.fromarray(img)
        im.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
       
    return json.dumps({'filename':file.filename,'reading':reading,'meter_type':meter_type})



app.run(host="0.0.0.0", port=5000)
