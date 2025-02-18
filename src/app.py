
from flask import Flask, request, jsonify, send_from_directory
from gevent import pywsgi
import numpy as np
import cv2
from api import detect_image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
  return send_from_directory('templates', 'index.html')

@app.route('/public/<path:path>', methods=['GET'])
def public(path):
  return send_from_directory('templates', path)

@app.route('/api/upload', methods=['POST'])
def upload_image():
  if 'file' not in request.files:
    return jsonify({'error': 'no file'}), 400
  
  file = request.files['file'].read()
  auto_detect_direction = bool(request.form.get("auto_detect_direction"))
  result_include_image = bool(request.form.get("result_include_image"))
  npimg = np.frombuffer(file, np.uint8)
  img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
  try:
    ocr, front, face = detect_image(img, auto_detect_direction = auto_detect_direction, result_include_image = result_include_image)
    data = []
    idx = 0
    for res in ocr:
      for line in res:
        data.insert(idx, {"type": "ocr", "text": line[1][0], "confidence": line[1][1]})
        idx += 1
    
    if (front is not None):
      data.insert(idx, {"type":"front", "content": front})
      idx += 1

    if (face is not None):
      data.insert(idx, {"type":"face", "content": face})

    return jsonify({
      'result': 'success',
      'data': data
    })
  
  except Exception as e:
    return jsonify({
      'result': 'fail',
      'error': e.args[0]
    })



# app.run(port=5000, debug=True)
server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
server.serve_forever()

    
