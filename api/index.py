import os

os.chdir("/var/task")

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_url_path='', static_folder='../public')

MODEL_LOADED = False
predict_func = None

def load_predict_module():
    global predict_func, MODEL_LOADED
    if not MODEL_LOADED:
        from model.predict_severity import predict as pred
        predict_func = pred
        MODEL_LOADED = True

@app.route('/')
def home():
    return send_from_directory('public', 'index.html')

@app.route('/assets/<path:filename>')
def serve_static(filename):
    return send_from_directory('public', f'assets/{filename}')

@app.route('/api/predict', methods=['POST'])
def predict_severity():
    load_predict_module()
    data = request.get_json()
    temperature = data['temperature']
    visibility = data['visibility']
    humidity = data['humidity']
    severity = predict_func(temperature, visibility, humidity)
    return jsonify({'severity': severity})

def handler(environ, start_response):
    return app(environ, start_response)