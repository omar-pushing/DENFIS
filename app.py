from flask import Flask, request, jsonify, render_template
from model.predict_severity import predict

app = Flask(__name__, static_url_path='/assets', static_folder='static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_severity():
    data = request.get_json()
    temperature = data['temperature']
    visibility = data['visibility']
    humidity = data['humidity']
    severity = predict(temperature, visibility, humidity)
    return jsonify({'severity': severity})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)