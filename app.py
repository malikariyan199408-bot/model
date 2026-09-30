from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    raise FileNotFoundError("model.pkl not found. Please train the model first using house.py")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['bedrooms']),
            float(request.form['bathrooms']),
            float(request.form['floors']),
            float(request.form['yr_built'])
        ]
        arr = np.array([features]).astype(np.float64)
        pred = model.predict(arr)[0]
        return render_template('index.html', data=int(pred.item()))
    except (KeyError, ValueError):
        return render_template('index.html', data="Invalid input")


if __name__ == '__main__':
    app.run(debug=True, port=8000)
