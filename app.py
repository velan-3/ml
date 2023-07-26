import numpy as np
from flask import Flask,jsonify,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_api/<int:feature1>/<int:feature2>/<int:feature3>/<int:feature4>', methods=['GET'])
def predict_api(feature1, feature2, feature3, feature4):
    # Convert the input features to a numpy array
    input_data = np.array([[feature1, feature2, feature3, feature4]])

    # Make the prediction using the model
    prediction = model.predict(input_data)

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
