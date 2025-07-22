from flask import Flask, request, render_template  
import pickle, numpy as np

app = Flask(__name__)  
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]  
    pred = model.predict([np.array(data)])  
    result = "High Risk" if pred[0] == 1 else "Low Risk"
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)

from waitress import serve
serve(app, host='0.0.0.0', port=5000)
