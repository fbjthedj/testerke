# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle

# app = Flask(__name__)

# # model = pickle.load(open('salary.pkl', 'rb'))

# model = pickle.load(open('salary.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods = ['POST'])
# def predict():
    
    
#     features = [float(x) for x in request.form.values()]

   
#     final_features = [np.array(features)]

#     prediction = model.predict(final_features)

#     outcome = round(prediction[0], 2)
  
#     return render_template('index.html', prediction_text='Your projected salary is $ {}'.format(outcome))

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
    
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#     output = prediction[0]

#     return jsonify(output)

# if __name__ == '__main__':
#     app.run(debug=True)          ## Running the app as debug==True

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('salary_predict.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [float(x) for x in request.form.values()]
        except ValueError:
            return "Invalid input data", 400
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        outcome = round(prediction[0], 2)
        return render_template('index.html', prediction_text='Your projected salary is {}'.format(outcome))
    return "Invalid Method", 405

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if request.method == 'POST':
        data = request.get_json(force=True)
        try:
            prediction = model.predict([np.array(list(data.values()))])
        except ValueError:
            return "Invalid input data", 400
        output = prediction[0]
        return jsonify(output)
    return "Invalid Method", 405

if __name__ == '__main__':
    app.run(debug=False)