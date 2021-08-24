from flask import Flask, render_template, request
import numpy as np
import pickle

# from gbc import GradientBoostingClassifier


model = pickle.load(open("irl.pkl", "rb"))
app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    data1 = request.form["a"]
    data2 = request.form["b"]
    data3 = request.form["c"]
    data4 = request.form["d"]
    data5 = request.form["e"]
    data6 = request.form["f"]
    data7 = request.form["g"]
    data8 = request.form["h"]
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8]])
    prediction = model.predict(arr)

    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
