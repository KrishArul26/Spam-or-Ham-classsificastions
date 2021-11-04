from flask import Flask, request, jsonify,render_template
import os
from flask_cors import CORS, cross_origin
from spams import prediction

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
@cross_origin()

def predictRoute():
    data = request.json["data"]
    pred = prediction(data)
    print(pred)
    print(type(pred))
    return {"Result": str(pred)}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000, debug=True)