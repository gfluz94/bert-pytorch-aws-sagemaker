import os
import sys
from flask import Flask, request, make_response, abort, jsonify

from predictor import PredictorService
from pyschemavalidator import validate_param
import logging

# Logger creation
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger()

# The flaks app to serve predictions
app = Flask(__name__)

PREFIX = "/opt/ml/"
MODEL_PATH = os.path.join(PREFIX, "model")
model = PredictorService(MODEL_PATH)
model.start()

THRESHOLD = 0.5


def health():
    """
    Sanity check to make sure the container is properly running.
    """
    return make_response("", 200)

@app.route("/health", methods=["GET"])
def home():
    return health()

@app.route("/ping", methods=["GET"])
def ping():
    return health()

@app.route("/invocations", methods=["POST"])
@validate_param(key="text", keytype=str, isrequired=True)
def invocations():
    """
        Online prediction on single data instance. Data is accepted as JSON and then properly parsed.
        Then, the model predicts whether the text contains humor or not.
    """
    data = request.get_json(silent=True)
    pred = model.predict(data["text"])[0]
    pred = round(pred, 2)
    isHumor = (pred >= THRESHOLD)
    return make_response(
        jsonify({
            "text": data["text"],
            "humor": isHumor,
            "score": pred
        }),
        200
    )