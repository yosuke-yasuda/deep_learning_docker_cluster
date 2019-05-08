from logging.handlers import RotatingFileHandler
from logging import INFO
import time
import io

import tensorflow as tf
import keras
import numpy as np
from PIL import Image

# Flask
from flask import (
    Flask,
    jsonify,
    request
)

app = Flask(__name__)
app.config.from_object('config.Config')

@app.errorhandler(404)
def page_not_found(e):
    response = jsonify(
        {
            "message": "url is not found",
            "success": False
        }
    )
    response.status_code = 404
    return response

@app.errorhandler(Exception)
def handle_exception(error):
    response = jsonify(
        {
            "message": str(error),
            "success": False
        }
    )
    app.logger.error(error, exc_info = True)
    response.status_code = 500
    return response

@app.route('/images', methods=['POST'])
def process_image():
    start = time.time()

    stream = request.files["img"].stream
    img = Image.open(io.BytesIO(bytearray(stream.read())))
    resized = img.resize(
        (
            app.config["MODEL_INPUT_SHAPE"][1], 
            app.config["MODEL_INPUT_SHAPE"][0]
        ) # W, H
    ) 
    img_tensor = np.asarray(resized)
    
    with graph.as_default():
        pred = model.predict(
            img_tensor[
                np.newaxis, # insert batch size dimention
                ...
            ]
        )[0] # extract first batch
    
    end = time.time()
    app.logger.info("Elapsed Time: %s" % (end - start))

    return jsonify(
        {
            "success": True,
            "result": float(pred[0])
        }
    )

if __name__ == '__main__':
    global graph
    graph = tf.get_default_graph()
    global model
    model = keras.applications.mobilenet.MobileNet(
        weights=None,
        input_shape=app.config["MODEL_INPUT_SHAPE"],
        classes=1
    )

    handler = RotatingFileHandler('log/info.log', maxBytes=10000, backupCount=1)
    handler.setLevel(INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(INFO)

    app.run(host= '0.0.0.0', port = 5000, debug = False)