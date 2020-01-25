import numpy as np
import cv2
import keras

def predict(graph, model, img_tensor, *args, **kwargs):
    input_shape = keras.backend.int_shape(model.inputs[0])
    input_tensor = cv2.resize(
        img_tensor, input_shape[1:3][::-1]
    )[np.newaxis, ...]
    with graph.as_default():
        pred_result = model.predict(input_tensor)
    processed = post_process(pred_result)
    return processed

def post_process(pred_result):
    return {
        "probs": pred_result[0].tolist()
    }
