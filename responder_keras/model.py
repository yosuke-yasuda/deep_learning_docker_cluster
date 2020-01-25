import keras

def build_model():
    return keras.applications.MobileNetV2(
        input_shape=(120, 250, 3),
        weights=None
    )