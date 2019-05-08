import keras

def train():
    input_shape = (150, 150, 3)
    model = keras.applications.mobilenet.MobileNet(
        weights=None,
        input_shape=input_shape,
        classes=1
    )
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = datagen.flow_from_directory(
        '/shared/data/sample',
        target_size=input_shape[:2],
        batch_size=1,
        class_mode='binary')

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam()
    )
    model.fit_generator(
        train_generator,
        steps_per_epoch=3
    )

if __name__ == "__main__":
    if len(keras.backend.tensorflow_backend._get_available_gpus()) == 0:
        raise Exception("gpu is not recognized")
    train()
    print("success")