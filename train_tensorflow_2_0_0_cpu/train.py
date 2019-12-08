import os
import datetime
import shutil
import tensorflow as tf
from sacred.observers import MongoObserver
from config import ex

# save learning metrics for omniboard
ex.observers.append(
    MongoObserver.create(
        url='logger:27017', db_name='train_log'
    )
)

# callback to save loss log in omniboard
def ex_log_callback(ex_run):
    def callback(epoch, logs):
        for key, val in logs.items():
            ex_run.log_scalar(key, val, epoch)
    return tf.keras.callbacks.LambdaCallback(on_epoch_end=callback)

@ex.automain
def train(input_shape, _run=None, *args, **kwargs):
    # create experiment id from sacred
    if _run is not None:
        run_id = _run._id
    else:
        run_id = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    model = tf.keras.applications.mobilenet.MobileNet(
        weights=None,
        input_shape=input_shape,
        classes=1
    )
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    train_generator = datagen.flow_from_directory(
        '/shared/data/sample',
        target_size=input_shape[:2],
        batch_size=1,
        class_mode='binary'
    )

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam()
    )

    weight_save_path = os.path.join(
        "./saved_model",
        str(run_id)
    )
    if os.path.exists(weight_save_path):
        shutil.rmtree(weight_save_path)
    os.makedirs(weight_save_path)
    
    tflog_save_path = os.path.join(
        "/shared/logger/tensorboard_logs", 
        "train_tf_2_0_0_cpu", 
        str(run_id)
    )
    if os.path.exists(tflog_save_path):
        shutil.rmtree(tflog_save_path)
    os.makedirs(tflog_save_path, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=tflog_save_path, 
            write_graph=False
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(
                weight_save_path,
                "weights_epoch{epoch:02d}.h5"
            ),
            monitor='loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=True,
            mode='auto',
            period=1
        )
    ]

    if _run is not None:
        callbacks.append(ex_log_callback(ex_run=_run))

    model.fit_generator(
        train_generator,
        callbacks=callbacks,
        steps_per_epoch=3,
        epochs=2
    )
