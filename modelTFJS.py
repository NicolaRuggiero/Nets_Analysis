import tensorflow as tf
import tensorflow_hub as hub
import tensorflowjs as tfjs

if __name__ == '__main__':
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5")
    ])
    model.build([None, 224, 224, 3])
    tfjs.converters.save_keras_model(model, 'C:/Users/Utente/Desktop/TFJS/V3Large/model.h5')