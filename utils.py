from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

image_size = 224

def load_model(model):
    model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
    return model

def process_image(image_path):
    image = np.asarray(Image.open(image_path))
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return np.expand_dims(image, axis=0)

def predict(image, model, top_k):
    predictions = model.predict(image)
    
    if top_k == None:
        max_class = np.argmax(predictions[0])
        return [predictions[0][max_class]], [max_class]
    
    classes = (-predictions[0]).argsort()[:top_k]
    probs = predictions[0][classes]
    return probs, classes

