import tensorflow as tf

file_name = 'best_fall_detector.h5'

model = tf.keras.models.load_model(file_name)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

with open('best_fall_detector.tflite', 'wb') as f:
    f.write(converter.convert())
