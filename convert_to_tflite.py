import tensorflow as tf

model = tf.keras.models.load_model("predictor/models/diabetes_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("predictor/models/diabetes_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully!")
