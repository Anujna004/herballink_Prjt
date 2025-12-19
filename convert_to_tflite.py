import tensorflow as tf

# Convert Leaf Model
leaf_model = tf.keras.models.load_model("model1.h5")
converter_leaf = tf.lite.TFLiteConverter.from_keras_model(leaf_model)
converter_leaf.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: reduces size
tflite_leaf_model = converter_leaf.convert()
with open("model1.tflite", "wb") as f:
    f.write(tflite_leaf_model)
print("Leaf model converted to model1.tflite")

# Convert Skin Model
skin_model = tf.keras.models.load_model("skin_disease_model.h5")
converter_skin = tf.lite.TFLiteConverter.from_keras_model(skin_model)
converter_skin.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_skin_model = converter_skin.convert()
with open("skin_disease_model.tflite", "wb") as f:
    f.write(tflite_skin_model)
print("Skin model converted to skin_disease_model.tflite")

