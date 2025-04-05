import os
import cv2
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

model_save_path = 'my_trained_model'

# Load the saved model
loaded_model = tf.keras.models.load_model(model_save_path)
print(f"Model loaded from {model_save_path}")

# Use the loaded model for predictions
test_image = 'test.png'
predicted_label_index = predict_image(test_image)

# Translate the predicted label using the PLU mapping
product_class = list(label_to_index.keys())[list(label_to_index.values()).index(predicted_label_index)]

product_name = plu_mapping[f"{product_class}"]

# Print the predicted PLU and product name
print(f"Predicted class: {product_class}")
print(f"Name of Product: {product_name}")
