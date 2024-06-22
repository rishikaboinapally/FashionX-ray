import os
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import gradio as gr

feature_list = np.array(pickle.load(open(r'C:\Users\rishika\anaconda3\Scripts\small_dataset\embeddings.pkl', 'rb')))
filenames = pickle.load(open(r'C:\Users\rishika\anaconda3\Scripts\small_dataset\filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

def predict(input_image):
    os.makedirs('uploads', exist_ok=True)
    image_path = "uploads/temp_image.jpg"
    input_image.save(image_path)
    
    features = feature_extraction(image_path, model)
    indices = recommend(features, feature_list)
    
    result_images = [filenames[indices[0][i]] for i in range(5)]
    return [Image.open(img_path) for img_path in result_images]

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[gr.Image(type="pil", label=f"Recommendation {i+1}") for i in range(5)],
    title="Amazon Lens",
    description="Upload an image of clothing, and the system will recommend similar items."
)

if __name__ == "__main__":
    iface.launch()
