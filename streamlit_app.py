

import io
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input, decode_predictions
import onnxruntime as ort
import cv2
from keras.utils import normalize
import matplotlib.pyplot as plt



st.title('**Сегментация объектов на снимках**')

def load_model ():

    model = ort.InferenceSession(
        r'C:\Users\katko\Desktop\Karina\Diplom\venv\multi_unet_model3.h5new.onnx',
        providers=['AzureExecutionProvider', 'CPUExecutionProvider']
        )

    return model

def load_image():
    uploaded_file = st.file_uploader(label='**Выберите изображение для сегментации**')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def preprocess_image(img):

    img = tf.image.rgb_to_grayscale (img, name=None)
    img = image.smart_resize(img, (256, 256))
    x = image.img_to_array(img)
    x = np.transpose(np.expand_dims(x, axis=0), (3, 2, 1, 0) )
    x = preprocess_input(x)

    return x

import numpy 

def get_predictions (model, x):
    #model.get_inputs()[0].shape
    #model.get_inputs()[0].type
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    preds =model.run([output_name], {input_name: x.astype(numpy.float32)})[0]
    #preds =model.run([output_name], {input_name: np.ones ((1,*model.get_inputs()[0].shape [1:]),dtype = np.float32)})[0]
    scores = np.argmax (preds, axis = 3)[0,:,:]
    return scores 



model = load_model()

img = load_image()
result = st.button('Создать маску')

if result:
    x = preprocess_image(img)
    scores = get_predictions(model,x) 
    st.title('**Предсказанная маска**')
    plt.figure(figsize=(12, 8))
    plt.subplot(233)
    plt.imshow(scores, cmap='twilight')

    st.pyplot()




   
 

