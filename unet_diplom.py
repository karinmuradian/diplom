



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

st.title('**Сегментация объектов на снимках**')

def load_model ():

    model = ort.InferenceSession(
        r'C:\Users\katko\Desktop\Karina\Diplom\venv\multi_unet_model2.onnx',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
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
    #img = Image.open(img).convert ('RGB')
    img = image.smart_resize(img, (256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=3)
    x = normalize(x, axis =1)
    x = preprocess_input(x)
    return x
     
def get_predictions (model, x):
    model.get_inputs()[0].shape
    model.get_inputs()[0].type
    x_norm =x[:,:,0][:,:,None]
    x_input = np.expand_dims(x_norm, 0)
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    preds =model.run(([output_name]), input_feed ={input_name: x_input})
    scores = np.argmax (preds, axis = 3)[0,:,:]
    st.image (scores, cmap = 'twilight')
    return scores 


model = load_model()
img = load_image()
result = st.button('Создать маску')
            
if result:
    x = preprocess_image(img)
    get_predictions(model,x)
    st.write('**Предсказанная маска**')


