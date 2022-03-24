import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
import streamlit as st

#import keras
from tensorflow.keras.models import load_model
import pickle

from PIL import Image

#Loading the Model

model = load_model('Capstone-Model-VGG19.h5')

#Name of Classes
a_file = open("plant_diseases.pkl", "rb")
ref = pickle.load(a_file)

a_file.close()


#Setting Title of App
#st.title("Vehicle Classification")


html_temp = """
<div style="background-color:dodgerblue;padding:10px">
<h2 style="color:white;text-align:center;">Plant Disease Classification App </h2>
</div>
    """
st.markdown(html_temp,unsafe_allow_html=True)


image = Image.open('plant-disease-image.jpeg')
st.image(image,use_column_width=True)

st.markdown("""
This web page leverages deep learning to classify plant disease images for the following crops:

* Maize / Corn
* Tomatoes

"""
)


#Uploading the image
plant_image = st.file_uploader("Upload Image below...", type=["jpg", "jpeg", "png"])
submit = st.button('Identify Disease')

with st.spinner('Identfying...'):
  #On predict button click
  if submit:


      if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR",width=512)
        
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
    
        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)

        #Make Prediction
        pred = np.argmax(model.predict(opencv_image))
        prediction = ref[pred]
        st.subheader(str("Image classified as "+prediction))