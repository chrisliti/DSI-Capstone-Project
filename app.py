## Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


import keras
import streamlit as st

from keras.models import load_model
import pickle

#Loading the Model
model = load_model('Capstone-Model-VGG19.h5')

#Name of Classes
a_file = open("plant_diseases.pkl", "rb")
ref = pickle.load(a_file)

a_file.close()

#Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")


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
        st.subheader(str("Disease identified as "+prediction))
