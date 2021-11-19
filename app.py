# Import libraries
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
predictor_model = load_model('./vgg.h5')
import matplotlib.pyplot as plt
import time

st.title('SHIP CLASSIFIER vs YOU')  #Title
fig = plt.figure()


#Main function to upload image and process prediction

def main():
    st.info("Just for fun! Check yourself if you can do better than Neural Network")
    st.write("Imagine yourself as a trained pilot who is trying to save the world   \n  "
             "by spotting the right kind of ship  \n  "
             "You got below information-  \n"
             "1. There are 3 types of ships below you  \n "
             "2. Your mission is to spot the ships right so army can attack only Battle ships"
             )
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])

    if file_uploaded is not None:
        file_bytes = np.asarray(bytearray(file_uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 0)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write(image.size)

    option = st.selectbox('Select the ship:', ['Bulk Carrier', 'Container', 'Tanker']) # User selection

    class_btn = st.button("Classify") # Button to classify

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                predictions = predict(image)
                if option == predictions:
                    st.write('You are right')
                    st.balloons()
                else:
                    st.write(option)
                    st.write(predictions)

                time.sleep(1)

'''
Load classifier and predict the ship type after 
clicking on 'classify'button
'''
def predict(image):
    classifier_model = "vgg.h5"
    model = load_model(classifier_model)
    img_r = cv2.resize(image, (128, 128))
    st.write(img_r.size)
    img_array = np.array(img_r)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    class_names = [
            'Bulk Carrier',
            'Container',
            'Tanker']

    result = f"{class_names[np.argmax(prediction)]} with a {(100 * np.max(prediction)).round(2)} % confidence."
    predicted_result = class_names[np.argmax(prediction)]
    st.write(result)
    return predicted_result

if __name__ == "__main__":
    main()