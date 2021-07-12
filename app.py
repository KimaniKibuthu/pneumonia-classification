import cv2
import numpy as np
import streamlit as st
import tensorflow as tf


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("tuned_model.h5")
    return model


def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Pneumonia Prediction Web App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(" ", type=["jpeg", "jpg"])
    model = load_model()

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(opencv_image, (224, 224))
        st.image(opencv_image, channels="RGB")
        resized = tf.keras.applications.inception_v3.preprocess_input(resized)
        img_reshape = resized[np.newaxis, ...]

    if st.button("Predict"):
        prediction = model.predict(img_reshape, batch_size=1)
        if prediction < 0.5:
            output = 'be healthy'

        else:
            output = 'have Pneumonia'

        st.write("Classifying...")
        st.success('You are likely to {}.'.format(output))

    if st.button("Disclaimer"):
        disclaimer = 'This is an estimate as to whether you have pneumonia or not'
        st.success(disclaimer)


if __name__ == '__main__':
    main()
