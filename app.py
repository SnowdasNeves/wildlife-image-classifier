import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def classifier():
    """Receives uploaded images and returns predictions."""
    pass


def plot_eda():
    """Plots EDA graphs for the CIFAR-10 and CCT20 datasets."""
    pass


# Page design
page = st.sidebar.selectbox(
    "Navigation", ["Home", "The CIFAR-10 Dataset", "The CCT20 Dataset"]
)


if page == "Home":
    st.title("Wildlife Monitoring with Convolutional Neural Networks")
    st.write("Upload images and run the classifier.")

    # Simulate folder upload using multiple file uploader
    uploaded_files = st.file_uploader(
        "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        st.write("### Uploaded Images")
        images = []
        for file in uploaded_files:
            image = Image.open(file)
            images.append(image)
            st.image(image, caption=file.name, width=150)

        # Button to run the classifier
        if st.button("Run Classifier"):
            st.write("### Predictions")
            predictions = {}
            for file, image in zip(uploaded_files, images):
                prediction = classifier(image)
                predictions[file.name] = prediction
            for fname, pred in predictions.items():
                st.write(f"**{fname}**: {pred}")
    else:
        st.info("Please upload images to begin.")


elif page == "The CIFAR-10 Dataset":
    st.title("The CIFAR-10 Dataset")
    st.write(
        """
        The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per
        class.
        It is widely used for developing and benchmarking image classification models.
    """
    )

    st.write("### Dataset Characteristics")
    st.write(
        """
                Below is an example graph from our EDA:
        """
    )
    eda_fig = plot_eda()
    st.pyplot(eda_fig)


elif page == "The CCT20 Dataset":
    st.title("The CCT20 Dataset")
    st.write(
        """
        The CCT20 dataset consists of 57,864 images of different species of North American mammals in the wild.
    """
    )

    st.write("### Dataset Characteristics")
    st.write(
        """
             Below is an example graph from our EDA:
    """
    )

    eda_fig = plot_eda()
    st.pyplot(eda_fig)
