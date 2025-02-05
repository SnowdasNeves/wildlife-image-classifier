import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def dummy_classifier(image):
    return "Dummy prediction"


def plot_eda():
    # Bar plot EDA placeholder
    data = pd.DataFrame(
        {
            "Category": ["Cat", "Dog", "Bird", "Other"],
            "Count": np.random.randint(10, 100, size=4),
        }
    )
    fig, ax = plt.subplots()
    ax.bar(data["Category"], data["Count"], color="skyblue")
    ax.set_title("Dummy EDA")
    return fig


# Sidebar for navigation between pages
page = st.sidebar.selectbox(
    "Navigation", ["Home", "The CIFAR-10 Dataset", "The CCT20 Dataset"]
)

if page == "Home":
    st.title("Animal Monitoring Using CNNs for Image Classification")
    st.write("Upload a folder of images and run the classifier.")

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

        # Button to run the pre-trained classifier (dummy classifier used here)
        if st.button("Run Classifier"):
            st.write("### Predictions")
            predictions = {}
            for file, image in zip(uploaded_files, images):
                prediction = dummy_classifier(image)
                predictions[file.name] = prediction
            for fname, pred in predictions.items():
                st.write(f"**{fname}**: {pred}")
    else:
        st.info("Please upload images to begin.")

elif page == "The CIFAR-10 Dataset":
    st.title("CIFAR-10 Dataset Introduction")
    st.write(
        """
        The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per
        class.
        It is widely used for developing and benchmarking image classification models.
    """
    )

    st.write("### Exploratory Data Analysis (EDA)")
    st.write("Below is an example graph from our EDA:")
    eda_fig = plot_eda()
    st.pyplot(eda_fig)

elif page == "The CCT20 Dataset":
    st.title("CCT20 Dataset Introduction")
    st.write(
        """
        The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per
        class.
        It is widely used for developing and benchmarking image classification models.
    """
    )

    st.write("### Exploratory Data Analysis (EDA)")
    st.write("Below is an example graph from our EDA:")
    eda_fig = plot_eda()
    st.pyplot(eda_fig)
