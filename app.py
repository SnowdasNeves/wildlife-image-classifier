import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms, models
from torch import nn

GITHUB_URL = "https://github.com/SnowdasNeves/cifar10-image-classifier"


def classifier(image, threshold):
    """Receives uploaded images and returns predictions."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=False)  # Base ResNet18 model

    # Custom fully connected layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 10),
    )

    model.load_state_dict(
        torch.load(
            os.path.join("trained-models", "cifar10_resnet4.pth"), weights_only=True
        )
    )

    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = nn.functional.softmax(output, dim=1)
        max_probs, predicted = torch.max(probabilities, 1)

    # Class dictionary for CIFAR-10
    classes = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    # Return prediction based on confidence threshold
    if max_probs.item() < threshold:
        return "none"
    else:
        return classes[predicted.item()]


# Page design
st.set_page_config(page_title="Wildlife AI Classifier", page_icon=":elephant:")

if st.session_state.get("predictions") is None:
    st.session_state["predictions"] = {}

tab_selector = st.pills(
    "Tabs",
    ["Home", "Datasets"],
    default="Home",
    selection_mode="single",
    label_visibility="collapsed",
)

if tab_selector == "Home":
    st.title("Wildlife Camera Trap AI Classifier")
    st.write(
        """
        This app uses a custom ResNet18 model to classify animal species in camera trap images.
        To begin, upload images and click the 'Run Classifier' button.
        You can find additional information on the datasets used for training in the 'Datasets' tab.
        """
    )
    st.write(
        "To learn more about the model, visit the project's [GitHub repo](GITHUB_URL)."
    )
    st.write("")

    st.write("### Upload images and run the classifier")
    uploaded_files = st.file_uploader(
        "Upload images and run the classifier",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    predictions = {}
    if uploaded_files:
        st.info(
            "If you want to clear all uploaded images just press the 'Home' tab twice."
        )
        st.write("")

        with st.expander("Classifier Settings", expanded=False):
            conf_threshold = st.slider(
                "Adjust the confidence threshold to filter out low-confidence predictions (default value is 0.8).",
                0.0,
                1.0,
                0.8,
                0.05,
            )

        images = []
        for file in uploaded_files:
            image = Image.open(file)
            images.append(image)

        # Button to run the classifier
        if st.button("Run Classifier", use_container_width=True, type="primary"):
            for file, image in zip(uploaded_files, images):
                prediction = classifier(image, conf_threshold)
                predictions[file.name] = prediction

            st.session_state["predictions"] = predictions

        predictions_df = pd.DataFrame(
            list(st.session_state.predictions.items()),
            columns=["File Name", "Identified Animal"],
        )

        # Creates a df to plot frequency of each class
        chart_df = pd.DataFrame(predictions_df["Identified Animal"].value_counts())

        st.divider()
        st.write("## Results preview")

        # Downloads CSV file with all the info
        if predictions_df.shape[0] > 0:
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                "Download Complete Results to CSV file",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )

        st.write("Number of images uploaded:", len(uploaded_files))
        st.write("")

        # col1, col2 = st.columns([1, 2])
        # with col1:
        st.dataframe(predictions_df.head(10), hide_index=True, use_container_width=True)
        st.markdown(
            f"<p style='font-size:14px; text-align: end'>(showing the first {predictions_df.shape[0]} results)</p>",
            unsafe_allow_html=True,
        )

        more_info = st.pills(
            "More Info",
            ["More info"],
            selection_mode="single",
            label_visibility="collapsed",
        )

        if more_info:
            # with col2:
            st.write("### Number of animals detected per species")
            st.write("")
            st.bar_chart(chart_df, x_label="Species", y_label="N. of occurrences")

    else:
        st.info("Please upload images to begin.")
