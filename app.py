import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms, models
from torch import nn
from ultralytics import YOLO

GITHUB_URL = "https://github.com/SnowdasNeves/cifar10-image-classifier"
CCT_URL = "https://lila.science/datasets/caltech-camera-traps"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar.html"

# Class dictionary for CIFAR-10
classes_cifar = {
    0: "airplane",
    1: "car",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

# Class dictionary for CCT20
classes_cct = {
    0: "bobcat",
    1: "opossum",
    2: "empty",
    3: "coyote",
    4: "raccoon",
    5: "bird",
    6: "dog",
    7: "cat",
    8: "squirrel",
    9: "rabbit",
    10: "skunk",
    11: "rodent",
    12: "car",  # car went from 14 to 12. Eliminated 12, 13 and 15
}


def classifier(image, threshold, class_dict, model_select):
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
        nn.Linear(512, len(class_dict)),
    )

    model.load_state_dict(
        torch.load(
            os.path.join("trained-models", model_select),
            weights_only=True,
            map_location=device,
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

    # Return prediction based on confidence threshold
    if max_probs.item() < threshold:
        return "none"
    else:
        if class_dict == classes_cifar and predicted.item() in {
            0,
            8,
            9,
        }:  # Removing vehicle predictions from CIFAR-10 trained model
            return "none"
        else:
            return class_dict[predicted.item()]


def detect_bounding_box(image_file, threshold):
    bbox_model = YOLO(os.path.join("trained-models", "yolo_bbox_detection.pt"))

    results = bbox_model.predict(image_file, threshold)
    results = list(results)
    results = results[0].boxes.xywh.tolist()[0]
    results[2], results[3] = results[0] + results[2], results[1] + results[3]
    return results


# Page design
st.set_page_config(page_title="Wildlife AI Classifier", page_icon=":elephant:")

if st.session_state.get("predictions") is None:
    st.session_state["predictions"] = {}

if st.session_state.get("images_cropped") is None:
    st.session_state["images_cropped"] = []

tab_selector = st.pills(
    "Tabs",
    ["Wildlife Classifier", "Bounding Box Detection", "Datasets"],
    default="Wildlife Classifier",
    selection_mode="single",
    label_visibility="collapsed",
)

if tab_selector == "Wildlife Classifier":
    st.title("Wildlife Camera Trap AI Classifier")
    st.write(
        """
        This app uses a custom ResNet18 model to classify animal species in camera trap images.
        To begin, upload images and click the 'Run Classifier' button. You can stay in this page
        for higher accuracy classification without automatic bounding box detection or head over
        to the 'Bounding Box Detection' tab for automatic bounding box detection at the cost of
        lower accuracy.
        """
    )
    st.write(
        "You can find additional information on the datasets used for training in the 'Datasets' tab."
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
        # st.info(
        #     "To clear all uploaded images press 'Clear Results' and then press the 'Home' tab twice."
        # )
        # st.divider()

        images = []
        for file in uploaded_files:
            image = Image.open(file)
            images.append(image)

        with st.expander("**Additional Settings**", expanded=True):
            st.write(
                "Adjust the confidence threshold to filter out low-confidence predictions (default value is 0.8)."
            )
            conf_threshold = st.slider(
                "Confidence Threshold.",
                0.0,
                1.0,
                0.8,
                0.01,
                label_visibility="collapsed",
            )

            show_images = st.toggle("Show uploaded images", False)

            if show_images:
                rows = math.ceil(len(uploaded_files) / 4)
                for row in range(rows):
                    cols = st.columns(4)
                    for col, file, image in zip(
                        cols,
                        uploaded_files[row * 4 : (row + 1) * 4],
                        images[row * 4 : (row + 1) * 4],
                    ):
                        col.image(image, caption=file.name, width=150)

        col1, col2 = st.columns(2)
        with col1:
            # Button to run the classifier
            run_button = st.button(
                "Run Classifier",
                use_container_width=True,
                type="primary",
            )

            if run_button:
                for file, image in zip(uploaded_files, images):
                    prediction = classifier(
                        image,
                        conf_threshold,
                        class_dict=classes_cifar,
                        model_select="cifar10_resnet18.pth",
                    )
                    predictions[file.name] = prediction

                st.session_state["predictions"] = predictions

        with col2:
            if st.button("Clear Results", use_container_width=True):
                st.session_state["predictions"] = {}

        predictions_df = pd.DataFrame(
            list(st.session_state.predictions.items()),
            columns=["File Name", "Identified Animal"],
        )

        # Creates a df to plot frequency of each class
        chart_df = pd.DataFrame(predictions_df["Identified Animal"].value_counts())

        if predictions_df.shape[0] > 0:
            st.write("## Results preview")

            # Downloads CSV file with all the info
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                "Download Complete Results to CSV file",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )

            st.write("Number of images uploaded:", len(uploaded_files))
            st.write("")

            st.dataframe(
                predictions_df.head(10), hide_index=True, use_container_width=True
            )
            st.markdown(
                f"""
                <p style='font-size:14px; text-align: end'>
                (showing the first {predictions_df.shape[0] if predictions_df.shape[0] < 10 else 10} results)
                </p>""",
                unsafe_allow_html=True,
            )

            more_info = st.pills(
                "More Info",
                ["More info"],
                selection_mode="single",
                label_visibility="collapsed",
            )

            if more_info:
                st.write("### Number of animals detected per species")
                st.write("")
                st.bar_chart(chart_df, x_label="Species", y_label="N. of occurrences")

    else:
        st.info("Please upload images to begin.")


if tab_selector == "Bounding Box Detection":
    st.title("Wildlife Camera Trap AI Classifier")
    st.write(
        """
        This app uses a custom ResNet18 model to classify animal species in camera trap images
        after identifying the subjects bounding box using a YOLOv5 based model.
        To begin, upload images and click the 'Run Classifier' button.
        You can stay in this page for for automatic bounding box detection at the cost of lower
        accuracy or head over to the 'Wildlife Classifier' tab for higher accuracy classification
        without automatic bounding box detection.
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
        # st.info(
        #     "To clear all uploaded images press 'Clear Results' and then press the 'Home' tab twice."
        # )
        # st.divider()

        images = []
        for file in uploaded_files:
            image = Image.open(file)
            images.append(image)

        with st.expander("**Additional Settings**", expanded=True):
            st.write(
                "Adjust the confidence threshold to filter out low-confidence predictions (default value is 0.8)."
            )
            conf_threshold = st.slider(
                "Confidence Threshold.",
                0.0,
                1.0,
                0.8,
                0.01,
                label_visibility="collapsed",
            )

            st.write(
                "Adjust the bbox detection confidence threshold (default value 0.5)."
            )
            bbox_conf_threshold = st.slider(
                "Confidence Threshold for BBox.",
                0.0,
                1.0,
                0.5,
                0.01,
                label_visibility="collapsed",
            )

            show_images = st.toggle("Show uploaded images", False)

            if show_images:
                rows = math.ceil(len(uploaded_files) / 4)
                for row in range(rows):
                    cols = st.columns(4)
                    for col, file, image in zip(
                        cols,
                        uploaded_files[row * 4 : (row + 1) * 4],
                        images[row * 4 : (row + 1) * 4],
                    ):
                        col.image(image, caption=file.name, width=150)

        col1, col2 = st.columns(2)
        with col1:
            # Button to run the classifier
            run_button = st.button(
                "Run Classifier",
                use_container_width=True,
                type="primary",
            )

            if run_button:
                for file, image in zip(uploaded_files, images):
                    bbox = detect_bounding_box(image, bbox_conf_threshold)
                    image_cropped = image.crop(bbox)
                    st.session_state.images_cropped.append(image_cropped)

                    prediction = classifier(
                        image_cropped,
                        conf_threshold,
                        class_dict=classes_cct,
                        model_select="cct20_resnet18.pth",
                    )
                    predictions[file.name] = prediction

                st.session_state["predictions"] = predictions

        with col2:
            if st.button("Clear Results", use_container_width=True):
                st.session_state["predictions"] = {}
                st.session_state["images_cropped"] = []

        predictions_df = pd.DataFrame(
            list(st.session_state.predictions.items()),
            columns=["File Name", "Identified Animal"],
        )

        # Creates a df to plot frequency of each class
        chart_df = pd.DataFrame(predictions_df["Identified Animal"].value_counts())

        if predictions_df.shape[0] > 0:
            st.write("## Results preview")

            # Downloads CSV file with all the info
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                "Download Complete Results to CSV file",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )

            st.write("Number of images uploaded:", len(uploaded_files))
            if st.toggle("Show detected bounding box", False):
                rows = math.ceil(len(uploaded_files) / 4)
                for row in range(rows):
                    cols = st.columns(4)
                    for col, file, image in zip(
                        cols,
                        uploaded_files[row * 4 : (row + 1) * 4],
                        st.session_state.images_cropped[row * 4 : (row + 1) * 4],
                    ):
                        col.image(image, caption=file.name, width=150)

            st.write("")

            st.dataframe(
                predictions_df.head(10), hide_index=True, use_container_width=True
            )
            st.markdown(
                f"""
                <p style='font-size:14px; text-align: end'>
                (showing the first {predictions_df.shape[0] if predictions_df.shape[0] < 10 else 10} results)
                </p>""",
                unsafe_allow_html=True,
            )

            more_info = st.pills(
                "More Info",
                ["More info"],
                selection_mode="single",
                label_visibility="collapsed",
            )

            if more_info:
                st.write("### Number of animals detected per species")
                st.write("")
                st.bar_chart(chart_df, x_label="Species", y_label="N. of occurrences")

    else:
        st.info("Please upload images to begin.")


if tab_selector == "Datasets":
    st.title("The CIFAR-10 and CCT20 Datasets")
    st.write(
        "To learn more about the datasets, visit [CIFAR-10](CIFAR_URL) and [CCT20](CCT_URL)."
    )
    st.write("")
