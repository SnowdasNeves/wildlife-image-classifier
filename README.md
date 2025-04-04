# Wildlife Image Classification and Detection

This project uses deep learning models to classify wildlife and detect bounding boxes around animals in camera trap images. The models were fine-tuned using the CIFAR-10 and CCT20 datasets and deployed via a Streamlit-based application.

To classify your own images without having to download the files in this repo you can access the [project's Hugging Face space](https://huggingface.co/spaces/diogoneves/wildlife-classifier).

## Project Structure

- **`dataset_analysis/`**: Exploratory Data Analysis for CIFAR-10 and CCT20.
- **`models/classification/cct20_resnet.ipynb`**: Training and testing of the ResNet18 model using the CCT20 dataset.
- **`models/classification/cifar10_resnet.ipynb`**: Training and testing of the ResNet18 model using the CIFAR-10 dataset.
- **`models/detection/yolo_bbox_detection.ipynb`**: Training of the YOLO model for bounding box detection.
- **`models/detection/yolo_bbox_detection_test.ipynb`**: Testing of the YOLO model for bounding box detection.
- **`saved-models/`**: Directory containing the trained model weights.
- **`streamlit_app/app.py`**: Streamlit application for real-time classification and detection.

## Models Used

1. **ResNet18**: Fine-tuned on CIFAR-10 and CCT20.
2. **YOLO**: Custom-trained for bounding box detection in camera trap images using the CCT20 benchmark subset.

Model evaluation metrics can be accessed in the respective testing files (see Project Structure above).

## Datasets

- **[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)**: General low resolution, standardized image dataset with centered subject. Can be downloaded directly from the `torchvision` library.
- **[CCT20](https://lila.science/datasets/caltech-camera-traps)**: Camera trap images of Southwestern USA animal species. The used benchmark subset can be downloaded to a local directory (6 GB). The code in `cct20_resnet.ipynb` assumes dataset images and metadata stored in `cct20-images` and `cct20-metadata` directories.

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd <repo_folder>

# Install dependencies
pip install -r requirements.txt
```

## Running the Application Locally

```bash
streamlit run app.py
```

## Application Features

1. **Wildlife Classifier**: Uses a custom version of ResNet18, fine-tuned on CIFAR-10, to classify uploaded images.
2. **Bounding Box Detection**: Applies YOLO to detect animals and then uses a custom version of ResNet18, trained on CCT20, to classify them.
3. **Dataset Information**: Provides insights into the CIFAR-10 and CCT20 datasets.

## Notes

- Make sure the `trained-models` folder contains the necessary model weights.
- Adjust the confidence thresholds as needed to balance precision and recall.

Feel free to contribute to the project by submitting issues or pull requests!
