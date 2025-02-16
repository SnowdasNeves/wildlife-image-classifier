# Wildlife Image Classification and Detection

This project uses deep learning models to classify wildlife images and detect bounding boxes around animals in camera trap images. The models were fine-tuned using the CIFAR-10 and CCT20 datasets and deployed via a Streamlit-based application.

## Project Structure

- **`yolo_bbox_detection.ipynb`**: Training of the YOLO model for bounding box detection.
- **`cct20_resnet.ipynb`**: Training of the ResNet18 model using the CCT20 dataset.
- **`cifar10_resnet.ipynb`**: Training of the ResNet18 model using the CIFAR-10 dataset.
- **`cifar10_eda.ipynb` & `cct20_eda.ipynb`**: Exploratory Data Analysis for CIFAR-10 and CCT20.
- **`app.py`**: Streamlit application for real-time classification and detection.
- **`requirements.txt`**: Dependencies for running the application.
- **`trained-models/`**: Directory containing the trained model weights.

## Models Used

1. **ResNet18**: Fine-tuned on CIFAR-10 and CCT20.
2. **YOLO**: Custom-trained for bounding box detection in camera trap images using the CCT20 benchmark subset.

## Datasets

- **[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)**: General low resolution, standardized image dataset with centered subject.
- **[CCT20](https://lila.science/datasets/caltech-camera-traps)**: Camera trap images of Southwestern USA animal species.

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd <repo_folder>

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

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

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)

Feel free to contribute to the project by submitting issues or pull requests!
