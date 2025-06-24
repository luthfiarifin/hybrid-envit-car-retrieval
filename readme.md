# Car Retrieval System for Indonesian Vehicles

This project is a comprehensive Car Retrieval System built to fulfill the requirements of the PNU ISLAB Technical AI Test 2025. The system is designed to detect multiple car instances from a video feed and then classify each detected vehicle by its type (e.g., MPV, Sedan, etc.).

The entire pipeline is broken down into four main stages, each represented by a Jupyter Notebook:

1.  **Data Collection**: Scraping a custom dataset of Indonesian cars from various online marketplaces.
2.  **Object Detection**: Fine-tuning a YOLO model to accurately detect cars in images.
3.  **Image Classification**: Training a custom model based on EfficientNet and attention mechanisms to classify the type of each detected car.
4.  **Inference**: Running the complete detection and classification pipeline on a test video.

## Project Objective

The primary objective is to create a dual-model system that first detects all car instances in a given video and then classifies the type of each detected car. The project mandates the use of separate models for detection and classification, with the classifier being developed in PyTorch or TensorFlow.

## Pipeline and Methodology

The project follows a sequential pipeline, from data acquisition to final model inference.

### 1\. Indonesian Car Image Dataset: Automated Scraping, Validation, and Preparation

*This notebook demonstrates a fully automated workflow for building a high-quality Indonesian car image dataset, ready for machine learning tasks such as classification and detection.*

The workflow covers scraping images from multiple sources, validating and cropping images, splitting the dataset, and generating comprehensive reports. Each step is designed to ensure the resulting dataset is clean, well-organized, and suitable for downstream ML projects.

  - **Notebook**: `1_run_scraper_into_dataset.ipynb`
  - **Sources Used**: OLX, Carmudi, Mobil123
  - **Overview**:
      - **Scraping**: Collect car images from multiple online sources, ensuring non-duplicate URLs.
      - **Validation & Cropping**: Check image quality and consistency, and crop to standardize.
      - **Dataset Splitting**: Organize validated images into training, validation, and test sets.
      - **Reporting**: Generate visual and tabular reports on scraping performance, validation status, and class distribution.

### 2\. Fine-tuning YOLOv12n for Vehicle Detection

*This notebook demonstrates the process of fine-tuning the YOLOv12n object detection model for custom vehicle detection tasks using a COCO-format dataset.*

This notebook provides a complete workflow for preparing data, training, and evaluating a YOLOv12n model on Indonesian vehicle images. The steps include environment setup, dataset preparation, model fine-tuning, and performance evaluation with visualizations.

  - **Notebook**: `2_finetune_the_detection_model.ipynb`
  - **Model Used**: **YOLOv12n**, a lightweight and efficient object detection model, suitable for real-time vehicle detection tasks.
  - **Overview**:
      - **Dataset**: COCO vehicle dataset, converted to YOLO format for training.
      - **Training Strategy**: Transfer learning using a pretrained YOLOv12n model with a custom training configuration.
      - **Evaluation**: Model performance assessment and confusion matrix visualization.

### 3\. Car Classification Model Training

*This notebook implements custom models based on the approach described in the paper: [EfficientNet with Hybrid Attention Mechanisms for Enhanced Breast Histopathology Classification: A Comprehensive Approach](https://arxiv.org/pdf/2410.22392v2).*

This notebook demonstrates the full pipeline for training and evaluating deep learning models for Indonesian car type classification using a custom dataset. The workflow includes dataset preparation, model training with class imbalance handling, evaluation, and visualization of results.

  - **Notebook**: `3_train_the_classification_model.ipynb`
  - **Dataset Overview**:
      - **Source**: The dataset was collected using a custom web scraper (see notebook `1_run_scraper_into_dataset.ipynb`) to gather Indonesian car images.
      - **Final Dataset**: The cleaned and organized dataset is available at [Kaggle: indonesian-cars-classification-dataset](https://www.kaggle.com/datasets/muhammadluthfiarifin/indonesian-cars-classification-dataset).
      - **Structure**: The dataset is split into train, validation, and test sets, each containing images for 8 car classes.
  - **Model Used**: **EfficientNetB4-CBAM**, which uses EfficientNet-B4 as the backbone and integrates Convolutional Block Attention Module (CBAM) blocks to enhance feature representation.
  - **Overview**:
      - **Classes**: 8 Indonesian car types (hatchback, mpv, offroad, pickup, sedan, suv, truck, van).
      - **Training Strategy**: Transfer learning, data augmentation, and weighted loss for class imbalance.
      - **Evaluation**: Per-class metrics, confusion matrix, and visualizations.

### 4\. Final Inference Pipeline

The final step is to integrate the two models into a single, cohesive pipeline that processes an input video and produces an annotated output video.

  - **Notebook**: `4_final_traffic_cam_test.ipynb`
  - **Description**: This notebook loads the input video (`traffic_test.mp4`). For each frame, it performs the following:
    1.  Uses the fine-tuned YOLO model to detect cars.
    2.  Crops the image of each detected car from the frame.
    3.  Passes the cropped image to the trained `EfficientNetB4-CBAM` model for classification.
    4.  Draws the bounding box and the predicted class label onto the output frame.
    5.  Saves the final annotated video.

## Project Structure

```
.
├── 1_run_scraper_into_dataset.ipynb
├── 2_finetune_the_detection_model.ipynb
├── 3_train_the_classification_model.ipynb
├── 4_final_traffic_cam_test.ipynb
├── data_processing
│   ├── coco_dataset
│   └── scraping
├── models
│   ├── classification
│   └── detection
├── traffic_test.mp4
└── requirements.txt
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/luthfiarifin/hybrid-envit-car-retrieval.git
    cd hybrid-envit-car-retrieval
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Pipeline

To execute the full pipeline from data collection to final inference, run the Jupyter notebooks in sequential order.

1.  **Run `1_run_scraper_into_dataset.ipynb`**: This will scrape the web to build the car dataset for the classification task.
2.  **Run `2_finetune_the_detection_model.ipynb`**: This will fine-tune the YOLOv12n object detection model.
3.  **Run `3_train_the_classification_model.ipynb`**: This will train the classification model on the scraped data.
4.  **Run `4_final_traffic_cam_test.ipynb`**: This will execute the final pipeline on the provided `traffic_test.mp4`, generating a `traffic_test_classified.mp4` with the results.

## Demo

The final output is a video where each detected car is enclosed in a bounding box with a label indicating its classified type.

<p align="center">
  <b>Sample Output:</b><br>
  <video src="traffic_test_classified.mp4" controls width="600"></video>
</p>

*(You can add a GIF or a screenshot of the output video here)*

## Disclaimer

The dataset provided and used in this project is intended solely for experimental and research purposes. Do not use the dataset or any derivative works for commercial purposes. The authors are not responsible for any misuse of the data.
