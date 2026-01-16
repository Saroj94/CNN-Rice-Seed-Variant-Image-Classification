# Rice Image Classification

This project implements a Convolutional Neural Network (CNN) to classify different varieties of rice grains from images. The model is trained to distinguish between five common rice types, providing an automated solution for rice variety identification.

## Project Overview

The goal of this project is to build an image classification system that can accurately identify the type of rice from an image. This can be useful in various applications, such as quality control in agriculture, food processing, and research.

## Dataset

The model is trained on the **"Rice Image Dataset"** from Kaggle ([https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)).

This dataset contains images of five different rice varieties:
*   Arborio
*   Basmati
*   Ipsala
*   Jasmine
*   Karacadag

The dataset is downloaded and extracted programmatically within the notebook.

## Model Architecture

The classification model is a sequential CNN built using TensorFlow/Keras. The architecture includes:
*   **Input Layer**: Rescales image pixels to the range [0, 1].
*   **Convolutional Layers**: Two `Conv2D` layers (64 and 32 filters respectively) with `relu` activation and `MaxPool2D` layers for feature extraction.
*   **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
*   **Dropout Layer**: A `Dropout` layer (0.4 rate) is included to prevent overfitting.
*   **Dense Layers**: A hidden `Dense` layer with 16 neurons and `relu` activation, followed by an output `Dense` layer with `softmax` activation for multiclass classification.

## Training and Evaluation

### Constants:
*   `IMG_HEIGHT`, `IMG_WIDTH`: 180x180 pixels
*   `BATCH_SIZE`: 16
*   `EPOCHS`: 10

The model is compiled with the `Adam` optimizer (learning rate 0.01) and `sparse_categorical_crossentropy` loss, using `accuracy` as the evaluation metric.

### Training Results:
After 10 epochs of training, the model achieved the following performance:
*   **Validation Loss**: Approximately 0.079
*   **Validation Accuracy**: Approximately 0.975

These metrics indicate good performance in classifying the rice varieties.

## Usage and Prediction

To use the trained model for predicting the class of a new image:

1.  **Prepare your image**: Ensure your image is accessible in the Colab environment.
2.  **Load and Preprocess**: Use the `load_preprocess` function, which resizes the image to 224x224 pixels, converts it to a numpy array, adds a batch dimension, and normalizes pixel values.
3.  **Predict**: Call the `predict` function with the trained model, image path, and class names mapping. The function will display the image and print the predicted rice class.

```python
# Prediction Example:
path_image = '/content/your_image.jpg' # Replace with image path
predict_class_name = predict(model, path_image, class_nidx)
print(f'The predicted class is: {predict_class_name}')
```

## Setup and Installation

To run this notebook, you will need:

1.  **Google Colab Environment**: The notebook is designed to run in Google Colab.
2.  **Kaggle API Key**: To download the dataset, need a `kaggle.json` file containing Kaggle username and API key. Upload this file to Colab environment.

### Dependencies:
The necessary libraries are imported at the beginning of the notebook, including `tensorflow`, `keras`, `numpy`, `matplotlib`, `PIL`, `shutil`, `zipfile`, and `kaggle`.

To install `kaggle` if not already present:
```bash
!pip install kaggle
```

## Future work
Hyperparameter tuing would be the next step to perform for better performance of the model.

**Note**: The notebook includes checks for corrupted images, although none were found in the provided dataset.