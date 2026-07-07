
# ASL Sign Language Reader

A real-time American Sign Language recognition project built with Python, computer vision, and machine learning. The model uses image data of ASL hand signs to classify letters and can run live through a webcam.

## Overview

This project was built to explore how machine learning can be used for accessibility and real-time visual recognition. It combines image preprocessing, model training, saved model outputs, and webcam-based prediction into one end-to-end ML application.

## Dataset

This project was trained using the [ASL Alphabet dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

The full dataset is not included in this repository because it contains thousands of image files and would make the repository unnecessarily large. Instead, the dataset should be downloaded directly from Kaggle and placed locally in the expected folder structure before training.

Expected structure:
```text
asl/
├── asl_alphabet_train/
│   ├── A/
│   ├── B/
│   ├── C/
│   └── ...
└── asl_alphabet_test/
```text

## Features

- Trains a machine learning model on ASL alphabet image data
- Uses a webcam for live prediction
- Processes image input for classification
- Saves model metadata and output files
- Demonstrates a full ML workflow from dataset to real-time use

## Files

The following files are included for the following purposes:
- train_asl_model.py — trains the ASL recognition model
- webcam_asl.py — runs real-time webcam prediction
- asl_model_output/ — contains model output metadata

## Dataset Note

The full ASL image dataset is not included in this repository because of size limits. The model was trained locally using an ASL alphabet image dataset organized by letter folders.

## Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Computer vision
- Image classification

## What I Learned

Through this project, I learned how to structure an end-to-end computer vision project, train a model on image data, manage large datasets locally, and connect a trained model to a real-time webcam application. It's one of the coolest things I've built :)
