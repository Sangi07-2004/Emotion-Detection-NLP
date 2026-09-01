# 🤖 Emotion Detection using NLP

### Machine Learning | Natural Language Processing | Streamlit

An NLP-based Machine Learning application that predicts the emotion expressed in a given text.

---

## 📌 Overview

Emotion Detection is a Natural Language Processing (NLP) task that aims to identify the emotional state expressed in text.

In this project, I built a **six-class emotion classification system** that takes a text sentence as input and predicts one of the following emotions:

**😢 Sadness · 😡 Anger · ❤️ Love · 😮 Surprise · 😨 Fear · 😊 Joy**

The project follows an end-to-end Machine Learning workflow:

**Text → Preprocessing → TF-IDF → Model Training → Model Comparison → Best Model → Prediction**

After training and evaluating multiple Machine Learning classification models, **Random Forest achieved the highest test accuracy of 88.44%** among the evaluated models. Therefore, Random Forest was selected as the final model and integrated into a Streamlit application.

---

## 🎯 Objective

The main objective of this project is to develop a Machine Learning system that can automatically detect emotions from text.

The application can:

- Accept text as user input
- Process the text using NLP techniques
- Convert text into numerical features using TF-IDF
- Classify the text into one of six emotions
- Display the predicted emotion through an interactive Streamlit interface

---

## 📊 Dataset

The dataset used in this project was obtained from **Kaggle**.

The dataset is provided in a `.txt` format and contains text samples along with their corresponding emotion labels.

### Emotion Classes

| Emotion | Label |
|---|---|
| 😢 Sadness | sadness |
| 😡 Anger | anger |
| ❤️ Love | love |
| 😮 Surprise | surprise |
| 😨 Fear | fear |
| 😊 Joy | joy |

> **Dataset Source:** Kaggle

---

## 🔄 Project Workflow

```text
                Dataset
                   │
                   ▼
            Data Preparation
                   │
                   ▼
           Text Preprocessing
                   │
                   ▼
            TF-IDF Vectorizer
                   │
                   ▼
          Feature Representation
                   │
                   ▼
        Multiple ML Classification
               Models
                   │
                   ▼
          Model Performance
             Comparison
                   │
                   ▼
        Random Forest Classifier
                   │
                   ▼
          Save Trained Model
                   │
                   ▼
          Streamlit Application
                   │
                   ▼
          Emotion Prediction