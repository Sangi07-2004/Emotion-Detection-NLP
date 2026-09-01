# 🤖 Emotion Detection using NLP

### Machine Learning | Natural Language Processing | Streamlit

An NLP-based Machine Learning application that predicts the emotion expressed in a given text.

---

## 📌 Overview

Emotion Detection is a Natural Language Processing (NLP) task that aims to identify the emotional state expressed in text.

In this project, I built a **six-class emotion classification system** that takes a text sentence as input and predicts one of the following emotions:

**😢 Sadness · 😡 Anger · ❤️ Love · 😮 Surprise · 😨 Fear · 😊 Joy**

The project follows an end-to-end Machine Learning workflow:

**Text → Preprocessing → TF-IDF → Multiple Model Training → Model Evaluation → Model Comparison → Hyperparameter Tuning → Best Model Selection → Prediction**

Instead of directly selecting a single Machine Learning algorithm, I **trained and evaluated multiple classification models** on the same dataset and compared their performance using evaluation metrics such as **Accuracy, Precision, Recall, and F1-Score**.

After comparing the different models, the best-performing model was selected for the final application. **Tuned Linear SVC achieved the highest test accuracy of 90.63%**, so it was selected as the final model and integrated into a Streamlit application.

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

> **Dataset Source:** *Kaggle*

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
          Model Evaluation
                   │
                   ▼
         Model Comparison
                   │
                   ▼
      Hyperparameter Tuning
                   │
                   ▼
        Best Model Selection
                   │
                   ▼
        Tuned Linear SVC
                   │
                   ▼
         Save Trained Model
                   │
                   ▼
        Streamlit Application
                   │
                   ▼
          Emotion Prediction