# Emotion Classification in Social Media

This project compares different NLP models for classifying emotions in English social-media text.

The goal is to identify one of six emotions:

- sadness
- joy
- love
- anger
- fear
- surprise

The project compares five models:

- Logistic Regression
- Linear SVM
- Naive Bayes
- LSTM + Word2Vec
- DistilBERT

The models are trained and evaluated using the same train, validation, and test protocol so their performance can be compared fairly.

The main evaluation metric is Macro-F1 because the dataset is multiclass and imbalanced.

## What the project includes

The project covers the complete NLP pipeline:

- data loading and cleaning
- duplicate and overlap removal
- text preprocessing
- TF-IDF representation
- Word2Vec embeddings
- traditional machine-learning models
- LSTM training
- DistilBERT fine-tuning
- model comparison
- confusion matrices
- error analysis
- additional unigram vs. bigram experiment
- live model comparison in Streamlit

## Main result

DistilBERT achieved the best overall performance with a Macro-F1 of 0.8852.

The strongest traditional model was Linear SVM with a Macro-F1 of 0.8549.

The error analysis showed that some emotions are harder to distinguish, especially:

- joy vs. love
- surprise vs. fear
- sadness vs. anger

## Project files

`app.py`  
Runs the Streamlit dashboard and displays the project results.

`train_models.py`  
Contains the complete training and evaluation pipeline.

`utils.py`  
Contains helper functions used by the project.

`requirements.txt`  
Contains the Python libraries required to run the project.

`FINAL NLP.pdf`  
Contains the final written report.

## How to run the project

Enter to: https://upnlpfinalproject-fnfld69vfngupeq5o8snju.streamlit.app/
