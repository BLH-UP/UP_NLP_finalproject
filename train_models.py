#Step 1
#Project Introduction
#This project studies single-label multiclass emotion classification because each text belongs to exactly one of six emotion categories.
#The goal is to compare traditional statistical NLP, a neural sequence model, and a pretrained Transformer under the same train/validation/test protocol.

PROJECT_TITLE = "Comparing Traditional, Neural, and Transformer Models for Multiclass Emotion Classification"

RESEARCH_QUESTION_1 = (
    "How do traditional machine-learning models compare with neural and "
    "Transformer-based models for multiclass emotion classification?"
)

RESEARCH_QUESTION_2 = (
    "Which emotion classes are the most difficult to classify and what recurring "
    "error patterns appear?"
)

HYPOTHESIS = (
    "The pretrained Transformer model is expected to obtain the highest Macro-F1 "
    "because contextual representations can capture relationships between words "
    "more effectively than sparse TF-IDF representations."
)

print(PROJECT_TITLE)
print("\nTask: Multiclass emotion classification")
print("Language: English")
print("Labels: sadness, joy, love, anger, fear, surprise")

print("\nResearch Question 1:")
print(RESEARCH_QUESTION_1)

print("\nResearch Question 2:")
print(RESEARCH_QUESTION_2)

print("\nHypothesis:")
print(HYPOTHESIS)



#Step 3
#Import the required libraries and fix random seeds.
#Seeds improve reproducibility by reducing variation caused by random sampling and neural-network initialization.

import os
import re
import html
import copy
import time
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import pickle
import json

warnings.filterwarnings("ignore")


#Terminal-friendly replacement for Jupyter display().
def display(obj):
    print(obj)


from datasets import load_dataset

from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.utils.class_weight import compute_class_weight

from gensim.models import Word2Vec

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import (
    pack_padded_sequence
)

from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)



#Create project output folders.
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


#Step 4
#Describe the dataset source and usage conditions.
#This information is included because the final project requires a clear description of source, language, licensing/usage conditions, and limitations.

DATASET_SOURCE = "Hugging Face - dair-ai/emotion"

DATASET_HOMEPAGE = (
    "https://huggingface.co/datasets/dair-ai/emotion"
)

ORIGINAL_PROJECT = (
    "CARER: Contextualized Affect Representations "
    "for Emotion Recognition"
)

DATASET_AUTHORS = (
    "Saravia, Liu, Huang, Wu, and Chen (2018)"
)

DATASET_USAGE = (
    "The Hugging Face dataset card states that the dataset "
    "should be used for educational and research purposes only."
)

print("Dataset:", DATASET_SOURCE)
print("Original work:", ORIGINAL_PROJECT)
print("Citation:", DATASET_AUTHORS)
print("Usage condition:", DATASET_USAGE)

print("""
DATASET DESCRIPTION

Task:
Single-label multiclass emotion classification.

Source:
English Twitter messages.

Language:
English.

Classes:
sadness, joy, love, anger, fear, surprise.

Why this dataset was selected:
- It contains more than the required 5,000 labeled observations.
- It already defines a clear multiclass NLP task.
- It provides fixed train, validation, and test splits.
- Six classes allow meaningful class-wise and error analysis.
- The complete dataset is computationally manageable in Google Colab.

Important limitations:
- Emotion annotation is subjective.
- Some texts may reasonably contain more than one emotional interpretation.
- The classes are imbalanced.
- The data contain only English text.
- The source domain is social-media text, so results may not generalize to formal writing or other domains.

Ethical considerations:
- Predictions are text-classification outputs, not psychological diagnoses.
- The model should not be used for high-stakes decisions about individuals.
- Biases in the original social-media data may be learned by the models.
""")


#Step 5
#Load the complete dataset instead of using a subset.
#All 20,000 observations are retained because the dataset is manageable and keeping the full sample preserves minority-class observations.

dataset = load_dataset(
    "dair-ai/emotion",
    "split"
)

print(dataset)


#Step 6
#Explain the fixed train/validation/test protocol.
#The validation set is used for tuning, while the test set remains untouched until final evaluation to prevent information leakage.

print("""
FIXED EXPERIMENTAL PROTOCOL

TRAIN:
Used to learn model parameters.

VALIDATION:
Used to select hyperparameters and the best neural-network epoch.

TEST:
Used only for final model evaluation.

The test set is never used to select hyperparameters.
This avoids test-set leakage and keeps the final comparison fair.
""")


#Step 7
#Inspect dataset dimensions and variables.
#Basic EDA is required before preprocessing so we understand the number of observations, variables, and target categories.

label_names = (
    dataset["train"]
    .features["label"]
    .names
)

dataset_structure = []

for split in dataset.keys():

    dataset_structure.append({
        "Split": split,
        "Rows": len(dataset[split]),
        "Variables": len(
            dataset[split].column_names
        ),
        "Variable Names": ", ".join(
            dataset[split].column_names
        )
    })

dataset_structure_df = pd.DataFrame(
    dataset_structure
)

display(dataset_structure_df)

print(
    "Number of emotion classes:",
    len(label_names)
)

print(
    "Classes:",
    label_names
)


#Step 8
#Convert the predefined dataset splits to pandas DataFrames.
#Pandas makes EDA, preprocessing inspection, duplicate control, and error analysis easier to display.

train_df = (
    dataset["train"]
    .to_pandas()
)

val_df = (
    dataset["validation"]
    .to_pandas()
)

test_df = (
    dataset["test"]
    .to_pandas()
)

for df in [
    train_df,
    val_df,
    test_df
]:

    df["emotion"] = (
        df["label"]
        .map(
            lambda x: label_names[x]
        )
    )

print("Original Train:", len(train_df))
print("Original Validation:", len(val_df))
print("Original Test:", len(test_df))
print(
    "Original Total:",
    len(train_df)
    + len(val_df)
    + len(test_df)
)


#Step 9
#Inspect missing values and empty texts.
#Missing or empty text would provide no useful linguistic information and should be identified before modeling.

def inspect_missing(
    df,
    split_name
):

    return {
        "Split": split_name,
        "Missing Text": df[
            "text"
        ].isna().sum(),
        "Missing Labels": df[
            "label"
        ].isna().sum(),
        "Empty Text": (
            df["text"]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )
    }

missing_table = pd.DataFrame([
    inspect_missing(
        train_df,
        "Train"
    ),
    inspect_missing(
        val_df,
        "Validation"
    ),
    inspect_missing(
        test_df,
        "Test"
    )
])

display(missing_table)


#Step 10
#Inspect duplicate texts and conflicting duplicate labels.
#Duplicates may inflate performance, while identical texts assigned to different emotions indicate annotation ambiguity.

def duplicate_summary(
    df,
    split_name
):

    duplicated_mask = (
        df["text"]
        .duplicated(
            keep=False
        )
    )

    duplicate_rows = (
        duplicated_mask.sum()
    )

    conflicts = (
        df[
            duplicated_mask
        ]
        .groupby(
            "text"
        )["label"]
        .nunique()
    )

    conflicting_texts = (
        conflicts > 1
    ).sum()

    return {
        "Split": split_name,
        "Duplicated Rows": duplicate_rows,
        "Conflicting Duplicate Texts": conflicting_texts
    }

duplicate_table = pd.DataFrame([
    duplicate_summary(
        train_df,
        "Train"
    ),
    duplicate_summary(
        val_df,
        "Validation"
    ),
    duplicate_summary(
        test_df,
        "Test"
    )
])

display(duplicate_table)


#Step 11
#Remove internally conflicting duplicate texts and exact duplicate observations.
#Conflicting duplicate texts are removed because one identical input should not provide contradictory target supervision.

def remove_internal_duplicates(
    df
):

    label_counts = (
        df.groupby(
            "text"
        )["label"]
        .nunique()
    )

    conflicting_texts = (
        label_counts[
            label_counts > 1
        ]
        .index
    )

    cleaned_df = (
        df[
            ~df["text"]
            .isin(
                conflicting_texts
            )
        ]
        .copy()
    )

    cleaned_df = (
        cleaned_df
        .drop_duplicates(
            subset=[
                "text",
                "label"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return cleaned_df


train_df = remove_internal_duplicates(
    train_df
)

val_df = remove_internal_duplicates(
    val_df
)

test_df = remove_internal_duplicates(
    test_df
)

print(
    "Train after internal duplicate removal:",
    len(train_df)
)

print(
    "Validation after internal duplicate removal:",
    len(val_df)
)

print(
    "Test after internal duplicate removal:",
    len(test_df)
)


#Step 12
#Check and remove duplicate texts across train, validation, and test.
#Cross-split duplicates could allow the same text to appear during both training and final testing, producing optimistic evaluation results.

train_texts = set(
    train_df["text"]
)

val_texts = set(
    val_df["text"]
)

test_texts = set(
    test_df["text"]
)

overlap_table = pd.DataFrame({
    "Comparison": [
        "Train vs Validation",
        "Train vs Test",
        "Validation vs Test"
    ],

    "Overlapping Texts": [
        len(
            train_texts.intersection(
                val_texts
            )
        ),

        len(
            train_texts.intersection(
                test_texts
            )
        ),

        len(
            val_texts.intersection(
                test_texts
            )
        )
    ]
})

display(overlap_table)

val_df = (
    val_df[
        ~val_df["text"]
        .isin(
            set(
                train_df["text"]
            )
        )
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

test_df = (
    test_df[
        ~test_df["text"]
        .isin(
            set(
                train_df["text"]
            )
            .union(
                set(
                    val_df["text"]
                )
            )
        )
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

final_split_table = pd.DataFrame({
    "Split": [
        "Train",
        "Validation",
        "Test"
    ],

    "Final Samples": [
        len(train_df),
        len(val_df),
        len(test_df)
    ]
})

display(final_split_table)

print(
    "Final total:",
    len(train_df)
    + len(val_df)
    + len(test_df)
)


#Step 13
#Analyze class distribution and quantify imbalance.
#A majority-to-minority ratio substantially above 1 indicates that some emotions are represented much more often than others.

class_counts = (
    train_df[
        "emotion"
    ]
    .value_counts()
    .reindex(
        label_names
    )
)

class_percentages = (
    class_counts
    / class_counts.sum()
    * 100
)

class_distribution_df = pd.DataFrame({
    "Emotion": label_names,
    "Count": class_counts.values,
    "Percentage": class_percentages.values
})

display(
    class_distribution_df
    .round(2)
)

imbalance_ratio = (
    class_counts.max()
    / class_counts.min()
)

print(
    "Majority / minority ratio:",
    round(
        imbalance_ratio,
        2
    )
)

if imbalance_ratio >= 1.5:

    print(
        "Interpretation: The dataset is class-imbalanced."
    )

else:

    print(
        "Interpretation: The distribution is relatively balanced."
    )

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    class_distribution_df[
        "Emotion"
    ],
    class_distribution_df[
        "Count"
    ]
)

plt.title(
    "Training Class Distribution"
)

plt.xlabel(
    "Emotion"
)

plt.ylabel(
    "Number of Texts"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()
plt.show()


#Step 14
#Calculate balanced class weights.
#Minority emotions receive larger loss weights so neural models do not optimize mainly for the most frequent classes.

classes = np.arange(
    len(label_names)
)

class_weight_values = (
    compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df[
            "label"
        ].values
    )
)

class_weight_tensor = (
    torch.tensor(
        class_weight_values,
        dtype=torch.float32
    )
    .to(device)
)

class_weight_df = pd.DataFrame({
    "Emotion": label_names,
    "Weight": class_weight_values
})

display(
    class_weight_df
    .round(4)
)


#Step 15
#Analyze text length.
#Word-count statistics help determine reasonable sequence lengths for LSTM padding and Transformer truncation.

for df in [
    train_df,
    val_df,
    test_df
]:

    df[
        "word_count"
    ] = (
        df[
            "text"
        ]
        .astype(str)
        .str.split()
        .str.len()
    )

display(
    train_df[
        "word_count"
    ]
    .describe()
    .to_frame(
        "Word Count"
    )
)

plt.figure(
    figsize=(9, 5)
)

plt.hist(
    train_df[
        "word_count"
    ],
    bins=40
)

plt.title(
    "Distribution of Training Text Length"
)

plt.xlabel(
    "Number of Words"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()
plt.show()


#Step 16
#Show representative examples from every emotion.
#Human-readable examples help identify semantic overlap before evaluating model errors.

for emotion in label_names:

    print(
        "\nEMOTION:",
        emotion.upper()
    )

    subset = (
        train_df[
            train_df[
                "emotion"
            ] == emotion
        ]
    )

    examples = (
        subset[
            "text"
        ]
        .sample(
            min(
                3,
                len(subset)
            ),
            random_state=SEED
        )
    )

    for text in examples:

        print(
            "-",
            text
        )


#Step 17
#Demonstrate preprocessing visually.
#Traditional ML and Word2Vec benefit from normalized text, but aggressive normalization is avoided because negation and emotional wording may contain important information.

demo_text = (
    "I'M sooo HAPPY!!! Visit https://example.com "
    "and tell @Friend because I can't believe it!!!"
)

def lowercase_stage(
    text
):

    return str(
        text
    ).lower()


def url_user_stage(
    text
):

    text = re.sub(
        r"http\S+|www\.\S+",
        " url ",
        text
    )

    text = re.sub(
        r"@\w+",
        " user ",
        text
    )

    return text


def punctuation_noise_stage(
    text
):

    #Most punctuation is removed for sparse statistical representations.
    #Apostrophes are preserved because contractions may express negation.

    text = re.sub(
        r"[^\w\s']",
        " ",
        text
    )

    return text


def whitespace_stage(
    text
):

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


demo_lower = lowercase_stage(
    demo_text
)

demo_normalized = url_user_stage(
    demo_lower
)

demo_noise = punctuation_noise_stage(
    demo_normalized
)

demo_final = whitespace_stage(
    demo_noise
)

preprocessing_demo_df = pd.DataFrame({
    "Stage": [
        "Original",
        "Lowercasing",
        "URL/User Normalization",
        "Punctuation/Noise Handling",
        "Whitespace Normalization"
    ],

    "Text": [
        demo_text,
        demo_lower,
        demo_normalized,
        demo_noise,
        demo_final
    ]
})

display(preprocessing_demo_df)


#Step 18
#Create the preprocessing function used by traditional ML and Word2Vec/LSTM.
#Stopword removal and stemming are intentionally avoided because words such as "not", "very", and "feel" may contribute to emotion classification.

def preprocess_classic(
    text
):

    text = html.unescape(
        str(text)
    )

    text = lowercase_stage(
        text
    )

    text = url_user_stage(
        text
    )

    text = punctuation_noise_stage(
        text
    )

    text = whitespace_stage(
        text
    )

    return text


for df in [
    train_df,
    val_df,
    test_df
]:

    df[
        "classic_text"
    ] = (
        df[
            "text"
        ]
        .apply(
            preprocess_classic
        )
    )

display(
    train_df[
        [
            "text",
            "classic_text",
            "emotion"
        ]
    ]
    .sample(
        10,
        random_state=SEED
    )
)


#Step 19
#Demonstrate word-level tokenization.
#Word2Vec and LSTM operate on ordered word sequences, so a word tokenizer is used for those models.

def word_tokenize_simple(
    text
):

    return re.findall(
        r"\b[\w']+\b",
        str(text).lower()
    )


token_demo_text = (
    "I can't believe I feel so happy today!"
)

word_tokens_demo = (
    word_tokenize_simple(
        preprocess_classic(
            token_demo_text
        )
    )
)

print(
    "Original text:"
)

print(
    token_demo_text
)

print(
    "\nWord-level tokens:"
)

print(
    word_tokens_demo
)


#Step 20
#Demonstrate DistilBERT subword tokenization.
#Transformers use pretrained subword tokenizers, so DistilBERT receives the original text rather than the aggressively normalized traditional representation.

TRANSFORMER_NAME = (
    "distilbert-base-uncased"
)

bert_tokenizer = (
    AutoTokenizer
    .from_pretrained(
        TRANSFORMER_NAME
    )
)

subword_tokens_demo = (
    bert_tokenizer
    .tokenize(
        token_demo_text
    )
)

tokenization_demo_df = pd.DataFrame({
    "Representation": [
        "Word-level tokenization",
        "DistilBERT subword tokenization"
    ],

    "Tokens": [
        str(
            word_tokens_demo
        ),
        str(
            subword_tokens_demo
        )
    ]
})

display(
    tokenization_demo_df
)


#Step 21
#Create the target arrays shared by every model.
#Using identical labels and final splits ensures that all final results are directly comparable.

y_train = (
    train_df[
        "label"
    ]
    .values
)

y_val = (
    val_df[
        "label"
    ]
    .values
)

y_test = (
    test_df[
        "label"
    ]
    .values
)

NUM_CLASSES = len(
    label_names
)

print(
    "Train labels:",
    len(y_train)
)

print(
    "Validation labels:",
    len(y_val)
)

print(
    "Test labels:",
    len(y_test)
)


#Step 22
#Create the TF-IDF representation.
#TF-IDF is selected because traditional classifiers cannot use raw text directly and TF-IDF captures term importance while reducing the influence of words common across many documents.

print("""
TF-IDF CONCEPT

TF:
How frequently a term appears in one document.

IDF:
How rare that term is across the collection.

TF-IDF:
Terms that are frequent in one document but uncommon across documents receive higher weights.
""")

tfidf_vectorizer = (
    TfidfVectorizer(
        lowercase=False,
        max_features=20000,
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
)

X_train_tfidf = (
    tfidf_vectorizer
    .fit_transform(
        train_df[
            "classic_text"
        ]
    )
)

X_val_tfidf = (
    tfidf_vectorizer
    .transform(
        val_df[
            "classic_text"
        ]
    )
)

X_test_tfidf = (
    tfidf_vectorizer
    .transform(
        test_df[
            "classic_text"
        ]
    )
)

print(
    "TF-IDF Train shape:",
    X_train_tfidf.shape
)

print(
    "TF-IDF Validation shape:",
    X_val_tfidf.shape
)

print(
    "TF-IDF Test shape:",
    X_test_tfidf.shape
)

print(
    "Vocabulary size:",
    len(
        tfidf_vectorizer
        .vocabulary_
    )
)


#Step 23
#Show a real TF-IDF example.
#Displaying individual term weights makes the representation interpretable rather than treating vectorization as a black box.

example_index = 0

example_text = (
    train_df
    .iloc[
        example_index
    ][
        "classic_text"
    ]
)

example_vector = (
    X_train_tfidf[
        example_index
    ]
)

feature_names = (
    tfidf_vectorizer
    .get_feature_names_out()
)

nonzero_positions = (
    example_vector
    .nonzero()[1]
)

tfidf_example_df = pd.DataFrame({
    "Term": feature_names[
        nonzero_positions
    ],

    "TF-IDF": example_vector[
        0,
        nonzero_positions
    ]
    .toarray()
    .flatten()
})

tfidf_example_df = (
    tfidf_example_df
    .sort_values(
        "TF-IDF",
        ascending=False
    )
)

print(
    "Example text:"
)

print(
    example_text
)

print(
    "\nHighest TF-IDF terms:"
)

display(
    tfidf_example_df
    .head(15)
    .round(4)
)


#Step 24
#Modeling Overview
#Logistic Regression, Linear SVM, and Multinomial Naive Bayes use TF-IDF; LSTM uses Word2Vec embeddings learned from the training corpus; DistilBERT uses pretrained contextual Transformer representations.
#This design compares statistical, neural-sequential, and pretrained contextual NLP approaches using the same final test set.


#Step 25
#Create a common evaluation function.
#Macro metrics are included because the dataset is imbalanced and each emotion should contribute equally to the main evaluation.

def evaluate_model(
    model_name,
    y_true,
    y_pred,
    training_time
):

    accuracy = (
        accuracy_score(
            y_true,
            y_pred
        )
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    _, _, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision Macro": precision_macro,
        "Recall Macro": recall_macro,
        "Macro F1": f1_macro,
        "Weighted F1": f1_weighted,
        "Training Time (s)": training_time
    }


final_results = []

prediction_dictionary = {}


#Step 26
#Tune Logistic Regression on validation Macro-F1.
#Logistic Regression is selected because it is a strong and interpretable linear baseline for sparse TF-IDF features.

lr_candidates = [
    0.5,
    1.0,
    2.0,
    4.0
]

lr_validation_rows = []

for c in lr_candidates:

    model = (
        LogisticRegression(
            C=c,
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED
        )
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    pred = (
        model
        .predict(
            X_val_tfidf
        )
    )

    _, _, f1_value, _ = (
        precision_recall_fscore_support(
            y_val,
            pred,
            average="macro",
            zero_division=0
        )
    )

    lr_validation_rows.append({
        "C": c,
        "Validation Macro F1": f1_value
    })

lr_validation_df = (
    pd.DataFrame(
        lr_validation_rows
    )
)

display(
    lr_validation_df
    .round(4)
)

best_lr_c = (
    lr_validation_df
    .sort_values(
        "Validation Macro F1",
        ascending=False
    )
    .iloc[0][
        "C"
    ]
)

print(
    "Selected C:",
    best_lr_c
)


#Step 27
#Train and evaluate the final Logistic Regression model.
#The test set is used only after validation has selected the hyperparameter.

start_time = (
    time.perf_counter()
)

logistic_model = (
    LogisticRegression(
        C=best_lr_c,
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED
    )
)

logistic_model.fit(
    X_train_tfidf,
    y_train
)

lr_training_time = (
    time.perf_counter()
    - start_time
)

logistic_test_pred = (
    logistic_model
    .predict(
        X_test_tfidf
    )
)

prediction_dictionary[
    "Logistic Regression"
] = logistic_test_pred

final_results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_test_pred,
        lr_training_time
    )
)

display(
    pd.DataFrame(
        [
            final_results[-1]
        ]
    )
    .round(4)
)


#Step 28
#Tune Linear SVM using validation Macro-F1.
#Linear SVM is selected because margin-based linear classifiers often perform strongly with sparse high-dimensional text representations.

svm_candidates = [
    0.25,
    0.5,
    1.0,
    2.0
]

svm_validation_rows = []

for c in svm_candidates:

    model = (
        LinearSVC(
            C=c,
            class_weight="balanced",
            random_state=SEED
        )
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    pred = (
        model
        .predict(
            X_val_tfidf
        )
    )

    _, _, f1_value, _ = (
        precision_recall_fscore_support(
            y_val,
            pred,
            average="macro",
            zero_division=0
        )
    )

    svm_validation_rows.append({
        "C": c,
        "Validation Macro F1": f1_value
    })

svm_validation_df = pd.DataFrame(
    svm_validation_rows
)

display(
    svm_validation_df
    .round(4)
)

best_svm_c = (
    svm_validation_df
    .sort_values(
        "Validation Macro F1",
        ascending=False
    )
    .iloc[0][
        "C"
    ]
)

print(
    "Selected C:",
    best_svm_c
)


#Step 29
#Train and evaluate the final Linear SVM.
#Class weighting is used because the emotion classes are not equally represented.

start_time = (
    time.perf_counter()
)

svm_model = (
    LinearSVC(
        C=best_svm_c,
        class_weight="balanced",
        random_state=SEED
    )
)

svm_model.fit(
    X_train_tfidf,
    y_train
)

svm_training_time = (
    time.perf_counter()
    - start_time
)

svm_test_pred = (
    svm_model
    .predict(
        X_test_tfidf
    )
)

prediction_dictionary[
    "Linear SVM"
] = svm_test_pred

final_results.append(
    evaluate_model(
        "Linear SVM",
        y_test,
        svm_test_pred,
        svm_training_time
    )
)

display(
    pd.DataFrame(
        [
            final_results[-1]
        ]
    )
    .round(4)
)


#Step 30
#Tune Multinomial Naive Bayes.
#Naive Bayes is included because it is a classic probabilistic text-classification baseline and works naturally with non-negative TF-IDF features.

nb_rows = []

for alpha_value in [
    0.1,
    0.5,
    1.0,
    2.0
]:

    for prior_setting in [
        True,
        False
    ]:

        model = (
            MultinomialNB(
                alpha=alpha_value,
                fit_prior=prior_setting
            )
        )

        model.fit(
            X_train_tfidf,
            y_train
        )

        pred = (
            model
            .predict(
                X_val_tfidf
            )
        )

        _, _, f1_value, _ = (
            precision_recall_fscore_support(
                y_val,
                pred,
                average="macro",
                zero_division=0
            )
        )

        nb_rows.append({
            "Alpha": alpha_value,
            "Fit Prior": prior_setting,
            "Validation Macro F1": f1_value
        })

nb_validation_df = (
    pd.DataFrame(
        nb_rows
    )
)

display(
    nb_validation_df
    .sort_values(
        "Validation Macro F1",
        ascending=False
    )
    .round(4)
)

best_nb_row = (
    nb_validation_df
    .sort_values(
        "Validation Macro F1",
        ascending=False
    )
    .iloc[0]
)

best_nb_alpha = (
    best_nb_row[
        "Alpha"
    ]
)

best_nb_prior = bool(
    best_nb_row[
        "Fit Prior"
    ]
)

print(
    "Selected alpha:",
    best_nb_alpha
)

print(
    "Selected fit_prior:",
    best_nb_prior
)


#Step 31
#Train and evaluate the final Multinomial Naive Bayes classifier.
#The validation-selected prior setting helps determine whether learned class priors or uniform priors are more appropriate for this imbalanced dataset.

start_time = (
    time.perf_counter()
)

nb_model = (
    MultinomialNB(
        alpha=best_nb_alpha,
        fit_prior=best_nb_prior
    )
)

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_training_time = (
    time.perf_counter()
    - start_time
)

nb_test_pred = (
    nb_model
    .predict(
        X_test_tfidf
    )
)

prediction_dictionary[
    "Naive Bayes"
] = nb_test_pred

final_results.append(
    evaluate_model(
        "Naive Bayes",
        y_test,
        nb_test_pred,
        nb_training_time
    )
)

display(
    pd.DataFrame(
        [
            final_results[-1]
        ]
    )
    .round(4)
)


#Step 32
#Create learning curves for the three traditional models.
#Traditional classifiers do not train over epochs, so their learning curves show validation Macro-F1 as progressively more training observations are used.

traditional_models = {
    "Logistic Regression": (
        LogisticRegression(
            C=best_lr_c,
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED
        )
    ),

    "Linear SVM": (
        LinearSVC(
            C=best_svm_c,
            class_weight="balanced",
            random_state=SEED
        )
    ),

    "Naive Bayes": (
        MultinomialNB(
            alpha=best_nb_alpha,
            fit_prior=best_nb_prior
        )
    )
}

training_fractions = [
    0.20,
    0.40,
    0.60,
    0.80,
    1.00
]

traditional_learning_curves = {}

for model_name, base_model in traditional_models.items():

    rows = []

    for fraction in training_fractions:

        if fraction < 1:

            splitter = (
                StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=fraction,
                    random_state=SEED
                )
            )

            subset_index, _ = next(
                splitter.split(
                    X_train_tfidf,
                    y_train
                )
            )

            X_subset = (
                X_train_tfidf[
                    subset_index
                ]
            )

            y_subset = (
                y_train[
                    subset_index
                ]
            )

        else:

            X_subset = (
                X_train_tfidf
            )

            y_subset = (
                y_train
            )

        model = clone(
            base_model
        )

        model.fit(
            X_subset,
            y_subset
        )

        pred = (
            model
            .predict(
                X_val_tfidf
            )
        )

        _, _, f1_value, _ = (
            precision_recall_fscore_support(
                y_val,
                pred,
                average="macro",
                zero_division=0
            )
        )

        rows.append({
            "Training Fraction": fraction,
            "Training Samples": len(
                y_subset
            ),
            "Validation Macro F1": f1_value
        })

    curve_df = pd.DataFrame(
        rows
    )

    traditional_learning_curves[
        model_name
    ] = curve_df

    display(
        curve_df
        .round(4)
    )

    plt.figure(
        figsize=(7, 4)
    )

    plt.plot(
        curve_df[
            "Training Samples"
        ],
        curve_df[
            "Validation Macro F1"
        ],
        marker="o"
    )

    plt.title(
        f"Learning Curve - {model_name}"
    )

    plt.xlabel(
        "Training Samples"
    )

    plt.ylabel(
        "Validation Macro F1"
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()
    plt.show()


#Step 33
#Train Word2Vec embeddings using only the training corpus.
#Word2Vec is not used as a classifier by itself; it learns dense semantic word vectors that will initialize the LSTM embedding layer.

tokenized_train_corpus = [
    word_tokenize_simple(
        text
    )
    for text
    in train_df[
        "classic_text"
    ]
]

WORD2VEC_DIM = 100

WORD2VEC_WINDOW = 5

WORD2VEC_MIN_COUNT = 2

WORD2VEC_EPOCHS = 20

word2vec_model = (
    Word2Vec(
        sentences=tokenized_train_corpus,
        vector_size=WORD2VEC_DIM,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        workers=1,
        sg=1,
        epochs=WORD2VEC_EPOCHS,
        seed=SEED
    )
)

print(
    "Word2Vec vocabulary size:",
    len(
        word2vec_model
        .wv
        .index_to_key
    )
)

print(
    "Embedding dimension:",
    WORD2VEC_DIM
)


#Step 34
#Show an example of semantic similarity learned by Word2Vec.
#Nearest neighbors make it visible that Word2Vec represents semantic relationships rather than simple counts.

for candidate_word in [
    "happy",
    "sad",
    "love",
    "angry",
    "fear"
]:

    if candidate_word in (
        word2vec_model
        .wv
    ):

        print(
            "\nNearest words to:",
            candidate_word
        )

        print(
            word2vec_model
            .wv
            .most_similar(
                candidate_word,
                topn=5
            )
        )

        break


#Step 35
#Build the LSTM vocabulary and embedding matrix.
#The matrix transfers the Word2Vec vectors learned from the training corpus into the LSTM input layer.

word2idx = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word in (
    word2vec_model
    .wv
    .index_to_key
):

    word2idx[
        word
    ] = len(
        word2idx
    )

embedding_matrix = (
    np.zeros(
        (
            len(
                word2idx
            ),
            WORD2VEC_DIM
        ),
        dtype=np.float32
    )
)

mean_embedding = (
    np.mean(
        word2vec_model
        .wv
        .vectors,
        axis=0
    )
)

embedding_matrix[
    word2idx[
        "<UNK>"
    ]
] = mean_embedding

for word, index in word2idx.items():

    if word in (
        word2vec_model
        .wv
    ):

        embedding_matrix[
            index
        ] = (
            word2vec_model
            .wv[
                word
            ]
        )

print(
    "LSTM vocabulary size:",
    len(
        word2idx
    )
)

print(
    "Embedding matrix shape:",
    embedding_matrix.shape
)


#Step 36
#Choose the LSTM maximum sequence length from the 95th percentile of training texts.
#This covers most observations while avoiding excessive padding caused by rare very long texts.

train_token_lengths = [
    len(
        word_tokenize_simple(
            text
        )
    )
    for text
    in train_df[
        "classic_text"
    ]
]

LSTM_MAX_LEN = int(
    np.percentile(
        train_token_lengths,
        95
    )
)

LSTM_MAX_LEN = max(
    10,
    min(
        LSTM_MAX_LEN,
        64
    )
)

print(
    "LSTM maximum sequence length:",
    LSTM_MAX_LEN
)


#Step 37
#Convert each text to a padded sequence while preserving its true sequence length.
#The real length is required so the LSTM can ignore padding instead of treating PAD tokens as meaningful language.

def text_to_lstm_sequence(
    text
):

    tokens = (
        word_tokenize_simple(
            preprocess_classic(
                text
            )
        )
    )

    sequence = [
        word2idx.get(
            token,
            word2idx[
                "<UNK>"
            ]
        )
        for token
        in tokens
    ]

    sequence = (
        sequence[
            :LSTM_MAX_LEN
        ]
    )

    real_length = max(
        1,
        len(
            sequence
        )
    )

    if len(
        sequence
    ) < LSTM_MAX_LEN:

        sequence = (
            sequence
            + [
                word2idx[
                    "<PAD>"
                ]
            ]
            * (
                LSTM_MAX_LEN
                - len(
                    sequence
                )
            )
        )

    return (
        sequence,
        real_length
    )


class EmotionLSTMDataset(
    Dataset
):

    def __init__(
        self,
        texts,
        labels
    ):

        self.texts = list(
            texts
        )

        self.labels = (
            np.array(
                labels
            )
        )

    def __len__(
        self
    ):

        return len(
            self.texts
        )

    def __getitem__(
        self,
        index
    ):

        sequence, length = (
            text_to_lstm_sequence(
                self.texts[
                    index
                ]
            )
        )

        return {
            "input_ids": (
                torch.tensor(
                    sequence,
                    dtype=torch.long
                )
            ),

            "length": (
                torch.tensor(
                    length,
                    dtype=torch.long
                )
            ),

            "label": (
                torch.tensor(
                    self.labels[
                        index
                    ],
                    dtype=torch.long
                )
            )
        }


train_lstm_dataset = (
    EmotionLSTMDataset(
        train_df[
            "text"
        ],
        y_train
    )
)

val_lstm_dataset = (
    EmotionLSTMDataset(
        val_df[
            "text"
        ],
        y_val
    )
)

test_lstm_dataset = (
    EmotionLSTMDataset(
        test_df[
            "text"
        ],
        y_test
    )
)


#Step 38
#Create LSTM DataLoaders.
#Training batches are shuffled, while validation and test batches preserve stable evaluation order.

LSTM_BATCH_SIZE = 128

train_lstm_loader = (
    DataLoader(
        train_lstm_dataset,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=True
    )
)

val_lstm_loader = (
    DataLoader(
        val_lstm_dataset,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False
    )
)

test_lstm_loader = (
    DataLoader(
        test_lstm_dataset,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False
    )
)


#Step 39
#Define the corrected Word2Vec + LSTM classifier.
#Packed sequences explicitly ignore padded positions, preventing the final LSTM state from being influenced by artificial PAD tokens.

class Word2VecLSTMClassifier(
    nn.Module
):

    def __init__(
        self,
        embedding_matrix,
        hidden_dim,
        num_classes,
        dropout
    ):

        super().__init__()

        embedding_tensor = (
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
            )
        )

        self.embedding = (
            nn.Embedding
            .from_pretrained(
                embedding_tensor,
                freeze=False,
                padding_idx=0
            )
        )

        self.lstm = (
            nn.LSTM(
                input_size=embedding_matrix.shape[
                    1
                ],
                hidden_size=hidden_dim,
                batch_first=True
            )
        )

        self.dropout = (
            nn.Dropout(
                dropout
            )
        )

        self.classifier = (
            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    def forward(
        self,
        input_ids,
        lengths
    ):

        embedded = (
            self.embedding(
                input_ids
            )
        )

        packed = (
            pack_padded_sequence(
                embedded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
        )

        _, (
            hidden,
            _
        ) = (
            self.lstm(
                packed
            )
        )

        final_hidden = (
            hidden[
                -1
            ]
        )

        final_hidden = (
            self.dropout(
                final_hidden
            )
        )

        logits = (
            self.classifier(
                final_hidden
            )
        )

        return logits


LSTM_HIDDEN_DIM = 128

LSTM_DROPOUT = 0.30

LSTM_LEARNING_RATE = 0.001

LSTM_MAX_EPOCHS = 20

EARLY_STOPPING_PATIENCE = 3

lstm_model = (
    Word2VecLSTMClassifier(
        embedding_matrix=embedding_matrix,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=LSTM_DROPOUT
    )
    .to(device)
)

print(
    lstm_model
)


#Step 40
#Train the corrected LSTM using weighted loss and early stopping.
#A maximum of 20 epochs gives the model enough time to learn, while early stopping prevents unnecessary training after validation Macro-F1 stops improving.

lstm_loss_function = (
    nn.CrossEntropyLoss(
        weight=class_weight_tensor
    )
)

lstm_optimizer = (
    torch.optim.Adam(
        lstm_model.parameters(),
        lr=LSTM_LEARNING_RATE
    )
)

best_lstm_f1 = -1

best_lstm_state = None

best_lstm_epoch = None

epochs_without_improvement = 0

lstm_history = []

start_time = (
    time.perf_counter()
)

for epoch in range(
    LSTM_MAX_EPOCHS
):

    lstm_model.train()

    epoch_train_loss = 0

    for batch in (
        train_lstm_loader
    ):

        input_ids = (
            batch[
                "input_ids"
            ]
            .to(device)
        )

        lengths = (
            batch[
                "length"
            ]
        )

        labels = (
            batch[
                "label"
            ]
            .to(device)
        )

        lstm_optimizer.zero_grad()

        logits = (
            lstm_model(
                input_ids,
                lengths
            )
        )

        loss = (
            lstm_loss_function(
                logits,
                labels
            )
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            lstm_model.parameters(),
            max_norm=1.0
        )

        lstm_optimizer.step()

        epoch_train_loss += (
            loss.item()
        )

    average_train_loss = (
        epoch_train_loss
        / len(
            train_lstm_loader
        )
    )

    lstm_model.eval()

    validation_predictions = []

    validation_labels = []

    with torch.no_grad():

        for batch in (
            val_lstm_loader
        ):

            input_ids = (
                batch[
                    "input_ids"
                ]
                .to(device)
            )

            lengths = (
                batch[
                    "length"
                ]
            )

            labels = (
                batch[
                    "label"
                ]
                .to(device)
            )

            logits = (
                lstm_model(
                    input_ids,
                    lengths
                )
            )

            predictions = (
                torch.argmax(
                    logits,
                    dim=1
                )
            )

            validation_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )

            validation_labels.extend(
                labels
                .cpu()
                .numpy()
            )

    _, _, validation_macro_f1, _ = (
        precision_recall_fscore_support(
            validation_labels,
            validation_predictions,
            average="macro",
            zero_division=0
        )
    )

    lstm_history.append({
        "Epoch": epoch + 1,
        "Train Loss": average_train_loss,
        "Validation Macro F1": validation_macro_f1
    })

    print(
        f"Epoch {epoch + 1}/{LSTM_MAX_EPOCHS} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Validation Macro F1: {validation_macro_f1:.4f}"
    )

    if (
        validation_macro_f1
        >
        best_lstm_f1
    ):

        best_lstm_f1 = (
            validation_macro_f1
        )

        best_lstm_epoch = (
            epoch + 1
        )

        best_lstm_state = (
            copy.deepcopy(
                lstm_model
                .state_dict()
            )
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    if (
        epochs_without_improvement
        >=
        EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping activated."
        )

        print(
            "Best epoch:",
            best_lstm_epoch
        )

        break

lstm_training_time = (
    time.perf_counter()
    - start_time
)

lstm_history_df = pd.DataFrame(
    lstm_history
)

display(
    lstm_history_df
    .round(4)
)


#Step 41
#Plot the LSTM learning curves.
#The curves show both optimization progress and whether validation performance eventually stops improving.

plt.figure(
    figsize=(7, 4)
)

plt.plot(
    lstm_history_df[
        "Epoch"
    ],
    lstm_history_df[
        "Train Loss"
    ],
    marker="o"
)

plt.title(
    "LSTM + Word2Vec Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Training Loss"
)

plt.tight_layout()
plt.show()


plt.figure(
    figsize=(7, 4)
)

plt.plot(
    lstm_history_df[
        "Epoch"
    ],
    lstm_history_df[
        "Validation Macro F1"
    ],
    marker="o"
)

plt.title(
    "LSTM + Word2Vec Learning Curve"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Validation Macro F1"
)

plt.ylim(
    0,
    1
)

plt.tight_layout()
plt.show()


#Step 42
#Evaluate the best LSTM checkpoint on the untouched test set.
#Only the validation-selected checkpoint is used, maintaining the separation between model selection and final evaluation.

lstm_model.load_state_dict(
    best_lstm_state
)

lstm_model.eval()

lstm_test_predictions = []

with torch.no_grad():

    for batch in (
        test_lstm_loader
    ):

        input_ids = (
            batch[
                "input_ids"
            ]
            .to(device)
        )

        lengths = (
            batch[
                "length"
            ]
        )

        logits = (
            lstm_model(
                input_ids,
                lengths
            )
        )

        predictions = (
            torch.argmax(
                logits,
                dim=1
            )
        )

        lstm_test_predictions.extend(
            predictions
            .cpu()
            .numpy()
        )

lstm_test_predictions = (
    np.array(
        lstm_test_predictions
    )
)

prediction_dictionary[
    "LSTM + Word2Vec"
] = lstm_test_predictions

final_results.append(
    evaluate_model(
        "LSTM + Word2Vec",
        y_test,
        lstm_test_predictions,
        lstm_training_time
    )
)

display(
    pd.DataFrame(
        [
            final_results[-1]
        ]
    )
    .round(4)
)


#Step 43
#Select a DistilBERT sequence length from training-data token lengths.
#The 95th percentile balances information retention and computational efficiency.

sample_for_length = (
    train_df[
        "text"
    ]
    .sample(
        min(
            5000,
            len(
                train_df
            )
        ),
        random_state=SEED
    )
)

bert_token_lengths = []

for text in sample_for_length:

    encoded = (
        bert_tokenizer(
            text,
            add_special_tokens=True,
            truncation=False
        )
    )

    bert_token_lengths.append(
        len(
            encoded[
                "input_ids"
            ]
        )
    )

BERT_MAX_LEN = int(
    np.percentile(
        bert_token_lengths,
        95
    )
)

BERT_MAX_LEN = max(
    32,
    min(
        BERT_MAX_LEN,
        128
    )
)

print(
    "DistilBERT maximum sequence length:",
    BERT_MAX_LEN
)


#Step 44
#Create Transformer datasets from the original uncleaned text.
#DistilBERT receives natural text because its pretrained tokenizer and representations were learned from contextual language rather than TF-IDF-style normalized text.

class TransformerEmotionDataset(
    Dataset
):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length
    ):

        self.texts = list(
            texts
        )

        self.labels = (
            np.array(
                labels
            )
        )

        self.tokenizer = (
            tokenizer
        )

        self.max_length = (
            max_length
        )

    def __len__(
        self
    ):

        return len(
            self.texts
        )

    def __getitem__(
        self,
        index
    ):

        encoding = (
            self.tokenizer(
                self.texts[
                    index
                ],
                truncation=True,
                max_length=self.max_length
            )
        )

        encoding[
            "labels"
        ] = int(
            self.labels[
                index
            ]
        )

        return encoding


bert_train_dataset = (
    TransformerEmotionDataset(
        train_df[
            "text"
        ],
        y_train,
        bert_tokenizer,
        BERT_MAX_LEN
    )
)

bert_val_dataset = (
    TransformerEmotionDataset(
        val_df[
            "text"
        ],
        y_val,
        bert_tokenizer,
        BERT_MAX_LEN
    )
)

bert_test_dataset = (
    TransformerEmotionDataset(
        test_df[
            "text"
        ],
        y_test,
        bert_tokenizer,
        BERT_MAX_LEN
    )
)


#Step 45
#Create dynamically padded DistilBERT batches.
#Dynamic padding avoids padding every observation to the global maximum sequence length.

bert_data_collator = (
    DataCollatorWithPadding(
        tokenizer=bert_tokenizer
    )
)

BERT_TRAIN_BATCH_SIZE = 32

BERT_EVAL_BATCH_SIZE = 64

bert_train_loader = (
    DataLoader(
        bert_train_dataset,
        batch_size=BERT_TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=bert_data_collator
    )
)

bert_val_loader = (
    DataLoader(
        bert_val_dataset,
        batch_size=BERT_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=bert_data_collator
    )
)

bert_test_loader = (
    DataLoader(
        bert_test_dataset,
        batch_size=BERT_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=bert_data_collator
    )
)


#Step 46
#Load pretrained DistilBERT with a new six-class classifier.
#DistilBERT was selected because it satisfies the pretrained-Transformer requirement while being smaller and faster than full BERT.

id2label = {
    i: label
    for i, label
    in enumerate(
        label_names
    )
}

label2id = {
    label: i
    for i, label
    in enumerate(
        label_names
    )
}

bert_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        TRANSFORMER_NAME,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id
    )
    .to(device)
)

BERT_LEARNING_RATE = 2e-5

BERT_WEIGHT_DECAY = 0.01

BERT_EPOCHS = 3

bert_loss_function = (
    nn.CrossEntropyLoss(
        weight=class_weight_tensor
    )
)

bert_optimizer = (
    AdamW(
        bert_model.parameters(),
        lr=BERT_LEARNING_RATE,
        weight_decay=BERT_WEIGHT_DECAY
    )
)


#Step 47
#Fine-tune DistilBERT and choose the best checkpoint using validation Macro-F1.
#The test set remains unused during fine-tuning and checkpoint selection.

best_bert_f1 = -1

best_bert_state = None

best_bert_epoch = None

bert_history = []

start_time = (
    time.perf_counter()
)

for epoch in range(
    BERT_EPOCHS
):

    bert_model.train()

    epoch_train_loss = 0

    for batch in (
        bert_train_loader
    ):

        input_ids = (
            batch[
                "input_ids"
            ]
            .to(device)
        )

        attention_mask = (
            batch[
                "attention_mask"
            ]
            .to(device)
        )

        labels = (
            batch[
                "labels"
            ]
            .to(device)
        )

        bert_optimizer.zero_grad()

        outputs = (
            bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        )

        loss = (
            bert_loss_function(
                outputs.logits,
                labels
            )
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            bert_model.parameters(),
            max_norm=1.0
        )

        bert_optimizer.step()

        epoch_train_loss += (
            loss.item()
        )

    average_train_loss = (
        epoch_train_loss
        / len(
            bert_train_loader
        )
    )

    bert_model.eval()

    validation_predictions = []

    validation_labels = []

    with torch.no_grad():

        for batch in (
            bert_val_loader
        ):

            input_ids = (
                batch[
                    "input_ids"
                ]
                .to(device)
            )

            attention_mask = (
                batch[
                    "attention_mask"
                ]
                .to(device)
            )

            labels = (
                batch[
                    "labels"
                ]
                .to(device)
            )

            outputs = (
                bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            )

            predictions = (
                torch.argmax(
                    outputs.logits,
                    dim=1
                )
            )

            validation_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )

            validation_labels.extend(
                labels
                .cpu()
                .numpy()
            )

    _, _, validation_macro_f1, _ = (
        precision_recall_fscore_support(
            validation_labels,
            validation_predictions,
            average="macro",
            zero_division=0
        )
    )

    bert_history.append({
        "Epoch": epoch + 1,
        "Train Loss": average_train_loss,
        "Validation Macro F1": validation_macro_f1
    })

    print(
        f"Epoch {epoch + 1}/{BERT_EPOCHS} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Validation Macro F1: {validation_macro_f1:.4f}"
    )

    if (
        validation_macro_f1
        >
        best_bert_f1
    ):

        best_bert_f1 = (
            validation_macro_f1
        )

        best_bert_epoch = (
            epoch + 1
        )

        best_bert_state = {
            key: (
                value
                .detach()
                .cpu()
                .clone()
            )
            for key, value
            in (
                bert_model
                .state_dict()
                .items()
            )
        }

bert_training_time = (
    time.perf_counter()
    - start_time
)

bert_history_df = pd.DataFrame(
    bert_history
)

display(
    bert_history_df
    .round(4)
)


#Step 48
#Plot DistilBERT learning curves.
#These plots show how training loss and validation Macro-F1 evolve during fine-tuning.

plt.figure(
    figsize=(7, 4)
)

plt.plot(
    bert_history_df[
        "Epoch"
    ],
    bert_history_df[
        "Train Loss"
    ],
    marker="o"
)

plt.title(
    "DistilBERT Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Training Loss"
)

plt.tight_layout()
plt.show()


plt.figure(
    figsize=(7, 4)
)

plt.plot(
    bert_history_df[
        "Epoch"
    ],
    bert_history_df[
        "Validation Macro F1"
    ],
    marker="o"
)

plt.title(
    "DistilBERT Learning Curve"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Validation Macro F1"
)

plt.ylim(
    0,
    1
)

plt.tight_layout()
plt.show()


#Step 49
#Evaluate the best DistilBERT checkpoint on the test set.
#This produces the final Transformer result under the same test protocol used by every other model.

bert_model.load_state_dict(
    best_bert_state
)

bert_model = (
    bert_model
    .to(device)
)

bert_model.eval()

bert_test_predictions = []

with torch.no_grad():

    for batch in (
        bert_test_loader
    ):

        input_ids = (
            batch[
                "input_ids"
            ]
            .to(device)
        )

        attention_mask = (
            batch[
                "attention_mask"
            ]
            .to(device)
        )

        outputs = (
            bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        )

        predictions = (
            torch.argmax(
                outputs.logits,
                dim=1
            )
        )

        bert_test_predictions.extend(
            predictions
            .cpu()
            .numpy()
        )

bert_test_predictions = (
    np.array(
        bert_test_predictions
    )
)

prediction_dictionary[
    "DistilBERT"
] = bert_test_predictions

final_results.append(
    evaluate_model(
        "DistilBERT",
        y_test,
        bert_test_predictions,
        bert_training_time
    )
)

display(
    pd.DataFrame(
        [
            final_results[-1]
        ]
    )
    .round(4)
)


#Step 50
#Create the main comparison table.
#Macro-F1 is the primary comparison metric because all six classes contribute equally regardless of their frequency.

results_df = (
    pd.DataFrame(
        final_results
    )
    .sort_values(
        "Macro F1",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

display(
    results_df
    .round(4)
)


#Step 51
#Visualize model Macro-F1 scores.
#The chart makes overall differences among traditional, neural, and Transformer approaches easy to communicate.

plot_df = (
    results_df
    .sort_values(
        "Macro F1",
        ascending=True
    )
)

plt.figure(
    figsize=(9, 5)
)

plt.barh(
    plot_df[
        "Model"
    ],
    plot_df[
        "Macro F1"
    ]
)

plt.xlabel(
    "Macro F1"
)

plt.ylabel(
    "Model"
)

plt.title(
    "Final Model Performance Comparison"
)

plt.xlim(
    0,
    1
)

plt.tight_layout()
plt.show()


#Step 52
#Print class-level reports for all five models.
#Class-specific precision, recall, and F1 reveal weaknesses that overall accuracy could hide.

for (
    model_name,
    predictions
) in (
    prediction_dictionary
    .items()
):

    print(
        "\n",
        model_name.upper()
    )

    print(
        classification_report(
            y_test,
            predictions,
            labels=list(
                range(
                    NUM_CLASSES
                )
            ),
            target_names=label_names,
            zero_division=0
        )
    )


#Step 53
#Display one normalized confusion matrix for each model.
#Confusion matrices reveal which emotions are repeatedly confused rather than only summarizing total performance.

for (
    model_name,
    predictions
) in (
    prediction_dictionary
    .items()
):

    matrix = (
        confusion_matrix(
            y_test,
            predictions,
            labels=list(
                range(
                    NUM_CLASSES
                )
            ),
            normalize="true"
        )
    )

    figure, axis = (
        plt.subplots(
            figsize=(7, 6)
        )
    )

    display_matrix = (
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=label_names
        )
    )

    display_matrix.plot(
        ax=axis,
        xticks_rotation=45,
        values_format=".2f",
        cmap=None
    )

    plt.title(
        f"Normalized Confusion Matrix - {model_name}"
    )

    plt.tight_layout()
    plt.show()


#Step 54
#Analyze class-level performance for the best model.
#This directly answers which emotion categories are easiest and hardest to classify.

best_model_name = (
    results_df
    .iloc[0][
        "Model"
    ]
)

best_model_predictions = (
    prediction_dictionary[
        best_model_name
    ]
)

class_precision, class_recall, class_f1, class_support = (
    precision_recall_fscore_support(
        y_test,
        best_model_predictions,
        labels=list(
            range(
                NUM_CLASSES
            )
        ),
        zero_division=0
    )
)

class_results_df = pd.DataFrame({
    "Emotion": label_names,
    "Precision": class_precision,
    "Recall": class_recall,
    "F1": class_f1,
    "Support": class_support
})

display(
    class_results_df
    .sort_values(
        "F1",
        ascending=False
    )
    .round(4)
)


#Step 55
#Perform quantitative error analysis.
#Recurring true-versus-predicted emotion pairs provide objective evidence of systematic failure patterns.

error_df = (
    test_df[
        [
            "text",
            "label",
            "emotion",
            "word_count"
        ]
    ]
    .copy()
)

error_df[
    "predicted_label"
] = (
    best_model_predictions
)

error_df[
    "predicted_emotion"
] = (
    error_df[
        "predicted_label"
    ]
    .map(
        lambda x: label_names[x]
    )
)

error_df[
    "correct"
] = (
    error_df[
        "label"
    ]
    ==
    error_df[
        "predicted_label"
    ]
)

errors_only = (
    error_df[
        error_df[
            "correct"
        ] == False
    ]
    .copy()
)

print(
    "Best model:",
    best_model_name
)

print(
    "Total test errors:",
    len(
        errors_only
    )
)

common_confusions = (
    errors_only
    .groupby(
        [
            "emotion",
            "predicted_emotion"
        ]
    )
    .size()
    .reset_index(
        name="Count"
    )
    .sort_values(
        "Count",
        ascending=False
    )
)

display(
    common_confusions
    .head(15)
)


#Step 56
#Create a manual qualitative error-analysis sample.
#Human inspection is necessary because ambiguity, negation, multiple emotions, and annotation uncertainty cannot be reliably inferred from labels alone.

manual_error_sample = (
    errors_only[
        [
            "text",
            "emotion",
            "predicted_emotion",
            "word_count"
        ]
    ]
    .sample(
        min(
            30,
            len(
                errors_only
            )
        ),
        random_state=SEED
    )
    .copy()
)

manual_error_sample[
    "Manual Failure Pattern"
] = ""

manual_error_sample[
    "Research Notes"
] = ""

display(
    manual_error_sample
)

manual_error_sample.to_csv(
    "manual_error_analysis.csv",
    index=False
)

print("""
Suggested qualitative categories:
- Similar emotions
- Emotional ambiguity
- Limited context
- Negation
- Implicit emotion
- Multiple emotional cues
- Possible annotation ambiguity
""")


#Step 57
#Analyze whether text length is associated with model errors.
#This checks whether very short or longer observations appear disproportionately difficult.

error_df[
    "Length Group"
] = (
    pd.cut(
        error_df[
            "word_count"
        ],
        bins=[
            0,
            10,
            20,
            30,
            1000
        ],
        labels=[
            "1-10",
            "11-20",
            "21-30",
            "31+"
        ]
    )
)

length_error_analysis = (
    error_df
    .groupby(
        "Length Group",
        observed=False
    )[
        "correct"
    ]
    .agg(
        [
            "count",
            "mean"
        ]
    )
    .reset_index()
)

length_error_analysis[
    "Error Rate"
] = (
    1
    -
    length_error_analysis[
        "mean"
    ]
)

display(
    length_error_analysis[
        [
            "Length Group",
            "count",
            "Error Rate"
        ]
    ]
    .round(4)
)


#Step 58
#Evaluate practical rather than purely numerical improvement.
#The comparison considers both predictive gain and additional training cost.

traditional_names = [
    "Logistic Regression",
    "Linear SVM",
    "Naive Bayes"
]

best_traditional_row = (
    results_df[
        results_df[
            "Model"
        ]
        .isin(
            traditional_names
        )
    ]
    .sort_values(
        "Macro F1",
        ascending=False
    )
    .iloc[0]
)

best_overall_row = (
    results_df
    .iloc[0]
)

macro_f1_gain = (
    best_overall_row[
        "Macro F1"
    ]
    -
    best_traditional_row[
        "Macro F1"
    ]
)

accuracy_gain = (
    best_overall_row[
        "Accuracy"
    ]
    -
    best_traditional_row[
        "Accuracy"
    ]
)

traditional_error = (
    1
    -
    best_traditional_row[
        "Accuracy"
    ]
)

overall_error = (
    1
    -
    best_overall_row[
        "Accuracy"
    ]
)

relative_error_reduction = (
    (
        traditional_error
        -
        overall_error
    )
    /
    traditional_error
    *
    100
)

training_time_ratio = (
    best_overall_row[
        "Training Time (s)"
    ]
    /
    max(
        best_traditional_row[
            "Training Time (s)"
        ],
        0.0001
    )
)

practical_significance_df = pd.DataFrame({
    "Comparison": [
        "Best overall model",
        "Best traditional model",
        "Macro-F1 gain",
        "Accuracy gain",
        "Relative accuracy-error reduction",
        "Training-time multiplier"
    ],

    "Value": [
        best_overall_row[
            "Model"
        ],

        best_traditional_row[
            "Model"
        ],

        f"{macro_f1_gain * 100:.2f} percentage points",

        f"{accuracy_gain * 100:.2f} percentage points",

        f"{relative_error_reduction:.2f}%",

        f"{training_time_ratio:.1f}x"
    ]
})

display(
    practical_significance_df
)


#Step 59
#Compare unigram TF-IDF with unigram-plus-bigram TF-IDF.
#This focused feature experiment tests whether adding short word-order information improves emotion classification.

tfidf_unigram = (
    TfidfVectorizer(
        lowercase=False,
        max_features=20000,
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
)

tfidf_unigram_bigram = (
    TfidfVectorizer(
        lowercase=False,
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
)

X_train_uni = (
    tfidf_unigram
    .fit_transform(
        train_df[
            "classic_text"
        ]
    )
)

X_val_uni = (
    tfidf_unigram
    .transform(
        val_df[
            "classic_text"
        ]
    )
)

X_train_uni_bi = (
    tfidf_unigram_bigram
    .fit_transform(
        train_df[
            "classic_text"
        ]
    )
)

X_val_uni_bi = (
    tfidf_unigram_bigram
    .transform(
        val_df[
            "classic_text"
        ]
    )
)

uni_model = (
    LogisticRegression(
        C=best_lr_c,
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED
    )
)

uni_bi_model = (
    LogisticRegression(
        C=best_lr_c,
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED
    )
)

uni_model.fit(
    X_train_uni,
    y_train
)

uni_bi_model.fit(
    X_train_uni_bi,
    y_train
)

uni_pred = (
    uni_model
    .predict(
        X_val_uni
    )
)

uni_bi_pred = (
    uni_bi_model
    .predict(
        X_val_uni_bi
    )
)

_, _, uni_f1, _ = (
    precision_recall_fscore_support(
        y_val,
        uni_pred,
        average="macro",
        zero_division=0
    )
)

_, _, uni_bi_f1, _ = (
    precision_recall_fscore_support(
        y_val,
        uni_bi_pred,
        average="macro",
        zero_division=0
    )
)

ngram_experiment_df = pd.DataFrame({
    "Representation": [
        "Unigrams",
        "Unigrams + Bigrams"
    ],

    "Validation Macro F1": [
        uni_f1,
        uni_bi_f1
    ],

    "Vocabulary Size": [
        len(
            tfidf_unigram
            .vocabulary_
        ),

        len(
            tfidf_unigram_bigram
            .vocabulary_
        )
    ]
})

display(
    ngram_experiment_df
    .round(4)
)

plt.figure(
    figsize=(7, 4)
)

plt.bar(
    ngram_experiment_df[
        "Representation"
    ],
    ngram_experiment_df[
        "Validation Macro F1"
    ]
)

plt.ylabel(
    "Validation Macro F1"
)

plt.title(
    "TF-IDF Feature Experiment"
)

plt.ylim(
    0,
    1
)

plt.tight_layout()
plt.show()


#Step 60
#Document the selected hyperparameters.
#This table makes the experiment reproducible and clearly states which settings were chosen using validation data.

hyperparameter_summary_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Representation": "TF-IDF unigram",
        "Main Hyperparameters": f"C={best_lr_c}",
        "Selection Procedure": "Validation Macro F1"
    },

    {
        "Model": "Linear SVM",
        "Representation": "TF-IDF unigram",
        "Main Hyperparameters": f"C={best_svm_c}",
        "Selection Procedure": "Validation Macro F1"
    },

    {
        "Model": "Naive Bayes",
        "Representation": "TF-IDF unigram",
        "Main Hyperparameters": (
            f"alpha={best_nb_alpha}, "
            f"fit_prior={best_nb_prior}"
        ),
        "Selection Procedure": "Validation Macro F1"
    },

    {
        "Model": "LSTM + Word2Vec",
        "Representation": "Word2Vec learned from training corpus",
        "Main Hyperparameters": (
            f"embedding={WORD2VEC_DIM}, "
            f"hidden={LSTM_HIDDEN_DIM}, "
            f"lr={LSTM_LEARNING_RATE}, "
            f"batch={LSTM_BATCH_SIZE}, "
            f"best_epoch={best_lstm_epoch}"
        ),
        "Selection Procedure": (
            "Validation Macro F1 + early stopping"
        )
    },

    {
        "Model": "DistilBERT",
        "Representation": "Pretrained contextual subwords",
        "Main Hyperparameters": (
            f"lr={BERT_LEARNING_RATE}, "
            f"batch={BERT_TRAIN_BATCH_SIZE}, "
            f"max_len={BERT_MAX_LEN}, "
            f"best_epoch={best_bert_epoch}"
        ),
        "Selection Procedure": "Validation Macro F1"
    }
])

display(
    hyperparameter_summary_df
)


#Step 61
#Save the principal experimental outputs.
#Saving the result tables supports reproducibility and consistent reporting.

results_df.to_csv(
    "final_model_comparison.csv",
    index=False
)

class_results_df.to_csv(
    "best_model_class_results.csv",
    index=False
)

common_confusions.to_csv(
    "common_confusions.csv",
    index=False
)

ngram_experiment_df.to_csv(
    "ngram_experiment.csv",
    index=False
)

hyperparameter_summary_df.to_csv(
    "hyperparameter_summary.csv",
    index=False
)

final_split_table.to_csv(
    "dataset_split_summary.csv",
    index=False
)

print(
    "Main result files saved."
)

# =========================================================
# SAVE TRAINED MODEL ARTIFACTS
# =========================================================
#The notebook originally kept trained models in memory.
#For GitHub + Streamlit, the fitted artifacts are saved so app.py can load them without retraining.

print("\nSaving trained model artifacts...")

#Traditional TF-IDF pipeline artifacts
joblib.dump(
    tfidf_vectorizer,
    "models/tfidf_vectorizer.joblib"
)

joblib.dump(
    logistic_model,
    "models/logistic_regression.joblib"
)

joblib.dump(
    svm_model,
    "models/linear_svm.joblib"
)

joblib.dump(
    nb_model,
    "models/naive_bayes.joblib"
)

#Word2Vec and LSTM artifacts
word2vec_model.save(
    "models/word2vec.model"
)

with open(
    "models/word2idx.pkl",
    "wb"
) as file:
    pickle.dump(
        word2idx,
        file
    )

torch.save(
    lstm_model.state_dict(),
    "models/lstm_model.pt"
)

lstm_metadata = {
    "embedding_dim": int(WORD2VEC_DIM),
    "hidden_dim": int(LSTM_HIDDEN_DIM),
    "dropout": float(LSTM_DROPOUT),
    "max_length": int(LSTM_MAX_LEN),
    "num_classes": int(NUM_CLASSES),
    "best_epoch": int(best_lstm_epoch),
    "labels": list(label_names)
}

with open(
    "models/lstm_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        lstm_metadata,
        file,
        indent=2
    )

#DistilBERT model and tokenizer
bert_output_dir = "models/distilbert"

bert_model.save_pretrained(
    bert_output_dir
)

bert_tokenizer.save_pretrained(
    bert_output_dir
)

bert_metadata = {
    "transformer_name": TRANSFORMER_NAME,
    "max_length": int(BERT_MAX_LEN),
    "num_classes": int(NUM_CLASSES),
    "best_epoch": int(best_bert_epoch),
    "labels": list(label_names)
}

with open(
    "models/distilbert_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        bert_metadata,
        file,
        indent=2
    )

#Copy the principal experimental tables into the results folder.
results_df.to_csv(
    "results/final_model_comparison.csv",
    index=False
)

class_results_df.to_csv(
    "results/best_model_class_results.csv",
    index=False
)

common_confusions.to_csv(
    "results/common_confusions.csv",
    index=False
)

ngram_experiment_df.to_csv(
    "results/ngram_experiment.csv",
    index=False
)

hyperparameter_summary_df.to_csv(
    "results/hyperparameter_summary.csv",
    index=False
)

final_split_table.to_csv(
    "results/dataset_split_summary.csv",
    index=False
)

if "manual_error_sample" in globals():
    manual_error_sample.to_csv(
        "results/manual_error_analysis.csv",
        index=False
    )

#Save label information for inference.
label_metadata = {
    "labels": list(label_names),
    "label2id": {
        str(label): int(index)
        for index, label
        in enumerate(label_names)
    }
}

with open(
    "models/labels.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        label_metadata,
        file,
        indent=2
    )

print("\nTraining pipeline completed.")
print("Saved model artifacts in: models/")
print("Saved result tables in: results/")
print("\nNext command:")
print("streamlit run app.py")
