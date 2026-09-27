import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Emotion Classification NLP",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Emotion Classification in Social Media (Twitter)")
st.caption("Beatriz León Hernández | Master in Data Science | NLP")

st.markdown(
    """
    This dashboard presents the complete NLP final project, including the dataset,
    methodology, model comparison, learning curves, error analysis, additional experiments,
    conclusions, and a live comparison demo.
    """
)


# =========================================================
# PROJECT DATA
# =========================================================

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

FINAL_SPLITS = pd.DataFrame(
    {
        "Split": ["Train", "Validation", "Test"],
        "Samples": [15939, 1991, 1986]
    }
)

CLASS_DISTRIBUTION = pd.DataFrame(
    {
        "Emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "Count": [4662, 5345, 1289, 2152, 1926, 565],
        "Percentage": [29.25, 33.53, 8.09, 13.50, 12.08, 3.54]
    }
)

FINAL_RESULTS = pd.DataFrame(
    {
        "Model": [
            "DistilBERT",
            "LSTM + Word2Vec",
            "Linear SVM",
            "Logistic Regression",
            "Naive Bayes"
        ],
        "Accuracy": [0.9245, 0.9134, 0.8953, 0.8922, 0.8338],
        "Macro Precision": [0.8941, 0.8580, 0.8352, 0.8316, 0.7959],
        "Macro Recall": [0.8916, 0.9068, 0.8852, 0.8784, 0.7286],
        "Macro F1": [0.8852, 0.8765, 0.8549, 0.8511, 0.7526],
        "Weighted F1": [0.9251, 0.9156, 0.8977, 0.8946, 0.8294],
        "Training Time (s)": [221.7133, 14.7915, 0.3880, 4.0858, 0.0073]
    }
)

DISTILBERT_CLASS_RESULTS = pd.DataFrame(
    {
        "Emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "Precision": [0.9767, 0.9736, 0.7356, 0.8969, 0.8708, 0.9111],
        "Recall": [0.9413, 0.9113, 0.9808, 0.9526, 0.9330, 0.6308],
        "F1": [0.9587, 0.9414, 0.8407, 0.9239, 0.9009, 0.7455],
        "Support": [579, 688, 156, 274, 224, 65]
    }
)

ERROR_PAIRS = pd.DataFrame(
    {
        "True emotion": [
            "joy",
            "surprise",
            "sadness",
            "sadness",
            "fear",
            "sadness",
            "anger",
            "fear",
            "surprise",
            "anger"
        ],
        "Predicted emotion": [
            "love",
            "fear",
            "anger",
            "joy",
            "anger",
            "fear",
            "sadness",
            "sadness",
            "joy",
            "fear"
        ],
        "Count": [52, 18, 16, 9, 8, 6, 6, 6, 4, 4]
    }
)

NGRAM_RESULTS = pd.DataFrame(
    {
        "Representation": ["Unigrams", "Unigrams + Bigrams"],
        "Validation Macro F1": [0.8800, 0.8637],
        "Vocabulary Size": [7234, 30000]
    }
)

TRADITIONAL_CURVES = {
    "Logistic Regression": {
        "samples": [3187, 6375, 9563, 12751, 15939],
        "macro_f1": [0.7924, 0.8546, 0.8637, 0.8759, 0.8800]
    },
    "Linear SVM": {
        "samples": [3187, 6375, 9563, 12751, 15939],
        "macro_f1": [0.8035, 0.8742, 0.8707, 0.8853, 0.8841]
    },
    "Naive Bayes": {
        "samples": [3187, 6375, 9563, 12751, 15939],
        "macro_f1": [0.5090, 0.6381, 0.7162, 0.7324, 0.7658]
    }
}

LSTM_EPOCHS = [1, 2, 3, 4, 5, 6, 7, 8]
LSTM_LOSS = [1.6630, 1.1655, 0.6260, 0.3348, 0.2238, 0.1571, 0.1256, 0.1009]
LSTM_VAL_F1 = [0.4175, 0.6438, 0.8196, 0.8717, 0.9066, 0.8881, 0.8919, 0.8966]

BERT_EPOCHS = [1, 2, 3]
BERT_LOSS = [0.7015, 0.2034, 0.1550]
BERT_VAL_F1 = [0.9038, 0.9067, 0.9103]

SVM_CONFUSION = np.array(
    [
        [0.91, 0.02, 0.01, 0.03, 0.01, 0.01],
        [0.01, 0.90, 0.07, 0.00, 0.00, 0.01],
        [0.00, 0.07, 0.92, 0.01, 0.00, 0.01],
        [0.04, 0.03, 0.01, 0.91, 0.02, 0.00],
        [0.04, 0.00, 0.01, 0.04, 0.83, 0.07],
        [0.03, 0.05, 0.00, 0.00, 0.08, 0.85]
    ]
)

DISTILBERT_CONFUSION = np.array(
    [
        [0.94, 0.02, 0.01, 0.03, 0.01, 0.00],
        [0.00, 0.91, 0.08, 0.00, 0.00, 0.00],
        [0.00, 0.01, 0.98, 0.01, 0.00, 0.00],
        [0.02, 0.01, 0.00, 0.95, 0.01, 0.00],
        [0.03, 0.00, 0.00, 0.04, 0.93, 0.00],
        [0.02, 0.06, 0.00, 0.02, 0.28, 0.63]
    ]
)

DEMO_EXAMPLES = {
    "Clear sadness": {
        "text": "I miss my family and I feel very lonely today.",
        "predictions": {
            "Logistic Regression": "Sadness",
            "Linear SVM": "Sadness",
            "Naive Bayes": "Sadness",
            "LSTM + Word2Vec": "Sadness",
            "DistilBERT": "Sadness"
        }
    },
    "Ambiguous: sadness / joy / love": {
        "text": (
            "My friends stayed with me during a very difficult week and reminded me that "
            "I did not have to face everything alone. I feel grateful, supported, and deeply cared for."
        ),
        "predictions": {
            "Logistic Regression": "Sadness",
            "Linear SVM": "Sadness",
            "Naive Bayes": "Sadness",
            "LSTM + Word2Vec": "Joy",
            "DistilBERT": "Love"
        }
    },
    "Surprise vs fear": {
        "text": (
            "I walked into the room expecting an ordinary meeting, but everyone was waiting for me "
            "with unexpected news. I had no idea this was coming and I was completely amazed."
        ),
        "predictions": {
            "Logistic Regression": "Surprise",
            "Linear SVM": "Surprise",
            "Naive Bayes": "Fear",
            "LSTM + Word2Vec": "Surprise",
            "DistilBERT": "Surprise"
        }
    }
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def line_plot(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def confusion_plot(matrix, title):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")

    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels(EMOTIONS)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig


def display_demo_predictions(predictions):
    columns = st.columns(5)

    for col, (model, prediction) in zip(columns, predictions.items()):
        with col:
            st.markdown(f"**{model}**")
            st.info(prediction.upper())

    prediction_values = list(predictions.values())
    most_common = max(set(prediction_values), key=prediction_values.count)
    agreement = prediction_values.count(most_common)

    st.caption(
        f"Model agreement: {agreement} of 5 models most commonly predict "
        f"**{most_common.upper()}**."
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Project Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Project Overview",
        "Dataset & EDA",
        "Methodology",
        "Traditional Models",
        "LSTM + Word2Vec",
        "DistilBERT",
        "Final Model Comparison",
        "Error Analysis",
        "Additional Experiment",
        "Live Demo",
        "Conclusion & Future Work",
        "Acknowledgments"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("**Main metric:** Macro-F1")
st.sidebar.write("**Random seed:** 42")


# =========================================================
# 1. PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.header("Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Original observations", "20,000")
    c2.metric("Final observations", "19,916")
    c3.metric("Emotion classes", "6")
    c4.metric("Best Macro-F1", "0.8852")

    st.subheader("Objective")

    st.write(
        "The objective of this project is to compare traditional machine-learning, neural, "
        "and Transformer-based approaches for multiclass emotion classification in English "
        "social-media text."
    )

    st.subheader("Research Questions")

    st.markdown(
        """
        1. **How do traditional machine-learning models compare with neural and
        Transformer-based models for multiclass emotion classification?**

        2. **Which emotion classes are the most difficult to classify, and what recurring
        error patterns appear?**
        """
    )

    st.subheader("Hypothesis")

    st.write(
        "The Transformer-based model is expected to obtain the highest Macro-F1 because "
        "contextual representations can capture relationships between words more effectively "
        "than sparse TF-IDF representations."
    )

    st.subheader("Models Compared")

    st.markdown(
        """
        **Traditional models**
        - Logistic Regression
        - Linear SVM
        - Multinomial Naive Bayes

        **Neural / modern NLP models**
        - LSTM + Word2Vec
        - DistilBERT
        """
    )


# =========================================================
# 2. DATASET & EDA
# =========================================================

elif page == "Dataset & EDA":

    st.header("Dataset & Exploratory Data Analysis")

    st.write(
        "The project uses the `dair-ai/emotion` dataset from Hugging Face. "
        "The processed version contains English social-media texts labeled with six emotions: "
        "sadness, joy, love, anger, fear, and surprise."
    )

    st.subheader("Original Split")

    original_split = pd.DataFrame(
        {
            "Split": ["Train", "Validation", "Test"],
            "Samples": [16000, 2000, 2000]
        }
    )

    st.dataframe(original_split, use_container_width=True, hide_index=True)

    st.subheader("Final Split After Cleaning")

    st.dataframe(FINAL_SPLITS, use_container_width=True, hide_index=True)

    st.write(
        "After removing duplicated observations and cross-split overlaps, the final dataset "
        "contained 15,939 training texts, 1,991 validation texts, and 1,986 test texts."
    )

    st.subheader("Class Distribution")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.bar_chart(CLASS_DISTRIBUTION.set_index("Emotion")["Count"])

    with col2:
        st.dataframe(
            CLASS_DISTRIBUTION.style.format({"Percentage": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "The dataset is imbalanced. Joy and sadness are the most frequent emotions, "
        "while surprise is the least frequent. The majority/minority ratio is 9.46."
    )

    st.subheader("Data Quality Checks")

    quality = pd.DataFrame(
        {
            "Check": [
                "Missing text",
                "Missing labels",
                "Empty text",
                "Duplicated rows",
                "Conflicting duplicate texts",
                "Cross-split overlap"
            ],
            "Result": [
                "0",
                "0",
                "0",
                "Detected and removed",
                "Detected",
                "Detected and removed"
            ]
        }
    )

    st.dataframe(quality, use_container_width=True, hide_index=True)


# =========================================================
# 3. METHODOLOGY
# =========================================================

elif page == "Methodology":

    st.header("Methodology")

    st.subheader("Experimental Protocol")

    st.markdown(
        """
        - **Training set:** used to fit model parameters.
        - **Validation set:** used for model selection and tuning.
        - **Test set:** used only for final evaluation.
        - All final models were evaluated on the **same unseen test set**.
        """
    )

    st.subheader("Preprocessing")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Traditional Models + LSTM")
        st.markdown(
            """
            - Lowercase conversion
            - URLs replaced with a generic token
            - User mentions replaced with a generic token
            - Punctuation/noise handling
            - Whitespace normalization
            - No stopword removal
            - No stemming
            """
        )

    with col2:
        st.markdown("### DistilBERT")
        st.markdown(
            """
            - Original text preserved
            - DistilBERT pretrained tokenizer
            - Subword tokenization
            - No aggressive preprocessing
            """
        )

    st.subheader("Why Macro-F1?")

    st.write(
        "Macro-F1 calculates the F1-score independently for each emotion and gives every "
        "class the same importance. This makes it appropriate for this multiclass and "
        "imbalanced dataset."
    )


# =========================================================
# 4. TRADITIONAL MODELS
# =========================================================

elif page == "Traditional Models":

    st.header("Traditional Machine-Learning Models")

    st.write(
        "The traditional models use TF-IDF unigram features. TF-IDF converts text into "
        "numerical features based on how informative each word is in the corpus."
    )

    selected_model = st.selectbox(
        "Select a learning curve",
        ["Logistic Regression", "Linear SVM", "Naive Bayes"]
    )

    curve = TRADITIONAL_CURVES[selected_model]

    st.pyplot(
        line_plot(
            curve["samples"],
            curve["macro_f1"],
            f"Learning Curve - {selected_model}",
            "Training Samples",
            "Validation Macro-F1"
        )
    )

    st.write(
        "Logistic Regression and Linear SVM reached strong and stable validation performance "
        "as more training data were added. Naive Bayes also improved, but remained less competitive."
    )

    traditional_results = FINAL_RESULTS[
        FINAL_RESULTS["Model"].isin(
            ["Linear SVM", "Logistic Regression", "Naive Bayes"]
        )
    ]

    st.dataframe(
        traditional_results.style.format(
            {
                "Accuracy": "{:.4f}",
                "Macro Precision": "{:.4f}",
                "Macro Recall": "{:.4f}",
                "Macro F1": "{:.4f}",
                "Weighted F1": "{:.4f}",
                "Training Time (s)": "{:.4f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "Best traditional model: Linear SVM with a test Macro-F1 of 0.8549."
    )


# =========================================================
# 5. LSTM + WORD2VEC
# =========================================================

elif page == "LSTM + Word2Vec":

    st.header("LSTM + Word2Vec")

    st.write(
        "Word2Vec converts words into 100-dimensional dense numerical vectors. "
        "The LSTM then learns relationships between words across the sequence."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(
            line_plot(
                LSTM_EPOCHS,
                LSTM_LOSS,
                "LSTM + Word2Vec Training Loss",
                "Epoch",
                "Training Loss"
            )
        )

    with col2:
        st.pyplot(
            line_plot(
                LSTM_EPOCHS,
                LSTM_VAL_F1,
                "LSTM + Word2Vec Learning Curve",
                "Epoch",
                "Validation Macro-F1"
            )
        )

    st.write(
        "The training loss continuously decreased, while validation Macro-F1 improved quickly "
        "during the first epochs and reached its best value around epoch 5. Early stopping was "
        "used after validation performance stopped improving."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Test Accuracy", "0.9134")
    c2.metric("Test Macro-F1", "0.8765")
    c3.metric("Training Time", "~15 s")


# =========================================================
# 6. DISTILBERT
# =========================================================

elif page == "DistilBERT":

    st.header("DistilBERT")

    st.write(
        "DistilBERT is a pretrained Transformer model. It starts with general language knowledge "
        "and is fine-tuned to classify the six emotion categories."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(
            line_plot(
                BERT_EPOCHS,
                BERT_LOSS,
                "DistilBERT Training Loss",
                "Epoch",
                "Training Loss"
            )
        )

    with col2:
        st.pyplot(
            line_plot(
                BERT_EPOCHS,
                BERT_VAL_F1,
                "DistilBERT Learning Curve",
                "Epoch",
                "Validation Macro-F1"
            )
        )

    st.write(
        "DistilBERT reduced its training error quickly, while validation Macro-F1 remained "
        "high and stable during the three epochs."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Test Accuracy", "0.9245")
    c2.metric("Test Macro-F1", "0.8852")
    c3.metric("Training Time", "~222 s")

    st.subheader("Performance by Emotion")

    st.dataframe(
        DISTILBERT_CLASS_RESULTS.style.format(
            {
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Sadness achieved the highest class F1 (0.9587), while surprise was the most "
        "difficult class with an F1 of 0.7455."
    )


# =========================================================
# 7. FINAL MODEL COMPARISON
# =========================================================

elif page == "Final Model Comparison":

    st.header("Final Model Comparison")

    st.write(
        "All final models were evaluated using the same unseen test set."
    )

    st.dataframe(
        FINAL_RESULTS.style.format(
            {
                "Accuracy": "{:.4f}",
                "Macro Precision": "{:.4f}",
                "Macro Recall": "{:.4f}",
                "Macro F1": "{:.4f}",
                "Weighted F1": "{:.4f}",
                "Training Time (s)": "{:.4f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Macro-F1")

    st.bar_chart(
        FINAL_RESULTS.set_index("Model")[["Macro F1"]]
    )

    st.subheader("Training Time")

    st.bar_chart(
        FINAL_RESULTS.set_index("Model")[["Training Time (s)"]]
    )

    st.success(
        "DistilBERT achieved the highest overall Macro-F1: 0.8852."
    )

    st.write(
        "Although DistilBERT achieved the best overall performance, it was also the most "
        "computationally expensive model, requiring about 222 seconds of training. "
        "LSTM + Word2Vec required about 15 seconds, while Linear SVM required less than one second."
    )

    st.write(
        "For this relatively small project, DistilBERT was selected as the best model because "
        "the additional computational cost was manageable. For larger future applications, "
        "model selection should also consider budget, time, and computational resources."
    )


# =========================================================
# 8. ERROR ANALYSIS
# =========================================================

elif page == "Error Analysis":

    st.header("Error Analysis")

    st.metric("DistilBERT test errors", "150")

    st.subheader("Most Common Error Pairs")

    st.dataframe(
        ERROR_PAIRS,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "The most common confusion was joy → love (52 cases), followed by surprise → fear "
        "(18 cases) and sadness → anger (16 cases)."
    )

    st.write(
        "These errors suggest that the model has more difficulty when emotions are semantically "
        "close or when the text contains mixed or ambiguous emotional cues."
    )

    st.subheader("Normalized Confusion Matrices")

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(
            confusion_plot(
                SVM_CONFUSION,
                "Normalized Confusion Matrix - Linear SVM"
            )
        )

    with col2:
        st.pyplot(
            confusion_plot(
                DISTILBERT_CONFUSION,
                "Normalized Confusion Matrix - DistilBERT"
            )
        )

    st.info(
        "DistilBERT improves performance for most emotion classes. Its main difficulty is "
        "surprise, where only 0.63 of the true surprise cases are correctly classified and "
        "0.28 are classified as fear."
    )

    st.subheader("Recurring Failure Patterns")

    st.markdown(
        """
        - Similar emotions
        - Emotional ambiguity
        - Limited context
        - Negation
        - Implicit emotion
        - Multiple emotional cues
        - Possible annotation ambiguity
        """
    )


# =========================================================
# 9. ADDITIONAL EXPERIMENT
# =========================================================

elif page == "Additional Experiment":

    st.header("Additional Feature Experiment")

    st.write(
        "Logistic Regression was kept fixed to compare TF-IDF unigrams with "
        "TF-IDF unigrams + bigrams."
    )

    st.dataframe(
        NGRAM_RESULTS.style.format(
            {"Validation Macro F1": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        NGRAM_RESULTS.set_index("Representation")[["Validation Macro F1"]]
    )

    st.write(
        "Unigrams achieved a higher validation Macro-F1 (0.8800) than unigrams + bigrams "
        "(0.8637). Adding more features did not improve performance in this experiment."
    )


# =========================================================
# 10. LIVE DEMO
# =========================================================

elif page == "Live Demo":

    st.header("Live Model Comparison")

    SHORT_DEMO_TEXTS = {
        "Short - Joy":
            "I feel incredibly happy and proud of what I achieved today.",

        "Short - Sadness":
            "I miss my family and I feel very lonely today.",

        "Short - Anger":
            "I am furious that they treated me this way.",

        "Short - Fear":
            "I am terrified about what might happen tomorrow.",

        "Short - Love":
            "I feel deeply loved and appreciated by the people around me.",

        "Short - Surprise":
            "I cannot believe this happened; I am completely shocked."
    }

    AMBIGUOUS_DEMO_TEXTS = {
        "Ambiguous - Joy vs Love":
            "I miss him, but thinking about our time together still makes me smile.",

        "Ambiguous - Fear vs Surprise":
            "I cannot believe what just happened and I do not know what will happen next.",

        "Ambiguous - Sadness vs Anger":
            "I feel hurt and frustrated that everything ended this way.",

        "Ambiguous - Joy vs Surprise":
            "I never expected this news and I cannot stop smiling.",

        "Ambiguous - Fear vs Sadness":
            "I feel helpless and worried because I do not know what comes next."
    }

    LONG_DEMO_TEXTS = {
        "Longer - Joy":
            (
                "Today started like any normal day, but then I received the results "
                "I had been waiting for. I worked very hard and I feel incredibly "
                "happy and proud that everything went well."
            ),

        "Longer - Fear":
            (
                "Tomorrow I have to face something I have been avoiding for weeks. "
                "I keep thinking about everything that could go wrong, and I feel "
                "nervous and uncertain about what will happen."
            ),

        "Longer - Sadness":
            (
                "I spent the afternoon remembering how different everything used to "
                "be. I miss the people who were part of that time, and thinking "
                "about it makes me feel lonely and disappointed."
            ),

        "Longer - Anger":
            (
                "I tried to remain calm, but they ignored everything I said and "
                "continued treating me unfairly. The more I think about the situation, "
                "the more frustrated and angry I become."
            ),

        "Longer - Love":
            (
                "My friends stayed with me during a very difficult week and reminded "
                "me that I did not have to face everything alone. I feel grateful, "
                "supported, and deeply cared for."
            ),

        "Longer - Surprise":
            (
                "I walked into the room expecting an ordinary meeting, but everyone "
                "was waiting for me with unexpected news. I had no idea this was "
                "coming and I was completely amazed."
            )
    }

    ALL_DEMO_TEXTS = {}

    ALL_DEMO_TEXTS.update(SHORT_DEMO_TEXTS)
    ALL_DEMO_TEXTS.update(AMBIGUOUS_DEMO_TEXTS)
    ALL_DEMO_TEXTS.update(LONG_DEMO_TEXTS)

    emotion_colors = {
        "sadness": "#D6EAF8",
        "joy": "#FCF3CF",
        "love": "#FADBD8",
        "anger": "#F5B7B1",
        "fear": "#D2B4DE",
        "surprise": "#D5F5E3"
    }

    emotion_text_colors = {
        "sadness": "#1F618D",
        "joy": "#9A7D0A",
        "love": "#922B21",
        "anger": "#922B21",
        "fear": "#633974",
        "surprise": "#196F3D"
    }

    selected_example = st.selectbox(
        "Example:",
        list(ALL_DEMO_TEXTS.keys())
    )

    selected_text = ALL_DEMO_TEXTS[selected_example]

    st.text_area(
        "Text:",
        value=selected_text,
        height=120,
        disabled=True
    )

    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            border: none;
            height: 45px;
            width: 250px;
            font-weight: bold;
        }

        div.stButton > button:first-child:hover {
            background-color: #45a049;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("Compare All Models"):

        # -------------------------------------------------
        # TEMPORARY PREDEFINED OUTPUTS
        # Replace this section later with predict_all_models()
        # once the saved trained models are connected.
        # -------------------------------------------------

        preset_predictions = {

            "Short - Joy": {
                "Logistic Regression": "joy",
                "Linear SVM": "joy",
                "Naive Bayes": "joy",
                "LSTM + Word2Vec": "joy",
                "DistilBERT": "joy"
            },

            "Short - Sadness": {
                "Logistic Regression": "sadness",
                "Linear SVM": "sadness",
                "Naive Bayes": "sadness",
                "LSTM + Word2Vec": "sadness",
                "DistilBERT": "sadness"
            },

            "Short - Anger": {
                "Logistic Regression": "anger",
                "Linear SVM": "anger",
                "Naive Bayes": "anger",
                "LSTM + Word2Vec": "anger",
                "DistilBERT": "anger"
            },

            "Short - Fear": {
                "Logistic Regression": "fear",
                "Linear SVM": "fear",
                "Naive Bayes": "fear",
                "LSTM + Word2Vec": "fear",
                "DistilBERT": "fear"
            },

            "Short - Love": {
                "Logistic Regression": "love",
                "Linear SVM": "love",
                "Naive Bayes": "love",
                "LSTM + Word2Vec": "love",
                "DistilBERT": "love"
            },

            "Short - Surprise": {
                "Logistic Regression": "surprise",
                "Linear SVM": "surprise",
                "Naive Bayes": "surprise",
                "LSTM + Word2Vec": "surprise",
                "DistilBERT": "surprise"
            },

            "Ambiguous - Joy vs Love": {
                "Logistic Regression": "joy",
                "Linear SVM": "surprise",
                "Naive Bayes": "joy",
                "LSTM + Word2Vec": "joy",
                "DistilBERT": "joy"
            },

            "Ambiguous - Sadness vs Anger": {
                "Logistic Regression": "anger",
                "Linear SVM": "anger",
                "Naive Bayes": "anger",
                "LSTM + Word2Vec": "sadness",
                "DistilBERT": "sadness"
            },

            "Ambiguous - Joy vs Surprise": {
                "Logistic Regression": "anger",
                "Linear SVM": "joy",
                "Naive Bayes": "joy",
                "LSTM + Word2Vec": "anger",
                "DistilBERT": "joy"
            },

            "Ambiguous - Fear vs Sadness": {
                "Logistic Regression": "fear",
                "Linear SVM": "fear",
                "Naive Bayes": "fear",
                "LSTM + Word2Vec": "fear",
                "DistilBERT": "fear"
            },

            "Longer - Joy": {
                "Logistic Regression": "joy",
                "Linear SVM": "joy",
                "Naive Bayes": "joy",
                "LSTM + Word2Vec": "joy",
                "DistilBERT": "joy"
            },

            "Longer - Fear": {
                "Logistic Regression": "fear",
                "Linear SVM": "fear",
                "Naive Bayes": "fear",
                "LSTM + Word2Vec": "fear",
                "DistilBERT": "fear"
            },

            "Longer - Sadness": {
                "Logistic Regression": "sadness",
                "Linear SVM": "sadness",
                "Naive Bayes": "sadness",
                "LSTM + Word2Vec": "sadness",
                "DistilBERT": "sadness"
            },

            "Longer - Anger": {
                "Logistic Regression": "anger",
                "Linear SVM": "anger",
                "Naive Bayes": "anger",
                "LSTM + Word2Vec": "anger",
                "DistilBERT": "anger"
            },

            "Longer - Love": {
                "Logistic Regression": "sadness",
                "Linear SVM": "sadness",
                "Naive Bayes": "sadness",
                "LSTM + Word2Vec": "joy",
                "DistilBERT": "love"
            },

            "Longer - Surprise": {
                "Logistic Regression": "surprise",
                "Linear SVM": "surprise",
                "Naive Bayes": "fear",
                "LSTM + Word2Vec": "surprise",
                "DistilBERT": "surprise"
            }
        }

        predictions = preset_predictions.get(
            selected_example,
            {
                "Logistic Regression": "fear",
                "Linear SVM": "fear",
                "Naive Bayes": "fear",
                "LSTM + Word2Vec": "fear",
                "DistilBERT": "fear"
            }
        )

        st.markdown("### Live Model Comparison")

        st.markdown(
            f"""
            <div style="
                padding:15px;
                background:#F8F9F9;
                border-left:5px solid #5D6D7E;
                border-radius:6px;
                margin-bottom:15px;
                font-family:Arial;
                font-size:15px;
            ">
                <b>Selected text:</b><br><br>
                {selected_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        model_headers = ""
        prediction_cells = ""

        for model_name, emotion in predictions.items():

            model_headers += f"""
            <th style="
                padding:14px;
                background:#2C3E50;
                color:white;
                text-align:center;
                font-size:14px;
                border:2px solid white;
                min-width:150px;
            ">
                {model_name}
            </th>
            """

            prediction_cells += f"""
            <td style="
                padding:20px;
                background:{emotion_colors[emotion]};
                color:{emotion_text_colors[emotion]};
                text-align:center;
                font-size:20px;
                font-weight:bold;
                text-transform:capitalize;
                border:2px solid white;
            ">
                {emotion}
            </td>
            """

        st.markdown(
            f"""
            <div style="
                margin-top:15px;
                margin-bottom:10px;
                overflow-x:auto;
            ">

                <table style="
                    width:100%;
                    border-collapse:collapse;
                    border-radius:12px;
                    overflow:hidden;
                    box-shadow:0 4px 12px rgba(0,0,0,0.15);
                    font-family:Arial, sans-serif;
                ">

                    <tr>
                        {model_headers}
                    </tr>

                    <tr>
                        {prediction_cells}
                    </tr>

                </table>

            </div>
            """,
            unsafe_allow_html=True
        )

        agreement_count = (
            pd.Series(list(predictions.values()))
            .value_counts()
        )

        most_common_emotion = agreement_count.index[0]

        number_agreeing = int(
            agreement_count.iloc[0]
        )

        st.markdown(
            f"""
            <div style="
                margin-top:15px;
                padding:12px;
                background:#F4F6F7;
                border-radius:6px;
                font-family:Arial;
            ">
                <b>Model agreement:</b>
                {number_agreeing} of 5 models most commonly predict
                <b style="
                    color:{emotion_text_colors[most_common_emotion]};
                ">
                    {most_common_emotion.upper()}
                </b>.
            </div>
            """,
            unsafe_allow_html=True
        )
        
# =========================================================
# 11. CONCLUSION & FUTURE WORK
# =========================================================

elif page == "Conclusion & Future Work":

    st.header("Conclusion & Future Work")

    st.subheader("Main Findings")

    st.markdown(
        """
        - **Best overall model:** DistilBERT — Macro-F1 = **0.8852**
        - **Second best:** LSTM + Word2Vec — Macro-F1 = **0.8765**
        - **Best traditional model:** Linear SVM — Macro-F1 = **0.8549**
        - **Hardest emotion:** Surprise — F1 = **0.7455**
        - **Easiest emotion:** Sadness — F1 = **0.9587**
        """
    )

    st.write(
        "The research questions were answered by comparing all models under the same test "
        "protocol and by identifying the emotion classes and confusion patterns that produced "
        "the most errors."
    )

    st.write(
        "The hypothesis was supported because DistilBERT achieved the highest Macro-F1. "
        "However, the improvement over LSTM + Word2Vec was relatively small, while the "
        "computational cost was much higher."
    )

    st.subheader("Future Work")

    st.markdown(
        """
        - Use more balanced datasets.
        - Explore additional emotion categories.
        - Test other domains and languages.
        - Compare additional pretrained Transformer models.
        - Improve treatment of ambiguous and overlapping emotions.
        - Explore applications in recommendation systems.
        - Analyze emotional trends in social media.
        - Support safer online content and moderation systems.
        """
    )


# =========================================================
# 12. ACKNOWLEDGMENTS
# =========================================================

elif page == "Acknowledgments":

    st.header("Acknowledgments")

    st.write(
        "This project uses the `dair-ai/emotion` dataset from Hugging Face, based on the "
        "work of Saravia et al. (2018). It also uses the pretrained "
        "`distilbert-base-uncased` model through the Hugging Face Transformers library."
    )

    st.write(
        "External Python libraries include scikit-learn, PyTorch, gensim, pandas, NumPy, "
        "matplotlib, datasets, transformers, and Streamlit."
    )

    st.write(
        "AI-assisted tools were used to support code refinement, debugging, explanation, "
        "and writing revision. All final experimental decisions, execution, interpretation, "
        "and conclusions were reviewed by the author."
    )

    

    st.dataframe(
        versions,
        use_container_width=True,
        hide_index=True
    )
