import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from transformers import pipeline
from langdetect import detect

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("🌍 Multilingual Sentiment Analysis Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Settings")

model_choice = st.sidebar.radio(
    "Choose Model",
    ["VADER", "TextBlob", "BERT", "LSTM"]
)

pos_thresh = st.sidebar.slider("Positive Threshold", 0.0, 1.0, 0.05)
neg_thresh = st.sidebar.slider("Negative Threshold", -1.0, 0.0, -0.05)

# ---------------- MODELS ----------------
analyzer = SentimentIntensityAnalyzer()

@st.cache_resource
def load_bert():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

bert_model = load_bert()

# ---------------- FUNCTIONS ----------------
def detect_language(text):
    try:
        lang = detect(text)
        if lang == "en":
            return "English"
        elif lang == "hi":
            return "Hindi"
        elif lang == "te":
            return "Telugu"
        else:
            return "Other"
    except:
        return "Unknown"

def vader_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= pos_thresh:
        return "Positive"
    elif score <= neg_thresh:
        return "Negative"
    return "Neutral"

def textblob_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"


def bert_sentiment(text):
    result = bert_model(text)[0]["label"]
    stars = int(result[0])
    if stars >= 4:
        return "Positive"
    elif stars == 3:
        return "Neutral"
    return "Negative"

def lstm_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    return "Neutral"

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📁 Upload Data", "📋 Results", "📊 Dashboard"])

# ---------------- TAB 1 (UPLOAD) ----------------
with tab1:
    st.subheader("Upload Dataset")

    file = st.file_uploader("Upload CSV / Excel file", type=["csv", "xlsx"])

    if file is not None:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)

            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)

            else:
                st.error("Unsupported file format")
                st.stop()

            st.success("File uploaded successfully!")
            st.dataframe(df.head())

            st.session_state["data"] = df

        except ImportError:
            st.error("⚠ Please install 'openpyxl' to read Excel files")

        except Exception as e:
            st.error(f"Error reading file: {e}")

# ---------------- TAB 2 (RESULTS) ----------------
with tab2:
    if "data" not in st.session_state:
        st.warning("Upload dataset first")
    else:
        df = st.session_state["data"]

        col = st.selectbox("Select Review Column", df.columns)
        df[col] = df[col].astype(str)

        languages = []
        sentiments = []

        for text in df[col]:
            languages.append(detect_language(text))

            if model_choice == "VADER":
                sentiments.append(vader_sentiment(text))
            elif model_choice == "TextBlob":
                sentiments.append(textblob_sentiment(text))
            elif model_choice == "BERT":
                sentiments.append(bert_sentiment(text))
            else:
                sentiments.append(lstm_sentiment(text))

        df["Language"] = languages
        df["Sentiment"] = sentiments

        st.session_state["result"] = df

        st.dataframe(df[[col, "Language", "Sentiment"]])

# ---------------- TAB 3 (DASHBOARD) ----------------
with tab3:
    if "result" not in st.session_state:
        st.warning("Run analysis first")
    else:
        df = st.session_state["result"]

        col1, col2 = st.columns(2)

        # PIE CHART
        with col1:
            counts = df["Sentiment"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]

            fig = px.pie(counts, names="Sentiment", values="Count", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        # LANGUAGE BAR
        with col2:
            lang_counts = df["Language"].value_counts().reset_index()
            lang_counts.columns = ["Language", "Count"]

            fig2 = px.bar(lang_counts, x="Language", y="Count", text="Count")
            st.plotly_chart(fig2, use_container_width=True)

        # ACCURACY COMPARISON
        st.subheader("Model Accuracy Comparison")

        if "Actual" in df.columns:
            review_col = df.columns[0]

            vader_preds = df[review_col].apply(vader_sentiment)
            tb_preds = df[review_col].apply(textblob_sentiment)
            bert_preds = df[review_col].apply(bert_sentiment)
            lstm_preds = df[review_col].apply(lstm_sentiment)

            acc_df = pd.DataFrame({
                "Model": ["VADER", "TextBlob", "BERT", "LSTM"],
                "Accuracy": [
                    accuracy_score(df["Actual"], vader_preds),
                    accuracy_score(df["Actual"], tb_preds),
                    accuracy_score(df["Actual"], bert_preds),
                    accuracy_score(df["Actual"], lstm_preds)
                ]
            })

            fig3 = px.bar(acc_df, x="Model", y="Accuracy", text="Accuracy")
            st.plotly_chart(fig3, use_container_width=True)

            best = acc_df.loc[acc_df["Accuracy"].idxmax()]
            st.success(f"🏆 Best Model: {best['Model']} ({best['Accuracy']:.2f})")

        else:
            st.info("Add 'Actual' column in dataset for accuracy comparison")

        # DOWNLOAD
        st.download_button(
            "⬇ Download Results",
            df.to_csv(index=False),
            file_name="results.csv",
            mime="text/csv"
        )