
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Spam Detector", layout="wide")
st.title("📧 Spam Email Detection with Machine Learning")
st.markdown("Upload a new dataset to retrain or test the spam detection model.")

# Paths
MODEL_PATH = "models/spam_classifier_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Sidebar: model insights
st.sidebar.header("📊 Model Info")

# Load saved model and vectorizer if available
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    return None, None

model, vectorizer = load_model()

# Section 1: Upload CSV and train model
st.subheader("📁 Upload Dataset (CSV)")

uploaded_file = st.file_uploader("Upload a CSV file with two columns: label (ham/spam) and text", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    try:
        # Only use first 2 columns
        df = df.iloc[:, :2]
        df.columns = ["label", "text"]
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        df.dropna(inplace=True)

        # Show dataset details in sidebar
        total_ham = (df["label"] == 0).sum()
        total_spam = (df["label"] == 1).sum()
        st.sidebar.write(f"📄 **Dataset Summary**")
        st.sidebar.write(f"📨 HAM messages: {total_ham}")
        st.sidebar.write(f"🚫 SPAM messages: {total_spam}")
        st.sidebar.write(f"📊 Total messages: {len(df)}")

        # Train model
        X = df["text"]
        y = df["label"]
        vectorizer = TfidfVectorizer(stop_words="english")
        X_vectorized = vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Save updated model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.success(f"✅ Model trained with accuracy: {acc:.2%}")

        # Sidebar: show confusion matrix
        st.sidebar.write(f"✅ **Accuracy:** {acc:.2%}")
        st.sidebar.write("🧮 **Confusion Matrix**")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=["HAM", "SPAM"], yticklabels=["HAM", "SPAM"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.sidebar.pyplot(fig)

        # Matrix interpretation
        tn, fp, fn, tp = cm.ravel()
        st.sidebar.markdown("### 🔍 Interpretation")
        st.sidebar.markdown(f"✅ **{tn}**: HAM correctly predicted as HAM")
        st.sidebar.markdown(f"✅ **{tp}**: SPAM correctly predicted as SPAM")
        st.sidebar.markdown(f"❌ **{fp}**: HAM incorrectly predicted as SPAM")
        st.sidebar.markdown(f"❌ **{fn}**: SPAM incorrectly predicted as HAM")

    except Exception as e:
        st.error(f"Error while processing file: {e}")

# Section 2: Predict custom message
st.subheader("✉️ Test a Message")
user_input = st.text_area("Enter a message to classify", "")

if st.button("Predict"):
    if not model or not vectorizer:
        st.warning("Please upload and train a model first.")
    elif user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]
        if prediction == 1:
            st.error("🚫 SPAM Message Detected!")
        else:
            st.success("✅ This is a HAM (not spam) message.")
