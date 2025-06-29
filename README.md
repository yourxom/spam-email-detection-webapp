# 📧 Spam Email Detection Using Machine Learning

This project uses Natural Language Processing (NLP) and Machine Learning (ML) to build a system that classifies email messages as **Spam** or **Ham (Not Spam)**. It also features a user-friendly **Streamlit web app** for real-time message testing and model retraining.

---

## 🚀 Features

* 🧹 Data cleaning and preprocessing
* 🧠 TF-IDF vectorization of email text
* 🤖 Trained with Multinomial Naive Bayes classifier
* 📊 Accuracy, Precision, Recall, and Confusion Matrix
* 🌐 Interactive Streamlit web app

---

## 📁 Project Structure

```
📦 spam-email-detection
├── models/
│   ├── spam_classifier_model.pkl
│   └── tfidf_vectorizer.pkl
├── spam.csv                  # Dataset
├── app_streamlit_final_interpreted.py  # Streamlit Web App
├── Email Spam Detection.ipynb          # Jupyter Notebook
├── README.md                # Project Overview
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/spam-email-detection.git
cd spam-email-detection
```

2. **Create a Virtual Environment (Optional but Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install Required Packages**

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn joblib
```

4. **Run the App**

```bash
streamlit run app_streamlit_final_interpreted.py
```

Then open your browser to: `http://localhost:8501`

---

## 🧪 Usage

* Upload `spam.csv` or a custom dataset with `label` and `text` columns
* View accuracy and confusion matrix on sidebar
* Test messages in real time using the input box

---

## 📦 Deploy on GitHub + Streamlit Cloud

1. **Push Project to GitHub**

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/spam-email-detection.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud**

* Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
* Click **'New App'**
* Connect your GitHub repo
* Choose:

  * Repository: `spam-email-detection`
  * Branch: `main`
  * App file: `app_streamlit_final_interpreted.py`
* Click **Deploy**

Your app will be live in minutes! 🚀

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Credits

* Project by Om Tripathi & \[Second Member Name]
* Built using Python, scikit-learn, and Streamlit

---

## 💬 Questions?

Feel free to open issues or contact us!
