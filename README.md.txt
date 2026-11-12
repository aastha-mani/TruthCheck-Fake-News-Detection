# TruthCheck – AI-Powered Fake News Detection System

## About the Project
TruthCheck is a machine learning-based web application that predicts whether a given news article is REAL or FAKE. It uses Natural Language Processing (NLP), TF-IDF vectorization, and Logistic Regression to analyze news text.

The application is built using Python and Streamlit, providing a simple interface where users can enter news content and receive a prediction.

## Features
- Detects potentially fake and real news articles.
- Uses NLP and TF-IDF for text processing.
- Logistic Regression classification model.
- Interactive Streamlit web interface.
- Live news coverage section using GDELT and Google News RSS.
- Displays prediction results in an easy-to-understand format.

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Joblib
- Requests

## Dataset Description
The model was trained using two datasets:

- Real news articles: 21,195
- Fake news articles: 17,906
- Total articles: 39,101

## Model Training
The project uses the following machine learning approach:

1. Load real and fake news datasets.
2. Combine news titles and article text.
3. Clean the data and remove duplicates.
4. Convert text into numerical features using TF-IDF.
5. Split the dataset into 80% training and 20% testing data.
6. Train a Logistic Regression classifier.
7. Evaluate the model using accuracy, precision, recall, and F1-score.

## Model Performance

**Test Accuracy: 98.43%**

The model achieved 98.43% accuracy on the held-out test dataset. This is an evaluation result on the prepared dataset, not a guarantee of accuracy on every real-world news article.

## Project Structure

```text
TruthCheck-Fake-News-Detection/
│
├── app.py
├── train_model.py
├── lr_model.jb
├── vectorizer.jb
├── requirements.txt
├── .gitignore
└── README.md
```

The training dataset is kept separately and is not included in this repository.

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd TruthCheck-Fake-News-Detection
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Future Scope
- Improve prediction performance on new and unseen news.
- Integrate reliable fact-checking APIs.
- Add source credibility analysis.
- Support multiple languages.
- Improve explainability of model predictions.

## Disclaimer
TruthCheck is an educational machine learning project. Its predictions are based on patterns learned from training data and should not be treated as definitive verification of news authenticity.

## Author
Developed as an academic project using Python, NLP, and Machine Learning.
