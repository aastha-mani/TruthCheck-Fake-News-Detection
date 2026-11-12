
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading datasets...")

# Load real and fake news datasets
real = pd.read_csv("dataset/True.csv")
fake = pd.read_csv("dataset/Fake.csv")

# Check required columns
for df in [real, fake]:
    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")

# Combine title and article text
def prepare_text(df):
    title = df["title"].fillna("") if "title" in df.columns else ""
    text = df["text"].fillna("")
    return (title + " " + text).astype(str).str.strip()

real_news = pd.DataFrame({
    "news": prepare_text(real),
    "label": 1
})

fake_news = pd.DataFrame({
    "news": prepare_text(fake),
    "label": 0
})

# Combine datasets
data = pd.concat([real_news, fake_news], ignore_index=True)

# Remove empty articles and duplicates
data = data[data["news"].str.len() > 30]
data = data.drop_duplicates(subset="news")

print("Total articles:", len(data))
print("Real articles:", (data["label"] == 1).sum())
print("Fake articles:", (data["label"] == 0).sum())

# Prepare input and output
X = data["news"]
y = data["label"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining TF-IDF vectorizer...")

# Convert news text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 2),
    max_features=100000
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("Training Logistic Regression model...")

# Train model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_vectorized, y_train)

# Evaluate model
predictions = model.predict(X_test_vectorized)

print("\n========== MODEL RESULTS ==========")

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["FAKE", "REAL"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save as NEW files first (do not overwrite old model)
joblib.dump(model, "lr_model_new.jb")
joblib.dump(vectorizer, "vectorizer_new.jb")

print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("Saved: lr_model_new.jb")
print("Saved: vectorizer_new.jb")
print("===================================")
