import os
import string
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Download NLTK data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return " ".join(words)

def main():
    data_path = os.path.join("data", "SMSSpamCollection")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run dataset_fetcher.py first.")
        return
        
    print("Loading data...")
    # The UCI SMS Spam collection is tab separated: label \t message
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])
    
    # Map labels to binary values
    df['labelNum'] = df.label.map({'ham': 0, 'spam': 1})
    if df['labelNum'].isnull().any():
        print("Warning: Some labels could not be mapped! Dropping invalid rows.")
        df = df.dropna(subset=['labelNum'])
        
    print("Preprocessing text... This might take a few moments.")
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    X = df['processed_message']
    y = df['labelNum']
    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (Linear)": SVC(kernel='linear', probability=True)
    }
    
    best_model_name = ""
    best_model = None
    best_score = 0
    
    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        
        # We prioritize precision to avoid classifying ham as spam
        score = acc + prec
        if score > best_score:
            best_score = score
            best_model_name = name
            best_model = model

    print(f"\nBest model selected: {best_model_name}")
    
    # Save the model and vectorizer
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, "spam_classifier_model.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    print(f"Model and vectorizer saved to '{models_dir}/' directory.")

if __name__ == "__main__":
    main()
