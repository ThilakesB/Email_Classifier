import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    
    # In a real deployed app, it's better to verify stopwords are present
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return " ".join(words)

def plot_top_words():
    print("Loading dataset to find top spam words...")
    data_path = os.path.join("data", "SMSSpamCollection")
    if not os.path.exists(data_path):
        print("Dataset not found. Cannot plot top words.")
        return
        
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])
    spam_msgs = df[df['label'] == 'spam']['message']
    
    WORDS = []
    stop_words = set(stopwords.words('english'))
    for msg in spam_msgs:
        text = str(msg).lower()
        text = "".join([c for c in text if c not in string.punctuation])
        for w in text.split():
            if w not in stop_words and len(w) > 2:
                WORDS.append(w)
                
    word_counts = pd.Series(WORDS).value_counts().head(20)
    
    plt.figure(figsize=(10, 6))
    word_counts.plot(kind='bar', color='red')
    plt.title('Top 20 Most Frequent Words in Spam Emails/SMS')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plot_path = "spam_top_words.png"
    plt.savefig(plot_path)
    print(f"Plot saved successfully as {plot_path}")

def main():
    model_path = os.path.join("models", "spam_classifier_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Model or vectorizer not found. Please run 'python train_model.py' first.")
        return
        
    print("Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    print("\n" + "="*50)
    print("📧 EMAIL/SMS SPAM CLASSIFIER CLI 📧")
    print("="*50)
    print("Commands:")
    print("  'exit' or 'quit' - stop the application")
    print("  'plot' - generate a bar plot of top spam words")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("\nEnter a message to classify: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting classifier. Goodbye!")
                break
            elif user_input.lower() == 'plot':
                plot_top_words()
                continue
            elif not user_input.strip():
                continue
                
            processed = preprocess_text(user_input)
            vec_input = vectorizer.transform([processed])
            prediction = model.predict(vec_input)[0]
            
            if prediction == 1:
                print("\n🚨 [RESULT]: SPAM")
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vec_input)[0]
                    print(f"   Confidence: {proba[1]*100:.2f}%")
            else:
                print("\n✅ [RESULT]: HAM (Not Spam)")
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vec_input)[0]
                    print(f"   Confidence: {proba[0]*100:.2f}%")
                    
        except KeyboardInterrupt:
            print("\nExiting classifier. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
