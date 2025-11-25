import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TweetCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Clean tweet text: remove URLs, mentions, special characters
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtag symbols but keep text
        text = re.sub(r'#', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize and lemmatize text
        """
        if not text:
            return []
        
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                  if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def preprocess_dataframe(self, df, text_column='text'):
        """
        Preprocess entire dataframe
        """
        df = df.copy()
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Tokenize
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        
        # Join tokens back to text
        df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        return df

class TextSequencer:
    def __init__(self, max_vocab_size=50000, max_sequence_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
    
    def fit_tokenizer(self, texts):
        """
        Fit tokenizer on texts
        """
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        return self.tokenizer
    
    def texts_to_sequences(self, texts):
        """
        Convert texts to sequences
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call fit_tokenizer first.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_sequence_length, 
                               padding='post', truncating='post')
        return padded
    
    def save_tokenizer(self, path='model/tokenizer.pkl'):
        """
        Save tokenizer to file
        """
        with open(path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def load_tokenizer(self, path='model/tokenizer.pkl'):
        """
        Load tokenizer from file
        """
        with open(path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        return self.tokenizer

def preprocess_pipeline(input_path, output_path, test_size=0.2, val_size=0.1):
    """
    Complete preprocessing pipeline
    """
    # Load data
    df = pd.read_csv(input_path)
    
    # Clean tweets
    cleaner = TweetCleaner()
    df = cleaner.preprocess_dataframe(df)
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    
    return df

if __name__ == '__main__':
    # Example usage
    preprocess_pipeline('data/raw_tweets.csv', 'data/processed_tweets.csv')
