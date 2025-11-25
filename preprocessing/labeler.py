import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class SentimentLabeler:
    """
    Handle sentiment labeling for tweets
    Uses Sentiment140 dataset format or manual labeling
    """
    
    def __init__(self):
        self.label_map = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }
    
    def load_sentiment140(self, path, encoding='latin-1', sample_size=None):
        """
        Load Sentiment140 dataset
        Format: target, ids, date, flag, user, text
        Target: 0 = negative, 4 = positive
        """
        columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(path, encoding=encoding, names=columns)
        
        # Map 0 -> negative, 4 -> positive
        df['sentiment'] = df['target'].map({0: 0, 4: 2})
        
        # Sample if needed
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        return df[['text', 'sentiment']]
    
    def add_neutral_class(self, df, neutral_keywords=None):
        """
        Optionally add neutral class using keyword-based heuristics
        """
        if neutral_keywords is None:
            neutral_keywords = ['maybe', 'perhaps', 'okay', 'fine', 'alright']
        
        # Simple heuristic: mark tweets with neutral keywords as neutral
        df = df.copy()
        for keyword in neutral_keywords:
            mask = df['text'].str.contains(keyword, case=False, na=False)
            df.loc[mask, 'sentiment'] = 1
        
        return df
    
    def split_dataset(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset into train, validation, and test sets
        """
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df['sentiment']
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=random_state,
            stratify=train_val['sentiment']
        )
        
        return train, val, test
    
    def save_splits(self, train, val, test, base_path='data/'):
        """
        Save train, validation, and test splits
        """
        train.to_csv(f'{base_path}train.csv', index=False)
        val.to_csv(f'{base_path}val.csv', index=False)
        test.to_csv(f'{base_path}test.csv', index=False)
        
        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        print(f"Saved to {base_path}")
    
    def get_class_distribution(self, df):
        """
        Get class distribution statistics
        """
        dist = df['sentiment'].value_counts().sort_index()
        print("\nClass Distribution:")
        for label, count in dist.items():
            label_name = self.label_map.get(label, 'unknown')
            percentage = (count / len(df)) * 100
            print(f"{label_name}: {count} ({percentage:.2f}%)")
        
        return dist

def prepare_labeled_dataset(sentiment140_path=None, output_base_path='data/'):
    """
    Main function to prepare labeled dataset
    """
    labeler = SentimentLabeler()
    
    if sentiment140_path:
        # Load Sentiment140 dataset
        print("Loading Sentiment140 dataset...")
        df = labeler.load_sentiment140(sentiment140_path, sample_size=500000)
    else:
        # Load already collected tweets (would need manual labeling)
        print("Loading collected tweets...")
        df = pd.read_csv('data/processed_tweets.csv')
        
        # For showcase, create synthetic labels based on simple heuristics
        # In production, use pre-labeled dataset or manual labeling
        print("Creating synthetic labels for demonstration...")
        df['sentiment'] = np.random.choice([0, 1, 2], size=len(df), p=[0.3, 0.2, 0.5])
    
    # Show distribution
    labeler.get_class_distribution(df)
    
    # Split dataset
    print("\nSplitting dataset...")
    train, val, test = labeler.split_dataset(df)
    
    # Save splits
    labeler.save_splits(train, val, test, base_path=output_base_path)
    
    return train, val, test

if __name__ == '__main__':
    # Example: Use Sentiment140 dataset
    # Download from: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
    # prepare_labeled_dataset('path/to/training.1600000.processed.noemoticon.csv')
    
    # Or use collected tweets with synthetic labels
    prepare_labeled_dataset()
