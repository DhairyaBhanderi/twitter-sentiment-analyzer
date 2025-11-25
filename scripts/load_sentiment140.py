"""Load and preprocess Sentiment140 dataset."""

import pandas as pd
import os

def load_sentiment140(input_path='../data/sentiment140.csv', output_path='../data/train.csv', sample_size=100000):
    """
    Load Sentiment140 dataset and convert to project format.
    
    Sentiment140 format: [polarity, id, date, query, user, text]
    Project format: [text, sentiment, timestamp]
    
    Polarity mapping: 0 (negative) -> 0, 4 (positive) -> 2
    """
    print(f"Loading Sentiment140 dataset from {input_path}...")
    
    # Column names for Sentiment140
    columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    
    # Load dataset
    df = pd.read_csv(
        input_path,
        encoding='ISO-8859-1',
        names=columns,
        header=None
    )
    
    print(f"✓ Loaded {len(df):,} tweets")
    
    # Sample if needed
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size:,} tweets...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Map polarities: 0 -> 0 (negative), 4 -> 2 (positive)
    print("Converting polarity labels...")
    df['sentiment'] = df['polarity'].map({0: 0, 4: 2})
    
    # Select relevant columns
    df = df[['text', 'sentiment', 'date']].copy()
    df.rename(columns={'date': 'timestamp'}, inplace=True)
    
    # Remove any null values
    df.dropna(inplace=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed dataset
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Successfully processed {len(df):,} tweets")
    print(f"✓ Saved to: {output_path}")
    print(f"\nSentiment distribution:")
    print(df['sentiment'].value_counts().sort_index())
    
    return df

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    input_file = sys.argv[1] if len(sys.argv) > 1 else '../data/sentiment140.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else '../data/train.csv'
    sample_size = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        print("\nTo use Sentiment140 dataset:")
        print("1. Download from: https://www.kaggle.com/datasets/kazanova/sentiment140")
        print("2. Extract to: data/sentiment140.csv")
        print("3. Run: python load_sentiment140.py")
        sys.exit(1)
    
    # Load and process
    load_sentiment140(input_file, output_file, sample_size)
