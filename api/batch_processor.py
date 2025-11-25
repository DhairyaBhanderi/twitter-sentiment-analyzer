import pandas as pd
import sys
import os
from datetime import datetime
from tqdm import tqdm

sys.path.append('..')

from model.evaluator import ModelEvaluator

class BatchProcessor:
    """
    Process tweets in batch for Power BI export
    """
    
    def __init__(self, model_path='../model/checkpoints/bilstm_best.h5',
                 tokenizer_path='../model/tokenizer.pkl'):
        self.evaluator = ModelEvaluator(model_path, tokenizer_path)
    
    def process_csv(self, input_path, output_path, batch_size=1000):
        """
        Process tweets from CSV and save predictions
        """
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        
        text_col = 'processed_text' if 'processed_text' in df.columns else 'text'
        
        print(f"Processing {len(df)} tweets...")
        
        predictions = []
        confidences = []
        sentiments = []
        
        # Process in batches
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df[text_col].iloc[i:i+batch_size].values
            pred, conf = self.evaluator.predict(batch)
            
            predictions.extend(pred)
            confidences.extend(conf)
            sentiments.extend([self.evaluator.label_map[p] for p in pred])
        
        # Add predictions to dataframe
        df['predicted_label'] = predictions
        df['sentiment'] = sentiments
        df['confidence'] = confidences
        df['prediction_timestamp'] = datetime.now().isoformat()
        
        # Save results
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        return df
    
    def generate_powerbi_export(self, input_path, output_path):
        """
        Generate aggregated data for Power BI
        """
        print(f"Generating Power BI export from {input_path}...")
        
        df = pd.read_csv(input_path)
        
        # Ensure we have predictions
        if 'sentiment' not in df.columns:
            df = self.process_csv(input_path, 'temp_predictions.csv')
        
        # Extract date from created_at or prediction_timestamp
        if 'created_at' in df.columns:
            df['date'] = pd.to_datetime(df['created_at']).dt.date
        elif 'prediction_timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['prediction_timestamp']).dt.date
        else:
            df['date'] = datetime.now().date()
        
        # Aggregate by date and sentiment
        aggregated = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
        
        # Add additional metrics
        if 'confidence' in df.columns:
            avg_confidence = df.groupby(['date', 'sentiment'])['confidence'].mean().reset_index()
            aggregated = aggregated.merge(avg_confidence, on=['date', 'sentiment'])
        
        # Save Power BI export
        aggregated.to_csv(output_path, index=False)
        print(f"Power BI export saved to {output_path}")
        
        return aggregated
    
    def generate_hashtag_analysis(self, input_path, output_path):
        """
        Generate hashtag-based sentiment analysis
        """
        print(f"Generating hashtag analysis from {input_path}...")
        
        df = pd.read_csv(input_path)
        
        # Ensure we have predictions
        if 'sentiment' not in df.columns:
            df = self.process_csv(input_path, 'temp_predictions.csv')
        
        # Filter rows with hashtags
        df_with_hashtags = df[df['hashtags'].notna() & (df['hashtags'] != '')]
        
        # Explode hashtags
        hashtag_rows = []
        for _, row in df_with_hashtags.iterrows():
            hashtags = str(row['hashtags']).split(',')
            for hashtag in hashtags:
                hashtag = hashtag.strip()
                if hashtag:
                    hashtag_rows.append({
                        'hashtag': hashtag,
                        'sentiment': row['sentiment'],
                        'confidence': row.get('confidence', 0)
                    })
        
        hashtag_df = pd.DataFrame(hashtag_rows)
        
        # Aggregate by hashtag and sentiment
        hashtag_analysis = hashtag_df.groupby(['hashtag', 'sentiment']).agg({
            'sentiment': 'count',
            'confidence': 'mean'
        }).rename(columns={'sentiment': 'count', 'confidence': 'avg_confidence'}).reset_index()
        
        # Save
        hashtag_analysis.to_csv(output_path, index=False)
        print(f"Hashtag analysis saved to {output_path}")
        
        return hashtag_analysis

def main():
    """
    Example usage
    """
    processor = BatchProcessor(
        model_path='../model/checkpoints/bilstm_best.h5',
        tokenizer_path='../model/tokenizer.pkl'
    )
    
    # Process raw tweets
    processor.process_csv(
        input_path='../data/processed_tweets.csv',
        output_path='../data/predictions.csv'
    )
    
    # Generate Power BI export
    processor.generate_powerbi_export(
        input_path='../data/predictions.csv',
        output_path='../powerbi/sentiment_data.csv'
    )
    
    # Generate hashtag analysis
    processor.generate_hashtag_analysis(
        input_path='../data/predictions.csv',
        output_path='../powerbi/hashtag_sentiment.csv'
    )

if __name__ == '__main__':
    main()
