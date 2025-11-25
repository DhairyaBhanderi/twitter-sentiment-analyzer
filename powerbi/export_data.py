import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append('..')

class PowerBIExporter:
    """
    Export sentiment data for Power BI dashboard
    """
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def load_predictions(self, path='../data/predictions.csv'):
        """
        Load predictions CSV
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Predictions file not found: {path}")
        
        return pd.read_csv(path)
    
    def create_daily_sentiment_summary(self, df, output_path='sentiment_data.csv'):
        """
        Create daily sentiment summary for Power BI
        """
        # Parse date
        if 'created_at' in df.columns:
            df['date'] = pd.to_datetime(df['created_at']).dt.date
        elif 'prediction_timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['prediction_timestamp']).dt.date
        else:
            df['date'] = datetime.now().date()
        
        # Aggregate by date and sentiment
        summary = df.groupby(['date', 'sentiment']).agg({
            'sentiment': 'count',
            'confidence': ['mean', 'min', 'max']
        }).reset_index()
        
        summary.columns = ['date', 'sentiment', 'count', 'avg_confidence', 'min_confidence', 'max_confidence']
        
        # Save
        summary.to_csv(output_path, index=False)
        print(f"Daily sentiment summary saved to {output_path}")
        
        return summary
    
    def create_hashtag_sentiment(self, df, output_path='hashtag_sentiment.csv', top_n=50):
        """
        Create hashtag sentiment analysis
        """
        # Filter rows with hashtags
        df_hashtags = df[df['hashtags'].notna() & (df['hashtags'] != '')]
        
        # Explode hashtags
        hashtag_data = []
        for _, row in df_hashtags.iterrows():
            hashtags = str(row['hashtags']).split(',')
            for hashtag in hashtags:
                hashtag = hashtag.strip()
                if hashtag:
                    hashtag_data.append({
                        'hashtag': hashtag,
                        'sentiment': row['sentiment'],
                        'confidence': row.get('confidence', 0)
                    })
        
        hashtag_df = pd.DataFrame(hashtag_data)
        
        # Get top hashtags by volume
        top_hashtags = hashtag_df['hashtag'].value_counts().head(top_n).index
        hashtag_df = hashtag_df[hashtag_df['hashtag'].isin(top_hashtags)]
        
        # Aggregate
        summary = hashtag_df.groupby(['hashtag', 'sentiment']).agg({
            'sentiment': 'count',
            'confidence': 'mean'
        }).reset_index()
        
        summary.columns = ['hashtag', 'sentiment', 'count', 'avg_confidence']
        
        # Save
        summary.to_csv(output_path, index=False)
        print(f"Hashtag sentiment saved to {output_path}")
        
        return summary
    
    def create_geography_sentiment(self, df, output_path='geography_sentiment.csv'):
        """
        Create geography-based sentiment analysis
        """
        # Check if location data exists
        if 'user_location' not in df.columns:
            print("No location data available. Skipping geography export.")
            return None
        
        # Filter rows with location
        df_geo = df[df['user_location'].notna() & (df['user_location'] != '')]
        
        if len(df_geo) == 0:
            print("No geographic data available.")
            return None
        
        # Aggregate by location and sentiment
        summary = df_geo.groupby(['user_location', 'sentiment']).agg({
            'sentiment': 'count',
            'confidence': 'mean'
        }).reset_index()
        
        summary.columns = ['location', 'sentiment', 'count', 'avg_confidence']
        
        # Save
        summary.to_csv(output_path, index=False)
        print(f"Geography sentiment saved to {output_path}")
        
        return summary
    
    def create_time_series(self, df, output_path='time_series.csv'):
        """
        Create time series data for trend analysis
        """
        if 'created_at' in df.columns:
            df['datetime'] = pd.to_datetime(df['created_at'])
        elif 'prediction_timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['prediction_timestamp'])
        else:
            print("No timestamp data available.")
            return None
        
        # Create hourly aggregation
        df['hour'] = df['datetime'].dt.floor('H')
        
        time_series = df.groupby(['hour', 'sentiment']).agg({
            'sentiment': 'count',
            'confidence': 'mean'
        }).reset_index()
        
        time_series.columns = ['hour', 'sentiment', 'count', 'avg_confidence']
        
        # Save
        time_series.to_csv(output_path, index=False)
        print(f"Time series data saved to {output_path}")
        
        return time_series
    
    def export_all(self, predictions_path='../data/predictions.csv', output_dir='./'):
        """
        Export all Power BI datasets
        """
        print("Loading predictions...")
        df = self.load_predictions(predictions_path)
        
        print(f"\nProcessing {len(df)} tweets...")
        
        # Create all exports
        self.create_daily_sentiment_summary(df, f'{output_dir}/sentiment_data.csv')
        self.create_hashtag_sentiment(df, f'{output_dir}/hashtag_sentiment.csv')
        self.create_geography_sentiment(df, f'{output_dir}/geography_sentiment.csv')
        self.create_time_series(df, f'{output_dir}/time_series.csv')
        
        print("\nAll Power BI exports completed!")

def main():
    """
    Main export function
    """
    exporter = PowerBIExporter()
    
    # Export all datasets
    exporter.export_all(
        predictions_path='../data/predictions.csv',
        output_dir='./'
    )

if __name__ == '__main__':
    main()
