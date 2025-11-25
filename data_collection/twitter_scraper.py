import tweepy
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class TwitterScraper:
    def __init__(self):
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_secret = os.getenv('TWITTER_ACCESS_SECRET')
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_secret,
            wait_on_rate_limit=True
        )
    
    def search_tweets(self, query, max_results=100, start_time=None, end_time=None):
        """
        Search tweets using Twitter API v2
        """
        tweets_data = []
        
        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'geo', 'public_metrics', 'entities'],
                expansions=['author_id', 'geo.place_id'],
                start_time=start_time,
                end_time=end_time
            )
            
            if tweets.data:
                for tweet in tweets.data:
                    hashtags = []
                    if tweet.entities and 'hashtags' in tweet.entities:
                        hashtags = [tag['tag'] for tag in tweet.entities['hashtags']]
                    
                    tweets_data.append({
                        'tweet_id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'hashtags': ','.join(hashtags),
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'quote_count': tweet.public_metrics['quote_count']
                    })
        
        except tweepy.errors.TweepyException as e:
            print(f"Error fetching tweets: {e}")
        
        return tweets_data
    
    def collect_tweets(self, queries, max_tweets=500000, batch_size=100):
        """
        Collect tweets for multiple queries
        """
        all_tweets = []
        collected = 0
        
        for query in queries:
            print(f"Collecting tweets for query: {query}")
            
            while collected < max_tweets:
                batch_tweets = self.search_tweets(query, max_results=batch_size)
                
                if not batch_tweets:
                    print(f"No more tweets found for query: {query}")
                    break
                
                all_tweets.extend(batch_tweets)
                collected += len(batch_tweets)
                
                print(f"Collected {collected}/{max_tweets} tweets")
                
                time.sleep(2)
                
                if collected >= max_tweets:
                    break
        
        return all_tweets
    
    def save_to_csv(self, tweets, output_path='data/raw_tweets.csv'):
        """
        Save tweets to CSV file
        """
        df = pd.DataFrame(tweets)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(tweets)} tweets to {output_path}")
        return df

def main():
    scraper = TwitterScraper()
    
    # Example queries - customize based on your needs
    queries = [
        'python lang:en -is:retweet',
        'machine learning lang:en -is:retweet',
        'AI lang:en -is:retweet',
        'data science lang:en -is:retweet'
    ]
    
    tweets = scraper.collect_tweets(queries, max_tweets=10000, batch_size=100)
    scraper.save_to_csv(tweets)

if __name__ == '__main__':
    main()
