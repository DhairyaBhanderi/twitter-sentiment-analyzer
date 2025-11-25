# Twitter API Configuration Template
# Copy this to config.py and fill in your credentials

TWITTER_CONFIG = {
    'api_key': 'your_api_key',
    'api_secret': 'your_api_secret',
    'access_token': 'your_access_token',
    'access_secret': 'your_access_secret',
    'bearer_token': 'your_bearer_token'
}

# Collection Parameters
COLLECTION_CONFIG = {
    'max_tweets': 500000,
    'batch_size': 100,
    'rate_limit_wait': 2,  # seconds
    'queries': [
        'politics lang:en -is:retweet',
        'sports lang:en -is:retweet',
        'technology lang:en -is:retweet',
        'entertainment lang:en -is:retweet'
    ]
}

# Output Paths
DATA_PATHS = {
    'raw_tweets': 'data/raw_tweets.csv',
    'processed_tweets': 'data/processed_tweets.csv',
    'train_data': 'data/train.csv',
    'test_data': 'data/test.csv',
    'val_data': 'data/val.csv'
}
