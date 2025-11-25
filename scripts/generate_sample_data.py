"""Generate synthetic sample tweets for testing and demonstration."""

import pandas as pd
import random
from datetime import datetime, timedelta
import os

# Sample tweet templates by sentiment
POSITIVE_TEMPLATES = [
    "I absolutely love {product}! Best purchase ever! {hashtag}",
    "Amazing experience with {company}. Highly recommended! {hashtag}",
    "{product} exceeded all my expectations. Simply perfect! {hashtag}",
    "Fantastic {service}! Will definitely use again. {hashtag}",
    "So happy with my {product}. Worth every penny! {hashtag}",
    "Incredible {service} from {company}. Five stars! {hashtag}",
    "Best {product} I've ever used. Absolutely brilliant! {hashtag}",
    "{company} is amazing! Great customer service! {hashtag}",
]

NEGATIVE_TEMPLATES = [
    "Terrible experience with {product}. Very disappointed. {hashtag}",
    "{company} has awful {service}. Never again! {hashtag}",
    "Waste of money. {product} is complete garbage. {hashtag}",
    "Horrible {service}. Worst purchase ever! {hashtag}",
    "{product} broke after one day. Terrible quality! {hashtag}",
    "Extremely dissatisfied with {company}. Avoid at all costs! {hashtag}",
    "{service} is a complete disaster. Save your money! {hashtag}",
    "Disappointed with {product}. Not worth the hype. {hashtag}",
]

NEUTRAL_TEMPLATES = [
    "Got the {product} today. It's okay, nothing special. {hashtag}",
    "{company} provides decent {service}. Average experience. {hashtag}",
    "The {product} works as described. No complaints. {hashtag}",
    "Used {service} from {company}. It's fine. {hashtag}",
    "{product} is acceptable for the price. {hashtag}",
    "Standard {service}. Nothing to write home about. {hashtag}",
    "{product} meets basic expectations. {hashtag}",
    "Neutral feelings about {company}. It's just okay. {hashtag}",
]

# Sample entities
PRODUCTS = ["iPhone", "laptop", "headphones", "coffee maker", "backpack", "watch", "camera", "shoes"]
COMPANIES = ["Amazon", "Apple", "Google", "Microsoft", "Tesla", "Netflix", "Spotify", "Uber"]
SERVICES = ["delivery", "support", "app", "website", "service", "product", "experience", "platform"]
HASHTAGS = ["#tech", "#review", "#product", "#shopping", "#customerservice", "#quality", "#experience", "#recommendation"]

def generate_tweet(sentiment):
    """Generate a single synthetic tweet."""
    if sentiment == 2:  # Positive
        template = random.choice(POSITIVE_TEMPLATES)
    elif sentiment == 0:  # Negative
        template = random.choice(NEGATIVE_TEMPLATES)
    else:  # Neutral
        template = random.choice(NEUTRAL_TEMPLATES)
    
    tweet = template.format(
        product=random.choice(PRODUCTS),
        company=random.choice(COMPANIES),
        service=random.choice(SERVICES),
        hashtag=random.choice(HASHTAGS)
    )
    
    return tweet

def generate_dataset(num_tweets=1000):
    """Generate synthetic dataset with balanced sentiments."""
    tweets = []
    sentiments = []
    timestamps = []
    
    base_time = datetime.now() - timedelta(days=30)
    
    # Generate balanced dataset
    per_class = num_tweets // 3
    
    for sentiment in [0, 1, 2]:
        for _ in range(per_class):
            tweets.append(generate_tweet(sentiment))
            sentiments.append(sentiment)
            timestamps.append(base_time + timedelta(
                minutes=random.randint(0, 43200)  # 30 days in minutes
            ))
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': tweets,
        'sentiment': sentiments,
        'timestamp': timestamps
    })
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating sample tweets...")
    df = generate_dataset(999)
    
    # Ensure data directory exists
    os.makedirs('../data', exist_ok=True)
    
    # Save to CSV
    output_path = '../data/sample_tweets.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} tweets")
    print(f"✓ Saved to: {output_path}")
    print(f"\nSentiment distribution:")
    print(df['sentiment'].value_counts().sort_index())
