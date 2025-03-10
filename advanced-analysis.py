import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import datetime
import re
from collections import Counter

# Download NLTK resources
nltk.download('vader_lexicon')

def load_processed_data(file_path='processed_security_data.csv'):
    """
    Load the processed data from the main analysis
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Processed data loaded with {df.shape[0]} rows")
        return df
    except Exception as e:
        print(f"Error loading processed data: {e}")
        print("Please run the main analysis script first")
        return None

def perform_sentiment_analysis(df):
    """
    Analyze sentiment of tweets to gauge severity
    """
    if 'Tweets' not in df.columns:
        print("Tweets column not found in dataset")
        return df
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Function to get sentiment scores
    def get_sentiment(text):
        if not isinstance(text, str):
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
        return sia.polarity_scores(text)
    
    # Apply sentiment analysis
    df['sentiment'] = df['Tweets'].apply(get_sentiment)
    
    # Extract sentiment components
    df['sentiment_negative'] = df['sentiment'].apply(lambda x: x['neg'])
    df['sentiment_neutral'] = df['sentiment'].apply(lambda x: x['neu'])
    df['sentiment_positive'] = df['sentiment'].apply(lambda x: x['pos'])
    df['sentiment_compound'] = df['sentiment'].apply(lambda x: x['compound'])
    
    # Drop the dictionary column
    df = df.drop('sentiment', axis=1)
    
    # Create a sentiment category
    def categorize_sentiment(compound):
        if compound <= -0.05:
            return 'negative'
        elif compound >= 0.05:
            return 'positive'
        else:
            return 'neutral'
    
    df['sentiment_category'] = df['sentiment_compound'].apply(categorize_sentiment)
    
    # Print sentiment distribution
    sentiment_dist = df['sentiment_category'].value_counts()
    print("Sentiment distribution:")
    print(sentiment_dist)
    
    return df

def topic_modeling(df, n_topics=5):
    """
    Perform topic modeling to identify main crime themes
    """
    if 'processed_tweet' not in df.columns:
        print("processed_tweet column not found - please run the main script first")
        return df
    
    # Create TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(df['processed_tweet'].fillna(''))
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # LDA Topic Modeling
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf)
    
    # Get the top words for each topic
    def print_top_words(model, feature_names, n_top_words=10):
        topics = {}
        for topic_idx, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics[f"Topic {topic_idx+1}"] = top_words
            print(f"Topic {topic_idx+1}: {' '.join(top_words)}")
        return topics
    
    print("\nTop words in each topic:")
    topics = print_top_words(lda, feature_names)
    
    # Assign topics to tweets
    topic_assignments = lda.transform(tfidf)
    df['primary_topic'] = np.argmax(topic_assignments, axis=1) + 1
    
    # Create a descriptive name for each topic
    topic_names = {
        1: "Violent Crimes",
        2: "Property Crimes",
        3: "Public Safety",
        4: "Police Activity",
        5: "Traffic/Incidents"
    }
    
    # Map the numerical topics to descriptive names
    df['topic_name'] = df['primary_topic'].map(lambda x: topic_names.get(x, f"Topic {x}"))
    
    return df, topics

def temporal_analysis(df):
    """
    Analyze crime patterns by time
    """
    if 'date' not in df.columns:
        print("date column not found in dataset")
        return
    
    # Convert to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extract time components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    
    # Analyze by day of week
    plt.figure(figsize=(12, 6))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.countplot(x='day_of_week', data=df, order=day_order)
    plt.title('Crime Reports by Day of Week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('crime_by_day.png')
    
    # Analyze by hour (time of day)
    plt.figure(figsize=(12, 6))
    hourly_crimes = df['hour'].value_counts().sort_index()
    hourly_crimes.plot(kind='bar')
    plt.title('Crime Reports by Hour of Day')
    plt.xlabel('Hour (24-hour format)')
    plt.ylabel('Number of Reports')
    plt.tight_layout()
    plt.savefig('crime_by_hour.png')
    
    # Create heatmap of day vs hour
    plt.figure(figsize=(12, 8))
    day_hour = pd.crosstab(df['day_of_week'], df['hour'])
    day_hour = day_hour.reindex(day_order)
    sns.heatmap(day_hour, cmap='YlOrRd')
    plt.title('Crime Reports by Day and Hour')
    plt.tight_layout()
    plt.savefig('crime_heatmap_day_hour.png')
    
    print("Temporal analysis visualizations saved")
    return df

def analyze_engagement(df):
    """
    Analyze social media engagement with crime reports
    """
    if 'likes' not in df.columns or 'retweets' not in df.columns:
        print("Engagement columns not found in dataset")
        return df
    
    # Create a total engagement metric
    df['total_engagement'] = df['likes'] + df['retweets']
    
    # Analyze engagement by crime category
    plt.figure(figsize=(12, 6))
    engagement_by_category = df.groupby('crime_category')['total_engagement'].mean().sort_values(ascending=False)
    engagement_by_category.plot(kind='bar')
    plt.title('Average Engagement by Crime Category')
    plt.ylabel('Average Likes + Retweets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('engagement_by_category.png')
    
    # Analyze engagement by security level
    plt.figure(figsize=(10, 6))
    engagement_by_severity = df.groupby('security_level')['total_engagement'].mean().reindex(['high', 'moderate', 'low', 'unknown'])
    engagement_by_severity.plot(kind='bar', color=['red', 'orange', 'green', 'gray'])
    plt.title('Average Engagement by Security Level')
    plt.ylabel('Average Likes + Retweets')
    plt.tight_layout()
    plt.savefig('engagement_by_severity.png')
    
    # Find most engaging tweets
    top_tweets = df.sort_values('total_engagement', ascending=False).head(10)[['Tweets', 'likes', 'retweets', 'total_engagement', 'crime_category']]
    print("\nTop 10 most engaging tweets:")
    print(top_tweets)
    
    return df

def generate_word_clouds(df):
    """
    Generate word clouds for different crime categories
    """
    if 'processed_tweet' not in df.columns or 'crime_category' not in df.columns:
        print("Required columns not found in dataset")
        return
    
    # Create word clouds for each crime category
    categories = df['crime_category'].unique()
    
    plt.figure(figsize=(15, 15))
    for i, category in enumerate(categories, 1):
        if len(categories) <= 6:
            plt.subplot(2, 3, i)
        else:
            plt.subplot(3, 3, i)
        
        # Get text for this category
        text = ' '.join(df[df['crime_category'] == category]['processed_tweet'].fillna(''))
        
        if text.strip():
            # Generate word cloud
            wordcloud = WordCloud(width=400, height=400, background_color='white', 
                                 max_words=100, contour_width=3, contour_color='steelblue').generate(text)
            
            # Display
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Word Cloud: {category}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('crime_wordclouds.png')
    print("Word clouds generated and saved as 'crime_wordclouds.png'")
    
    return

def create_security_dashboard(df):
    """
    Create a comprehensive security dashboard
    """
    # Create a figure for the dashboard
    plt.figure(figsize=(20, 15))
    
    # 1. Security levels by borough
    plt.subplot(2, 2, 1)
    security_by_location = pd.crosstab(df['primary_location'], df['security_level'])
    # Filter to only include major boroughs
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    borough_security = security_by_location.loc[security_by_location.index.isin(boroughs)]
    
    # Reorder for visualization
    if not borough_security.empty:
        borough_security = borough_security.reindex(columns=['high', 'moderate', 'low', 'unknown'])
        borough_security.plot(kind='bar', stacked=True, color=['red', 'orange', 'green', 'gray'])
        plt.title('Security Incidents by Borough')
        plt.xticks(rotation=45)
    
    # 2. Sentiment analysis results
    plt.subplot(2, 2, 2)
    if 'sentiment_category' in df.columns:
        sentiment_counts = df['sentiment_category'].value_counts()
        colors = {'negative': 'red', 'neutral': 'gray', 'positive': 'green'}
        sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=[colors[c] for c in sentiment_counts.index])
        plt.title('Tweet Sentiment Analysis')
        plt.ylabel('')
    
    # 3. Top crime categories
    plt.subplot(2, 2, 3)
    category_counts = df['crime_category'].value_counts().head(6)
    category_counts.plot(kind='bar')
    plt.title('Top Crime Categories')
    plt.xticks(rotation=45)
    
    # 4. Time series of incidents
    plt