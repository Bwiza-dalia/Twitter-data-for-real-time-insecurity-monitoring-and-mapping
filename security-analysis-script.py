import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# No need for punkt_tab as we'll modify our approach

# Step 1: Load the data
def load_data(file_path):
    """
    Load the Twitter dataset from CSV
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Step 2: Preprocess the tweets
def preprocess_tweets(df):
    """
    Clean and preprocess the tweets
    """
    if 'Tweets' not in df.columns:
        print("Column 'Tweets' not found in the dataset")
        return df
    
    # Create a copy to avoid changing the original data
    df_processed = df.copy()
    
    # Function to clean individual tweets
    def clean_tweet(tweet):
        if not isinstance(tweet, str):
            return ""
        
        # Remove URLs
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
        
        # Remove user mentions
        tweet = re.sub(r'@\w+', '', tweet)
        
        # Remove hashtags symbol (but keep the text)
        tweet = re.sub(r'#', '', tweet)
        
        # Remove special characters and numbers
        tweet = re.sub(r'[^\w\s]', '', tweet)
        tweet = re.sub(r'\d+', '', tweet)
        
        # Convert to lowercase
        tweet = tweet.lower()
        
        # Remove extra spaces
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        
        return tweet
    
    # Apply the cleaning function
    df_processed['cleaned_tweet'] = df_processed['Tweets'].apply(clean_tweet)
    
    # Tokenization and stopword removal
    # Use a simpler approach to avoid NLTK issues
    stop_words = set(stopwords.words('english'))
    
    def simple_tokenize(text):
        if not isinstance(text, str):
            return ""
        # Split on whitespace instead of using word_tokenize
        tokens = text.split()
        # Filter out stopwords
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    df_processed['processed_tweet'] = df_processed['cleaned_tweet'].apply(simple_tokenize)
    
    print("Tweet preprocessing completed")
    return df_processed

# Step 3: Extract locations from tweets
def extract_locations(df):
    """
    Extract location mentions from tweets using keyword matching
    """
    if 'Tweets' not in df.columns:
        print("Column 'Tweets' not found in the dataset")
        return df
    
    # Create a copy
    df_with_locations = df.copy()
    
    # Define NYC boroughs and neighborhoods
    nyc_locations = {
        'manhattan': 'Manhattan',
        'brooklyn': 'Brooklyn',
        'queens': 'Queens',
        'bronx': 'Bronx',
        'staten island': 'Staten Island',
        'harlem': 'Manhattan',
        'times square': 'Manhattan',
        'central park': 'Manhattan',
        'greenwich village': 'Manhattan',
        'soho': 'Manhattan',
        'midtown': 'Manhattan',
        'upper east side': 'Manhattan',
        'upper west side': 'Manhattan',
        'lower east side': 'Manhattan',
        'downtown': 'Manhattan',
        'williamsburg': 'Brooklyn',
        'bushwick': 'Brooklyn',
        'flatbush': 'Brooklyn',
        'brownsville': 'Brooklyn',
        'park slope': 'Brooklyn',
        'astoria': 'Queens',
        'flushing': 'Queens',
        'jamaica': 'Queens',
        'south bronx': 'Bronx',
        'fordham': 'Bronx',
        'riverdale': 'Bronx',
        'st. george': 'Staten Island',
        'new springville': 'Staten Island'
    }
    
    # Function to extract locations using dictionary matching
    def get_locations(tweet):
        if not isinstance(tweet, str):
            return []
        
        tweet_lower = tweet.lower()
        found_locations = []
        
        for loc in nyc_locations.keys():
            if loc in tweet_lower:
                found_locations.append(nyc_locations[loc])
        
        # If no location found, mark as Unknown
        if not found_locations:
            return ['Unknown']
            
        return list(set(found_locations))  # Remove duplicates
    
    # Apply the location extraction
    df_with_locations['extracted_locations'] = df_with_locations['Tweets'].apply(get_locations)
    
    # Create a column with the primary location (first one found)
    df_with_locations['primary_location'] = df_with_locations['extracted_locations'].apply(
        lambda x: x[0] if len(x) > 0 else 'Unknown')
    
    # Count the number of locations found
    location_counts = df_with_locations['primary_location'].value_counts()
    print(f"Locations extracted: {len(location_counts)} unique locations found")
    
    return df_with_locations

# Step 4: Analyze crime categories and severity
def classify_crime_severity(df):
    """
    Classify tweets by crime type and severity
    """
    if 'Tweets' not in df.columns:
        print("Column 'Tweets' not found in the dataset")
        return df
    
    # Create a copy
    df_classified = df.copy()
    
    # Define crime types and associated keywords
    crime_categories = {
        'violent_crime': ['murder', 'homicide', 'shooting', 'shot', 'killed', 'stabbing', 'assault', 'attack', 'weapon', 'gun', 'knife', 'armed', 'robbery', 'violent'],
        'property_crime': ['burglary', 'theft', 'stolen', 'robbery', 'break-in', 'larceny', 'shoplifting', 'vandalism', 'property'],
        'drug_crime': ['drug', 'narcotics', 'cocaine', 'heroin', 'marijuana', 'pills', 'controlled substance'],
        'fraud': ['fraud', 'scam', 'identity theft', 'counterfeit', 'scheme'],
        'traffic_incident': ['crash', 'collision', 'traffic', 'vehicle', 'driving', 'drunk driving', 'dui'],
        'public_disturbance': ['noise', 'disturbance', 'fight', 'disorderly', 'trespassing', 'harassment'],
        'other': ['suspicious', 'activity', 'incident', 'investigation', 'arrest']
    }
    
    # Define severity levels for each category
    severity_levels = {
        'violent_crime': 'high',
        'property_crime': 'moderate',
        'drug_crime': 'moderate',
        'fraud': 'low',
        'traffic_incident': 'low',
        'public_disturbance': 'low',
        'other': 'unknown'
    }
    
    # Function to classify tweet by crime type
    def classify_crime(tweet):
        if not isinstance(tweet, str):
            return 'unknown'
        
        tweet_lower = tweet.lower()
        for category, keywords in crime_categories.items():
            for keyword in keywords:
                if keyword in tweet_lower:
                    return category
        
        return 'other'
    
    # Apply classification
    df_classified['crime_category'] = df_classified['Tweets'].apply(classify_crime)
    
    # Map categories to severity levels
    df_classified['security_level'] = df_classified['crime_category'].map(severity_levels)
    
    # Count incidents by category
    category_counts = df_classified['crime_category'].value_counts()
    print("Crime categories distribution:")
    print(category_counts)
    
    return df_classified

# Step 5: Visualize the results
def visualize_results(df):
    """
    Create basic visualizations of the analyzed data
    """
    if df is None or df.empty:
        print("No data to visualize")
        return
    
    # Create a figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot crime categories
    plt.subplot(2, 2, 1)
    df['crime_category'].value_counts().plot(kind='bar')
    plt.title('Crime Categories')
    plt.tight_layout()
    
    # Plot security levels
    plt.subplot(2, 2, 2)
    security_count = df['security_level'].value_counts()
    bars = plt.bar(security_count.index, security_count.values)
    
    # Color based on security level
    colors = {'high': 'red', 'moderate': 'orange', 'low': 'green', 'unknown': 'gray'}
    for i, level in enumerate(security_count.index):
        if level in colors:
            bars[i].set_color(colors[level])
    
    plt.title('Security Level Distribution')
    plt.tight_layout()
    
    # Plot locations
    plt.subplot(2, 2, 3)
    location_counts = df['primary_location'].value_counts().head(10)  # Top 10 locations
    location_counts.plot(kind='bar')
    plt.title('Top 10 Locations Mentioned')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Generate a table of security levels by location
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Create a cross-tabulation
    location_security = pd.crosstab(df['primary_location'], df['security_level'])
    location_security = location_security.sort_values(by=['high', 'moderate', 'low'], 
                                                    ascending=False).head(10)
    
    # Convert to string for display
    location_table = location_security.to_string()
    plt.text(0, 0.5, f"Security Level by Location:\n\n{location_table}", 
             fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.savefig('crime_analysis_results.png')
    print("Visualizations saved as 'crime_analysis_results.png'")
    
    return

# Generate a CSV security report
def generate_security_report(df):
    """
    Generate a CSV report of security levels by location
    """
    # Create a cross-tabulation
    location_security = pd.crosstab(df['primary_location'], df['security_level'])
    
    # Calculate total incidents for each location
    location_security['total'] = location_security.sum(axis=1)
    
    # Sort by high security incidents, then moderate, then low
    location_security = location_security.sort_values(
        by=['high', 'moderate', 'low'], ascending=False)
    
    # Calculate percentage of each security level
    for level in ['high', 'moderate', 'low', 'unknown']:
        if level in location_security.columns:
            location_security[f'{level}_pct'] = (
                location_security[level] / location_security['total'] * 100).round(1)
    
    # Reorder columns for better readability
    cols = []
    for level in ['high', 'moderate', 'low', 'unknown']:
        if level in location_security.columns:
            cols.extend([level, f'{level}_pct'])
    cols.append('total')
    
    # Ensure all columns exist before filtering
    existing_cols = [col for col in cols if col in location_security.columns]
    location_security = location_security[existing_cols]
    
    # Save to CSV
    location_security.to_csv('security_report_by_location.csv')
    print("Security report saved as 'security_report_by_location.csv'")
    
    return location_security

# Main function to run the analysis pipeline
def main(file_path):
    """
    Run the complete analysis pipeline
    """
    print("Starting Twitter security analysis...")
    
    # Step 1: Load the data
    df = load_data(file_path)
    if df is None:
        return
    
    # Display basic info about the dataset
    print("\nBasic dataset information:")
    print(df.info())
    print("\nSample tweets:")
    print(df['Tweets'].head())
    
    # Step 2: Preprocess the tweets
    print("\nPreprocessing tweets...")
    df_processed = preprocess_tweets(df)
    
    # Step 3: Extract locations
    print("\nExtracting locations...")
    df_with_locations = extract_locations(df_processed)
    
    # Step 4: Classify tweets by crime severity
    print("\nClassifying crime severity...")
    df_classified = classify_crime_severity(df_with_locations)
    
    # Step 5: Visualize results
    print("\nCreating visualizations...")
    visualize_results(df_classified)
    
    # Step 6: Generate security report
    print("\nGenerating security report...")
    generate_security_report(df_classified)
    
    # Save the processed data
    df_classified.to_csv('processed_security_data.csv', index=False)
    print("\nAnalysis complete! Processed data saved to 'processed_security_data.csv'")
    
    return df_classified

# Run the analysis if executed as script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Twitter security data')
    parser.add_argument('--input', type=str, default='nyc_crime_tweets.csv', 
                        help='Path to input CSV file (default: nyc_crime_tweets.csv)')
    
    args = parser.parse_args()
    
    # Run the analysis
    main(args.input)