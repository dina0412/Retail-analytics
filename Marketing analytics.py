import pandas as pd
import pyodbc
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Defining a function to fetch data from the SQL database
def fetch_data_from_sql():
    # Setting up the connection string to connect to the SQL Server
    conn_str = (
        "Driver={SQL Server};"  
        "Server=dina_laptop;"  # Specifying SQL Server instance
        "Database=PortfolioProject_MarketingAnalytics;" 
        "Trusted_Connection=yes;"
    )
    # Establishing the connection to the database
    conn = pyodbc.connect(conn_str)

    # Defining the SQL query to select the required data from the customer_reviews table
    query = "SELECT ReviewID, CustomerID, ProductID, ReviewDate, Rating, ReviewText FROM customer_reviews"

    # Executing the query and storing the result in a DataFrame
    df = pd.read_sql(query, conn)

    conn.close()
    return df


customer_reviews_df = fetch_data_from_sql()

sia = SentimentIntensityAnalyzer()

# Define a function to calculate sentiment scores using VADER
def calculate_sentiment(review):
    sentiment = sia.polarity_scores(review)
    return sentiment['compound']


# Defining a function to categorize sentiment using both the sentiment score and the review rating
def categorize_sentiment(score, rating):
    if score > 0.05:  # Positive sentiment score
        if rating >= 4:
            return 'Positive'
        elif rating == 3:
            return 'Mixed Positive'
        else:
            return 'Mixed Negative'
    elif score < -0.05:  # Negative sentiment score
        if rating <= 2:
            return 'Negative'
        elif rating == 3:
            return 'Mixed Negative'
        else:
            return 'Mixed Positive'
    else:  # Neutral sentiment score
        if rating >= 4:
            return 'Positive'
        elif rating <= 2:
            return 'Negative'
        else:
            return 'Neutral'

# Defining a function to bucket sentiment scores into text ranges
def sentiment_bucket(score):
    if score >= 0.5:
        return '0.5 to 1.0'
    elif 0.0 <= score < 0.5:
        return '0.0 to 0.49'
    elif -0.5 <= score < 0.0:
        return '-0.49 to 0.0'
    else:
        return '-1.0 to -0.5'  

customer_reviews_df['SentimentScore'] = customer_reviews_df['ReviewText'].apply(calculate_sentiment)

customer_reviews_df['SentimentCategory'] = customer_reviews_df.apply(
    lambda row: categorize_sentiment(row['SentimentScore'], row['Rating']), axis=1)

customer_reviews_df['SentimentBucket'] = customer_reviews_df['SentimentScore'].apply(sentiment_bucket)

print(customer_reviews_df.head())

# Saving the DataFrame with sentiment scores, categories, and buckets to a new CSV file
customer_reviews_df.to_csv('fact_customer_reviews_with_sentiment.csv', index=False)
