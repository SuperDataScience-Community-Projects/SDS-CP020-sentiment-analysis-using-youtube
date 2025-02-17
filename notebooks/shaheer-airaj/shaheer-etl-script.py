import os
import mysql.connector
from googleapiclient.discovery import build
from dotenv import load_dotenv
load_dotenv()

# Load API Key and MySQL credentials from environment variables
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
MYSQL_USERNAME = os.getenv("MYSQL_USERNAME")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not loaded. Please check your .env file.")

if not MYSQL_USERNAME:
    raise ValueError("MYSQL_USERNAME not loaded. Please check your .env file.")

if not MYSQL_PASSWORD:
    raise ValueError("MYSQL_PASSWORD not loaded. Please check your .env file.")

# Database connection
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user=MYSQL_USERNAME,
        password=MYSQL_PASSWORD,
        database="youtube_api_project"
    )

# Fetch YouTube comments
def fetch_youtube_comments(video_id):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    # Fetch top-level comments
    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id
    )
    response = request.execute()

    # Extract required fields
    comments = [
        {
            "author": item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"],
            "comment": item["snippet"]["topLevelComment"]["snippet"]["textDisplay"],
            "likes": item["snippet"]["topLevelComment"]["snippet"]["likeCount"],
            "rating": item["snippet"]["topLevelComment"]["snippet"].get("viewerRating", "none"),
            "publishedTime": item["snippet"]["topLevelComment"]["snippet"]["publishedAt"],
        }
        for item in response["items"]
    ]
    return comments

# Load new comments into the database
def load_comments_to_db(comments):
    connection = connect_to_db()
    cursor = connection.cursor()

    # Ensure the table exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS comments_table (
        id INT AUTO_INCREMENT PRIMARY KEY,
        authorName VARCHAR(255),
        comment MEDIUMTEXT,
        likes INT,
        rating INT,
        publishedTime VARCHAR(255)
    );
    """
    cursor.execute(create_table_query)

    # Check for existing comments and insert new ones
    insert_query = """
    INSERT INTO comments_table (authorName, comment, likes, rating, publishedTime)
    VALUES (%s, %s, %s, %s, %s)
    """
    select_query = """
    SELECT comment FROM comments_table WHERE comment = %s
    """
    
    new_rows = 0
    for comment in comments:
        cursor.execute(select_query, (comment["comment"],))
        if not cursor.fetchone():  # If the comment does not exist
            cursor.execute(insert_query, (
                comment["author"],
                comment["comment"],
                comment["likes"],
                0 if comment["rating"] == "none" else comment["rating"],
                comment["publishedTime"]
            ))
            new_rows += 1

    connection.commit()
    print(f"{new_rows} new comments were inserted.")
    cursor.close()
    connection.close()

# Main ETL function
def etl_youtube_comments(video_id):
    print("Starting ETL process...")
    comments = fetch_youtube_comments(video_id)
    print(f"Fetched {len(comments)} comments from YouTube.")
    load_comments_to_db(comments)
    print("ETL process completed.")

# Run the ETL process
if __name__ == "__main__":
    VIDEO_ID = "-2k1rcRzsLA"
    etl_youtube_comments(VIDEO_ID)
