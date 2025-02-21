from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from airflow.providers.postgres.hooks.postgres import PostgresHook
import googleapiclient.discovery
import os
import pandas as pd
import re
import string
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MAX_WORDS = 50
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1)
}

with DAG(dag_id='yt_comments_etl_test',
         default_args=default_args,
         schedule_interval=None,  # Run manually
         catchup=False) as dag:

    @task()
    def extract_video_metadata(**kwargs):
        """Extract metadata of a YouTube video."""
        video_id = kwargs['dag_run'].conf.get('video_id', 'SIm2W9TtzR0') 
        print("video_id:",video_id)
        youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=os.getenv("ytb_api")
        )
        
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        )
        response = request.execute()

        if "items" in response and len(response["items"]) > 0:
            video = response["items"][0]
            metadata = {
                "video_id": video_id,
                "title": video["snippet"]["title"],
                "description": video["snippet"]["description"][:500],
                "channel_title": video["snippet"]["channelTitle"],
                "published_at": video["snippet"]["publishedAt"],
                "view_count": video["statistics"].get("viewCount", 0),
                "like_count": video["statistics"].get("likeCount", 0),
                "comment_count": video["statistics"].get("commentCount", 0),
                "duration": video["contentDetails"]["duration"]
            }
        else:
            raise ValueError("No metadata found for the given video ID.")

        return metadata

    @task()
    def extract_youtube_comments(**kwargs):
        """Extract YouTube comments using the YouTube API."""
        video_id = kwargs['dag_run'].conf.get('video_id', 'SIm2W9TtzR0')
        
        youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=os.getenv("ytb_api"))
        
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=30
        )
        response = request.execute()
        return response

    @task()
    def transform_youtube_comments(youtube_data):
        """Transform extracted YouTube comments by cleaning text."""
        def remove_URL(text):
            return re.sub(r'https?://\S+|www\.\S+', '', text)

        def remove_punct(text):
            return text.translate(str.maketrans('', '', string.punctuation))
        
        def truncate_and_filter(text, max_words=MAX_WORDS):
            words = text.split()
            if len(words) > max_words:
                return None  # Drop comments exceeding max word count
            return " ".join(words)  # Keep comments that fit the limit
        
        comments = []
        for item in youtube_data['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                comment['textOriginal']
            ])

        transformed_data = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
        transformed_data['text'] = transformed_data['text'].apply(remove_URL).apply(remove_punct).apply(truncate_and_filter)
        transformed_data = transformed_data.dropna(subset=['text'])
        transformed_data = transformed_data[transformed_data['text'].str.strip() != ""]
        
        return transformed_data.to_json()

    @task()
    def load_video_metadata(video_metadata):
        """Drop & Create VideoMetadata table, then load metadata."""
        pg_hook = PostgresHook(postgres_conn_id='youtube_connection')
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        # Drop table if exists
        cursor.execute("DROP TABLE IF EXISTS VideoMetadata CASCADE;")

        # Create table
        cursor.execute("""
            CREATE TABLE VideoMetadata (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                channel_title TEXT,
                published_at TIMESTAMP,
                view_count INT,
                like_count INT,
                comment_count INT,
                duration TEXT
            );
        """)

        # Insert data
        insert_query = """
        INSERT INTO VideoMetadata (video_id, title, description, channel_title, published_at, view_count, like_count, comment_count, duration)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """

        cursor.execute(insert_query, tuple(video_metadata.values()))
        conn.commit()
        cursor.close()
        conn.close()

        return "Video metadata loaded successfully!"

    @task()
    def load_youtube_comments(transformed_data, **kwargs):
        """Drop & Create Comments table, then load comments."""
        pg_hook = PostgresHook(postgres_conn_id='youtube_connection')
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        # Drop table if exists
        cursor.execute("DROP TABLE IF EXISTS Comments CASCADE;")

        # Create table
        cursor.execute("""
            CREATE TABLE Comments (
                id SERIAL PRIMARY KEY,
                video_id TEXT REFERENCES VideoMetadata(video_id) ON DELETE CASCADE,
                author TEXT,
                published_at TIMESTAMP,
                updated_at TIMESTAMP,
                like_count INT,
                text TEXT
            );
        """)

        # Load data
        transformed_df = pd.read_json(transformed_data)
        video_id = kwargs['dag_run'].conf.get('video_id', 'SIm2W9TtzR0')

        insert_query = """
        INSERT INTO Comments (video_id, author, published_at, updated_at, like_count, text)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        
        records = [(video_id, *tuple(row)) for row in transformed_df.itertuples(index=False, name=None)]
        cursor.executemany(insert_query, records)

        conn.commit()
        cursor.close()
        conn.close()

        return "YouTube comments loaded successfully!"

    ## DAG Workflow
    video_metadata = extract_video_metadata()
    youtube_comments = extract_youtube_comments()
    transformed_comments = transform_youtube_comments(youtube_comments)

    load_video_metadata(video_metadata)
    load_youtube_comments(transformed_comments)


# from airflow import DAG
# from airflow.decorators import task
# from airflow.utils.dates import days_ago
# from airflow.providers.postgres.hooks.postgres import PostgresHook
# import googleapiclient.discovery
# import os
# import pandas as pd
# import re
# import string
# from dotenv import load_dotenv

# load_dotenv()

# default_args = {
#     'owner': 'airflow',
#     'start_date': days_ago(1)
# }

# with DAG(dag_id='yt_comments_etl_test',
#          default_args=default_args,
#          schedule_interval=None,  # Run only when triggered manually
#          catchup=False) as dag:

#     @task()
#     def extract_youtube_data(**kwargs):
#         """Extract YouTube comments using YouTube API."""
#         video_id = kwargs['dag_run'].conf.get('video_id', 'SIm2W9TtzR0')  # Default ID if none provided
        
#         youtube = googleapiclient.discovery.build(
#             "youtube", "v3", developerKey=os.getenv("ytb_api"))
        
#         request = youtube.commentThreads().list(
#             part="snippet",
#             videoId=video_id,
#             maxResults=20
#         )
#         response = request.execute()
#         return response

#     @task()
#     def transform_youtube_data(youtube_data):
#         # from deep_translator import GoogleTranslator
#         # from langdetect import detect
#         """Transform the extracted YouTube data by cleaning comments."""
#         def remove_URL(text):
#             return re.sub(r'https?://\S+|www\.\S+', '', text)

#         def remove_emoji(text):
#             emoji_pattern = re.compile("["
#                                        u"\U0001F600-\U0001F64F"
#                                        u"\U0001F300-\U0001F5FF"
#                                        u"\U0001F680-\U0001F6FF"
#                                        u"\U0001F1E0-\U0001F1FF"
#                                        u"\U00002702-\U000027B0"
#                                        u"\U000024C2-\U0001F251"
#                                        "]+", flags=re.UNICODE)
#             return emoji_pattern.sub(r'', text)

#         def remove_punct(text):
#             return text.translate(str.maketrans('', '', string.punctuation))
        
#         # def translate_to_english(text):
#         #     try:
#         #         detected_lang = detect(text)
#         #         if detected_lang != "en":  # If not English, translate to English
#         #             return GoogleTranslator(source='auto', target='en').translate(text)
#         #         return text  # If already in English, return as-is
#         #     except Exception as e:
#         #         return text  # If language detection fails, return original text
            
#         comments = []
#         for item in youtube_data['items']:
#             comment = item['snippet']['topLevelComment']['snippet']
#             comments.append([
#                 comment['authorDisplayName'],
#                 comment['publishedAt'],
#                 comment['updatedAt'],
#                 comment['likeCount'],
#                 comment['textOriginal']
#             ])

#         transformed_data = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
#         transformed_data['text'] = transformed_data['text'].apply(remove_URL).apply(remove_punct)
#         # Translate to English if needed
#         # transformed_data['text'] = transformed_data['text'].apply(translate_to_english)
#         # transformed_data['text'] = transformed_data['text'].apply(remove_URL).apply(remove_emoji).apply(remove_punct)
        
#         return transformed_data.to_json()

#     @task()
#     def load_youtube_data(transformed_data):
#         """Load transformed data into PostgreSQL."""
#         transformed_df = pd.read_json(transformed_data)
        
#         pg_hook = PostgresHook(postgres_conn_id='youtube_connection')
#         conn = pg_hook.get_conn()
#         cursor = conn.cursor()
        
#         cursor.execute("DROP TABLE IF EXISTS Comments CASCADE;")
#         cursor.execute("""
#             CREATE TABLE Comments (
#                 author TEXT,
#                 published_at TIMESTAMP,
#                 updated_at TIMESTAMP,
#                 like_count INT,
#                 text TEXT
#             );
#         """)

#         insert_query = """
#         INSERT INTO Comments (author, published_at, updated_at, like_count, text)
#         VALUES (%s, %s, %s, %s, %s)
#         """
#         records = [tuple(row) for row in transformed_df.itertuples(index=False, name=None)]
#         cursor.executemany(insert_query, records)

#         conn.commit()
#         cursor.close()
#         conn.close()

#         return "Data loaded successfully!"

#     ## DAG Workflow
#     youtube_data = extract_youtube_data()
#     transformed_data = transform_youtube_data(youtube_data)
#     load_youtube_data(transformed_data)

