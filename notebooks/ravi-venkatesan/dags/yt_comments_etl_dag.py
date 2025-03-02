from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from airflow.utils.dates import days_ago

import os
import string
import re
import pandas as pd
import googleapiclient.discovery
from dotenv import load_dotenv
load_dotenv()

default_args = {
    'owner':'airflow',
    'start_date': days_ago(1)
}

with DAG(dag_id='yt_comments_etl_pipeline',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dag:
   
    @task()
    def extract_video_metadata(**kwargs):
        """Extract youtube comments from youtube API."""
        video_id = kwargs['dag_run'].conf.get('video_id', 'pZzXEJ3bcw4')
        print('video_id:',video_id)
        youtube = googleapiclient.discovery.build(
            'youtube',
            'v3',
            developerKey=os.getenv('youtube_api_key')
        )
        request = youtube.videos().list(
            part="snippet, statistics, contentDetails",
            id=video_id,
        )
        response=request.execute()

        if "items" in response and len(response["items"]) > 0:
            video = response["items"][0]
            metadata = {
                "video_id": video_id,
                "title": video["snippet"]["title"],
                "description": video["snippet"]["description"][:1000],
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
    def extract_comments():
        """Extract youtube comments from youtube API."""
        input_vars = { 'video_id' : 'pZzXEJ3bcw4',
                'api_service_name' : 'youtube',
                'api_version' : 'v3',
                'DEVELOPER_KEY': os.getenv('youtube_api_key')
                }
        print('os.getenv(youtube_api_key):', os.getenv('youtube_api_key'))
        youtube = googleapiclient.discovery.build(
            input_vars['api_service_name'],
            input_vars['api_version'],
            developerKey=input_vars['DEVELOPER_KEY']
        )
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=input_vars["video_id"],
            maxResults=50
        )
        response=request.execute()
        return response


    @task()
    def transform_comments(youtube_data):
        """Transform the extracted youtube comments"""
        def convert_comments_to_lowercase(text):
            return text.lower()

        def remove_html(text):
            return re.sub('<[^<]+?>', '', str(text))

        def remove_punctuation(text):
            for char in string.punctuation:
                text = text.replace(char, '')
            return text

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
       
        # Apply transformations to the data
        transformed_data = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
        transformed_data['text'] = transformed_data['text'].apply(convert_comments_to_lowercase)
        transformed_data['text'] = transformed_data['text'].apply(remove_html)
        transformed_data['text'] = transformed_data['text'].apply(remove_punctuation)
        print("transformed_data.head():", transformed_data.head())
        return transformed_data.to_dict()  # Convert to dict for XCom serialization

    @task()
    def load_to_postgres(transformed_data_dict):
        """Load transformed data into PostgreSQL."""
        # Convert dict back to DataFrame
        transformed_data = pd.DataFrame.from_dict(transformed_data_dict)
       
        pg_hook = PostgresHook(postgres_conn_id='yt_conn')
        conn = pg_hook.get_conn()
        cursor = conn.cursor()
       
        # Check if table exists and drop if needed
        cursor.execute("SELECT * FROM pg_tables WHERE tablename = 'yt_comments';")
        cursor.execute("DROP TABLE IF EXISTS yt_comments CASCADE;")
       
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS yt_comments (
                author TEXT,
                published_at TIMESTAMP,
                updated_at TIMESTAMP,
                like_count INT,
                text TEXT
            );
            """)
       
        # Insert transformed data into the table
        insert_query = """
        INSERT INTO yt_comments (author, published_at, updated_at, like_count, text)
        VALUES (%s, %s, %s, %s, %s)
        """
       
        # Convert DataFrame rows to tuples for insertion with proper Python types
        records = [(row.author, row.published_at, row.updated_at, int(row.like_count),
                    row.text) for row in transformed_data.itertuples(index=False)]
       
        for record in records:
            cursor.execute(insert_query, record)                
        conn.commit()
        cursor.close()
        conn.close()
        return "Data loaded to Postgres successfully"
       
    @task
    def perform_sentiment_analysis(load_result):
        # Simple placeholder for sentiment analysis
        print("Sentiment analysis complete")
        return "Sentiment analysis completed"
       
    ## Correct TaskFlow API connection syntax:
    comments_data = extract_comments()
    transformed_data = transform_comments(comments_data)
    load_result = load_to_postgres(transformed_data)
    analysis_result = perform_sentiment_analysis(load_result)
