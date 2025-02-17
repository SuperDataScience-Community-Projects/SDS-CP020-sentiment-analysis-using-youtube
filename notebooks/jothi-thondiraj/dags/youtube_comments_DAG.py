# #dag - directed acyclic graph
# #tasks : 1) fetch amazon data (extract) 2) clean data (transform) 3) create and store data in table on postgres (load)
# #operators : Python Operator and PostgresOperator
# #hooks - allows connection to postgres
# #dependencies
from airflow import DAG
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from airflow.utils.dates import days_ago
import string
import requests
import json
import random
import re
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
#
print("os.getenv(ytb_api):1",os.getenv("ytb_api"))
default_args={
    'owner':'airflow',
    'start_date':days_ago(1)
}

## DAG
with DAG(dag_id='youtube_comments_etl_pipeline',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dags:
    
    @task()
    def extract_youtube_data():
        """Extract youtube comments from youtube API."""
        input_params = { "video_id" : "SIm2W9TtzR0", 
                "api_service_name" : "youtube",
                "api_version" : "v3",
                "DEVELOPER_KEY": os.getenv("ytb_api")
                }
        print("os.getenv(ytb_api):2",os.getenv("ytb_api"))
        youtube = googleapiclient.discovery.build(
            input_params["api_service_name"], input_params["api_version"], developerKey=input_params["DEVELOPER_KEY"])
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=input_params["video_id"],
            maxResults=5
        )
        response = request.execute()
        return response       
        
    @task()
    def transform_youtube_data(youtube_data):
        """Transform the extracted youtube data."""
        # def contains_url(text):
        #     url_pattern = re.compile(r'https?://\S+|www\.\S+')
        #     return bool(url_pattern.search(text))
        def remove_URL(text):
            url = re.compile(r'https?://\S+|www\.\S+')
            return url.sub(r'',text)
        def remove_emoji(text):
            emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
            return emoji_pattern.sub(r'', text)
        def remove_punct(text):
            table=str.maketrans('','',string.punctuation)
            return text.translate(table)        
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
        #Check if the text data has URL: Source: ChatGPT
        transformed_data = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
        print("transformed_data.head() before cleaning:",transformed_data.head())
        transformed_data['text'] = transformed_data['text'].apply(lambda x : remove_URL(x))
        transformed_data['text'] = transformed_data['text'].apply(lambda x : remove_emoji(x))
        transformed_data['text'] = transformed_data['text'].apply(lambda x : remove_punct(x))
        # transformed_data['like_count'] = transformed_data['like_count'].astype(int)
        print("transformed_data.head():",transformed_data.head())
        return transformed_data
    
    @task()
    def load_youtube_data(transformed_data):
        """Load transformed data into PostgreSQL."""
        pg_hook = PostgresHook(postgres_conn_id='youtube_connection')
        conn = pg_hook.get_conn()
        cursor = conn.cursor()
        cursor.execute("""
                        SELECT * FROM pg_tables WHERE tablename = 'Comments'; 
                        DROP TABLE IF EXISTS Comments CASCADE;
                       """)
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Comments (
                author TEXT,
                published_at TIMESTAMP,
                updated_at TIMESTAMP,
                like_count INT,
                text TEXT
            );
            """)
        # Insert transformed data into the table
        insert_query = """
        INSERT INTO Comments (author, published_at, updated_at, like_count, text)
        VALUES (%s, %s, %s, %s, %s)
        """
        # Convert DataFrame rows to tuples for insertion
        # records = transformed_data.to_records(index=False)
        # Convert DataFrame rows to tuples with proper Python types
        records = [(row.author, row.published_at, row.updated_at, int(row.like_count), 
                    row.text) for row in transformed_data.itertuples(index=False) ]
        # print("records:",records)
        # print("type(records):",[type(rec) for rec in records[0]])
        for record in records:
            cursor.execute(insert_query, record)                
        conn.commit()
        # extracted_data = pd.read_sql_query("SELECT * FROM users", conn)
        cursor.close()
        conn.close()
        return extracted_data

    @task
    def perform_sentiment_analysis(clean_data):
        st.write(clean_data)
        st.write("Hello")
        
    ## DAG Worflow- ETL Pipeline
    weather_data = extract_youtube_data()
    transformed_data = transform_youtube_data(weather_data)
    clean_data = load_youtube_data(transformed_data)
    perform_sentiment_analysis(clean_data)



