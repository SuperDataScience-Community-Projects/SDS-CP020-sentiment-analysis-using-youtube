import os
import googleapiclient.discovery
import re

from dotenv import load_dotenv
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YTB_API_KEY")
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

default_args={
    'owner':'airflow',
    'start_date':days_ago(1)
}

## DAG
with DAG(dag_id='youtube_comments_etl_pipeline',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dags:
    

    @task
    def retrieve_comments_via_api(video_id):
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

        youtube = googleapiclient.discovery.build(
            API_SERVICE_NAME, API_VERSION, developerKey = YOUTUBE_API_KEY)

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id
        )
        response = request.execute()

        return response

    def clean_comment(comment):
        def remove_repeated_chars(text):
            return re.sub(r'(.)\1+', r'\1\1', text)
        
        comment = comment.lower()
        comment = re.sub(r"http\S+|www\S+|https\S+", "", comment)
        comment = re.sub(r"@\w+", "", comment)
        #comment = remove_emojis(comment)
        comment = re.sub(r"[^\w\s]", "", comment)
        comment = re.sub(r"\d+", "", comment)
        #comment = remove_stopwords(comment)
        comment = remove_repeated_chars(comment)
        
        #tokens = word_tokenize(comment)
        
        return comment
    
    @task
    def clean_comments_data(youtube_response):
        cleaned_youtube_response = [
            {
                "author": item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"],
                "comment": clean_comment(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]),
                "likes": item["snippet"]["topLevelComment"]["snippet"]["likeCount"],
                "rating": item["snippet"]["topLevelComment"]["snippet"].get("viewerRating", "none"),
                "publishedTime": item["snippet"]["topLevelComment"]["snippet"]["publishedAt"],
            }
            for item in youtube_response["items"]
        ]

        return cleaned_youtube_response

    @task
    def store_comments_in_db(cleaned_youtube_response):
        pg_hook = PostgresHook(postgres_conn_id='aryan-postgres-1')
        db_conn = pg_hook.get_conn()
        db_cursor = db_conn.cursor()
        db_cursor.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                author TEXT,
                comment TEXT,
                like_count INT,
                rating INT,
                published_at TIMESTAMP
            );
            """)
        
        insert_query = """
        INSERT INTO comments (author, comment, like_count, rating, published_at)
        VALUES (%s, %s, %s, %s, %s)
        """

        select_query = """
        SELECT comment FROM comments WHERE comment = %s
        """

        new_rows = 0
        for comment in cleaned_youtube_response:
            db_cursor.execute(select_query, (comment["comment"],))
            if not db_cursor.fetchone():  # If the comment does not exist
                db_cursor.execute(insert_query, (
                    comment["author"],
                    comment["comment"],
                    comment["likes"],
                    0 if comment["rating"] == "none" else comment["rating"],
                    comment["publishedTime"]
                ))
                new_rows += 1  
        db_conn.commit()
        db_cursor.close()
        db_conn.close()

    # 1. Retrieve comments via API (Extract)
    youtube_response = retrieve_comments_via_api(video_id='G8ZiyDlbHBQ')

    #2. Clean comments Data (Transform)
    cleaned_youtube_response = clean_comments_data(youtube_response)

    #3. Load data in DB (Load)
    store_comments_in_db(cleaned_youtube_response)