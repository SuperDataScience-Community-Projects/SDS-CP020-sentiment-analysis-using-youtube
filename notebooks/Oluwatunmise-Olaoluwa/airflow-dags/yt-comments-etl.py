from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from googleapiclient.discovery import build
from dotenv import load_dotenv
from googleapiclient.errors import HttpError
import os

load_dotenv() # take environment variables from .env.

# Add any imports needed for your ETL script here
# import pandas as pd
# import requests
# import sqlalchemy
# etc.

def run_etl_process():
    """
    Your complete ETL script goes here.
    This function will be executed when the DAG runs.
    """
    try:
        print("Starting ETL process...")
        
        # Place your entire ETL script here
        # For example:
        api_key = os.getenv("MY_API_KEY")

        youtube = build('youtube', 'v3', developerKey=api_key)

        request = youtube.search().list(
            part="snippet",
            q="schafer5",
            type="channel",
            maxResults=1  # Assuming there's only one channel with this name
        )
        response = request.execute()
        items = response.get("items", [])
        if items:
            channel_id = items[0]["snippet"]["channelId"]
            # Now use the channel ID to fetch statistics
            request = youtube.channels().list(
                part='statistics',
                id=channel_id
            )
            channel_response = request.execute()
            print(channel_response)
        else:
            print("No channel found")


        def get_comments(youtube, video_id, max_results=100):
            comments = []
            next_page_token = None

            while True:
                try:
                    response = youtube.commentThreads().list(
                        part="snippet,replies",
                        videoId=video_id,
                        maxResults=max_results,
                        pageToken=next_page_token
                    ).execute()

                    for item in response['items']:
                        comment = item['snippet']['topLevelComment']['snippet']
                        comments.append({
                            'text': comment['textDisplay'],
                            'author': comment['authorDisplayName'],
                            'likeCount': comment['likeCount']
                        })

                        # If the comment has replies, fetch them as well
                        if 'replies' in item:
                            for reply_item in item['replies']['comments']:
                                reply = reply_item['snippet']
                                comments.append({
                                    'text': reply['textDisplay'],
                                    'author': reply['authorDisplayName'],
                                    'likeCount': reply['likeCount']
                                })

                    # Check if there are more pages
                    next_page_token = response.get('nextPageToken')
                    if not next_page_token:
                        break

                except HttpError as e:
                    print(f'An HTTP error {e.resp.status} occurred:\n{e.content}')
                    break

            return comments

        # Build the service object
        youtube = build('youtube', 'v3', developerKey=api_key)
        # Video ID from the URL you provided
        video_id = '_o2v4GZ4x64'

        # Fetch comments
        all_comments = get_comments(youtube, video_id)

        # comment_list=[]
        # for comment in all_comments:
        #     comment_list.append(comment['text'])

        comment_list = [comment['text'] for comment in all_comments]

        # save comment list to plain text file
        with open('comments.txt', 'w') as f:
            for comment in comment_list:
                f.write(comment + '\n')
        
        print("ETL process completed successfully")
    except Exception as e:
        print(f"ETL process failed: {str(e)}")
        raise

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1)
}

# Create the DAG
dag = DAG(
    'daily_etl_process',
    default_args=default_args,
    description='Daily ETL process using Python',
    schedule_interval='0 0 * * *',  # Run at midnight every day
    catchup=False
)

# Define the task
etl_task = PythonOperator(
    task_id='run_etl',
    python_callable=run_etl_process,
    dag=dag
)

# If you need to add any environment variables or configurations:
'''
from airflow.models import Variable

# Set up environment variables in Airflow
# API_KEY = Variable.get("API_KEY")
'''