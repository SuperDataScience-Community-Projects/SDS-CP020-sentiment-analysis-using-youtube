from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from googleapiclient.discovery import build
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('youtube_api')
video_url = 'https://www.youtube.com/watch?v=0UEhMDAlX1Y'

# Function to get comments using YouTube Data API
def get_comments(video_url, api_key):
    video_id = video_url.split('v=')[1]
    youtube = build('youtube', 'v3', developerKey=api_key)

    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=100
    )
    response = request.execute()

    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append({
            'author': comment['authorDisplayName'],
            'comment': comment['textOriginal'],
            'date': comment['publishedAt']
        })

    return comments

# Function to analyze comments with Hugging Face InferenceClient
def analyze_comments(comments):
    token = os.getenv('hugging_face_token')
    model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    client = InferenceClient(model=model, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model)
    max_length = 512  # Maximum token length for the model

    for comment in comments:
        # Truncate the comment if it exceeds the maximum token length
        tokens = tokenizer.tokenize(comment['comment'])
        if len(tokens) > max_length:
            truncated_comment = tokenizer.convert_tokens_to_string(tokens[:max_length])
        else:
            truncated_comment = comment['comment']

        sentiment = client.text_classification(truncated_comment)[0]
        comment['sentiment'] = sentiment

    return comments

def make_output_file(video_url, api_key):
    comments = get_comments(video_url, api_key)
    analyzed_comments = analyze_comments(comments=comments)
    
    # Save the results to a file
    with open('sentiment_analysis.json', 'w') as f:
        json.dump(analyzed_comments, f)

default_args = {
    'owner': 'sekine',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 7),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(days=1),
}

with DAG('youtube_sentiment_analysis', default_args=default_args, schedule_interval=timedelta(days=1)) as dag:
    get_comments_task = PythonOperator(
        task_id='get_comments',
        python_callable=get_comments,
        op_args=[video_url, api_key]
    )

    analyze_comments_task = PythonOperator(
        task_id='analyze_comments',
        python_callable=analyze_comments,
        op_args=['{{ task_instance.xcom_pull(task_ids="get_comments") }}']
    )

    make_output_file_task = PythonOperator(
        task_id='make_output_file',
        python_callable=make_output_file,
        op_args=[video_url, api_key]
    )

    get_comments_task >> analyze_comments_task >> make_output_file_task
