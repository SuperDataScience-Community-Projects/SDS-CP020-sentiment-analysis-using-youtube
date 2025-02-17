from airflow import DAG
from airflow.models import Variable
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago
import json



# Defining  DAGastro deployment list
with DAG(
      
    dag_id = 'youtube_comment_postgres',
    start_date = days_ago(1),
    description='A DAG to fetch and store YouTube comments',
    schedule_interval = '@daily',
    catchup = False

) as dag:

   # Steps to creating the etl 

# step 1: Create the table if it doesnt exists

 @task
 def create_table():
     
     postgres_hook =  PostgresHook(postgres_conn_id = "my_postgres_connection")
     create_table_query ="""
     CREATE TABLE IF NOT EXISTS youtube_comments (
         id TEXT PRIMARY KEY,
         text_original TEXT,
         author_display_name VARCHAR(100)

    );
"""
     postgres_hook.run(create_table_query)

# step 2: Extract the youtube api comments data

 extract_youtube_comments = SimpleHttpOperator(
        task_id='extract_youtube_comments',
        http_conn_id='youtube_api',  
        endpoint='commentThreads',  
        method='GET',
        data={
            "part": "snippet",
            "videoId": Variable.get("youtube_video_id", default_var="DEFAULT_VIDEO_ID"), 
            "maxResults": 50,
            "key": "{{ conn.youtube_api.extra_dejson.api_key }}"  
        },
        response_filter=lambda response: response.json(),
    )

# step 3:  Trasform the data i.e pick needed the information to save from the metadata

 @task
 def transform_youtube_data(response):
        comments_data = []
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']
            comments_data.append({
                'id': item['id'],
                'text_original': comment.get('textOriginal', ''),
                'author_display_name': comment.get('authorDisplayName', ''),
            })
        return comments_data

# step 4: Load the data into Postgres SQL

 @task
 def load_data_to_postgres(comments_data):
        postgres_hook = PostgresHook(postgres_conn_id='my_postgres_connection')

        insert_query = """
        INSERT INTO youtube_comments (id, text_original, author_display_name)
        VALUES (%s, %s, %s)
        ON CONFLICT (id) DO NOTHING;
        """

        for comment in comments_data:
            postgres_hook.run(insert_query, parameters=(
                comment['id'],
                comment['text_original'],
                comment['author_display_name'],
            ))



# step 5: Define the task dependencies

 create_table() >> extract_youtube_comments
youtube_response = extract_youtube_comments.output # extract
transformed_data = transform_youtube_data(youtube_response) # transform
load_data_to_postgres(transformed_data)# load