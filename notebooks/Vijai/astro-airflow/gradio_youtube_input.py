import gradio as gr
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import psycopg2
import requests  # Import requests to call Airflow API
import pandas as pd

# Load environment variables
load_dotenv()

# YouTube API key
API_KEY = os.getenv('YOUTUBE_API_KEY')

# PostgresQL password

DB_PW = os.getenv('DB_passwd')

# PostgreSQL connection details
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'postgres'
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'



# Airflow API details
AIRFLOW_URL = "http://localhost:8080/api/v1/dags/youtube_comment_extractor/dagRuns"
AIRFLOW_USERNAME = "admin"
AIRFLOW_PASSWORD = "admin"

# Ensure the tables exist before inserting data
def create_tables():
    create_input_youtubeid_table = """
    CREATE TABLE IF NOT EXISTS input_youtubeid (
        id SERIAL PRIMARY KEY,
        video_id TEXT UNIQUE NOT NULL,
        video_title TEXT,
        date_added TIMESTAMP DEFAULT NOW()
    );
    """

    create_youtube_comments_table = """
    CREATE TABLE IF NOT EXISTS youtube_comments (
        comment_id TEXT PRIMARY KEY,
        author TEXT,
        comment TEXT,
        cleaned_comment TEXT,
        likes INT,
        published_at TIMESTAMP,
        video_id TEXT REFERENCES input_youtubeid(video_id) ON DELETE CASCADE,
        sentiment TEXT
    );
    """



    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(create_input_youtubeid_table)
            cursor.execute(create_youtube_comments_table)
    conn.close()

# Call this function at startup to ensure tables exist
create_tables()

def fetch_sentiment_data():
    query = """
    SELECT i.video_id, i.video_title, i.date_added,
           COUNT(c.sentiment) FILTER (WHERE c.sentiment = 'positive') AS positive_count,
           COUNT(c.sentiment) FILTER (WHERE c.sentiment = 'negative') AS negative_count
    FROM input_youtubeid i
    LEFT JOIN youtube_comments c ON i.video_id = c.video_id
    GROUP BY i.video_id, i.video_title, i.date_added
    ORDER BY i.date_added DESC;
    """
    
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
    
    if not results:
        return "<p>No data available.</p>"

    # Create HTML table with sentiment meters
    table_html = "<table style='width:100%; border-collapse: collapse;'>"
    table_html += """
    <tr style='background-color: #222; color: white;'>
        <th style='padding: 8px;'>Date Added</th>
        <th style='padding: 8px;'>Video ID</th>
        <th style='padding: 8px;'>Title</th>
        <th style='padding: 8px;'>Positive Sentiment</th>
    </tr>
    """

    from datetime import datetime
    for video_id, title, date_added, pos_count, neg_count in results:
        total = (pos_count or 0) + (neg_count or 0)

        # ✅ Fix: Show "Processing" if no sentiment is available yet
        if total == 0:
            progress_bar_html = "<p style='color: yellow; text-align: center;'>⏳ Processing...</p>"
        else:
            positive_percentage = (pos_count / total) * 100
            progress_bar_html = f"""
            <div style="width: 100%; background-color: #444; border-radius: 5px; padding: 2px;">
                <div style="width: {positive_percentage}%; height: 15px; background-color: #4CAF50; border-radius: 5px;"></div>
            </div>
            <p style="font-size: 12px; text-align: center; margin: 5px 0;">{positive_percentage:.1f}% Positive</p>
            """

        if isinstance(date_added, str):  
            try:
                date_added = datetime.strptime(date_added, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        formatted_date = date_added.strftime("%Y-%m-%d %H:%M:%S") if isinstance(date_added, datetime) else date_added

        table_html += f"""
        <tr style='background-color: #333; color: white;'>
            <td style='padding: 8px; text-align: center;'>{formatted_date}</td>
            <td style='padding: 8px; text-align: center;'>{video_id}</td>
            <td style='padding: 8px;'>{title}</td>
            <td style='padding: 8px; text-align: center;'>{progress_bar_html}</td>
        </tr>
        """

    table_html += "</table>"
    return table_html

# Function to validate YouTube video ID and get title
def validate_youtube_id(video_id):
    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        request = youtube.videos().list(
            part='snippet',
            id=video_id
        )
        response = request.execute()

        if response['items']:
            title = response['items'][0]['snippet']['title']
            return True, title  
        else:
            return False, None  
    except Exception as e:
        return False, str(e)

# Function to insert video ID and title into PostgreSQL
def insert_video_id(video_id, video_title):
    insert_query = """
    INSERT INTO input_youtubeid (video_id, video_title, date_added)
    VALUES (%s, %s, NOW())
    ON CONFLICT (video_id) DO NOTHING;
    """
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(insert_query, (video_id, video_title))
    conn.close()

# Function to trigger Airflow DAG
def trigger_airflow_dag(video_id):
    headers = {"Content-Type": "application/json"}
    data = {"conf": {"video_id": video_id}}  

    response = requests.post(AIRFLOW_URL, auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD), json=data, headers=headers)

    if response.status_code == 200:
        return "✅ Airflow DAG triggered successfully!"
    else:
        return f"❌ Failed to trigger Airflow DAG: {response.text}"

# Function to handle video submission
def submit_video_id(video_id):
    is_valid, video_title = validate_youtube_id(video_id)
    
    if is_valid:
        insert_video_id(video_id, video_title)
        airflow_response = trigger_airflow_dag(video_id)
        new_sentiment_table_html = fetch_sentiment_data()
        return f"✅ Video '{video_id}' added successfully!\n📺 Title: {video_title}\n\n{airflow_response}", new_sentiment_table_html
    else:
        return f"❌ Invalid Video ID: {video_title if video_title else 'Video not found'}", fetch_sentiment_data()

# **Gradio Interface**
with gr.Blocks() as iface:
    gr.Markdown("# 🎬 YouTube Video ID Submission & Sentiment Analysis")
    
    with gr.Row():
        video_input = gr.Textbox(label="Enter YouTube Video ID")
        submit_button = gr.Button("Submit")
        refresh_button = gr.Button("🔄 Refresh Data")

    sentiment_output = gr.Textbox(label="Status", interactive=False)
    sentiment_table_display = gr.HTML(value=fetch_sentiment_data())

    submit_button.click(submit_video_id, inputs=[video_input], outputs=[sentiment_output, sentiment_table_display])
    refresh_button.click(fetch_sentiment_data, outputs=[sentiment_table_display])

iface.launch()