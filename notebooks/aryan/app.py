import streamlit as st
import requests
import json
import os
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

AIRFLOW_BASE_URL = "http://localhost:8080/api/v1"  # Change if running on a different host
AIRFLOW_DAG_ID = "youtube_comments_etl_pipeline"
AIRFLOW_USERNAME = "airflow"
AIRFLOW_PASSWORD = "airflow"
global dag_run_id

HF_TOKEN = os.getenv("HGF_API_TOKEN")
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)

def set_airflow_variable(video_id):
    url = f"{AIRFLOW_BASE_URL}/variables/yt_video_id"
    headers = {"Content-Type": "application/json"}
    auth = (AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
    
    data = {"key": "yt_video_id", "value": video_id}
    response = requests.patch(url, auth=auth, headers=headers, data=json.dumps(data))
    
    return response.status_code == 200

def trigger_dag():
    """Trigger the Airflow DAG"""
    url = f"{AIRFLOW_BASE_URL}/dags/{AIRFLOW_DAG_ID}/dagRuns"
    headers = {"Content-Type": "application/json"}
    auth = (AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
    
    data = {"conf": {}}  # Empty config since video ID is set via Variable
    response = requests.post(url, auth=auth, headers=headers, data=json.dumps(data))
    dag_run_id = response.json()['dag_run_id']
    
    return (response.status_code == 200, dag_run_id)

def determine_dag_status(dag_run_id):
    url = f"{AIRFLOW_BASE_URL}/dags/{AIRFLOW_DAG_ID}/dagRuns/{dag_run_id}"
    status_response = requests.get(url, auth=("airflow", "airflow"))
    dag_status = status_response.json()['state']
    return dag_status

def extract_data_from_db():
    conn = psycopg2.connect(dbname="postgres", user="airflow", password="airflow", host="localhost", port="5432")
    df = pd.read_sql_query("SELECT * FROM comments", conn)
    return df

def run_inference_model():
    comments_df = extract_data_from_db()
    sentiments = []
    for text in comments_df["comment"]:
        result = client.text_classification(text)
        sentiments.append(result[0]["label"])
    comments_df["sentiment"] = sentiments
    return comments_df
# Streamlit UI
st.title("YouTube Comments Sentiment Analysis")
video_id = st.text_input("Enter YouTube Video ID", "")

if st.button("Run ETL Pipeline"):
    if not video_id:
        st.warning("Please enter a YouTube Video ID.")

    if not set_airflow_variable(video_id):
        st.error("Failed to set Airflow variable. Check API permissions.")
            
    (pipeline_status, dag_run_id) = trigger_dag()
    if not pipeline_status:
        st.error("Failed to trigger DAG. Check Airflow API logs.")

    time.sleep(30)
    dag_status = determine_dag_status(dag_run_id)

    if dag_status == "success":
        st.success(f"ETL pipeline completed successfully for Video ID: {video_id}")
        df = run_inference_model()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📺 Video Preview")
            st.video(f"https://www.youtube.com/embed/{video_id}")

        with col2:
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df["sentiment"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis("equal")
            st.pyplot(fig)
    else:
        st.error(f"ETL pipeline failed for Video ID: {video_id}")
    

    
    