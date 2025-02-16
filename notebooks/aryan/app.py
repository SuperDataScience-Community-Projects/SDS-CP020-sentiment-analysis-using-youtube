import streamlit as st
import requests
import json
import os

from huggingface_hub import InferenceClient

AIRFLOW_BASE_URL = "http://localhost:8080/api/v1"  # Change if running on a different host
AIRFLOW_DAG_ID = "youtube_comments_etl_pipeline"
AIRFLOW_USERNAME = "airflow"
AIRFLOW_PASSWORD = "airflow"

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
    
    return response.status_code == 200

# Streamlit UI
st.title("YouTube Comments Sentiment Analysis")
video_id = st.text_input("Enter YouTube Video ID", "")

if st.button("Run ETL Pipeline"):
    if video_id:
        if set_airflow_variable(video_id):
            if trigger_dag():
                st.success(f"ETL pipeline triggered successfully for Video ID: {video_id}")
            else:
                st.error("Failed to trigger DAG. Check Airflow API logs.")
        else:
            st.error("Failed to set Airflow variable. Check API permissions.")
    else:
        st.warning("Please enter a YouTube Video ID.")
