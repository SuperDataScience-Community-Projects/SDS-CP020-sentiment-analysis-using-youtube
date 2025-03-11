import streamlit as st
import sys
import psycopg2
import requests 
import pandas as pd
import time 
import matplotlib.pyplot as plt
import os 
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

#Load env variables
load_dotenv()

# Airflow API URL
AIRFLOW_TRIGGER_URL = "http://localhost:8080/api/v1/dags/yt_comments_etl_pipeline/dagRuns"
AIRFLOW_DAG_STATUS_URL = "http://localhost:8080/api/v1/dags/yt_comments_etl_pipeline/dagRuns/"

#Hugging Face API 
hf_token = os.getenv('hf_token')
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
client = InferenceClient(model=MODEL_NAME, token=hf_token)

# Streamlit UI Title
st.set_page_config(page_title="YouTube Sentiment Analysis", layout="wide")
st.title("🎬 YouTube Comments Sentiment Analysis")

# Pin the title at the top using markdown and CSS
st.markdown(
    """
    <style>
    .title {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: white;
        padding: 10px 0;
        z-index: 100;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    <div class="title">🎬 YouTube Comments Sentiment Analysis</div>
    """,
    unsafe_allow_html=True
)

# Add some spacing to prevent content from being hidden under the fixed title
st.write("\n\n")
# User Input for YouTube Video ID
st.markdown("### 🔎 Enter YouTube Video ID")
video_id = st.text_input("Paste the Video ID below and run the pipeline:", "")


# Run ETL Pipeline
if st.button("🚀 Run ETL Pipeline"):
    if not video_id:
        st.warning("⚠️ Please enter a valid Video ID!")
    else:
        # Trigger the Airflow DAG
        response = requests.post(AIRFLOW_TRIGGER_URL, json={"conf": {"video_id": video_id}}, auth=("airflow", "airflow"))

        if response.status_code == 200:
            dag_run_id = response.json()["dag_run_id"]
            st.success(f"✅ DAG Triggered Successfully! Run ID: `{dag_run_id}`")

            # Polling for DAG completion
            with st.spinner("⏳ Waiting for DAG execution to complete..."):
                while True:
                    status_response = requests.get(AIRFLOW_DAG_STATUS_URL + dag_run_id, auth=("airflow", "airflow"))
                    dag_status = status_response.json().get("state", "")

                    if dag_status in ["success", "failed"]:
                        break
                    time.sleep(5)  # Wait before checking status again
            
            # Check DAG execution status
            if dag_status == "success":
                st.success("🎉 DAG Execution Completed Successfully!")

                # Fetch Results from PostgreSQL
                conn = psycopg2.connect(dbname="airflow", user="airflow", password="airflow", host='localhost', port="5432")

                # Fetch Video Metadata
                st.markdown("### 🎥 YouTube Video Metadata")
                metadata_query = f"""
                SELECT title AS video_title, channel_title, published_at AS video_posted_date, comment_count, view_count, like_count  
                FROM VideoMetadata WHERE video_id = '{video_id}';
                """
                metadata_df = pd.read_sql_query(metadata_query, conn)
                # # st.dataframe(metadata_df, width=800)
                st.dataframe(metadata_df.style.hide(axis="index"), width=800)
                # # Apply bold styling to column headers and hide index
                # styled_df = metadata_df.style.set_table_styles(
                #     [{"selector": "th", "props": [("font-weight", "bold")]}]
                # ).hide(axis="index")

                # # Display the dataframe in Streamlit
                # st.dataframe(styled_df, width=800)
                # Fetch Comments
                df = pd.read_sql_query("SELECT * FROM yt_comments", conn)
                # Display Comments Table with Sentiment
                # st.markdown("### 💬 Comments")
                # st.dataframe(df[["author", "published_at", "like_count", "text"]], height=400)
                # Perform Sentiment Analysis
                st.markdown("### 🤖 Sentiment Analysis on Comments")
                # df["text"] = df["text"].apply(truncate_text)  # Truncate long comments
                # sentiments = [client.text_classification(text)[0]["label"] for text in df["text"]]
                # df["sentiment"] = sentiments
                # df["sentiment"] = df["text"].apply(lambda x: client.text_classification(text=x)[0]["label"])
                sentiments = []
                for text in df["text"]:
                    # print()
                    result = client.text_classification(text)
                    sentiments.append(result[0]["label"])  # Extract sentiment label
                
                df["sentiment"] = sentiments

                # Layout: Display Video & Sentiment Distribution Side-by-Side
                col1, col2 = st.columns(2)

                # Display YouTube Video
                with col1:
                    st.subheader("📺 Video Preview")
                    st.video(f"https://www.youtube.com/embed/{video_id}")

                # Sentiment Distribution Pie Chart
                with col2:
                    st.subheader("📊 Sentiment Distribution")
                    sentiment_counts = df["sentiment"].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.axis("equal")
                    st.pyplot(fig)

                # # Display Comments Table with Sentiment
                # st.markdown("### 💬 Comments with Sentiment")
                # st.dataframe(df[["author", "published_at", "like_count", "text", "sentiment"]], height=400)

                conn.close()
            else:
                st.error("❌ DAG Execution Failed!")
        else:
            st.error("❌ Failed to trigger DAG. Check Airflow API and authentication.")



