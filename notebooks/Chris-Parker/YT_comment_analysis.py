# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 11:52:14 2025

@author: PC
"""
#  !pip install --upgrade google-api-python-client
# Some initial SQLite setup:
#      connect to a database, create if it doesn't exist
#      create a table for video details, if it doesnt exist
#
# The intention is to query the database for the selected video and if data is already
# recorded and the number of comments is unchanged then simply load the output from the 
# database rather than re-classifying all the comments.
# 
# mport sqlite3
import pandas as pd
from dotenv import load_dotenv
import os

# connection = sqlite3.connect(r'./tempDb')
# cursor = connection.cursor()
# 
# sql = '''create table if not exists video_data(
#           video_id text primary key,
#           video_title text,
#           video_owner text,
#           comment_count integer,
#           likes_count integer,
#           viewed_count integer,
#           positive_count integer,
#           neutral_count integer,
#           negative_count integer
#           )
# '''
# cursor.execute(sql)
#%%
#       Get video id for sentiment analysis of comments
#

import streamlit as st
import matplotlib.pyplot as plt
import sys


st.title("Youtube Video - Sentiment Analysis")

# Description
st.write(
    """This prototype application analyzes the sentiment of comments from a YouTube video. 
    Simply input the video ID to retrieve and analyze comments.
    The analysis classifies comments into three sentiment categories: Positive, Neutral, and Negative.
    """
)

# Input for YouTube Video ID
video_id = st.text_input("Enter YouTube Video ID:", placeholder="e.g., dQw4w9WgXcQ")

if video_id:

    st.write(f"Analyzing comments for video ID: {video_id}...")
    from googleapiclient.discovery import build

    SECRET_KEY = os.getenv('SECRET_KEY')
    api_key = SECRET_KEY

    youtube = build('youtube', 'v3', developerKey=api_key)

    stats_request = youtube.videos().list(
            part="snippet,contentDetails,statistics ",
            id=video_id
    #        textFormat="plainText" 
        )
    stats_response = stats_request.execute()

    df_columns = ['video_id',
        'video_title' ,
        'video_owner' ,
        'comment_count' ,
        'likes_count' ,
        'viewed_count' ]
    #    'positive_count' ,
    #    'neutral_count' ,
    #    'negative_count']  
    for item in stats_response['items']:
        video_title = item['snippet']['title']
        video_owner = item['snippet']['channelTitle']
        comment_count = item['statistics']['commentCount']
        likes_count = item['statistics']['likeCount']
        viewed_count = item['statistics']['viewCount']

    data = [video_id, video_title, video_owner, comment_count, likes_count, viewed_count]
    stats_df = pd.DataFrame({"Metric": df_columns, "Value" : data})


    request2 = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100  )

    comment_response = request2.execute()

    video_comments = []
    for item in comment_response['items']:
        high_level_comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        video_comments.append(high_level_comment)

        


    #%%
    import emoji 

    i = 0
    for comment in video_comments:    
        comment = emoji.demojize(comment)
        video_comments[i] = comment
        i = i + 1


    #%%

    from transformers import pipeline
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    sa_tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased-finetuned-sst-2-english')
    sa_model = DistilBertForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased-finetuned-sst-2-english')

    sentiment_analysis = pipeline('sentiment-analysis', 
                                model=sa_model, 
                                tokenizer=sa_tokenizer,
                                max_length=512, truncation=True)

    results = sentiment_analysis(video_comments)

    
    for outcome in results:
        if outcome['score'] < 0.70:
            outcome['label'] = 'Neutral'

    # Placeholder for retrieving and processing YouTube data
    
    import pandas
    df = pandas.DataFrame(results)
    df2 = df[['label']].copy()
   
    # Count occurrences of each sentiment label
    sentiment_counts = df2["label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    # Define colors for specific labels
    color_mapping = {
        "POSITIVE": "#0cf55a",
        "Neutral": "#f0e68c",
        "NEGATIVE": "#f5100c"
    }
    colors = [color_mapping[label] for label in sentiment_counts["Sentiment"]]

    col1, col2 = st.columns(2)
    video_stats = col1.container(border=True)
    video_stats.subheader("Real video stats here")
    video_stats.table(stats_df)


    # Display Statistics
    sentiment_stats = col2.container(border=True)
    sentiment_stats.subheader("Sentiment Statistics")
    sentiment_stats.dataframe(sentiment_counts)

    # Plotting a Pie Chart
    fig, ax = plt.subplots()
    ax.pie(
        sentiment_counts["Count"], 
        labels=sentiment_counts["Sentiment"], 
        autopct="%1.1f%%", 
        startangle=90, 
        colors=colors
    )
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    sentiment_stats.pyplot(fig)    

    st.write("Analysis complete! The pie chart above shows the sentiment distribution of the comments.")
else:
    st.write("Please enter a YouTube video ID to begin analysis.")