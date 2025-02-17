import os
import googleapiclient.discovery

from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YTB_API_KEY")
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

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

def clean_comments_data(youtube_response):
    cleaned_youtube_response = [
        {
            "author": item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"],
            "comment": item["snippet"]["topLevelComment"]["snippet"]["textDisplay"],
            "likes": item["snippet"]["topLevelComment"]["snippet"]["likeCount"],
            "rating": item["snippet"]["topLevelComment"]["snippet"].get("viewerRating", "none"),
            "publishedTime": item["snippet"]["topLevelComment"]["snippet"]["publishedAt"],
        }
        for item in youtube_response["items"]
    ]

    print(cleaned_youtube_response)

    return cleaned_youtube_response

def connect_to_db():
    pass

def store_comments_in_db(cleaned_youtube_response):
    pass

def main(video_id):
    # 1. Retrieve comments via API (Extract)
    youtube_response = retrieve_comments_via_api(video_id=video_id)

    #2. Clean comments Data (Transform)
    cleaned_youtube_response = clean_comments_data(youtube_response)

    #3. Load data in DB (Load)
    store_comments_in_db(cleaned_youtube_response)

if __name__ == "__main__":
    video_id = 'Y_vQyMljDsE'
    main(video_id)