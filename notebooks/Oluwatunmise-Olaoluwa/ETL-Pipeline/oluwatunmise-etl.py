from googleapiclient.discovery import build
from dotenv import load_dotenv
from googleapiclient.errors import HttpError
import os

load_dotenv() # take environment variables from .env.

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