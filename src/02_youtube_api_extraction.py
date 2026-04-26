import os
import json
from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime

# API Key Placeholder
API_KEY = "YOUR_YOUTUBE_API_KEY_HERE"

def run():
    print("\n--- [2/9] YouTube Data v3 API Extraction ---")
    if API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
        print("API Key not configured. Skipping live extraction and deferring to existing youtube_data1.xlsx.")
        return

    print("Connecting to YouTube API...")
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
    # Example logic to search for videos by hashtag (e.g., #India)
    request = youtube.search().list(
        q="#India",
        part="snippet",
        maxResults=50,
        type="video"
    )
    response = request.execute()
    
    extracted_data = []
    for item in response.get('items', []):
        video_id = item['id']['videoId']
        snippet = item['snippet']
        
        # Get video statistics
        stat_request = youtube.videos().list(part="statistics,snippet", id=video_id)
        stat_response = stat_request.execute()
        
        if not stat_response.get('items'):
            continue
            
        stats = stat_response['items'][0]['statistics']
        vid_snippet = stat_response['items'][0]['snippet']
        
        extracted_data.append({
            'video_id': video_id,
            'channel_id': snippet.get('channelId', ''),
            'channel_title': snippet.get('channelTitle', ''),
            'title': snippet.get('title', ''),
            'description': snippet.get('description', ''),
            'tags': ','.join(vid_snippet.get('tags', [])),
            'default_language': vid_snippet.get('defaultLanguage', 'Unknown'),
            'default_audio_language': vid_snippet.get('defaultAudioLanguage', 'Unknown'),
            'category_id': vid_snippet.get('categoryId', 0),
            'published_at': snippet.get('publishedAt', ''),
            'view_count': int(stats.get('viewCount', 0)),
            'like_count': int(stats.get('likeCount', 0)),
            'comment_count': int(stats.get('commentCount', 0)),
            'collected_at': datetime.now().isoformat()
        })
        
    df_extracted = pd.DataFrame(extracted_data)
    print(f"Extracted {len(df_extracted)} records from API.")
    df_extracted.to_csv(os.path.join('..', 'data', 'youtube_extracted.csv'), index=False)

if __name__ == "__main__":
    run()
