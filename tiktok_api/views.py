import json
import requests
import openai
import re
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from apify_client import ApifyClient

@api_view(['GET'])
def get_profile_videos(request):
    try:
        username = request.GET.get('username', '')
        if not username:
            return Response(
                {'error': 'Username parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        print(f"Fetching videos for profile: {username}")
        
        # Initialize the ApifyClient
        client = ApifyClient(settings.APIFY_API_TOKEN)

        # Prepare the actor input
        run_input = {
            "profiles": [username],
            "profileScrapeSections": ["videos"],
            "profileSorting": "latest",
            "resultsPerPage": 100,
            "excludePinnedPosts": False,
            "shouldDownloadVideos": False,
            "shouldDownloadCovers": False,
            "shouldDownloadSubtitles": False,
            "shouldDownloadSlideshowImages": False,
            "shouldDownloadAvatars": False
        }

        print(f"Running Apify actor with input: {run_input}")
        
        try:
            # Run the actor and wait for it to finish
            run = client.actor("0FXVyOXXEmdGcV88a").call(run_input=run_input)
            
            # Fetch actor results from the run's dataset
            print("Fetching results from dataset...")
            items = client.dataset(run["defaultDatasetId"]).list_items().items
            print(f"Retrieved {len(items)} items from dataset")
            print("Sample raw item:", json.dumps(items[0] if items else {}, indent=2))
            
            if not items:
                print(f"No videos found for profile {username}")
                return Response(
                    {'error': f'No videos found for profile {username}'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            videos = []
            for item in items:
                try:
                    print(f"Processing item with raw video URL: {item.get('webVideoUrl', '')} or {item.get('video', {}).get('playUrl', '')}")
                    print(f"Raw item: {json.dumps(item, indent=2)}")
                    print(f"Raw diggCount: {item.get('diggCount')}")
                    print(f"Raw commentCount: {item.get('commentCount')}")
                    print(f"Raw shareCount: {item.get('shareCount')}")
                    print(f"Raw playCount: {item.get('playCount')}")
                    digg_count = int(item.get('videoMeta', {}).get('diggCount', 0) or item.get('diggCount', 0))
                    play_count = int(item.get('videoMeta', {}).get('playCount', 0) or item.get('playCount', 0))
                    conversion = round(float(digg_count * 100.0 / max(play_count, 1)), 2)
                    
                    video_data = {
                        'id': str(item.get('id', '')),
                        'description': item.get('text', ''),
                        'thumbnail': item.get('thumbnail', ''),
                        'createTime': item.get('createTime', 0),
                        'stats': {
                            'diggCount': digg_count,
                            'commentCount': int(item.get('videoMeta', {}).get('commentCount', 0) or item.get('commentCount', 0)),
                            'shareCount': int(item.get('videoMeta', {}).get('shareCount', 0) or item.get('shareCount', 0)),
                            'playCount': play_count,
                            'followerConversion': conversion
                        },
                        'author': {
                            'name': item.get('authorMeta', {}).get('name', ''),
                            'nickname': item.get('authorMeta', {}).get('nickName', ''),
                            'avatar': item.get('authorMeta', {}).get('avatar', '')
                        },
                        'videoUrl': item.get('webVideoUrl', '') or item.get('video', {}).get('playUrl', ''),
                        'duration': item.get('videoMeta', {}).get('duration', 0),
                        'musicMeta': {
                            'musicName': item.get('musicMeta', {}).get('musicName', ''),
                            'musicAuthor': item.get('musicMeta', {}).get('musicAuthor', ''),
                            'musicOriginal': item.get('musicMeta', {}).get('musicOriginal', False)
                        }
                    }
                    
                    if all(k in video_data for k in ['id', 'description', 'thumbnail', 'createTime', 'stats', 'videoUrl', 'musicMeta']):
                        videos.append(video_data)
                    else:
                        print(f"Warning: Skipping video due to missing required fields. Found fields: {list(video_data.keys())}")
                        
                except Exception as e:
                    print(f"Error processing video data: {str(e)}")
                    continue
            
            if not videos:
                print("Warning: No valid videos could be processed")
                return Response(
                    {'error': f'No valid videos found for profile {username}'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Sort videos by creation time
            videos.sort(key=lambda x: x['createTime'], reverse=True)
            
            print(f"Successfully processed {len(videos)} videos")
            response_data = {
                'username': username,
                'videos': videos,
                'total': len(videos)
            }
            return Response(response_data)
            
        except Exception as e:
            print(f"Error during Apify actor run: {str(e)}")
            return Response(
                {'error': 'Error running Apify actor'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
    except Exception as e:
        print(f"Error in get_profile_videos: {str(e)}")
        return Response(
            {'error': 'Failed to fetch profile videos'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def search_hashtag_videos(request):
    print("search_hashtag_videos endpoint called")
    try:
        hashtag = request.GET.get('hashtag', '').strip('#')
        if not hashtag:
            return Response(
                {'error': 'Hashtag parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        print("=== Debug Info ===")
        print(f"Apify API Token: {settings.APIFY_API_TOKEN}")
        print(f"Searching for hashtag: {hashtag}")
        
        # Make direct API call to Apify
        api_url = "https://api.apify.com/v2/acts/clockworks~tiktok-hashtag-scraper/runs"
        run_input = {
            "hashtags": [hashtag],
            "resultsPerPage": 100  # Get more results to sort by recency and views
        }

        print(f"Making API request to: {api_url}")
        print(f"With input: {json.dumps(run_input, indent=2)}")

        try:
            # Make the API call
            response = requests.post(
                api_url,
                params={"token": settings.APIFY_API_TOKEN},
                json=run_input
            )
            
            if not response.ok:
                print(f"API request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return Response(
                    {'error': 'Failed to fetch hashtag data from TikTok API'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            run_data = response.json()
            dataset_id = run_data.get('data', {}).get('defaultDatasetId')
            
            if not dataset_id:
                print("No dataset ID in response")
                return Response(
                    {'error': 'Invalid API response'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Fetch the results from the dataset
            dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
            print(f"Fetching results from dataset: {dataset_url}")
            
            dataset_response = requests.get(
                dataset_url,
                params={"token": settings.APIFY_API_TOKEN}
            )
            
            if not dataset_response.ok:
                print(f"Dataset request failed with status {dataset_response.status_code}")
                print(f"Response: {dataset_response.text}")
                return Response(
                    {'error': 'Failed to fetch hashtag data'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            items = dataset_response.json()
            print(f"Retrieved {len(items)} items from dataset")
            
            if not items:
                print("Error: No videos found for hashtag")
                return Response(
                    {'error': f'No videos found for hashtag #{hashtag}'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            videos = []
            for item in items:
                try:
                    create_time = item.get('createTime', 0)
                    print(f"Raw item: {json.dumps(item, indent=2)}")
                    # Extract stats with fallback paths
                    stats = item.get('stats', {}) or item.get('videoMeta', {})
                    author = item.get('author', {}) or item.get('authorMeta', {})
                    
                    video_data = {
                        'id': str(item.get('id', '')),
                        'description': item.get('text', '') or item.get('desc', ''),
                        'thumbnail': (
                            item.get('thumbnail', '') or 
                            item.get('coverUrl', '') or 
                            item.get('video', {}).get('cover', '') or
                            item.get('video', {}).get('originCover', '')
                        ),
                        'createTime': create_time,
                        'stats': {
                            'diggCount': int(stats.get('diggCount', 0)),
                            'commentCount': int(stats.get('commentCount', 0)),
                            'shareCount': int(stats.get('shareCount', 0)),
                            'playCount': int(stats.get('playCount', 0))
                        },
                        'author': {
                            'name': author.get('name', '') or author.get('uniqueId', ''),
                            'nickname': author.get('nickname', '') or author.get('nickName', ''),
                            'avatar': author.get('avatar', '') or author.get('avatarLarger', '')
                        },
                        'videoUrl': (
                            item.get('webVideoUrl', '') or 
                            item.get('videoUrl', '') or 
                            item.get('video', {}).get('playUrl', '') or
                            item.get('video', {}).get('downloadAddr', '')
                        )
                    }
                    
                    # Print processed video data for debugging
                    print(f"Processed video data: {json.dumps(video_data, indent=2)}")
                    
                    if all(k in video_data for k in ['id', 'description', 'thumbnail', 'createTime', 'stats', 'videoUrl']):
                        videos.append(video_data)
                    else:
                        print(f"Warning: Skipping video due to missing required fields. Found fields: {list(video_data.keys())}")
                        print(f"Raw item data: {item}")
                        
                except Exception as e:
                    print(f"Error processing video data: {str(e)}")
                    print(f"Problematic video data: {item}")
                    continue
            
            if not videos:
                print("Warning: No valid videos could be processed")
                return Response(
                    {'error': f'No valid videos found for hashtag #{hashtag}'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Sort videos by views and creation time
            videos.sort(key=lambda x: (x['createTime'], x['stats']['playCount']), reverse=True)
            
            # Take top 20 videos
            videos = videos[:20]
                
            print(f"Successfully processed {len(videos)} videos")
            print("Sample video data structure:", videos[0] if videos else "No videos")
            response_data = {
                'hashtag': hashtag,
                'videos': videos,
                'total': len(videos)
            }
            print("Full response data:", response_data)
            return Response(response_data)
            
        except Exception as e:
            print(f"Error during Apify actor run: {str(e)}")
            return Response(
                {'error': 'Error running Apify actor'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
    except Exception as e:
        print(f"Error in search_trending_videos: {str(e)}")
        return Response(
            {'error': 'Failed to fetch TikTok data'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def get_ai_insights(request):
    videos = request.data.get('videos')
    if not videos:
        return Response({'error': 'Videos data is required'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        if not isinstance(videos, list) or len(videos) == 0:
            return Response(
                {'error': 'No videos provided for analysis'},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Format video data for OpenAI
        video_descriptions = []
        for i, video in enumerate(videos[:10]):  # Limit to 10 videos for analysis
            desc = video.get('description', '').strip()
            if not desc:
                continue
                
            stats = video.get('stats', {})
            video_descriptions.append(
                f"Video {i+1}:\n" +
                f"Description: {desc}\n" +
                f"Stats: {stats.get('diggCount', 0)} likes, " +
                f"{stats.get('commentCount', 0)} comments, " +
                f"{stats.get('playCount', 0)} views\n"
            )
            
        if not video_descriptions:
            return Response(
                {'error': 'No valid video descriptions found for analysis'},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        video_descriptions_text = "\n".join(video_descriptions)

        prompt = f"""Analyze these TikTok videos for #{request.data.get('hashtag', 'unknown')} and provide insights:

{video_descriptions_text}

Please provide:
1. Common patterns in successful videos using this hashtag
2. Content strategy recommendations specific to this hashtag
3. Tips for maximizing engagement with this hashtag's audience
4. Trending elements and themes within this hashtag community
5. Suggestions for related hashtags to combine with
"""

        openai.api_key = settings.OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a TikTok content strategy expert."},
                {"role": "user", "content": prompt}
            ]
        )

        insights = response.choices[0].message.content

        return Response({'insights': insights})

    except Exception as e:
        print(f"Error in get_ai_insights: {str(e)}")
        return Response(
            {'error': 'Failed to generate AI insights'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def analyze_post_timing(request):
    try:
        username = request.GET.get('username', '')
        if not username:
            return Response(
                {'error': 'Username parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        print(f"Analyzing timing for profile: {username}")
        
        # Initialize the ApifyClient
        client = ApifyClient(settings.APIFY_API_TOKEN)

        # Prepare the actor input - using the same scraper as profile videos
        run_input = {
            "profiles": [username],
            "profileScrapeSections": ["videos"],
            "profileSorting": "latest",
            "resultsPerPage": 100,
            "excludePinnedPosts": False,
            "shouldDownloadVideos": False,
            "shouldDownloadCovers": False,
            "shouldDownloadSubtitles": False,
            "shouldDownloadSlideshowImages": False,
            "shouldDownloadAvatars": False
        }

        print(f"Running Apify actor with input: {run_input}")
        
        try:
            # Run the actor and wait for it to finish
            run = client.actor("0FXVyOXXEmdGcV88a").call(run_input=run_input)
            
            # Fetch actor results from the run's dataset
            print("Fetching results from dataset...")
            items = client.dataset(run["defaultDatasetId"]).list_items().items
            print(f"Retrieved {len(items)} items from dataset")
            
            if not items:
                print(f"No videos found for profile {username}")
                return Response(
                    {'error': f'No videos found for profile {username}'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Analyze posting times and engagement
            hourly_engagement = {i: {'count': 0, 'total_engagement': 0} for i in range(24)}
            weekday_engagement = {i: {'count': 0, 'total_engagement': 0} for i in range(7)}

            for item in items:
                try:
                    create_time = item.get('createTime', '')
                    if not create_time:
                        continue

                    # Convert timestamp to datetime
                    from datetime import datetime
                    post_time = datetime.fromtimestamp(create_time)
                    
                    # Calculate engagement using the same stats structure as profile videos
                    engagement = (
                        int(item.get('videoMeta', {}).get('diggCount', 0) or item.get('diggCount', 0)) +
                        int(item.get('videoMeta', {}).get('commentCount', 0) or item.get('commentCount', 0)) +
                        int(item.get('videoMeta', {}).get('shareCount', 0) or item.get('shareCount', 0))
                    )

                    print(f"Processing video posted at {post_time} with engagement {engagement}")

                    # Add to hourly stats
                    hour = post_time.hour
                    hourly_engagement[hour]['count'] += 1
                    hourly_engagement[hour]['total_engagement'] += engagement

                    # Add to weekday stats
                    weekday = post_time.weekday()
                    weekday_engagement[weekday]['count'] += 1
                    weekday_engagement[weekday]['total_engagement'] += engagement

                except Exception as e:
                    print(f"Error processing video timing: {str(e)}")
                    continue

            # Calculate average engagement for each hour and weekday
            best_times = []
            for hour, stats in hourly_engagement.items():
                if stats['count'] > 0:
                    avg_engagement = stats['total_engagement'] / stats['count']
                    # Format hour in 12-hour format with AM/PM
                    hour_12 = hour % 12
                    if hour_12 == 0:
                        hour_12 = 12
                    ampm = 'AM' if hour < 12 else 'PM'
                    label = f"{hour_12}:00 {ampm}"
                    best_times.append({
                        'hour': hour,
                        'engagement': int(avg_engagement),
                        'label': label
                    })

            weekday_stats = []
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for day_num, stats in weekday_engagement.items():
                if stats['count'] > 0:
                    avg_engagement = stats['total_engagement'] / stats['count']
                    weekday_stats.append({
                        'day': days[day_num],
                        'engagement': int(avg_engagement)
                    })

            # Sort best times by engagement
            best_times.sort(key=lambda x: x['engagement'], reverse=True)
            best_times = best_times[:5]  # Keep top 5 hours

            # Get top performing days
            sorted_days = sorted(weekday_stats, key=lambda x: x['engagement'], reverse=True)
            top_days = [day['day'] for day in sorted_days[:2]]

            # Generate insights based on the data
            insights = [
                f"Peak engagement occurs between {best_times[0]['label']} - {best_times[1]['label']}",
                f"{sorted_days[0]['day']} posts receive highest engagement",
                "Posting 3-4 times during peak hours maximizes reach",
                f"Followers are most active on {' and '.join(top_days)}"
            ]

            response_data = {
                'username': username,
                'bestTimes': best_times,
                'weekdayStats': weekday_stats,
                'insights': insights
            }

            print("Analysis complete. Response data:", response_data)
            return Response(response_data)
            
        except Exception as e:
            print(f"Error during Apify actor run: {str(e)}")
            return Response(
                {'error': 'Error running Apify actor'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
    except Exception as e:
        print(f"Error in analyze_post_timing: {str(e)}")
        return Response(
            {'error': 'Failed to analyze post timing'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def analyze_comments(request):
    print("analyze_comments endpoint called")
    try:
        video_url = request.data.get('url')
        if not video_url:
            return Response(
                {'error': 'Video URL is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        print(f"Received video URL: {video_url}")

        try:
            # Make direct API call to Apify
            api_url = "https://api.apify.com/v2/acts/clockworks~tiktok-comments-scraper/runs"
            run_input = {
                "postURLs": [video_url],
                "commentsPerPost": 100
            }

            print(f"Making API request to: {api_url}")
            print(f"With input: {json.dumps(run_input, indent=2)}")

            # Make the API call
            response = requests.post(
                api_url,
                params={"token": settings.APIFY_API_TOKEN},
                json=run_input
            )
            
            if not response.ok:
                print(f"API request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return Response(
                    {'error': 'Failed to fetch comments from TikTok API'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            run_data = response.json()
            dataset_id = run_data.get('data', {}).get('defaultDatasetId')
            
            if not dataset_id:
                print("No dataset ID in response")
                return Response(
                    {'error': 'Invalid API response'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Fetch the results from the dataset
            dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
            print(f"Fetching results from dataset: {dataset_url}")
            
            dataset_response = requests.get(
                dataset_url,
                params={"token": settings.APIFY_API_TOKEN}
            )
            
            if not dataset_response.ok:
                print(f"Dataset request failed with status {dataset_response.status_code}")
                print(f"Response: {dataset_response.text}")
                return Response(
                    {'error': 'Failed to fetch comments data'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            comments = dataset_response.json()
            print(f"Retrieved {len(comments)} comments from dataset")

            if not comments:
                return Response(
                    {'error': 'No comments found for this video'},
                    status=status.HTTP_404_NOT_FOUND
                )
        except Exception as apify_error:
            print(f"Apify error: {str(apify_error)}")
            return Response(
                {'error': 'Failed to fetch comments from TikTok'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Sort comments by likes (diggCount)
        top_comments = sorted(comments, key=lambda x: x.get('diggCount', 0), reverse=True)[:10]

        try:
            # Prepare comments for sentiment analysis
            comments_text = "\n".join([f"Comment: {c.get('text', '')}" for c in comments[:100]])
            
            # Use OpenAI for sentiment analysis and recommendations
            openai.api_key = settings.OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a TikTok content strategy and sentiment analysis expert. Analyze comments to provide both sentiment analysis and actionable recommendations."},
                    {"role": "user", "content": f"""Analyze these TikTok comments and provide:
1. A positive sentiment score (0-1)
2. A negative sentiment score (0-1)
3. Key sentiment patterns and insights
4. Content recommendations based on:
   - What viewers seem to enjoy most
   - Common requests or suggestions
   - Areas for improvement
   - Content ideas that could resonate with this audience
5. Engagement strategy recommendations based on:
   - Common questions or concerns
   - Types of comments that get the most likes
   - Opportunities for community interaction

Comments:
{comments_text}
"""}
                ]
            )

            # Extract sentiment scores from AI response
            ai_analysis = response.choices[0].message.content

            # Parse the AI response to extract scores
            positive_score = 0.5  # default score
            negative_score = 0.5  # default score
            
            try:
                # Look for score patterns in the AI response
                if "positive sentiment score:" in ai_analysis.lower():
                    pos_match = re.search(r"positive sentiment score:?\s*(0\.\d+)", ai_analysis.lower())
                    if pos_match:
                        positive_score = float(pos_match.group(1))
                
                if "negative sentiment score:" in ai_analysis.lower():
                    neg_match = re.search(r"negative sentiment score:?\s*(0\.\d+)", ai_analysis.lower())
                    if neg_match:
                        negative_score = float(neg_match.group(1))
            except Exception as e:
                print(f"Error parsing sentiment scores: {str(e)}")

            # Extract recommendations from AI response
            recommendations = {
                'content': [],
                'engagement': []
            }
            
            try:
                # Look for content recommendations section
                content_match = re.search(r"Content recommendations:(.*?)(?=Engagement strategy recommendations:|$)", ai_analysis, re.DOTALL | re.IGNORECASE)
                if content_match:
                    content_text = content_match.group(1).strip()
                    recommendations['content'] = [rec.strip('- ') for rec in content_text.split('\n') if rec.strip('- ')]

                # Look for engagement recommendations section
                engagement_match = re.search(r"Engagement strategy recommendations:(.*?)(?=$)", ai_analysis, re.DOTALL | re.IGNORECASE)
                if engagement_match:
                    engagement_text = engagement_match.group(1).strip()
                    recommendations['engagement'] = [rec.strip('- ') for rec in engagement_text.split('\n') if rec.strip('- ')]
            except Exception as e:
                print(f"Error parsing recommendations: {str(e)}")

            return Response({
                'positiveScore': positive_score,
                'negativeScore': negative_score,
                'analysis': ai_analysis,
                'topComments': [{
                    'text': comment.get('text', ''),
                    'diggCount': comment.get('diggCount', 0),
                    'createTime': comment.get('createTimeISO', '')
                } for comment in top_comments],
                'recommendations': recommendations
            })
        except Exception as openai_error:
            print(f"OpenAI API error: {str(openai_error)}")
            # Return just the comments data if AI analysis fails
            return Response({
                'topComments': [{
                    'text': comment.get('text', ''),
                    'diggCount': comment.get('diggCount', 0),
                    'createTime': comment.get('createTimeISO', '')
                } for comment in top_comments]
            })

    except Exception as e:
        print(f"Error in analyze_comments: {str(e)}")
        return Response(
            {'error': 'Failed to analyze comments'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
