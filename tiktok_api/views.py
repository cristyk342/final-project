import logging
import traceback
import json
import requests
import openai
import re
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from apify_client import ApifyClient

# Set up logging
logger = logging.getLogger(__name__)

# ==============================================================================
# ========================== get_profile_videos VIEW =========================
# ==============================================================================

@api_view(['GET'])
def get_profile_videos(request):
    logger.info("get_profile_videos endpoint called")
    try:
        username = request.GET.get('username', '')
        if not username:
            logger.warning("Username parameter is missing")
            return Response(
                {'error': 'Username parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        logger.info(f"Fetching videos for profile: {username}")

        # Initialize the ApifyClient
        client = ApifyClient(settings.APIFY_API_TOKEN) # Use settings

        # Prepare the actor input
        run_input = {
            "profiles": [username],
            "profileScrapeSections": ["videos"],
            "profileSorting": "latest",
            "resultsPerPage": 20, # Limited to 20 videos per request
            "excludePinnedPosts": False,
            "shouldDownloadVideos": False,
            "shouldDownloadCovers": False,
            "shouldDownloadSubtitles": False,
            "shouldDownloadSlideshowImages": False,
            "shouldDownloadAvatars": False
        }

        logger.info(f"Running Apify actor '0FXVyOXXEmdGcV88a' with input: {run_input}")
        run = None
        try:
            # Run the actor and wait for it to finish
            # Assuming '0FXVyOXXEmdGcV88a' is the correct actor ID for profile videos
            run = client.actor("0FXVyOXXEmdGcV88a").call(run_input=run_input)
            logger.info(f"Apify actor run completed. Run details: {run}")

        except Exception as apify_call_error:
            logger.error(f"Error calling Apify actor for profile videos: {str(apify_call_error)}", exc_info=True)
            return Response(
                {'error': 'Failed to execute profile data scraping job.', 'details': str(apify_call_error)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Fetch actor results
        items = []
        dataset_id = run.get("defaultDatasetId") if run else None
        if not dataset_id:
             logger.error("Apify run did not return a defaultDatasetId for profile videos.")
             return Response(
                 {'error': 'Failed to get dataset ID from profile video run.'},
                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
             )

        try:
            dataset = client.dataset(dataset_id)
            # Use iterate_items for potentially large datasets
            for item in dataset.iterate_items():
                items.append(item)
            logger.info(f"Retrieved {len(items)} items from dataset {dataset_id} for profile {username}")
            
            # Debug log the first item's structure
            if items:
                logger.info(f"First item structure: {json.dumps(items[0], indent=2)}")
                logger.info(f"Stats from first item - diggCount: {items[0].get('diggCount')}, playCount: {items[0].get('playCount')}, shareCount: {items[0].get('shareCount')}, commentCount: {items[0].get('commentCount')}")

        except Exception as dataset_error:
            logger.error(f"Error iterating Apify dataset ({dataset_id}) for profile videos: {str(dataset_error)}", exc_info=True)
            return Response(
                {'error': 'Failed to retrieve results from profile video scraping job.', 'details': str(dataset_error)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        if not items:
             logger.warning(f"No videos found for profile {username} in dataset {dataset_id}")
             return Response(
                  {'message': f'No videos found for profile {username}', 'videos': []},
                  status=status.HTTP_200_OK
              )

        # Process items into video data (similar structure to search_hashtag_videos but adapted)
        videos = []
        logger.info(f"Processing {len(items)} raw profile video items...")
        for i, item in enumerate(items):
             try:
                 # Log raw item for debugging
                 logger.info(f"Processing video item {i}: {json.dumps(item)}")
                 
                 # Extract data using exact field names from Apify response
                 author = item.get('authorMeta', {})
                 music = item.get('musicMeta', {})

                 video_data = {
                     'id': str(item.get('id', '')),
                     'description': item.get('text', ''),
                     'thumbnail': item.get('thumbnail', ''),
                     'createTime': item.get('createTime', 0),
                     'duration': item.get('videoMeta', {}).get('duration', 0),
                     'stats': {
                         'diggCount': int(item.get('diggCount', 0)) if isinstance(item.get('diggCount'), (int, str)) and str(item.get('diggCount', '')).isdigit() else 0,
                         'commentCount': int(item.get('commentCount', 0)) if isinstance(item.get('commentCount'), (int, str)) and str(item.get('commentCount', '')).isdigit() else 0,
                         'shareCount': int(item.get('shareCount', 0)) if isinstance(item.get('shareCount'), (int, str)) and str(item.get('shareCount', '')).isdigit() else 0,
                         'playCount': int(item.get('playCount', 0)) if isinstance(item.get('playCount'), (int, str)) and str(item.get('playCount', '')).isdigit() else 0
                     },
                     'author': { # Author might already be top-level in profile results
                         'name': author.get('name', '') or author.get('uniqueId', ''),
                         'nickname': author.get('nickname', '') or author.get('nickName', ''),
                         'avatar': author.get('avatar', '') or author.get('avatarLarger', '')
                     },
                     'videoUrl': item.get('webVideoUrl', '') or item.get('videoUrl', '') or item.get('video', {}).get('playUrl', ''),
                     'musicMeta': {
                            'musicName': music.get('musicName', ''),
                            'musicAuthor': music.get('musicAuthor', ''),
                            'musicOriginal': music.get('musicOriginal', False)
                        }
                     # Add other relevant fields if needed
                 }

                 if video_data.get('id'): # Basic check
                     # Log the processed stats for debugging
                     logger.info(f"Processed stats for video {i}: {json.dumps(video_data['stats'])}")
                     videos.append(video_data)
                 else:
                     logger.warning(f"Skipping profile video item {i} due to missing 'id'. Item Keys: {list(item.keys())}")

             except Exception as processing_error:
                 logger.error(f"Error processing profile video item {i} (ID: {item.get('id', 'N/A')}): {str(processing_error)}", exc_info=True)
                 continue

        if not videos:
              logger.warning(f"No valid videos could be processed for profile {username}")
              return Response(
                  {'message': f'No processable videos found for profile {username}', 'videos': []},
                  status=status.HTTP_200_OK
              )

        # Optional: Sort videos? By creation time?
        videos.sort(key=lambda x: x['createTime'], reverse=True)

        logger.info(f"Successfully processed {len(videos)} videos for profile {username}. Returning video list.")
        response_data = {
             'username': username,
             'videos': videos,
             'total': len(videos)
         }
        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        logger.critical(f"Unexpected error in get_profile_videos: {str(e)}", exc_info=True)
        return Response(
            {'error': 'An unexpected server error occurred while fetching profile videos.', 'details': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# ==============================================================================
# ========================== search_hashtag_videos VIEW ========================
# ==============================================================================

@api_view(['GET'])
def search_hashtag_videos(request):
    print("search_hashtag_videos endpoint called")
    logger.info("search_hashtag_videos endpoint called")
    try:
        hashtag = request.GET.get('hashtag', '').strip('#')
        if not hashtag:
            logger.warning("Hashtag parameter is missing")
            return Response(
                {'error': 'Hashtag parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        print("=== Debug Info ===")
        print(f"Apify API Token Used (First 10 chars): apify_api_...")
        print(f"Searching for hashtag: {hashtag}")
        logger.info(f"Searching for hashtag: {hashtag}")

        # Initialize the ApifyClient
        client = ApifyClient(settings.APIFY_API_TOKEN)

        # Prepare the Actor input
        run_input = {
            "hashtags": [hashtag],
            "resultsPerPage": 10,
            "shouldDownloadVideos": False,
            "shouldDownloadCovers": False,
            "shouldDownloadSubtitles": False,
            "shouldDownloadSlideshowImages": False,
        }

        print(f"Running Apify actor 'f1ZeP0K58iwlqG2pY' with input: {run_input}")
        logger.info(f"Running Apify actor 'f1ZeP0K58iwlqG2pY' with input: {run_input}")

        run = None
        try:
            # Run the Actor and wait for it to finish
            run = client.actor("f1ZeP0K58iwlqG2pY").call(run_input=run_input)
            logger.info(f"Apify actor run completed. Run details: {run}")

        except Exception as apify_call_error:
            print(f"!!! Error calling Apify actor: {str(apify_call_error)}")
            print(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Error calling Apify actor: {str(apify_call_error)}", exc_info=True)
            return Response(
                {'error': 'Failed to execute data scraping job.', 'details': str(apify_call_error)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Fetch actor results from the run's dataset
        print("Fetching results from dataset...")
        logger.info("Fetching results from dataset...")
        items = []
        dataset_id = run.get("defaultDatasetId") if run else None

        if not dataset_id:
            logger.error("Apify run did not return a defaultDatasetId.")
            return Response(
                {'error': 'Failed to get dataset ID from Apify run.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        try:
            dataset = client.dataset(dataset_id)
            for item in dataset.iterate_items():
                items.append(item)
            print(f"Retrieved {len(items)} items from dataset {dataset_id}")
            logger.info(f"Retrieved {len(items)} items from dataset {dataset_id}")

        except Exception as dataset_error:
            print(f"!!! Error iterating Apify dataset ({dataset_id}): {str(dataset_error)}")
            print(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Error iterating Apify dataset ({dataset_id}): {str(dataset_error)}", exc_info=True)
            return Response(
                {'error': 'Failed to retrieve results from data scraping job.', 'details': str(dataset_error)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        if not items:
            print(f"Warning: No videos found for hashtag #{hashtag}")
            logger.warning(f"No videos found for hashtag #{hashtag} in dataset {dataset_id}")
            return Response(
                 {'message': f'No videos found for hashtag #{hashtag}', 'videos': []},
                 status=status.HTTP_200_OK
             )

        videos = []
        print(f"Processing {len(items)} raw items...")
        logger.info(f"Processing {len(items)} raw items...")
        for i, item in enumerate(items):
            try:
                # Log raw item for debugging
                logger.info(f"Processing hashtag video item {i}: {json.dumps(item)}")
                
                # Extract data using exact field names from Apify response
                author = item.get('authorMeta', {})

                video_data = {
                    'id': str(item.get('id', '')),
                    'description': item.get('text', ''),
                    'thumbnail': item.get('thumbnail', ''),
                    'createTime': item.get('createTime', 0),
                    'stats': {
                        'diggCount': int(item.get('diggCount', 0)) if isinstance(item.get('diggCount'), (int, str)) and str(item.get('diggCount', '')).isdigit() else 0,
                        'commentCount': int(item.get('commentCount', 0)) if isinstance(item.get('commentCount'), (int, str)) and str(item.get('commentCount', '')).isdigit() else 0,
                        'shareCount': int(item.get('shareCount', 0)) if isinstance(item.get('shareCount'), (int, str)) and str(item.get('shareCount', '')).isdigit() else 0,
                        'playCount': int(item.get('playCount', 0)) if isinstance(item.get('playCount'), (int, str)) and str(item.get('playCount', '')).isdigit() else 0
                    },
                    'author': {
                        'name': author.get('name', '') or author.get('uniqueId', ''),
                        'nickname': author.get('nickname', '') or author.get('nickName', ''),
                        'avatar': author.get('avatar', '') or author.get('avatarLarger', '')
                    },
                    'videoUrl': item.get('webVideoUrl', '') or item.get('videoUrl', '') or item.get('video', {}).get('playUrl', '') or item.get('video', {}).get('downloadAddr', '')
                }

                if video_data.get('id') and video_data.get('description'):
                     videos.append(video_data)
                else:
                    print(f"Warning: Skipping item {i} due to missing 'id' or 'description'. Raw item keys: {list(item.keys())}")
                    logger.warning(f"Skipping item {i} due to missing 'id' or 'description'. Item ID (if available): {item.get('id')}")

            except Exception as processing_error:
                print(f"!!! Error processing video item {i}: {str(processing_error)}")
                problematic_keys = list(item.keys())
                problematic_id = item.get('id', 'N/A')
                print(f"Problematic video item keys: {problematic_keys}, ID: {problematic_id}")
                print(f"Traceback: {traceback.format_exc()}")
                logger.error(f"Error processing video item {i} (ID: {problematic_id}): {str(processing_error)}", exc_info=True)
                continue

        if not videos:
             print("Warning: No valid videos could be processed from the retrieved items.")
             logger.warning("No valid videos could be processed from the retrieved items.")
             return Response(
                 {'message': f'No processable videos found for hashtag #{hashtag}', 'videos': []},
                 status=status.HTTP_200_OK
             )

        print(f"Successfully processed {len(videos)} videos for hashtag #{hashtag}")
        logger.info(f"Successfully processed {len(videos)} videos for hashtag #{hashtag}. Returning video list.")
        return Response({'videos': videos}, status=status.HTTP_200_OK)

    except Exception as e:
        print(f"!!! Unexpected error in search_hashtag_videos: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        logger.critical(f"Unexpected error in search_hashtag_videos: {str(e)}", exc_info=True)
        return Response(
            {'error': 'An unexpected server error occurred.', 'details': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# ==============================================================================
# ============================= get_ai_insights VIEW ===========================
# ==============================================================================

@api_view(['POST', 'OPTIONS'])
@permission_classes([AllowAny])
def get_ai_insights(request):
    logger.info("\n=== AI Insights Request Received ===")
    logger.info(f"Request method: {request.method}")

    if request.method == 'OPTIONS':
        logger.info("Responding to OPTIONS request")
        return Response(status=status.HTTP_200_OK)

    if request.method == 'POST':
        logger.info("Processing POST request for AI insights")
        try:
            data = json.loads(request.body)
            videos = data.get('videos', [])
            analysis_type = data.get('type', 'general')  # Can be 'like_rate', 'conversion', or 'general'
            logger.info(f"Received {len(videos)} videos for {analysis_type} analysis.")

            if not videos:
                logger.warning("No videos provided in the request body.")
                return Response(
                    {'error': 'No videos provided for analysis'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if not settings.OPENAI_API_KEY:
                logger.error("OpenAI API key is not configured in settings.")
                return Response(
                    {'error': 'OpenAI API key not configured.'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            openai.api_key = settings.OPENAI_API_KEY

            # Prepare the prompt based on analysis type
            if analysis_type == 'like_rate':
                prompt_content = """Analyze the following TikTok video data focusing on like rate performance. Structure your response like this:

1. Like Rate Analysis: Analyze patterns in videos with high like-to-view ratios
2. Viewer Engagement: Identify what makes viewers more likely to like the content
3. Content Optimization: Suggest ways to improve like rate performance
4. Best Practices: Share tactics specifically for increasing like rates
5. Performance Insights: Compare performance against typical TikTok benchmarks

Base your analysis ONLY on the following video data:

"""
            elif analysis_type == 'conversion':
                prompt_content = """Analyze the following TikTok video data focusing on conversion and viewer interaction. Structure your response like this:

1. Conversion Analysis: Analyze patterns in videos with high comment and share rates
2. Viewer Interaction: Identify what drives viewers to comment and share
3. Call-to-Action Strategy: Suggest effective ways to prompt viewer interaction
4. Engagement Tactics: Share specific methods to increase comments and shares
5. Performance Insights: Compare interaction rates against platform benchmarks

Base your analysis ONLY on the following video data:

"""
            else:
                prompt_content = """Analyze the following TikTok video data and provide insights in a numbered list format. Structure your response like this:

1. Common Patterns: Identify recurring themes, formats, or styles in successful videos
2. Content Strategy: Provide specific recommendations for creating content
3. Engagement Tips: Share tactics for maximizing viewer interaction
4. Trending Elements: Point out popular features, sounds, or approaches
5. Related Hashtags: Suggest complementary hashtags to use

Base your analysis ONLY on the following video data:

"""
            for video in videos[:10]: # Limit the number of videos sent to OpenAI to manage cost/token limits
                prompt_content += f"- Description: {video.get('description', 'N/A')}\n"
                prompt_content += f"  Views: {video.get('stats', {}).get('playCount', 'N/A')}\n"
                prompt_content += f"  Likes: {video.get('stats', {}).get('diggCount', 'N/A')}\n"
                prompt_content += f"  Comments: {video.get('stats', {}).get('commentCount', 'N/A')}\n"
                prompt_content += f"  Shares: {video.get('stats', {}).get('shareCount', 'N/A')}\n\n"

            logger.info("Sending request to OpenAI API...")
            # logger.debug(f"OpenAI Prompt: {prompt_content}") # Uncomment for debugging prompt issues

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant analyzing TikTok data."},
                        {"role": "user", "content": prompt_content}
                    ],
                    max_tokens=300, # Adjust token limit as needed
                    temperature=0.5
                )
                logger.info("Received response from OpenAI API.")
                # logger.debug(f"OpenAI Response: {response}") # Uncomment for debugging OpenAI response

                insights = response.choices[0].message['content'].strip()
                logger.info("Successfully extracted insights from OpenAI response.")
                return Response({'insights': insights}, status=status.HTTP_200_OK)

            except openai.error.OpenAIError as openai_error:
                logger.error(f"OpenAI API Error: {str(openai_error)}", exc_info=True)
                return Response(
                    {'error': 'Failed to get insights from AI.', 'details': str(openai_error)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            except Exception as api_call_err:
                 logger.error(f"Unexpected error during OpenAI API call: {str(api_call_err)}", exc_info=True)
                 return Response(
                     {'error': 'An unexpected error occurred calling the AI service.', 'details': str(api_call_err)},
                     status=status.HTTP_500_INTERNAL_SERVER_ERROR
                 )

        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from request body.")
            return Response(
                {'error': 'Invalid JSON format in request body'},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.critical(f"Unexpected error in get_ai_insights POST handler: {str(e)}", exc_info=True)
            return Response(
                {'error': 'An unexpected server error occurred while generating AI insights.', 'details': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    else:
        logger.warning(f"Received unsupported method: {request.method}")
        return Response(
            {'error': 'Method Not Allowed'},
            status=status.HTTP_405_METHOD_NOT_ALLOWED
        )

# ==============================================================================
# =========================== analyze_post_timing VIEW =========================
# ==============================================================================

@api_view(['GET'])
def analyze_post_timing(request):
    pass # TODO: Implement or restore function logic
    # try:
    #     username = request.GET.get('username', '')
    # ... rest of the code remains the same ...

# ==============================================================================
# =========================== analyze_comments VIEW ============================
# ==============================================================================

@api_view(['POST', 'OPTIONS'])
@permission_classes([AllowAny]) # Adjust permissions as needed
def analyze_comments(request):
    # --- ADDED LOG ---
    logger.info("--- analyze_comments VIEW ENTERED ---") 
    logger.info(f"Request method: {request.method}")

    if request.method == 'OPTIONS':
        logger.info("Responding to OPTIONS request for analyze_comments")
        return Response(status=status.HTTP_200_OK)

    if request.method == 'POST':
        logger.info("Processing POST request for analyze_comments")
        try:
            data = json.loads(request.body)
            username = data.get('username', '')
            video_id = data.get('videoId', '') 

            if not video_id or not username:
                logger.warning("Missing 'videoId' or 'username' in analyze_comments request body.")
                return Response(
                    {'error': 'Missing videoId or username parameter in request body'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            logger.info(f"Analyzing comments for video ID: {video_id} (User: {username})")

            # --- Scrape Comments using Apify ---
            client = ApifyClient(settings.APIFY_API_TOKEN)
            # Use the correct actor ID for TikTok comments
            comment_actor_id = "BDec00yAmCm1QbMEI" # clockworks/tiktok-comments-scraper
            # Construct the full URL required by the actor
            full_video_url = f"https://www.tiktok.com/@{username}/video/{video_id}"
            run_input = { 
                "postURLs": [full_video_url], 
                "commentsPerPost": 50 # Limit comments for performance/cost
            }
            logger.info(f"Running Apify actor '{comment_actor_id}' for comments with input: {run_input}")
            
            comments_items = []
            run_details = None # Initialize run_details
            try:
                # --- ADDED LOG ---
                logger.info(f"--- Attempting to call Apify actor {comment_actor_id} ---") 
                run = client.actor(comment_actor_id).call(run_input=run_input)
                run_details = run # Store run details 
                # logger.info(f"Apify comment actor run details: {json.dumps(run_details)}") # REMOVED: Causes TypeError
                dataset_id = run.get("defaultDatasetId")
                if dataset_id:
                    logger.info(f"Attempting to fetch comments from dataset {dataset_id}")
                    try: # Add specific try/except around dataset operations
                        dataset = client.dataset(dataset_id)
                        logger.info(f"Dataset object created for {dataset_id}. Starting iteration...")
                        item_count = 0 
                        for item in dataset.iterate_items():
                            # Log processing for each item, including keys
                            logger.debug(f"Processing comment item {item_count} from dataset {dataset_id}. Item keys: {list(item.keys())}") 
                            comments_items.append(item)
                            item_count += 1
                        # Log after loop finishes
                        logger.info(f"Finished iterating dataset {dataset_id}. Total items processed: {item_count}. comments_items length: {len(comments_items)}")
                    except Exception as dataset_iteration_error:
                         logger.error(f"Error iterating dataset {dataset_id}: {str(dataset_iteration_error)}", exc_info=True)
                         # Let it continue, comments_items will be empty
                else:
                    logger.error("Apify run for comments did not return a dataset ID.")
                    # Let it continue, comments_items will be empty
                    
            except Exception as apify_call_error:
                # Log the error but allow the function to continue
                logger.error(f"Error calling Apify actor {comment_actor_id}: {str(apify_call_error)}", exc_info=True)
                # comments_items remains empty


            # --- Analyze Comments using OpenAI ---
            analysis_result = "AI analysis could not be performed."
            if comments_items:
                 # Log first few comment items to check structure
                logger.info(f"First 3 comment items structure: {json.dumps(comments_items[:3], indent=2)}")
                
                # Prepare comments text for OpenAI
                comments_text = "\n".join([f"- {item.get('text', '')}" for item in comments_items if item.get('text')]) # Ensure text exists
                logger.info(f"Generated comments_text for OpenAI (first 500 chars): {comments_text[:500]}") # Log part of the text
                
                if not comments_text:
                     logger.warning("comments_text is empty after processing items. Skipping OpenAI call.")
                     analysis_result = "Could not extract valid comment text to analyze."
                elif not settings.OPENAI_API_KEY:
                    logger.error("OpenAI API key is not configured.")
                else:
                    openai.api_key = settings.OPENAI_API_KEY
                    prompt = f"""Analyze the sentiment and key topics in the following TikTok comments. Provide a summary including:
1. Overall Sentiment (Positive/Negative/Neutral percentage estimate).
2. Key Topics/Themes mentioned in the comments.
3. Suggestions for the creator based on the comments.

Comments:
{comments_text[:3500]} 

Analysis:""" # Limit prompt length

                    logger.info("Sending comment analysis request to OpenAI...")
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are an assistant analyzing TikTok comment sections."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=250,
                            temperature=0.5
                        )
                        analysis_result = response.choices[0].message['content'].strip()
                        logger.info("Comment analysis complete.")
                    except openai.error.OpenAIError as openai_error:
                        logger.error(f"OpenAI API Error during comment analysis: {str(openai_error)}", exc_info=True)
                        analysis_result = "AI analysis failed due to API error."
                    except Exception as general_error:
                        logger.error(f"Unexpected error during OpenAI call for comments: {str(general_error)}", exc_info=True)
                        analysis_result = "An unexpected error occurred during AI analysis."
            else:
                 analysis_result = "No comments were retrieved to analyze."

            # --- Parse OpenAI Response for Scores ---
            positive_score = 0.0
            negative_score = 0.0
            # Try to extract percentages using regex (adjust regex as needed based on actual OpenAI output format)
            try:
                # Look for patterns like "Positive: XX%", "Negative: YY%"
                pos_match = re.search(r'Positive.*?(\d+)%', analysis_result, re.IGNORECASE)
                neg_match = re.search(r'Negative.*?(\d+)%', analysis_result, re.IGNORECASE)
                if pos_match:
                    positive_score = float(pos_match.group(1)) / 100.0
                if neg_match:
                    negative_score = float(neg_match.group(1)) / 100.0
                logger.info(f"Parsed sentiment scores - Positive: {positive_score}, Negative: {negative_score}")
            except Exception as parse_error:
                 logger.error(f"Error parsing sentiment scores from AI response: {parse_error}")
                 # Keep scores as 0.0 if parsing fails

            # --- Prepare Structured Response ---
            structured_analysis = {
                'summary': analysis_result,
                'positiveScore': positive_score,
                'negativeScore': negative_score
                # TODO: Add parsing for topics/recommendations if needed later
            }

            # Extract top comments (e.g., by likes if available)
            top_comments_data = []
            if comments_items:
                 # Sort by 'diggCount' if it exists, otherwise take first few
                 try:
                     sorted_comments = sorted(comments_items, key=lambda x: int(x.get('diggCount', 0)), reverse=True)
                     top_comments_data = [{'text': c.get('text', ''), 'diggCount': c.get('diggCount', 0)} for c in sorted_comments[:5]] # Top 5
                 except Exception as sort_err:
                      logger.warning(f"Could not sort comments: {sort_err}. Taking first 5.")
                      top_comments_data = [{'text': c.get('text', ''), 'diggCount': c.get('diggCount', 'N/A')} for c in comments_items[:5]]


            return Response({ 
                'videoId': video_id,
                'analysis': structured_analysis, # Use structured analysis object
                'topComments': top_comments_data 
            }, status=status.HTTP_200_OK)

        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from analyze_comments request body.")
            return Response(
                {'error': 'Invalid JSON format in request body'},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.critical(f"Unexpected error in analyze_comments POST handler: {str(e)}", exc_info=True)
            return Response(
                {'error': 'An unexpected server error occurred while analyzing comments.', 'details': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    else:
        logger.warning(f"Received unsupported method for analyze_comments: {request.method}")
        return Response(
            {'error': 'Method Not Allowed'},
            status=status.HTTP_405_METHOD_NOT_ALLOWED
        )
