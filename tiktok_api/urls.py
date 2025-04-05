from django.urls import path
from . import views

urlpatterns = [
    path('profile/videos/', views.get_profile_videos, name='get_profile_videos'),
    path('hashtag/videos/', views.search_hashtag_videos, name='search_hashtag_videos'),
    path('insights/', views.get_ai_insights, name='get_ai_insights'),
    path('timing/', views.analyze_post_timing, name='analyze_post_timing'),
    path('comments/', views.analyze_comments, name='analyze_comments'),
]
