"""api URL Configuration"""
from django.urls import path, include
from django.http import JsonResponse

def api_root(request):
    return JsonResponse({
        'name': 'TikTok Profile API',
        'version': '1.0.0',
        'description': 'API for fetching TikTok profile data and analytics',
        'endpoints': {
            'profile_videos': {
                'url': '/api/profile/videos/',
                'method': 'GET',
                'params': {
                    'username': 'TikTok username (e.g. gordonramsay)'
                },
                'example': '/api/profile/videos/?username=gordonramsay'
            }
        },
        'documentation': 'For more information, visit /api/'
    }, json_dumps_params={'indent': 2})

urlpatterns = [
    path('', api_root, name='api-root'),
    path('api/', include('tiktok_api.urls')),
]
