"""api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
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
    path('admin/', admin.site.urls),
    path('api/', include('tiktok_api.urls')),
]
