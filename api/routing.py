"""
WebSocket URL routing for the API app.
"""

from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Real-time text chat
    re_path(
        r'ws/chat/(?P<flow_id>[0-9a-f-]+)/$',
        consumers.ChatConsumer.as_asgi(),
        name='ws_chat',
    ),
    
    # Voice streaming
    re_path(
        r'ws/voice/(?P<flow_id>[0-9a-f-]+)/$',
        consumers.VoiceConsumer.as_asgi(),
        name='ws_voice',
    ),
    
    # Gemini Live streaming
    re_path(
        r'ws/live/(?P<flow_id>[0-9a-f-]+)/$',
        consumers.LiveStreamConsumer.as_asgi(),
        name='ws_live',
    ),
]

