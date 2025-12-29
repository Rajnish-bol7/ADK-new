"""
API URL configuration.
"""

from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path("auth/login/", views.login_view, name="api_login"),
    path("auth/logout/", views.logout_view, name="api_logout"),
    path("auth/me/", views.current_user, name="api_current_user"),
    
    # Flows CRUD
    path("flows/", views.flow_list, name="api_flow_list"),
    path("flows/<uuid:flow_id>/", views.flow_detail, name="api_flow_detail"),
    
    # Flow execution
    path("flows/<uuid:flow_id>/chat/", views.flow_chat_test, name="api_flow_chat"),  # Test-friendly
    path("flows/<uuid:flow_id>/chat/stream/", views.flow_chat_stream, name="api_flow_chat_stream"),
    path("flows/<uuid:flow_id>/chat/auth/", views.flow_chat, name="api_flow_chat_auth"),  # Requires auth
    
    # Sessions
    path("flows/<uuid:flow_id>/sessions/", views.session_list, name="api_session_list"),
    path("sessions/<uuid:session_id>/", views.session_detail, name="api_session_detail"),
    path("sessions/<uuid:session_id>/messages/", views.session_messages, name="api_session_messages"),
    
    # Webhooks
    path("webhooks/<str:webhook_path>/", views.webhook_trigger, name="api_webhook_trigger"),
    
    # Cache management
    path("cache/invalidate/", views.invalidate_cache, name="api_invalidate_cache"),
    
    # Background execution
    path("flows/<uuid:flow_id>/run-async/", views.run_flow_background, name="api_run_flow_background"),
    
    # Scheduled flows
    path("flows/<uuid:flow_id>/schedule/", views.schedule_flow, name="api_schedule_flow"),
]

