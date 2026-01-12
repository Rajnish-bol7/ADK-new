"""
URL configuration for ADK Flow Platform.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from core.views import chat_tester, list_tenants

urlpatterns = [
    # Admin
    path("admin/", admin.site.urls),
    
    # Chat Tester UI (WhatsApp-like interface)
    path("", chat_tester, name="chat_tester"),
    path("chat/", chat_tester, name="chat_tester_alt"),
    
    # Webhook endpoints
    path("webhook/", include("webhook.urls")),
    
    # API endpoints
    path("api/v1/", include("api.urls")),
    path("api/v1/tenants/", list_tenants, name="api_list_tenants"),
    
    # Health check
    path("health/", include("core.urls")),
]

# Serve static and media files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
