"""
API URL routes for WhatsApp message sending
"""
from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    path('send-message/', views.send_message, name='send_message'),
]

