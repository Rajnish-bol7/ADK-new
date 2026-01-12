from django.urls import path
from . import views

app_name = 'webhook'

urlpatterns = [
    path('', views.whatsapp_webhook, name='whatsapp_webhook'),  # Static route
    path('whatsapp/<str:webhook_secret>/', 
         views.whatsapp_webhook_dynamic, 
         name='whatsapp_webhook_dynamic'),  # Dynamic route
]

