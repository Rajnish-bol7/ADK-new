"""
Celery configuration for ADK Flow Platform.

This enables:
1. Background task execution
2. Scheduled/periodic flow execution
3. Async webhook processing
"""

import os
from celery import Celery

# Set the default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

# Create Celery app
app = Celery('adk_flow_platform')

# Load config from Django settings
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks in all installed apps
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    """Debug task to test Celery."""
    print(f'Request: {self.request!r}')

