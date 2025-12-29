"""
Celery tasks for background and scheduled flow execution.

This module provides:
1. Background flow execution
2. Scheduled/periodic flow runs
3. Webhook processing
4. Batch message processing
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def run_flow_async(
    self,
    tenant_id: str,
    flow_id: str,
    user_id: str,
    session_id: str,
    message: str,
    callback_url: Optional[str] = None,
):
    """
    Run a flow in the background.
    
    This is useful for:
    - Long-running flows
    - Webhook-triggered flows
    - Batch processing
    
    Args:
        tenant_id: Tenant identifier
        flow_id: Flow identifier
        user_id: User identifier
        session_id: Session identifier
        message: Input message
        callback_url: Optional URL to POST the result
    
    Returns:
        The agent's response text
    """
    import asyncio
    import os
    from asgiref.sync import sync_to_async
    
    from flows.models import Flow, FlowExecution
    from adk_integration.executor.flow_executor import FlowExecutor
    from google.adk.sessions import InMemorySessionService
    
    logger.info(f"Starting background flow execution: {flow_id}")
    
    # Create execution log
    execution = None
    try:
        flow = Flow.objects.get(id=flow_id)
        execution = FlowExecution.objects.create(
            flow=flow,
            status='running',
            input_message=message,
        )
    except Exception as e:
        logger.error(f"Failed to create execution log: {e}")
    
    async def _run():
        async def get_flow(tid, fid):
            f = await sync_to_async(Flow.objects.select_related('tenant').get)(id=fid)
            return f.get_flow_schema()
        
        async def get_keys(tid):
            return {'google': os.environ.get('GOOGLE_API_KEY', '')}
        
        executor = FlowExecutor(
            get_flow_callback=get_flow,
            get_api_keys_callback=get_keys,
            session_service_factory=lambda t: InMemorySessionService(),
        )
        
        try:
            response = await executor.run_flow_sync(
                tenant_id=tenant_id,
                flow_id=flow_id,
                user_id=user_id,
                session_id=session_id,
                message=message,
            )
            return response
        finally:
            await executor.shutdown()
    
    try:
        # Run the async flow
        response = asyncio.run(_run())
        
        # Update execution log
        if execution:
            execution.status = 'completed'
            execution.output_message = response
            execution.completed_at = timezone.now()
            execution.latency_ms = int(
                (execution.completed_at - execution.started_at).total_seconds() * 1000
            )
            execution.save()
        
        # Send callback if provided
        if callback_url:
            _send_callback(callback_url, {
                'status': 'success',
                'response': response,
                'flow_id': flow_id,
                'session_id': session_id,
            })
        
        logger.info(f"Background flow execution completed: {flow_id}")
        return response
        
    except Exception as e:
        logger.error(f"Background flow execution failed: {e}")
        
        # Update execution log
        if execution:
            execution.status = 'failed'
            execution.error_message = str(e)
            execution.completed_at = timezone.now()
            execution.save()
        
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@shared_task
def run_scheduled_flow(
    flow_id: str,
    input_message: str = "This is a scheduled execution.",
):
    """
    Run a scheduled flow.
    
    This task is called by Celery Beat for periodic flow execution.
    
    Args:
        flow_id: Flow identifier
        input_message: Default message for the scheduled run
    """
    from flows.models import Flow
    
    try:
        flow = Flow.objects.select_related('tenant').get(id=flow_id, is_active=True)
        
        # Create unique session for scheduled run
        session_id = f"scheduled_{flow_id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Queue the flow execution
        run_flow_async.delay(
            tenant_id=str(flow.tenant.id),
            flow_id=flow_id,
            user_id='scheduler',
            session_id=session_id,
            message=input_message,
        )
        
        logger.info(f"Scheduled flow queued: {flow_id}")
        
    except Flow.DoesNotExist:
        logger.error(f"Scheduled flow not found: {flow_id}")
    except Exception as e:
        logger.error(f"Failed to queue scheduled flow: {e}")


@shared_task
def process_webhook(
    webhook_id: str,
    payload: dict,
    headers: dict,
):
    """
    Process a webhook trigger asynchronously.
    
    Args:
        webhook_id: Webhook identifier
        payload: Request payload
        headers: Request headers
    """
    from flows.models import Webhook
    
    try:
        webhook = Webhook.objects.select_related('flow', 'flow__tenant').get(
            id=webhook_id,
            is_active=True,
        )
        
        # Extract message from payload
        message = payload.get('message', payload.get('text', str(payload)))
        session_id = payload.get('session_id', f"webhook_{webhook_id}")
        
        # Queue the flow execution
        run_flow_async.delay(
            tenant_id=str(webhook.flow.tenant.id),
            flow_id=str(webhook.flow.id),
            user_id=f"webhook_{webhook_id}",
            session_id=session_id,
            message=message,
            callback_url=payload.get('callback_url'),
        )
        
        # Update last triggered
        webhook.last_triggered_at = timezone.now()
        webhook.save(update_fields=['last_triggered_at'])
        
        logger.info(f"Webhook processed: {webhook_id}")
        
    except Webhook.DoesNotExist:
        logger.error(f"Webhook not found: {webhook_id}")
    except Exception as e:
        logger.error(f"Failed to process webhook: {e}")


@shared_task
def cleanup_expired_sessions(days_old: int = 7):
    """
    Clean up expired sessions.
    
    Args:
        days_old: Delete sessions older than this many days
    """
    from flows.models import Session
    
    cutoff = timezone.now() - timedelta(days=days_old)
    
    deleted, _ = Session.objects.filter(
        updated_at__lt=cutoff,
        status__in=['completed', 'expired'],
    ).delete()
    
    logger.info(f"Cleaned up {deleted} expired sessions")


@shared_task
def cleanup_old_executions(days_old: int = 30):
    """
    Clean up old execution logs.
    
    Args:
        days_old: Delete executions older than this many days
    """
    from flows.models import FlowExecution
    
    cutoff = timezone.now() - timedelta(days=days_old)
    
    deleted, _ = FlowExecution.objects.filter(
        started_at__lt=cutoff,
    ).delete()
    
    logger.info(f"Cleaned up {deleted} old execution logs")


def _send_callback(url: str, data: dict):
    """Send callback to webhook URL."""
    import httpx
    
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(url, json=data)
            response.raise_for_status()
            logger.info(f"Callback sent to {url}")
    except Exception as e:
        logger.error(f"Failed to send callback to {url}: {e}")

