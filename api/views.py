"""
API views for the ADK Flow Platform.

This module provides REST API endpoints for managing flows, sessions,
and executing agent conversations.
"""

import asyncio
import json
import logging
from functools import wraps
from uuid import UUID

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from asgiref.sync import async_to_sync, sync_to_async

from core.models import Tenant, User
from flows.models import Flow, Session, Message, FlowExecution

from adk_integration.schema.flow_schema import FlowSchema
from adk_integration.executor.flow_executor import FlowExecutor, init_executor

logger = logging.getLogger(__name__)


# =============================================================================
# Flow Executor Initialization
# =============================================================================

_executor = None


def get_executor() -> FlowExecutor:
    """Get or create the global flow executor."""
    global _executor
    
    if _executor is None:
        _executor = init_executor(
            get_flow_callback=_get_flow_from_db,
            get_api_keys_callback=_get_tenant_api_keys,
            session_service_factory=_create_session_service,
            cache_max_size=settings.FLOW_EXECUTOR_CACHE_MAX_SIZE,
            cache_ttl_seconds=settings.FLOW_EXECUTOR_CACHE_TTL,
        )
    
    return _executor


async def _get_flow_from_db(tenant_id: str, flow_id: str) -> FlowSchema | None:
    """Fetch flow from database."""
    try:
        flow = await sync_to_async(Flow.objects.get)(
            tenant_id=tenant_id,
            id=flow_id,
            is_active=True,
        )
        return flow.get_flow_schema()
    except Flow.DoesNotExist:
        return None


async def _get_tenant_api_keys(tenant_id: str) -> dict:
    """Fetch tenant's API keys."""
    try:
        tenant = await sync_to_async(Tenant.objects.get)(id=tenant_id)
        keys = tenant.get_api_keys()
        
        # Fallback to global keys if tenant doesn't have them
        if not keys.get("google"):
            keys["google"] = settings.GOOGLE_API_KEY
        if not keys.get("openai"):
            keys["openai"] = settings.OPENAI_API_KEY
        if not keys.get("anthropic"):
            keys["anthropic"] = settings.ANTHROPIC_API_KEY
            
        return keys
    except Tenant.DoesNotExist:
        return {
            "google": settings.GOOGLE_API_KEY,
            "openai": settings.OPENAI_API_KEY,
            "anthropic": settings.ANTHROPIC_API_KEY,
        }


def _create_session_service(tenant_id: str):
    """Create session service for a tenant."""
    from google.adk.sessions import InMemorySessionService
    # TODO: Replace with DatabaseSessionService for production
    return InMemorySessionService()


# =============================================================================
# Helper Decorators
# =============================================================================

def async_view(view_func):
    """Decorator to run async views in Django's sync context."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        return async_to_sync(view_func)(request, *args, **kwargs)
    return wrapper


def require_auth(view_func):
    """Decorator to require authentication."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)
        return view_func(request, *args, **kwargs)
    return wrapper


def require_tenant(view_func):
    """Decorator to require tenant association."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)
        if not request.user.tenant:
            return JsonResponse({"error": "No tenant associated"}, status=403)
        return view_func(request, *args, **kwargs)
    return wrapper


# =============================================================================
# Authentication Views
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
def login_view(request):
    """Login endpoint."""
    try:
        data = json.loads(request.body)
        username = data.get("username")
        password = data.get("password")
        
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return JsonResponse({
                "status": "success",
                "user": {
                    "id": str(user.id),
                    "username": user.username,
                    "email": user.email,
                    "tenant_id": str(user.tenant.id) if user.tenant else None,
                },
            })
        else:
            return JsonResponse({"error": "Invalid credentials"}, status=401)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def logout_view(request):
    """Logout endpoint."""
    logout(request)
    return JsonResponse({"status": "success"})


@require_http_methods(["GET"])
@require_auth
def current_user(request):
    """Get current user info."""
    user = request.user
    return JsonResponse({
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "tenant_id": str(user.tenant.id) if user.tenant else None,
        "tenant_name": user.tenant.name if user.tenant else None,
    })


# =============================================================================
# Flow CRUD Views
# =============================================================================

@csrf_exempt
@require_http_methods(["GET", "POST"])
def flow_list(request):
    """List flows or create a new flow."""
    if request.method == "GET":
        # For GET, allow listing all flows (for tester UI) or filter by tenant
        tenant_id = request.GET.get("tenant_id")
        
        if request.user.is_authenticated and request.user.tenant:
            # Authenticated user: show their tenant's flows
            flows = Flow.objects.filter(tenant=request.user.tenant, is_active=True)
        elif tenant_id:
            # Filter by specific tenant
            flows = Flow.objects.filter(tenant_id=tenant_id, is_active=True)
        else:
            # Show all active flows (for tester)
            flows = Flow.objects.filter(is_active=True).select_related("tenant")
        
        def get_flow_capabilities(flow):
            """Determine chat/call capabilities from flow JSON."""
            nodes = flow.flow_json.get("nodes", []) if flow.flow_json else []
            has_call_trigger = any(n.get("type") == "call_start" for n in nodes)
            return {
                "chat_enabled": True,  # Chat is ALWAYS enabled (default)
                "call_enabled": has_call_trigger,  # Call only if call_start node exists
            }
        
        return JsonResponse({
            "flows": [
                {
                    "id": str(f.id),
                    "name": f.name,
                    "description": f.description,
                    "trigger_type": f.trigger_type,
                    "is_active": f.is_active,
                    "is_published": f.is_published,
                    "version": f.version,
                    "tenant_id": str(f.tenant.id) if f.tenant else None,
                    "tenant_name": f.tenant.name if f.tenant else None,
                    "flow_json": f.flow_json,
                    "created_at": f.created_at.isoformat(),
                    "updated_at": f.updated_at.isoformat(),
                    # Chat is always enabled, Call only if trigger exists
                    **get_flow_capabilities(f),
                }
                for f in flows
            ]
        })
    
    else:  # POST
        # Require authentication for creating flows
        if not request.user.is_authenticated or not request.user.tenant:
            return JsonResponse({"error": "Authentication required to create flows"}, status=401)
        
        try:
            data = json.loads(request.body)
            
            flow = Flow.objects.create(
                tenant=request.user.tenant,
                created_by=request.user,
                name=data.get("name", "Untitled Flow"),
                description=data.get("description", ""),
                flow_json=data.get("flow_json", {"nodes": [], "connections": []}),
                trigger_type=data.get("trigger_type", "chat"),
            )
            
            return JsonResponse({
                "status": "created",
                "flow": {
                    "id": str(flow.id),
                    "name": flow.name,
                },
            }, status=201)
            
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)


@csrf_exempt
@require_http_methods(["GET", "PUT", "DELETE"])
@require_tenant
def flow_detail(request, flow_id: UUID):
    """Get, update, or delete a specific flow."""
    try:
        flow = Flow.objects.get(id=flow_id, tenant=request.user.tenant)
    except Flow.DoesNotExist:
        return JsonResponse({"error": "Flow not found"}, status=404)
    
    if request.method == "GET":
        return JsonResponse({
            "id": str(flow.id),
            "name": flow.name,
            "description": flow.description,
            "flow_json": flow.flow_json,
            "trigger_type": flow.trigger_type,
            "is_active": flow.is_active,
            "is_published": flow.is_published,
            "version": flow.version,
            "created_at": flow.created_at.isoformat(),
            "updated_at": flow.updated_at.isoformat(),
        })
    
    elif request.method == "PUT":
        try:
            data = json.loads(request.body)
            
            # Update fields
            if "name" in data:
                flow.name = data["name"]
            if "description" in data:
                flow.description = data["description"]
            if "flow_json" in data:
                flow.flow_json = data["flow_json"]
                flow.version += 1
            if "trigger_type" in data:
                flow.trigger_type = data["trigger_type"]
            if "is_active" in data:
                flow.is_active = data["is_active"]
            if "is_published" in data:
                flow.is_published = data["is_published"]
                if flow.is_published:
                    flow.published_at = timezone.now()
            
            flow.save()
            
            # Invalidate executor cache
            executor = get_executor()
            async_to_sync(executor.invalidate_flow_cache)(
                str(request.user.tenant.id),
                str(flow.id),
            )
            
            return JsonResponse({
                "status": "updated",
                "flow": {
                    "id": str(flow.id),
                    "name": flow.name,
                    "version": flow.version,
                },
            })
            
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    
    else:  # DELETE
        flow.delete()
        return JsonResponse({"status": "deleted"})


# =============================================================================
# Flow Execution Views
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@require_tenant
def flow_chat(request, flow_id: UUID):
    """
    Non-streaming chat endpoint.
    
    POST /api/v1/flows/{flow_id}/chat/
    Body: {"session_id": "...", "message": "..."}
    """
    try:
        data = json.loads(request.body)
        session_id = data.get("session_id", str(flow_id))
        message = data.get("message", "")
        
        if not message:
            return JsonResponse({"error": "Message is required"}, status=400)
        
        # Verify flow exists
        try:
            flow = Flow.objects.get(id=flow_id, tenant=request.user.tenant, is_active=True)
        except Flow.DoesNotExist:
            return JsonResponse({"error": "Flow not found"}, status=404)
        
        executor = get_executor()
        
        # Run flow synchronously
        response = async_to_sync(executor.run_flow_sync)(
            tenant_id=str(request.user.tenant.id),
            flow_id=str(flow_id),
            user_id=str(request.user.id),
            session_id=session_id,
            message=message,
        )
        
        # Log message
        _log_message(flow, session_id, request.user, message, response)
        
        return JsonResponse({
            "response": response,
            "session_id": session_id,
        })
        
    except Exception as e:
        logger.error(f"Error in flow_chat: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@require_tenant
def flow_chat_stream(request, flow_id: UUID):
    """
    Streaming chat endpoint using Server-Sent Events.
    
    POST /api/v1/flows/{flow_id}/chat/stream/
    Body: {"session_id": "...", "message": "..."}
    """
    try:
        data = json.loads(request.body)
        session_id = data.get("session_id", str(flow_id))
        message = data.get("message", "")
        
        if not message:
            return JsonResponse({"error": "Message is required"}, status=400)
        
        # Verify flow exists
        try:
            flow = Flow.objects.get(id=flow_id, tenant=request.user.tenant, is_active=True)
        except Flow.DoesNotExist:
            return JsonResponse({"error": "Flow not found"}, status=404)
        
        tenant_id = str(request.user.tenant.id)
        user_id = str(request.user.id)
        
        def event_stream():
            """Generate SSE events."""
            executor = get_executor()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            final_response = ""
            
            try:
                async_gen = executor.run_flow(
                    tenant_id=tenant_id,
                    flow_id=str(flow_id),
                    user_id=user_id,
                    session_id=session_id,
                    message=message,
                )
                
                while True:
                    try:
                        event = loop.run_until_complete(async_gen.__anext__())
                        
                        # Extract text for logging
                        if event.content and event.content.parts:
                            for part in event.content.parts:
                                if hasattr(part, "text") and part.text:
                                    final_response = part.text
                        
                        yield f"data: {event.model_dump_json()}\n\n"
                        
                    except StopAsyncIteration:
                        break
                    except Exception as e:
                        logger.error(f"Error in stream: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        break
                
                # Log the conversation
                _log_message(flow, session_id, request.user, message, final_response)
                
            finally:
                loop.close()
        
        return StreamingHttpResponse(
            event_stream(),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Error in flow_chat_stream: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def _log_message(flow, session_id, user, user_message, assistant_response):
    """Log messages to database."""
    try:
        # Get or create session
        session, _ = Session.objects.get_or_create(
            id=session_id,
            defaults={
                "flow": flow,
                "user": user,
            }
        )
        
        # Log user message
        Message.objects.create(
            session=session,
            role="user",
            content=user_message,
        )
        
        # Log assistant response
        if assistant_response:
            Message.objects.create(
                session=session,
                role="assistant",
                content=assistant_response,
            )
            
    except Exception as e:
        logger.warning(f"Failed to log message: {e}")


# =============================================================================
# Test Chat Endpoint (for Chat Tester UI - simplified auth)
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
def flow_chat_test(request, flow_id: UUID):
    """
    Simplified chat endpoint for the tester UI.
    Works with or without authentication.
    
    POST /api/v1/flows/{flow_id}/chat/
    Body: {"session_id": "...", "message": "..."}
    """
    import concurrent.futures
    import asyncio
    import os
    
    try:
        data = json.loads(request.body)
        session_id = data.get("session_id", f"test_{flow_id}")
        message = data.get("message", "")
        
        if not message:
            return JsonResponse({"error": "Message is required"}, status=400)
        
        # Get flow (without strict tenant check for testing)
        try:
            flow = Flow.objects.select_related("tenant").get(id=flow_id, is_active=True)
        except Flow.DoesNotExist:
            return JsonResponse({"error": "Flow not found"}, status=404)
        
        # Get user info
        user_id = "anonymous"
        if request.user.is_authenticated:
            user_id = str(request.user.id)
        
        # Get all data we need from Django ORM first (sync context)
        tenant_id = str(flow.tenant.id)
        flow_id_str = str(flow_id)
        flow_schema = flow.get_flow_schema()
        
        # Get API keys
        api_keys = flow.tenant.get_api_keys()
        if not api_keys.get("google"):
            api_keys["google"] = os.environ.get("GOOGLE_API_KEY", "")
        if not api_keys.get("openai"):
            api_keys["openai"] = os.environ.get("OPENAI_API_KEY", "")
        
        def run_in_thread():
            """Run the flow in a separate thread with its own event loop."""
            import asyncio
            from google.adk import Agent, Runner
            from google.adk.sessions import InMemorySessionService
            from google.genai import types
            from adk_integration.builder.flow_builder import FlowBuilder
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def execute():
                    # Build the flow directly (no callbacks needed)
                    builder = FlowBuilder(
                        tenant_id=tenant_id,
                        api_keys=api_keys,
                    )
                    build_result = await builder.build(flow_schema)
                    
                    # Create session service
                    session_service = InMemorySessionService()
                    
                    # Create runner
                    runner = Runner(
                        agent=build_result.root_agent,
                        app_name=f"flow_{flow_id_str}",
                        session_service=session_service,
                    )
                    
                    # Create session - always create new for simplicity
                    app_name = f"flow_{flow_id_str}"
                    await session_service.create_session(
                        app_name=app_name,
                        user_id=user_id,
                        session_id=session_id,
                    )
                    
                    # Run the agent
                    response_text = ""
                    content = types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=message)]
                    )
                    
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=session_id,
                        new_message=content,
                    ):
                        if event.content and event.content.parts:
                            for part in event.content.parts:
                                if hasattr(part, "text") and part.text:
                                    response_text = part.text
                    
                    # Cleanup
                    await build_result.cleanup()
                    
                    return response_text
                
                return loop.run_until_complete(execute())
            finally:
                loop.close()
        
        # Run in a thread pool to avoid async context issues
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_in_thread)
            response = future.result(timeout=120)  # 2 minute timeout
        
        # Log message if user is authenticated
        if request.user.is_authenticated:
            _log_message(flow, session_id, request.user, message, response)
        
        return JsonResponse({
            "response": response,
            "session_id": session_id,
            "flow_name": flow.name,
        })
        
    except concurrent.futures.TimeoutError:
        return JsonResponse({"error": "Request timed out"}, status=504)
    except Exception as e:
        logger.error(f"Error in flow_chat_test: {e}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


# =============================================================================
# Session Views
# =============================================================================

@require_http_methods(["GET"])
@require_tenant
def session_list(request, flow_id: UUID):
    """List sessions for a flow."""
    try:
        flow = Flow.objects.get(id=flow_id, tenant=request.user.tenant)
    except Flow.DoesNotExist:
        return JsonResponse({"error": "Flow not found"}, status=404)
    
    sessions = Session.objects.filter(flow=flow)
    return JsonResponse({
        "sessions": [
            {
                "id": str(s.id),
                "status": s.status,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "message_count": s.messages.count(),
            }
            for s in sessions
        ]
    })


@require_http_methods(["GET", "DELETE"])
@require_tenant
def session_detail(request, session_id: UUID):
    """Get or delete a session."""
    try:
        session = Session.objects.get(
            id=session_id,
            flow__tenant=request.user.tenant,
        )
    except Session.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)
    
    if request.method == "GET":
        return JsonResponse({
            "id": str(session.id),
            "flow_id": str(session.flow.id),
            "flow_name": session.flow.name,
            "status": session.status,
            "state": session.state,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        })
    
    else:  # DELETE
        session.delete()
        return JsonResponse({"status": "deleted"})


@require_http_methods(["GET"])
@require_tenant
def session_messages(request, session_id: UUID):
    """Get messages for a session."""
    try:
        session = Session.objects.get(
            id=session_id,
            flow__tenant=request.user.tenant,
        )
    except Session.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)
    
    messages = session.messages.all()
    return JsonResponse({
        "messages": [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ]
    })


# =============================================================================
# Webhook Views
# =============================================================================

@csrf_exempt
@require_http_methods(["GET", "POST"])
def webhook_trigger(request, webhook_path: str):
    """
    Trigger a flow via webhook.
    
    GET/POST /api/v1/webhooks/{webhook_path}/
    """
    from flows.models import Webhook
    
    try:
        webhook = Webhook.objects.get(path=webhook_path, is_active=True)
    except Webhook.DoesNotExist:
        return JsonResponse({"error": "Webhook not found"}, status=404)
    
    # Check method
    if request.method not in webhook.allowed_methods:
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    # Get message from body or query params
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            message = data.get("message", "")
        except json.JSONDecodeError:
            message = request.body.decode("utf-8")
    else:
        message = request.GET.get("message", "")
    
    if not message:
        return JsonResponse({"error": "Message is required"}, status=400)
    
    # Execute flow
    executor = get_executor()
    session_id = request.GET.get("session_id", f"webhook_{webhook_path}")
    
    try:
        response = async_to_sync(executor.run_flow_sync)(
            tenant_id=str(webhook.flow.tenant.id),
            flow_id=str(webhook.flow.id),
            user_id=f"webhook_{webhook_path}",
            session_id=session_id,
            message=message,
        )
        
        # Update last triggered
        webhook.last_triggered_at = timezone.now()
        webhook.save(update_fields=["last_triggered_at"])
        
        return JsonResponse({
            "response": response,
            "session_id": session_id,
        })
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JsonResponse({"error": str(e)}, status=500)


# =============================================================================
# Cache Management Views
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@require_tenant
def invalidate_cache(request):
    """Invalidate flow executor cache for the tenant."""
    executor = get_executor()
    count = async_to_sync(executor.invalidate_tenant_cache)(
        str(request.user.tenant.id)
    )
    
    return JsonResponse({
        "status": "invalidated",
        "count": count,
    })


# =============================================================================
# Background Execution Views
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@require_tenant
def run_flow_background(request, flow_id: UUID):
    """
    Run a flow in the background using Celery.
    
    Useful for long-running flows or when you don't need immediate response.
    
    Request body:
    {
        "message": "User input message",
        "session_id": "optional-session-id",
        "callback_url": "optional-url-for-result"
    }
    
    Response:
    {
        "task_id": "celery-task-id",
        "status": "queued"
    }
    """
    from flows.tasks import run_flow_async
    
    try:
        flow = Flow.objects.get(id=flow_id, tenant=request.user.tenant, is_active=True)
    except Flow.DoesNotExist:
        return JsonResponse({"error": "Flow not found"}, status=404)
    
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    message = data.get("message", "").strip()
    if not message:
        return JsonResponse({"error": "Message is required"}, status=400)
    
    session_id = data.get("session_id") or f"bg_{flow_id}_{request.user.id}"
    callback_url = data.get("callback_url")
    
    # Queue the task
    task = run_flow_async.delay(
        tenant_id=str(request.user.tenant.id),
        flow_id=str(flow_id),
        user_id=str(request.user.id),
        session_id=session_id,
        message=message,
        callback_url=callback_url,
    )
    
    return JsonResponse({
        "task_id": task.id,
        "status": "queued",
        "session_id": session_id,
    })


@csrf_exempt
@require_http_methods(["POST"])
@require_tenant
def schedule_flow(request, flow_id: UUID):
    """
    Create a scheduled task for periodic flow execution.
    
    Request body:
    {
        "name": "Daily Report",
        "schedule_type": "interval|crontab",
        "interval_minutes": 60,  // for interval type
        "crontab": "0 9 * * *",  // for crontab type (cron expression)
        "input_message": "Generate daily report",
        "enabled": true
    }
    
    Response:
    {
        "schedule_id": "periodic-task-id",
        "status": "created"
    }
    """
    from django_celery_beat.models import PeriodicTask, IntervalSchedule, CrontabSchedule
    
    try:
        flow = Flow.objects.get(id=flow_id, tenant=request.user.tenant)
    except Flow.DoesNotExist:
        return JsonResponse({"error": "Flow not found"}, status=404)
    
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    name = data.get("name", f"Scheduled: {flow.name}")
    schedule_type = data.get("schedule_type", "interval")
    input_message = data.get("input_message", "Scheduled execution")
    enabled = data.get("enabled", True)
    
    try:
        if schedule_type == "interval":
            minutes = int(data.get("interval_minutes", 60))
            schedule, _ = IntervalSchedule.objects.get_or_create(
                every=minutes,
                period=IntervalSchedule.MINUTES,
            )
            
            task, created = PeriodicTask.objects.update_or_create(
                name=f"{name} ({flow_id})",
                defaults={
                    "task": "flows.tasks.run_scheduled_flow",
                    "interval": schedule,
                    "args": json.dumps([str(flow_id), input_message]),
                    "enabled": enabled,
                }
            )
            
        elif schedule_type == "crontab":
            cron_expr = data.get("crontab", "0 9 * * *")
            parts = cron_expr.split()
            
            if len(parts) != 5:
                return JsonResponse(
                    {"error": "Invalid crontab expression. Expected: minute hour day_of_month month_of_year day_of_week"},
                    status=400
                )
            
            schedule, _ = CrontabSchedule.objects.get_or_create(
                minute=parts[0],
                hour=parts[1],
                day_of_month=parts[2],
                month_of_year=parts[3],
                day_of_week=parts[4],
            )
            
            task, created = PeriodicTask.objects.update_or_create(
                name=f"{name} ({flow_id})",
                defaults={
                    "task": "flows.tasks.run_scheduled_flow",
                    "crontab": schedule,
                    "args": json.dumps([str(flow_id), input_message]),
                    "enabled": enabled,
                }
            )
            
        else:
            return JsonResponse(
                {"error": f"Invalid schedule_type: {schedule_type}. Use 'interval' or 'crontab'"},
                status=400
            )
        
        return JsonResponse({
            "schedule_id": task.id,
            "name": task.name,
            "status": "created" if created else "updated",
            "enabled": task.enabled,
        })
        
    except Exception as e:
        logger.error(f"Failed to create schedule: {e}")
        return JsonResponse({"error": str(e)}, status=500)
