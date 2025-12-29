# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Django Integration Views for ADK Flow Executor.

This module provides example Django views for integrating the ADK flow
executor with your Django backend. These views handle:
- Chat endpoint (streaming SSE)
- Flow save/update
- Session management
- Health checks

IMPORTANT: This is an example implementation. You should adapt it to your
specific Django project structure, authentication system, and database models.
"""

from __future__ import annotations

import asyncio
import json
import logging
from functools import wraps
from typing import Any, Callable, Dict

# Django imports - adjust based on your Django version
try:
    from django.http import (
        HttpRequest,
        HttpResponse,
        JsonResponse,
        StreamingHttpResponse,
    )
    from django.views.decorators.csrf import csrf_exempt
    from django.views.decorators.http import require_http_methods
    from asgiref.sync import async_to_sync, sync_to_async

    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    # Provide stubs for type checking
    HttpRequest = Any
    HttpResponse = Any
    JsonResponse = Any
    StreamingHttpResponse = Any

from ..executor.flow_executor import FlowExecutor, init_executor
from ..schema.flow_schema import FlowSchema

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration - Replace with your actual implementations
# =============================================================================


async def get_flow_from_db(
    tenant_id: str,
    flow_id: str,
) -> FlowSchema | None:
    """
    Fetch flow from your Django database.

    REPLACE THIS with your actual database query.

    Example implementation:
        from your_app.models import Flow

        @sync_to_async
        def _get_flow():
            try:
                flow_obj = Flow.objects.get(tenant_id=tenant_id, id=flow_id)
                return FlowSchema(**flow_obj.flow_json)
            except Flow.DoesNotExist:
                return None

        return await _get_flow()
    """
    # Placeholder - replace with your implementation
    logger.warning("get_flow_from_db not implemented - using placeholder")
    return None


async def get_tenant_api_keys(tenant_id: str) -> Dict[str, str]:
    """
    Fetch tenant's API keys from your database.

    REPLACE THIS with your actual implementation.

    Example implementation:
        from your_app.models import TenantAPIKeys

        @sync_to_async
        def _get_keys():
            try:
                keys = TenantAPIKeys.objects.get(tenant_id=tenant_id)
                return {
                    "google": keys.google_api_key,
                    "openai": keys.openai_api_key,
                    "anthropic": keys.anthropic_api_key,
                }
            except TenantAPIKeys.DoesNotExist:
                return {}

        return await _get_keys()
    """
    # Placeholder - replace with your implementation
    logger.warning("get_tenant_api_keys not implemented - using placeholder")
    return {}


def create_session_service(tenant_id: str):
    """
    Create a session service for a tenant.

    REPLACE THIS with your actual implementation for persistent sessions.

    For production, you might use:
    - DatabaseSessionService with your PostgreSQL
    - Redis-backed session service
    - Spanner-backed session service (for GCP)
    """
    from google.adk.sessions import InMemorySessionService

    # For development/testing, use in-memory
    # For production, use a persistent session service
    return InMemorySessionService()


# =============================================================================
# Initialize Executor - Call this at Django startup
# =============================================================================


def initialize_executor() -> FlowExecutor:
    """
    Initialize the global FlowExecutor.

    Call this from your Django AppConfig.ready() method or wsgi.py/asgi.py.

    Example in your_app/apps.py:
        from django.apps import AppConfig

        class YourAppConfig(AppConfig):
            name = 'your_app'

            def ready(self):
                from adk_integration.api.views import initialize_executor
                initialize_executor()
    """
    return init_executor(
        get_flow_callback=get_flow_from_db,
        get_api_keys_callback=get_tenant_api_keys,
        session_service_factory=create_session_service,
        cache_max_size=1000,
        cache_ttl_seconds=3600,
    )


# Global executor instance - initialized lazily
_executor: FlowExecutor | None = None


def get_executor() -> FlowExecutor:
    """Get or create the global executor."""
    global _executor
    if _executor is None:
        _executor = initialize_executor()
    return _executor


# =============================================================================
# Helper Decorators
# =============================================================================


def async_view(view_func: Callable) -> Callable:
    """
    Decorator to run async views in Django's sync context.

    Usage:
        @async_view
        async def my_view(request):
            result = await some_async_operation()
            return JsonResponse(result)
    """

    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        return async_to_sync(view_func)(request, *args, **kwargs)

    return wrapper


# =============================================================================
# API Views
# =============================================================================


if DJANGO_AVAILABLE:

    @csrf_exempt
    @require_http_methods(["POST"])
    def chat_with_flow(
        request: HttpRequest,
        tenant_id: str,
        flow_id: str,
    ) -> StreamingHttpResponse:
        """
        Chat endpoint - streams events via Server-Sent Events (SSE).

        POST /api/flows/{tenant_id}/{flow_id}/chat

        Request Body:
            {
                "session_id": "unique-session-id",
                "message": "User's message"
            }

        Response:
            Server-Sent Events stream with ADK events
        """
        try:
            body = json.loads(request.body)
            session_id = body.get("session_id", "default")
            message = body.get("message", "")
            user_id = str(getattr(request.user, "id", "anonymous"))

            if not message:
                return JsonResponse({"error": "Message is required"}, status=400)

            executor = get_executor()

            def event_stream():
                """Generate SSE events."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    async_gen = executor.run_flow(
                        tenant_id=tenant_id,
                        flow_id=flow_id,
                        user_id=user_id,
                        session_id=session_id,
                        message=message,
                    )

                    while True:
                        try:
                            event = loop.run_until_complete(async_gen.__anext__())
                            yield f"data: {event.model_dump_json()}\n\n"
                        except StopAsyncIteration:
                            break
                        except Exception as e:
                            logger.error(f"Error in event stream: {e}")
                            yield f"data: {json.dumps({'error': str(e)})}\n\n"
                            break

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
            logger.error(f"Error in chat_with_flow: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    @csrf_exempt
    @require_http_methods(["POST"])
    @async_view
    async def chat_with_flow_simple(
        request: HttpRequest,
        tenant_id: str,
        flow_id: str,
    ) -> JsonResponse:
        """
        Simple chat endpoint - returns final response (non-streaming).

        POST /api/flows/{tenant_id}/{flow_id}/chat/simple

        Request Body:
            {
                "session_id": "unique-session-id",
                "message": "User's message"
            }

        Response:
            {
                "response": "Agent's response text"
            }
        """
        try:
            body = json.loads(request.body)
            session_id = body.get("session_id", "default")
            message = body.get("message", "")
            user_id = str(getattr(request.user, "id", "anonymous"))

            if not message:
                return JsonResponse({"error": "Message is required"}, status=400)

            executor = get_executor()

            response = await executor.run_flow_sync(
                tenant_id=tenant_id,
                flow_id=flow_id,
                user_id=user_id,
                session_id=session_id,
                message=message,
            )

            return JsonResponse({"response": response})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except ValueError as e:
            return JsonResponse({"error": str(e)}, status=404)
        except Exception as e:
            logger.error(f"Error in chat_with_flow_simple: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    @csrf_exempt
    @require_http_methods(["POST", "PUT"])
    @async_view
    async def save_flow(
        request: HttpRequest,
        tenant_id: str,
        flow_id: str,
    ) -> JsonResponse:
        """
        Save/update a flow.

        POST/PUT /api/flows/{tenant_id}/{flow_id}

        Request Body:
            Flow JSON definition

        Response:
            {
                "status": "saved",
                "flow_id": "...",
                "validation_errors": []
            }
        """
        try:
            flow_json = json.loads(request.body)

            # Validate with Pydantic
            flow = FlowSchema(
                tenant_id=tenant_id,
                flow_id=flow_id,
                **flow_json,
            )

            # Validate flow structure
            validation_errors = flow.validate_flow()
            if validation_errors:
                return JsonResponse(
                    {
                        "status": "error",
                        "validation_errors": validation_errors,
                    },
                    status=400,
                )

            # TODO: Save to your database
            # Example:
            #   await sync_to_async(Flow.objects.update_or_create)(
            #       tenant_id=tenant_id,
            #       id=flow_id,
            #       defaults={"flow_json": flow.model_dump()},
            #   )

            # Invalidate cache so next request uses new version
            executor = get_executor()
            await executor.invalidate_flow_cache(tenant_id, flow_id)

            return JsonResponse(
                {
                    "status": "saved",
                    "flow_id": flow_id,
                    "validation_errors": [],
                }
            )

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error(f"Error in save_flow: {e}")
            return JsonResponse({"error": str(e)}, status=400)

    @csrf_exempt
    @require_http_methods(["DELETE"])
    @async_view
    async def delete_flow(
        request: HttpRequest,
        tenant_id: str,
        flow_id: str,
    ) -> JsonResponse:
        """
        Delete a flow.

        DELETE /api/flows/{tenant_id}/{flow_id}
        """
        try:
            # TODO: Delete from your database
            # Example:
            #   await sync_to_async(
            #       Flow.objects.filter(tenant_id=tenant_id, id=flow_id).delete
            #   )()

            # Invalidate cache
            executor = get_executor()
            await executor.invalidate_flow_cache(tenant_id, flow_id)

            return JsonResponse({"status": "deleted", "flow_id": flow_id})

        except Exception as e:
            logger.error(f"Error in delete_flow: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    @require_http_methods(["GET"])
    def health_check(request: HttpRequest) -> JsonResponse:
        """
        Health check endpoint.

        GET /api/health
        """
        try:
            executor = get_executor()
            cache_stats = executor.get_cache_stats()

            return JsonResponse(
                {
                    "status": "healthy",
                    "cache": cache_stats,
                }
            )

        except Exception as e:
            return JsonResponse(
                {
                    "status": "unhealthy",
                    "error": str(e),
                },
                status=500,
            )

    @require_http_methods(["POST"])
    @async_view
    async def invalidate_cache(
        request: HttpRequest,
        tenant_id: str,
    ) -> JsonResponse:
        """
        Invalidate all cached flows for a tenant.

        POST /api/flows/{tenant_id}/invalidate-cache
        """
        try:
            executor = get_executor()
            count = await executor.invalidate_tenant_cache(tenant_id)

            return JsonResponse(
                {
                    "status": "invalidated",
                    "count": count,
                }
            )

        except Exception as e:
            logger.error(f"Error in invalidate_cache: {e}")
            return JsonResponse({"error": str(e)}, status=500)


# =============================================================================
# URL Patterns - Add to your urls.py
# =============================================================================

"""
Example urls.py:

from django.urls import path
from adk_integration.api.views import (
    chat_with_flow,
    chat_with_flow_simple,
    save_flow,
    delete_flow,
    health_check,
    invalidate_cache,
)

urlpatterns = [
    # Chat endpoints
    path(
        'api/flows/<str:tenant_id>/<str:flow_id>/chat',
        chat_with_flow,
        name='chat_with_flow'
    ),
    path(
        'api/flows/<str:tenant_id>/<str:flow_id>/chat/simple',
        chat_with_flow_simple,
        name='chat_with_flow_simple'
    ),

    # Flow management
    path(
        'api/flows/<str:tenant_id>/<str:flow_id>',
        save_flow,
        name='save_flow'
    ),
    path(
        'api/flows/<str:tenant_id>/<str:flow_id>/delete',
        delete_flow,
        name='delete_flow'
    ),

    # Cache management
    path(
        'api/flows/<str:tenant_id>/invalidate-cache',
        invalidate_cache,
        name='invalidate_cache'
    ),

    # Health
    path('api/health', health_check, name='health_check'),
]
"""
