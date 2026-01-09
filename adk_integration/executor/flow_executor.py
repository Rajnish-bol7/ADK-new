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
Flow Executor for running flows and managing sessions.

This module provides the main execution engine for running flows with
proper session management, caching, and multi-tenant support.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)

from google.adk import Agent, Runner
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types

from ..schema.flow_schema import FlowSchema
from ..builder.flow_builder import FlowBuilder, FlowBuildResult

logger = logging.getLogger(__name__)


# Type aliases for callback functions
GetFlowCallback = Callable[[str, str], Awaitable[Optional[FlowSchema]]]
GetApiKeysCallback = Callable[[str], Awaitable[Dict[str, str]]]
SessionServiceFactory = Callable[[str], Any]


class RunnerCache:
    """
    Caches Runner instances per tenant/flow combination.

    This avoids rebuilding agents on every request, significantly improving
    performance for frequently-used flows.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the runner cache.

        Args:
            max_size: Maximum number of runners to cache
            ttl_seconds: Time-to-live for cached runners
        """
        self._cache: Dict[str, Tuple[Runner, FlowBuildResult, str, float]] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()

    def _make_key(self, tenant_id: str, flow_id: str) -> str:
        """Create a cache key from tenant and flow IDs."""
        return f"{tenant_id}:{flow_id}"

    def _make_version_hash(self, flow: FlowSchema) -> str:
        """Create a hash to detect flow changes."""
        flow_str = json.dumps(flow.model_dump(), sort_keys=True)
        return hashlib.md5(flow_str.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry is expired."""
        return (datetime.now().timestamp() - timestamp) > self._ttl_seconds

    async def get_or_create(
        self,
        tenant_id: str,
        flow_id: str,
        flow: FlowSchema,
        api_keys: Dict[str, str],
        session_service: Any,
    ) -> Tuple[Runner, FlowBuildResult]:
        """
        Get cached runner or create new one.

        Args:
            tenant_id: Tenant identifier
            flow_id: Flow identifier
            flow: Flow schema
            api_keys: API keys for the tenant
            session_service: Session service instance

        Returns:
            Tuple of (Runner, FlowBuildResult)
        """
        key = self._make_key(tenant_id, flow_id)
        version_hash = self._make_version_hash(flow)
        now = datetime.now().timestamp()

        async with self._lock:
            if key in self._cache:
                runner, build_result, cached_hash, timestamp = self._cache[key]

                # Check if still valid (same version and not expired)
                if cached_hash == version_hash and not self._is_expired(timestamp):
                    logger.debug(f"Cache hit for flow {flow_id}")
                    return runner, build_result

                # Flow changed or expired, cleanup old and rebuild
                logger.info(f"Cache invalidated for flow {flow_id}")
                await build_result.cleanup()
                del self._cache[key]

            # Build new runner
            logger.info(f"Building new runner for flow {flow_id}")
            builder = FlowBuilder(tenant_id, api_keys)
            build_result = await builder.build(flow)

            runner = Runner(
                agent=build_result.root_agent,
                app_name=f"{tenant_id}_{flow_id}",
                session_service=session_service,
            )

            # Evict oldest entries if cache is full
            if len(self._cache) >= self._max_size:
                await self._evict_oldest()

            # Cache the new runner
            self._cache[key] = (runner, build_result, version_hash, now)

            return runner, build_result

    async def _evict_oldest(self) -> None:
        """Evict the oldest cache entry."""
        if not self._cache:
            return

        # Find oldest entry
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k][3],  # timestamp
        )

        # Cleanup and remove
        _, build_result, _, _ = self._cache.pop(oldest_key)
        await build_result.cleanup()
        logger.debug(f"Evicted oldest cache entry: {oldest_key}")

    async def invalidate(self, tenant_id: str, flow_id: str) -> bool:
        """
        Remove a flow from cache.

        Call this when a flow is updated.

        Args:
            tenant_id: Tenant identifier
            flow_id: Flow identifier

        Returns:
            True if entry was found and removed
        """
        key = self._make_key(tenant_id, flow_id)

        async with self._lock:
            if key in self._cache:
                _, build_result, _, _ = self._cache.pop(key)
                await build_result.cleanup()
                logger.info(f"Invalidated cache for flow {flow_id}")
                return True

        return False

    async def invalidate_tenant(self, tenant_id: str) -> int:
        """
        Remove all flows for a tenant from cache.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Number of entries removed
        """
        prefix = f"{tenant_id}:"
        count = 0

        async with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]

            for key in keys_to_remove:
                _, build_result, _, _ = self._cache.pop(key)
                await build_result.cleanup()
                count += 1

        if count > 0:
            logger.info(f"Invalidated {count} cache entries for tenant {tenant_id}")

        return count

    async def cleanup_all(self) -> None:
        """Cleanup all cached runners."""
        async with self._lock:
            for key, (_, build_result, _, _) in self._cache.items():
                try:
                    await build_result.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up {key}: {e}")

            self._cache.clear()
            logger.info("Cleared all runner cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "entries": list(self._cache.keys()),
        }


class FlowExecutor:
    """
    Main executor for running flows.

    This class provides the interface between your Django backend and the
    ADK agent system. It handles:
    - Loading flows from your database
    - Building agents from flow definitions
    - Managing sessions per user/conversation
    - Caching runners for performance
    - Streaming events back to clients

    Example:
        executor = FlowExecutor(
            get_flow_callback=get_flow_from_db,
            get_api_keys_callback=get_tenant_api_keys,
            session_service_factory=create_session_service,
        )

        async for event in executor.run_flow(
            tenant_id="tenant_123",
            flow_id="flow_456",
            user_id="user_789",
            session_id="session_abc",
            message="Hello!",
        ):
            # Send event to client
            send_to_client(event)
    """

    def __init__(
        self,
        get_flow_callback: GetFlowCallback,
        get_api_keys_callback: GetApiKeysCallback,
        session_service_factory: Optional[SessionServiceFactory] = None,
        cache_max_size: int = 1000,
        cache_ttl_seconds: int = 3600,
    ):
        """
        Initialize the executor.

        Args:
            get_flow_callback: Async function to get flow from database.
                Signature: async (tenant_id, flow_id) -> FlowSchema | None
            get_api_keys_callback: Async function to get tenant's API keys.
                Signature: async (tenant_id) -> Dict[str, str]
            session_service_factory: Optional function to create session service.
                Signature: (tenant_id) -> SessionService
                If not provided, uses InMemorySessionService.
            cache_max_size: Maximum number of runners to cache
            cache_ttl_seconds: Time-to-live for cached runners
        """
        self.get_flow = get_flow_callback
        self.get_api_keys = get_api_keys_callback
        self.session_service_factory = (
            session_service_factory or self._default_session_factory
        )
        self.runner_cache = RunnerCache(
            max_size=cache_max_size,
            ttl_seconds=cache_ttl_seconds,
        )
        # Cache session services per tenant to ensure sessions are shared
        self._session_service_cache: Dict[str, Any] = {}

    def _default_session_factory(self, tenant_id: str) -> InMemorySessionService:
        """Default session service factory using in-memory storage."""
        return InMemorySessionService()

    def _get_session_service(self, tenant_id: str) -> Any:
        """
        Get or create a session service for a tenant (cached).
        
        This ensures that all runners for the same tenant share the same
        session service instance, allowing sessions to be properly shared
        across different flow executions.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Session service instance for the tenant
        """
        if tenant_id not in self._session_service_cache:
            self._session_service_cache[tenant_id] = self.session_service_factory(tenant_id)
        return self._session_service_cache[tenant_id]

    async def run_flow(
        self,
        tenant_id: str,
        flow_id: str,
        user_id: str,
        session_id: str,
        message: str,
    ) -> AsyncGenerator[Event, None]:
        """
        Execute a flow for a user message.

        This is the main method for running a flow. It:
        1. Loads the flow definition from your database
        2. Builds or retrieves cached agents
        3. Runs the agent and streams events

        Args:
            tenant_id: Tenant identifier
            flow_id: Flow identifier
            user_id: User identifier
            session_id: Session/conversation identifier
            message: User's message

        Yields:
            ADK Event objects

        Raises:
            ValueError: If flow not found or not active
        """
        # Get flow definition
        flow = await self.get_flow(tenant_id, flow_id)
        if not flow:
            raise ValueError(f"Flow not found: {tenant_id}/{flow_id}")

        if not flow.is_active:
            raise ValueError(f"Flow is not active: {flow_id}")

        # Get tenant's API keys
        api_keys = await self.get_api_keys(tenant_id)
        if not api_keys:
            raise ValueError(f"No API keys found for tenant: {tenant_id}")

        # Get session service for this tenant (cached to ensure session sharing)
        session_service = self._get_session_service(tenant_id)

        # Get or create runner
        runner, build_result = await self.runner_cache.get_or_create(
            tenant_id=tenant_id,
            flow_id=flow_id,
            flow=flow,
            api_keys=api_keys,
            session_service=session_service,
        )

        # Ensure session exists (create if not)
        existing_session = await session_service.get_session(
            app_name=f"{tenant_id}_{flow_id}",
            user_id=user_id,
            session_id=session_id,
        )
        if not existing_session:
            await session_service.create_session(
                app_name=f"{tenant_id}_{flow_id}",
                user_id=user_id,
                session_id=session_id,
            )
            logger.debug(f"Created new session: {session_id}")

        # Run the agent
        logger.info(f"Running flow {flow_id} for user {user_id}, session {session_id}")

        # Convert string message to Content object
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)],
        )

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            yield event

    async def run_flow_sync(
        self,
        tenant_id: str,
        flow_id: str,
        user_id: str,
        session_id: str,
        message: str,
    ) -> str:
        """
        Execute a flow and return the final text response.

        This is a convenience method for simple integrations that don't
        need streaming.

        Args:
            tenant_id: Tenant identifier
            flow_id: Flow identifier
            user_id: User identifier
            session_id: Session identifier
            message: User's message

        Returns:
            The final text response from the agent
        """
        final_response = ""

        async for event in self.run_flow(
            tenant_id=tenant_id,
            flow_id=flow_id,
            user_id=user_id,
            session_id=session_id,
            message=message,
        ):
            # Collect text parts from model responses
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_response = part.text

        return final_response

    async def run_flow_live(
        self,
        tenant_id: str,
        flow_id: str,
        user_id: str,
        session_id: str,
        audio_stream: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[bytes, None]:
        """
        Execute a flow with live audio streaming.

        This method handles bidirectional audio streaming for voice calls.

        Args:
            tenant_id: Tenant identifier
            flow_id: Flow identifier
            user_id: User identifier
            session_id: Session identifier
            audio_stream: Async generator yielding audio bytes

        Yields:
            Audio bytes from the agent's response

        Note:
            This requires a flow with a call_start trigger and a model
            that supports live audio (e.g., gemini-2.5-flash-native-audio-latest)
        """
        # Get flow and verify it's a live flow
        flow = await self.get_flow(tenant_id, flow_id)
        if not flow:
            raise ValueError(f"Flow not found: {tenant_id}/{flow_id}")

        # Get API keys
        api_keys = await self.get_api_keys(tenant_id)
        # Get session service for this tenant (cached to ensure session sharing)
        session_service = self._get_session_service(tenant_id)

        # Get runner
        runner, build_result = await self.runner_cache.get_or_create(
            tenant_id=tenant_id,
            flow_id=flow_id,
            flow=flow,
            api_keys=api_keys,
            session_service=session_service,
        )

        if not build_result.is_live:
            raise ValueError(f"Flow {flow_id} does not support live streaming")

        # Use run_live for bidirectional streaming
        # Note: This requires proper live streaming setup
        async for event in runner.run_live(
            session_id=session_id,
            live_request_queue=audio_stream,  # type: ignore
        ):
            # Extract audio from events
            if hasattr(event, "audio") and event.audio:
                yield event.audio

    async def invalidate_flow_cache(
        self,
        tenant_id: str,
        flow_id: str,
    ) -> bool:
        """
        Invalidate cached runner when flow is updated.

        Call this from your Django save endpoint whenever a flow is modified.

        Args:
            tenant_id: Tenant identifier
            flow_id: Flow identifier

        Returns:
            True if cache entry was found and removed
        """
        return await self.runner_cache.invalidate(tenant_id, flow_id)

    async def invalidate_tenant_cache(self, tenant_id: str) -> int:
        """
        Invalidate all cached runners for a tenant.

        Call this when tenant settings change (e.g., API keys updated).

        Args:
            tenant_id: Tenant identifier

        Returns:
            Number of cache entries removed
        """
        return await self.runner_cache.invalidate_tenant(tenant_id)

    async def shutdown(self) -> None:
        """Cleanup all resources on shutdown."""
        await self.runner_cache.cleanup_all()
        logger.info("FlowExecutor shutdown complete")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return self.runner_cache.get_stats()


# Singleton instance for global access
_executor_instance: Optional[FlowExecutor] = None


def get_executor() -> Optional[FlowExecutor]:
    """Get the global executor instance."""
    return _executor_instance


def init_executor(
    get_flow_callback: GetFlowCallback,
    get_api_keys_callback: GetApiKeysCallback,
    session_service_factory: Optional[SessionServiceFactory] = None,
    **kwargs,
) -> FlowExecutor:
    """
    Initialize the global executor instance.

    Call this once at application startup.

    Args:
        get_flow_callback: Function to get flow from database
        get_api_keys_callback: Function to get tenant API keys
        session_service_factory: Optional session service factory
        **kwargs: Additional arguments for FlowExecutor

    Returns:
        The initialized FlowExecutor instance
    """
    global _executor_instance

    _executor_instance = FlowExecutor(
        get_flow_callback=get_flow_callback,
        get_api_keys_callback=get_api_keys_callback,
        session_service_factory=session_service_factory,
        **kwargs,
    )

    return _executor_instance
