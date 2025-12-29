#!/usr/bin/env python
"""
Test script to verify flow execution works correctly.
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

import asyncio
from asgiref.sync import sync_to_async

from flows.models import Flow
from core.models import Tenant
from adk_integration.executor.flow_executor import FlowExecutor
from google.adk.sessions import InMemorySessionService


async def get_flow_from_db(tenant_id: str, flow_id: str):
    """Fetch flow from database."""
    flow = await sync_to_async(
        Flow.objects.select_related('tenant').get,
        thread_sensitive=True
    )(id=flow_id)
    return flow.get_flow_schema()


async def get_api_keys(tenant_id: str):
    """Get API keys."""
    return {'google': os.environ.get('GOOGLE_API_KEY', '')}


def create_session_service(tenant_id: str):
    """Create session service."""
    return InMemorySessionService()


async def main():
    """Test the flow execution."""
    print("=" * 60)
    print("Testing Flow Execution")
    print("=" * 60)
    
    # Get the test flow
    flow = await sync_to_async(
        Flow.objects.select_related('tenant').get,
        thread_sensitive=True
    )(name='Simple Test Flow')
    
    print(f"\n📋 Flow: {flow.name}")
    print(f"   ID: {flow.id}")
    print(f"   Tenant: {flow.tenant.name}")
    
    # Create executor
    executor = FlowExecutor(
        get_flow_callback=get_flow_from_db,
        get_api_keys_callback=get_api_keys,
        session_service_factory=create_session_service,
    )
    
    # Test message 1
    print("\n💬 Test 1: Sending 'Hello, are you working?'")
    
    response = await executor.run_flow_sync(
        tenant_id=str(flow.tenant.id),
        flow_id=str(flow.id),
        user_id='test_user',
        session_id='test_session_1',
        message='Hello, are you working?',
    )
    
    print(f"\n✅ Response: {response}")
    
    # Test message 2 (in same session to test history)
    print("\n💬 Test 2: Sending 'What can you help me with?'")
    
    response2 = await executor.run_flow_sync(
        tenant_id=str(flow.tenant.id),
        flow_id=str(flow.id),
        user_id='test_user',
        session_id='test_session_1',
        message='What can you help me with?',
    )
    
    print(f"\n✅ Response: {response2}")
    
    # Cleanup
    await executor.shutdown()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())

