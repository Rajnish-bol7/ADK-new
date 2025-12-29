"""
WebSocket consumers for real-time chat and voice streaming.

This module provides WebSocket support for:
1. Real-time text chat with streaming responses
2. Voice/audio streaming with Gemini Live models
"""

import json
import logging
import asyncio
from typing import Optional
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async

from flows.models import Flow, Session, Message
from adk_integration.executor.flow_executor import FlowExecutor
from google.adk.sessions import InMemorySessionService

logger = logging.getLogger(__name__)


# Global executor instance
_ws_executor: Optional[FlowExecutor] = None


async def get_ws_executor() -> FlowExecutor:
    """Get or create the WebSocket executor."""
    global _ws_executor
    
    if _ws_executor is None:
        _ws_executor = FlowExecutor(
            get_flow_callback=_get_flow,
            get_api_keys_callback=_get_api_keys,
            session_service_factory=_create_session_service,
        )
    
    return _ws_executor


async def _get_flow(tenant_id: str, flow_id: str):
    """Fetch flow from database."""
    import os
    flow = await sync_to_async(
        Flow.objects.select_related('tenant').get,
        thread_sensitive=True
    )(id=flow_id, is_active=True)
    return flow.get_flow_schema()


async def _get_api_keys(tenant_id: str):
    """Get API keys for tenant."""
    import os
    from core.models import Tenant
    
    try:
        tenant = await sync_to_async(Tenant.objects.get)(id=tenant_id)
        keys = tenant.get_api_keys()
        
        # Fallback to environment variables
        if not keys.get('google'):
            keys['google'] = os.environ.get('GOOGLE_API_KEY', '')
        if not keys.get('openai'):
            keys['openai'] = os.environ.get('OPENAI_API_KEY', '')
            
        return keys
    except Exception:
        return {
            'google': os.environ.get('GOOGLE_API_KEY', ''),
            'openai': os.environ.get('OPENAI_API_KEY', ''),
        }


def _create_session_service(tenant_id: str):
    """Create session service."""
    return InMemorySessionService()


class ChatConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time text chat with streaming responses.
    
    Connect: ws://localhost:8000/ws/chat/{flow_id}/
    
    Send message:
    {
        "type": "message",
        "content": "Hello!",
        "session_id": "optional-session-id"
    }
    
    Receive events:
    {
        "type": "text",
        "content": "Hello! How can I help?",
        "is_final": true
    }
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_id: Optional[str] = None
        self.tenant_id: Optional[str] = None
        self.user_id: str = "anonymous"
        self.session_id: Optional[str] = None
        self.flow: Optional[Flow] = None
    
    async def connect(self):
        """Handle WebSocket connection."""
        self.flow_id = self.scope['url_route']['kwargs']['flow_id']
        
        # Get user from scope (if authenticated)
        user = self.scope.get('user')
        if user and user.is_authenticated:
            self.user_id = str(user.id)
            if hasattr(user, 'tenant') and user.tenant:
                self.tenant_id = str(user.tenant.id)
        
        # Verify flow exists and get tenant
        try:
            self.flow = await sync_to_async(
                Flow.objects.select_related('tenant').get,
                thread_sensitive=True
            )(id=self.flow_id, is_active=True)
            
            if not self.tenant_id:
                self.tenant_id = str(self.flow.tenant.id)
            
            await self.accept()
            
            # Send connection confirmation
            await self.send(json.dumps({
                "type": "connected",
                "flow_id": self.flow_id,
                "flow_name": self.flow.name,
            }))
            
            logger.info(f"WebSocket connected: flow={self.flow_id}, user={self.user_id}")
            
        except Flow.DoesNotExist:
            await self.close(code=4004)
            logger.warning(f"Flow not found: {self.flow_id}")
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        logger.info(f"WebSocket disconnected: flow={self.flow_id}, code={close_code}")
    
    async def receive(self, text_data):
        """Handle incoming messages."""
        try:
            data = json.loads(text_data)
            msg_type = data.get('type', 'message')
            
            if msg_type == 'message':
                await self.handle_message(data)
            elif msg_type == 'ping':
                await self.send(json.dumps({"type": "pong"}))
            else:
                await self.send(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                }))
                
        except json.JSONDecodeError:
            await self.send(json.dumps({
                "type": "error",
                "message": "Invalid JSON"
            }))
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def handle_message(self, data: dict):
        """Process a chat message and stream the response."""
        content = data.get('content', '').strip()
        if not content:
            await self.send(json.dumps({
                "type": "error",
                "message": "Empty message"
            }))
            return
        
        # Use provided session_id or create one
        self.session_id = data.get('session_id') or self.session_id or f"ws_{self.flow_id}_{self.user_id}"
        
        # Send acknowledgment
        await self.send(json.dumps({
            "type": "ack",
            "session_id": self.session_id,
        }))
        
        # Get executor and run flow
        executor = await get_ws_executor()
        
        try:
            # Stream events
            full_response = ""
            async for event in executor.run_flow(
                tenant_id=self.tenant_id,
                flow_id=self.flow_id,
                user_id=self.user_id,
                session_id=self.session_id,
                message=content,
            ):
                # Extract text from event
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            full_response = part.text
                            
                            # Send streaming update
                            await self.send(json.dumps({
                                "type": "text",
                                "content": part.text,
                                "is_final": False,
                            }))
                
                # Check for function calls
                if hasattr(event, 'tool_calls') and event.tool_calls:
                    for tool_call in event.tool_calls:
                        await self.send(json.dumps({
                            "type": "tool_call",
                            "name": getattr(tool_call, 'name', 'unknown'),
                        }))
            
            # Send final message
            await self.send(json.dumps({
                "type": "text",
                "content": full_response,
                "is_final": True,
            }))
            
            # Log messages to database
            await self._log_messages(content, full_response)
            
        except Exception as e:
            logger.error(f"Error running flow: {e}")
            await self.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def _log_messages(self, user_message: str, assistant_response: str):
        """Log messages to database."""
        try:
            # Get or create session
            session, _ = await sync_to_async(Session.objects.get_or_create)(
                id=self.session_id,
                defaults={
                    'flow': self.flow,
                }
            )
            
            # Log user message
            await sync_to_async(Message.objects.create)(
                session=session,
                role='user',
                content=user_message,
            )
            
            # Log assistant response
            if assistant_response:
                await sync_to_async(Message.objects.create)(
                    session=session,
                    role='assistant',
                    content=assistant_response,
                )
                
        except Exception as e:
            logger.warning(f"Failed to log messages: {e}")


class VoiceConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time voice/audio streaming.
    
    Connect: ws://localhost:8000/ws/voice/{flow_id}/
    
    Send audio:
    Binary audio data (PCM, 16-bit, 16kHz mono)
    
    Or control messages:
    {
        "type": "start",
        "session_id": "optional-session-id",
        "audio_format": "pcm16"
    }
    {
        "type": "stop"
    }
    
    Receive:
    Binary audio data (agent's voice response)
    Or status messages as JSON
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_id: Optional[str] = None
        self.tenant_id: Optional[str] = None
        self.user_id: str = "anonymous"
        self.session_id: Optional[str] = None
        self.is_streaming: bool = False
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.stream_task: Optional[asyncio.Task] = None
    
    async def connect(self):
        """Handle WebSocket connection for voice."""
        self.flow_id = self.scope['url_route']['kwargs']['flow_id']
        
        # Get user from scope
        user = self.scope.get('user')
        if user and user.is_authenticated:
            self.user_id = str(user.id)
            if hasattr(user, 'tenant') and user.tenant:
                self.tenant_id = str(user.tenant.id)
        
        # Verify flow exists
        try:
            flow = await sync_to_async(
                Flow.objects.select_related('tenant').get,
                thread_sensitive=True
            )(id=self.flow_id, is_active=True)
            
            if not self.tenant_id:
                self.tenant_id = str(flow.tenant.id)
            
            # Accept connection (allow testing voice for any flow type)
            # In production, you may want to restrict to flow.trigger_type == 'call'
            await self.accept()
            
            await self.send(json.dumps({
                "type": "connected",
                "flow_id": self.flow_id,
                "voice_enabled": True,
            }))
            
            logger.info(f"Voice WebSocket connected: flow={self.flow_id}")
            
        except Flow.DoesNotExist:
            await self.close(code=4004)
    
    async def disconnect(self, close_code):
        """Handle disconnection."""
        self.is_streaming = False
        
        if self.stream_task:
            self.stream_task.cancel()
            
        logger.info(f"Voice WebSocket disconnected: flow={self.flow_id}")
    
    async def receive(self, text_data=None, bytes_data=None):
        """Handle incoming data (text control messages or binary audio)."""
        if text_data:
            # Control message
            try:
                data = json.loads(text_data)
                msg_type = data.get('type')
                
                if msg_type == 'start':
                    await self.start_streaming(data)
                elif msg_type == 'stop':
                    await self.stop_streaming()
                elif msg_type == 'ping':
                    await self.send(json.dumps({"type": "pong"}))
                    
            except json.JSONDecodeError:
                await self.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON"
                }))
                
        elif bytes_data:
            # Audio data
            if self.is_streaming:
                await self.audio_queue.put(bytes_data)
    
    async def start_streaming(self, data: dict):
        """Start voice streaming session."""
        self.session_id = data.get('session_id') or f"voice_{self.flow_id}_{self.user_id}"
        self.is_streaming = True
        
        await self.send(json.dumps({
            "type": "streaming_started",
            "session_id": self.session_id,
        }))
        
        # Start the streaming task
        self.stream_task = asyncio.create_task(self.process_audio_stream())
    
    async def stop_streaming(self):
        """Stop voice streaming session."""
        self.is_streaming = False
        
        if self.stream_task:
            self.stream_task.cancel()
            
        await self.send(json.dumps({
            "type": "streaming_stopped",
        }))
    
    async def process_audio_stream(self):
        """Process incoming audio and send responses."""
        try:
            # Note: Full implementation would use ADK's run_live method
            # with Gemini Live models. This is a placeholder structure.
            
            while self.is_streaming:
                try:
                    # Get audio chunk with timeout
                    audio_chunk = await asyncio.wait_for(
                        self.audio_queue.get(),
                        timeout=30.0
                    )
                    
                    # TODO: Process audio through Gemini Live
                    # For now, just acknowledge receipt
                    await self.send(json.dumps({
                        "type": "audio_received",
                        "bytes": len(audio_chunk),
                    }))
                    
                except asyncio.TimeoutError:
                    # Send keepalive
                    await self.send(json.dumps({"type": "keepalive"}))
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in audio stream: {e}")
            await self.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))


class LiveStreamConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for Gemini Live bidirectional streaming.
    
    This consumer handles real-time audio input/output using
    Gemini's native live streaming capabilities.
    
    Connect: ws://localhost:8000/ws/live/{flow_id}/
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_id: Optional[str] = None
        self.tenant_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.live_session = None
    
    async def connect(self):
        """Handle connection."""
        self.flow_id = self.scope['url_route']['kwargs']['flow_id']
        
        try:
            flow = await sync_to_async(
                Flow.objects.select_related('tenant').get,
                thread_sensitive=True
            )(id=self.flow_id, is_active=True)
            
            self.tenant_id = str(flow.tenant.id)
            await self.accept()
            
            await self.send(json.dumps({
                "type": "connected",
                "flow_id": self.flow_id,
                "live_streaming": True,
            }))
            
        except Flow.DoesNotExist:
            await self.close(code=4004)
    
    async def disconnect(self, close_code):
        """Handle disconnection."""
        if self.live_session:
            # Close Gemini live session
            pass
    
    async def receive(self, text_data=None, bytes_data=None):
        """Handle incoming data."""
        if text_data:
            data = json.loads(text_data)
            
            if data.get('type') == 'start_live':
                await self.start_live_session(data)
            elif data.get('type') == 'end_live':
                await self.end_live_session()
                
        elif bytes_data:
            # Forward audio to Gemini Live
            if self.live_session:
                # Send audio to live session
                pass
    
    async def start_live_session(self, data: dict):
        """Initialize Gemini Live session."""
        self.session_id = data.get('session_id', f"live_{self.flow_id}")
        
        # TODO: Initialize Gemini Live connection
        # This would use google.adk's live streaming capabilities
        
        await self.send(json.dumps({
            "type": "live_started",
            "session_id": self.session_id,
        }))
    
    async def end_live_session(self):
        """End Gemini Live session."""
        if self.live_session:
            self.live_session = None
            
        await self.send(json.dumps({
            "type": "live_ended",
        }))

