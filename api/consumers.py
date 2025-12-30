"""
WebSocket consumers for real-time chat and voice streaming.

This module provides WebSocket support for:
1. Real-time text chat with streaming responses
2. Voice/audio streaming with Gemini Live models
"""

import json
import logging
import asyncio
import time
from typing import Optional
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from pydantic import ValidationError

from flows.models import Flow, Session, Message
from adk_integration.executor.flow_executor import FlowExecutor
from google.adk.sessions import InMemorySessionService
from google.adk.agents.live_request_queue import LiveRequest, LiveRequestQueue
from google.genai import types as genai_types
from contextlib import aclosing as Aclosing

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
    WebSocket consumer for real-time voice/audio streaming using Gemini Live.
    
    Based on ADK's run_live pattern for bidirectional audio streaming.
    
    Connect: ws://localhost:8000/ws/voice/{flow_id}/
    
    Send control messages (JSON):
    {
        "type": "content",
        "text": "Hello"  // Optional text message
    }
    {
        "type": "activity_start"  // Signal start of user speaking
    }
    {
        "type": "activity_end"    // Signal end of user speaking
    }
    
    Send audio:
    Binary audio data (will be converted to Blob and sent to Gemini Live)
    
    Receive:
    JSON events from ADK (model responses, transcriptions, etc.)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_id: Optional[str] = None
        self.tenant_id: Optional[str] = None
        self.user_id: str = "anonymous"
        self.session_id: Optional[str] = None
        self.live_request_queue: Optional[LiveRequestQueue] = None
        self.forward_task: Optional[asyncio.Task] = None
        self.process_task: Optional[asyncio.Task] = None
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_queue: asyncio.Queue = asyncio.Queue()
    
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
            
            await self.accept()
            
            # Create session ID
            self.session_id = f"voice_{self.flow_id}_{self.user_id}_{int(time.time())}"
            
            # Initialize LiveRequestQueue for bidirectional streaming
            self.live_request_queue = LiveRequestQueue()
            
            await self.send(json.dumps({
                "type": "connected",
                "flow_id": self.flow_id,
                "session_id": self.session_id,
                "voice_enabled": True,
            }))
            
            logger.info(f"Voice WebSocket connected: flow={self.flow_id}, session={self.session_id}")
            
            # Start the live streaming session
            await self.start_live_streaming()
            
        except Flow.DoesNotExist:
            await self.close(code=4004)
            logger.warning(f"Flow not found for voice: {self.flow_id}")
        except Exception as e:
            logger.error(f"Error connecting voice WebSocket: {e}")
            await self.close(code=4000)
    
    async def start_live_streaming(self):
        """Start the live streaming session with ADK using run_live."""
        executor = await get_ws_executor()
        
        # Create forward_events task (streams events from ADK to WebSocket)
        async def forward_events():
            try:
                # Get flow to verify it exists
                flow = await sync_to_async(
                    Flow.objects.select_related('tenant').get,
                    thread_sensitive=True
                )(id=self.flow_id, is_active=True)
                flow_schema = flow.get_flow_schema()
                
                # Get API keys
                from core.models import Tenant
                tenant = await sync_to_async(Tenant.objects.get)(id=self.tenant_id)
                api_keys = tenant.get_api_keys()
                import os
                if not api_keys.get('google'):
                    api_keys['google'] = os.environ.get('GOOGLE_API_KEY', '')
                
                # Get session service
                session_service = _create_session_service(self.tenant_id)
                
                # Get runner from cache
                runner, build_result = await executor.runner_cache.get_or_create(
                    tenant_id=self.tenant_id,
                    flow_id=self.flow_id,
                    flow=flow_schema,
                    api_keys=api_keys,
                    session_service=session_service,
                )
                
                # Ensure session exists before calling run_live
                # IMPORTANT: Use the runner's session_service instance (not a new one)
                # This ensures the session is visible to the Runner
                app_name = runner.app_name
                runner_session_service = runner.session_service
                existing_session = await runner_session_service.get_session(
                    app_name=app_name,
                    user_id=self.user_id,
                    session_id=self.session_id,
                )
                if not existing_session:
                    # Create session if it doesn't exist using the runner's session service
                    await runner_session_service.create_session(
                        app_name=app_name,
                        user_id=self.user_id,
                        session_id=self.session_id,
                    )
                    # Verify it was created
                    existing_session = await runner_session_service.get_session(
                        app_name=app_name,
                        user_id=self.user_id,
                        session_id=self.session_id,
                    )
                    if not existing_session:
                        raise ValueError(f"Failed to create session: {self.session_id}")
                
                # Run live streaming
                # run_live returns an async generator, wrap it with Aclosing for proper cleanup
                async_gen = runner.run_live(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    live_request_queue=self.live_request_queue,
                )
                
                # Forward events to WebSocket using Aclosing for proper cleanup
                logger.info("Starting to receive events from run_live...")
                async with Aclosing(async_gen) as agen:
                    event_count = 0
                    async for event in agen:
                        event_count += 1
                        event_type = type(event).__name__
                        
                        # Log event details (content type, author, etc.)
                        event_summary = event_type
                        if hasattr(event, 'content') and event.content:
                            if hasattr(event.content, 'parts') and event.content.parts:
                                part_types = [type(p).__name__ for p in event.content.parts]
                                event_summary += f" (content parts: {part_types})"
                        if hasattr(event, 'author'):
                            event_summary += f" [author: {event.author}]"
                        
                        logger.debug(f"Received ADK event #{event_count}: {event_summary}")
                        
                        # Serialize event to JSON and send
                        try:
                            event_json = event.model_dump_json(exclude_none=True, by_alias=True)
                            await self.send(event_json)
                            # Only log first few events to avoid spam, but include content preview
                            if event_count <= 5:
                                # Extract a preview of the event content for debugging
                                preview = event_summary
                                if hasattr(event, 'content') and event.content and event.content.parts:
                                    for i, part in enumerate(event.content.parts[:2]):  # First 2 parts only
                                        if hasattr(part, 'text') and part.text:
                                            preview += f" [text: {part.text[:50]}...]"
                                        elif hasattr(part, 'inline_data') and part.inline_data:
                                            preview += f" [inline_data: {part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else 'data'}]"
                                logger.info(f"Sent ADK event #{event_count} to WebSocket: {preview}")
                        except Exception as e:
                            logger.warning(f"Error serializing event: {e}", exc_info=True)
                            # Send error message
                            await self.send(json.dumps({
                                "type": "error",
                                "message": f"Error processing event: {str(e)}"
                            }))
                        
            except Exception as e:
                logger.error(f"Error in forward_events: {e}", exc_info=True)
                await self.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
        
        # Create process_messages task (processes incoming WebSocket messages from queue)
        async def process_messages():
            try:
                while True:
                    # Wait for messages from queue (put there by receive method)
                    message = await self.message_queue.get()
                    
                    if message.get("type") == "text":
                        # Text control message
                        try:
                            data = json.loads(message["data"])
                            await self.handle_control_message(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in control message: {e}")
                            await self.send(json.dumps({
                                "type": "error",
                                "message": "Invalid JSON"
                            }))
                    elif message.get("type") == "bytes":
                        # Audio blob data
                        logger.info(f"Received audio data in queue: {len(message['data'])} bytes")
                        await self.handle_audio_data(message["data"])
                        
            except Exception as e:
                logger.error(f"Error in process_messages: {e}", exc_info=True)
                if "closed" not in str(e).lower():
                    await self.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
        
        # Start both tasks concurrently
        self.forward_task = asyncio.create_task(forward_events())
        self.process_task = asyncio.create_task(process_messages())
    
    async def handle_control_message(self, data: dict):
        """Handle control messages from client."""
        if not self.live_request_queue:
            return
        
        msg_type = data.get('type')
        
        try:
            if msg_type == 'content':
                # Send text content to the live queue
                text = data.get('text', '')
                if text:
                    content = genai_types.Content(
                        role="user",
                        parts=[genai_types.Part.from_text(text=text)]
                    )
                    self.live_request_queue.send_content(content)
                    
            # Note: activity_start and activity_end are not supported when Gemini Live
            # has automatic activity detection enabled (which is the default)
            # elif msg_type == 'activity_start':
            #     self.live_request_queue.send_activity_start()
            # elif msg_type == 'activity_end':
            #     self.live_request_queue.send_activity_end()
                
            elif msg_type == 'close':
                # Close the queue
                self.live_request_queue.close()
                
            elif msg_type == 'ping':
                # Respond to ping
                await self.send(json.dumps({"type": "pong"}))
                
        except Exception as e:
            logger.error(f"Error handling control message: {e}")
            await self.send(json.dumps({
                "type": "error",
                "message": f"Error processing control message: {str(e)}"
            }))
    
    async def handle_audio_data(self, audio_bytes: bytes):
        """Handle incoming audio blob data."""
        if not self.live_request_queue or not audio_bytes:
            return
        
        try:
            # Convert audio bytes to Blob
            # Frontend now sends PCM format (linear16, 16-bit, 16kHz mono) via Web Audio API
            blob = genai_types.Blob(
                data=audio_bytes,
                mime_type="audio/pcm"  # PCM format as expected by Gemini Live
            )
            
            logger.debug(f"Sending audio blob to live queue: {len(audio_bytes)} bytes, mime_type=audio/pcm")
            
            # Send to live queue as realtime input
            self.live_request_queue.send_realtime(blob)
            
        except Exception as e:
            logger.error(f"Error handling audio data: {e}", exc_info=True)
            await self.send(json.dumps({
                "type": "error",
                "message": f"Error processing audio: {str(e)}"
            }))
    
    async def disconnect(self, close_code):
        """Handle disconnection."""
        logger.info(f"Voice WebSocket disconnecting: flow={self.flow_id}, code={close_code}")
        
        # Cancel tasks
        if self.forward_task:
            self.forward_task.cancel()
            try:
                await self.forward_task
            except asyncio.CancelledError:
                pass
        
        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass
        
        # Close the live request queue
        if self.live_request_queue:
            try:
                self.live_request_queue.close()
            except Exception as e:
                logger.warning(f"Error closing live request queue: {e}")
        
        logger.info(f"Voice WebSocket disconnected: flow={self.flow_id}")
    
    async def receive(self, text_data=None, bytes_data=None):
        """Handle incoming WebSocket messages - queues them for processing."""
        try:
            if text_data:
                # Queue text message for processing
                await self.message_queue.put({"type": "text", "data": text_data})
            elif bytes_data:
                # Queue binary audio data for processing
                await self.message_queue.put({"type": "bytes", "data": bytes_data})
        except Exception as e:
            logger.error(f"Error queueing message: {e}")


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

