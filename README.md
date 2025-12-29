# ADK Flow Platform

A Django-based platform for building AI agents from visual flow definitions, powered by Google's Agent Development Kit (ADK).

## Features

- **Visual Flow Builder**: Create AI agent flows with a drag-and-drop interface
- **Multi-Tenant**: Isolated data and API keys per tenant
- **Multi-Model Support**: Gemini, GPT-4, Claude, and more
- **Real-Time Chat**: Streaming responses via Server-Sent Events
- **WebSocket Support**: Real-time text and voice streaming
- **Background Execution**: Async flow execution with Celery
- **Scheduled Flows**: Periodic/scheduled flow execution with Celery Beat
- **Webhooks**: Receive flow configurations from external platforms
- **Session Management**: Persistent conversations across requests

## Project Structure

```
adk-python/
├── config/                  # Django project settings
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py              # WebSocket support
│   ├── celery.py            # Celery configuration
│   └── wsgi.py
├── core/                    # Core app (users, tenants)
│   ├── models.py            # User, Tenant, APIKey
│   └── admin.py
├── flows/                   # Flows app
│   ├── models.py            # Flow, Session, Message
│   ├── tasks.py             # Celery tasks
│   └── admin.py
├── api/                     # REST API
│   ├── urls.py
│   ├── views.py
│   ├── consumers.py         # WebSocket consumers
│   └── routing.py           # WebSocket routing
├── adk_integration/         # ADK integration layer
│   ├── schema/              # Flow JSON validation
│   ├── builder/             # Agent building
│   └── executor/            # Flow execution
├── src/google/adk/          # Core ADK library
├── manage.py
├── requirements.txt
└── .env
```

## Quick Start

### 1. Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Run migrations
python manage.py migrate

# Credentials
# Username: admin, Password: admin123
```

### 2. Configure API Keys

Edit `.env` and add your API keys:

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 3. Run the Server

```bash
# Development server
python manage.py runserver

# With WebSocket support (recommended)
daphne config.asgi:application -p 8000
```

### 4. Start Celery (for background tasks)

```bash
# Start Celery worker
celery -A config worker -l info

# Start Celery Beat (for scheduled tasks)
celery -A config beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler
```

### 5. Access the Admin

Open http://localhost:8000/admin/

- Username: `admin`
- Password: `admin123`

## API Endpoints

### Authentication

```
POST /api/v1/auth/login/         # Login
POST /api/v1/auth/logout/        # Logout
GET  /api/v1/auth/me/            # Current user
```

### Flows

```
GET  /api/v1/flows/              # List flows
POST /api/v1/flows/              # Create flow
GET  /api/v1/flows/{id}/         # Get flow
PUT  /api/v1/flows/{id}/         # Update flow
DELETE /api/v1/flows/{id}/       # Delete flow
```

### Chat

```
POST /api/v1/flows/{id}/chat/           # Non-streaming chat
POST /api/v1/flows/{id}/chat/stream/    # Streaming chat (SSE)
```

### Background Execution

```
POST /api/v1/flows/{id}/run-async/      # Run flow in background
POST /api/v1/flows/{id}/schedule/       # Create scheduled task
```

### Sessions

```
GET  /api/v1/flows/{id}/sessions/       # List sessions
GET  /api/v1/sessions/{id}/             # Get session
GET  /api/v1/sessions/{id}/messages/    # Get messages
DELETE /api/v1/sessions/{id}/           # Delete session
```

### Webhooks

```
POST /api/v1/webhook/flow/       # Receive flow JSON configuration from external platform
```

**Expected JSON payload:**
```json
{
  "id": "flow-uuid",
  "tenant_id": "tenant-uuid",
  "flow_name": "my_flow",
  "description": "Optional description",
  "nodes": [...],
  "edges": [...]
}
```

## WebSocket Endpoints

### Real-Time Chat

```javascript
// Connect
const ws = new WebSocket('ws://localhost:8000/ws/chat/{flow_id}/');

// Send message
ws.send(JSON.stringify({
    type: 'message',
    content: 'Hello!',
    session_id: 'optional-session-id'
}));

// Receive streaming response
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'text') {
        console.log(data.content, data.is_final);
    }
};
```

### Voice Streaming

```javascript
// Connect
const ws = new WebSocket('ws://localhost:8000/ws/voice/{flow_id}/');

// Start streaming
ws.send(JSON.stringify({ type: 'start' }));

// Send audio (PCM 16-bit, 16kHz mono)
ws.send(audioBuffer);

// Receive audio response
ws.onmessage = (event) => {
    if (event.data instanceof Blob) {
        // Play audio
    }
};
```

### Gemini Live Streaming

```javascript
// Connect
const ws = new WebSocket('ws://localhost:8000/ws/live/{flow_id}/');

// Start live session
ws.send(JSON.stringify({ type: 'start_live' }));

// Bidirectional audio streaming
```

## Background & Scheduled Execution

### Run Flow in Background

```bash
curl -X POST http://localhost:8000/api/v1/flows/{id}/run-async/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Process this in background",
    "callback_url": "https://your-webhook.com/result"
  }'
```

### Create Scheduled Flow

```bash
# Run every hour
curl -X POST http://localhost:8000/api/v1/flows/{id}/schedule/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Hourly Report",
    "schedule_type": "interval",
    "interval_minutes": 60,
    "input_message": "Generate hourly report"
  }'

# Run daily at 9 AM
curl -X POST http://localhost:8000/api/v1/flows/{id}/schedule/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Daily Report",
    "schedule_type": "crontab",
    "crontab": "0 9 * * *",
    "input_message": "Generate daily report"
  }'
```

## Flow JSON Format

```json
{
  "nodes": [
    {
      "id": "agent_1",
      "type": "agent",
      "position": {"x": 100, "y": 100},
      "data": {
        "agentName": "my_agent",
        "model": "gemini-2.5-flash",
        "promptDescription": "You are a helpful assistant.",
        "promptInstructions": "Be concise."
      }
    }
  ],
  "connections": [
    {
      "id": "conn_1",
      "source": "trigger_1",
      "target": "agent_1"
    }
  ]
}
```

## Supported Models

| Provider | Models |
|----------|--------|
| Google | gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash |
| OpenAI | gpt-4, gpt-4-turbo, gpt-4o, gpt-3.5-turbo |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3.5-sonnet |

## Development

### Run Test Script

```bash
python test_flow.py
```

### Test Background Tasks

```bash
# Make sure Celery worker is running
celery -A config worker -l info

# Then trigger a background task via API
```

## Production Deployment

### Requirements

- Redis (for Celery broker)
- PostgreSQL (recommended for production)

### Environment Variables

```env
DEBUG=False
DJANGO_SECRET_KEY=your-secure-secret-key
DATABASE_URL=postgres://user:pass@localhost/dbname
CELERY_BROKER_URL=redis://localhost:6379/0
REDIS_URL=redis://localhost:6379/0
```

### Run with Gunicorn + Daphne

```bash
# HTTP (gunicorn)
gunicorn config.wsgi:application

# WebSockets (daphne)
daphne config.asgi:application -p 8001
```

## License

Apache 2.0
