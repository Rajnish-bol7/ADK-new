#!/usr/bin/env python3
"""
Script to create a test flow for voice calling.
"""

import json
import uuid
import requests
import sys

# Configuration
BASE_URL = "http://localhost:8000"
WEBHOOK_URL = f"{BASE_URL}/api/v1/webhook/flow/"

# Generate unique IDs
flow_id = str(uuid.uuid4())
tenant_id = str(uuid.uuid4())

# Test flow configuration
flow_config = {
    "id": flow_id,
    "tenant_id": tenant_id,
    "flow_name": "Test Voice Call Flow",
    "description": "A simple test flow for testing voice calling functionality",
    "nodes": [
        {
            "id": "node_0",
            "type": "call_start",
            "position": {
                "x": 200,
                "y": 200
            },
            "data": {
                "callType": "incoming",
                "phoneNumber": "",
                "callerName": "Test User"
            },
            "width": 120,
            "height": 120
        },
        {
            "id": "node_1",
            "type": "agent",
            "position": {
                "x": 500,
                "y": 200
            },
            "data": {
                "agentName": "VoiceAssistant",
                "agentType": "agent",
                "model": "gemini-2.0-flash-exp",
                "promptDescription": "You are a helpful voice assistant. Respond naturally and conversationally.",
                "promptInstructions": "Keep responses concise and friendly. Ask clarifying questions if needed.",
                "includeChatHistory": True,
                "reasoningEffort": "medium",
                "outputFormat": "text",
                "knowledgeSource": "auto"
            },
            "width": 200,
            "height": 138
        }
    ],
    "edges": [
        {
            "id": "edge_0",
            "source": "node_0",
            "target": "node_1",
            "sourceHandle": "call-start-out",
            "targetHandle": "agent-in"
        }
    ]
}

def create_flow():
    """Create the test flow via webhook."""
    print("Creating test voice call flow...")
    print(f"Flow ID: {flow_id}")
    print(f"Tenant ID: {tenant_id}")
    print(f"Flow Name: {flow_config['flow_name']}")
    print()
    
    try:
        response = requests.post(
            WEBHOOK_URL,
            json=flow_config,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Flow created successfully!")
            print()
            print("Response:")
            print(json.dumps(result, indent=2))
            print()
            print("=" * 60)
            print("TESTING INSTRUCTIONS:")
            print("=" * 60)
            print(f"1. Open: {BASE_URL}")
            print(f"2. Select tenant: {tenant_id}")
            print(f"3. Select flow: {flow_config['flow_name']}")
            print("4. Click the Call button (phone icon)")
            print("5. Click 'Start Call'")
            print("6. Grant microphone permissions")
            print("7. Speak into your microphone")
            print()
            print(f"Flow will be available at: {BASE_URL}/api/v1/flows/{flow_id}/")
            return True
        else:
            print(f"❌ Error creating flow: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server.")
        print(f"   Make sure the server is running at {BASE_URL}")
        print("   Run: python manage.py runserver 0.0.0.0:8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = create_flow()
    sys.exit(0 if success else 1)

