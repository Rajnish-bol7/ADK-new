# Voice Call Fix - Activity Signals Removed

## Issue
The error was: **"Explicit activity control is not supported when automatic activity detection is enabled."**

## Root Cause
Gemini Live has automatic activity detection enabled by default. When we send `activity_start` and `activity_end` signals explicitly, it conflicts with the automatic detection.

## Fix Applied

### Frontend (templates/chat.html)
- Removed `activity_start` signal when starting audio capture
- Removed `activity_end` signal when stopping audio capture
- Removed `window.activityStarted` flag

### Backend (api/consumers.py)
- Commented out `activity_start` handler
- Commented out `activity_end` handler

## What Changed

**Before:**
```javascript
// Sent activity_start when audio started
window.voiceWs.send(JSON.stringify({ type: 'activity_start' }));
```

**After:**
```javascript
// Just send audio directly - Gemini Live handles activity detection automatically
window.voiceWs.send(arrayBuffer);
```

## Testing

1. Restart the server
2. Connect to voice WebSocket
3. Send audio directly (no activity signals)
4. Gemini Live will automatically detect when you're speaking

## Notes

- Gemini Live uses automatic activity detection by default
- We only need to send the audio data (binary)
- The system automatically detects when user is speaking vs. not speaking
- This is more reliable than manual activity signals

