# AI Coding Agent Instructions: Implement Real-Time Status Updates for AI Chatbot

## Objective
Implement a real-time status update system that provides live feedback to users about the progress of their query processing. This system should show intermediate steps during document retrieval, validation, and answer generation phases.

## System Architecture Overview
The status update system consists of three main components:
1. **Backend Status Callback System** - Captures and queues status messages
2. **Server-Sent Events (SSE) Endpoint** - Streams status updates to frontend
3. **Frontend JavaScript** - Receives and displays status updates in real-time

## Implementation Requirements

### 1. Backend Status Infrastructure

#### A. Status Callback Factory Function
Create a function that generates session-specific status callback functions:

```python
import queue
from threading import Lock

# Global storage for status queues (session_id -> Queue)
status_queues = {}
status_lock = Lock()

def create_status_callback(session_id):
    """Create a status callback function for a specific session"""
    def status_callback(message):
        print(f"Status callback for session {session_id}: {message}")  # Debug logging
        with status_lock:
            if session_id in status_queues:
                try:
                    status_queues[session_id].put(message, block=False)
                    print(f"Successfully queued status: {message}")  # Debug logging
                except queue.Full:
                    print(f"Queue full, skipped status: {message}")  # Debug logging
            else:
                print(f"No queue found for session {session_id}")  # Debug logging
    return status_callback
```

#### B. Queue Management
Add queue initialization and cleanup:

```python
def initialize_status_queue(session_id):
    """Initialize a status queue for a session"""
    with status_lock:
        if session_id not in status_queues:
            status_queues[session_id] = queue.Queue(maxsize=50)
            print(f"Created status queue for session: {session_id}")

def cleanup_status_queue(session_id):
    """Clean up status queue for a session"""
    with status_lock:
        if session_id in status_queues:
            del status_queues[session_id]
            print(f"Cleaned up status queue for session: {session_id}")
```

### 2. Query Processing Integration

#### A. Status Callback Parameter
Modify your main query processing function to accept and use a status callback:

```python
def process_query(query, retriever, llm, conversation_history, debug_mode=False, status_callback=None):
    """
    Main query processing function with status updates
    
    Args:
        query: User's question
        retriever: Document retrieval system
        llm: Language model for answer generation
        conversation_history: Chat history
        debug_mode: Enable debug output
        status_callback: Function to call with status updates (callable that takes string)
    """
    
    # Example status updates throughout processing:
    if status_callback:
        status_callback("Processing query")
    
    # During document retrieval
    if status_callback:
        status_callback("Searching knowledge base")
    
    # During validation/ranking
    if status_callback:
        status_callback("Analyzing content relevance")
    
    # During answer generation
    if status_callback:
        status_callback("Generating answer")
    
    # When complete
    if status_callback:
        status_callback("Complete")
```

#### B. Retrieval Class Integration
If using a retrieval class, add status callback support:

```python
class YourRetriever:
    def __init__(self, ...):
        self.status_callback = None
    
    def set_status_callback(self, callback):
        """Set callback function for status updates"""
        self.status_callback = callback
    
    def _update_status(self, message: str):
        """Update status via callback if available"""
        if self.status_callback:
            self.status_callback(message)
    
    def retrieve_documents(self, query):
        self._update_status("Finding relevant information")
        # ... retrieval logic ...
        
        self._update_status("Ranking results")
        # ... ranking logic ...
```

### 3. Flask/Web Framework Routes

#### A. Main Query Route
Integrate status callback into your main query processing route:

```python
from flask import Flask, request, session, jsonify
import uuid

@app.route('/query', methods=['POST'])
def handle_query():
    # Get or create session ID
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    # Initialize status queue for this session
    initialize_status_queue(session_id)
    
    try:
        query = request.json.get('query', '')
        
        # Create status callback for this session
        status_callback = create_status_callback(session_id)
        
        # Process query with status updates
        result = process_query(
            query, 
            retriever, 
            llm, 
            conversation_history, 
            debug_mode=False, 
            status_callback=status_callback
        )
        
        # Final status update
        status_callback("Complete")
        
        return jsonify(result)
        
    except Exception as e:
        if 'status_callback' in locals():
            status_callback("Error occurred")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up status queue after a delay (or implement proper cleanup)
        pass
```

#### B. Server-Sent Events Endpoint
Create an SSE endpoint to stream status updates:

```python
from flask import Response
import json
import time

@app.route('/status/<session_id>')
def stream_status(session_id):
    """Stream status updates for a specific session via Server-Sent Events"""
    
    def generate_status_events():
        # Initialize queue if it doesn't exist
        initialize_status_queue(session_id)
        
        while True:
            try:
                with status_lock:
                    if session_id in status_queues:
                        queue_obj = status_queues[session_id]
                    else:
                        break
                
                try:
                    # Wait for status update with timeout
                    status = queue_obj.get(timeout=1.0)
                    
                    # Format as SSE
                    data = json.dumps({"status": status})
                    yield f"data: {data}\\n\\n"
                    
                    # Break if complete or error
                    if status in ["Complete", "Error occurred"]:
                        break
                        
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'heartbeat': True})}\\n\\n"
                    continue
                    
            except Exception as e:
                print(f"Error in status stream: {e}")
                break
        
        # Clean up when done
        cleanup_status_queue(session_id)
    
    return Response(
        generate_status_events(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

# Alternative endpoint for current session
@app.route('/status/current')
def stream_current_status():
    session_id = session.get('session_id')
    if session_id:
        return stream_status(session_id)
    else:
        return Response("", mimetype='text/event-stream')
```

### 4. Frontend JavaScript Implementation

#### A. HTML Structure
Add a status display element to your chat interface:

```html
<div id="chatContainer">
    <div id="loadingStatus" class="loading-status" style="display: none;">
        Processing...
    </div>
    <!-- Your existing chat UI -->
</div>
```

#### B. CSS Styling
Add appropriate styling for the status display:

```css
.loading-status {
    background: #f0f0f0;
    padding: 8px 16px;
    border-radius: 4px;
    margin: 8px 0;
    font-style: italic;
    color: #666;
    font-size: 14px;
    text-align: center;
    border-left: 3px solid #4285f4;
}
```

#### C. JavaScript Status Handler
Implement the frontend status update logic:

```javascript
let eventSource = null;

function setupStatusUpdates() {
    // Get session ID from your session management system
    const sessionId = getSessionId(); // Implement based on your session handling
    
    if (sessionId && typeof(EventSource) !== "undefined") {
        try {
            // Close existing connection if any
            if (eventSource) {
                eventSource.close();
            }
            
            // Start listening for status updates
            const statusUrl = `/status/${sessionId}`;
            console.log('Opening EventSource to:', statusUrl);
            eventSource = new EventSource(statusUrl);
            
            eventSource.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    if (data.status && data.status !== "Complete" && data.status !== "Error occurred") {
                        updateLoadingStatus(data.status);
                    } else if (data.status === "Complete" || data.status === "Error occurred") {
                        hideLoadingStatus();
                        eventSource.close();
                        eventSource = null;
                    }
                } catch (e) {
                    console.error('Error parsing SSE data:', e);
                }
            };
            
            eventSource.onerror = function(error) {
                console.error('EventSource error:', error);
                hideLoadingStatus();
                if (eventSource) {
                    eventSource.close();
                    eventSource = null;
                }
            };
            
        } catch (sseError) {
            console.log('SSE not available, continuing without status updates:', sseError);
        }
    }
}

function updateLoadingStatus(status) {
    const statusElement = document.getElementById('loadingStatus');
    if (statusElement) {
        statusElement.textContent = status;
        statusElement.style.display = 'block';
    }
}

function hideLoadingStatus() {
    const statusElement = document.getElementById('loadingStatus');
    if (statusElement) {
        statusElement.style.display = 'none';
    }
}

// Call when submitting a query
function submitQuery() {
    // Show initial loading status
    updateLoadingStatus("Processing query...");
    
    // Set up status updates
    setupStatusUpdates();
    
    // Your existing query submission logic
    // ...
}
```

### 5. Suggested Status Messages by Processing Stage

Customize these status messages based on your specific processing pipeline:

#### Document Retrieval Stage:
- `"Searching knowledge base"`
- `"Finding relevant information"`
- `"Found cached result"`
- `"Retrieving documents"`

#### Content Analysis Stage:
- `"Analyzing content relevance"`
- `"Validating information"`
- `"Ranking results"`
- `"Ranking best matches"`

#### Context Expansion Stage:
- `"Answer not found, expanding context (attempt 1)"`
- `"Answer not found, expanding context (attempt 2)"`
- `"Using best available matches"`

#### Answer Generation Stage:
- `"Generating answer"`
- `"Synthesizing response"`
- `"Finalizing answer"`

#### Completion:
- `"Complete"`
- `"Error occurred"` (for error states)

### 6. Error Handling and Cleanup

#### A. Connection Management
```javascript
// Clean up on page unload
window.addEventListener('beforeunload', function() {
    if (eventSource) {
        eventSource.close();
    }
});

// Implement timeout for status updates
let statusTimeout = null;

function setupStatusTimeout() {
    // Clear existing timeout
    if (statusTimeout) {
        clearTimeout(statusTimeout);
    }
    
    // Set timeout for 30 seconds
    statusTimeout = setTimeout(() => {
        console.log('Status update timeout, hiding status');
        hideLoadingStatus();
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    }, 30000);
}
```

#### B. Backend Cleanup
```python
# Add cleanup in your application shutdown or periodic cleanup
def cleanup_old_status_queues():
    """Clean up old/unused status queues periodically"""
    current_time = time.time()
    with status_lock:
        # Implement cleanup logic based on your needs
        # For example, remove queues older than 10 minutes
        pass
```

### 7. Integration Checklist

When implementing this system:

1. ✅ **Backend Infrastructure**
   - [ ] Add status queue management
   - [ ] Create status callback factory
   - [ ] Implement SSE endpoint
   - [ ] Add cleanup mechanisms

2. ✅ **Query Processing Integration**
   - [ ] Add status_callback parameter to main processing function
   - [ ] Add _update_status calls throughout processing pipeline
   - [ ] Set status callback in retrieval classes

3. ✅ **Frontend Implementation**
   - [ ] Add status display HTML element
   - [ ] Implement JavaScript EventSource handling
   - [ ] Add CSS styling for status display
   - [ ] Add error handling and cleanup

4. ✅ **Testing**
   - [ ] Test status updates appear in real-time
   - [ ] Verify proper cleanup after completion
   - [ ] Test error scenarios
   - [ ] Verify multiple concurrent sessions work

### 8. Customization Notes

- **Status Message Timing**: Adjust status messages based on your specific processing steps
- **Queue Size**: Modify queue.Queue(maxsize=50) based on your expected message volume
- **Timeout Values**: Adjust SSE timeout and cleanup intervals for your use case
- **Session Management**: Integrate with your existing session handling system
- **Error Messages**: Customize error status messages for your application

This implementation provides a robust, real-time status update system that enhances user experience by showing processing progress during potentially long-running operations.
