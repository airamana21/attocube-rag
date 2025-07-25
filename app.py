import os
import json
import base64
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from google.auth.transport import requests
from google.oauth2 import id_token
from functools import wraps
import secrets
from rag import initialize_rag_system, process_query, ConversationHistory

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
CORS(app)

# Google OAuth settings
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
ALLOWED_DOMAIN = "@lbl.gov"

# Local development mode - set this environment variable to bypass authentication
LOCAL_DEV_MODE = os.environ.get("LOCAL_DEV_MODE", "false").lower() == "true"

# Log the current mode for debugging
if LOCAL_DEV_MODE:
    print("🔧 RUNNING IN LOCAL DEVELOPMENT MODE - Authentication bypassed")
    print(f"   • GOOGLE_CLIENT_ID: {'SET' if GOOGLE_CLIENT_ID else 'NOT SET'}")
else:
    print("🚀 RUNNING IN PRODUCTION MODE - Authentication required")
    print(f"   • GOOGLE_CLIENT_ID: {'SET' if GOOGLE_CLIENT_ID else 'NOT SET'}")
    print(f"   • Allowed domain: {ALLOWED_DOMAIN}")

print(f"LOCAL_DEV_MODE: {LOCAL_DEV_MODE}")

# Global RAG components (initialized on startup)
retriever = None
llm = None
conversations = {}  # Store conversations per session
rag_initialized = False
initialization_error = None

# Initialize RAG system on startup
def initialize_rag_on_startup():
    global retriever, llm, rag_initialized, initialization_error
    try:
        print("Starting RAG system initialization...")
        retriever, llm = initialize_rag_system()
        rag_initialized = True
        print("RAG system ready!")
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        initialization_error = str(e)
        raise e

# Start initialization in background when app starts
print("App starting - checking if we should initialize RAG...")
if not rag_initialized and not initialization_error:
    try:
        print("Starting RAG initialization on app startup...")
        initialize_rag_on_startup()
        print(f"App startup: RAG initialization complete. rag_initialized={rag_initialized}")
    except Exception as e:
        print(f"App startup: RAG initialization failed: {e}")
else:
    print(f"App startup: Skipping initialization. rag_initialized={rag_initialized}, initialization_error={initialization_error}")

def ensure_rag_initialized():
    """Ensure RAG system is initialized"""
    global rag_initialized, initialization_error
    if not rag_initialized:
        if initialization_error:
            raise Exception(f"RAG system failed to initialize: {initialization_error}")
        else:
            raise Exception("RAG system is still initializing, please wait...")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip authentication in local development mode
        if LOCAL_DEV_MODE:
            # Set a dummy user session for local development
            if 'user_email' not in session:
                session['user_email'] = 'dev@localhost'
                session['user_name'] = 'Local Developer'
            return f(*args, **kwargs)
        
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    # Skip authentication redirect in local development mode
    if LOCAL_DEV_MODE:
        # Set a dummy user session for local development
        if 'user_email' not in session:
            session['user_email'] = 'dev@localhost'
            session['user_name'] = 'Local Developer'
        return render_template('index.html')
    
    # Redirect to login if user is not authenticated
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login')
def login():
    # In local dev mode, automatically "log in" the user
    if LOCAL_DEV_MODE:
        session['user_email'] = 'dev@localhost'
        session['user_name'] = 'Local Developer'
        return redirect(url_for('index'))
    
    # In production, ensure we have the Google Client ID
    if not GOOGLE_CLIENT_ID:
        return "Error: Google OAuth not configured. Contact system administrator.", 500
    
    return render_template('login.html', client_id=GOOGLE_CLIENT_ID)

@app.route('/auth', methods=['POST'])
def auth():
    # In local dev mode, skip OAuth verification
    if LOCAL_DEV_MODE:
        session['user_email'] = 'dev@localhost'
        session['user_name'] = 'Local Developer'
        return jsonify({'success': True})
    
    # In production, require proper OAuth
    if not GOOGLE_CLIENT_ID:
        return jsonify({'error': 'OAuth not configured'}), 500
    
    try:
        token = request.json['credential']
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), GOOGLE_CLIENT_ID
        )
        
        email = idinfo['email']
        
        # Check if email ends with @lbl.gov
        if not email.endswith(ALLOWED_DOMAIN):
            return jsonify({'error': 'Unauthorized domain'}), 403
        
        # Store user info in session
        session['user_email'] = email
        session['user_name'] = idinfo.get('name', email)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 401

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    # Ensure RAG system is initialized
    ensure_rag_initialized()
    
    data = request.json
    query = data.get('query', '')
    debug_mode = data.get('debug_mode', False)
    
    # Get or create conversation history for this session
    session_id = session.get('session_id', secrets.token_hex(16))
    session['session_id'] = session_id
    
    if session_id not in conversations:
        conversations[session_id] = ConversationHistory()
    
    conversation_history = conversations[session_id]
    
    try:
        # Process the query
        result = process_query(query, retriever, llm, conversation_history, debug_mode)
        
        # Convert images to base64 for JSON transmission
        images_b64 = []
        for img in result.get('images', []):
            img_b64 = {
                "id": img["id"],
                "source": img["source"],
                "page": img["page"],
                "doc_type": img["doc_type"],
                "width": img["width"],
                "height": img["height"],
                "image_index": img["image_index"],
                "image_data": base64.b64encode(img["image_data"]).decode('utf-8')
            }
            images_b64.append(img_b64)
        
        return jsonify({
            'success': True,
            'response': result['answer'],
            'sources': result['sources'],
            'images': images_b64,
            'debug_info': result['debug_info']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/clear', methods=['POST'])
@login_required
def clear_conversation():
    session_id = session.get('session_id')
    if session_id and session_id in conversations:
        conversations[session_id].clear()
    return jsonify({'success': True})

@app.route('/image/<image_id>')
@login_required
def get_image(image_id):
    """Serve individual images by ID (if needed for direct access)"""
    # This endpoint could be used for direct image access if needed
    # For now, images are embedded in the chat response as base64
    return jsonify({'error': 'Images are embedded in chat responses'}), 404

@app.route('/health')
def health():
    """Health check endpoint with mode information"""
    global rag_initialized, initialization_error
    
    try:
        base_response = {
            'mode': 'local_development' if LOCAL_DEV_MODE else 'production',
            'local_dev_mode': LOCAL_DEV_MODE,
            'oauth_configured': bool(GOOGLE_CLIENT_ID),
            'allowed_domain': ALLOWED_DOMAIN if not LOCAL_DEV_MODE else 'bypassed'
        }
        
        if rag_initialized:
            return jsonify({
                **base_response,
                'status': 'healthy', 
                'rag_initialized': True,
                'message': 'RAG system is ready'
            })
        elif initialization_error:
            return jsonify({
                **base_response,
                'status': 'error', 
                'rag_initialized': False,
                'error': initialization_error
            }), 500
        else:
            return jsonify({
                **base_response,
                'status': 'initializing', 
                'rag_initialized': False,
                'message': 'RAG system is still initializing...'
            })
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'rag_initialized': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)