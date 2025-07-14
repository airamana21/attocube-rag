import os
import json
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

# Global RAG components (initialized once)
retriever = None
llm = None
conversations = {}  # Store conversations per session

# Initialize RAG system on startup
print("Starting RAG system initialization...")
retriever, llm = initialize_rag_system()
print("RAG system ready!")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html', client_id=GOOGLE_CLIENT_ID)

@app.route('/login')
def login():
    return render_template('login.html', client_id=GOOGLE_CLIENT_ID)

@app.route('/auth', methods=['POST'])
def auth():
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
        
        return jsonify({
            'success': True,
            'response': result['answer'],
            'sources': result['sources'],
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

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)