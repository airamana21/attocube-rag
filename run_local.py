#!/usr/bin/env python3
"""
Local development server for the Attocube RAG system.
This script sets the LOCAL_DEV_MODE environment variable to bypass authentication.
"""

import os
import sys

def main():
    # Set environment variables for local development
    os.environ["LOCAL_DEV_MODE"] = "true"
    os.environ["FLASK_ENV"] = "development"
    os.environ["FLASK_DEBUG"] = "1"
    
    # Set default GCP project if not already set
    if not os.environ.get("GCP_PROJECT_ID"):
        print("Warning: GCP_PROJECT_ID not set. You may need to set this for the RAG system to work.")
        print("You can set it with: set GCP_PROJECT_ID=your-project-id")
    
    print("=" * 60)
    print("🚀 Starting Attocube RAG System in LOCAL DEVELOPMENT MODE")
    print("=" * 60)
    print("• Authentication is BYPASSED")
    print("• You will be automatically logged in as 'dev@localhost'")
    print("• Debug mode is ENABLED")
    print("• This mode is ONLY for local development")
    print("• Cloud deployment will use normal authentication")
    print("=" * 60)
    print()
    
    # Import and run the Flask app
    try:
        from app import app
        print("Starting Flask development server...")
        print("Open your browser to: http://127.0.0.1:5000")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run the Flask app
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            use_reloader=True
        )
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("🛑 Server stopped by user")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
