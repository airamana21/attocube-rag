@echo off
echo ================================================================
echo 🚀 Starting Attocube RAG System in LOCAL DEVELOPMENT MODE
echo ================================================================
echo • Authentication is BYPASSED
echo • You will be automatically logged in as 'dev@localhost'
echo • Debug mode is ENABLED
echo • This mode is ONLY for local development
echo • Cloud deployment will use normal authentication
echo ================================================================
echo.

REM Set environment variables for local development
set LOCAL_DEV_MODE=true
set FLASK_ENV=development
set FLASK_DEBUG=1

REM Check if GCP_PROJECT_ID is set
if "%GCP_PROJECT_ID%"=="" (
    echo Warning: GCP_PROJECT_ID not set. You may need to set this for the RAG system to work.
    echo You can set it with: set GCP_PROJECT_ID=your-project-id
    echo.
)

echo Starting Flask development server...
echo Open your browser to: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo ================================================================

REM Run the Python script
python run_local.py

pause
