#!/usr/bin/env python3
"""
Test script to verify that the deployment environment variables are correctly set.
This helps ensure the same code works in both local and cloud environments.
"""

import os

def test_deployment_config():
    """Test configuration for different deployment environments"""
    
    print("🔍 Testing Deployment Configuration")
    print("=" * 60)
    
    # Test environment variables
    local_dev_mode = os.environ.get("LOCAL_DEV_MODE", "false").lower() == "true"
    google_client_id = os.environ.get("GOOGLE_CLIENT_ID")
    gcp_project_id = os.environ.get("GCP_PROJECT_ID")
    
    print(f"LOCAL_DEV_MODE: {local_dev_mode}")
    print(f"GOOGLE_CLIENT_ID: {'SET' if google_client_id else 'NOT SET'}")
    print(f"GCP_PROJECT_ID: {'SET (' + gcp_project_id + ')' if gcp_project_id else 'NOT SET'}")
    
    print("\n📊 Environment Analysis:")
    print("-" * 40)
    
    if local_dev_mode:
        print("✅ LOCAL DEVELOPMENT MODE")
        print("   • Authentication will be bypassed")
        print("   • Google OAuth not required")
        print("   • User will be auto-logged in as 'dev@localhost'")
        
        if not gcp_project_id:
            print("⚠️  WARNING: GCP_PROJECT_ID not set")
            print("   • RAG system may not work without GCP credentials")
        else:
            print(f"✅ GCP Project: {gcp_project_id}")
            
    else:
        print("🚀 PRODUCTION MODE")
        print("   • Full authentication required")
        print("   • Google OAuth will be enforced")
        print("   • Domain restriction: @lbl.gov")
        
        if not google_client_id:
            print("❌ ERROR: GOOGLE_CLIENT_ID not set")
            print("   • Authentication will fail in production")
            print("   • Set this environment variable for Cloud Run")
        else:
            print("✅ Google OAuth configured")
            
        if not gcp_project_id:
            print("❌ ERROR: GCP_PROJECT_ID not set")
            print("   • RAG system will not work")
        else:
            print(f"✅ GCP Project: {gcp_project_id}")
    
    print("\n🛠️  Deployment Readiness:")
    print("-" * 40)
    
    if local_dev_mode:
        print("✅ Ready for local development")
        print("💡 To deploy to Cloud Run:")
        print("   1. Do NOT set LOCAL_DEV_MODE in Cloud Run")
        print("   2. Ensure GOOGLE_CLIENT_ID is set in Cloud Run")
        print("   3. Ensure GCP_PROJECT_ID is set in Cloud Run")
    else:
        if google_client_id and gcp_project_id:
            print("✅ Ready for Cloud Run deployment")
        else:
            print("❌ NOT ready for deployment")
            print("   Missing required environment variables")
    
    print("=" * 60)

if __name__ == "__main__":
    test_deployment_config()
