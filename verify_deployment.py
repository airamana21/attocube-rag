#!/usr/bin/env python3
"""
Pre-deployment verification script.
Run this before deploying to Cloud Run to ensure everything is configured correctly.
"""

import os
import subprocess
import sys

def check_gcloud_config():
    """Check gcloud configuration"""
    try:
        # Check if gcloud is installed
        result = subprocess.run(['gcloud', 'version'], 
                              capture_output=True, text=True, check=True)
        print("✅ gcloud CLI is installed")
        
        # Check current project
        result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                              capture_output=True, text=True, check=True)
        project = result.stdout.strip()
        if project:
            print(f"✅ gcloud project: {project}")
            return project
        else:
            print("❌ No gcloud project set")
            return None
            
    except subprocess.CalledProcessError:
        print("❌ gcloud CLI not found or not configured")
        return None
    except FileNotFoundError:
        print("❌ gcloud CLI not installed")
        return None

def verify_deployment_ready():
    """Verify the application is ready for deployment"""
    
    print("🚀 Pre-Deployment Verification")
    print("=" * 60)
    
    issues = []
    
    # Check that LOCAL_DEV_MODE is not set (shouldn't be in Cloud Run)
    if os.environ.get("LOCAL_DEV_MODE", "false").lower() == "true":
        issues.append("LOCAL_DEV_MODE is set - this should only be for local development")
    else:
        print("✅ LOCAL_DEV_MODE not set (good for production)")
    
    # Check required files exist
    required_files = [
        "app.py",
        "rag.py", 
        "requirements.txt",
        "Dockerfile",
        ".gcloudignore"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            issues.append(f"Missing required file: {file}")
    
    # Check gcloud configuration
    gcloud_project = check_gcloud_config()
    
    # Check .gcloudignore excludes local dev files
    try:
        with open('.gcloudignore', 'r') as f:
            gcloudignore_content = f.read()
            
        local_dev_files = ['run_local.py', 'run_local.bat', 'check_dependencies.py', 'test_config.py']
        for dev_file in local_dev_files:
            if dev_file in gcloudignore_content:
                print(f"✅ {dev_file} excluded from deployment")
            else:
                issues.append(f"{dev_file} not excluded in .gcloudignore")
                
    except FileNotFoundError:
        issues.append(".gcloudignore file not found")
    
    print("\n📋 Deployment Checklist:")
    print("-" * 40)
    print("The following environment variables should be set in Cloud Run:")
    print("  • GOOGLE_CLIENT_ID (your OAuth client ID)")
    print("  • GCP_PROJECT_ID (your GCP project ID)")
    print("  • GCS_BUCKET_NAME (your PDF storage bucket)")
    print("  • Do NOT set LOCAL_DEV_MODE (should remain unset)")
    
    print("\n🔍 Issues Found:")
    print("-" * 40)
    if issues:
        for issue in issues:
            print(f"❌ {issue}")
        print(f"\n⚠️  Found {len(issues)} issue(s) that should be resolved before deployment")
        return False
    else:
        print("✅ No issues found - ready for deployment!")
        return True
    
def main():
    ready = verify_deployment_ready()
    
    if ready:
        print("\n🎉 Your application is ready for Cloud Run deployment!")
        print("\n📝 To deploy, run:")
        print("gcloud run deploy attocube-support --source . --platform managed --region us-central1")
        print("\n💡 Don't forget to set the required environment variables in Cloud Run console!")
        sys.exit(0)
    else:
        print("\n🛑 Please fix the issues above before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()
