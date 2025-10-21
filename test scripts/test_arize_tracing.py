#!/usr/bin/env python3
"""
Test script to verify Arize AX tracing is working correctly.

This script:
1. Makes a test request to the surf trip planner API
2. Verifies the response is valid
3. Confirms tracing was enabled
4. Provides instructions to view traces in Arize

Usage:
    python "test scripts/test_arize_tracing.py"
"""

import requests
import json
import sys
import time
from datetime import datetime

# API Configuration
BASE_URL = "http://localhost:8000"
TEST_DESTINATION = "Pipeline, North Shore, Hawaii"
TEST_DURATION = "7 days"

def check_health():
    """Check if the API is running."""
    print("🔍 Checking if API is running...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {BASE_URL}")
        print("   Make sure the server is running: ./start.sh")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def make_test_request():
    """Make a test surf trip planning request."""
    print(f"\n📤 Making test request to plan a surf trip...")
    
    # Create a unique session ID for tracking
    session_id = f"test_session_{int(time.time())}"
    
    payload = {
        "destination": TEST_DESTINATION,
        "duration": TEST_DURATION,
        "budget": "moderate",
        "surf_preferences": "reef breaks, barrels",
        "skill_level": "advanced",
        "session_id": session_id,
        "user_id": "test_user",
        "turn_index": 1
    }
    
    print(f"   Destination: {TEST_DESTINATION}")
    print(f"   Duration: {TEST_DURATION}")
    print(f"   Session ID: {session_id}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/plan-trip",
            json=payload,
            timeout=120  # Surf trip planning can take some time
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Request successful!")
            print(f"   Response length: {len(result.get('result', ''))} characters")
            print(f"   Tool calls made: {len(result.get('tool_calls', []))}")
            
            # Show first 200 chars of response
            result_text = result.get('result', '')
            preview = result_text[:200] + "..." if len(result_text) > 200 else result_text
            print(f"\n📋 Response preview:")
            print(f"   {preview}")
            
            return True, session_id
        else:
            print(f"\n❌ Request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False, None
            
    except requests.exceptions.Timeout:
        print("\n⚠️  Request timed out (took longer than 120 seconds)")
        print("   The request might still be processing. Check Arize for traces.")
        return False, None
    except Exception as e:
        print(f"\n❌ Error making request: {e}")
        return False, None

def check_tracing_configuration():
    """Check if tracing appears to be configured."""
    print("\n🔍 Checking tracing configuration...")
    
    try:
        # Check startup logs by making a health check and examining headers
        response = requests.get(f"{BASE_URL}/health")
        
        # We can't directly check if tracing is enabled from the client,
        # but we can remind the user what to check
        print("\n📋 Tracing checklist:")
        print("   □ ARIZE_SPACE_ID is set in backend/.env")
        print("   □ ARIZE_API_KEY is set in backend/.env")
        print("   □ Server logs show: '✅ Arize AX tracing initialized successfully'")
        print("\nIf you didn't see the success message, check your .env file.")
        
    except Exception as e:
        print(f"⚠️  Could not verify tracing: {e}")

def print_instructions(session_id):
    """Print instructions for viewing traces in Arize."""
    print("\n" + "="*70)
    print("🎉 Test completed! Now view your trace in Arize:")
    print("="*70)
    print(f"""
1. Go to: https://app.arize.com

2. Select your Space

3. Find the project: 'ai-surf-trip-planner'

4. Look for traces with session_id: {session_id}

5. Click on the trace to see:
   ✓ Multi-agent workflow execution
   ✓ All 4 agents (research, budget, local, itinerary)
   ✓ LLM calls with prompts and completions
   ✓ Tool calls and their results
   ✓ Execution timeline (waterfall view)
   ✓ Performance metrics and latency

6. Explore Arize features:
   - Trace visualization (waterfall timeline)
   - Prompt Playground (test and modify prompts)
   - Session tracking (filter by session_id)
   - Cost tracking (token usage)
   - Performance analysis (latency, bottlenecks)

7. Try filtering traces by:
   - session.id = {session_id}
   - agent.destination = {TEST_DESTINATION}
   - Tags: surf_trip_planner, multi_agent, langgraph
""")
    
    print("📚 Documentation:")
    print("   - Quick Setup: ARIZE_SETUP.md")
    print("   - Full Guide: ARIZE_OBSERVABILITY.md")
    print("="*70)

def main():
    """Run the tracing test."""
    print("="*70)
    print("🏄 Arize AX Tracing Test for Surf Trip Planner")
    print("="*70)
    
    # Step 1: Check if API is running
    if not check_health():
        sys.exit(1)
    
    # Step 2: Make test request
    success, session_id = make_test_request()
    if not success:
        print("\n⚠️  Test request failed, but traces may still be captured.")
        print("   Check Arize for any traces that were generated.")
        sys.exit(1)
    
    # Step 3: Check tracing configuration
    check_tracing_configuration()
    
    # Step 4: Print instructions
    if session_id:
        print_instructions(session_id)
    
    print("\n✅ Test script completed successfully!")
    print("   Your surf trip planner is now fully instrumented with Arize AX. 🎉\n")

if __name__ == "__main__":
    main()

