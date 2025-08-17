"""
Integration tests for API
"""

import pytest
import requests
import time

def test_api_health():
    """Test API health endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "api_status" in data
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")

def test_api_estates():
    """Test API estates endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/estates", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "estates" in data
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")
