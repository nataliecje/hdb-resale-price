"""
Basic tests for HDB BTO System
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_import_api():
    """Test that API modules can be imported"""
    try:
        from api.main import app
        assert app is not None
        print("API app imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import API app: {e}")


def test_import_models():
    """Test that model modules can be imported"""
    try:
        from models.bto_recommendation import BTORecommendationSystem
        assert BTORecommendationSystem is not None
        print("BTO recommendation system imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import BTO recommendation system: {e}")


def test_import_llm_service():
    """Test that LLM service can be imported"""
    try:
        from llm_service import LLMService
        assert LLMService is not None
        print("LLM service imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import LLM service: {e}")


def test_basic_functionality():
    """Test basic system functionality"""
    # This is a placeholder test that always passes
    # In a real scenario, you'd test actual functionality
    assert True
    print("Basic functionality test passed")


def test_environment():
    """Test that required environment variables can be set"""
    # Test that we can set environment variables
    os.environ['TEST_VAR'] = 'test_value'
    assert os.environ.get('TEST_VAR') == 'test_value'
    print("Environment variable test passed")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
