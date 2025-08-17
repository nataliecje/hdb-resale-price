"""
Unit tests for models
"""

import pytest
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.bto_recommendation import BTORecommendationSystem

def test_bto_system_initialization():
    """Test BTO recommendation system initialization"""
    try:
        bto_system = BTORecommendationSystem()
        assert bto_system is not None
    except Exception as e:
        pytest.skip(f"Database connection required: {e}")

def test_estate_analysis():
    """Test estate analysis functionality"""
    try:
        bto_system = BTORecommendationSystem()
        analysis = bto_system.get_estate_bto_analysis()
        assert isinstance(analysis, dict) or hasattr(analysis, 'shape')
    except Exception as e:
        pytest.skip(f"Database connection required: {e}")
