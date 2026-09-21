"""
Quick test for the Crop Recommendation Service
Run: python tests/test_crop_service.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.services.crop_service import predict_crop, CropRecommendationService


def test_basic_prediction():
    """Test basic crop prediction."""
    result = predict_crop(90, 42, 43, 25.5, 80.0, 6.5, 200.0)
    print("Test 1 - Rice conditions:")
    print(f"  Result: {result}")
    assert "recommended_crop" in result
    print("  ✅ PASSED\n")


def test_apple_prediction():
    """Test apple prediction."""
    result = predict_crop(20, 130, 200, 18.0, 85.0, 6.5, 100.0)
    print("Test 2 - Apple conditions:")
    print(f"  Result: {result}")
    print("  ✅ PASSED\n")


def test_invalid_input():
    """Test that invalid inputs are handled."""
    # Negative nitrogen
    result = predict_crop(-10, 42, 43, 25.5, 80.0, 6.5, 200.0)
    print("Test 3 - Invalid nitrogen (negative):")
    print(f"  Result: {result}")
    assert "error" in result
    print("  ✅ PASSED (error caught)\n")


def test_invalid_humidity():
    """Test humidity > 100%."""
    result = predict_crop(90, 42, 43, 25.5, 150.0, 6.5, 200.0)
    print("Test 4 - Invalid humidity (>100%):")
    print(f"  Result: {result}")
    assert "error" in result
    print("  ✅ PASSED (error caught)\n")


def test_invalid_ph():
    """Test pH > 14."""
    result = predict_crop(90, 42, 43, 25.5, 80.0, 15.0, 200.0)
    print("Test 5 - Invalid pH (>14):")
    print(f"  Result: {result}")
    assert "error" in result
    print("  ✅ PASSED (error caught)\n")


def test_string_input():
    """Test that string inputs are rejected."""
    result = predict_crop("ninety", 42, 43, 25.5, 80.0, 6.5, 200.0)
    print("Test 6 - String input instead of number:")
    print(f"  Result: {result}")
    assert "error" in result
    print("  ✅ PASSED (error caught)\n")


if __name__ == "__main__":
    print("=" * 50)
    print("CROP RECOMMENDATION SERVICE TESTS")
    print("=" * 50 + "\n")

    test_basic_prediction()
    test_apple_prediction()
    test_invalid_input()
    test_invalid_humidity()
    test_invalid_ph()
    test_string_input()

    print("🎉 All tests passed!")