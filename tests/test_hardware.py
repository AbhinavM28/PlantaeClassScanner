"""
Unit tests for hardware abstraction layer
Tests both mock (WSL) and real (RasPi) modes

Author: Abhinav M
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hardware.gpio_interface import (
    TriggerButton, 
    StatusLED, 
    Camera, 
    HARDWARE_AVAILABLE
)

try:
    from src.utils.logger import setup_logger
    logger = setup_logger("HardwareTest")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("HardwareTest")


def test_trigger():
    """Test trigger button functionality"""
    logger.info("🧪 Testing Trigger Button...")
    
    trigger_count = [0]  # Use list for mutable counter in closure
    
    def on_trigger():
        trigger_count[0] += 1
        logger.info(f"  🔫 Trigger pressed! Count: {trigger_count[0]}")
    
    trigger = TriggerButton()
    trigger.on_press(on_trigger)
    
    if not HARDWARE_AVAILABLE:
        logger.info("  💡 Mock mode: Press ENTER 3 times (10 second window)...")
        start_time = time.time()
        while time.time() - start_time < 10:
            time.sleep(0.1)
    else:
        logger.info("  Press hardware trigger 3 times (10 second window)...")
        time.sleep(10)
    
    trigger.cleanup()
    logger.info(f"  ✅ Trigger test complete. Total presses: {trigger_count[0]}")
    return trigger_count[0] > 0


def test_led():
    """Test LED functionality"""
    logger.info("🧪 Testing Status LED...")
    
    led = StatusLED()
    
    # Test on/off
    logger.info("  Testing on/off...")
    led.on()
    time.sleep(0.5)
    led.off()
    time.sleep(0.5)
    
    # Test blink
    logger.info("  Testing blink (3 times)...")
    for i in range(3):
        led.blink(0.2)
        time.sleep(0.3)
    
    logger.info("  ✅ LED test complete")
    return True


def test_camera():
    """Test camera functionality"""
    logger.info("🧪 Testing Camera...")
    
    camera = Camera()
    
    # Capture test image
    logger.info("  Capturing test image...")
    img = camera.capture()
    logger.info(f"  Captured image shape: {img.shape}")
    
    # Verify dimensions
    if img.shape[2] != 3:
        logger.error(f"  ❌ Expected 3 channels (RGB), got {img.shape[2]}")
        camera.close()
        return False
    
    camera.close()
    logger.info("  ✅ Camera test complete")
    return True


def run_all_tests():
    """Run complete hardware test suite"""
    logger.info("=" * 70)
    logger.info("🔬 HARDWARE ABSTRACTION LAYER TEST SUITE")
    logger.info("=" * 70)
    logger.info(f"Mode: {'RasPi Hardware' if HARDWARE_AVAILABLE else 'Development/Mock'}")
    logger.info("")
    
    results = {
        "trigger": False,
        "led": False,
        "camera": False
    }
    
    # Run tests
    try:
        results["led"] = test_led()
    except Exception as e:
        logger.error(f"LED test failed: {e}")
    
    logger.info("")
    
    try:
        results["camera"] = test_camera()
    except Exception as e:
        logger.error(f"Camera test failed: {e}")
    
    logger.info("")
    
    try:
        results["trigger"] = test_trigger()
    except Exception as e:
        logger.error(f"Trigger test failed: {e}")
    
    # Print results
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST RESULTS")
    logger.info("=" * 70)
    
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {component.upper()}")
    
    all_passed = all(results.values())
    
    logger.info("=" * 70)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.warning("⚠️  Some tests failed. Check logs above.")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)