"""
Hardware abstraction layer for PlantaeClassScanner
Allows development on non-RasPi systems (WSL, macOS, Windows)

Author: Abhinav M
"""

import sys
import platform

# Detect if running on Raspberry Pi
IS_RASPBERRY_PI = platform.machine() in ['aarch64', 'armv7l']

if IS_RASPBERRY_PI:
    try:
        import RPi.GPIO as GPIO
        # import gpiozero # hardware specific
        from picamera2 import Picamera2
        HARDWARE_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️  RasPi hardware libraries not available: {e}")
        HARDWARE_AVAILABLE = False
else:
    print("💻 Running in development mode (non-RasPi platform)")
    HARDWARE_AVAILABLE = False


class TriggerButton:
    """Abstraction for trigger button with mock support"""
    
    def __init__(self, pin=17):
        self.pin = pin
        self.callback = None
        
        if HARDWARE_AVAILABLE:
            self._setup_real_gpio()
        else:
            self._setup_mock_gpio()
    
    def _setup_real_gpio(self):
        """Setup actual GPIO on RasPi"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print(f"✅ GPIO trigger initialized on pin {self.pin}")
    
    def _setup_mock_gpio(self):
        """Setup keyboard simulation for development"""
        print(f"🔧 Mock GPIO: Press ENTER to simulate trigger on pin {self.pin}")
    
    def on_press(self, callback):
        """Register callback for trigger press"""
        self.callback = callback
        
        if HARDWARE_AVAILABLE:
            GPIO.add_event_detect(self.pin, GPIO.FALLING, 
                                callback=lambda ch: callback(), 
                                bouncetime=200)
        else:
            # Mock: user presses ENTER
            import threading
            def mock_loop():
                while True:
                    input()  # Wait for ENTER
                    if self.callback:
                        self.callback()
            
            thread = threading.Thread(target=mock_loop, daemon=True)
            thread.start()
    
    def cleanup(self):
        """Cleanup GPIO resources"""
        if HARDWARE_AVAILABLE:
            GPIO.cleanup()


class StatusLED:
    """Abstraction for status LED with mock support"""
    
    def __init__(self, pin=27):
        self.pin = pin
        self.state = False
        
        if HARDWARE_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW)
            print(f"✅ LED initialized on pin {self.pin}")
        else:
            print(f"🔧 Mock LED on pin {self.pin}")
    
    def on(self):
        """Turn LED on"""
        self.state = True
        if HARDWARE_AVAILABLE:
            GPIO.output(self.pin, GPIO.HIGH)
        else:
            print("💡 LED: ON")
    
    def off(self):
        """Turn LED off"""
        self.state = False
        if HARDWARE_AVAILABLE:
            GPIO.output(self.pin, GPIO.LOW)
        else:
            print("💡 LED: OFF")
    
    def blink(self, duration=0.1):
        """Blink LED briefly"""
        import time
        self.on()
        time.sleep(duration)
        self.off()


class Camera:
    """Abstraction for camera with mock support"""
    
    def __init__(self):
        if HARDWARE_AVAILABLE:
            try:
                self.camera = Picamera2()
                self.camera.configure(self.camera.create_still_configuration())
                self.camera.start()
                print("✅ Camera initialized")
            except Exception as e:
                print(f"⚠️  Camera initialization failed: {e}")
                self.camera = None
        else:
            self.camera = None
            print("🔧 Mock camera (will use test images)")
    
    def capture(self):
        """Capture image from camera"""
        if HARDWARE_AVAILABLE and self.camera:
            import numpy as np
            # Capture image as numpy array
            image = self.camera.capture_array()
            return image
        else:
            # Return mock image for testing
            import numpy as np
            print("📸 Mock capture: Using test image")
            # Return blank 800x480 RGB image (matching your display)
            return np.zeros((480, 800, 3), dtype=np.uint8)
    
    def close(self):
        """Release camera resources"""
        if HARDWARE_AVAILABLE and self.camera:
            self.camera.stop()
            self.camera.close()


# Quick test
if __name__ == "__main__":
    print("🧪 Testing Hardware Abstraction Layer")
    print("=" * 50)
    
    # Test trigger
    trigger = TriggerButton()
    led = StatusLED()
    camera = Camera()
    
    def on_trigger():
        print("🔫 Trigger pressed!")
        led.blink()
    
    trigger.on_press(on_trigger)
    
    if not HARDWARE_AVAILABLE:
        print("\n💡 Press ENTER to simulate trigger (Ctrl+C to exit)")
    
    try:
        import time
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        trigger.cleanup()
        camera.close()