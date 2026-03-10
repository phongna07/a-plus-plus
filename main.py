import rumps
import subprocess
import os
import threading
import logging
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from PIL import Image
from pynput import keyboard

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize client with primary API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Secondary API key for fallback
SECONDARY_API_KEY = os.getenv("GEMINI_API_KEY_SECONDARY")

# System prompt for the grading agent
GRADING_PROMPT = """
You are a specialized grading agent. Your only function is to output a single, uppercase character that represents the correct answer choice. You must provide no other text, explanation, or punctuation.

Task:

Analyze the provided screenshot of a computer screen.

Identify the first complete multiple-choice question that appears on the screen.

Based only on the content of the screenshot, determine the correct answer for that question.

Output Rules:

Respond with the single uppercase letter of the correct answer (e.g., 'A', 'B', 'C', 'D', 'E', or 'F').

If no multiple-choice question is found, your single and only output must be the question mark ('?').

Remember: Output only the single character.

Here is the screenshot:"""


def call_gemini_api(gemini_client: genai.Client, image: Image.Image) -> str:
    """Call Gemini API with the given client and image."""
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[GRADING_PROMPT, image],
    )
    return response.text or ""


class AwesomeStatusBarApp(rumps.App):
    def __init__(self):
        super(AwesomeStatusBarApp, self).__init__("20ºC")
        self.screenshot_count = 0
        self.actual_answer = "?"
        self.show_answer = False
        self.lock = threading.Lock()
        self.capture_timer = None

        # Create screenshots folder if it doesn't exist
        self.screenshots_dir = os.path.join(os.getcwd(), "screenshots")
        try:
            os.makedirs(self.screenshots_dir, exist_ok=True)
            # Empty the screenshots folder
            for filename in os.listdir(self.screenshots_dir):
                filepath = os.path.join(self.screenshots_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
        except Exception as e:
            logger.error(f"Failed to create or empty screenshots directory: {e}")
            sys.exit(1)

        # Start keyboard listener for shift key detection
        self._start_keyboard_listener()

        # Add 1 seconds delay since initialization before scheduling the first capture
        time.sleep(1)
        # Start the first capture after the delay
        self._schedule_next_capture()

    def _schedule_next_capture(self):
        """Schedule the next screenshot capture after 1 seconds."""
        if self.capture_timer:
            self.capture_timer.cancel()

        self.capture_timer = threading.Timer(1, self._capture_and_schedule)
        self.capture_timer.daemon = True
        self.capture_timer.start()

    def _update_title(self):
        """Update menu bar title based on show_answer state."""
        with self.lock:
            if self.show_answer:
                self.title = f"{self.screenshot_count}{self.actual_answer}"
            else:
                self.title = "20ºC"

    def _start_keyboard_listener(self):
        """Start a keyboard listener to detect shift key presses."""

        def on_press(key):
            if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
                self.show_answer = True
                self._update_title()

        def on_release(key):
            if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
                self.show_answer = False
                self._update_title()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

    def _capture_and_schedule(self):
        """Capture a screenshot and schedule the next one."""
        thread = threading.Thread(target=self._take_screenshot, daemon=True)
        thread.start()

    def _analyze_screenshot_with_gemini(self, filepath: str) -> str:
        """Analyze screenshot using Gemini 2.5 Flash API."""
        try:
            # Load image with Pillow
            image = Image.open(filepath)

            # Call Gemini 2.5 Flash API using the primary client
            solution = call_gemini_api(client, image)

            # Print full solution to console with context
            print(
                f"\n[{datetime.now().isoformat()}] Screenshot: {os.path.basename(filepath)}"
            )
            print(f"Solution: {solution}\n")

            # Return first character
            return solution[0] if solution else "?"

        except Exception as e:
            logger.error(f"Gemini API analysis failed with primary key: {e}")

            # Try with secondary API key if available
            if SECONDARY_API_KEY:
                try:
                    logger.info("Attempting analysis with secondary API key...")
                    secondary_client = genai.Client(api_key=SECONDARY_API_KEY)

                    # Load image again
                    image = Image.open(filepath)

                    # Call Gemini 2.5 Flash API using the secondary client
                    solution = call_gemini_api(secondary_client, image)

                    # Print full solution to console with context
                    print(
                        f"\n[{datetime.now().isoformat()}] Screenshot: {os.path.basename(filepath)} (using secondary API key)"
                    )
                    print(f"Solution: {solution}\n")

                    # Return first character
                    return solution[0] if solution else "?"

                except Exception as secondary_e:
                    logger.error(
                        f"Gemini API analysis also failed with secondary key: {secondary_e}"
                    )
                    sys.exit(1)
            else:
                logger.error("No secondary API key available for fallback")
                sys.exit(1)

    def _take_screenshot(self):
        """Execute screencapture command in a non-blocking manner."""
        try:
            # Generate ISO datetime filename (replace colons with hyphens for filesystem compatibility)
            timestamp = datetime.now().isoformat().replace(":", "-")
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(self.screenshots_dir, filename)

            # Execute screencapture command
            result = subprocess.run(
                ["screencapture", filepath], capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                logger.error(
                    f"screencapture failed with return code {result.returncode}: {result.stderr}"
                )
                sys.exit(1)

            # Analyze screenshot with Gemini and update title
            first_char = self._analyze_screenshot_with_gemini(filepath)

            with self.lock:
                self.screenshot_count += 1
                self.actual_answer = first_char

            self._update_title()

        except Exception as e:
            logger.error(f"Exception during screenshot capture: {e}")
            sys.exit(1)
        finally:
            # Schedule the next capture after this one completes
            self._schedule_next_capture()


if __name__ == "__main__":
    AwesomeStatusBarApp().run()
