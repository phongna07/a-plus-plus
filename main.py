import rumps
import subprocess
import os
import threading
import logging
import sys
import re
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

PRIMARY_MODEL = "gemini-3-flash-preview"
FALLBACK_MODEL = "gemini-2.5-flash"

# System prompt for the grading agent
GRADING_PROMPT = """
You are a specialized grading agent. Your only function is to output uppercase answer choices only, with no explanation, punctuation, or extra text.

Task:

Analyze the provided screenshot of a computer screen.

Identify the first complete multiple-choice question that appears on the screen.

The question may be either:
- single-answer (radio style), or
- multiple-answer (checkbox style, select all that apply).

Based only on the screenshot, determine the correct answer choice(s) for that first complete question.

Output Rules:

- If the question has one correct choice, output one uppercase letter (A-F), for example: B
- If the question has multiple correct choices, output all correct uppercase letters in alphabetical order with no separators, for example: AC or BDF
- If no complete multiple-choice question is found, output exactly: ?

Remember: Output only the answer token(s).

Here is the screenshot:"""


def parse_answer_choices(raw_solution: str) -> str:
    """Normalize model output to '?' or one/more uppercase choices (A-F)."""
    if not raw_solution:
        return "?"

    cleaned = raw_solution.strip().upper()
    if not cleaned:
        return "?"

    if cleaned == "?":
        return "?"

    # Preferred format: standalone choice tokens such as "A", "A C", or "A, C".
    token_matches = re.findall(r"\b([A-F])\b", cleaned)
    if token_matches:
        return "".join(sorted(set(token_matches)))

    # Fallback: compact format such as "AC".
    compact = re.sub(r"[^A-F]", "", cleaned)
    if compact:
        return "".join(sorted(set(compact)))

    return "?"


def call_gemini_api(gemini_client: genai.Client, image: Image.Image, model: str) -> str:
    """Call Gemini API with the given client, image, and model."""
    response = gemini_client.models.generate_content(
        model=model,
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
        self.left_ctrl_pressed = False

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

    def _update_title(self):
        """Update menu bar title based on show_answer state."""
        with self.lock:
            if self.show_answer:
                self.title = f"{self.screenshot_count}{self.actual_answer}"
            else:
                self.title = "20ºC"

    def _start_keyboard_listener(self):
        """Start a keyboard listener for shift visibility and left-control capture."""

        def on_press(key):
            if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
                self.show_answer = True
                self._update_title()
            elif key == keyboard.Key.ctrl_l and not self.left_ctrl_pressed:
                self.left_ctrl_pressed = True
                self._trigger_capture()

        def on_release(key):
            if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
                self.show_answer = False
                self._update_title()
            elif key == keyboard.Key.ctrl_l:
                self.left_ctrl_pressed = False

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

    def _trigger_capture(self):
        """Capture a screenshot in a background thread when triggered by hotkey."""
        thread = threading.Thread(target=self._take_screenshot, daemon=True)
        thread.start()

    def _analyze_screenshot_with_gemini(self, filepath: str) -> str:
        """Analyze screenshot using Gemini with model/API-key fallback layers."""
        try:
            image = Image.open(filepath)
        except Exception as e:
            logger.error(f"Failed to open screenshot image: {e}")
            sys.exit(1)

        secondary_client = genai.Client(api_key=SECONDARY_API_KEY) if SECONDARY_API_KEY else None

        # Fallback order:
        # 1) gemini-3-flash-preview + primary key
        # 2) gemini-3-flash-preview + secondary key
        # 3) gemini-2.5-flash + primary key
        # 4) gemini-2.5-flash + secondary key
        attempts = [
            (PRIMARY_MODEL, client, "primary API key"),
            (PRIMARY_MODEL, secondary_client, "secondary API key"),
            (FALLBACK_MODEL, client, "primary API key"),
            (FALLBACK_MODEL, secondary_client, "secondary API key"),
        ]

        for model, attempt_client, key_label in attempts:
            if attempt_client is None:
                continue

            try:
                solution = call_gemini_api(attempt_client, image, model)

                print(
                    f"\n[{datetime.now().isoformat()}] Screenshot: {os.path.basename(filepath)}"
                )
                print(f"Model: {model} ({key_label})")
                print(f"Solution: {solution}\n")

                return parse_answer_choices(solution)
            except Exception as e:
                logger.error(f"Gemini API analysis failed for {model} with {key_label}: {e}")

        logger.error("Gemini API analysis failed for all fallback layers")
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


if __name__ == "__main__":
    AwesomeStatusBarApp().run()
