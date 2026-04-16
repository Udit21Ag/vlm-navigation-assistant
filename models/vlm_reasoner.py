import base64
import json
import os
from urllib import error, request
import urllib
from PIL import Image
import cv2
import google.generativeai as genai


class VLMReasoner:
    def __init__(self, model_name=None):

        self.timeout_seconds = float(os.getenv("VLM_TIMEOUT_SECONDS", "10"))
        self.provider = os.getenv("VLM_PROVIDER", "").strip().lower()
        self.available = False
        self.model = None

        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        if not self.provider:
            self.provider = "openrouter" if openrouter_key else "gemini" if gemini_key else "off"

        try:
            if self.provider == "openrouter":
                self.api_key = openrouter_key
                self.model_name = model_name or os.getenv(
                    "OPENROUTER_MODEL",
                    "qwen/qwen2.5-vl-7b-instruct:free",
                )
                self.api_url = os.getenv(
                    "OPENROUTER_API_URL",
                    "https://openrouter.ai/api/v1/chat/completions",
                )
                self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
                self.app_title = os.getenv("OPENROUTER_APP_TITLE", "dl-navigation-assistant")

                if not self.api_key:
                    raise RuntimeError("Missing OPENROUTER_API_KEY")

                self.available = True
                print(f"[VLM] Using OpenRouter ({self.model_name}).")

            elif self.provider == "gemini":
                self.api_key = gemini_key
                self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

                if not self.api_key:
                    raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY)")

                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self.available = True

                print(f"[VLM] Using Gemini API ({self.model_name}).")

            else:
                self.model_name = model_name or "off"
                print("[VLM] Disabled by configuration.")

        except Exception as exc:
            print(f"[VLM] Disabled: {exc}")

    # ------------------------------------------------------
    # Build prompt from YOUR pipeline outputs
    # ------------------------------------------------------
    def _build_prompt(self, temporal_objects, instruction):

        if not temporal_objects:
            scene_desc = "No major obstacles."
        else:
            parts = []
            for obj in temporal_objects[:3]:  # 🔥 limit for speed
                label = obj.get("label", "object")
                zone = obj.get("zone", "center")
                motion = obj.get("motion", "stationary")
                parts.append(f"{label} {motion} in {zone}")

            scene_desc = ", ".join(parts)

        return (
            "You are helping a visually impaired person navigate safely.\n"
            f"Observed scene: {scene_desc}.\n"
            f"Current navigation suggestion: {instruction}.\n"
            "Generate one short, clear navigation instruction.\n"
            "Mention the safest direction.\n"
            "Keep the answer under 10 words."
        )

    # ------------------------------------------------------
    # Fallback (VERY IMPORTANT)
    # ------------------------------------------------------
    def _fallback_caption(self, instruction, temporal_objects):

        if not temporal_objects:
            return instruction

        top = temporal_objects[0]
        label = top.get("label", "object")
        zone = top.get("zone", "center")
        motion = top.get("motion", "stationary")

        if motion == "approaching":
            return f"{instruction}. {label} approaching from {zone}."
        return f"{instruction}. {label} detected on the {zone}."

    def _prepare_image_data_url(self, frame):
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("Failed to encode frame for VLM request")

        encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_generation_config(self):
        return {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 16,
            "max_output_tokens": 32,
        }

    def _call_openrouter(self, frame, prompt):
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._prepare_image_data_url(frame)
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "top_p": 0.8,
            "max_tokens": 32,
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "LowVisionNavigation"
            }
        )

        with request.urlopen(req, timeout=self.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))

        return str(content).strip()

    # ------------------------------------------------------
    # Main inference
    # ------------------------------------------------------
    def generate(self, frame, temporal_objects, instruction):

        # 🔥 If model failed → fallback
        if not self.available:
            return self._fallback_caption(instruction, temporal_objects)

        try:
            prompt = self._build_prompt(temporal_objects, instruction)

            if self.provider == "openrouter":
                text = self._call_openrouter(frame, prompt)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)

                response = self.model.generate_content(
                    [prompt, image],
                    generation_config=self._build_generation_config(),
                    request_options={"timeout": self.timeout_seconds},
                )
                text = (response.text or "").strip()

            if not text:
                return self._fallback_caption(instruction, temporal_objects)

            text = text.replace("ASSISTANT:", "").strip()

            if len(text) > 180:
                text = text[:180].rsplit(" ", 1)[0].strip()

            return text

        except (error.URLError, error.HTTPError, TimeoutError, Exception) as e:
            print("[VLM ERROR]", e)
            return self._fallback_caption(instruction, temporal_objects)