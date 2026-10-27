import os
import sys
import threading
import time
import datetime

import addonHandler
import globalPluginHandler
import ui
import tones
import speech

addonHandler.initTranslation()

# FaceGuide Modern - accessible spoken corrections and movement beeps.
LIB = os.path.join(os.path.dirname(__file__), "lib")
BASE = os.path.dirname(os.path.dirname(__file__))

if LIB not in sys.path:
    sys.path.insert(0, LIB)

if hasattr(os, "add_dll_directory"):
    for dll_dir in (
        os.path.join(LIB, "opencv_python.libs"),
        os.path.join(LIB, "numpy.libs"),
    ):
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except OSError:
                pass

import cv2


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
    scriptCategory = "FaceGuide Modern"

    # A new state must be seen consistently before it is announced.
    STABLE_FRAMES = 4
    DIRECTION_REPEAT_SECONDS = 10.0
    CENTER_STABLE_SECONDS = 0.8
    DARK_THRESHOLD = 58.0
    TILT_ANGLE_THRESHOLD = 14.0
    TILT_CLEAR_THRESHOLD = 8.0
    DARK_CONFIRM_SECONDS = 2.0
    DARK_REPEAT_SECONDS = 15.0
    LIGHT_OK_THRESHOLD = 72.0

    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        self.worker = None
        self.last_message = ""
        self.last_message_time = 0.0
        self.last_tone = ""
        self.last_tone_time = 0.0
        self._last_guidance_beep = 0.0
        self._candidate_state = None
        self._candidate_count = 0
        self._active_state = None
        self._active_since = 0.0
        self._last_state_announcement = 0.0
        self._dark_since = None
        self._last_dark_announcement = 0.0
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._selfie_thread = None
        self.selfie_active = False
        self._last_tilt_state = None
        self._last_tilt_announcement = 0.0
        self._last_lighting_ok_announcement = 0.0
        self._lighting_state = None
        self._speech_priority_until = 0.0
        self._tilt_clear_since = None
        self._camera_face_detected = False
        self._camera_distance_state = "unknown"
        self._camera_lighting_state = "unknown"
        self._last_status = ""

    def terminate(self):
        self.stop(silent=True)
        super().terminate()

    def speak(self, message, force=False):
        now = time.monotonic()
        if force or message != self.last_message or now - self.last_message_time >= 2.0:
            self.last_message = message
            self.last_message_time = now
            ui.message(message)

    def play_guidance_tone(self, direction):
        now = time.monotonic()
        if direction == self.last_tone and now - self.last_tone_time < 0.15:
            return
        self.last_tone = direction
        self.last_tone_time = now
        try:
            frequencies = {
                "left": (520, 90), "right": (900, 90),
                "up": (1100, 120), "down": (380, 120),
                "closer": (760, 70), "farther": (600, 70),
                "center": (1200, 140),
            }
            if direction in frequencies:
                tones.beep(*frequencies[direction])
        except Exception:
            pass

    def start(self):
        if self.running:
            self.speak("FaceGuide Modern is already running.", force=True)
            return
        self.running = True
        self._candidate_state = None
        self._candidate_count = 0
        self._active_state = None
        self._dark_since = None
        self.speak("FaceGuide Modern started.", force=True)
        try:
            tones.beep(880, 80)
        except Exception:
            pass
        self.worker = threading.Thread(target=self.camera_loop, name="FaceGuideModernCamera", daemon=True)
        self.worker.start()

    def stop(self, silent=False):
        self.running = False
        self._camera_face_detected = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if not silent:
            self.speak("FaceGuide Modern stopped.", force=True)
            try:
                tones.beep(440, 80)
            except Exception:
                pass

    def _classify_position(self, center_x, center_y, width, height, face_ratio, face_width=None):
        """Return one actionable correction, prioritising distance independently of alignment."""
        nx = center_x / float(width)
        ny = center_y / float(height)
        previous = self._active_state
        width_ratio = (face_width / float(width)) if face_width is not None else (face_ratio ** 0.5)

        # Use face width, rather than area alone, as a more stable rough distance proxy.
        # These are camera-relative estimates, not physical distance measurements.
        if previous == "closer" and width_ratio < 0.22:
            return "closer", "Your face is small in the camera view. Move about 5 centimeters closer."
        if previous == "farther" and width_ratio > 0.40:
            return "farther", "Your face is large in the camera view. Move about 5 centimeters away."
        if width_ratio < 0.16:
            return "closer", "Your face is small in the camera view. Move about 5 centimeters closer."
        if width_ratio > 0.48:
            return "farther", "Your face is large in the camera view. Move about 5 centimeters away."

        # Horizontal corrections use physical movement directions.
        if previous == "left" and nx < 0.40:
            return "left", "Your face is left of the target area. Move your head about 2 centimeters to your RIGHT."
        if previous == "right" and nx > 0.60:
            return "right", "Your face is right of the target area. Move your head about 2 centimeters to your LEFT."
        if nx < 0.30:
            return "left", "Your face is left of the target area. Move your head about 2 centimeters to your RIGHT."
        if nx > 0.70:
            return "right", "Your face is right of the target area. Move your head about 2 centimeters to your LEFT."

        # Image y grows downward: a face high in the image calls for lowering the head.
        if previous == "up" and ny < 0.40:
            return "up", "Your face is too high. Lower your head about 2 centimeters."
        if previous == "down" and ny > 0.60:
            return "down", "Your face is too low. Raise your head about 2 centimeters."
        if ny < 0.30:
            return "up", "Your face is too high. Lower your head about 2 centimeters."
        if ny > 0.70:
            return "down", "Your face is too low. Raise your head about 2 centimeters."
        return "center", "Face position is good. Keep your head still."

    def _detect_tilt(self, gray, face_rect):
        """Return roll tilt in degrees when both eyes can be detected."""
        try:
            x, y, w, h = face_rect
            eye_path = os.path.join(BASE, "faceGuide_data", "haarcascade_eye.xml")
            if not os.path.exists(eye_path):
                eye_path = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_eye.xml")
            eyes_detector = getattr(self, "_eyes_detector", None)
            if eyes_detector is None:
                eyes_detector = cv2.CascadeClassifier(eye_path)
                self._eyes_detector = eyes_detector
            if eyes_detector.empty():
                return None
            face_gray = gray[y:y+h, x:x+w]
            eyes = eyes_detector.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12))
            # Keep eyes in the upper half and use the two most horizontally separated.
            eyes = [(ex, ey, ew, eh) for ex, ey, ew, eh in eyes if ey < h * 0.65]
            if len(eyes) < 2:
                return None
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye, right_eye = eyes[0], eyes[-1]
            lx, ly = left_eye[0] + left_eye[2] / 2.0, left_eye[1] + left_eye[3] / 2.0
            rx, ry = right_eye[0] + right_eye[2] / 2.0, right_eye[1] + right_eye[3] / 2.0
            if rx <= lx:
                return None
            return float(__import__("math").degrees(__import__("math").atan2(ry - ly, rx - lx)))
        except Exception:
            return None

    def _announce_tilt_if_needed(self, angle):
        """Announce persistent tilt; tolerate brief eye-detector failures."""
        now = time.monotonic()
        if angle is None:
            # Do not forget a tilt after a single frame where the eyes are missed.
            if self._tilt_clear_since is not None and now - self._tilt_clear_since >= 1.5:
                self._last_tilt_state = None
            return
        if abs(angle) < self.TILT_CLEAR_THRESHOLD:
            if self._tilt_clear_since is None:
                self._tilt_clear_since = now
            if now - self._tilt_clear_since >= 1.0:
                self._last_tilt_state = None
            return
        self._tilt_clear_since = None
        if abs(angle) < self.TILT_ANGLE_THRESHOLD:
            return
        state = "tilt_right" if angle > 0 else "tilt_left"
        if state != self._last_tilt_state or now - self._last_tilt_announcement >= 15.0:
            if angle > 0:
                self.speak("Your head is tilted to the right. To straighten it, raise your right ear slightly. Move slowly.")
            else:
                self.speak("Your head is tilted to the left. To straighten it, raise your left ear slightly. Move slowly.")
            self._last_tilt_state = state
            self._last_tilt_announcement = now

    def _update_guidance(self, state, message, announce=True):
        now = time.monotonic()
        if state == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = state
            self._candidate_count = 1

        if self._candidate_count < self.STABLE_FRAMES:
            return

        changed = state != self._active_state
        if changed:
            self._active_state = state
            self._active_since = now
            # Only announce centered after it remains stable briefly.
            if state == "center":
                self._last_state_announcement = 0.0
                return
            self._last_state_announcement = now
            if announce and now >= self._speech_priority_until:
                self.speak(message)
            if state == "center":
                self._last_guidance_beep = 0.0
            else:
                self.play_guidance_tone(state)
                self._last_guidance_beep = now
            return

        if state == "center":
            if now - self._active_since >= self.CENTER_STABLE_SECONDS and self._last_state_announcement == 0.0:
                self.speak("Face position is good. Keep your head still.")
                self.play_guidance_tone("center")
                self._last_state_announcement = now
            return

        # Reversing-style feedback: slower while well outside the target, faster
        # as the face approaches the acceptable region. The state classifier's
        # hysteresis keeps the beeps from flickering at the boundary.
        interval = 0.70
        if state in ("left", "right"):
            nx_error = abs((self._latest_face_nx if hasattr(self, "_latest_face_nx") else 0.0) - 0.5)
            interval = 0.70 if nx_error >= 0.20 else (0.45 if nx_error >= 0.12 else 0.28)
        elif state in ("up", "down"):
            ny_error = abs((self._latest_face_ny if hasattr(self, "_latest_face_ny") else 0.0) - 0.5)
            interval = 0.70 if ny_error >= 0.20 else (0.45 if ny_error >= 0.12 else 0.28)
        elif state in ("closer", "farther"):
            interval = 0.55
        if now - self._last_guidance_beep >= interval:
            self.play_guidance_tone(state)
            self._last_guidance_beep = now
        if now - self._last_state_announcement >= self.DIRECTION_REPEAT_SECONDS:
            if now >= self._speech_priority_until:
                self.speak(message)
            self._last_state_announcement = now

    def _check_lighting(self, face_gray, announce_recovery=True):
        """Return current lighting state and announce only meaningful state changes."""
        now = time.monotonic()
        if face_gray is None or face_gray.size == 0:
            return self._lighting_state
        brightness = float(face_gray.mean())
        if brightness < self.DARK_THRESHOLD:
            if self._dark_since is None:
                self._dark_since = now
            if now - self._dark_since >= self.DARK_CONFIRM_SECONDS:
                if self._lighting_state != "dark" or now - self._last_dark_announcement >= self.DARK_REPEAT_SECONDS:
                    self.speak("Lighting is too low on your face. Add light in front of you.")
                    self._speech_priority_until = now + 2.5
                    self._last_dark_announcement = now
                self._lighting_state = "dark"
                self._camera_lighting_state = "too dark"
        elif brightness >= self.LIGHT_OK_THRESHOLD:
            self._dark_since = None
            if self._lighting_state != "good":
                self._lighting_state = "good"
                self._camera_lighting_state = "good"
                if announce_recovery:
                    self.speak("Lighting is good.")
                    self._speech_priority_until = now + 2.0
                self._last_lighting_ok_announcement = now
        else:
            # In the middle band, preserve the previous state to prevent chatter.
            self._dark_since = None
        return self._lighting_state

    def _detect_faces_with_fallback(self, gray, primary_detector, fallback_detectors):
        """Try equalised and raw frames, then alternate cascades only if needed."""
        equalized = cv2.equalizeHist(gray)
        for image in (equalized, gray):
            faces = primary_detector.detectMultiScale(image, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40))
            if len(faces):
                return faces
        for detector in fallback_detectors:
            if detector is None or detector.empty():
                continue
            faces = detector.detectMultiScale(equalized, scaleFactor=1.08, minNeighbors=4, minSize=(40, 40))
            if len(faces):
                return faces
        return ()

    def camera_loop(self):
        cap = None
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap = cap
            if not cap.isOpened():
                self.speak("Camera could not be opened.", force=True)
                return

            cascade_path = os.path.join(BASE, "faceGuide_data", "haarcascade_frontalface_default.xml")
            if not os.path.exists(cascade_path):
                cascade_path = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
            detector = cv2.CascadeClassifier(cascade_path)
            alt_path = os.path.join(BASE, "faceGuide_data", "haarcascade_frontalface_alt2.xml")
            fallback_detector = cv2.CascadeClassifier(alt_path) if os.path.exists(alt_path) else None
            if detector.empty():
                self.speak("Face detector could not be loaded.", force=True)
                return

            self.speak("Camera ready. Looking for your face.", force=True)
            no_face_since = None
            no_face_announced = False

            while self.running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.25)
                    continue

                with self._frame_lock:
                    self._latest_frame = frame.copy()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._detect_faces_with_fallback(gray, detector, (fallback_detector,))
                height, width = gray.shape[:2]

                if len(faces) == 0:
                    self._camera_face_detected = False
                    self._camera_distance_state = "unknown"
                    now = time.monotonic()
                    if no_face_since is None:
                        no_face_since = now
                    if not self.selfie_active and now - no_face_since >= 2.5 and not no_face_announced:
                        self.speak("No face detected. Face the camera and improve the light in front of you.")
                        no_face_announced = True
                    self._candidate_state = None
                    self._candidate_count = 0
                    time.sleep(0.12)
                    continue

                no_face_since = None
                no_face_announced = False
                self._camera_face_detected = True
                x, y, face_width, face_height = max(faces, key=lambda face: int(face[2]) * int(face[3]))
                center_x = x + face_width / 2.0
                center_y = y + face_height / 2.0
                face_ratio = (face_width * face_height) / float(width * height)
                face_gray = gray[y:y + face_height, x:x + face_width]
                self._latest_face_nx = center_x / float(width)
                self._latest_face_ny = center_y / float(height)
                # Selfie mode runs its own lighting and framing checks.
                if not self.selfie_active:
                    previous_lighting = self._lighting_state
                    self._check_lighting(face_gray)
                    lighting_changed = self._lighting_state != previous_lighting
                    tilt_angle = self._detect_tilt(gray, (x, y, face_width, face_height))
                    if not lighting_changed and time.monotonic() >= self._speech_priority_until:
                        self._announce_tilt_if_needed(tilt_angle)
                    state, message = self._classify_position(center_x, center_y, width, height, face_ratio, face_width)
                    self._camera_distance_state = state if state in ("closer", "farther") else "acceptable"
                    self._update_guidance(state, message, announce=not lighting_changed)
                time.sleep(0.12)

        except Exception:
            self.speak("FaceGuide Modern encountered a camera error.", force=True)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.cap = None
            self.running = False

    def script_toggle(self, gesture):
        if self.running:
            self.stop()
        else:
            self.start()


    def script_capture_selfie(self, gesture):
        """Start an accessible smile-and-steady selfie flow without blocking NVDA."""
        if self._selfie_thread is not None and self._selfie_thread.is_alive():
            self.speak("Selfie capture is already in progress. Keep your face in view and smile.", force=True)
            return
        self.selfie_active = True
        try:
            speech.cancelSpeech()
        except Exception:
            pass
        self._selfie_thread = threading.Thread(
            target=self._capture_selfie_flow, name="FaceGuideSelfieCapture", daemon=True
        )
        self._selfie_thread.start()

    def _capture_selfie_flow(self):
        temporary_cap = None
        try:
            face_path = os.path.join(BASE, "faceGuide_data", "haarcascade_frontalface_default.xml")
            smile_path = os.path.join(BASE, "faceGuide_data", "haarcascade_smile.xml")
            if not os.path.exists(face_path):
                face_path = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
            if not os.path.exists(smile_path):
                smile_path = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_smile.xml")
            face_detector = cv2.CascadeClassifier(face_path)
            smile_detector = cv2.CascadeClassifier(smile_path)
            alt_path = os.path.join(BASE, "faceGuide_data", "haarcascade_frontalface_alt2.xml")
            fallback_face_detector = cv2.CascadeClassifier(alt_path) if os.path.exists(alt_path) else None
            if face_detector.empty() or smile_detector.empty():
                self.speak("Selfie capture is unavailable because the face or smile detector could not be loaded.", force=True)
                return

            use_running_camera = self.running
            if not use_running_camera:
                temporary_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not temporary_cap.isOpened():
                    self.speak("Selfie failed. The camera is unavailable.", force=True)
                    return

            self.speak("Selfie mode. Face the camera. Keep your head still and smile naturally. I will count three, two, one before taking the picture.", force=True)
            started = time.monotonic()
            last_reminder = started
            stable_count = 0
            last_box = None
            countdown_phase = None
            countdown_stage_at = 0.0
            captured_frame = None
            selfie_lighting_state = None

            while time.monotonic() - started < 35.0:
                if use_running_camera:
                    with self._frame_lock:
                        frame = self._latest_frame.copy() if self._latest_frame is not None else None
                    time.sleep(0.12)
                else:
                    ok, frame = temporary_cap.read()
                    if not ok:
                        frame = None
                    time.sleep(0.08)

                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._detect_faces_with_fallback(gray, face_detector, (fallback_face_detector,))
                if len(faces) == 0:
                    stable_count = 0
                    last_box = None
                    countdown_phase = None
                    if time.monotonic() - last_reminder >= 8.0:
                        self.speak("I cannot detect your face. Turn your face toward the camera and place a light in front of you.")
                        last_reminder = time.monotonic()
                    continue

                x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
                box = (x, y, w, h)
                if last_box is not None:
                    dx = abs(x - last_box[0])
                    dy = abs(y - last_box[1])
                    dw = abs(w - last_box[2])
                    dh = abs(h - last_box[3])
                    if dx <= max(14, w * 0.06) and dy <= max(14, h * 0.06) and dw <= max(12, w * 0.07) and dh <= max(12, h * 0.07):
                        stable_count += 1
                    else:
                        stable_count = 0
                        countdown_phase = None
                last_box = box

                roi = gray[y:y+h, x:x+w]
                face_brightness = float(roi.mean()) if roi.size else 0.0
                if face_brightness < self.DARK_THRESHOLD:
                    if selfie_lighting_state != "dark" and time.monotonic() - last_reminder >= 1.0:
                        self.speak("Lighting is too low on your face. Add light in front of you.")
                        last_reminder = time.monotonic()
                    selfie_lighting_state = "dark"
                elif face_brightness >= self.LIGHT_OK_THRESHOLD and selfie_lighting_state != "good":
                    self.speak("Lighting is good.")
                    selfie_lighting_state = "good"
                # Smile cascades generally detect the mouth in the lower half of the face.
                mouth_region = roi[int(h * 0.30):, :]
                smiles = smile_detector.detectMultiScale(
                    cv2.equalizeHist(mouth_region), scaleFactor=1.1, minNeighbors=12,
                    minSize=(max(15, int(w * 0.15)), max(8, int(h * 0.06)))
                )
                smiling = len(smiles) > 0

                if stable_count < 5:
                    countdown_phase = None
                    if time.monotonic() - last_reminder >= 8.0:
                        self.speak("Hold still with your face toward the camera.")
                        last_reminder = time.monotonic()
                    continue
                if not smiling:
                    countdown_phase = None
                    if time.monotonic() - last_reminder >= 8.0:
                        self.speak("I can see your face. Give a natural smile for the countdown.")
                        last_reminder = time.monotonic()
                    continue

                now = time.monotonic()
                if countdown_phase is None:
                    self.speak("Smile detected. Hold still. Countdown begins in two seconds.", force=True)
                    countdown_phase = "intro"
                    countdown_stage_at = now
                    continue
                if countdown_phase == "intro" and now - countdown_stage_at >= 2.5:
                    self.speak("Three", force=True)
                    countdown_phase = "three"
                    countdown_stage_at = now
                    continue
                if countdown_phase == "three" and now - countdown_stage_at >= 1.2:
                    self.speak("Two", force=True)
                    countdown_phase = "two"
                    countdown_stage_at = now
                    continue
                if countdown_phase == "two" and now - countdown_stage_at >= 1.2:
                    self.speak("One", force=True)
                    countdown_phase = "one"
                    countdown_stage_at = now
                    continue
                if countdown_phase == "one" and now - countdown_stage_at >= 1.3:
                    # Capture only if face stability and smile were confirmed this frame.
                    captured_frame = frame.copy()
                    break

            if captured_frame is None:
                self.speak("Selfie not captured. I could not confirm a steady face and detectable smile within 35 seconds. Try better lighting and a natural smile.", force=True)
                return

            pictures = os.path.join(os.path.expanduser("~"), "Pictures", "FaceGuide Selfies")
            os.makedirs(pictures, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = os.path.join(pictures, "FaceGuide_Selfie_" + stamp + ".jpg")
            if not cv2.imwrite(path, captured_frame):
                self.speak("Selfie could not be saved. Please check the Pictures folder permissions.", force=True)
                return
            self.speak("Selfie captured and saved in Pictures, FaceGuide Selfies.", force=True)
        except Exception:
            self.speak("Selfie capture failed. Check that the camera is available and try again.", force=True)
        finally:
            self.selfie_active = False
            if temporary_cap is not None:
                try:
                    temporary_cap.release()
                except Exception:
                    pass

    def script_status(self, gesture):
        camera = "running" if self.running else "stopped"
        face = "detected" if self._camera_face_detected else "not currently detected"
        lighting = self._camera_lighting_state
        distance = self._camera_distance_state
        selfie = "active" if self.selfie_active else "inactive"
        try:
            ui.message(
                "FaceGuide Modern status. Camera " + camera + ". Face " + face +
                ". Lighting " + lighting + ". Distance " + distance +
                ". Selfie mode " + selfie + ". OpenCV " + cv2.__version__ + " is loaded."
            )
        except Exception:
            ui.message("FaceGuide Modern status unavailable.")

    __gestures = {
        "kb:NVDA+alt+g": "toggle",
        "kb:NVDA+alt+shift+g": "status",
        "kb:NVDA+alt+shift+c": "capture_selfie",
        "kb:NVDA+alt+shift+p": "capture_selfie",
    }
