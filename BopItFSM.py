import random
import asyncio
import cv2 as cv
import mediapipe as mp
import math
import time
import numpy as np
from flask import Flask, Response, request
from threading import Thread
from queue import Queue
from bleak import BleakClient, BleakScanner
import socket
#
# ── Config ──────────────────────────────────────────────
DEVICE_NAME = "Bluefruit"
UART_RX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
#UART_RX_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

PHONE_USB_IP = "192.168.159.106"

def send_to_phone(message: str):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect((PHONE_USB_IP, 9000))
        sock.sendall((message + "\n").encode())
        sock.close()
        print(f"[TCP] Sent '{message}' to phone")
    except Exception as e:
        print(f"[TCP] Failed to send '{message}': {e}")

# ── Flask ────────────────────────────────────────────────
flask_app = Flask(__name__)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
frame_queue = Queue(maxsize=2)

@flask_app.route('/frame', methods=['POST'])
def receive_frame():
    jpeg_data = request.get_data()
    if not jpeg_data:
        return "No data", 400
    np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
    frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    if frame is None:
        return "Bad image", 400
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except:
            pass
    frame_queue.put(frame)
    return "OK", 200

# ── MediaPipe ────────────────────────────────────────────
mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils

# ── Game State ───────────────────────────────────────────
score = 0
level = 1
ppl = 20
game_started = False

def timeLimit(level: int) -> float:
    return max(1.5, 6.0 - (level - 1) * 0.5)

# ── FSM ─────────────────────────────────────────────────
class BopItFSM:
    def __init__(self):
        self.current_state = "idle"
        self.current_prompt = None
        self._timeout_task = None

    def prompt(self):
        global level
        self.current_prompt = random.choice(["ATTACK", "BLOCK", "FORCE"])
        self.current_state = "waiting"
        time_lim = timeLimit(level)
        print(f"\n>>> {self.current_prompt}! (Level {level} - {time_lim:.1f}s)")
        send_to_phone(self.current_prompt)
        if self._timeout_task:
            self._timeout_task.cancel()
        self._timeout_task = asyncio.create_task(self._timeout_after(time_lim))

    async def _timeout_after(self, seconds: float):
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        if self.current_state == "waiting":
            print("Too Slow!")
            self.game_over("timeout")

    def evaluate(self, detected: str):
        global score, level
        if self.current_state != "waiting":
            return
        if self._timeout_task:
            self._timeout_task.cancel()
            self._timeout_task = None
        if detected == self.current_prompt:
            print("✓ Correct!")
            send_to_phone("CORRECT!")
            score += 1
            if score % ppl == 0:
                level += 1
                print("Level Up!")
                send_to_phone("LEVEL UP!")
            print(f"Score: {score} | Level: {level}")
            self.current_state = "idle"
            asyncio.create_task(self._next_prompt())
        else:
            print(f"✗ Wrong! Expected {self.current_prompt}, got {detected}")
            self.game_over("wrong")

    def game_over(self, reason: str):
        global game_started
        self.current_state = "game_over"
        self.current_prompt = None
        if self._timeout_task:
            self._timeout_task.cancel()
            self._timeout_task = None
        print("\n========== GAME OVER ==========")
        print(f"Reason: {reason}")
        print(f"Final Score: {score} | Level Reached: {level}")
        print("===============================")
        send_to_phone(f"Score: {score}")
        game_started = False
        # Schedule the "PRESS START" message without blocking the event loop
        asyncio.create_task(self._announce_game_over())

    async def _announce_game_over(self):
        await asyncio.sleep(3)
        send_to_phone("GAME_OVER! PRESS START")

    async def _next_prompt(self):
        await asyncio.sleep(1.0)
        if self.current_state != "game_over":
            self.prompt()

    def reset(self):
        global score, level
        if self._timeout_task:
            self._timeout_task.cancel()
            self._timeout_task = None
        score = 0
        level = 1
        self.current_state = "idle"
        self.current_prompt = None
        print("Game reset.")

fsm = BopItFSM()

# ── Helpers that run ON the asyncio loop thread ──────────
def _handle_start():
    """Called via call_soon_threadsafe so we're on the loop thread."""
    global game_started
    fsm.reset()
    game_started = True
    fsm.prompt()

# ── Force Detection + Camera Loop ────────────────────────
latest_frame = b''
last_force = False

def camera_loop():
    global latest_frame, last_force

    print("[CAMERA] MediaPipe loop started, waiting for frames from Android...")

    with mphands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            frame = frame_queue.get()
            frame_height, frame_width = frame.shape[:2]

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            process_frames = hands.process(rgb_frame)

            force_detected = False

            # Flip this to "Right" if your camera feed isn't mirrored
            # (MediaPipe labels assume a mirrored selfie-style image)
            TARGET_LABEL = "Right"

            if process_frames.multi_hand_landmarks and process_frames.multi_handedness:
                # Collect all hands matching our target handedness
                candidates = []
                for lm, handedness in zip(
                    process_frames.multi_hand_landmarks,
                    process_frames.multi_handedness
                ):
                    if handedness.classification[0].label == TARGET_LABEL:
                        candidates.append(lm)

                # Pick the one closest to horizontal center (the person in the middle)
                if candidates:
                    lm = min(candidates, key=lambda h: abs(h.landmark[0].x - 0.5))

                    mpdrawing.draw_landmarks(frame, lm, mphands.HAND_CONNECTIONS)

                    thumb_tip  = lm.landmark[4]
                    index_tip  = lm.landmark[8]
                    ring_tip   = lm.landmark[16]
                    pinky_tip  = lm.landmark[20]
                    wrist      = lm.landmark[0]
                    mid_mcp    = lm.landmark[9]

                    ref_dist = math.sqrt(
                        (wrist.x - mid_mcp.x)**2 + (wrist.y - mid_mcp.y)**2
                    )
                    if ref_dist >= 0.001:
                        norm_distance_1 = math.sqrt(
                            (thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2
                        ) / ref_dist

                        norm_distance_2 = math.sqrt(
                            (ring_tip.x - pinky_tip.x)**2 + (ring_tip.y - pinky_tip.y)**2
                        ) / ref_dist

                        if norm_distance_1 > 0.5 and norm_distance_2 > 0.5:
                            force_detected = True
                            cv.putText(frame, "Using the force",
                                       (frame_width // 2 - 150, 50),
                                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if force_detected and not last_force:
                main_loop.call_soon_threadsafe(fsm.evaluate, "FORCE")
            last_force = force_detected

            cv.putText(frame, f"Score: {score}  Level: {level}",
                       (20, frame_height - 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if fsm.current_state == "game_over":
                cv.putText(frame, "GAME OVER",
                           (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            elif fsm.current_prompt:
                cv.putText(frame, f"DO: {fsm.current_prompt}",
                           (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

            _, buffer = cv.imencode('.jpg', frame)
            latest_frame = buffer.tobytes()

def generate():
    while True:
        if latest_frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   latest_frame + b'\r\n')

@flask_app.route('/output')
def output():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ── BLE Handler ──────────────────────────────────────────
ble_buffer = ""

def on_data(sender, data: bytearray):
    global ble_buffer
    ble_buffer += data.decode("utf-8", errors="ignore")
    while True:
        ble_buffer = ble_buffer.strip()
        if not ble_buffer:
            break
        if "START" in ble_buffer:
            ble_buffer = ble_buffer.replace("START", "", 1).strip()
            print("[BLE] Received: START — Beginning game!")
            main_loop.call_soon_threadsafe(_handle_start)
        elif "ATTACK" in ble_buffer:
            ble_buffer = ble_buffer.replace("ATTACK", "", 1).strip()
            print("[BLE] Received: ATTACK")
            main_loop.call_soon_threadsafe(fsm.evaluate, "ATTACK")
        elif "BLOCK" in ble_buffer:
            ble_buffer = ble_buffer.replace("BLOCK", "", 1).strip()
            print("[BLE] Received: BLOCK")
            main_loop.call_soon_threadsafe(fsm.evaluate, "BLOCK")
        else:
            break

# ── Main ─────────────────────────────────────────────────
async def main():
    global main_loop
    main_loop = asyncio.get_running_loop()
    cam_thread = Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    flask_thread = Thread(target=lambda: flask_app.run(
        host='0.0.0.0', port=5000, use_reloader=False), daemon=True)
    flask_thread.start()

    print(f"Flask running on port 5000")
    print(f"  - Android posts frames to http://[PI-IP]:5000/frame")
    print(f"  - Debug view at http://[PI-IP+]:5000/output")

    print("Scanning for Bluefruit...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME)
    if not device:
        print(f"Could not find '{DEVICE_NAME}'. Running with camera only...")
        while True:
            await asyncio.sleep(1)
        return

    print(f"Found {device.name}, connecting...")
    async with BleakClient(device) as client:
        print("Connected! Starting game...\n")
        send_to_phone("Waiting for Start!")
        await client.start_notify(UART_RX_UUID, on_data)
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nGame over!")
            print(f"Final score: {score}")
            await client.stop_notify(UART_RX_UUID)

asyncio.run(main())
