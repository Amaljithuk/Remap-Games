import sys
import cv2
import numpy as np
import random
import math
import time
import pygame
import os
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget, QFrame, QSizePolicy, QGraphicsOpacityEffect)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QBrush, QColor, QLinearGradient, QGradient

# Initialize Audio
pygame.mixer.init()

# =====================================================
# EXE PATH HELPER (CRITICAL FOR PYINSTALLER)
# =====================================================
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# =====================================================
# ASSET MANAGER
# =====================================================
class Assets:
    def __init__(self):
        self.images = {}
        self.sounds = {}
        self.load_images()
        self.load_sounds()

    def load_images(self):
        def load_img(path, size=None):
            # WRAP PATH FOR EXE
            full_path = resource_path(path)
            
            if os.path.exists(full_path):
                img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    if size:
                        img = cv2.resize(img, size)
                    return img
            return None

        self.images['head'] = load_img("assets/snake_head.png", (80, 80)) 
        self.images['food'] = load_img("assets/apple.png", (65, 65))
        self.images['crosshair'] = load_img("assets/crosshair.png", (40, 40))
        # HUD REMOVED as requested
        self.images['heart_full'] = load_img("assets/heart_full.png", (40, 40))
        self.images['heart_empty'] = load_img("assets/heart_empty.png", (40, 40))

    def load_sounds(self):
        def load_snd(filename):
            # WRAP PATH FOR EXE
            full_path = resource_path(filename)
            
            if os.path.exists(full_path):
                return pygame.mixer.Sound(full_path)
            return None
        
        self.sounds['eat'] = load_snd("eat.mp3")
        self.sounds['hit'] = load_snd("hit.mp3")
        self.sounds['game_over'] = load_snd("game_over.mp3")
        
        # Fallback for click sound
        self.sounds['click'] = load_snd("click.mp3")
        if self.sounds['click'] is None:
            self.sounds['click'] = load_snd("hit.mp3")

        # BG Music Check
        bg_music = resource_path("hits.mp3")
        if os.path.exists(bg_music):
            pygame.mixer.music.load(bg_music)
            pygame.mixer.music.set_volume(0.2)

assets = Assets()

# =====================================================
# CUSTOM UI WIDGETS
# =====================================================
class LoadingOverlay(QWidget):
    """A translucent overlay with a spinning text to block input during loads."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False) # Block clicks
        self.setStyleSheet("background-color: rgba(0, 0, 0, 200);")
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        self.lbl_text = QLabel("LOADING...")
        self.lbl_text.setStyleSheet("""
            color: #FFD700; 
            font-size: 32px; 
            font-weight: bold; 
            font-family: 'Segoe UI';
            background: transparent;
        """)
        layout.addWidget(self.lbl_text)
        self.setLayout(layout)
        self.hide()

def get_camera_index():
    """Checks for external camera (Index 1). If found, returns 1. If not, returns 0 (Webcam)."""
    cap = cv2.VideoCapture(1)
    if cap is not None and cap.isOpened():
        ret, _ = cap.read()
        cap.release()
        if ret:
            print("✅ External Camera Found (Index 1)")
            return 1
    print("⚠️ Using Default Webcam (Index 0)")
    return 0

# =====================================================
# VISUAL HELPER FUNCTIONS
# =====================================================
def overlay_image(bg, overlay, x, y):
    if overlay is None: return bg
    h, w = overlay.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
    if x2 <= x1 or y2 <= y1: return bg
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    bg_crop = bg[y1:y2, x1:x2]
    overlay_crop = overlay[oy1:oy2, ox1:ox2]
    if overlay.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(3):
            bg_crop[:, :, c] = (alpha * overlay_crop[:, :, c] + alpha_inv * bg_crop[:, :, c])
    else:
        bg_crop[:] = overlay_crop
    bg[y1:y2, x1:x2] = bg_crop
    return bg

def rotate_image(image, angle):
    if image is None: return None
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def draw_shadow_text(img, text, pos, scale, color, thickness):
    x, y = pos
    cv2.putText(img, text, (x+3, y+3), cv2.FONT_HERSHEY_TRIPLEX, scale, (0, 0, 0), thickness+2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_TRIPLEX, scale, color, thickness)

def spawn_food(diff, trail, w, h, bounds=None):
    if bounds:
        min_x, max_x, min_y, max_y = bounds
        # Add margin
        min_x += 40; max_x -= 40; min_y += 40; max_y -= 40
    else:
        min_x, max_x = 60, w - 60
        min_y, max_y = 140, h - 60
    
    cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
    width_span = (max_x - min_x) // 2
    height_span = (max_y - min_y) // 2

    # --- DIFFICULTY LOGIC FOR SPAWNING ---
    # Easy: Restrict spawning to near center
    if diff == "easy":
        spawn_range_x = int(width_span * 0.3) # 30% of width
        spawn_range_y = int(height_span * 0.3)
    # Medium: Restrict spawning to moderate area
    elif diff == "medium":
        spawn_range_x = int(width_span * 0.6) # 60% of width
        spawn_range_y = int(height_span * 0.6)
    # Hard: Full area
    else:
        spawn_range_x = width_span
        spawn_range_y = height_span

    for _ in range(50):
        # Generate coordinates based on difficulty range around center
        dx = random.randint(-spawn_range_x, spawn_range_x)
        dy = random.randint(-spawn_range_y, spawn_range_y)
        x = cx + dx
        y = cy + dy
        
        # Ensure within strict bounds just in case
        x = max(min_x, min(max_x, x))
        y = max(min_y, min(max_y, y))

        collision = False
        for sx, sy in trail:
            if math.hypot(x - sx, y - sy) < 60:
                collision = True
                break
        if not collision: return (x, y)
    return (cx, cy)

# =====================================================
# WORKERS
# =====================================================
class CalibrationWorker(QThread):
    frame_update = pyqtSignal(QImage)
    calibration_complete = pyqtSignal(tuple)

    def __init__(self, mode="standing"):
        super().__init__()
        self.mode = mode
        self.running = True
        self.hold_time = 3.0
        self.hold_timer = 0
        self.calib_min_x, self.calib_max_x = 1280, 0
        self.calib_min_y, self.calib_max_y = 720, 0

    def run(self):
        cam_idx = get_camera_index()
        cap = cv2.VideoCapture(cam_idx)
        cap.set(3, 1280)
        cap.set(4, 720)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5)
        mp_draw = mp.solutions.drawing_utils
        last_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret: break
            dt = time.time() - last_time
            last_time = time.time()
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            within = False
            
            if self.mode == "sitting":
                results_pose = pose.process(img_rgb)
                if results_pose.pose_landmarks:
                    mp_draw.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    lw = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                    rw = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                    lx, ly = int(lw.x * w), int(lw.y * h)
                    rx, ry = int(rw.x * w), int(rw.y * h)
                    self.calib_min_x = min(self.calib_min_x, lx, rx)
                    self.calib_max_x = max(self.calib_max_x, lx, rx)
                    self.calib_min_y = min(self.calib_min_y, ly, ry)
                    self.calib_max_y = max(self.calib_max_y, ly, ry)
                    if abs(lx - rx) > w * 0.4: within = True
            else:
                results_hands = hands.process(img_rgb)
                if results_hands.multi_hand_landmarks:
                    for lm in results_hands.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                        within = True
                self.calib_min_x, self.calib_max_x = 50, w - 50
                self.calib_min_y, self.calib_max_y = 100, h - 50

            if within: self.hold_timer += dt
            else: self.hold_timer = max(0, self.hold_timer - dt)
            
            bx, by = w//2 - 200, h - 100
            cv2.rectangle(frame, (bx, by), (bx + 400, by + 30), (50, 50, 50), -1)
            prog = int(400 * (self.hold_timer / self.hold_time))
            color = (0, 255, 0) if within else (0, 165, 255)
            if prog > 0: cv2.rectangle(frame, (bx, by), (bx + prog, by + 30), color, -1)
            msg = "STRETCH ARMS!" if self.mode == "sitting" else "STAND CLEAR!"
            draw_shadow_text(frame, msg, (w//2 - 150, h - 140), 1.0, color, 2)
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_update.emit(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888))
            
            if self.hold_timer >= self.hold_time:
                final_bounds = (max(0, self.calib_min_x - 20), min(w, self.calib_max_x + 20),
                                max(0, self.calib_min_y - 20), min(h, self.calib_max_y + 20))
                self.calibration_complete.emit(final_bounds)
                break
        cap.release()

class GameWorker(QThread):
    frame_update = pyqtSignal(QImage)
    game_over = pyqtSignal()
    score = 0

    def __init__(self, difficulty, duration, bounds):
        super().__init__()
        self.difficulty = difficulty
        self.duration = duration
        self.bounds = bounds
        self.running = True
        self.paused = False

    def run(self):
        cam_idx = get_camera_index()
        cap = cv2.VideoCapture(cam_idx)
        cap.set(3, 1280)
        cap.set(4, 720)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        snake_trail = []
        smooth_points = []
        max_trail = 5
        snake_pos = (640, 360)
        self.score = 0
        lives = 3
        invincible, inv_time = False, 0
        angle = 0
        start_time = time.time()
        food = None
        bound_min_x, bound_max_x, bound_min_y, bound_max_y = self.bounds

        # --- DIFFICULTY LOGIC (SPEED) ---
        if self.difficulty == "easy":
            speed_factor = 0.10
        elif self.difficulty == "medium":
            speed_factor = 0.20
        else: # hard
            speed_factor = 0.35

        while self.running:
            if self.paused: time.sleep(0.1); continue
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if food is None: food = spawn_food(self.difficulty, snake_trail, w, h, self.bounds)
            elapsed = time.time() - start_time
            remaining = max(0, self.duration - int(elapsed))

            if invincible and time.time() - inv_time > 2.0: invincible = False
            if remaining <= 0 or lives <= 0:
                self.game_over.emit()
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            finger_detected = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm = hand_landmarks.landmark[8]
                    wx, wy = int(lm.x * w), int(lm.y * h)
                    finger_detected = True
                    smooth_points.append((wx, wy))
                    if len(smooth_points) > 5: smooth_points.pop(0)
                    frame = overlay_image(frame, assets.images['crosshair'], wx-20, wy-20)

            if finger_detected and len(smooth_points) > 0:
                avg_x = int(sum(p[0] for p in smooth_points) / len(smooth_points))
                avg_y = int(sum(p[1] for p in smooth_points) / len(smooth_points))
                dx = avg_x - snake_pos[0]
                dy = avg_y - snake_pos[1]
                
                angle = math.degrees(math.atan2(dy, dx)) + 90
                # Move snake based on difficulty speed
                snake_pos = (snake_pos[0] + dx * speed_factor, snake_pos[1] + dy * speed_factor)

            x, y = int(snake_pos[0]), int(snake_pos[1])

            if not invincible:
                if (x < bound_min_x or x > bound_max_x or y < bound_min_y or y > bound_max_y):
                    lives -= 1
                    if assets.sounds['hit']: assets.sounds['hit'].play()
                    snake_pos = ((bound_min_x + bound_max_x)//2, (bound_min_y + bound_max_y)//2)
                    snake_trail.clear()
                    smooth_points.clear()
                    invincible, inv_time = True, time.time()
                    continue

            x = max(bound_min_x, min(bound_max_x, x))
            y = max(bound_min_y, min(bound_max_y, y))
            snake_pos = (x, y)
            snake_trail.append(snake_pos)
            if len(snake_trail) > max_trail: snake_trail.pop(0)

            if math.hypot(x - food[0], y - food[1]) < 55:
                if assets.sounds['eat']: assets.sounds['eat'].play()
                self.score += 1
                max_trail += 3
                food = spawn_food(self.difficulty, snake_trail, w, h, self.bounds)

            # --- NEON BORDER ---
            # Outer glow
            cv2.rectangle(frame, (bound_min_x-5, bound_min_y-5), (bound_max_x+5, bound_max_y+5), (0, 165, 255), 2)
            # Inner bright line
            cv2.rectangle(frame, (bound_min_x, bound_min_y), (bound_max_x, bound_max_y), (0, 255, 255), 4)
            
            if assets.images['food'] is not None:
                frame = overlay_image(frame, assets.images['food'], food[0]-32, food[1]-32)
            else:
                cv2.circle(frame, food, 25, (0, 0, 255), -1)

            if len(snake_trail) > 1:
                pts = np.array(snake_trail, np.int32).reshape((-1, 1, 2))
                overlay = frame.copy()
                cv2.polylines(overlay, [pts], False, (0, 165, 255), 40)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                cv2.polylines(frame, [pts], False, (0, 215, 255), 15)

            if assets.images['head'] is not None:
                rotated_head = rotate_image(assets.images['head'], -angle)
                frame = overlay_image(frame, rotated_head, x-40, y-40)
            else:
                cv2.circle(frame, (x, y), 35, (0, 215, 255), -1)

            # Draw Stats Text (No HUD Frame)
            draw_shadow_text(frame, f"SCORE: {self.score}", (bound_min_x, bound_min_y - 20), 1.2, (255, 255, 255), 2)
            
            time_color = (0, 255, 255) if remaining > 10 else (0, 0, 255)
            draw_shadow_text(frame, f"TIME: {remaining}", (bound_max_x - 200, bound_min_y - 20), 1.2, time_color, 2)

            start_x = (bound_min_x + bound_max_x) // 2 - 75
            for i in range(3):
                h_img = assets.images['heart_full'] if i < lives else assets.images['heart_empty']
                frame = overlay_image(frame, h_img, start_x + (i * 50), bound_min_y - 50)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_update.emit(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888))

        cap.release()

# =====================================================
# UI MAIN WINDOW
# =====================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake AR - Jungle Adventure")
        self.setGeometry(100, 100, 1280, 720)
        
        # Load Background with WRAPPED PATH
        bg_path = resource_path("assets/bg.jpg")
        self.bg_pixmap = None
        if os.path.exists(bg_path):
            self.bg_pixmap = QImage(bg_path)
            self.update_background()
        else:
            self.setStyleSheet("QMainWindow { background-color: #1a1a2e; }")

        # Resolve Wood Button Path for CSS
        btn_wood_path = resource_path("assets/btn_wood.png").replace("\\", "/")
        
        # --- REFINED STYLE SHEET ---
        self.setStyleSheet(self.styleSheet() + f"""
            /* STATIC BUTTONS (RESTORED BG IMAGE) */
            QPushButton {{
                /* Real Image (Your Custom Asset) */
                background-image: url({btn_wood_path});
                background-repeat: no-repeat;
                background-position: center;
                background-size: cover; 
                color: #FFE4B5; 
                border: 3px solid #DAA520; 
                border-radius: 10px;
                font-family: 'Verdana';
                font-size: 22px;
                font-weight: bold;
                padding: 15px;
                margin: 5px;
            }}
            
            QPushButton:hover {{
                background: #CD853F;
                border: 3px solid #FFD700;
                color: white;
            }}
            
            /* GAME OVER CARD */
            QFrame#GameOverCard {{
                background-color: rgba(0, 0, 0, 180);
                border: 4px solid #FFD700;
                border-radius: 20px;
            }}
            
            .score_title {{
                color: #FFD700;
                font-size: 24px;
                font-weight: bold;
                letter-spacing: 2px;
            }}
            
            .score_val {{
                color: #FFFFFF;
                font-size: 90px;
                font-weight: 900;
            }}
        """)

        self.worker = None
        self.difficulty = "medium"
        self.duration = 60
        self.calibration_mode = "standing"
        self.calibrated_bounds = (50, 1230, 100, 670)
        
        # Stack for Screens
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        # Loading Overlay
        self.loader = LoadingOverlay(self)
        self.loader.resize(1280, 720)

        # Init Screens
        self.init_start_screen()
        self.init_mode_screen()
        self.init_calibration_screen()
        self.init_difficulty_screen()
        self.init_duration_screen()
        self.init_instruction_screen()
        self.init_game_screen()
        self.init_gameover_screen()

    def resizeEvent(self, event):
        self.update_background()
        self.loader.resize(self.size())
        super().resizeEvent(event)

    def update_background(self):
        if self.bg_pixmap is not None:
            sImage = self.bg_pixmap.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            palette = QPalette()
            palette.setBrush(QPalette.Window, QBrush(sImage))
            self.setPalette(palette)

    # --- HELPER FOR STATIC TITLES ---
    def get_title_widget(self, image_name, fallback_text):
        lbl = QLabel("") 
        # WRAP PATH
        path = resource_path(image_name) 
        
        if os.path.exists(path):
            pix = QPixmap(path)
            scaled = pix.scaled(500, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl.setPixmap(scaled)
        else:
            print(f"⚠️ Missing Title Image: {path}") 
            lbl.setText(fallback_text)
            lbl.setStyleSheet("font-family: 'Verdana'; font-size: 60px; font-weight: 900; color: #FFD700;")
        lbl.setAlignment(Qt.AlignCenter)
        return lbl

    # --- HELPER TO PLAY SOUND THEN SWITCH ---
    def play_sound_and_switch(self, index):
        if assets.sounds['click']:
            assets.sounds['click'].play()
        self.central_widget.setCurrentIndex(index)

    # --- SCREENS ---
    
    def init_start_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # Title Image
        title = self.get_title_widget("assets/download.png", "SNAKE AR")

        btn_start = QPushButton("▶ PLAY GAME")
        btn_start.setFixedSize(300, 70)
        # Fix: Call helper to play sound
        btn_start.clicked.connect(lambda: self.play_sound_and_switch(1))

        btn_exit = QPushButton("❌ EXIT")
        btn_exit.setFixedSize(300, 70)
        btn_exit.clicked.connect(self.close)

        layout.addWidget(title)
        layout.addWidget(btn_start)
        layout.addWidget(btn_exit)
        page.setLayout(layout)
        self.central_widget.addWidget(page)

    def init_mode_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        title = self.get_title_widget("assets/MODE.png", "SELECT MODE")
        
        btn_stand = QPushButton("🧍 STANDING")
        btn_stand.setFixedSize(350, 70)
        btn_stand.clicked.connect(lambda: self.start_calibration("standing"))
        
        btn_sit = QPushButton("🪑 SITTING")
        btn_sit.setFixedSize(350, 70)
        btn_sit.clicked.connect(lambda: self.start_calibration("sitting"))
        
        layout.addWidget(title)
        layout.addWidget(btn_stand)
        layout.addWidget(btn_sit)
        page.setLayout(layout)
        self.central_widget.addWidget(page)

    def init_calibration_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        self.calib_label = QLabel()
        self.calib_label.setAlignment(Qt.AlignCenter)
        self.calib_label.setScaledContents(True)
        self.calib_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.calib_label)
        page.setLayout(layout)
        self.central_widget.addWidget(page)

    def init_difficulty_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        title = self.get_title_widget("assets/difficulty.png", "DIFFICULTY")
        
        layout.addWidget(title)
        for d in ["EASY", "MEDIUM", "HARD"]:
            btn = QPushButton(d)
            btn.setFixedSize(350, 70)
            btn.clicked.connect(lambda c, diff=d.lower(): self.select_difficulty(diff))
            layout.addWidget(btn)
        page.setLayout(layout)
        self.central_widget.addWidget(page)

    def init_duration_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        title = self.get_title_widget("assets/duration.png", "DURATION")
        
        t_layout = QHBoxLayout()
        self.time_label = QLabel("60 SEC")
        self.time_label.setStyleSheet("font-size: 40px; color: #FFE4B5; font-weight: bold;")
        
        b_minus = QPushButton("➖")
        b_minus.setFixedSize(80, 60)
        b_minus.clicked.connect(lambda: self.change_time(-10))
        
        b_plus = QPushButton("➕")
        b_plus.setFixedSize(80, 60)
        b_plus.clicked.connect(lambda: self.change_time(10))
        
        t_layout.setAlignment(Qt.AlignCenter)
        t_layout.addWidget(b_minus); t_layout.addWidget(self.time_label); t_layout.addWidget(b_plus)
        
        btn = QPushButton("🚀 NEXT")
        btn.setFixedSize(350, 70)
        # Fix: Call helper to play sound
        btn.clicked.connect(lambda: self.play_sound_and_switch(5))
        
        # --- RESTORED ALIGNMENT LOGIC ---
        launch_layout = QHBoxLayout()
        # 1. Add spring on the LEFT
        launch_layout.addStretch(1) 
        # 2. Add invisible space BEFORE the button
        launch_layout.addSpacing(10) 
        # 3. Add the Button
        launch_layout.addWidget(btn)
        # 4. Add spring on the RIGHT
        launch_layout.addStretch(1)
        # --------------------------------
        
        layout.addWidget(title)
        layout.addLayout(t_layout)
        layout.addLayout(launch_layout)
        page.setLayout(layout)
        self.central_widget.addWidget(page)

    def init_instruction_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(30)

        title = self.get_title_widget("assets/instructions.png", "HOW TO PLAY")
        
        card = QFrame()
        card.setStyleSheet("background-color: rgba(0, 0, 0, 150); border: 2px solid #FFD700; border-radius: 20px;")
        card.setFixedSize(800, 350)
        
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignCenter)
        
        steps = [
            "👋 1. CALIBRATE your hand position.",
            "🍎 2. Move your INDEX FINGER to guide the snake.",
            "🐍 3. Eat APPLES to grow longer.",
            "❌ 4. Avoid hitting WALLS or your own TAIL!"
        ]
        
        for step in steps:
            lbl = QLabel(step)
            lbl.setStyleSheet("color: white; font-size: 28px; font-weight: bold; font-family: 'Segoe UI'; padding: 5px;")
            lbl.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(lbl)

        btn_go = QPushButton("GO!")
        btn_go.setFixedSize(200, 70)
        btn_go.clicked.connect(self.trigger_game_load)

        layout.addWidget(title)
        layout.addWidget(card)
        layout.addWidget(btn_go, 0, Qt.AlignCenter)
        
        page.setLayout(layout)
        self.central_widget.addWidget(page)

    def init_game_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.game_label = QLabel()
        self.game_label.setScaledContents(True)
        self.game_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.game_label)
        page.setLayout(layout)
        self.central_widget.addWidget(page)

    def init_gameover_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        card = QFrame()
        card.setObjectName("GameOverCard")
        card.setFixedSize(500, 400)
        
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignCenter)
        card_layout.setSpacing(20)
        
        lbl_over = QLabel("GAME OVER")
        lbl_over.setStyleSheet("color: white; font-size: 32px; font-weight: bold;")
        
        lbl_text = QLabel("FINAL SCORE")
        lbl_text.setProperty("class", "score_title")
        
        self.final_score = QLabel("0")
        self.final_score.setProperty("class", "score_val")
        
        btn_restart = QPushButton("🔄 PLAY AGAIN")
        btn_restart.setFixedSize(250, 60)
        btn_restart.clicked.connect(self.restart)
        
        btn_quit = QPushButton("❌ QUIT")
        btn_quit.setFixedSize(250, 60)
        btn_quit.clicked.connect(self.close)

        card_layout.addWidget(lbl_over, 0, Qt.AlignCenter)
        card_layout.addWidget(lbl_text, 0, Qt.AlignCenter)
        card_layout.addWidget(self.final_score, 0, Qt.AlignCenter)
        card_layout.addWidget(btn_restart, 0, Qt.AlignCenter)
        card_layout.addWidget(btn_quit, 0, Qt.AlignCenter)
        
        layout.addWidget(card)
        page.setLayout(layout)
        self.central_widget.addWidget(page)

    # --- LOGIC ---
    
    def show_loader(self, callback):
        self.loader.raise_()
        self.loader.show()
        QTimer.singleShot(1500, lambda: self._hide_loader_and_call(callback))
        
    def _hide_loader_and_call(self, callback):
        self.loader.hide()
        callback()

    def start_calibration(self, mode):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.calibration_mode = mode
        self.show_loader(self._real_start_calibration)

    def _real_start_calibration(self):
        self.central_widget.setCurrentIndex(2) 
        self.worker = CalibrationWorker(self.calibration_mode)
        self.worker.frame_update.connect(lambda img: self.calib_label.setPixmap(QPixmap.fromImage(img)))
        self.worker.calibration_complete.connect(self.on_calibration_done)
        self.worker.start()

    def on_calibration_done(self, bounds):
        self.calibrated_bounds = bounds
        self.central_widget.setCurrentIndex(3)

    def select_difficulty(self, diff):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.difficulty = diff
        self.central_widget.setCurrentIndex(4)

    def change_time(self, delta):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.duration = max(10, min(300, self.duration + delta))
        self.time_label.setText(f"{self.duration} SEC")

    def trigger_game_load(self):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.show_loader(self.start_game)

    def start_game(self):
        if self.worker:
            self.worker.running = False
            self.worker.wait()
        if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
        bg_music = resource_path("hits.mp3")
        if os.path.exists(bg_music): pygame.mixer.music.play(-1)

        self.central_widget.setCurrentIndex(6) # Game Screen is now index 6
        self.worker = GameWorker(self.difficulty, self.duration, self.calibrated_bounds)
        self.worker.frame_update.connect(lambda img: self.game_label.setPixmap(QPixmap.fromImage(img)))
        self.worker.game_over.connect(self.game_over)
        self.worker.start()

    def game_over(self):
        if assets.sounds['game_over']: assets.sounds['game_over'].play()
        if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
        self.worker.running = False
        self.final_score.setText(str(self.worker.score))
        QTimer.singleShot(100, lambda: self.central_widget.setCurrentIndex(7)) # Game Over is index 7

    def restart(self):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.central_widget.setCurrentIndex(0)

    def closeEvent(self, event):
        if self.worker:
            self.worker.running = False
            self.worker.wait()
        pygame.mixer.quit()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())