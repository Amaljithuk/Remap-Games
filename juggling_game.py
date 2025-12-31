import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import random
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget, 
                             QFrame, QSizePolicy, QButtonGroup, QRadioButton, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QSize, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QBrush, QColor, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist

# --- EXE PATH HELPER (REQUIRED FOR PYINSTALLER) ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- GAME CONFIGURATION ---
LEVEL_CONFIG = {
    1: { "balls": 1, "gravity": 0.2, "base_speed": 4.0, "desc": "Level 1: Warm Up" },
    2: { "balls": 1, "gravity": 0.3, "base_speed": 6.0, "desc": "Level 2: Standard" },
    3: { "balls": 2, "gravity": 0.4, "base_speed": 7.0, "desc": "Level 3: Double Trouble" },
    4: { "balls": 2, "gravity": 0.55, "base_speed": 9.0, "desc": "Level 4: Expert Mode" }
}

# --- STYLESHEET ---
STYLESHEET = """
    QWidget {
        font-family: 'Segoe UI', sans-serif;
    }
    QLabel#Header {
        font-size: 32px;
        font-weight: bold;
        color: #00fff5;
        background-color: transparent;
    }
    QLabel#SubHeader {
        font-size: 18px;
        color: #e0e0e0;
        background-color: transparent;
    }
    QLabel#TimeLabel {
        font-size: 36px;
        font-weight: bold;
        color: #ffcc00;
        background-color: transparent;
    }
    QLabel#GameOverTitle {
        font-size: 60px;
        font-weight: bold;
        color: #ff2a6d;
    }
    QLabel#FinalScore {
        font-size: 40px;
        font-weight: bold;
        color: #00ff00;
    }
    QLabel#FinalStats {
        font-size: 24px;
        color: #ffffff;
    }
    QPushButton {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #16213e, stop:1 #0f3460);
        color: white;
        border: 2px solid #00fff5;
        border-radius: 15px;
        padding: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #e94560;
        border-color: #ff2a6d;
    }
    QPushButton:pressed {
        background-color: #1a1a2e;
        margin-top: 2px;
    }
    QPushButton.SmallBtn {
        border-radius: 10px;
        padding: 5px;
        font-size: 24px;
        min-width: 40px;
        max-width: 40px;
    }
    QRadioButton {
        color: white;
        font-size: 20px;
        spacing: 10px;
        background-color: transparent;
    }
    QRadioButton::indicator {
        width: 20px;
        height: 20px;
        border-radius: 10px;
        border: 2px solid #00fff5;
    }
    QRadioButton::indicator:checked {
        background-color: #00fff5;
        border: 2px solid white;
    }
    QFrame#HUD {
        background-color: rgba(0, 0, 0, 150);
        border-bottom: 2px solid #00fff5;
        border-radius: 0px;
    }
    QLabel#ScoreLabel { color: #00ff00; font-size: 24px; font-weight: bold; }
    QLabel#LevelLabel { color: #ffcc00; font-size: 24px; font-weight: bold; }
    QLabel#TimerLabel { color: #00fff5; font-size: 24px; font-weight: bold; }
"""

class GameLogic:
    def __init__(self, width=1280, height=720):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.width = width
        self.height = height
        
        # State
        self.active = False
        self.paused = False
        self.calibration_active = False
        self.calibration_mode = "Sitting"
        self.game_over = False
        
        # Calibration
        self.calibration_y = height - 100 
        self.play_area_min_x = 0
        self.play_area_max_x = width
        self.calibrated = False
        
        # Game Stats
        self.current_level = 1
        self.score = 0
        self.total_airtime = 0
        
        # Time Management
        self.selected_duration = 60
        self.time_remaining = 60
        self.last_time_check = 0
        
        # Physics
        self.balls = []
        self.prev_hand_positions = {}
        self.last_spawn_time = 0
        self.hand_radius = 50
        self.push_force = 15
        self.max_push_angle = 35

    def reset_game(self, level):
        self.current_level = level
        self.score = 0
        self.balls = []
        self.total_airtime = 0
        self.game_over = False
        self.paused = False
        self.last_spawn_time = time.time()
        self.prev_hand_positions = {}
        self.time_remaining = self.selected_duration
        self.last_time_check = time.time()

    def calibrate(self, hand_landmarks_list):
        if len(hand_landmarks_list) < 2: return False
        h, w = self.height, self.width
        wrist_1_x = hand_landmarks_list[0].landmark[0].x * w
        wrist_1_y = hand_landmarks_list[0].landmark[0].y * h
        wrist_2_x = hand_landmarks_list[1].landmark[0].x * w
        wrist_2_y = hand_landmarks_list[1].landmark[0].y * h
        avg_y = (wrist_1_y + wrist_2_y) / 2
        self.calibration_y = int(avg_y) - 50 
        min_x = min(wrist_1_x, wrist_2_x)
        max_x = max(wrist_1_x, wrist_2_x)
        if self.calibration_mode == "Sitting":
            self.play_area_min_x = int(min_x)
            self.play_area_max_x = int(max_x)
        else:
            span = max_x - min_x
            center = (min_x + max_x) / 2
            self.play_area_min_x = int(max(0, center - (span * 0.75)))
            self.play_area_max_x = int(min(self.width, center + (span * 0.75)))
        self.calibrated = True
        return True

    def spawn_ball(self):
        config = LEVEL_CONFIG[self.current_level]
        if len(self.balls) < config["balls"]:
            safe_min = self.play_area_min_x + 30
            safe_max = self.play_area_max_x - 30
            if safe_max <= safe_min: safe_min, safe_max = 50, self.width - 50
            x = random.randint(safe_min, safe_max)
            ball = {'pos': np.array([x, 50], dtype=float), 'vel': np.array([0, config["base_speed"]], dtype=float), 'radius': 25, 'color': (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)), 'airtime': 0, 'last_hit_time': 0}
            self.balls.append(ball)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        h, w, c = frame.shape
        hand_centers = []
        
        if self.calibrated or self.calibration_active:
            cv2.line(frame, (0, self.calibration_y), (w, self.calibration_y), (0, 255, 255), 2)
            cv2.line(frame, (self.play_area_min_x, 0), (self.play_area_min_x, h), (255, 0, 0), 2)
            cv2.line(frame, (self.play_area_max_x, 0), (self.play_area_max_x, h), (255, 0, 0), 2)
            cv2.putText(frame, f"Mode: {self.calibration_mode}", (self.play_area_min_x + 10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                if self.active and not self.paused:
                    indices = [0, 5, 9, 13, 17]
                    cx = int(np.mean([hand_landmarks.landmark[i].x * w for i in indices]))
                    cy = int(np.mean([hand_landmarks.landmark[i].y * h for i in indices]))
                    center = (cx, cy)
                    hand_centers.append(center)
                    if hand_idx in self.prev_hand_positions:
                        prev = self.prev_hand_positions[hand_idx]
                        velocity = np.array(center) - np.array(prev)
                    else: velocity = np.array([0,0])
                    self.prev_hand_positions[hand_idx] = center
                    self.check_collisions(center, velocity)
                    cv2.circle(frame, center, self.hand_radius, (0, 255, 0), 2)
            if self.calibration_active: self.calibrate(results.multi_hand_landmarks)
        
        if self.active and not self.paused and not self.game_over:
            current_time = time.time()
            if current_time - self.last_time_check >= 1.0:
                self.time_remaining -= 1
                self.last_time_check = current_time
                if self.time_remaining <= 0:
                    self.game_over = True
                    self.time_remaining = 0
            self.update_physics()
            self.spawn_ball()
            self.draw_balls(frame)
        return frame

    def check_collisions(self, hand_center, hand_vel):
        current_time = time.time()
        for ball in self.balls:
            dist = np.linalg.norm(ball['pos'] - np.array(hand_center))
            if dist < (ball['radius'] + self.hand_radius) and (current_time - ball['last_hit_time']) > 0.3:
                direction = ball['pos'] - np.array(hand_center)
                norm = np.linalg.norm(direction)
                if norm > 0: direction /= norm
                angle = np.degrees(np.arctan2(direction[0], -direction[1]))
                angle = np.clip(angle, -self.max_push_angle, self.max_push_angle)
                rad = np.radians(angle)
                direction = np.array([np.sin(rad), -np.cos(rad)])
                speed = np.linalg.norm(hand_vel)
                force = self.push_force + min(speed * 0.5, 15)
                ball['vel'] = direction * force
                ball['last_hit_time'] = current_time
                self.score += 1

    def update_physics(self):
        config = LEVEL_CONFIG[self.current_level]
        balls_to_remove = []
        for i, ball in enumerate(self.balls):
            ball['vel'][1] += config['gravity']
            ball['pos'] += ball['vel']
            if ball['pos'][0] < self.play_area_min_x + ball['radius']:
                ball['vel'][0] *= -0.8
                ball['pos'][0] = self.play_area_min_x + ball['radius']
            elif ball['pos'][0] > self.play_area_max_x - ball['radius']:
                ball['vel'][0] *= -0.8
                ball['pos'][0] = self.play_area_max_x - ball['radius']
            if ball['pos'][1] < ball['radius']:
                ball['vel'][1] *= -0.5
                ball['pos'][1] = ball['radius']
            if ball['pos'][1] > self.height + 50:
                balls_to_remove.append(i)
                self.total_airtime += ball['airtime']
            else: ball['airtime'] += 0.03
        for i in reversed(balls_to_remove): self.balls.pop(i)

    def draw_balls(self, frame):
        for ball in self.balls:
            pos = tuple(ball['pos'].astype(int))
            cv2.circle(frame, pos, ball['radius'], ball['color'], -1)
            cv2.circle(frame, pos, ball['radius'], (255,255,255), 2)

# --- WORKER THREAD ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    stats_signal = pyqtSignal(dict)
    def __init__(self, game_logic):
        super().__init__()
        self.game = game_logic
        self.running = True
    
    def run(self):
        # --- CAMERA CHECK LOGIC ---
        source = 0
        test_cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if test_cap.isOpened():
            source = 1
            test_cap.release()
        else:
            source = 0
        
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                processed_frame = self.game.process_frame(frame)
                stats = {
                    "score": self.game.score, 
                    "level": self.game.current_level, 
                    "game_over": self.game.game_over, 
                    "calibrated": self.game.calibrated,
                    "time": self.game.time_remaining,
                    "airtime": self.game.total_airtime
                }
                self.stats_signal.emit(stats)
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_Qt_format.scaled(1280, 720, Qt.AspectRatioMode.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)
            time.sleep(0.01)
        cap.release()
    def stop(self):
        self.running = False
        self.wait()

# --- MAIN WINDOW ---
class JugglingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual Juggling Pro")
        self.setGeometry(100, 100, 1280, 720)
        self.game_logic = GameLogic()
        
        # --- SOUND INITIALIZATION ---
        self.init_sounds()
        
        self.last_known_score = 0
        
        self.setStyleSheet(STYLESHEET)
        
        self.central_widget = QWidget()
        self.central_widget.setObjectName("CentralScreen")
        self.setCentralWidget(self.central_widget)
        
        self.set_background_image()

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.stack = QStackedWidget()
        self.main_layout.addWidget(self.stack)
        
        self.init_menu_screen()
        self.init_calibration_screen()
        self.init_game_screen()
        self.init_result_screen()
        
        self.thread = VideoThread(self.game_logic)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.stats_signal.connect(self.update_stats)
        self.thread.start()
        
        self.stack.setCurrentIndex(0)
        self.setFocusPolicy(Qt.StrongFocus)

    def init_sounds(self):
        """Setup media players for music and sfx with PATH WRAPPING"""
        # Music Player
        self.music_player = QMediaPlayer()
        self.playlist = QMediaPlaylist()
        
        bg_path = resource_path("bg.mp3")
        if os.path.exists(bg_path):
            self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(bg_path)))
            self.playlist.setPlaybackMode(QMediaPlaylist.Loop)
            self.music_player.setPlaylist(self.playlist)
            self.music_player.setVolume(20)
        
        # Hit Sound
        self.hit_player = QMediaPlayer()
        hit_path = resource_path("jug_hit.mp3")
        if os.path.exists(hit_path):
            self.hit_player.setMedia(QMediaContent(QUrl.fromLocalFile(hit_path)))
            self.hit_player.setVolume(100)
            
        # Click Sound
        self.click_player = QMediaPlayer()
        click_path = resource_path("click.mp3")
        if os.path.exists(click_path):
            self.click_player.setMedia(QMediaContent(QUrl.fromLocalFile(click_path)))
            self.click_player.setVolume(100)

    def play_click(self):
        if self.click_player.mediaStatus() != QMediaPlayer.NoMedia:
            if self.click_player.state() == QMediaPlayer.PlayingState:
                self.click_player.stop()
            self.click_player.play()

    def play_hit(self):
        if self.hit_player.mediaStatus() != QMediaPlayer.NoMedia:
            if self.hit_player.state() == QMediaPlayer.PlayingState:
                self.hit_player.stop()
            self.hit_player.play()

    def start_music(self):
        if self.music_player.playlist() is not None:
            self.music_player.play()

    def stop_music(self):
        self.music_player.stop()

    def set_background_image(self):
        # WRAP PATH
        bg_path = resource_path("assets/jug_logo.png")
        style = "#CentralScreen { background-color: #1a1a2e; }"
        if os.path.exists(bg_path):
            clean_path = bg_path.replace("\\", "/")
            style = f"#CentralScreen {{ border-image: url({clean_path}) 0 0 0 0 stretch stretch; }}"
        self.central_widget.setStyleSheet(style)

    def add_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 200))
        shadow.setOffset(2, 2)
        widget.setGraphicsEffect(shadow)

    def keyPressEvent(self, event):
        if self.stack.currentIndex() == 0:
            if event.key() == Qt.Key_Right or event.key() == Qt.Key_Up:
                self.change_duration(30)
            elif event.key() == Qt.Key_Left or event.key() == Qt.Key_Down:
                self.change_duration(-30)
        super().keyPressEvent(event)

    def change_duration(self, amount):
        self.play_click()
        new_time = self.game_logic.selected_duration + amount
        if 30 <= new_time <= 300:
            self.game_logic.selected_duration = new_time
            self.update_time_label()

    def update_time_label(self):
        mins = self.game_logic.selected_duration // 60
        secs = self.game_logic.selected_duration % 60
        self.lbl_time_display.setText(f"{mins:02d}:{secs:02d}")

    def init_menu_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(100, 30, 100, 30)
        layout.setSpacing(15)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        logo_label = QLabel("VIRTUAL JUGGLING")
        logo_label.setObjectName("Header")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_shadow(logo_label)
        
        # WRAP PATH
        jug_bg_path = resource_path("assets/jug_bg.png")
        if os.path.exists(jug_bg_path):
            pixmap = QPixmap(jug_bg_path).scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setText("")

        settings_group = QFrame()
        settings_group.setStyleSheet("background-color: rgba(0,0,0,100); border-radius: 10px; padding: 10px;")
        settings_layout = QVBoxLayout(settings_group)
        
        lbl_mode = QLabel("STANCE")
        lbl_mode.setObjectName("SubHeader")
        lbl_mode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        radio_layout = QHBoxLayout()
        self.rb_sitting = QRadioButton(" Sitting")
        self.rb_standing = QRadioButton(" Standing")
        self.rb_sitting.setChecked(True)
        # Add click sounds to radios
        self.rb_sitting.clicked.connect(self.play_click)
        self.rb_standing.clicked.connect(self.play_click)
        
        radio_layout.addStretch()
        radio_layout.addWidget(self.rb_sitting)
        radio_layout.addWidget(self.rb_standing)
        radio_layout.addStretch()
        
        lbl_time = QLabel("GAME DURATION")
        lbl_time.setObjectName("SubHeader")
        lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        time_layout = QHBoxLayout()
        btn_minus = QPushButton("<")
        btn_minus.setProperty("class", "SmallBtn")
        btn_minus.clicked.connect(lambda: self.change_duration(-30))
        
        self.lbl_time_display = QLabel("01:00")
        self.lbl_time_display.setObjectName("TimeLabel")
        self.lbl_time_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_shadow(self.lbl_time_display)
        
        btn_plus = QPushButton(">")
        btn_plus.setProperty("class", "SmallBtn")
        btn_plus.clicked.connect(lambda: self.change_duration(30))
        
        time_layout.addStretch()
        time_layout.addWidget(btn_minus)
        time_layout.addSpacing(20)
        time_layout.addWidget(self.lbl_time_display)
        time_layout.addSpacing(20)
        time_layout.addWidget(btn_plus)
        time_layout.addStretch()
        
        settings_layout.addWidget(lbl_mode)
        settings_layout.addLayout(radio_layout)
        settings_layout.addSpacing(10)
        settings_layout.addWidget(lbl_time)
        settings_layout.addLayout(time_layout)
        
        btn_layout = QVBoxLayout()
        for lvl, data in LEVEL_CONFIG.items():
            btn = QPushButton(data['desc'])
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(45)
            btn.clicked.connect(lambda checked, l=lvl: self.start_calibration_setup(l))
            btn_layout.addWidget(btn)

        btn_quit = QPushButton("EXIT GAME")
        btn_quit.setStyleSheet("background-color: #4a0e0e; border-color: #ff0000;")
        btn_quit.setMinimumHeight(45)
        btn_quit.clicked.connect(self.close)
        btn_quit.clicked.connect(self.play_click) # Though app closes fast, good practice

        layout.addWidget(logo_label)
        layout.addWidget(settings_group)
        layout.addLayout(btn_layout)
        layout.addWidget(btn_quit)
        
        page.setLayout(layout)
        self.stack.addWidget(page)

    def init_calibration_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        instr_frame = QFrame()
        instr_frame.setStyleSheet("background-color: rgba(0,0,0,200); padding: 20px;")
        instr_layout = QVBoxLayout(instr_frame)
        
        lbl_title = QLabel("CALIBRATION REQUIRED")
        lbl_title.setObjectName("Header")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_calib_instr = QLabel("Step 1: Stretch BOTH hands out to your sides.\nStep 2: Hold steady until lines appear.\nStep 3: Click Confirm.")
        self.lbl_calib_instr.setObjectName("SubHeader")
        self.lbl_calib_instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        instr_layout.addWidget(lbl_title)
        instr_layout.addWidget(self.lbl_calib_instr)
        
        self.video_label_calib = QLabel()
        self.video_label_calib.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label_calib.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        btn_frame = QFrame()
        btn_frame.setStyleSheet("background-color: transparent;")
        btn_layout = QHBoxLayout(btn_frame)
        
        btn_set = QPushButton("CONFIRM CALIBRATION")
        btn_set.setMinimumHeight(60)
        btn_set.setCursor(Qt.PointingHandCursor)
        btn_set.clicked.connect(self.finish_calibration)
        
        btn_back = QPushButton("BACK")
        btn_back.setFixedWidth(150)
        btn_back.setMinimumHeight(60)
        btn_back.clicked.connect(self.return_to_menu)
        
        btn_layout.addWidget(btn_back)
        btn_layout.addWidget(btn_set)
        
        layout.addWidget(instr_frame)
        layout.addWidget(self.video_label_calib)
        layout.addWidget(btn_frame)
        
        page.setLayout(layout)
        self.stack.addWidget(page)

    def init_game_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        hud_frame = QFrame()
        hud_frame.setObjectName("HUD")
        hud_layout = QHBoxLayout(hud_frame)
        hud_layout.setContentsMargins(20, 10, 20, 10)
        
        self.lbl_score = QLabel("SCORE: 000")
        self.lbl_score.setObjectName("ScoreLabel")
        self.add_shadow(self.lbl_score)
        
        self.lbl_level = QLabel("LEVEL: 1")
        self.lbl_level.setObjectName("LevelLabel")
        self.add_shadow(self.lbl_level)
        
        self.lbl_timer = QLabel("TIME: 01:00")
        self.lbl_timer.setObjectName("TimerLabel")
        self.add_shadow(self.lbl_timer)
        
        self.lbl_status = QLabel("ACTIVE")
        self.lbl_status.setStyleSheet("color: white; font-weight: bold; font-size: 18px;")
        
        btn_menu = QPushButton("PAUSE / MENU")
        btn_menu.setFixedSize(150, 40)
        btn_menu.setStyleSheet("font-size: 14px; padding: 5px;")
        btn_menu.clicked.connect(self.return_to_menu)
        
        hud_layout.addWidget(self.lbl_score)
        hud_layout.addSpacing(30)
        hud_layout.addWidget(self.lbl_level)
        hud_layout.addSpacing(30)
        hud_layout.addWidget(self.lbl_timer)
        hud_layout.addStretch()
        hud_layout.addWidget(self.lbl_status)
        hud_layout.addSpacing(20)
        hud_layout.addWidget(btn_menu)
        
        self.video_label_game = QLabel()
        self.video_label_game.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label_game.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout.addWidget(hud_frame)
        layout.addWidget(self.video_label_game)
        
        page.setLayout(layout)
        self.stack.addWidget(page)

    def init_result_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        container = QFrame()
        container.setStyleSheet("background-color: rgba(0,0,0,200); border: 2px solid #ff2a6d; border-radius: 20px; padding: 50px;")
        con_layout = QVBoxLayout(container)
        con_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_title = QLabel("TIME'S UP!")
        lbl_title.setObjectName("GameOverTitle")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_shadow(lbl_title)
        
        self.lbl_final_score = QLabel("FINAL SCORE: 0")
        self.lbl_final_score.setObjectName("FinalScore")
        self.lbl_final_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_shadow(self.lbl_final_score)
        
        self.lbl_final_stats = QLabel("Total Airtime: 0s")
        self.lbl_final_stats.setObjectName("FinalStats")
        self.lbl_final_stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        btn_row = QHBoxLayout()
        btn_row.setSpacing(20)
        
        btn_restart = QPushButton("MAIN MENU")
        btn_restart.setMinimumWidth(200)
        btn_restart.clicked.connect(self.return_to_menu)
        
        btn_quit = QPushButton("QUIT GAME")
        btn_quit.setMinimumWidth(200)
        btn_quit.setStyleSheet("background-color: #4a0e0e; border-color: #ff0000;")
        btn_quit.clicked.connect(self.close)
        
        btn_row.addWidget(btn_restart)
        btn_row.addWidget(btn_quit)
        
        con_layout.addWidget(lbl_title)
        con_layout.addSpacing(20)
        con_layout.addWidget(self.lbl_final_score)
        con_layout.addWidget(self.lbl_final_stats)
        con_layout.addSpacing(40)
        con_layout.addLayout(btn_row)
        
        layout.addWidget(container)
        page.setLayout(layout)
        self.stack.addWidget(page)

    def start_calibration_setup(self, level):
        self.play_click()
        self.selected_level = level
        mode = "Sitting" if self.rb_sitting.isChecked() else "Standing"
        self.game_logic.calibration_mode = mode
        self.game_logic.active = False
        self.game_logic.calibration_active = True
        self.game_logic.calibrated = False
        self.stack.setCurrentIndex(1)

    def finish_calibration(self):
        self.play_click()
        if self.game_logic.calibrated:
            self.game_logic.calibration_active = False
            self.start_game()
        else:
            self.lbl_calib_instr.setText("❌ HANDS NOT DETECTED! Please stretch arms fully.")
            self.lbl_calib_instr.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 18px;")

    def start_game(self):
        self.game_logic.reset_game(self.selected_level)
        self.last_known_score = 0
        self.game_logic.active = True
        self.stack.setCurrentIndex(2)
        self.start_music() # Start background music

    def return_to_menu(self):
        self.play_click()
        self.stop_music() # Stop music
        self.game_logic.active = False
        self.game_logic.calibration_active = False
        self.stack.setCurrentIndex(0)

    def update_image(self, qt_img):
        idx = self.stack.currentIndex()
        if idx == 1:
            self.video_label_calib.setPixmap(QPixmap.fromImage(qt_img))
        elif idx == 2:
            self.video_label_game.setPixmap(QPixmap.fromImage(qt_img))

    def update_stats(self, stats):
        if self.stack.currentIndex() == 2:
            current_score = stats['score']
            self.lbl_score.setText(f"SCORE: {current_score:03d}")
            self.lbl_level.setText(f"LEVEL: {stats['level']}")
            
            # --- DETECT HIT (Sound Logic) ---
            # If score increased from last frame, play hit sound
            if current_score > self.last_known_score:
                self.play_hit()
                self.last_known_score = current_score
            
            rem_sec = stats['time']
            mins = rem_sec // 60
            secs = rem_sec % 60
            self.lbl_timer.setText(f"TIME: {mins:02d}:{secs:02d}")
            if rem_sec < 10:
                 self.lbl_timer.setStyleSheet("color: red; font-size: 24px; font-weight: bold;")
            else:
                 self.lbl_timer.setStyleSheet("color: #00fff5; font-size: 24px; font-weight: bold;")
            
            if stats['game_over']:
                self.game_logic.active = False
                self.show_results(stats)

    def show_results(self, stats):
        self.stop_music()
        self.lbl_final_score.setText(f"FINAL SCORE: {stats['score']}")
        self.lbl_final_stats.setText(f"Level: {stats['level']}  |  Total Airtime: {stats['airtime']:.1f}s")
        self.stack.setCurrentIndex(3)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JugglingWindow()
    window.show()
    sys.exit(app.exec())