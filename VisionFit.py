import cv2
import mediapipe as mp
import numpy as np
import time
import math


# ---------------------- Utilities ----------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def draw_text_center(frame, text, y, scale=1.0, color=(255, 255, 255), thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(10, (frame.shape[1] - tw) // 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def safe_norm(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def format_time_min_sec(total_seconds):
    mins = total_seconds // 60
    secs = total_seconds % 60
    return f"{mins} min {secs} sec"


# ---------------------- Config ----------------------
WINDOW_NAME = "VISION-FIT"

# ---------------------- Initialization ----------------------
cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

COLOR_ACCENT = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)

# ---------------------- App State ----------------------
mode = "start"
selected_exercise = "Push-Ups"
start_time = None
end_time = None

counter = 0
calories = 0.0
pu_stage = None

jj_ankle_base = None
jj_wrist_base = None
jj_state = "closed"
jj_last_count_time = 0.0
JJ_MIN_COUNT_INTERVAL = 0.45

session_finished = False
final_frame = None

# ---------------------- Main ----------------------
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ---------------------------------------------------------
        #  START SCREEN
        # ---------------------------------------------------------
        if mode == "start":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            disp = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

            # Auto scalable fonts
            title_scale = max(w / 1300, 1.1)
            option_scale = max(w / 2100, 0.75)
            small_scale = max(w / 2600, 0.55)

            draw_text_center(disp, "🏆 VISION-FIT", int(h * 0.22), title_scale, (0, 200, 255), 4)
            draw_text_center(disp,
                             "Select Exercise: P = Push-Ups    J = Jumping Jacks",
                             int(h * 0.36), option_scale, (220, 220, 220), 1)
            draw_text_center(disp,
                             f"Current selection: {selected_exercise}",
                             int(h * 0.46), option_scale, (255, 255, 255), 2)
            draw_text_center(disp,
                             "Press S to START  •  ESC to EXIT",
                             int(h * 0.60), option_scale, (200, 200, 200), 1)
            draw_text_center(disp,
                             "Press Q anytime during workout to finish session",
                             int(h * 0.88), small_scale, (180, 180, 180), 1)

            cv2.imshow(WINDOW_NAME, disp)

            k = cv2.waitKey(10) & 0xFF
            if k in (ord('p'), ord('P')):
                selected_exercise = "Push-Ups"
            elif k in (ord('j'), ord('J')):
                selected_exercise = "Jumping Jacks"
            elif k in (ord('s'), ord('S')):
                mode = "workout"
                start_time = time.time()
                counter = 0
                calories = 0.0
                session_finished = False
                final_frame = None
                pu_stage = None
                jj_ankle_base = None
                jj_wrist_base = None
                jj_state = "closed"
                jj_last_count_time = 0.0
            elif k == 27:
                break
            continue

        # ---------------------------------------------------------
        #   WORKOUT MODE
        # ---------------------------------------------------------
        if mode == "workout":
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            landmarks = None
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

            # ---------------- PUSH-UPS ----------------
            if selected_exercise == "Push-Ups":
                angle = None
                if landmarks:
                    try:
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]

                        angle = calculate_angle(shoulder, elbow, wrist)
                        cv2.putText(frame, f'Elbow: {int(angle)}°', (30, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_ACCENT, 2)

                        if angle > 160:
                            pu_stage = "up"
                        if angle < 90 and pu_stage == "up":
                            pu_stage = "down"
                            counter += 1
                            calories = counter * 0.5
                    except:
                        angle = None

                # --- FIXED PUSHUP MOVEMENT BAR ---
                try:
                    a = angle if angle is not None else calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h],
                        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h],
                        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
                    )
                    bar = int(np.clip(np.interp(a, (175, 75), (0, 350)), 0, 350))
                except:
                    bar = 0

                bar_top = int(h * 0.20)
                bar_bottom = int(h * 0.82)
                left_x1 = int(w * 0.055)
                left_x2 = int(w * 0.095)
                right_x1 = int(w * 0.905) - (left_x2 - left_x1)
                right_x2 = int(w * 0.905)

                cv2.rectangle(frame, (left_x1, bar_top), (left_x2, bar_bottom), (255, 255, 255), 2)
                cv2.rectangle(frame, (left_x1, bar_top + bar), (left_x2, bar_bottom), (0, 255, 255), -1)

                cv2.rectangle(frame, (right_x1, bar_top), (right_x2, bar_bottom), (255, 255, 255), 2)
                cv2.rectangle(frame, (right_x1, bar_top + bar), (right_x2, bar_bottom), (0, 255, 255), -1)

            # ---------------- JUMPING JACKS ----------------
            elif selected_exercise == "Jumping Jacks":
                if landmarks:
                    try:
                        l_wr = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
                        r_wr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
                        l_ank = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
                        r_ank = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
                        l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                        r_sh = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]

                        cur_ank_dist = safe_norm(l_ank, r_ank)
                        cur_wr_dist = safe_norm(l_wr, r_wr)

                        if jj_ankle_base is None and cur_ank_dist is not None:
                            jj_ankle_base = cur_ank_dist
                        if jj_wrist_base is None and cur_wr_dist is not None:
                            jj_wrist_base = cur_wr_dist

                        hands_up = (l_wr[1] < l_sh[1] * 0.92) and (r_wr[1] < r_sh[1] * 0.92)

                        ank_ratio = cur_ank_dist / (jj_ankle_base + 1e-8)
                        wr_ratio = cur_wr_dist / (jj_wrist_base + 1e-8)

                        now = time.time()
                        opened = (ank_ratio > 1.25) and (wr_ratio > 1.20 or hands_up)
                        closed = (ank_ratio < 1.15) and (wr_ratio < 1.15) and (not hands_up)

                        if jj_state == "closed" and opened:
                            jj_state = "open"

                        elif jj_state == "open" and closed:
                            if now - jj_last_count_time > JJ_MIN_COUNT_INTERVAL:
                                counter += 1
                                calories += 0.15
                                jj_last_count_time = now
                            jj_state = "closed"

                        else:
                            alpha = 0.01
                            if cur_ank_dist is not None:
                                jj_ankle_base = (1 - alpha) * jj_ankle_base + alpha * cur_ank_dist
                            if cur_wr_dist is not None:
                                jj_wrist_base = (1 - alpha) * jj_wrist_base + alpha * cur_wr_dist

                    except:
                        pass

                # Draw progress bars
                bar_top = int(h * 0.20)
                bar_bottom = int(h * 0.82)
                bar_height = bar_bottom - bar_top

                left_x1 = int(w * 0.055)
                left_x2 = int(w * 0.095)
                right_x1 = int(w * 0.905) - (left_x2 - left_x1)
                right_x2 = int(w * 0.905)

                try:
                    fill = int(np.interp(cur_wr_dist, (jj_wrist_base * 0.6, jj_wrist_base * 1.6), (0, bar_height)))
                except:
                    fill = 0


                def fill_color(val):
                    frac = val / float(bar_height) if bar_height else 0
                    return (0, int(255 * frac), int(255 * (1 - frac)))


                if fill > 0:
                    cv2.rectangle(frame, (left_x1, bar_bottom - fill), (left_x2, bar_bottom), fill_color(fill), -1)
                    cv2.rectangle(frame, (right_x1, bar_bottom - fill), (right_x2, bar_bottom), fill_color(fill), -1)

                cv2.rectangle(frame, (left_x1, bar_top), (left_x2, bar_bottom), (255, 255, 255), 1)
                cv2.rectangle(frame, (right_x1, bar_top), (right_x2, bar_bottom), (255, 255, 255), 1)

            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=2)
                )

            # UI HUD
            cv2.putText(frame, f"{selected_exercise}: {counter}",
                        (int(w * 0.05) + 60, int(h * 0.08)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # --- TIMER AT BOTTOM RIGHT ---
            elapsed = int(time.time() - start_time)
            time_text = f"Time: {elapsed // 60:02d}:{elapsed % 60:02d}"
            (text_w, text_h), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(frame, time_text, (w - text_w - 30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

            # Calories at top
            cv2.putText(frame, f"Calories: {calories:.1f}", (int(w * 0.65), int(h * 0.04) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

            draw_text_center(frame, "Press Q to End Session", int(h * 0.95), 0.6, (200, 200, 200), 1)

            cv2.imshow(WINDOW_NAME, frame)

            k = cv2.waitKey(10)
            if k == ord('q'):
                session_finished = True
                end_time = time.time()
                final_frame = frame.copy()
                mode = "finish"
                continue
            if k == ord('r'):
                mode = "start"
                selected_exercise = "Push-Ups"
                continue
            if k == 27:
                if not session_finished:
                    session_finished = True
                    end_time = time.time()
                    final_frame = frame.copy()
                    mode = "finish"
                else:
                    break
            continue

        # ---------------------------------------------------------
        #   FINISH SCREEN
        # ---------------------------------------------------------
        if mode == "finish":
            if final_frame is None:
                final_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            frame_final = final_frame.copy()
            overlay = frame_final.copy()
            cv2.rectangle(overlay, (int(w * 0.12), int(h * 0.18)),
                          (int(w * 0.88), int(h * 0.82)), (0, 0, 0), -1)
            frame_final = cv2.addWeighted(overlay, 0.75, frame_final, 0.25, 0)

            total_time = int(end_time - start_time) if start_time and end_time else 0
            time_str = format_time_min_sec(total_time)

            draw_text_center(frame_final, "🔥 GREAT WORK 🔥", int(h * 0.28), max(1.0, w / 1200), (0, 200, 50), 4)
            draw_text_center(frame_final, "YOU HAVE FINISHED YOUR WORKOUT!", int(h * 0.36), max(0.6, w / 2000),
                             (230, 230, 230), 1)
            draw_text_center(frame_final, f"Exercise: {selected_exercise}", int(h * 0.46), max(0.6, w / 2000),
                             (255, 255, 255), 1)
            draw_text_center(frame_final, f"Reps Completed: {counter}", int(h * 0.52), max(0.7, w / 1600),
                             (255, 255, 255), 2)
            draw_text_center(frame_final, f"Time: {time_str}", int(h * 0.58), max(0.6, w / 2000), (255, 255, 255), 1)
            draw_text_center(frame_final, f"Calories: {calories:.1f} kcal", int(h * 0.64), max(0.6, w / 2000),
                             (255, 255, 255), 1)
            draw_text_center(frame_final, "🏆 KEEP GOING — YOU ARE UNSTOPPABLE 🏆", int(h * 0.72), max(0.5, w / 2600),
                             (255, 200, 0), 1)
            draw_text_center(frame_final, "Press ESC or Q to Exit, or R to Return to Menu", int(h * 0.80),
                             max(0.45, w / 3000), (200, 200, 200), 1)

            cv2.imshow(WINDOW_NAME, frame_final)
            k = cv2.waitKey(10)
            if k in (ord('q'), 27):
                break
            if k == ord('r'):
                session_finished = False
                mode = "start"
                selected_exercise = "Push-Ups"
                counter = 0
                calories = 0.0
                start_time = None
                end_time = None
                final_frame = None
                continue

# Cleanup
cap.release()
cv2.destroyAllWindows()
