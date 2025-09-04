import face_recognition
import cv2
import os
import numpy as np
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from PIL import Image, ImageDraw, ImageFont
import pyttsx3   # 語音播報

# === 初始化 TTS 引擎 ===
engine = pyttsx3.init()
engine.setProperty("rate", 170)   # 語速
engine.setProperty("volume", 1.0) # 音量

# === Google Sheets 認證 ===
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open("Attendance").sheet1  # 預設第一個工作表

# === 學生照片資料夾 ===
STUDENTS_DIR = "students"

# 儲存 encoding 和姓名
known_encodings = []
known_names = []

# 載入學生資料
for student_name in os.listdir(STUDENTS_DIR):
    student_folder = os.path.join(STUDENTS_DIR, student_name)
    if not os.path.isdir(student_folder):
        continue

    for filename in os.listdir(student_folder):
        filepath = os.path.join(student_folder, filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(student_name)

print(f"已載入 {len(set(known_names))} 位學生的臉")

# === 出席紀錄 ===
today = datetime.date.today().strftime("%Y-%m-%d")
attendance = {name: {"狀態": "缺席", "時間": ""} for name in set(known_names)}

# 遲到判斷時間 (07:30)
late_time = datetime.time(7, 30, 0)

# === 中文繪字函式 (用 PIL) ===
def draw_chinese_text(frame, text, pos, color=(0, 255, 0), font_size=28):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("msjh.ttc", font_size, encoding="utf-8")
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# === 繪製簽到看板 ===
def draw_attendance_board(frame, attendance, x_start=650, y_start=30):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("msjh.ttc", 24, encoding="utf-8")

    draw.text((x_start, y_start), "📋 出席狀態", font=font, fill=(255, 255, 0))
    y_offset = y_start + 40

    for name, record in attendance.items():
        status = record["狀態"]
        time = record["時間"]
        text = f"{name} - {status} {time}"
        color = (0, 255, 0) if status == "出席" else (0, 0, 255) if status == "遲到" else (200, 200, 200)
        draw.text((x_start, y_offset), text, font=font, fill=color)
        y_offset += 30

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# === 開啟 USB 攝影機 ===
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        min_distance = np.min(distances)
        best_match_index = np.argmin(distances)

        if min_distance < 0.45:  # 閾值可調整
            name = known_names[best_match_index]

            if attendance[name]["狀態"] == "缺席":
                now = datetime.datetime.now()
                arrival_time = now.strftime("%H:%M:%S")

                if now.time() <= late_time:
                    status = "出席"
                else:
                    status = "遲到"

                attendance[name] = {"狀態": status, "時間": arrival_time}
                display_text = f"{name} - {status}"

                # ⚡ 播報語音
                engine.say(f"{name} 已簽到")
                engine.runAndWait()

            else:
                display_text = f"{name} - 已打卡"
        else:
            display_text = "未知 - Unknown"

        # 畫框
        top, right, bottom, left = face_location
        top, right, bottom, left = top*4, right*4, bottom*4, left*4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        frame = draw_chinese_text(frame, display_text, (left, top-40))

    frame = draw_attendance_board(frame, attendance)

    cv2.imshow("人臉點名系統", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# === 上傳到 Google Sheet ===
sheet.append_row(["日期", "姓名", "到校時間", "狀態"])  # 標題列
for name, record in attendance.items():
    sheet.append_row([today, name, record["時間"], record["狀態"]])

print("✅ 點名完成，已上傳 Google Sheet")
