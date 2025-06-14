import cv2
import mediapipe as mp

# 1. Initialisation des solutions MediaPipe
mp_pose  = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cant open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Préparation de l'image (bgr -> rgb)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 3. Détection
    res_pose  = pose.process(img_rgb)
    res_hands = hands.process(img_rgb)

    # 4. Rendu sur l'image d'origine
    h, w, _ = frame.shape
    # — Trace le bras complet (épaule → coude → poignet)
    if res_pose.pose_landmarks:
        lm = res_pose.pose_landmarks.landmark
        # pour le bras droit
        shoulder = (int(lm[12].x * w), int(lm[12].y * h))
        elbow    = (int(lm[14].x * w), int(lm[14].y * h))
        wrist    = (int(lm[16].x * w), int(lm[16].y * h))
        cv2.line(frame, shoulder, elbow, (0, 255, 0), 4)
        cv2.line(frame, elbow,    wrist, (0, 255, 0), 4)


    # — Trace la main détectée
    if res_hands.multi_hand_landmarks:
        for hand_landmarks in res_hands.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow("Arm + Hand", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Échap pour quitter
        break

cap.release()
cv2.destroyAllWindows()
