import cv2
import mediapipe as mp

capture = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = capture.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks and results.multi_handedness:

        for idx, handLms in enumerate(results.multi_hand_landmarks):

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            label = results.multi_handedness[idx].classification[0].label

            thumb_tip = handLms.landmark[4]
            pinky_tip = handLms.landmark[20]

            # Konversi ke pixel
            h, w, _ = img.shape
            thumb_x = int(thumb_tip.x * w)
            pinky_x = int(pinky_tip.x * w)

            if label == "Right":
                if thumb_x < pinky_x:
                    posisi = "TELAPAK (DEPAN)"
                else:
                    posisi = "PUNGGUNG (BELAKANG)"
            else:  # Left hand
                if thumb_x > pinky_x:
                    posisi = "TELAPAK (DEPAN)"
                else:
                    posisi = "PUNGGUNG (BELAKANG)"

            cv2.putText(img, posisi, (50,50),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        (0,255,0), 2)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()