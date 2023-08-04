import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Accessing the landmarks by their position
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append((lm.x, lm.y))  # Store the normalized (x, y) coordinates in lm_list

            # Draw blue circles around the fingertips
            for tip_id in finger_tips:
                x, y = int(lm_list[tip_id][0] * w), int(lm_list[tip_id][1] * h)
                cv2.circle(img, (x, y), 10, (255, 0, 0), -1)

            # Check if fingers are folded or not
            finger_fold_status = []
            for i in range(len(finger_tips) - 1):
                x_tip = int(lm_list[finger_tips[i]][0] * w)
                x_next_tip = int(lm_list[finger_tips[i + 1]][0] * w)

                if x_tip < x_next_tip:
                    cv2.circle(img, (x_tip, y), 10, (0, 255, 0), -1)  # Draw green circle
                    finger_fold_status.append(True)
                else:
                    cv2.circle(img, (x_tip, y), 10, (0, 0, 255), -1)  # Draw red circle
                    finger_fold_status.append(False)

            # Check if all fingers are folded
            if all(finger_fold_status):
                # Check if the thumb is raised up or down
                y_tip = int(lm_list[finger_tips[0]][1] * h)
                y_prev_tip = int(lm_list[thumb_tip][1] * h)

                if y_tip < y_prev_tip:
                    print("LIKE")
                    cv2.putText(img, "LIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                else:
                    print("DISLIKE")
                    cv2.putText(img, "DISLIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("hand tracking", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
