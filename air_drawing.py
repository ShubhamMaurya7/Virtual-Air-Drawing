import cv2
import numpy as np
from utils.hand_detector import HandDetector

cap = cv2.VideoCapture(0)
canvas = None

detector = HandDetector()
xp, yp = 0, 0  # Previous finger positions

# Drawing settings
draw_color = (255, 0, 0)
brush_thickness = 7

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    frame = detector.find_hands(frame)
    lm_list = detector.find_position(frame)

    if lm_list:
        x1, y1 = lm_list[8][1:]  

        if len(lm_list) > 12:
            index_tip_y = lm_list[8][2]
            index_base_y = lm_list[6][2]
            middle_tip_y = lm_list[12][2]
            middle_base_y = lm_list[10][2]

            index_up = index_tip_y < index_base_y
            middle_up = middle_tip_y < middle_base_y

            if index_up and not middle_up:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0

    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, mask)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Virtual Air Drawing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
