import cv2
from pathlib import Path
import torch
from ultralytics import YOLO
import numpy as np

p1_score = 0
p2_score = 0

prev_pos = np.array([0, 0])
prev_vect = np.array([0, 0])

events_dict = {0: 'bounce', 1: 'empty_event', 2: 'net'}
serve_order = ['empty_event', 'bounce', 'net', 'bounce']
events_order = ['empty_event', 'net', 'bounce']
expected_event = ""
event_class = ""
prev_event = ""

count = 0

ball_model = YOLO('ball_best.pt')
events_model = YOLO('medium_events_best.pt')

video_path = 'test_2.mp4'
cap = cv2.VideoCapture(video_path)

output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 2.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    balls = ball_model.predict(frame)
    events = events_model.predict(frame)

    for event in events:
        try:
            print("Frame: ", count)
            if event.probs.top1conf.item() >= 0.99:
                print("Class: ", event.probs.top1)
                print("Confidence: ", event.probs.top1conf.item())
                event_class = events_dict[event.probs.top1] # Get the class predicted based on label given
        except Exception as e:
            print(f"Error processing event: {e}")

    try:
        x, y, w, h = balls[0].boxes.xywh.tolist()[0]
        x1, y1, x2, y2 = balls[0].boxes.xyxy.tolist()[0]
        new_pos = np.array([x, y])
        diff_vect = new_pos - prev_pos  # Get the new resulting vector based on the new position and previous position

        if int(diff_vect[1]) < 0 and int(prev_vect[1]) >= 0:
            if event_class == "bounce" and (expected_event == "bounce" or expected_event == ""):
                expected_event = "empty_event"
                prev_event = "bounce"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,
                            "Bounce",
                            (200, 50),
                            font, 1,
                            (0, 255, 255),
                            2,
                            cv2.LINE_4)
            elif expected_event != "bounce" and event_class != prev_event:
                print("Somebody scored")
                expected_event = ""
                p1_score += 1
        if int(diff_vect[0]) >= 0 and int(prev_vect[0]) < 0:
            if expected_event == "empty_event" or expected_event == "":
                expected_event = "net"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,
                            "Player 1 hit the ball",
                            (200, 50),
                            font, 1,
                            (0, 255, 255),
                            2,
                            cv2.LINE_4)
            elif expected_event != "empty_event" and event_class != "empty_event" and prev_event != "event_event":
                print("Somebody scored")
                expected_event = ""
                p1_score += 1
        elif int(diff_vect[0]) < 0 and int(prev_vect[0]) >= 0:
            if expected_event == "empty_event" or expected_event == "":
                expected_event = "net"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,
                            "Player 2 hit the ball",
                            (200, 50),
                            font, 1,
                            (0, 255, 255),
                            2,
                            cv2.LINE_4)
            elif expected_event != "empty_event" and event_class != "empty_event" and prev_event != "event_event":
                print("Somebody scored")
                expected_event = ""
                p1_score += 1
        if event_class == "net":
            if expected_event == "net" or expected_event == "":
                expected_event = "bounce"
                prev_event = "net"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,
                            "Net",
                            (200, 50),
                            font, 1,
                            (0, 255, 255),
                            2,
                            cv2.LINE_4)
            elif expected_event != "net" and event_class != prev_event:
                print("Somebody scored")
                expected_event = ""
                p1_score += 1

        prev_pos = new_pos
        prev_vect = diff_vect

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    except Exception as e:
        print(e)
        if expected_event == "bounce":
            print("Somebody scored")
            expected_event = ""
            p1_score += 1

    print('----------------------------------------------------------------------------------------')

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
                str(p1_score) + ' | ' + str(p2_score),
                (900, 50),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)
    # cv2.imshow('Video Player', frame)
    cv2.waitKey(1000//120)
    closeButton = cv2.getWindowProperty('Video Player', cv2.WND_PROP_VISIBLE) < 1

    out.write(frame)
    count += 1


cap.release()
out.release()

cv2.destroyAllWindows()