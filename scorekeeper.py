import cv2
from ultralytics import YOLO

last_player_hit = 1
p1_score = 0
p2_score = 0

score_cooldown = 5

diff_pos_x = []
diff_pos_y = []
prev_vect_x = 0
prev_vect_y = 0

events_dict = {0: 'bounce', 1: 'empty_event', 2: 'net'}
expected_event = ""
event_class = ""
prev_event = ""
frames_not_seen = 0

count = 0

ball_model = YOLO('ball_best.pt')
events_model = YOLO('events_best.pt')

video_path = 'test_2.mp4'
cap = cv2.VideoCapture(video_path)

output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 60.0, (int(cap.get(3)), int(cap.get(4))))

font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    balls = ball_model.predict(frame)
    events = events_model.predict(frame)

    if score_cooldown >= 1:
        for event in events:
            try:
                print("Frame: ", count)
                if event.probs.top1conf.item() >= 0.99:
                    print("Class: ", event.probs.top1)
                    print("Confidence: ", event.probs.top1conf.item())
                    event_class = events_dict[event.probs.top1]
            except Exception as e:
                print(f"Error processing event: {e}")

        try:
            x, y, w, h = balls[0].boxes.xywh.tolist()[0]
            x1, y1, x2, y2 = balls[0].boxes.xyxy.tolist()[0]

            diff_vect_x = 0
            if len(diff_pos_x) >= 4:
                diff_pos_x.pop(0)
                diff_pos_x.append(x)
                for index, i in enumerate(diff_pos_x):
                    if index < len(diff_pos_x)-1:
                        diff_vect_x += diff_pos_x[index] - diff_pos_x[index+1]
                if prev_vect_x == 0:
                    prev_vect_x = diff_vect_x
            else:
                diff_pos_x.append(x)

            diff_vect_y = 0
            if len(diff_pos_y) >= 4:
                diff_pos_y.pop(0)
                diff_pos_y.append(y)
                for index, i in enumerate(diff_pos_y):
                    if index < len(diff_pos_y) - 1:
                        diff_vect_y += diff_pos_y[index] - diff_pos_y[index + 1]
                if prev_vect_y == 0:
                    prev_vect_y = diff_vect_y
            else:
                diff_pos_y.append(y)

            frames_not_seen = 0

            if int(diff_vect_y) >= 0 and int(prev_vect_y) < 0:
                if event_class == "bounce" and (expected_event == "bounce" or expected_event == ""):
                    expected_event = "empty_event"
                    prev_event = "bounce"
                    cv2.putText(frame,
                                "Bounce",
                                (200, 50),
                                font, 1,
                                (0, 255, 255),
                                2,
                                cv2.LINE_4)
                elif expected_event != "bounce" and event_class != prev_event:
                    expected_event = ""
                    event_class = ""
                    if last_player_hit == 1:
                        p1_score += 1
                    else:
                        p2_score += 1
                    score_cooldown = 7
            if int(diff_vect_x) < 0 and int(prev_vect_x) >= 0:
                if expected_event == "empty_event" or expected_event == "":
                    expected_event = "net"
                    last_player_hit = 1
                    cv2.putText(frame,
                                "Player 1 hit the ball",
                                (200, 50),
                                font, 1,
                                (0, 255, 255),
                                2,
                                cv2.LINE_4)
                elif expected_event != "empty_event" and event_class != "empty_event" and prev_event != "event_event":
                    expected_event = ""
                    event_class = ""
                    if last_player_hit == 1:
                        p2_score += 1
                    else:
                        p1_score += 1
                    score_cooldown = 7
            elif int(diff_vect_x) >= 0 and int(prev_vect_x) < 0:
                if expected_event == "empty_event" or expected_event == "":
                    expected_event = "net"
                    last_player_hit = 2
                    cv2.putText(frame,
                                "Player 2 hit the ball",
                                (200, 50),
                                font, 1,
                                (0, 255, 255),
                                2,
                                cv2.LINE_4)
                elif expected_event != "empty_event" and event_class != "empty_event" and prev_event != "event_event":
                    expected_event = ""
                    event_class = ""
                    if last_player_hit == 1:
                        p2_score += 1
                    else:
                        p1_score += 1
                    score_cooldown = 7
            if event_class == "net":
                if expected_event == "net" or expected_event == "":
                    expected_event = "bounce"
                    prev_event = "net"
                    cv2.putText(frame,
                                "Net",
                                (200, 50),
                                font, 1,
                                (0, 255, 255),
                                2,
                                cv2.LINE_4)
                elif expected_event != "net" and event_class != prev_event and event_class != "empty_event":
                    expected_event = ""
                    event_class = ""
                    if last_player_hit == 1:
                        p2_score += 1
                    else:
                        p1_score += 1
                    score_cooldown = 7

            prev_vect_x = diff_vect_x
            prev_vect_y = diff_vect_y

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        except Exception as e:
            print(e)
            frames_not_seen += 1
            if expected_event == "bounce" and frames_not_seen >= 10:
                expected_event = ""
                event_class = ""
                if last_player_hit == 1:
                    p2_score += 1
                else:
                    p1_score += 1

                score_cooldown = 7
                frames_not_seen = 0
                prev_vect_x = 0
                prev_vect_y = 0

    else:
        score_cooldown -= 1

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