import cmath
import cv2
import mediapipe as mp
import math

import numpy as np

meter2pixel = 100

black_screen = np.zeros((800, 1280, 3), dtype=np.uint8)
anchor_a1_position = (640, 0)
anchor_a2_position = (740, 0)
def calculate_angle(a, b, c):
   
    angle = math.degrees(math.acos((a**2 + b**2 - c**2) / (2 * a * b)))
    return angle

def draw_uwb_tag(x, y):
    global black_screen  
    black_screen.fill(0)  

    height, width, _ = black_screen.shape
    center_x = width // 2
    center_y = height //2

    cv2.circle(black_screen, (center_x, 0), 30, (0, 0, 255), -1)
    cv2.circle(black_screen, (center_x+100, 0), 30, (0, 0, 255), -1)
    # pos_x = center_x + int(x * meter2pixel) if x < 0 else center_x - int(x * meter2pixel)
    # pos_y = center_y + int(y * meter2pixel) if y < 0 else center_y - int(y * meter2pixel)
    pos_x = center_x - int(x * meter2pixel)
    pos_y = center_y - int(y * meter2pixel)
    r = 10
    cv2.circle(black_screen, (pos_x, pos_y), r, (0, 0, 255), -1)

    cv2.namedWindow('Black Screen', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Black Screen', width, height)

    cv2.imshow('Black Screen', black_screen)


def tag_pos(a, b, c):
    # p = (a + b + c) / 2.0
    # s = cmath.sqrt(p * (p - a) * (p - b) * (p - c))
    # y = 2.0 * s / c
    # x = cmath.sqrt(b * b - y * y)
    if(a==0 or b==0):
        return
    cos_a = (b * b + c*c - a * a) / (2 * b * c)
    x = b * cos_a
    y = b * cmath.sqrt(1 - cos_a * cos_a)

    return round(x.real, 1), round(y.real, 1)

def calculate_distance_to_camera(known_width, focal_length, perceived_width):
   
    distance = (known_width * focal_length) / perceived_width
    return distance / 100  

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
   
    known_width_cm = 50  #  omuz genişliği (cm)
    known_width_mm = known_width_cm * 10  #  omuz genişliği (mm)
    focal_length_mm = 0.15  # Odak uzaklığı (mm)

    cap1 = cv2.VideoCapture(2)  
    cap2 = cv2.VideoCapture(3)  

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap1.isOpened() and cap2.isOpened():
           
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            image_rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            image_rgb1.flags.writeable = False

            image_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            image_rgb2.flags.writeable = False

            results1 = pose.process(image_rgb1)
            results2 = pose.process(image_rgb2)

            if results1.pose_landmarks and results2.pose_landmarks:
               
                left_shoulder1 = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder1 = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                
                left_distance = abs(calculate_distance_to_camera(known_width_mm, focal_length_mm, abs(left_shoulder1.z)))
                right_distance = abs(calculate_distance_to_camera(known_width_mm, focal_length_mm, abs(right_shoulder1.z)))
                percieved_width = math.sqrt((left_shoulder1.x - right_shoulder1.x) ** 2 + (left_shoulder1.y - right_shoulder1.y) ** 2)
                distance1 = calculate_distance_to_camera(known_width_mm, 0.1, percieved_width)
                cv2.putText(frame1, f"posx: {distance1:.2f} p ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"distance1 : {distance1}")

                
            
                
                left_shoulder2 = results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder2 = results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

               
                left_distance = abs(calculate_distance_to_camera(known_width_mm, focal_length_mm, abs(left_shoulder2.z)))
                right_distance = abs(calculate_distance_to_camera(known_width_mm, focal_length_mm, abs(right_shoulder2.z)))
                percieved_width1 = math.sqrt((left_shoulder2.x - right_shoulder2.x) ** 2 + (left_shoulder2.y - right_shoulder2.y) ** 2)
                distance2= calculate_distance_to_camera(known_width_mm, focal_length_mm, percieved_width1)

                print(f"distance2 : {distance2}")
            
                    
                x, y = tag_pos(distance2, distance1, 1.0)
                cv2.putText(frame2, f"posx: {x:.2f} posy :{y} ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
              
                draw_uwb_tag(x, y)
                
            mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Camera 1', frame1)
            cv2.imshow('Camera 2', frame2)
            cv2.imshow('Black Screen', black_screen)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
