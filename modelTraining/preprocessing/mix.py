import cv2
import mediapipe as mp
import os
import csv

import featureExtract

csv_file = open("modelTraining/dataset.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
# Definisci la lista delle colonne
columns = ['Video']

for i in range(1, 31):
    columns.extend([
        f'R_ARM_LN{i}', f'L_ARM_LN{i}', f'R_LEG_LN{i}', f'L_LEG_LN{i}',
        f'R_SHOULDER_AG{i}', f'L_SHOULDER_AG{i}', f'R_HIP_AG{i}', f'L_HIP_AG{i}',
        f'SHOULDER_LN{i}', f'TORSO_LN{i}', f'RLTORSO_RAT{i}', f'LL_TORSO_RAT{i}',
        f'MASS_CENTER_X{i}', f'MASS_CENTER_Y{i}'
    ])
    for j in range(1, 17):
        columns.extend([
            f'ADD{j}_{i}'
        ])

# Scrivi la riga nel file CSV
csv_writer.writerow(columns)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

video_folder = "modelTraining/videos/segmenti"

BG_COLOR = (192, 192, 192)  # gray

for filename in os.listdir(video_folder):
    video_path = os.path.join(video_folder, filename)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))




    def swap(a, b):
        return b, a

    with mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5) as pose:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = 30  # Numero di frame da elaborare per ogni segmento di video
        interval = frame_count // num_frames
        frame_num = 0
        frame_index = 0
        video_name = os.path.splitext(filename)[0]  # Extract video name without extension
        video_name = video_name[-3:]
        first_frame = True
        print(video_name)
        new_row = [video_name]
        counter = 1
        print("ho aggiunto " + str(counter))
        previous_distanceShoulder = None
        previous_distanceHip = None
        previous_distanceElbow = None
        previous_distanceWrist = None
        previous_distanceKnee = None
        previous_distanceAnkle =None

        previous_XrightShoulder = None
        previous_YrightShoulder = None

        previous_XleftShoulder = None
        previous_YleftShoulder = None

        previous_XrightElbow = None
        previous_YrightElbow = None

        previous_XleftElbow = None
        previous_YleftElbow = None

        previous_XrightWrist = None
        previous_YrightWrist = None

        previous_XleftWrist = None
        previous_YleftWrist =  None

        previous_XrightHip =  None
        previous_YrightHip =  None

        previous_XleftHip =  None
        previous_YleftHip = None

        previous_XrightKnee =  None
        previous_YrightKnee =  None

        previous_XleftKnee =  None
        previous_YleftKnee =  None

        previous_XrightAnkle =  None
        previous_YrightAnkle =  None

        previous_XleftAnkle =  None
        previous_YleftAnkle = None






        while cap.isOpened():
            print(first_frame)
            print(video_name)
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % interval == 0:
                # Resize frame to fit the window
                #frame = cv2.resize(frame, (new_width, new_height))

                # Convert the BGR frame to RGB before processing.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_height, image_width, _ = image.shape
                print(image_width, image_height)
                # Process the frame with Mediapipe pose detection.
                results = pose.process(image)

                if not results.pose_landmarks:
                    continue
                #prendo i punti importanti
                XrightShoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
                YrightShoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height

                XleftShoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
                YleftShoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height

                XrightElbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width
                YrightElbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height

                XleftElbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width
                YleftElbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height

                XrightWrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width
                YrightWrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height

                XleftWrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width
                YleftWrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height

                XrightHip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width
                YrightHip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height

                XleftHip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width
                YleftHip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height

                XrightKnee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width
                YrightKnee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height

                XleftKnee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width
                YleftKnee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height

                XrightAnkle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width
                YrightAnkle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_height

                XleftAnkle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width
                YleftAnkle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height

                #calcolo diszanza tra punti per controllo di swap
                distanceShoulder = (XleftShoulder - XrightShoulder)
                distanceElbow = (XleftElbow - XrightElbow)
                distanceWrist = (XleftWrist - XrightWrist)
                distanceHip = (XleftHip - XrightHip)
                distanceKnee = (XleftKnee - XrightKnee)
                distanceAnkle = (XleftAnkle - XrightAnkle)

                #controllo di swap e aggiornamento
                if distanceShoulder < 0:
                    XrightShoulder, XleftShoulder = swap(XrightShoulder, XleftShoulder)
                    YrightShoulder, YleftShoulder = swap(YrightShoulder, YleftShoulder)
                    distanceShoulder = (XleftShoulder - XrightShoulder)


                if distanceElbow < 0:
                    XrightElbow, XleftElbow = swap(XrightElbow, XleftElbow)
                    YrightElbow, YleftElbow = swap(YrightElbow, YleftElbow)
                    distanceElbow = (XleftElbow - XrightElbow)


                if distanceWrist < 0:
                    XrightWrist, XleftWrist = swap(XrightWrist, XleftWrist)
                    YrightWrist, YleftWrist = swap(YrightWrist, YleftWrist)
                    distanceWrist = (XleftWrist - XrightWrist)


                if distanceHip < 0:
                    XrightHip, XleftHip = swap(XrightHip, XleftHip)
                    YrightHip, YleftHip = swap(YrightHip, YleftHip)
                    distanceHip = (XleftHip - XrightHip)


                if distanceKnee < 0:
                    XrightKnee, XleftKnee = swap(XrightKnee, XleftKnee)
                    YrightKnee, YleftKnee = swap(YrightKnee, YleftKnee)
                    distanceKnee = (XleftKnee - XrightKnee)


                if distanceAnkle < 0:
                    XrightAnkle, XleftAnkle = swap(XrightAnkle, XleftAnkle)
                    YrightAnkle, YleftAnkle = swap(YrightAnkle, YleftAnkle)
                    distanceAnkle = (XleftAnkle - XrightAnkle)


                #controllo bug swap
                if not first_frame and previous_distanceShoulder is not None and previous_distanceHip is not None:
                    shoulder_diff = distanceShoulder - previous_distanceShoulder
                    hip_diff = distanceHip - previous_distanceHip
                    if (shoulder_diff < 0 and hip_diff < 0) or (shoulder_diff < -20 or hip_diff < -15):
                        #riprisina a valori precedenti

                        XrightShoulder = previous_XrightShoulder
                        YrightShoulder = previous_YrightShoulder

                        XleftShoulder = previous_XleftShoulder
                        YleftShoulder = previous_YleftShoulder

                        XrightElbow = previous_XrightElbow
                        YrightElbow = previous_YrightElbow

                        XleftElbow = previous_XleftElbow
                        YleftElbow = previous_YleftElbow

                        XrightWrist = previous_XrightWrist
                        YrightWrist = previous_YrightWrist

                        XleftWrist = previous_XleftWrist
                        YleftWrist = previous_YleftWrist

                        XrightHip = previous_XrightHip
                        YrightHip = previous_YrightHip

                        XleftHip = previous_XleftHip
                        YleftHip = previous_YleftHip

                        XrightKnee = previous_XrightKnee
                        YrightKnee = previous_YrightKnee

                        XleftKnee = previous_XleftKnee
                        YleftKnee = previous_YleftKnee

                        XrightAnkle = previous_XrightAnkle
                        YrightAnkle = previous_YrightAnkle

                        XleftAnkle = previous_XleftAnkle
                        YleftAnkle = previous_YleftAnkle

                        distanceShoulder = previous_distanceShoulder
                        distanceHip = previous_distanceHip
                        distanceElbow = previous_distanceElbow
                        distanceWrist = previous_distanceWrist
                        distanceKnee = previous_distanceKnee
                        distanceAnkle = previous_distanceAnkle

                        #fine ripristrino


                #aggiorno i previous per i successivi frame
                previous_distanceShoulder = distanceShoulder
                previous_distanceHip = distanceHip
                previous_distanceElbow = distanceElbow
                previous_distanceWrist = distanceWrist
                previous_distanceKnee = distanceKnee
                previous_distanceAnkle = distanceAnkle


                previous_XrightShoulder = XrightShoulder
                previous_YrightShoulder = YrightShoulder

                previous_XleftShoulder = XleftShoulder
                previous_YleftShoulder = YleftShoulder

                previous_XrightElbow = XrightElbow
                previous_YrightElbow = YrightElbow

                previous_XleftElbow = XleftElbow
                previous_YleftElbow = YleftElbow

                previous_XrightWrist = XrightWrist
                previous_YrightWrist = YrightWrist

                previous_XleftWrist = XleftWrist
                previous_YleftWrist = YleftWrist

                previous_XrightHip = XrightHip
                previous_YrightHip = YrightHip

                previous_XleftHip = XleftHip
                previous_YleftHip = YleftHip

                previous_XrightKnee = XrightKnee
                previous_YrightKnee = YrightKnee

                previous_XleftKnee = XleftKnee
                previous_YleftKnee = YleftKnee

                previous_XrightAnkle = XrightAnkle
                previous_YrightAnkle = YrightAnkle

                previous_XleftAnkle = XleftAnkle
                previous_YleftAnkle = YleftAnkle



                #punti estratti logica feature
                #aggiungere la logica del file csv

                points = {
                    'XrightShoulder': XrightShoulder,
                    'YrightShoulder': YrightShoulder,
                    'XleftShoulder': XleftShoulder,
                    'YleftShoulder': YleftShoulder,
                    'XrightElbow': XrightElbow,
                    'YrightElbow': YrightElbow,
                    'XleftElbow': XleftElbow,
                    'YleftElbow': YleftElbow,
                    'XrightWrist': XrightWrist,
                    'YrightWrist': YrightWrist,
                    'XleftWrist': XleftWrist,
                    'YleftWrist': YleftWrist,
                    'XrightHip': XrightHip,
                    'YrightHip': YrightHip,
                    'XleftHip': XleftHip,
                    'YleftHip': YleftHip,
                    'XrightKnee': XrightKnee,
                    'YrightKnee': YrightKnee,
                    'XleftKnee': XleftKnee,
                    'YleftKnee': YleftKnee,
                    'XrightAnkle': XrightAnkle,
                    'YrightAnkle': YrightAnkle,
                    'XleftAnkle': XleftAnkle,
                    'YleftAnkle': YleftAnkle
                }
                features = featureExtract.calculate_postural_features(points)
                features_normalized = featureExtract.normalize_data(features)
                print(features)
                print(features_normalized)

                for feature in features_normalized:
                    print(feature)
                    new_row.extend([str(feature)])

                counter = counter + len(features_normalized)

                print(
                    f' Shoulder distance: ('
                    f'{distanceShoulder}) LX:{XleftShoulder} LY:{YleftShoulder}       RX:{XrightShoulder} RY:{YrightShoulder}\n'
                    f' Elbow distance: ('
                    f'{distanceElbow}) LX:{XleftElbow} LY:{YleftElbow}       RX:{XrightElbow} RY:{YrightElbow}\n'
                    f' Wrist distance: ('
                    f'{distanceWrist}) LX:{XleftWrist} LY:{YleftWrist}       RX:{XrightWrist} RY:{YrightWrist}\n'
                    f' Hip distance: ('
                    f'{distanceHip}) LX:{XleftHip} LY:{YleftHip}       RX:{XrightHip} RY:{YrightHip}\n'
                    f' Ankle distance: ('
                    f'{distanceAnkle}) LX:{XleftAnkle} LY:{YleftAnkle}       RX:{XrightAnkle} RY:{YrightAnkle}\n'
                )

                print("ho aggiunto " + str(counter))
                cv2.rectangle(frame, (0, 0), (frame_width,frame_height), BG_COLOR, -1)

                # Draw pose landmarks on the frame.
                mp_drawing.draw_landmarks(frame,
                                          results.pose_landmarks,
                                          mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())



                frame_index += 1

            first_frame = False
            frame_num += 1

        print(frame_num)
        print("aggiunto " + str(counter) +"FINE")
        csv_writer.writerow(new_row)

    cap.release()

    cv2.destroyAllWindows()

