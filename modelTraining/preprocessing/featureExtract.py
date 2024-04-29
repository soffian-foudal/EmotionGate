import numpy as np
import math

def calcola_centro_massa(punti, massa_totale):
    somma_x = 0
    somma_y = 0

    for punto in punti:
        x, y = punto
        somma_x += x
        somma_y += y

    centro_di_massa_x = somma_x / massa_totale
    centro_di_massa_y = somma_y / massa_totale

    return centro_di_massa_x, centro_di_massa_y

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data

def calculate_area_ratio(x1, y1, x2, y2, x3, y3):
    # Calcola l'area del triangolo formato dai punti (x1, y1), (x2, y2), (x3, y3)
    area_triangle = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

    # Calcola l'area del triangolo formato dai punti (x1, y1), (x2, y2), (0, 0)
    area_reference = abs(x1 * y2 - x2 * y1) / 2

    # Calcola il rapporto tra le due aree
    area_ratio = area_triangle / area_reference

    return area_ratio
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Calcola le caratteristiche posturali
def calculate_postural_features(points):
    features = []

    # Calcola la lunghezza dei segmenti corporei
    right_arm_length = euclidean_distance(points['XrightShoulder'], points['YrightShoulder'], points['XrightWrist'],
                                          points['YrightWrist'])
    left_arm_length = euclidean_distance(points['XleftShoulder'], points['YleftShoulder'], points['XleftWrist'],
                                         points['YleftWrist'])
    right_leg_length = euclidean_distance(points['XrightHip'], points['YrightHip'], points['XrightAnkle'],
                                          points['YrightAnkle'])
    left_leg_length = euclidean_distance(points['XleftHip'], points['YleftHip'], points['XleftAnkle'],
                                         points['YleftAnkle'])

    features.extend([right_arm_length, left_arm_length, right_leg_length, left_leg_length])

    # Calcola gli angoli delle articolazioni
    right_shoulder_angle = calculate_angle(points['XrightShoulder'], points['YrightShoulder'], points['XrightElbow'],
                                           points['YrightElbow'], points['XrightWrist'], points['YrightWrist'])
    left_shoulder_angle = calculate_angle(points['XleftShoulder'], points['YleftShoulder'], points['XleftElbow'],
                                          points['YleftElbow'], points['XleftWrist'], points['YleftWrist'])
    right_hip_angle = calculate_angle(points['XrightHip'], points['YrightHip'], points['XrightKnee'],
                                      points['YrightKnee'], points['XrightAnkle'], points['YrightAnkle'])
    left_hip_angle = calculate_angle(points['XleftHip'], points['YleftHip'], points['XleftKnee'], points['YleftKnee'],
                                     points['XleftAnkle'], points['YleftAnkle'])

    features.extend([right_shoulder_angle, left_shoulder_angle, right_hip_angle, left_hip_angle])

    # Aggiungi ulteriori feature posturali
    neck_length = euclidean_distance(points['XrightShoulder'], points['YrightShoulder'], points['XleftShoulder'],
                                     points['YleftShoulder'])
    torso_length = euclidean_distance(points['XrightShoulder'], points['YrightShoulder'], points['XrightHip'],
                                      points['YrightHip'])
    right_leg_to_torso_ratio = right_leg_length / torso_length
    left_leg_to_torso_ratio = left_leg_length / torso_length

    features.extend([neck_length, torso_length, right_leg_to_torso_ratio, left_leg_to_torso_ratio])

    # Calcola il centro di massa
    punti = [
        (points['XrightShoulder'], points['YrightShoulder']),
        (points['XleftShoulder'], points['YleftShoulder']),
        (points['XrightElbow'], points['YrightElbow']),
        (points['XleftElbow'], points['YleftElbow']),
        (points['XrightWrist'], points['YrightWrist']),
        (points['XleftWrist'], points['YleftWrist']),
        (points['XrightHip'], points['YrightHip']),
        (points['XleftHip'], points['YleftHip']),
        (points['XrightKnee'], points['YrightKnee']),
        (points['XleftKnee'], points['YleftKnee']),
        (points['XrightAnkle'], points['YrightAnkle']),
        (points['XleftAnkle'], points['YleftAnkle'])
    ]
    massa_totale = 12  # Esempio di massa totale uguale per tutti i punti
    centro_di_massa_x, centro_di_massa_y = calcola_centro_massa(punti, massa_totale)

    features.extend([centro_di_massa_x, centro_di_massa_y])

    # Aggiungi altre feature desiderate...

    features = calculate_additional_features(features, points)
    return features


# Calcola l'angolo utilizzando tre punti
def calculate_angle(x1, y1, x2, y2, x3, y3):
    v1 = np.array([x1 - x2, y1 - y2])
    v2 = np.array([x3 - x2, y3 - y2])
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


# Esempio di utilizzo
sample_points = {
    'XrightShoulder': 10,
    'YrightShoulder': 20,
    'XleftShoulder': 15,
    'YleftShoulder': 25,
    'XrightElbow': 30,
    'YrightElbow': 40,
    'XleftElbow': 35,
    'YleftElbow': 45,
    'XrightWrist': 50,
    'YrightWrist': 60,
    'XleftWrist': 55,
    'YleftWrist': 65,
    'XrightHip': 70,
    'YrightHip': 80,
    'XleftHip': 75,
    'YleftHip': 85,
    'XrightKnee': 90,
    'YrightKnee': 100,
    'XleftKnee': 95,
    'YleftKnee': 105,
    'XrightAnkle': 110,
    'YrightAnkle': 120,
    'XleftAnkle': 115,
    'YleftAnkle': 125
}




def calculate_additional_features(features, points):


    points['Xneck'] = (points['XleftShoulder'] + points['XrightShoulder']) / 2
    points['Yneck'] = (points['YleftShoulder'] + points['YrightShoulder']) / 2
    points['Xhead'] = points['Xneck']
    points['Yhead'] = points['Yneck']
    points['Xroot'] = (points['XleftAnkle'] + points['XrightAnkle']) / 2
    points['Yroot'] = (points['YleftAnkle'] + points['YrightAnkle']) / 2




    # Calculate angles
    shoulder_lower_back_angle = calculate_angle(points['XleftShoulder'], points['YleftShoulder'],
                                               points['XrightShoulder'], points['YrightShoulder'],
                                               points['XrightHip'], points['YrightHip'])
    left_shoulder_elbow_angle = calculate_angle(points['XleftShoulder'], points['YleftShoulder'],
                                                points['XleftElbow'], points['YleftElbow'],
                                                points['XleftWrist'], points['YleftWrist'])
    right_shoulder_elbow_angle = calculate_angle(points['XrightShoulder'], points['YrightShoulder'],
                                                 points['XrightElbow'], points['YrightElbow'],
                                                 points['XrightWrist'], points['YrightWrist'])

    head_left_knee_root_angle = calculate_angle(points['Xhead'], points['Yhead'],
                                                points['XleftKnee'], points['YleftKnee'],
                                                points['Xroot'], points['Yroot'])
    head_right_knee_root_angle = calculate_angle(points['Xhead'], points['Yhead'],
                                                 points['XrightKnee'], points['YrightKnee'],
                                                 points['Xroot'], points['Yroot'])
    left_ankle_right_ankle_root_angle = calculate_angle(points['XleftAnkle'], points['YleftAnkle'],
                                                        points['XrightAnkle'], points['YrightAnkle'],
                                                        points['Xroot'], points['Yroot'])
    left_hip_ankle_knee_angle = calculate_angle(points['XleftHip'], points['YleftHip'],
                                                points['XleftAnkle'], points['YleftAnkle'],
                                                points['XleftKnee'], points['YleftKnee'])
    right_hip_ankle_knee_angle = calculate_angle(points['XrightHip'], points['YrightHip'],
                                                 points['XrightAnkle'], points['YrightAnkle'],
                                                 points['XrightKnee'], points['YrightKnee'])

    # Calculate distance ratios
    left_wrist_neck_ratio = euclidean_distance(points['XleftWrist'], points['YleftWrist'],
                                               points['Xneck'], points['Yneck']) / euclidean_distance(points['XleftWrist'],
                                                                                                      points['YleftWrist'],
                                                                                                      points['Xroot'],
                                                                                                      points['Yroot'])
    left_wrist_root_ratio = euclidean_distance(points['XleftWrist'], points['YleftWrist'],
                                               points['Xroot'], points['Yroot']) / euclidean_distance(points['XleftWrist'],
                                                                                                    points['YleftWrist'],
                                                                                                    points['Xneck'],
                                                                                                    points['Yneck'])
    right_wrist_neck_ratio = euclidean_distance(points['XrightWrist'], points['YrightWrist'],
                                                points['Xneck'], points['Yneck']) / euclidean_distance(points['XrightWrist'],
                                                                                                       points['YrightWrist'],
                                                                                                       points['Xroot'],
                                                                                                       points['Yroot'])
    right_wrist_root_ratio = euclidean_distance(points['XrightWrist'], points['YrightWrist'],
                                                points['Xroot'], points['Yroot']) / euclidean_distance(points['XrightWrist'],
                                                                                                     points['YrightWrist'],
                                                                                                     points['Xneck'],
                                                                                                     points['Yneck'])
    left_wrist_right_wrist_neck_ratio = euclidean_distance(points['XleftWrist'], points['YleftWrist'],
                                                           points['XrightWrist'], points['YrightWrist']) / euclidean_distance(
        points['XleftWrist'], points['YleftWrist'], points['Xneck'], points['Yneck'])

    left_ankle_right_ankle_neck_ratio = euclidean_distance(points['XleftAnkle'], points['YleftAnkle'],
                                                           points['XrightAnkle'], points['YrightAnkle']) / euclidean_distance(
        points['XleftAnkle'], points['YleftAnkle'], points['Xneck'], points['Yneck'])

    # Calculate area ratios
    shoulders_lower_back_area_ratio = calculate_area_ratio(points['XleftShoulder'], points['YleftShoulder'],
                                                           points['XrightShoulder'], points['YrightShoulder'],
                                                           points['XrightHip'], points['YrightHip'])
    wrists_lower_back_area_ratio = calculate_area_ratio(points['XleftWrist'], points['YleftWrist'],
                                                        points['XrightWrist'], points['YrightWrist'],
                                                        points['XrightHip'], points['YrightHip'])
    wrists_neck_area_ratio = calculate_area_ratio(points['XleftWrist'], points['YleftWrist'],
                                                  points['XrightWrist'], points['YrightWrist'],
                                                  points['Xneck'], points['Yneck'])
    ankles_root_area_ratio = calculate_area_ratio(points['XleftAnkle'], points['YleftAnkle'],
                                                  points['XrightAnkle'], points['YrightAnkle'],
                                                  points['Xroot'], points['Yroot'])

    # Append the new features
    features.extend([shoulder_lower_back_angle, left_shoulder_elbow_angle, right_shoulder_elbow_angle,
                      head_left_knee_root_angle,
                     head_right_knee_root_angle,left_hip_ankle_knee_angle,
                     right_hip_ankle_knee_angle, left_wrist_neck_ratio, left_wrist_root_ratio,
                     right_wrist_neck_ratio, right_wrist_root_ratio, left_wrist_right_wrist_neck_ratio,
                     left_ankle_right_ankle_neck_ratio, shoulders_lower_back_area_ratio, wrists_lower_back_area_ratio,
                     wrists_neck_area_ratio])

    return features
