import cv2
from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm
import face_recognition
import mediapipe as mp

# Constantes
SCALE_FACTOR = 0.25
SCALE_INVERSE = int(1 / SCALE_FACTOR)
IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")
DEFAULT_LABEL = "Desconhecido"
UNKNOWN_EMOTION = "unknown"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_COLOR = (36, 255, 12)
FONT_THICKNESS = 2
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
TEXT_MARGIN_TOP = 10
TEXT_MARGIN_BOTTOM = 25
TEXT_MIN_Y = 20
ROTATIONS = [
    (None, None),
    (cv2.ROTATE_90_CLOCKWISE, 90),
]

# Anomalia: magnitude acima deste limiar é considerada anômala
ANOMALY_THRESHOLD = 8.0
ANOMALY_COLOR = (0, 0, 255)
ANOMALY_LABEL_POSITION = (10, 30)

# Atividades
ACTIVITY_LABEL_COLOR = (255, 255, 0)
ACTIVITY_LABEL_POSITION = (10, 60)

# Índices dos landmarks do MediaPipe Pose
LM_NOSE = 0
LM_LEFT_EYE = 2
LM_RIGHT_EYE = 5
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24

# Conexões do esqueleto para desenho manual
POSE_CONNECTIONS = [
    (LM_NOSE, LM_LEFT_EYE), (LM_NOSE, LM_RIGHT_EYE),
    (LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER),
    (LM_LEFT_SHOULDER, LM_LEFT_ELBOW), (LM_LEFT_ELBOW, LM_LEFT_WRIST),
    (LM_RIGHT_SHOULDER, LM_RIGHT_ELBOW), (LM_RIGHT_ELBOW, LM_RIGHT_WRIST),
    (LM_LEFT_SHOULDER, LM_LEFT_HIP), (LM_RIGHT_SHOULDER, LM_RIGHT_HIP),
    (LM_LEFT_HIP, LM_RIGHT_HIP),
]


def load_known_faces(folder):
    """Carrega encodings e nomes a partir de imagens de referência."""
    encodings = []
    names = []

    for filename in os.listdir(folder):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(folder, filename)
        image = face_recognition.load_image_file(image_path)
        face_encs = face_recognition.face_encodings(image)

        if face_encs:
            encodings.append(face_encs[0])
            name = os.path.splitext(filename)[0][:-1]
            names.append(name)

    return encodings, names


def _convert_rotated_coords(loc, angle, frame_h):
    """Converte coordenadas de um frame rotacionado para o sistema original."""
    top, right, bottom, left = loc

    if angle == 90:
        return (left, frame_h - top, right, frame_h - bottom)

    return loc


def _is_duplicate(loc, existing_locations):
    """Verifica se uma detecção é duplicata de outra já existente."""
    top, right, bottom, left = loc
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    half_w = (right - left) // 2
    half_h = (bottom - top) // 2

    for f_top, f_right, f_bottom, f_left in existing_locations:
        f_cx = (f_left + f_right) // 2
        f_cy = (f_top + f_bottom) // 2
        if abs(cx - f_cx) < half_w and abs(cy - f_cy) < half_h:
            return True

    return False


def detect_faces_with_rotations(rgb_frame):
    """Detecta faces no frame original e rotacionado em 90°.
    Retorna face_locations e face_encodings no sistema de coordenadas original."""
    h = rgb_frame.shape[0]
    all_locations = []
    all_encodings = []

    for rotate_code, angle in ROTATIONS:
        rotated = rgb_frame if rotate_code is None else cv2.rotate(rgb_frame, rotate_code)

        locations = face_recognition.face_locations(rotated)
        encodings = face_recognition.face_encodings(rotated, locations)

        for loc, enc in zip(locations, encodings):
            orig_loc = _convert_rotated_coords(loc, angle, h)
            all_locations.append(orig_loc)
            all_encodings.append(enc)

    filtered_locations = []
    filtered_encodings = []
    for loc, enc in zip(all_locations, all_encodings):
        if not _is_duplicate(loc, filtered_locations):
            filtered_locations.append(loc)
            filtered_encodings.append(enc)

    return filtered_locations, filtered_encodings


def identify_faces(face_encodings_list, known_encodings, known_names):
    """Identifica cada face comparando com os encodings conhecidos."""
    names = []
    for face_encoding in face_encodings_list:
        name = DEFAULT_LABEL
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_index = np.argmin(distances)
            if matches[best_index]:
                name = known_names[best_index]
        names.append(name)
    return names


def analyze_emotion(face_crop):
    """Analisa a emoção dominante de um recorte facial."""
    try:
        result = DeepFace.analyze(
            face_crop, actions=['emotion'],
            enforce_detection=False, detector_backend='skip'
        )
        return result[0]['dominant_emotion']
    except Exception:
        return UNKNOWN_EMOTION


def safe_crop(frame, top, right, bottom, left):
    """Recorta uma região do frame garantindo coordenadas válidas."""
    h, w = frame.shape[:2]
    top = max(0, min(top, h))
    bottom = max(0, min(bottom, h))
    left = max(0, min(left, w))
    right = max(0, min(right, w))

    if top >= bottom or left >= right:
        return None

    return frame[top:bottom, left:right]


def draw_face_label(frame, top, right, bottom, left, label):
    """Desenha o retângulo e o label sobre a face no frame."""
    cv2.rectangle(frame, (left, top), (right, bottom), BOX_COLOR, BOX_THICKNESS)

    text_y = top - TEXT_MARGIN_TOP if top - TEXT_MARGIN_TOP > TEXT_MIN_Y else bottom + TEXT_MARGIN_BOTTOM
    text_x = max(left, 0)
    cv2.putText(frame, label, (text_x, text_y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)


def detect_anomaly(prev_gray, curr_gray):
    """Detecta anomalia comparando dois frames via optical flow.
    Retorna (is_anomaly, magnitude)."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_magnitude = np.mean(magnitude)
    return mean_magnitude >= ANOMALY_THRESHOLD, mean_magnitude


def draw_anomaly_alert(frame):
    """Desenha alerta de anomalia no frame."""
    cv2.putText(
        frame, "ANOMALIA DETECTADA",
        ANOMALY_LABEL_POSITION, FONT, FONT_SCALE,
        ANOMALY_COLOR, FONT_THICKNESS
    )


# --- Detecção de Atividades via MediaPipe Pose ---

def create_pose_landmarker(model_path):
    """Cria o PoseLandmarker usando a nova API mp.tasks."""
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def detect_pose(landmarker, frame):
    """Detecta pose no frame usando PoseLandmarker.
    Retorna a lista de landmarks ou None."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        return result.pose_landmarks[0]

    return None


def classify_activity(landmarks):
    """Classifica a atividade com base nos landmarks do MediaPipe Pose.
    Retorna uma lista de atividades detectadas no frame."""
    activities = []

    left_eye = landmarks[LM_LEFT_EYE]
    right_eye = landmarks[LM_RIGHT_EYE]
    left_shoulder = landmarks[LM_LEFT_SHOULDER]
    right_shoulder = landmarks[LM_RIGHT_SHOULDER]
    left_elbow = landmarks[LM_LEFT_ELBOW]
    right_elbow = landmarks[LM_RIGHT_ELBOW]
    left_wrist = landmarks[LM_LEFT_WRIST]
    right_wrist = landmarks[LM_RIGHT_WRIST]
    left_hip = landmarks[LM_LEFT_HIP]
    right_hip = landmarks[LM_RIGHT_HIP]
    nose = landmarks[LM_NOSE]

    # Braço levantado: cotovelo acima dos olhos
    if left_elbow.y < left_eye.y or right_elbow.y < right_eye.y:
        activities.append("Braco levantado")

    # Mão no rosto: pulso próximo ao nariz
    nose_to_left_wrist = abs(left_wrist.x - nose.x) + abs(left_wrist.y - nose.y)
    nose_to_right_wrist = abs(right_wrist.x - nose.x) + abs(right_wrist.y - nose.y)
    if nose_to_left_wrist < 0.15 or nose_to_right_wrist < 0.15:
        activities.append("Mao no rosto")

    # Inclinação do corpo: ombros desalinhados verticalmente
    if abs(left_shoulder.y - right_shoulder.y) > 0.08:
        activities.append("Corpo inclinado")

    # Sentado vs em pé: relação quadril-ombro
    avg_hip_y = (left_hip.y + right_hip.y) / 2
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    torso_height = avg_hip_y - avg_shoulder_y
    if torso_height < 0.2:
        activities.append("Sentado")
    else:
        activities.append("Em pe")

    # Braços cruzados: pulsos próximos aos ombros opostos
    left_to_right = abs(left_wrist.x - right_shoulder.x) + abs(left_wrist.y - right_shoulder.y)
    right_to_left = abs(right_wrist.x - left_shoulder.x) + abs(right_wrist.y - left_shoulder.y)
    if left_to_right < 0.15 and right_to_left < 0.15:
        activities.append("Bracos cruzados")

    if not activities:
        activities.append("Parado")

    return activities


def draw_pose_on_frame(frame, landmarks):
    """Desenha os landmarks e conexões da pose no frame."""
    h, w = frame.shape[:2]

    # Desenhar conexões
    for start_idx, end_idx in POSE_CONNECTIONS:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        pt1 = (int(start.x * w), int(start.y * h))
        pt2 = (int(end.x * w), int(end.y * h))
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

    # Desenhar pontos
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)


def draw_activity_info(frame, activities):
    """Desenha as atividades detectadas no frame."""
    activity_text = "Atividade: " + ", ".join(activities)
    cv2.putText(
        frame, activity_text,
        ACTIVITY_LABEL_POSITION, FONT, 0.7,
        ACTIVITY_LABEL_COLOR, FONT_THICKNESS
    )


# --- Funções de I/O ---

def open_video(input_path):
    """Abre o vídeo e retorna o capture junto com suas propriedades."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {input_path}")

    props = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, props


def save_report(registros_emocoes, registros_atividades, anomalias, total_frames, frames_analisados, output_video_path):
    """Salva o relatório completo com emoções, atividades e anomalias."""
    txt_path = os.path.splitext(output_video_path)[0] + '_relatorio.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("RELATÓRIO DE ANÁLISE DE VÍDEO\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total de frames no vídeo: {total_frames}\n")
        f.write(f"Frames analisados: {frames_analisados}\n")
        f.write(f"Anomalias detectadas: {len(anomalias)}\n\n")

        # Resumo de emoções
        f.write("-" * 40 + "\n")
        f.write("RESUMO DE EMOÇÕES\n")
        f.write("-" * 40 + "\n")
        contagem_emocoes = {}
        for registro in registros_emocoes:
            partes = registro.split(" - ")
            if len(partes) == 2:
                emotion = partes[1]
                contagem_emocoes[emotion] = contagem_emocoes.get(emotion, 0) + 1
        for emotion, count in sorted(contagem_emocoes.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {emotion}: {count} ocorrências\n")

        # Resumo de atividades
        f.write("\n" + "-" * 40 + "\n")
        f.write("RESUMO DE ATIVIDADES\n")
        f.write("-" * 40 + "\n")
        contagem_atividades = {}
        for atividades_frame in registros_atividades:
            for atividade in atividades_frame:
                contagem_atividades[atividade] = contagem_atividades.get(atividade, 0) + 1
        for atividade, count in sorted(contagem_atividades.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {atividade}: {count} frames\n")

        # Detalhes das anomalias
        if anomalias:
            f.write("\n" + "-" * 40 + "\n")
            f.write("DETALHES DAS ANOMALIAS\n")
            f.write("-" * 40 + "\n")
            for frame_num, magnitude in anomalias:
                f.write(f"  Frame {frame_num}: magnitude {magnitude:.2f}\n")

        # Log completo de emoções
        f.write("\n" + "-" * 40 + "\n")
        f.write("LOG DE EMOÇÕES POR FRAME\n")
        f.write("-" * 40 + "\n")
        for registro in registros_emocoes:
            f.write(f"  {registro}\n")

    print(f"\nRelatório salvo em: {txt_path}", flush=True)


# --- Processamento de Frame ---

def process_frame(frame, known_encodings, known_names, activities=None):
    """Processa um frame: detecta faces, identifica e analisa emoções.
    Retorna a lista de registros e desenha as anotações no frame."""
    small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    rgb_small = np.ascontiguousarray(small_frame[:, :, ::-1])

    face_locations_small, face_encodings_list = detect_faces_with_rotations(rgb_small)

    face_locations = [
        (t * SCALE_INVERSE, r * SCALE_INVERSE, b * SCALE_INVERSE, l * SCALE_INVERSE)
        for (t, r, b, l) in face_locations_small
    ]

    face_names = identify_faces(face_encodings_list, known_encodings, known_names)

    registros = []
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        face_crop = safe_crop(frame, top, right, bottom, left)
        emotion = analyze_emotion(face_crop) if face_crop is not None else UNKNOWN_EMOTION

        activity_str = ", ".join(activities) if activities else "N/A"
        tqdm.write(f"Detectado: {name} | emoção: {emotion} | atividade: {activity_str}")
        registros.append(f"{name} - {emotion} - {activity_str}")

        label = f"{name} | {emotion}"
        draw_face_label(frame, top, right, bottom, left, label)

    return registros


def main():
    print("Iniciando detecção de expressões com reconhecimento facial...", flush=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir, 'input/input_video.mp4')
    output_video_path = os.path.join(script_dir, 'output/output_video.mp4')
    image_folder = os.path.join(script_dir, 'images')
    model_path = os.path.join(script_dir, 'models/pose_landmarker_lite.task')

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    known_encodings, known_names = load_known_faces(image_folder)
    cap, props = open_video(input_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, props['fps'], (props['width'], props['height']))

    # Inicializar MediaPipe PoseLandmarker
    landmarker = create_pose_landmarker(model_path)

    registros_emocoes = []
    registros_atividades = []
    anomalias = []
    prev_gray = None
    frames_analisados = 0

    for frame_num in tqdm(range(props['total_frames']), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        frames_analisados += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecção de anomalia via optical flow
        if prev_gray is not None:
            anomaly_detected, flow_magnitude = detect_anomaly(prev_gray, curr_gray)
            if anomaly_detected:
                anomalias.append((frame_num, flow_magnitude))
                draw_anomaly_alert(frame)

        prev_gray = curr_gray

        # Detecção de atividades via MediaPipe Pose
        pose_landmarks = detect_pose(landmarker, frame)

        activities = ["Nenhuma pose detectada"]
        if pose_landmarks:
            draw_pose_on_frame(frame, pose_landmarks)
            activities = classify_activity(pose_landmarks)

        registros_atividades.append(activities)
        draw_activity_info(frame, activities)

        # Detecção facial e análise de emoções
        registros = process_frame(frame, known_encodings, known_names, activities)
        registros_emocoes.extend(registros)
        out.write(frame)

    landmarker.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    save_report(
        registros_emocoes, registros_atividades, anomalias,
        props['total_frames'], frames_analisados, output_video_path
    )


if __name__ == '__main__':
    main()
