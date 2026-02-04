import cv2
from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm
import face_recognition


def load_images_from_folder(folder):
    known_face_encodings = []
    known_face_names = []

    # Percorrer todos os arquivos na pasta fornecida
    for filename in os.listdir(folder):
        # Verificar se o arquivo é uma imagem
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Carregar a imagem
            image_path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(image_path)
            # Obter as codificações faciais (assumindo uma face por imagem)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                face_encoding = face_encodings[0]
                # Extrair o nome do arquivo, removendo o sufixo numérico e a extensão
                name = os.path.splitext(filename)[0][:-1]
                # Adicionar a codificação e o nome às listas
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

    return known_face_encodings, known_face_names


def match_deepface_to_name(deepface_region, face_locations, face_names):
    """Associa uma região do DeepFace ao nome mais próximo do face_recognition.
    face_locations usa coordenadas já escaladas para o tamanho original."""
    x, y, w, h = deepface_region['x'], deepface_region['y'], deepface_region['w'], deepface_region['h']
    # Centro da face detectada pelo DeepFace
    cx_deep = x + w // 2
    cy_deep = y + h // 2

    best_name = "Desconhecido"
    best_dist = float('inf')

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Centro da face detectada pelo face_recognition
        cx_fr = (left + right) // 2
        cy_fr = (top + bottom) // 2
        dist = abs(cx_deep - cx_fr) + abs(cy_deep - cy_fr)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name


def main():
    print("Iniciando detecção de expressões com reconhecimento facial...", flush=True)
    # Caminho para o arquivo de vídeo na mesma pasta do script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir, 'input/input_video.mp4')
    output_video_path = os.path.join(script_dir, 'output/output_video.mp4')

    image_folder = os.path.join(script_dir, 'images')  # Caminho para a pasta de imagens
    known_face_encodings, known_face_names = load_images_from_folder(image_folder)

    # Vetor para armazenar nome + emoção
    registros_emocoes = []

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(input_video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Loop para processar cada frame do vídeo
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        # Ler um frame do vídeo
        ret, frame = cap.read()

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        # Redimensionar o frame para 1/2 para acelerar o reconhecimento facial
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])  # Converter BGR para RGB

        # Localizar faces e obter encodings no frame reduzido
        face_locations_small = face_recognition.face_locations(rgb_small_frame)
        face_encodings_list = face_recognition.face_encodings(rgb_small_frame, face_locations_small)

        # Escalar as coordenadas de volta para o tamanho original (x2)
        face_locations = [(top * 2, right * 2, bottom * 2, left * 2)
                          for (top, right, bottom, left) in face_locations_small]

        # Identificar cada face reconhecida
        face_names = []
        for face_encoding in face_encodings_list:
            name = "Desconhecido"
            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            face_names.append(name)

        # Iterar sobre cada face detectada pelo face_recognition
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Recortar a face do frame original
            face_crop = frame[top:bottom, left:right]

            # Analisar a emoção apenas do rosto recortado (skip = não re-detectar)
            dominant_emotion = "unknown"
            try:
                result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, detector_backend='skip')
                dominant_emotion = result[0]['dominant_emotion']
            except Exception:
                pass

            tqdm.write(f"Detectado: {name} com emoção {dominant_emotion}")
            # Adicionar ao vetor de registros
            registros_emocoes.append(f"{name} - {dominant_emotion}")

            # Desenhar um retângulo ao redor da face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Escrever nome + emoção acima da face
            full_identification = f"{name} | {dominant_emotion}"
            text_y = top - 10 if top - 10 > 20 else bottom + 25
            text_x = max(left, 0)
            cv2.putText(frame, full_identification, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

    # Salvar registros de emoções em um arquivo .txt
    txt_path = os.path.splitext(output_video_path)[0] + '_emocoes.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        for registro in registros_emocoes:
            f.write(registro + '\n')
    print(f"\nEmoções salvas em: {txt_path}", flush=True)

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

