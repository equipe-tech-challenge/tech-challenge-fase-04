import os
from typing import Dict, Any, List
import cv2
import mediapipe as mp
import numpy as np


class PhysioAnalyzer:
    """
    - mede amplitude (ROM) aproximada do braço (ombro -> punho)
    - mede simetria (diferença entre lados)
    - mede estabilidade (variação frame a frame)
    """

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _angle(a, b, c) -> float:
        """
        Ângulo ABC em graus (b é o ponto central).
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
        if denom == 0:
            return 0.0

        cosang = np.dot(ba, bc) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    def analyze_video(self, video_path: str, job_id: str, fps_sample: int = 5) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_interval = max(int(video_fps / fps_sample), 1)

        storage_root = os.getenv("STORAGE_ROOT", "/app/storage")

        out_dir = os.path.join(storage_root, "results", job_id)
        os.makedirs(out_dir, exist_ok=True)

        output_video_path = os.path.join(out_dir, "physio_detected.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, float(video_fps), (w, h))

        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Não foi possível criar o vídeo de saída: {output_video_path}")

        frame_idx = 0

        left_elbow_angles: List[float] = []
        right_elbow_angles: List[float] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)

            annotated = frame.copy()

            if res.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=res.pose_landmarks,
                    connections=self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                )

            out.write(annotated)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                l_sh = (lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                l_el = (lm[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                        lm[self.mp_pose.PoseLandmark.LEFT_ELBOW].y)
                l_wr = (lm[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                        lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y)

                r_sh = (lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                        lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                r_el = (lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                        lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y)
                r_wr = (lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                        lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].y)

                left_angle = self._angle(l_sh, l_el, l_wr)
                right_angle = self._angle(r_sh, r_el, r_wr)

                left_elbow_angles.append(left_angle)
                right_elbow_angles.append(right_angle)

            frame_idx += 1

        cap.release()
        out.release()

        def safe_stats(arr):
            if not arr:
                return 0.0, 0.0, 0.0
            return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))

        l_mean, l_min, l_max = safe_stats(left_elbow_angles)
        r_mean, r_min, r_max = safe_stats(right_elbow_angles)

        l_rom = max(0.0, l_max - l_min)
        r_rom = max(0.0, r_max - r_min)

        symmetry = 1.0
        if max(l_rom, r_rom) > 0:
            symmetry = 1.0 - (abs(l_rom - r_rom) / max(l_rom, r_rom))
        symmetry = float(np.clip(symmetry, 0.0, 1.0))

        rom_score = float(np.clip((l_rom + r_rom) / 240.0, 0.0, 1.0))
        symmetry_score = symmetry

        execution_quality_score = float(np.clip(0.55 * rom_score + 0.45 * symmetry_score, 0.0, 1.0))

        events = []
        if symmetry_score < 0.55 and (l_rom + r_rom) > 30:
            events.append({
                "type": "movement_asymmetry",
                "score": round(1.0 - symmetry_score, 3),
                "description": "Assimetria de movimento detectada (diferença entre lados).",
            })

        if execution_quality_score < 0.45 and (l_rom + r_rom) > 20:
            events.append({
                "type": "low_execution_quality",
                "score": round(1.0 - execution_quality_score, 3),
                "description": "Qualidade do movimento abaixo do esperado (ROM baixo ou assimetria).",
            })

        return {
            "metrics": {
                "left_elbow_mean": round(l_mean, 3),
                "left_elbow_min": round(l_min, 3),
                "left_elbow_max": round(l_max, 3),
                "right_elbow_mean": round(r_mean, 3),
                "right_elbow_min": round(r_min, 3),
                "right_elbow_max": round(r_max, 3),
                "left_rom": round(l_rom, 3),
                "right_rom": round(r_rom, 3),
            },
            "scores": {
                "rom_score": round(rom_score, 3),
                "symmetry_score": round(symmetry_score, 3),
                "execution_quality_score": round(execution_quality_score, 3),
            },
            "events": events,
        }
