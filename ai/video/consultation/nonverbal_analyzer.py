import os
from typing import Dict, Any, List
import cv2
import mediapipe as mp
import numpy as np


class NonVerbalAnalyzer:
    """
    - mede agitação (movimento dos pulsos/mãos)
    - mede postura fechada (distância ombro-ombro + posição braços)
    - mede evitamento (cabeça virada / inclinação)
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
    def _dist(a, b) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def analyze_video(
        self,
        video_path: str,
        job_id: str,
        fps_sample: int = 5,
        min_seconds_for_event: float = 4.0,
    ) -> Dict[str, Any]:
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

        output_video_path = os.path.join(out_dir, "nonverbal_detected.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, float(video_fps), (w, h))

        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Não foi possível criar o vídeo de saída: {output_video_path}")

        frame_idx = 0

        wrist_motion_series: List[float] = []
        shoulder_width_series: List[float] = []
        head_turn_series: List[float] = []

        prev_wrists = None

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

                left_wrist = (lm[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                              lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y)
                right_wrist = (lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                               lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].y)

                left_shoulder = (lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                 lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                right_shoulder = (lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                  lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y)

                nose = (lm[self.mp_pose.PoseLandmark.NOSE].x,
                        lm[self.mp_pose.PoseLandmark.NOSE].y)

                shoulder_width = self._dist(left_shoulder, right_shoulder)
                shoulder_width_series.append(shoulder_width)

                shoulders_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
                head_turn = abs(nose[0] - shoulders_center_x)
                head_turn_series.append(head_turn)

                if prev_wrists is not None:
                    lw_prev, rw_prev = prev_wrists
                    motion = self._dist(left_wrist, lw_prev) + self._dist(right_wrist, rw_prev)
                    wrist_motion_series.append(motion)
                prev_wrists = (left_wrist, right_wrist)

            frame_idx += 1

        cap.release()
        out.release()

        agitation = float(np.mean(wrist_motion_series)) if wrist_motion_series else 0.0
        agitation_max = float(np.max(wrist_motion_series)) if wrist_motion_series else 0.0

        shoulder_mean = float(np.mean(shoulder_width_series)) if shoulder_width_series else 0.0
        shoulder_min = float(np.min(shoulder_width_series)) if shoulder_width_series else 0.0

        head_turn_mean = float(np.mean(head_turn_series)) if head_turn_series else 0.0
        head_turn_max = float(np.max(head_turn_series)) if head_turn_series else 0.0

        agitation_score = min(1.0, agitation / 0.08) if agitation > 0 else 0.0
        avoidance_score = min(1.0, head_turn_mean / 0.08) if head_turn_mean > 0 else 0.0

        closed_posture_score = 0.0
        if shoulder_mean > 0:
            ratio = shoulder_min / shoulder_mean
            closed_posture_score = min(1.0, max(0.0, (0.85 - ratio) / 0.25))

        nonverbal_discomfort_score = min(
            1.0,
            0.45 * agitation_score + 0.35 * avoidance_score + 0.20 * closed_posture_score
        )

        events = []
        if nonverbal_discomfort_score >= 0.65:
            events.append({
                "type": "nonverbal_discomfort_high",
                "score": round(nonverbal_discomfort_score, 3),
                "description": "Indicadores não-verbais elevados (agitação/evitamento/postura). Revisão recomendada.",
            })

        if agitation_score >= 0.75:
            events.append({
                "type": "high_fidgeting",
                "score": round(agitation_score, 3),
                "description": "Agitação (movimento de mãos/pulsos) elevada.",
            })

        if avoidance_score >= 0.75:
            events.append({
                "type": "high_avoidance",
                "score": round(avoidance_score, 3),
                "description": "Possível evitamento (cabeça/olhar desviando com frequência).",
            })

        return {
            "metrics": {
                "agitation_mean": round(agitation, 6),
                "agitation_max": round(agitation_max, 6),
                "shoulder_mean": round(shoulder_mean, 6),
                "shoulder_min": round(shoulder_min, 6),
                "head_turn_mean": round(head_turn_mean, 6),
                "head_turn_max": round(head_turn_max, 6),
            },
            "scores": {
                "agitation_score": round(agitation_score, 3),
                "avoidance_score": round(avoidance_score, 3),
                "closed_posture_score": round(closed_posture_score, 3),
                "nonverbal_discomfort_score": round(nonverbal_discomfort_score, 3),
            },
            "events": events,
        }
