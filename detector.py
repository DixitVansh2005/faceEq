"""
detector.py
-----------
Core facial attribute analysis module.
Wraps DeepFace to run age, gender, emotion and race prediction
on a preprocessed numpy image array.

Models used internally by DeepFace:
  - Age/Gender : VGG-Face backbone trained on VGGFace2 dataset
  - Emotion    : Mini-Xception trained on FER-2013 dataset
  - Race       : VGG-16 trained on UTKFace dataset
  - Detector   : RetinaFace (most accurate, default)
"""

import numpy as np
import time


# Which attributes to analyse — change this list to run fewer models
ANALYSIS_ACTIONS = ['age', 'gender', 'emotion', 'race']

# DeepFace face detector backend
# Options: 'retinaface', 'mtcnn', 'opencv', 'mediapipe'
# retinaface = most accurate, slightly slower
DETECTOR_BACKEND = 'retinaface'


def _load_deepface():
    """Lazy import so app starts fast before first analysis."""
    from deepface import DeepFace
    return DeepFace


def analyse(img_array: np.ndarray) -> dict:
    """
    Run full facial attribute analysis on a preprocessed image array.

    Parameters
    ----------
    img_array : np.ndarray
        RGB image as numpy array, shape (H, W, 3), uint8.

    Returns
    -------
    dict with keys:
        age          : int   — estimated age in years
        gender       : str   — 'Man' or 'Woman'
        gender_conf  : float — confidence % for predicted gender
        emotion      : str   — dominant emotion label
        emotions     : dict  — all emotions with % scores
        race         : str   — dominant ethnicity label
        race_scores  : dict  — all ethnicities with % scores
        inference_ms : float — time taken in milliseconds
        face_count   : int   — number of faces detected
    """
    DeepFace = _load_deepface()

    t0 = time.time()

    raw = DeepFace.analyze(
        img_path=img_array,
        actions=ANALYSIS_ACTIONS,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        silent=True
    )

    elapsed_ms = (time.time() - t0) * 1000

    # DeepFace returns a list (one entry per detected face)
    faces = raw if isinstance(raw, list) else [raw]
    face_count = len(faces)

    # Use the first (most prominent) face
    r = faces[0]

    # Normalise gender field — older DeepFace versions return dict, newer return str
    gender_raw = r.get('dominant_gender', r.get('gender', 'Unknown'))
    if isinstance(gender_raw, dict):
        gender_label = max(gender_raw, key=gender_raw.get)
        gender_confidence = gender_raw.get(gender_label, 0.0)
    else:
        gender_label = str(gender_raw)
        gender_dict = r.get('gender', {})
        gender_confidence = gender_dict.get(gender_label, 0.0) if isinstance(gender_dict, dict) else 0.0

    return {
        'age':          int(r.get('age', 0)),
        'gender':       gender_label,
        'gender_conf':  round(float(gender_confidence), 1),
        'emotion':      str(r.get('dominant_emotion', 'unknown')),
        'emotions':     r.get('emotion', {}),
        'race':         str(r.get('dominant_race', 'unknown')),
        'race_scores':  r.get('race', {}),
        'inference_ms': round(elapsed_ms, 1),
        'face_count':   face_count,
    }


def analyse_multi(img_array: np.ndarray) -> list[dict]:
    """
    Run analysis on ALL detected faces in the image.
    Returns a list of result dicts (one per face).
    Useful when multiple people are in the frame.
    """
    DeepFace = _load_deepface()

    raw = DeepFace.analyze(
        img_path=img_array,
        actions=ANALYSIS_ACTIONS,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        silent=True
    )

    faces = raw if isinstance(raw, list) else [raw]
    results = []

    for r in faces:
        gender_raw = r.get('dominant_gender', r.get('gender', 'Unknown'))
        if isinstance(gender_raw, dict):
            gender_label = max(gender_raw, key=gender_raw.get)
            gender_confidence = gender_raw.get(gender_label, 0.0)
        else:
            gender_label = str(gender_raw)
            gender_dict = r.get('gender', {})
            gender_confidence = gender_dict.get(gender_label, 0.0) if isinstance(gender_dict, dict) else 0.0

        results.append({
            'age':         int(r.get('age', 0)),
            'gender':      gender_label,
            'gender_conf': round(float(gender_confidence), 1),
            'emotion':     str(r.get('dominant_emotion', 'unknown')),
            'emotions':    r.get('emotion', {}),
            'race':        str(r.get('dominant_race', 'unknown')),
            'race_scores': r.get('race', {}),
            'region':      r.get('region', {}),
        })

    return results