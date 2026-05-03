"""
detector.py
-----------
Core facial attribute analysis module.
"""

import numpy as np
import time
import streamlit as st


ANALYSIS_ACTIONS = ['age', 'gender', 'emotion', 'race']
DETECTOR_BACKEND = 'opencv'  # lightweight, no extra download, fine for most photos


@st.cache_resource(show_spinner="Loading models...")
def _get_deepface():
    """Load DeepFace once and cache in memory across reruns."""
    from deepface import DeepFace
    # Warm up the models so first analysis isn't slow
    import numpy as np
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        DeepFace.analyze(
            img_path=dummy,
            actions=ANALYSIS_ACTIONS,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            silent=True
        )
    except Exception:
        pass
    return DeepFace


def analyse(img_array: np.ndarray) -> dict:
    DeepFace = _get_deepface()

    t0 = time.time()

    raw = DeepFace.analyze(
        img_path=img_array,
        actions=ANALYSIS_ACTIONS,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        silent=True
    )

    elapsed_ms = (time.time() - t0) * 1000

    faces = raw if isinstance(raw, list) else [raw]
    face_count = len(faces)
    r = faces[0]

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
    DeepFace = _get_deepface()

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