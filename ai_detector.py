"""Optional local ONNX Pokémon classifier. No network/API calls are made here."""
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("pokedetector.ai")
MODEL_PATH = Path(os.getenv("AI_MODEL_PATH", "models/pokemon_mobilenet.onnx"))
LABELS_PATH = Path(os.getenv("AI_LABELS_PATH", "models/pokemon_labels.json"))
CONFIDENCE = float(os.getenv("AI_CONFIDENCE", "0.65"))
_session = None
_labels = None
_load_attempted = False


def model_status():
    return {
        "available": MODEL_PATH.is_file() and LABELS_PATH.is_file(),
        "model": str(MODEL_PATH),
        "labels": str(LABELS_PATH),
    }


def _load():
    global _session, _labels, _load_attempted
    if _session is not None:
        return _session, _labels
    if _load_attempted:
        return None, None
    _load_attempted = True
    if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
        return None, None
    try:
        import onnxruntime as ort
        with LABELS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        _labels = data["classes"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = False
        _session = ort.InferenceSession(
            str(MODEL_PATH), sess_options=options,
            providers=["CPUExecutionProvider"]
        )
        log.info("Loaded local AI model with %d Pokémon classes.", len(_labels))
        return _session, _labels
    except Exception:
        log.exception("Could not load local ONNX model.")
        return None, None


def predict_image(raw):
    """Return (Pokémon name, confidence) or None. Call from a worker thread."""
    session, labels = _load()
    if session is None:
        return None
    try:
        import numpy as np
        from PIL import Image
        from io import BytesIO
        with Image.open(BytesIO(raw)) as source:
            image = source.convert("RGB")
            width, height = image.size
            # Spawn subjects are generally near the image center; preserve aspect
            # ratio, then use the same center crop as the training pipeline.
            scale = 256 / min(width, height)
            resized = image.resize((max(256, int(width * scale)),
                                    max(256, int(height * scale))), Image.Resampling.BILINEAR)
            left = (resized.width - 224) // 2
            top = (resized.height - 224) // 2
            image = resized.crop((left, top, left + 224, top + 224))
        array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        array = ((array - mean) / std).transpose(2, 0, 1)[None, ...]
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: array})[0][0]
        logits = logits - np.max(logits)
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        if index >= len(labels) or confidence < CONFIDENCE:
            return None
        return labels[index], confidence
    except Exception:
        log.exception("Local AI prediction failed.")
        return None
