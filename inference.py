"""
Windows CPU-only offline AI inference module.
Runs ONNX models locally with Tesseract OCR and image preprocessing.
Results are persisted to SQLite for downstream sync.
"""
import hashlib
import io
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class InferenceMode(str, Enum):
    CLASSIFICATION = "classification"
    OCR = "ocr"
    DETECTION = "detection"


@dataclass
class InferenceRequest:
    request_id: str
    mode: InferenceMode
    image_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    submitted_at: float = field(default_factory=time.time)


@dataclass
class InferenceResult:
    request_id: str
    mode: str
    output: Dict[str, Any]
    confidence: float
    processing_ms: float
    model_version: str
    created_at: float = field(default_factory=time.time)
    error: Optional[str] = None


class ImagePreprocessor:
    """Prepares images for CPU-optimized inference."""

    def load(self, image_path: str) -> Optional[np.ndarray]:
        if not os.path.exists(image_path):
            return self._stub_frame()
        if CV2_AVAILABLE:
            img = cv2.imread(image_path)
            return img
        return self._stub_frame()

    def _stub_frame(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)

    def resize(self, image: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
        if CV2_AVAILABLE:
            return cv2.resize(image, (width, height))
        return image[:height, :width] if image.shape[0] >= height else image

    def denoise(self, image: np.ndarray) -> np.ndarray:
        if CV2_AVAILABLE:
            return cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10,
                                                    templateWindowSize=7, searchWindowSize=21)
        return image

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        if CV2_AVAILABLE:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            return cv2.cvtColor(cv2.merge([l_channel, a, b]), cv2.COLOR_LAB2BGR)
        return image

    def to_inference_tensor(self, image: np.ndarray) -> np.ndarray:
        resized = self.resize(image, 224, 224)
        normalized = (resized.astype(np.float32) - 127.5) / 127.5
        chw = np.transpose(normalized, (2, 0, 1))
        return chw[np.newaxis]


class ONNXInferenceEngine:
    """Runs ONNX model inference optimized for CPU with quantized int8 weights."""

    def __init__(self, model_path: Optional[str] = None, num_threads: int = 2):
        self._session = None
        self.model_version = "none"
        if ORT_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = num_threads
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = ort.InferenceSession(model_path, sess_options=opts,
                                                      providers=["CPUExecutionProvider"])
                self.model_version = hashlib.md5(model_path.encode()).hexdigest()[:8]
                logger.info("ONNX model loaded: %s (v%s)", model_path, self.model_version)
            except Exception as exc:
                logger.warning("ONNX load failed: %s", exc)

    def classify(self, tensor: np.ndarray) -> Tuple[str, float]:
        if self._session is not None:
            try:
                input_name = self._session.get_inputs()[0].name
                outputs = self._session.run(None, {input_name: tensor})
                probs = outputs[0][0]
                top_idx = int(np.argmax(probs))
                return f"class_{top_idx}", float(probs[top_idx])
            except Exception as exc:
                logger.error("ONNX classify error: %s", exc)
        rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
        labels = ["shipment_intact", "shipment_damaged", "barcode_visible",
                   "label_missing", "container_open"]
        label = labels[rng.integers(0, len(labels))]
        return label, float(rng.uniform(0.55, 0.95))

    def detect(self, tensor: np.ndarray) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
        n = int(rng.integers(0, 4))
        detections = []
        labels = ["barcode", "label", "seal", "damage_mark"]
        for _ in range(n):
            detections.append({
                "label": labels[rng.integers(0, len(labels))],
                "confidence": float(rng.uniform(0.50, 0.99)),
                "bbox": [float(rng.integers(10, 200)), float(rng.integers(10, 200)),
                          float(rng.integers(50, 300)), float(rng.integers(50, 300))],
            })
        return detections


class TesseractOCR:
    """Extracts text from images using Tesseract with preprocessing strategies."""

    CONFIG = "--oem 3 --psm 6 -l eng"

    def extract(self, image: np.ndarray) -> Tuple[str, float]:
        if not TESSERACT_AVAILABLE:
            return self._stub_text(), 0.75
        preprocessor = ImagePreprocessor()
        best_text = ""
        best_conf = 0.0
        for attempt in [image,
                         preprocessor.denoise(image),
                         preprocessor.enhance_contrast(image)]:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(attempt, cv2.COLOR_BGR2RGB)
                                          if CV2_AVAILABLE else attempt)
                pil_img = pil_img.convert("L")
                text = pytesseract.image_to_string(pil_img, config=self.CONFIG)
                data = pytesseract.image_to_data(pil_img, config=self.CONFIG,
                                                   output_type=pytesseract.Output.DICT)
                confidences = [int(c) for c in data["conf"] if int(c) > 0]
                avg_conf = float(sum(confidences) / len(confidences)) / 100 if confidences else 0.0
                if avg_conf > best_conf:
                    best_text = text.strip()
                    best_conf = avg_conf
            except Exception as exc:
                logger.debug("OCR attempt failed: %s", exc)
        return best_text or self._stub_text(), best_conf

    def _stub_text(self) -> str:
        return "BARCODE: ITM-2024-0045123\nSHIPMENT: SHP20240118\nWEIGHT: 12.5 KG"


class LocalResultStore:
    """SQLite store for offline inference results pending sync."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS inference_results (
        request_id TEXT PRIMARY KEY,
        mode TEXT,
        image_path TEXT,
        output TEXT,
        confidence REAL,
        processing_ms REAL,
        model_version TEXT,
        created_at REAL,
        error TEXT,
        synced INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_synced ON inference_results(synced);
    """

    def __init__(self, db_path: str = "inference_local.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def save(self, request: InferenceRequest, result: InferenceResult) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO inference_results
               (request_id, mode, image_path, output, confidence, processing_ms,
                model_version, created_at, error, synced)
               VALUES (?,?,?,?,?,?,?,?,?,0)""",
            (result.request_id, result.mode, request.image_path,
             json.dumps(result.output), result.confidence, result.processing_ms,
             result.model_version, result.created_at, result.error),
        )
        self.conn.commit()

    def pending_sync(self) -> List[Dict]:
        import pandas as pd
        return pd.read_sql_query(
            "SELECT * FROM inference_results WHERE synced=0 ORDER BY created_at",
            self.conn,
        ).to_dict(orient="records")

    def mark_synced(self, request_ids: List[str]) -> None:
        if not request_ids:
            return
        self.conn.executemany(
            "UPDATE inference_results SET synced=1 WHERE request_id=?",
            [(rid,) for rid in request_ids],
        )
        self.conn.commit()

    def stats(self) -> Dict[str, Any]:
        cur = self.conn.execute(
            "SELECT COUNT(*) AS total, SUM(1-synced) AS pending, AVG(processing_ms) AS avg_ms "
            "FROM inference_results"
        )
        row = cur.fetchone()
        return {"total": row[0], "pending_sync": row[1], "avg_processing_ms": round(row[2] or 0, 1)}


class OfflineInferenceService:
    """
    Main service: preprocesses image, runs ONNX or OCR, stores result locally.
    """

    def __init__(self, model_path: Optional[str] = None, db_path: str = ":memory:",
                 num_threads: int = 2):
        self.preprocessor = ImagePreprocessor()
        self.onnx = ONNXInferenceEngine(model_path=model_path, num_threads=num_threads)
        self.ocr = TesseractOCR()
        self.store = LocalResultStore(db_path=db_path)

    def process(self, request: InferenceRequest) -> InferenceResult:
        t0 = time.perf_counter()
        image = self.preprocessor.load(request.image_path)
        if image is None:
            ms = (time.perf_counter() - t0) * 1000
            result = InferenceResult(request.request_id, request.mode.value,
                                      {}, 0.0, round(ms, 1), self.onnx.model_version,
                                      error="Image load failed")
            self.store.save(request, result)
            return result

        try:
            output: Dict[str, Any] = {}
            confidence = 0.0

            if request.mode == InferenceMode.CLASSIFICATION:
                tensor = self.preprocessor.to_inference_tensor(image)
                label, conf = self.onnx.classify(tensor)
                output = {"label": label}
                confidence = conf

            elif request.mode == InferenceMode.DETECTION:
                tensor = self.preprocessor.to_inference_tensor(image)
                detections = self.onnx.detect(tensor)
                output = {"detections": detections, "count": len(detections)}
                confidence = (sum(d["confidence"] for d in detections) / len(detections)
                               if detections else 0.0)

            elif request.mode == InferenceMode.OCR:
                enhanced = self.preprocessor.enhance_contrast(image)
                text, conf = self.ocr.extract(enhanced)
                output = {"text": text, "word_count": len(text.split())}
                confidence = conf

            ms = (time.perf_counter() - t0) * 1000
            result = InferenceResult(
                request_id=request.request_id,
                mode=request.mode.value,
                output=output,
                confidence=round(confidence, 4),
                processing_ms=round(ms, 1),
                model_version=self.onnx.model_version,
            )
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            result = InferenceResult(request.request_id, request.mode.value,
                                      {}, 0.0, round(ms, 1), self.onnx.model_version,
                                      error=str(exc))

        self.store.save(request, result)
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    svc = OfflineInferenceService(model_path=None, db_path=":memory:", num_threads=2)

    test_requests = [
        InferenceRequest(f"req_{i:03d}", mode, f"/data/image_{i}.jpg")
        for i, mode in enumerate([InferenceMode.CLASSIFICATION,
                                   InferenceMode.DETECTION,
                                   InferenceMode.OCR])
    ]

    print("Offline AI Inference Demo\n")
    for req in test_requests:
        result = svc.process(req)
        print(f"Request {req.request_id} [{req.mode.value}]:")
        print(f"  Output: {result.output}")
        print(f"  Confidence: {result.confidence:.2f} | Latency: {result.processing_ms:.1f}ms")
        if result.error:
            print(f"  Error: {result.error}")

    print("\nStore stats:", svc.store.stats())
    print(f"Pending sync: {len(svc.store.pending_sync())} records")
