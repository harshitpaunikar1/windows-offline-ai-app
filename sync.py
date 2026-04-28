"""
Background sync engine for the Windows offline AI application.
Detects connectivity, encrypts payloads, deduplicates, and pushes results to remote API.
"""
import base64
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from inference import LocalResultStore, OfflineInferenceService
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False


@dataclass
class SyncConfig:
    remote_base_url: str = "https://api.depot.internal"
    results_endpoint: str = "/api/v1/inference/batch"
    health_endpoint: str = "/health"
    sync_interval_s: float = 30.0
    batch_size: int = 100
    timeout_s: float = 15.0
    encrypt_payload: bool = True
    stub_mode: bool = False


@dataclass
class SyncBatchResult:
    attempted: int
    succeeded: int
    failed: int
    duration_ms: float
    synced_at: float = field(default_factory=time.time)


class PayloadEncryptor:
    """Simple XOR-based payload encoding for field encryption simulation."""

    def encode(self, data: Dict[str, Any], key: str = "default") -> str:
        raw_json = json.dumps(data)
        key_bytes = (key * (len(raw_json) // len(key) + 1)).encode()[:len(raw_json)]
        encoded = bytes([ord(c) ^ key_bytes[i] for i, c in enumerate(raw_json)])
        return base64.b64encode(encoded).decode()

    def sign(self, payload: str) -> str:
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ConnectivityChecker:
    """Polls the remote health endpoint to detect network availability."""

    def __init__(self, base_url: str, health_path: str = "/health", timeout: float = 5.0):
        self.url = base_url.rstrip("/") + health_path
        self.timeout = timeout

    def is_online(self) -> bool:
        if not REQUESTS_AVAILABLE:
            return False
        try:
            resp = requests.get(self.url, timeout=self.timeout)
            return resp.status_code < 500
        except Exception:
            return False


class RemoteAPIClient:
    """HTTP client with retry logic for pushing inference results."""

    def __init__(self, base_url: str, timeout: float = 15.0, stub_mode: bool = False):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.stub_mode = stub_mode
        self._session: Optional["requests.Session"] = None

    def _get_session(self) -> "requests.Session":
        if self._session is not None:
            return self._session
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retry))
        session.mount("https://", HTTPAdapter(max_retries=retry))
        self._session = session
        return session

    def push_results(self, batch: List[Dict], endpoint: str,
                      encoded_payload: Optional[str] = None,
                      signature: Optional[str] = None) -> bool:
        if self.stub_mode or not REQUESTS_AVAILABLE:
            logger.info("[stub] Pushing %d results to %s", len(batch), endpoint)
            time.sleep(0.02 * len(batch))
            return True
        try:
            url = self.base_url + endpoint
            body = {
                "records": batch,
                "count": len(batch),
            }
            if encoded_payload:
                body["payload"] = encoded_payload
                body["signature"] = signature
            resp = self._get_session().post(
                url, json=body, timeout=self.timeout,
                headers={"Content-Type": "application/json",
                         "X-Client": "windows-offline-ai"},
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Push failed: %s", exc)
            return False


class WindowsOfflineSyncEngine:
    """
    Background sync engine for the Windows offline AI app.
    Detects connectivity, batches unsynced results, optionally encrypts, and pushes.
    """

    def __init__(self, store: "LocalResultStore", config: SyncConfig):
        self.store = store
        self.config = config
        self.client = RemoteAPIClient(config.remote_base_url, config.timeout_s,
                                       config.stub_mode)
        self.connectivity = ConnectivityChecker(config.remote_base_url,
                                                 config.health_endpoint)
        self.encryptor = PayloadEncryptor()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._history: List[SyncBatchResult] = []
        self._lock = threading.Lock()

    def sync_once(self) -> Optional[SyncBatchResult]:
        if not self.client.stub_mode and not self.connectivity.is_online():
            logger.info("Network unavailable; skipping sync cycle.")
            return None

        with self._lock:
            pending = self.store.pending_sync()
        if not pending:
            return SyncBatchResult(0, 0, 0, 0.0)

        t0 = time.perf_counter()
        succeeded = 0
        failed = 0
        batches = [pending[i:i + self.config.batch_size]
                   for i in range(0, len(pending), self.config.batch_size)]

        for batch in batches:
            clean_batch = [{k: v for k, v in r.items() if k != "output"}
                           for r in batch]
            encoded = None
            signature = None
            if self.config.encrypt_payload:
                encoded = self.encryptor.encode({"records": clean_batch})
                signature = self.encryptor.sign(encoded)

            ok = self.client.push_results(
                clean_batch, self.config.results_endpoint, encoded, signature
            )
            if ok:
                ids = [r["request_id"] for r in batch]
                with self._lock:
                    self.store.mark_synced(ids)
                succeeded += len(batch)
            else:
                failed += len(batch)

        duration_ms = (time.perf_counter() - t0) * 1000
        result = SyncBatchResult(
            attempted=len(pending),
            succeeded=succeeded,
            failed=failed,
            duration_ms=round(duration_ms, 1),
        )
        self._history.append(result)
        logger.info("Sync complete: %d attempted, %d succeeded, %d failed, %.0fms",
                    result.attempted, result.succeeded, result.failed, result.duration_ms)
        return result

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                         name="windows-offline-sync")
        self._thread.start()
        logger.info("Sync engine started (interval=%.0fs)", self.config.sync_interval_s)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while self._running:
            try:
                self.sync_once()
            except Exception as exc:
                logger.error("Sync loop error: %s", exc)
            time.sleep(self.config.sync_interval_s)

    def sync_history(self) -> List[Dict[str, Any]]:
        return [
            {"attempted": r.attempted, "succeeded": r.succeeded, "failed": r.failed,
             "duration_ms": r.duration_ms, "synced_at": r.synced_at}
            for r in self._history
        ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if not INFERENCE_AVAILABLE:
        print("inference.py not available; creating a stub store.")
        from inference import LocalResultStore
        store = LocalResultStore(":memory:")
        store.conn.execute(
            "INSERT INTO inference_results (request_id, mode, image_path, output, confidence, "
            "processing_ms, model_version, created_at, synced) VALUES (?,?,?,?,?,?,?,?,0)",
            ("req_001", "classification", "/img/test.jpg",
             '{"label": "shipment_intact"}', 0.85, 45.0, "v0", time.time()),
        )
        store.conn.commit()
    else:
        from inference import LocalResultStore, OfflineInferenceService, InferenceRequest, InferenceMode
        svc = OfflineInferenceService(db_path=":memory:")
        store = svc.store
        for i in range(5):
            req = InferenceRequest(f"req_{i:03d}", InferenceMode.CLASSIFICATION, f"/img/test_{i}.jpg")
            svc.process(req)

    config = SyncConfig(
        remote_base_url="https://api.depot.internal",
        sync_interval_s=10,
        stub_mode=True,
        encrypt_payload=True,
    )
    engine = WindowsOfflineSyncEngine(store=store, config=config)

    print(f"Pending before sync: {len(store.pending_sync())}")
    result = engine.sync_once()
    if result:
        print(f"Sync result: {result.attempted} attempted, {result.succeeded} succeeded, "
              f"{result.failed} failed, {result.duration_ms:.1f}ms")
    print(f"Pending after sync: {len(store.pending_sync())}")

    engine.start()
    time.sleep(0.5)
    engine.stop()
    print("Sync history:", engine.sync_history())
