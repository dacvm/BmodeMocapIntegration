
"""Live-preview widget that streams from the selected USB camera into the Qt UI."""

# Standard library helpers for launching a Qt application and accepting CLI args.
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
# Optional typing helper preserved for future extensions or annotations.
from typing import Optional

# Third-party libs for video capture, image encoding, and Qt widgets/layouts.
import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPixmap, QScreen, QTextCursor
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QWidget

# Generated UI classes that wrap the Qt Designer forms into Python-friendly fields.
from ui.bmode_ui import Ui_Form as Ui_BModeV2
# Class to parse the Calibration XML.
from helpers.xml_stream_parser import (
    CoordinateDefinitionsExtractor,
    VideoDeviceExtractor,
    XmlStreamParser,
)

IMAGE_RECORD_QUEUE_MAXSIZE = 300
JPEG_QUALITY = 85
IMAGE_RECORD_DROP_WINDOW_SECONDS = 2.0
IMAGE_RECORD_DROP_RATE_THRESHOLD = 0.05
CAMERA_STOP_JOIN_TIMEOUT_SEC = 0.5
CAMERA_STOP_RETRY_JOIN_TIMEOUT_SEC = 0.5
CAMERA_REOPEN_COOLDOWN_SEC = 0.15
CAMERA_STARTUP_MAX_RETRIES = 2
CAMERA_STARTUP_RETRY_DELAY_SEC = 0.20
CAMERA_STARTUP_WARMUP_TIMEOUT_SEC = 0.80
CAMERA_STARTUP_MAX_WARMUP_FRAMES = 24

logger = logging.getLogger(__name__)


# Summary:
# - Build a monotonic timestamp in milliseconds for frame packets.
# - What it does: Uses `time.monotonic()` so image timestamps share the same stable
#   clock source as mocap timestamps for software coupling.
# - Input: None.
# - Returns: Monotonic milliseconds (int).
def _now_ms() -> int:
    # Use a monotonic clock so timestamp deltas are stable if system time changes.
    return int(time.monotonic() * 1000)


# Summary:
# - Lightweight container for one streamed frame and its metadata.
# - What it does: Stores timestamp, dimensions, format label, and raw bytes for UI rendering/recording.
@dataclass
class FramePacket:
    timestamp_ms: int
    width: int
    height: int
    fmt: str
    data: bytes


# Summary:
# - Qt signal bridge used to safely move frames/state from worker loops into the Qt UI thread.
# - What it does: Exposes signals for frame data, running state, and error messages.
class BModeStreamProxy(QObject):
    frame_ready = Signal(object)
    state_changed = Signal(bool, str)
    error_message = Signal(str)
    record_message = Signal(str)
    record_stop = Signal(str)

    # Summary:
    # - Initialize the proxy and assign a stable objectName for slot naming consistency.
    # - Input: `self`.
    # - Returns: None.
    def __init__(self) -> None:
        super().__init__()
        # Give the proxy a name so slot handlers can follow the naming convention.
        self.setObjectName("bmodeStreamProxy")


# Summary:
# - Background worker that owns the camera capture loop.
# - What it does: Opens the camera on a thread, reads frames, and emits FramePacket via the proxy.
class CameraStreamWorker:
    # Summary:
    # - Initialize the worker and its thread state.
    # - Input: `self`, `proxy` (BModeStreamProxy).
    # - Returns: None.
    def __init__(self, proxy: BModeStreamProxy) -> None:
        # Store the proxy so background threads can emit UI-safe signals.
        self._proxy = proxy

        # Track thread state so start/stop are repeat-safe.
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._lock = threading.Lock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._active = False

        # Track a monotonic session id so stale threads can never update the current stream state.
        # Some camera drivers can keep old reads "alive" briefly, and their late signals can otherwise
        # flip the UI back to idle/blank even though a new stream already started.
        self._session_id = 0

        # Track the last stop time so reopen attempts can wait briefly for USB drivers to settle.
        # Many USB frame grabbers behave better if we wait a short moment before reopening.
        self._last_stop_ts = 0.0

    # Summary:
    # - Start the camera acquisition thread.
    # - Input: `self`, `camera_index` (int), `fps` (int; used as ms interval),
    #   `clip_rect` (tuple[int, int, int, int] | None) for optional calibrated cropping.
    # - Returns: None.
    def start(
        self,
        camera_index: int,
        fps: int = 33,
        clip_rect: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        # Guard against double-starts to keep the worker repeat-safe.
        with self._lock:
            if self._active:
                return
            self._active = True

            # Create a unique session marker so we can ignore late events from older threads.
            # We increment it here (inside the lock) so the worker thread always knows which session
            # it belongs to, and so stop() can invalidate the session immediately.
            self._session_id += 1
            session_id = self._session_id

            # Create a fresh stop event for this streaming session.
            self._stop_event = threading.Event()

            # Start the background thread so camera I/O never blocks the UI thread.
            self._thread = threading.Thread(
                target=self._run,
                args=(
                    int(session_id),
                    int(camera_index),
                    int(fps),
                    self._stop_event,
                    clip_rect,
                ),
                daemon=True,
            )
            # Thread start: kick off the worker loop.
            self._thread.start()

    # Summary:
    # - Stop the active camera thread without blocking the UI for long.
    # - Input: `self`.
    # - Returns: None.
    def stop(self) -> None:
        # Lock to avoid racing against start.
        with self._lock:
            if not self._active and self._thread is None:
                return
            self._active = False
            
            # Invalidate the current session so stale thread callbacks are ignored immediately.
            # This protects the UI state from older threads that might still be emitting signals.
            self._session_id += 1

            stop_event = self._stop_event
            worker_thread = self._thread

            # Clear references so a new start can rebuild cleanly.
            self._stop_event = None
            self._thread = None
            # Track stop time so the next open can respect a short cooldown.
            self._last_stop_ts = time.monotonic()

        # Signal the worker loop to exit.
        if stop_event is not None:
            stop_event.set()

        # Join briefly so stop does not freeze the UI.
        if worker_thread is not None:
            worker_thread.join(timeout=CAMERA_STOP_JOIN_TIMEOUT_SEC)

            # Some camera drivers block inside read(); force-release can help the loop unwind.
            # If the thread is still alive after a short join, we try releasing the capture handle.
            # This can "unstick" read() so the worker thread can exit promptly.
            if worker_thread.is_alive():
                cap = self._cap
                if cap is not None:
                    try:
                        cap.release()
                        logger.debug(
                            "Forced VideoCapture release during stop to unblock camera thread."
                        )
                    except Exception:
                        logger.debug(
                            "Forced VideoCapture release failed during stop.",
                            exc_info=True,
                        )
                worker_thread.join(timeout=CAMERA_STOP_RETRY_JOIN_TIMEOUT_SEC)

    # Summary:
    # - Emit a proxy signal safely when the UI may have already been destroyed.
    # - Input: `self`, `emit_fn` (callable), `args` (tuple[object, ...]).
    # - Returns: None.
    def _safe_emit(self, emit_fn, *args) -> None:
        # Avoid crashing if Qt deletes the proxy while the thread is still running.
        try:
            emit_fn(*args)
        except RuntimeError:
            # Ignore emits after shutdown is already in progress.
            return

    # Summary:
    # - Check whether a worker callback still belongs to the active camera stream session.
    # - Input: `self`, `session_id` (int) for the worker thread emitting events.
    # - Returns: `True` when the session is still current, otherwise `False`.
    def _is_current_session(self, session_id: int) -> bool:
        # Read under lock so session checks stay consistent with start/stop updates.
        with self._lock:
            return session_id == self._session_id

    # Summary:
    # - Emit proxy signals only when the worker still owns the active stream session.
    # - Input: `self`, `session_id` (int), `emit_fn` (callable), `args` (tuple[object, ...]).
    # - Returns: None.
    def _emit_for_session(self, session_id: int, emit_fn, *args) -> None:
        # Suppress stale thread callbacks that can otherwise flip UI state after a new stream starts.
        if not self._is_current_session(session_id):
            # This log is intentionally debug-level so normal runs stay quiet.
            # If you need to debug a rare camera issue, enable debug logging to see these events.
            logger.debug(
                "Suppressed stale camera event from session %s.",
                session_id,
            )
            return
        self._safe_emit(emit_fn, *args)

    # Summary:
    # - Convert an OpenCV backend constant into a short human-readable debug label.
    # - Input: `self`, `backend` (int) backend id used for VideoCapture.
    # - Returns: Backend label (str).
    def _backend_name(self, backend: int) -> str:
        # Keep labels explicit so startup logs are easy to compare across attempts.
        if backend == getattr(cv2, "CAP_DSHOW", -1):
            return "DSHOW"
        if backend == getattr(cv2, "CAP_MSMF", -1):
            return "MSMF"
        if backend == getattr(cv2, "CAP_ANY", -1):
            return "ANY"
        return str(int(backend))

    # Summary:
    # - Build the backend order for startup retries using the selected fallback strategy.
    # - Input: `self`.
    # - Returns: Ordered list of backend ids (list[int]) for each startup attempt.
    def _build_backend_attempt_sequence(self) -> list[int]:
        # Preferred strategy: DSHOW -> MSMF -> DSHOW.
        # DirectShow works well for many USB cameras, but some frame grabbers initialize more reliably
        # using Media Foundation. We try a deterministic sequence so behavior is reproducible.
        dshow = getattr(cv2, "CAP_DSHOW", None)
        msmf = getattr(cv2, "CAP_MSMF", None)
        attempt_order: list[int] = []
        if dshow is not None:
            attempt_order.append(int(dshow))
        if msmf is not None:
            attempt_order.append(int(msmf))
        if dshow is not None:
            attempt_order.append(int(dshow))

        # Fallback to CAP_ANY when backend constants are unavailable in this OpenCV build.
        if not attempt_order:
            attempt_order = [int(getattr(cv2, "CAP_ANY", 0))]

        # Keep total attempts aligned with retry policy (first try + configured retries).
        max_attempts = CAMERA_STARTUP_MAX_RETRIES + 1
        if len(attempt_order) < max_attempts:
            attempt_order.extend([attempt_order[-1]] * (max_attempts - len(attempt_order)))
        return attempt_order[:max_attempts]

    # Summary:
    # - Sleep in short slices so stop requests can interrupt waits quickly.
    # - Input: `self`, `stop_event` (threading.Event), `seconds` (float).
    # - Returns: `True` if stop was requested during sleep, otherwise `False`.
    def _sleep_with_stop(self, stop_event: threading.Event, seconds: float) -> bool:
        # Long blocking sleeps can delay shutdown and worsen rapid camera switching behavior.
        # We sleep in tiny slices so stop() can interrupt quickly and the UI "Stop Stream" feels immediate.
        end_time = time.monotonic() + max(float(seconds), 0.0)
        while time.monotonic() < end_time:
            if stop_event.is_set():
                return True
            remaining = end_time - time.monotonic()
            time.sleep(min(0.01, max(remaining, 0.0)))
        return stop_event.is_set()

    # Summary:
    # - Respect a short cooldown after stop before reopening the camera device.
    # - Input: `self`, `stop_event` (threading.Event).
    # - Returns: `True` when cooldown completed, `False` when stop interrupted startup.
    def _wait_for_start_cooldown(self, stop_event: threading.Event) -> bool:
        # Read cooldown timing atomically so start uses the latest stop timestamp.
        with self._lock:
            elapsed = time.monotonic() - self._last_stop_ts
        remaining = CAMERA_REOPEN_COOLDOWN_SEC - elapsed
        if remaining <= 0:
            return True
        # The short cooldown reduces the chance that the device starts in a bad state.
        # This is especially helpful for USB frame grabbers that sometimes return blank frames right after open.
        logger.debug("Waiting %.3fs camera reopen cooldown.", remaining)
        return not self._sleep_with_stop(stop_event, remaining)

    # Summary:
    # - Open a camera and validate startup by requiring at least one non-empty, non-black frame.
    # - Input: `self`, `camera_index` (int), `backend` (int).
    # - Returns: `(capture, first_frame, reason)` where capture/frame are None on failure.
    def _open_validated_capture(
        self,
        camera_index: int,
        backend: int,
    ) -> tuple[Optional[cv2.VideoCapture], Optional[np.ndarray], str]:
        # Open with the requested backend so retries can apply backend fallback deterministically.
        cap = cv2.VideoCapture(camera_index, backend)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None, None, f"backend={self._backend_name(backend)} open failed"

        # Warm up startup frames and reject all-black output to catch blank-initialization states.
        # For the Epiphan frame grabber issue, the device can appear "opened" but only deliver black frames.
        # We treat a warmup of only-black frames as a failed startup so the code can auto-retry/reopen.
        #
        # Tradeoff note: If the real scene is truly black (e.g., lens cap on), this will also be treated as
        # a startup failure. That is acceptable here because the goal is a reliable "not blank" preview.
        deadline = time.monotonic() + CAMERA_STARTUP_WARMUP_TIMEOUT_SEC
        checked_frames = 0
        while (
            checked_frames < CAMERA_STARTUP_MAX_WARMUP_FRAMES
            and time.monotonic() < deadline
        ):
            checked_frames += 1
            try:
                ret, frame = cap.read()
            except Exception:
                cap.release()
                return (
                    None,
                    None,
                    f"backend={self._backend_name(backend)} read raised exception",
                )
            if not ret or frame is None or frame.size == 0:
                # Keep warming up: some devices return a few empty frames right after open.
                continue
            if np.count_nonzero(frame) == 0:
                # Treat "all zero pixels" as blank output and keep warming up/retrying.
                continue
            return cap, frame, ""

        cap.release()
        return (
            None,
            None,
            (
                f"backend={self._backend_name(backend)} warmup returned empty/black frames "
                f"({checked_frames} checks)"
            ),
        )

    # Summary:
    # - Convert one BGR frame into the stream packet format with optional calibrated crop checks.
    # - Input: `self`, `frame` (np.ndarray), `clip_rect` (tuple[int, int, int, int] | None).
    # - Returns: `(packet, error_message)` where `packet` is None on conversion/validation failure.
    def _build_packet_from_frame(
        self,
        frame: np.ndarray,
        clip_rect: Optional[tuple[int, int, int, int]],
    ) -> tuple[Optional[FramePacket], Optional[str]]:
        # Validate the source payload before color conversion so invalid driver outputs fail explicitly.
        # Some camera backends can return a "successful read" but still provide an empty array.
        if frame is None or frame.size == 0:
            return None, "The camera returned an empty frame. The stream has been closed."

        # Convert to grayscale because downstream reconstruction expects single-channel uint8 images.
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            # If color conversion fails, the frame is unusable downstream, so fail the stream cleanly.
            return None, "Failed to convert a camera frame to grayscale. The stream has been closed."

        # Apply optional calibration crop only after conversion so dimensions match stream payload.
        if clip_rect is not None:
            clip_origin_x, clip_origin_y, clip_width, clip_height = clip_rect
            frame_height, frame_width = gray.shape
            clip_end_x = clip_origin_x + clip_width
            clip_end_y = clip_origin_y + clip_height
            # Keep strict bounds mode so invalid calibration never produces partial/out-of-range crops.
            if (
                clip_origin_x < 0
                or clip_origin_y < 0
                or clip_end_x > frame_width
                or clip_end_y > frame_height
            ):
                # A bad crop means our calibration doesn't match the camera resolution.
                # Failing fast is safer than showing misleading/shifted images.
                return (
                    None,
                    "Calibration crop rectangle is outside camera frame bounds. The stream has been closed.",
                )
            gray = gray[clip_origin_y:clip_end_y, clip_origin_x:clip_end_x]

        height, width = gray.shape
        packet = FramePacket(
            _now_ms(),
            width,
            height,
            "gray8",
            gray.tobytes(),
        )
        return packet, None

    # Summary:
    # - Worker loop that opens the camera and streams frames until stopped.
    # - Input: `self`, `session_id` (int), `camera_index` (int), `fps` (int), `stop_event`
    #   (threading.Event),
    #   `clip_rect` (tuple[int, int, int, int] | None) for optional calibrated cropping.
    # - Returns: None.
    def _run(
        self,
        session_id: int,
        camera_index: int,
        fps: int,
        stop_event: threading.Event,
        clip_rect: Optional[tuple[int, int, int, int]],
    ) -> None:
        cap: Optional[cv2.VideoCapture] = None
        first_frame: Optional[np.ndarray] = None
        streaming_started = False

        try:
            # Respect a short stop->start cooldown so USB frame-grabber drivers have time to settle.
            if not self._wait_for_start_cooldown(stop_event):
                logger.debug(
                    "Camera startup interrupted during cooldown (session=%s).",
                    session_id,
                )
                return

            # Attempt startup using backend fallback plus retry policy to reduce blank initialization states.
            backend_attempts = self._build_backend_attempt_sequence()
            startup_failure_reason = "Unknown startup failure."
            for attempt_index, backend in enumerate(backend_attempts, start=1):
                # If the user already pressed stop or a newer stream started, do not waste time opening.
                if stop_event.is_set() or not self._is_current_session(session_id):
                    logger.debug(
                        "Camera startup canceled before attempt %s (session=%s).",
                        attempt_index,
                        session_id,
                    )
                    return

                cap, first_frame, startup_failure_reason = self._open_validated_capture(
                    camera_index,
                    backend,
                )
                if cap is not None and first_frame is not None:
                    logger.debug(
                        "Camera startup success (session=%s, backend=%s, attempt=%s/%s).",
                        session_id,
                        self._backend_name(backend),
                        attempt_index,
                        len(backend_attempts),
                    )
                    break

                logger.debug(
                    "Camera startup failed (session=%s, backend=%s, attempt=%s/%s): %s",
                    session_id,
                    self._backend_name(backend),
                    attempt_index,
                    len(backend_attempts),
                    startup_failure_reason,
                )
                if attempt_index < len(backend_attempts):
                    # Wait briefly before retrying to avoid hammering the driver during startup recovery.
                    if self._sleep_with_stop(stop_event, CAMERA_STARTUP_RETRY_DELAY_SEC):
                        logger.debug(
                            "Camera startup retry sleep interrupted by stop event (session=%s).",
                            session_id,
                        )
                        return

            if cap is None or first_frame is None:
                # Emit one clear startup message after all fallback attempts are exhausted.
                self._emit_for_session(
                    session_id,
                    self._proxy.error_message.emit,
                    (
                        "Failed to initialize the selected camera after startup retries and backend fallback. "
                        f"Reason: {startup_failure_reason}"
                    ),
                )
                self._emit_for_session(session_id, self._proxy.state_changed.emit, False, "camera failed")
                return

            # Register capture only if this worker still owns the active stream session.
            with self._lock:
                if session_id != self._session_id:
                    # Another stream started while we were initializing.
                    # Release immediately so the new stream can own the device.
                    cap.release()
                    return
                self._cap = cap

            # Validate crop size early so invalid calibration data fails before steady-state loop starts.
            if clip_rect is not None:
                _, _, clip_width, clip_height = clip_rect
                if clip_width <= 0 or clip_height <= 0:
                    # A non-positive crop size would crash or produce empty output; fail cleanly.
                    self._emit_for_session(
                        session_id,
                        self._proxy.error_message.emit,
                        "Invalid clip rectangle size detected. The stream has been closed.",
                    )
                    self._emit_for_session(
                        session_id,
                        self._proxy.state_changed.emit,
                        False,
                        "camera failed",
                    )
                    return

            # Notify the UI only after validated startup succeeds to avoid false-running blank streams.
            self._emit_for_session(session_id, self._proxy.state_changed.emit, True, "camera streaming")
            streaming_started = True

            # Convert the requested interval (ms) into seconds for the paced capture loop.
            frame_interval = max(int(fps), 1) / 1000.0

            # Use the first validated warmup frame as the first delivered packet.
            first_packet, first_frame_error = self._build_packet_from_frame(first_frame, clip_rect)
            if first_frame_error is not None:
                # If the first validated frame still cannot be processed, stop before entering the loop.
                self._emit_for_session(session_id, self._proxy.error_message.emit, first_frame_error)
                return
            if first_packet is not None:
                self._emit_for_session(session_id, self._proxy.frame_ready.emit, first_packet)

            # Keep reading frames until stop is requested or the session is replaced.
            while not stop_event.is_set() and self._is_current_session(session_id):
                ret, frame = cap.read()
                if not ret:
                    # Notify once then exit so the stream can restart cleanly.
                    self._emit_for_session(
                        session_id,
                        self._proxy.error_message.emit,
                        "The camera stopped sending frames. The stream has been closed.",
                    )
                    break

                packet, frame_error = self._build_packet_from_frame(frame, clip_rect)
                if frame_error is not None:
                    # Any transform/crop failure means we can't safely stream this frame downstream.
                    self._emit_for_session(session_id, self._proxy.error_message.emit, frame_error)
                    break
                if packet is not None:
                    self._emit_for_session(session_id, self._proxy.frame_ready.emit, packet)

                # Sleep in stop-aware slices so stream switching remains responsive.
                if self._sleep_with_stop(stop_event, frame_interval):
                    break
        finally:
            # Always release the local capture handle when the worker loop ends.
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    logger.debug(
                        "Camera release raised during worker shutdown.",
                        exc_info=True,
                    )

            emit_stopped_state = False
            with self._lock:
                # Clear global capture pointer only when it still points to this worker's capture.
                if self._cap is cap:
                    self._cap = None
                # Only the active session should mutate active/thread state.
                if session_id == self._session_id:
                    # Only the current session can change the "active" flag.
                    # This prevents an older thread from marking the worker idle while a newer stream is running.
                    self._active = False
                    self._thread = None
                    emit_stopped_state = streaming_started

            # Emit stopped only for the active stream session to prevent stale UI resets.
            if emit_stopped_state:
                self._emit_for_session(
                    session_id,
                    self._proxy.state_changed.emit,
                    False,
                    "camera stopped",
                )
            elif streaming_started:
                logger.debug(
                    "Suppressed stale stop-state emit (session=%s).",
                    session_id,
                )


# Summary:
# - UI-thread worker that captures a screen region using QScreen.grabWindow.
# - What it does: Uses a QTimer on the UI thread because grabWindow must run on the GUI thread and is
#   lightweight enough for this use case, so a separate thread is unnecessary.
class ScreenGrabWorker(QObject):
    # Summary:
    # - Initialize the worker with a proxy and a UI-thread timer.
    # - Input: `self`, `proxy` (BModeStreamProxy), `parent` (QObject | None).
    # - Returns: None.
    def __init__(self, proxy: BModeStreamProxy, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        # Store the proxy so we can emit UI-safe signals.
        self._proxy = proxy

        # Create the UI-thread timer that drives screen grabs.
        self._timer = QTimer(self)
        self._timer.setObjectName("bmodeScreenTimer")
        # Signal connection: grab a frame on each timer tick.
        self._timer.timeout.connect(self._on_bmodeScreenTimer_timeout)

        # Cache the current screen and capture rectangle for the active session.
        self._screen: Optional[QScreen] = None
        self._rect: Optional[tuple[int, int, int, int]] = None

    # Summary:
    # - Start grabbing the screen region on a timer.
    # - Input: `self`, `screen` (QScreen), `rect` (tuple[int, int, int, int]), `fps` (int; ms interval).
    # - Returns: None.
    def start(self, screen: QScreen, rect: tuple[int, int, int, int], fps: int = 33) -> None:
        # Avoid double-starts so repeated calls are safe.
        if self._timer.isActive():
            return

        # Validate the input screen before starting the timer.
        if screen is None:
            self._proxy.error_message.emit(
                "No screen was detected. Cannot start screen streaming."
            )
            self._proxy.state_changed.emit(False, "screen failed")
            return

        origin_x, origin_y, size_width, size_height = rect
        if size_width <= 0 or size_height <= 0:
            self._proxy.error_message.emit(
                "Invalid clip rectangle size detected. The stream has been closed."
            )
            self._proxy.state_changed.emit(False, "screen failed")
            return

        # Store the capture settings so the timer callback can use them.
        self._screen = screen
        self._rect = rect

        # Start the UI-thread timer at the requested interval.
        self._timer.setInterval(max(int(fps), 1))
        self._timer.start()

        # Notify the UI that screen streaming is active.
        self._proxy.state_changed.emit(True, "screen streaming")

    # Summary:
    # - Stop the active screen-grab timer and clear cached state.
    # - Input: `self`.
    # - Returns: None.
    def stop(self) -> None:
        # Skip work if the timer is already stopped.
        if not self._timer.isActive():
            return

        # Stop the UI-thread timer so no more grabs occur.
        self._timer.stop()

        # Clear cached screen info so stale settings are not reused.
        self._screen = None
        self._rect = None

        # Notify the UI that screen streaming ended.
        self._proxy.state_changed.emit(False, "screen stopped")

    # Summary:
    # - Slot function that captures a screen frame on each timer tick.
    # - Input: `self`.
    # - Returns: None.
    def _on_bmodeScreenTimer_timeout(self) -> None:
        # Skip if the screen or rect is missing (should not happen while active).
        if self._screen is None or self._rect is None:
            return

        # Ensure the selected screen is still connected.
        if self._screen not in QGuiApplication.screens():
            self._proxy.error_message.emit(
                "The screen is no longer available. The stream has been closed."
            )
            self.stop()
            return

        origin_x, origin_y, size_width, size_height = self._rect
        if size_width <= 0 or size_height <= 0:
            self._proxy.error_message.emit(
                "Invalid clip rectangle size detected. The stream has been closed."
            )
            self.stop()
            return

        # Grab the calibrated region from the screen on the UI thread.
        pixmap = self._screen.grabWindow(
            0, origin_x, origin_y, size_width, size_height
        )
        if pixmap.isNull():
            self._proxy.error_message.emit(
                "Failed to capture the screen region. The stream has been closed."
            )
            self.stop()
            return

        # Convert the pixmap into grayscale so downstream payloads are single-channel uint8.
        qimg = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        bytes_per_line = qimg.bytesPerLine()
        expected_bytes_per_line = qimg.width()
        height = qimg.height()

        # Extract the raw bytes from the QImage memory.
        bits = qimg.bits()
        buffer_size = qimg.sizeInBytes()
        if hasattr(bits, "setsize"):
            # PySide builds that expose sip.voidptr need setsize to view the buffer.
            bits.setsize(buffer_size)
            raw = bytes(bits)
        else:
            # Some builds return a memoryview; slice the reported size.
            raw = bytes(bits[:buffer_size])

        if bytes_per_line != expected_bytes_per_line:
            # Remove per-line padding so the payload is tightly packed grayscale bytes.
            packed_lines = []
            for row in range(height):
                start = row * bytes_per_line
                packed_lines.append(raw[start : start + expected_bytes_per_line])
            data = b"".join(packed_lines)
        else:
            data = raw

        # Emit the frame packet for UI rendering.
        packet = FramePacket(
            _now_ms(),
            qimg.width(),
            qimg.height(),
            "gray8",
            data,
        )
        self._proxy.frame_ready.emit(packet)


# Summary:
# - Background worker that writes incoming B-mode frames as JPEG files on a thread.
# - What it does: Accepts frames on the UI thread, enqueues them, and encodes/writes on a worker thread.
class ImageRecordWorker:
    # Summary:
    # - Initialize the image recording worker and its thread state.
    # - Input: `self`, `proxy` (BModeStreamProxy).
    # - Returns: None.
    def __init__(self, proxy: BModeStreamProxy) -> None:
        # Store the proxy so background threads can send UI-safe signals.
        self._proxy = proxy

        # Track whether recording is active so we can ignore frames when stopped.
        self._active = False
        # Store thread resources that are created on start and released on stop.
        self._queue: Optional[queue.Queue] = None
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._session_dir: Optional[str] = None

        # Track indexing so filenames are sorted by capture order.
        self._frame_index = 0

        # Track drop statistics to detect sustained overload on the UI thread.
        self._record_drop_window_start: Optional[float] = None
        self._record_drop_window_total = 0
        self._record_drop_window_dropped = 0
        self._record_overload_triggered = False

        # Guard start/stop so we do not race if the UI toggles quickly.
        self._lock = threading.Lock()

    # Summary:
    # - Build a unique session directory path for a new recording session.
    # - Input: `record_dir` (str).
    # - Returns: Full session directory path (str).
    @staticmethod
    def _build_session_dir(record_dir: str) -> str:
        # Use UTC in the folder name so sessions are sortable and unambiguous.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        base_name = f"bmode_{timestamp}"
        session_dir = os.path.join(record_dir, base_name)

        # Add a numeric suffix when a folder with the same name already exists.
        if not os.path.exists(session_dir):
            return session_dir

        for suffix in range(1, 1000):
            candidate = os.path.join(record_dir, f"{base_name}_{suffix:03d}")
            if not os.path.exists(candidate):
                return candidate

        # Fall back to the base name if we somehow exhausted all suffixes.
        return session_dir

    # Summary:
    # - Start the JPEG writer thread and reset recording state for a new session.
    # - Input: `self`, `record_dir` (str), `queue_maxsize` (int | None).
    # - Returns: Created session directory path (str).
    def start(
        self,
        record_dir: str,
        queue_maxsize: Optional[int] = None,
    ) -> str:
        # Ensure only one recording session is active at a time.
        with self._lock:
            if self._active:
                return self._session_dir or record_dir
            self._active = True

            # Normalize and ensure the base record directory exists.
            record_dir = os.path.abspath(record_dir)
            try:
                os.makedirs(record_dir, exist_ok=True)
                session_dir = self._build_session_dir(record_dir)
                os.makedirs(session_dir, exist_ok=True)
            except OSError:
                # Reset active state when the directory cannot be created.
                self._active = False
                self._session_dir = None
                raise

            # Reset per-session state before the writer thread starts.
            self._session_dir = session_dir
            self._frame_index = 0
            self._record_drop_window_start = None
            self._record_drop_window_total = 0
            self._record_drop_window_dropped = 0
            self._record_overload_triggered = False

            # Use the requested queue size so external sync can avoid drops.
            if queue_maxsize is None:
                queue_maxsize = IMAGE_RECORD_QUEUE_MAXSIZE
            # Create fresh queue/stop event per session so writer state is isolated.
            self._queue = queue.Queue(maxsize=max(int(queue_maxsize), 0))
            self._stop_event = threading.Event()

            # Start the writer thread so JPEG encoding and disk I/O never block the UI.
            self._thread = threading.Thread(
                target=self._writer_loop,
                args=(session_dir, self._queue, self._stop_event),
                daemon=True,
            )
            # Thread start: begin draining frames to disk asynchronously.
            self._thread.start()

            return session_dir

    # Summary:
    # - Stop the active recording session and release writer resources safely.
    # - Input: `self`, `reason` (str).
    # - Returns: None.
    def stop(self, reason: str = "") -> None:
        # The UI owns user-facing messaging, so we only stop the worker here.
        _ = reason

        # Ensure stop is idempotent and thread-safe.
        with self._lock:
            if not self._active:
                return
            self._active = False

            stop_event = self._stop_event
            writer_thread = self._thread

            # Clear references so new sessions start cleanly.
            self._queue = None
            self._thread = None
            self._stop_event = None
            self._session_dir = None

        # Signal the writer thread to drain and exit.
        if stop_event is not None:
            stop_event.set()

        # Join briefly so we do not freeze the UI while final frames flush.
        if writer_thread is not None:
            writer_thread.join(timeout=0.5)

    # Summary:
    # - Check whether recording is currently active.
    # - Input: `self`.
    # - Returns: True when recording is active (bool).
    def is_active(self) -> bool:
        # Lock to avoid racing a start/stop toggle.
        with self._lock:
            return self._active

    # Summary:
    # - Emit a proxy signal safely when the UI may have already been destroyed.
    # - Input: `self`, `emit_fn` (callable), `args` (tuple[object, ...]).
    # - Returns: None.
    def _safe_emit(self, emit_fn, *args) -> None:
        # Avoid crashing if Qt deletes the proxy while background threads still run.
        try:
            emit_fn(*args)
        except RuntimeError:
            # Ignore emits after the proxy is gone; shutdown is in progress.
            return

    # Summary:
    # - Enqueue a frame for JPEG recording without blocking the UI thread.
    # - Input: `self`, `packet` (FramePacket).
    # - Returns: None.
    def handle_frame(self, packet: FramePacket) -> None:
        # Snapshot state under lock so stop/start does not race frame handling.
        with self._lock:
            if not self._active:
                return
            record_queue = self._queue
            stop_event = self._stop_event

        # Skip enqueue work when the queue is missing or a stop is pending.
        if record_queue is None or (stop_event is not None and stop_event.is_set()):
            return

        # Only pass the minimal payload needed for the writer thread.
        payload = (packet.timestamp_ms, packet.width, packet.height, packet.data)

        # Enqueue without blocking to keep the UI thread responsive.
        dropped = False
        try:
            record_queue.put_nowait(payload)
        except queue.Full:
            dropped = True

        # Track drops over a time window to detect sustained overload.
        now = time.monotonic()
        if self._record_drop_window_start is None:
            self._record_drop_window_start = now
            self._record_drop_window_total = 0
            self._record_drop_window_dropped = 0
        self._record_drop_window_total += 1
        if dropped:
            self._record_drop_window_dropped += 1

        window_elapsed = now - self._record_drop_window_start
        if window_elapsed >= IMAGE_RECORD_DROP_WINDOW_SECONDS:
            drop_rate = 0.0
            if self._record_drop_window_total > 0:
                drop_rate = (
                    self._record_drop_window_dropped / self._record_drop_window_total
                )
            # Reset the window counters for the next measurement period.
            self._record_drop_window_start = now
            self._record_drop_window_total = 0
            self._record_drop_window_dropped = 0

            if drop_rate >= IMAGE_RECORD_DROP_RATE_THRESHOLD:
                if not self._record_overload_triggered:
                    # Stop recording once when sustained overload is detected.
                    self._record_overload_triggered = True
                    self._safe_emit(
                        self._proxy.record_stop.emit,
                        "Recording stopped because the encoder queue is overloaded.",
                    )

    # Summary:
    # - Writer loop that encodes queued frames and saves them as JPEG files.
    # - Input: `self`, `session_dir` (str), `record_queue` (queue.Queue),
    #   `stop_event` (threading.Event).
    # - Returns: None.
    def _writer_loop(
        self,
        session_dir: str,
        record_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        # Drain the queue until stopped, then exit once it is empty.
        while True:
            if stop_event.is_set():
                # If a stop is requested, drain any remaining frames without blocking.
                try:
                    payload = record_queue.get_nowait()
                except queue.Empty:
                    break
            else:
                # When running, wait briefly for the next frame to reduce CPU usage.
                try:
                    payload = record_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

            timestamp_ms, width, height, data = payload
            # Validate against a single-channel payload size because streaming now uses uint8 grayscale.
            expected_size = int(width) * int(height)
            if len(data) < expected_size:
                # Stop recording if frame data is incomplete to avoid corrupt files.
                self._safe_emit(
                    self._proxy.record_stop.emit,
                    "Recording stopped because a frame was incomplete.",
                )
                stop_event.set()
                break

            try:
                # Decode the packed frame buffer as a 2D grayscale image.
                gray = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
            except ValueError:
                # Stop recording if the byte buffer cannot be reshaped.
                self._safe_emit(
                    self._proxy.record_stop.emit,
                    "Recording stopped because a frame could not be decoded.",
                )
                stop_event.set()
                break

            # Encode grayscale JPEG directly to keep disk writes lightweight.
            ok, buffer = cv2.imencode(
                ".jpg",
                gray,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            if not ok:
                self._safe_emit(
                    self._proxy.record_stop.emit,
                    "Recording stopped because JPEG encoding failed.",
                )
                stop_event.set()
                break

            # Build the output path with a stable filename pattern.
            filename = f"frame_{self._frame_index:06d}_{timestamp_ms}.jpg"
            output_path = os.path.join(session_dir, filename)

            try:
                with open(output_path, "wb") as file_handle:
                    file_handle.write(buffer.tobytes())
            except OSError:
                self._safe_emit(
                    self._proxy.record_stop.emit,
                    "Recording stopped because a file could not be written.",
                )
                stop_event.set()
                break

            # Increment the frame index so filenames remain ordered.
            self._frame_index += 1


# Summary: Main Qt window for the B-mode streaming UI.
# What it does: Builds the widgets from the Designer `.ui`, wires signals, and manages stream state.
# Input: Created with no args; uses `self.ui` widgets and internal state to react to user actions.
# Returns: A QWidget subclass instance that can be shown in a QApplication.
class BModeWidget(QWidget):
    """Lightweight window that simply renders the V2 form."""
    # Signal: emit one image packet per received frame for downstream coupling.
    sig_image_packet = Signal(int, object)

    # Summary: Initialize the window and connect UI signals.
    # What it does: Sets up the generated UI, prepares stream workers, and hooks buttons/combos to handlers.
    # Input: `self` (the new window instance).
    # Returns: `None` (constructor).
    def __init__(self) -> None:
        super().__init__()  # Call the QWidget constructor to initialize Qt internals.
        self.ui = Ui_BModeV2()  # Instantiate the generated UI helper for the V2 layout.
        self.ui.setupUi(self)  # Wire the widgets from the .ui file to this QWidget instance.
        # UI state change: keep the text stream read-only so it behaves like a status/event console.
        self.ui.plainTextEdit_bmode_textStream.setReadOnly(True)
        # UI state change: disable undo history because logs are append-only and should stay lightweight.
        self.ui.plainTextEdit_bmode_textStream.setUndoRedoEnabled(False)
        # UI state change: force stylesheet background/border painting on this QLabel container.
        # WHY: some QWidget/QLabel styles render inconsistently without WA_StyledBackground enabled.
        self.ui.label_bmode_image.setAttribute(Qt.WA_StyledBackground, True)
        # Cache the original image-label stylesheet so recording indicator changes keep the base background color.
        self._bmode_image_label_base_stylesheet = self.ui.label_bmode_image.styleSheet()
        # Keep a fixed border thickness for the recording indicator to avoid magic numbers in style updates.
        self._bmode_recording_indicator_border_px = 6
        # UI state change: reserve border space from startup with a transparent border.
        # WHY: this prevents layout/size-hint growth when recording turns the border red.
        self.ui.label_bmode_image.setStyleSheet(
            self._build_bmode_recording_indicator_stylesheet(
                border_color="rgba(255, 0, 0, 0)"
            )
        )

        # Cache the calibration-derived screen rectangle used for screen streaming.
        self._screen_rect: Optional[tuple[int, int, int, int]] = None

        # Create the proxy and workers that handle acquisition outside the UI widgets.
        self._proxy = BModeStreamProxy()
        self._camera_worker = CameraStreamWorker(self._proxy)
        self._screen_worker = ScreenGrabWorker(self._proxy, parent=self)
        # Create the recording worker so the UI can offload JPEG writes.
        self._image_record_worker = ImageRecordWorker(self._proxy)

        # Cache the most recent frame packet so the UI can render on a timer.
        self._latest_frame_packet: Optional[FramePacket] = None
        # Cache the latest image packet fields for coupled stream consumers.
        self._latest_image_ts_ms: Optional[int] = None
        self._latest_image_data: Optional[object] = None
        # Track the current stream state to keep the open/stop button repeat-safe.
        self._is_streaming = False
        # Track external recording mode so image saving can be driven by mocap timestamps.
        self._external_recording_active = False
        # Keep the text stream bounded so long sessions do not grow the widget memory without limit.
        self._textstream_max_lines = 200
        # Track the most recent formatted line to debounce repeated worker-state events.
        self._last_textstream_line = ""
        self._last_textstream_line_ts = 0.0
        # Track a small health window so we can report stream FPS without per-frame log spam.
        self._health_window_start_ts = time.monotonic()
        self._health_window_frame_count = 0
        # Track health-log cadence separately so rate-limiting follows the same pattern as mocap.
        self._last_health_log_ts = 0.0
        self._health_log_interval_sec = 1.0

        # Timer to throttle rendering so the UI thread stays responsive.
        self._display_timer = QTimer(self)
        self._display_timer.setObjectName("bmodeDisplayTimer")
        self._display_timer.setInterval(33)  # ~33 FPS
        # Signal connection: render the latest cached frame on each tick.
        self._display_timer.timeout.connect(self._on_bmodeDisplayTimer_timeout)

        # Signal connection: cache incoming frames from workers.
        self._proxy.frame_ready.connect(self._on_bmodeStreamProxy_frame_ready)
        # Signal connection: react to start/stop state changes from workers.
        self._proxy.state_changed.connect(self._on_bmodeStreamProxy_state_changed)
        # Signal connection: surface worker errors on the UI thread.
        self._proxy.error_message.connect(self._on_bmodeStreamProxy_error_message)
        # Signal connection: stop recording when the writer thread reports a failure.
        self._proxy.record_stop.connect(self._on_bmodeStreamProxy_record_stop)
        # Signal connection: surface non-fatal recording messages on the UI thread.
        self._proxy.record_message.connect(self._on_bmodeStreamProxy_record_message)

        # Fill the stream port dropdown with any live camera indices we can open.
        self._populate_cameras()

        # Signal connection: update UI when the stream option changes.
        self.ui.comboBox_bmode_streamOption.currentTextChanged.connect(
            self._on_comboBox_bmode_streamOption_changed
        )
        # Apply the current selection immediately so the GUI is consistent on startup.
        self._on_comboBox_bmode_streamOption_changed(
            self.ui.comboBox_bmode_streamOption.currentText()
        )

        # Signal connection: trigger the stream-opening flow on button click.
        self.ui.pushButton_bmode_openStream.clicked.connect(
            self._on_pushButton_bmode_openStream_clicked
        )
        # Signal connection: open a file dialog when the user browses for calibration.
        self.ui.pushButton_bmode_calibBrowse.clicked.connect(
            self._on_pushButton_bmode_calibBrowse_clicked
        )
        # Signal connection: clear the calibration path when the clear button is clicked.
        self.ui.pushButton_bmode_calibClear.clicked.connect(
            self._on_pushButton_bmode_calibClear_clicked
        )
        # Signal connection: clear the record directory when the clear button is clicked.
        self.ui.pushButton_bmode_recorddirClear.clicked.connect(
            self._on_pushButton_bmode_recorddirClear_clicked
        )
        # Signal connection: open a directory picker for the record directory.
        self.ui.pushButton_bmode_recorddirBrowse.clicked.connect(
            self._on_pushButton_bmode_recorddirBrowse_clicked
        )
        # Signal connection: toggle recording when the record button is clicked.
        self.ui.pushButton_bmode_recordStream.clicked.connect(
            self._on_pushButton_bmode_recordStream_clicked
        )
        # Initial feedback: make the first state explicit so users know how to begin.
        self._log_bmode_event(
            "info",
            level="INFO",
            message="Idle. Select stream source and press Open Stream.",
        )

    # Summary: Append a single formatted line to the B-mode text stream.
    # What it does: Prefixes each message with local time/level and trims old lines to keep memory stable.
    # Input: `self`, `level` (str), `message` (str).
    # Returns: `None`.
    def _append_bmode_textstream(self, level: str, message: str) -> None:
        # Normalize and ignore empty messages so log output stays concise and useful.
        message_text = str(message).strip()
        if not message_text:
            return

        # Build a consistent line format that is easy to scan during live streaming.
        timestamp_text = datetime.now().strftime("%H:%M:%S")
        level_text = (level or "INFO").upper()
        formatted_line = f"[{timestamp_text}] [{level_text}] {message_text}"
        self.ui.plainTextEdit_bmode_textStream.appendPlainText(formatted_line)

        # Trim old lines from the top so the text widget remains bounded over long sessions.
        document = self.ui.plainTextEdit_bmode_textStream.document()
        while document.blockCount() > int(self._textstream_max_lines):
            cursor = QTextCursor(document)
            cursor.movePosition(QTextCursor.Start)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            # Remove the leftover newline after deleting the first block.
            cursor.deleteChar()

        # Keep the latest line visible after append/trim so the console behaves like a live activity feed.
        cursor = self.ui.plainTextEdit_bmode_textStream.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.ui.plainTextEdit_bmode_textStream.setTextCursor(cursor)
        self.ui.plainTextEdit_bmode_textStream.ensureCursorVisible()

    # Summary: Build and log a generic B-mode event message for the text stream.
    # What it does: Converts structured event context into one human-readable line and applies duplicate debouncing.
    # Input: `self`, `event` (str), `level` (str), `context` (keyword context fields).
    # Returns: `None`.
    def _log_bmode_event(self, event: str, *, level: str = "INFO", **context) -> None:
        # Normalize the event key so callers can pass stable, case-insensitive identifiers.
        event_key = str(event).strip().lower()
        message_text = ""

        # Build stream-start/stop request messages with source-specific context.
        if event_key == "stream_request":
            action = str(context.get("action", "start")).strip().lower() or "start"
            source = str(context.get("source", "unknown")).strip().lower() or "unknown"
            detail_parts: list[str] = []
            if source == "camera":
                camera_index = context.get("camera_index")
                if camera_index is not None:
                    detail_parts.append(f"camera={camera_index}")
            if source == "screen":
                screen_name = str(context.get("screen_name", "")).strip()
                if screen_name:
                    detail_parts.append(f"screen={screen_name}")
                screen_size = str(context.get("screen_size", "")).strip()
                if screen_size:
                    detail_parts.append(f"size={screen_size}")
            clip_rect = context.get("clip_rect")
            if clip_rect is not None:
                detail_parts.append(f"clip={clip_rect}")
            details_text = f" ({', '.join(detail_parts)})" if detail_parts else ""
            message_text = f"Stream {action} requested: {source}{details_text}."

        # Build stream running-state messages from worker signals.
        elif event_key == "stream_state":
            state_text = "started" if bool(context.get("is_running", False)) else "stopped"
            reason_text = str(context.get("reason", "")).strip()
            if reason_text:
                message_text = f"Stream {state_text}: {reason_text}."
            else:
                message_text = f"Stream {state_text}."

        # Build recording start messages with destination and mode details.
        elif event_key == "record_start":
            mode_text = str(context.get("mode", "local")).strip() or "local"
            target_text = str(context.get("target", "")).strip()
            if not target_text:
                target_text = str(context.get("session_dir", "")).strip()
            if target_text:
                message_text = f"Recording started ({mode_text}) -> {target_text}."
            else:
                message_text = f"Recording started ({mode_text})."

        # Build recording stop messages with a clear reason when available.
        elif event_key == "record_stop":
            reason_text = str(context.get("reason", "")).strip()
            if reason_text:
                message_text = f"Recording stopped: {reason_text}"
            else:
                message_text = "Recording stopped."

        # Build compact health snapshots from stream-health logic.
        elif event_key == "stream_health":
            message_text = str(context.get("message", "")).strip()

        # Build generic informational and warning/error messages.
        elif event_key in {"warning", "error", "info"}:
            message_text = str(context.get("message", "")).strip()

        # Fallback formatting: include event name and structured key/value fields.
        else:
            detail_parts = [f"{key}={context[key]}" for key in sorted(context.keys())]
            details_text = f"({', '.join(detail_parts)})" if detail_parts else ""
            message_text = f"Event {event_key}{details_text}"

        # Skip blank messages so malformed event context does not create empty rows.
        if not message_text:
            return

        # Debounce identical consecutive messages so repeated worker callbacks do not flood the UI.
        now_ts = time.monotonic()
        line_key = f"{(level or 'INFO').upper()}::{message_text}"
        if line_key == self._last_textstream_line and (
            now_ts - self._last_textstream_line_ts
        ) < 1.0:
            return
        self._last_textstream_line = line_key
        self._last_textstream_line_ts = now_ts
        self._append_bmode_textstream(level, message_text)

    # Summary: Log periodic stream health snapshots at a fixed cadence.
    # What it does: Rate-limits health logs, builds one compact status line, and avoids per-frame text spam.
    # Input: `self`.
    # Returns: `None`.
    def _log_stream_health(self) -> None:
        # Only emit health snapshots while stream state is active.
        if not self._is_streaming:
            return
        if self._health_window_frame_count <= 0:
            return

        # Rate-limit health lines with a dedicated timestamp gate to avoid per-frame text spam.
        now_ts = time.monotonic()
        if (now_ts - self._last_health_log_ts) < self._health_log_interval_sec:
            return
        self._last_health_log_ts = now_ts

        # Keep using the rolling frame window to estimate observed stream cadence.
        elapsed = now_ts - self._health_window_start_ts
        if elapsed <= 0.0:
            return

        # Convert sampled frames into an FPS estimate for this health window.
        fps_estimate = self._health_window_frame_count / elapsed

        # Reset the rolling window after each emitted health snapshot.
        frame_count = int(self._health_window_frame_count)
        self._health_window_start_ts = now_ts
        self._health_window_frame_count = 0

        # Build a compact health line that mirrors the mocap status style.
        recording_text = "on" if self._image_record_worker.is_active() else "off"
        health_line = (
            "Stream OK | "
            f"FPS~{fps_estimate:.1f} | "
            f"REC={recording_text}"
        )
        self._log_bmode_event("stream_health", level="INFO", message=health_line)

    # Summary: Scan for available cameras and show them in the dropdown.
    # What it does: Tries a small set of camera indices and adds the ones that open successfully.
    # Input: `self`.
    # Returns: `None`.
    def _populate_cameras(self) -> None:
        """List available camera indices in the stream port combo box."""
        # Start with an empty list to avoid stale entries between refreshes.
        self.ui.comboBox_bmode_streamPort.clear()
        for index in range(5):
            # Try to open each index and only keep it if the camera responds.
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if (cap is None) or (not cap.isOpened()):
                if cap is not None:
                    cap.release()
                # Skip indices that either failed to initialize or do not exist.
                continue
            # Add the working camera to the dropdown so the user can select it.
            self.ui.comboBox_bmode_streamPort.addItem(f"Camera {index}", index)
            cap.release()  # Close the camera as we only need to query its availability.

    # Summary: Slot function that updates UI when the stream option changes.
    # What it does: Enables/disables the camera port selector depending on whether we stream a camera or the screen.
    # Input: `self`, `text` (the current stream option label from the combo box).
    # Returns: `None`.
    def _on_comboBox_bmode_streamOption_changed(self, text: str) -> None:
        """Disable the port selector when the user picks screen streaming on this PC."""
        # When streaming to the PC screen, we do not need a camera index, so gray out the selector.
        self.ui.comboBox_bmode_streamPort.setEnabled(text != "Stream Screen (This PC)")

    # Summary: Slot function that opens/closes streaming when the user clicks the button.
    # What it does: Validates the current UI selections, then starts streaming or stops it if already running.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_openStream_clicked(self) -> None:
        # If we are already streaming, stop immediately so the button is repeat-safe.
        if self._is_streaming:
            # Normalize source labels so stop requests use the same camera/screen vocabulary as start requests.
            source_text = "camera"
            if (
                self.ui.comboBox_bmode_streamOption.currentText()
                == "Stream Screen (This PC)"
            ):
                source_text = "screen"
            # Log user intent so stop actions are visible in the text stream history.
            self._log_bmode_event(
                "stream_request",
                level="INFO",
                action="stop",
                source=source_text,
            )
            self._stop_stream()
            return

        # If the user selected local screen streaming, ensure the calibration path is set.
        stream_option = self.ui.comboBox_bmode_streamOption.currentText()
        calib_path = self.ui.lineEdit_bmode_calibPath.text().strip()
        if stream_option == "Stream Screen (This PC)" and not calib_path:
            # Log the validation failure so users can see why stream start was rejected.
            self._log_bmode_event(
                "warning",
                level="WARN",
                message="Calibration path is required before streaming the screen.",
            )
            QMessageBox.warning(
                self,
                "Calibration Path Missing",
                "Calibration path is required before streaming the screen.",  # replace this text as needed
            )
            return

        # Ensure we have at least one camera entry to use when streaming the camera.
        if stream_option == "Stream Image" and self.ui.comboBox_bmode_streamPort.count() == 0:
            # Log missing hardware so users have a persistent explanation in the text stream.
            self._log_bmode_event(
                "warning",
                level="WARN",
                message="No camera ports were detected. Plug in a camera and try again.",
            )
            QMessageBox.warning(
                self,
                "No Camera Found",
                "No available camera ports were detected. Please plug in a camera and try again.",
            )
            return

        # Start the stream after the warning so the flow continues.
        self._start_stream()

    # Summary: Slot function that lets the user choose a calibration XML file.
    # What it does: Opens a file picker, stores the chosen path in the UI, and parses the XML for clip rectangle data.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_calibBrowse_clicked(self) -> None:
        """Prompt the user to select a calibration XML file and store the path."""
        # Resolve the directory where this Python file lives (i.e., the app folder).
        # Using an absolute path avoids issues when the app is launched from a different working directory.
        base_dir = Path(__file__).resolve().parent
        # Build the path to the local "configs" folder where calibration XMLs are stored.
        configs_dir = base_dir / "configs"
        # Prefer a specific default calibration file so the file dialog opens directly on it.
        # This helps users find the correct XML quickly.
        default_xml = (
            configs_dir
            / "PlusDeviceSet_fCal_Epiphan_NDIPolaris_RadboudUMC_20241219_150400.xml"
        )
        # Choose what initial path the file dialog should show:
        # - If the default XML exists, open the dialog with that file selected.
        # - Otherwise, fall back to the configs directory (so the user can still browse).
        dialog_path = str(default_xml if default_xml.exists() else configs_dir)

        # Show a standard "Open File" dialog.
        # Returns a tuple: (selected_file_path, selected_filter).
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            # Title text shown at the top of the dialog window.
            "Select Calibration XML",
            # Initial directory or filename the dialog opens to.
            dialog_path,
            # Filter list the user can pick from; by default we focus on .xml files.
            "XML Files (*.xml);;All Files (*)",
        )
        # If the user cancels the dialog, Qt returns an empty string; do nothing in that case.
        if not file_path:
            return
        # Store the chosen file path in the UI so other actions (e.g., starting a stream)
        # can read and use this calibration file.
        self.ui.lineEdit_bmode_calibPath.setText(file_path)

        # Parse and validate the selected calibration so the screen stream can reuse the clip rectangle.
        try:
            self._screen_rect = self._load_clip_rectangle_from_calibration_path(file_path)
        except ValueError as error:
            QMessageBox.warning(
                self,
                "Calibration Error",
                str(error),
            )
            self._screen_rect = None
            return

    # Summary: Load and validate the clip rectangle from a calibration XML path.
    # What it does: Parses the file using XmlStreamParser and returns
    # `(origin_x, origin_y, size_width, size_height)` for stream cropping.
    # Input: `self`, `calibration_path` (str; path to the calibration XML file).
    # Returns: `tuple[int, int, int, int]` with clip rectangle values.
    # Raises: `ValueError` when the file cannot be parsed or does not contain valid clip rectangle values.
    def _load_clip_rectangle_from_calibration_path(
        self, calibration_path: str
    ) -> tuple[int, int, int, int]:
        # Validate early so callers get a clear message instead of a parser stack trace.
        normalized_path = calibration_path.strip()
        if not normalized_path:
            raise ValueError("Calibration path is empty.")

        # Parse the XML and extract only the VideoDevice attributes needed for clipping.
        parser = XmlStreamParser(str(normalized_path))
        parser.register(VideoDeviceExtractor())
        try:
            data = parser.parse()
        except Exception as error:
            # Convert parser/file failures into a user-friendly calibration error.
            raise ValueError(
                f"Failed to read the selected calibration file: {error}"
            ) from error

        # Read clip rectangle fields from the extracted VideoDevice attributes.
        video_device = data.get("VideoDevice") or {}
        video_device_attrib = video_device.get("attrib") or {}
        clip_rectangle_origin = video_device_attrib.get("ClipRectangleOrigin")
        clip_rectangle_size = video_device_attrib.get("ClipRectangleSize")
        if not clip_rectangle_origin or not clip_rectangle_size:
            raise ValueError(
                "Clip rectangle origin/size is missing in the selected calibration file."
            )

        # Convert the space-separated coordinates into integer values.
        try:
            origin_x, origin_y = map(int, clip_rectangle_origin.split())
            size_width, size_height = map(int, clip_rectangle_size.split())
        except ValueError as error:
            raise ValueError(
                "Clip rectangle origin/size values are invalid in the selected calibration file."
            ) from error

        # Reject non-positive sizes because streams require a non-empty crop region.
        if size_width <= 0 or size_height <= 0:
            raise ValueError(
                "Clip rectangle size must be greater than zero in the selected calibration file."
            )
        return (origin_x, origin_y, size_width, size_height)

    # Summary: Slot function that clears the calibration path and related cached screen settings.
    # What it does: Removes the calibration file path from the UI and clears any data that depends on it.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_calibClear_clicked(self) -> None:
        """Clear the calibration path text and reset calibration-derived state."""
        # Clear the stored path so no old file path is left in the UI.
        self.ui.lineEdit_bmode_calibPath.setText("")
        # Reset the placeholder to the default drive hint for the user.
        self.ui.lineEdit_bmode_calibPath.setPlaceholderText("D:\\")
        # Remove any cached calibration data so streaming can't reuse a stale file.
        self._screen_rect = None

    # Summary: Slot function that clears the record directory line edit.
    # What it does: Removes the record directory path so the user can pick a new one.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_recorddirClear_clicked(self) -> None:
        # UI state change: clear the selected record directory text.
        self.ui.lineEdit_bmode_recorddir.setText("")

    # Summary: Slot function that opens a directory picker for the record directory.
    # What it does: Opens a directory-only dialog and stores the chosen path in the UI.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_recorddirBrowse_clicked(self) -> None:
        # Use the current text as the starting folder when it exists.
        current_path = self.ui.lineEdit_bmode_recorddir.text().strip()
        # Open a directory-only picker so the user can select a folder.
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Record Directory",
            current_path,
            QFileDialog.ShowDirsOnly,
        )
        # Do nothing if the user cancels the dialog.
        if not selected_dir:
            return
        # Normalize to an absolute path so downstream code is consistent.
        absolute_path = os.path.abspath(selected_dir)
        # UI state change: store the selected directory in the line edit.
        self.ui.lineEdit_bmode_recorddir.setText(absolute_path)

    # Summary: Slot function that starts or stops image recording.
    # What it does: Toggles the JPEG writer thread based on current recording state.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_recordStream_clicked(self) -> None:
        # Toggle based on current recording state to keep the button behavior predictable.
        if self._image_record_worker.is_active():
            self._stop_recording()
            return
        self._start_recording()

    # Summary: Start a new JPEG recording session on the UI thread.
    # What it does: Validates the record directory, starts the writer thread, and locks UI controls.
    # Input: `self`.
    # Returns: `None`.
    def _start_recording(self) -> None:
        # Require an active stream so recordings stay aligned to live frames.
        if not self._is_streaming:
            # Log validation feedback so users can review why recording did not start.
            self._log_bmode_event(
                "warning",
                level="WARN",
                message="Start streaming before recording.",
            )
            QMessageBox.warning(
                self,
                "Stream Not Active",
                "Start streaming before recording.",
            )
            return

        # Validate the record directory so we fail fast on missing paths.
        record_dir = self.ui.lineEdit_bmode_recorddir.text().strip()
        if not record_dir:
            # Log missing path validation so users can see the required prerequisite.
            self._log_bmode_event(
                "warning",
                level="WARN",
                message="Please choose a record directory before recording.",
            )
            QMessageBox.warning(
                self,
                "Missing Record Directory",
                "Please choose a record directory before recording.",
            )
            return

        # Normalize to an absolute path so sessions are created consistently.
        record_dir = os.path.abspath(record_dir)

        # Ensure the directory exists so the writer can create a session folder.
        try:
            os.makedirs(record_dir, exist_ok=True)
        except OSError:
            # Log directory creation failures so users can inspect filesystem-related issues.
            self._log_bmode_event(
                "error",
                level="ERROR",
                message="The selected record directory could not be created.",
            )
            QMessageBox.warning(
                self,
                "Record Directory Error",
                "The selected record directory could not be created.",
            )
            return

        if not os.path.isdir(record_dir):
            # Log invalid path state so users can diagnose unexpected filesystem behavior.
            self._log_bmode_event(
                "error",
                level="ERROR",
                message="The selected record directory does not exist.",
            )
            QMessageBox.warning(
                self,
                "Invalid Record Directory",
                "The selected record directory does not exist.",
            )
            return
        if not os.access(record_dir, os.W_OK):
            # Log permission issues so users know recording failed due to write access.
            self._log_bmode_event(
                "error",
                level="ERROR",
                message="The selected record directory is not writable.",
            )
            QMessageBox.warning(
                self,
                "Record Directory Not Writable",
                "The selected directory is not writable.",
            )
            return

        # Start the writer thread so JPEG encoding and disk I/O stay off the UI thread.
        try:
            session_dir = self._image_record_worker.start(record_dir)
        except OSError:
            # Log session creation failures to preserve the exact stop reason in text feedback.
            self._log_bmode_event(
                "error",
                level="ERROR",
                message="Failed to create the recording session directory.",
            )
            QMessageBox.warning(
                self,
                "Record Directory Error",
                "Failed to create the recording session directory.",
            )
            return

        # UI state change: lock the record directory controls while recording is active.
        self.ui.lineEdit_bmode_recorddir.setEnabled(False)
        self.ui.pushButton_bmode_recorddirBrowse.setEnabled(False)
        self.ui.pushButton_bmode_recorddirClear.setEnabled(False)

        # UI state change: update the record button to show stop intent.
        self.ui.pushButton_bmode_recordStream.setText("Stop Recording")
        
        # UI state change: show a clear visual indicator on the image label while recording is active.
        self._set_bmode_recording_indicator(active=True)
        # Log active recording destination so users can quickly find saved images.
        self._log_bmode_event(
            "record_start",
            level="INFO",
            mode="local",
            target=session_dir,
        )

    # Summary: Stop the active JPEG recording session and release writer resources.
    # What it does: Stops the background thread, restores UI controls, and reports any reason.
    # Input: `self`, `reason` (str).
    # Returns: `None`.
    def _stop_recording(self, reason: str = "") -> None:
        # Avoid redundant work when recording is already stopped.
        if not self._image_record_worker.is_active():
            return

        # Stop the recording worker so it can flush and close cleanly.
        self._image_record_worker.stop(reason)

        # Reset external state so normal per-frame recording can resume later.
        self._external_recording_active = False

        # UI state change: unlock the record directory controls.
        self.ui.lineEdit_bmode_recorddir.setEnabled(True)
        self.ui.pushButton_bmode_recorddirBrowse.setEnabled(True)
        self.ui.pushButton_bmode_recorddirClear.setEnabled(True)

        # UI state change: restore the record button text.
        self.ui.pushButton_bmode_recordStream.setText("Record")

        # UI state change: hide the recording indicator while keeping reserved border space.
        self._set_bmode_recording_indicator(active=False)

        # Log recording-stop feedback so users can see user stops and forced stops in one place.
        stop_reason = reason.strip() if reason else "Recording stopped by user."
        self._log_bmode_event(
            "record_stop",
            level="WARN" if reason else "INFO",
            reason=stop_reason,
        )

        # Show the stop reason so the user understands why recording ended.
        if reason:
            QMessageBox.warning(self, "Recording Stopped", reason)

    # Summary:
    # - Build the scoped stylesheet used by the B-mode recording indicator.
    # - What it does: Preserves the original label style and appends a fixed-width border
    #   whose color can be toggled between red (active) and transparent (idle).
    # - Input: `self`, `border_color` (str).
    # - Returns: Complete stylesheet string (str).
    def _build_bmode_recording_indicator_stylesheet(self, border_color: str) -> str:
        # Use the cached base style so the UI-defined background color remains unchanged.
        base_stylesheet = self._bmode_image_label_base_stylesheet.strip()
        selector = "#label_bmode_image"
        # Keep border width fixed to reserve layout space in both idle and recording states.
        border_rule = (
            f"border: {self._bmode_recording_indicator_border_px}px solid {border_color};"
        )
        # Support selector-based base styles by appending a scoped override rule.
        if "{" in base_stylesheet and "}" in base_stylesheet:
            return f"{base_stylesheet}\n{selector} {{ {border_rule} }}"
        # Support property-only base styles by wrapping them in the label selector.
        if base_stylesheet:
            normalized_base = base_stylesheet.rstrip(";")
            return f"{selector} {{ {normalized_base}; {border_rule} }}"
        # Fall back to border-only style when no base declarations exist.
        return f"{selector} {{ {border_rule} }}"

    # Summary: Update the image-label border to indicate whether recording is active.
    # What it does: Keeps a fixed border width at all times, then toggles only border color
    # between red (active) and transparent (idle) for stable layout geometry.
    # Input: `self`, `active` (bool).
    # Returns: `None`.
    def _set_bmode_recording_indicator(self, active: bool) -> None:
        # Switch border color only, while keeping width fixed to avoid geometry jumps on toggle.
        if active:
            # UI state change: make the reserved border visible during active recording.
            self.ui.label_bmode_image.setStyleSheet(
                self._build_bmode_recording_indicator_stylesheet(
                    border_color="rgb(255, 0, 0)"
                )
            )
            return

        # UI state change: keep reserved space but hide the border when recording is not active.
        self.ui.label_bmode_image.setStyleSheet(
            self._build_bmode_recording_indicator_stylesheet(
                border_color="rgba(255, 0, 0, 0)"
            )
        )

    # Summary:
    # - Set the B-mode recording indicator from an external controller.
    # - What it does: Exposes a small public API so parent windows can toggle the same indicator
    #   used by this widget's local record button without duplicating stylesheet logic.
    # - Input: `self`, `active` (bool).
    # - Returns: None.
    def set_recording_indicator(self, active: bool) -> None:
        # Keep indicator style ownership inside this widget even when recording is started elsewhere.
        self._set_bmode_recording_indicator(active=active)

    # Summary:
    # - Start an externally-controlled image recording session.
    # - What it does: Starts the image writer thread and marks external mode so snapshots can be driven
    #   by mocap timestamps instead of every incoming frame.
    # - Input: `self`, `record_dir` (str).
    # - Returns: Created session directory path (str).
    def start_external_image_record(self, record_dir: str) -> str:
        # Start the writer thread with an unbounded queue so snapshots are never dropped.
        session_dir = self._image_record_worker.start(record_dir, queue_maxsize=0)
        # Track external state so per-frame auto recording is disabled.
        self._external_recording_active = True
        # Log externally-controlled start so coupled recording actions are visible in this widget.
        self._log_bmode_event(
            "record_start",
            level="INFO",
            mode="external",
            target=session_dir,
        )
        return session_dir

    # Summary:
    # - Stop an externally-controlled image recording session.
    # - What it does: Clears external mode and stops the image writer thread.
    # - Input: `self`, `reason` (str).
    # - Returns: None.
    def stop_external_image_record(self, reason: str = "") -> None:
        # Clear external mode so per-frame recording logic can resume when needed.
        self._external_recording_active = False
        # Stop the writer thread so file handles flush and close.
        self._image_record_worker.stop(reason)
        # Log externally-controlled stop so coupled flows remain transparent to users.
        stop_reason = reason.strip() if reason else "External recording stopped."
        self._log_bmode_event(
            "record_stop",
            level="WARN" if reason else "INFO",
            reason=stop_reason,
        )

    # Summary:
    # - Record a snapshot of the latest frame using an injected timestamp.
    # - What it does: Reuses the most recent frame bytes but overrides the timestamp to align
    #   filenames with mocap CSV row times.
    # - Input: `self`, `ts_ms` (int).
    # - Returns: None.
    def record_latest_frame_with_ts(self, ts_ms: int) -> None:
        # Ignore snapshot requests when external recording is not active.
        if not self._external_recording_active:
            return
        # Skip recording when no frame has been cached yet.
        if self._latest_frame_packet is None:
            return

        # Build a new packet with the injected timestamp but the same frame payload.
        latest_packet = self._latest_frame_packet
        snapshot_packet = FramePacket(
            int(ts_ms),
            latest_packet.width,
            latest_packet.height,
            latest_packet.fmt,
            latest_packet.data,
        )
        # Enqueue the snapshot without blocking the UI thread.
        self._image_record_worker.handle_frame(snapshot_packet)

    # Summary:
    # - Save one externally-controlled image snapshot directly to disk.
    # - What it does: Creates a B-mode session folder, validates one frame packet, encodes it
    #   with the same JPEG settings as continuous recording, and writes one image file.
    # - Input: `self`, `record_dir` (str), `ts_ms` (int), `image_data` (object).
    # - Returns: Full JPEG output path (str).
    def save_external_image_snapshot(
        self, record_dir: str, ts_ms: int, image_data: object
    ) -> str:
        # Normalize/create the parent directory so snapshot output has a stable target.
        normalized_record_dir = os.path.abspath(os.path.expanduser(str(record_dir)))
        os.makedirs(normalized_record_dir, exist_ok=True)

        # Reuse the existing B-mode session-folder convention under the coupled session.
        session_dir = ImageRecordWorker._build_session_dir(normalized_record_dir)
        os.makedirs(session_dir, exist_ok=True)

        # Validation/transform: accept the same FramePacket-style payload used by streaming.
        width = getattr(image_data, "width", None)
        height = getattr(image_data, "height", None)
        data_obj = getattr(image_data, "data", None)
        try:
            width_int = int(width)
            height_int = int(height)
        except (TypeError, ValueError) as exc:
            raise ValueError("Snapshot image dimensions are missing or invalid") from exc
        if width_int <= 0 or height_int <= 0:
            raise ValueError("Snapshot image dimensions must be positive")

        # Validation/transform: convert raw bytes-like payloads into the grayscale frame buffer.
        if isinstance(data_obj, memoryview):
            frame_bytes = data_obj.tobytes()
        elif isinstance(data_obj, (bytes, bytearray)):
            frame_bytes = bytes(data_obj)
        else:
            raise ValueError("Snapshot image payload is not bytes-like")

        expected_size = width_int * height_int
        if len(frame_bytes) < expected_size:
            raise ValueError("Snapshot image payload is incomplete")

        try:
            gray = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                (height_int, width_int)
            )
        except ValueError as exc:
            raise ValueError("Snapshot image payload could not be decoded") from exc

        # Encode with the same JPEG quality as the continuous image recorder.
        ok, buffer = cv2.imencode(
            ".jpg",
            gray,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        if not ok:
            raise IOError("Snapshot JPEG encoding failed")

        # Store the snapshot using the same frame filename pattern as continuous recording.
        output_path = os.path.join(session_dir, f"frame_000000_{int(ts_ms)}.jpg")
        with open(output_path, "wb") as file_handle:
            file_handle.write(buffer.tobytes())

        # Log the direct snapshot path so externally-controlled output remains visible.
        self._log_bmode_event(
            "snapshot_saved",
            level="INFO",
            target=output_path,
        )
        return output_path

    # Summary: Start streaming based on the selected stream option.
    # What it does: Chooses camera streaming or screen streaming, then starts the appropriate worker.
    # Input: `self`.
    # Returns: `None`.
    def _start_stream(self) -> None:
        """Start the selected stream source via the worker architecture."""
        stream_option = self.ui.comboBox_bmode_streamOption.currentText()
        if stream_option == "Stream Image":
            self._start_camera_stream()
        elif stream_option == "Stream Screen (This PC)":
            self._start_screen_stream()
        else:
            # Defensive guard: unknown option text should fail safely and guide the user.
            self._log_bmode_event(
                "error",
                level="ERROR",
                message=f"Unsupported stream option selected: {stream_option}",
            )
            QMessageBox.warning(
                self,
                "Unknown Stream Option",
                f"Unsupported stream option selected: {stream_option}",
            )

    # Summary: Start streaming frames from the selected camera.
    # What it does: Resolves the camera index, loads optional calibration crop data, and starts the camera worker.
    # Input: `self`.
    # Returns: `None`.
    def _start_camera_stream(self) -> None:
        """Begin camera streaming via the background worker thread."""
        # Read the selected camera index (stored as item data).
        selected_index = self.ui.comboBox_bmode_streamPort.currentData()
        if selected_index is None:
            selected_index = self.ui.comboBox_bmode_streamPort.currentIndex()

        # Read the optional calibration path so camera streaming can crop when a file is provided.
        calib_path = self.ui.lineEdit_bmode_calibPath.text().strip()
        camera_clip_rect: Optional[tuple[int, int, int, int]] = None
        if calib_path:
            # Validate and convert calibration clip values before starting the worker.
            try:
                camera_clip_rect = self._load_clip_rectangle_from_calibration_path(
                    calib_path
                )
            except ValueError as error:
                # Log calibration parsing failures so users can diagnose camera-start failures.
                self._log_bmode_event(
                    "warning",
                    level="WARN",
                    message=str(error),
                )
                QMessageBox.warning(
                    self,
                    "Calibration Error",
                    str(error),
                )
                return

        # Log the stream request context so source selection is visible in the text stream.
        self._log_bmode_event(
            "stream_request",
            level="INFO",
            action="start",
            source="camera",
            camera_index=int(selected_index),
            clip_rect=camera_clip_rect,
        )
        # Start the camera worker; it will emit state changes and frames via the proxy.
        self._camera_worker.start(
            int(selected_index),
            fps=1,
            clip_rect=camera_clip_rect,
        )

    # Summary: Start streaming a calibrated region from the local screen.
    # What it does: Requires calibration rectangle data, asks the user which screen to use, then starts the worker.
    # Input: `self`.
    # Returns: `None`.
    def _start_screen_stream(self) -> None:
        """Begin streaming a cropped region from the local screen."""
        if not self._screen_rect:
            # Log missing calibration so screen-mode prerequisites are explicit.
            self._log_bmode_event(
                "warning",
                level="WARN",
                message="Please load a calibration file before streaming the screen.",
            )
            QMessageBox.warning(
                self,
                "Calibration Required",
                "Please load a calibration file before streaming the screen.",
            )
            return

        screen = self._select_screen_for_stream()
        if screen is None:
            if not QGuiApplication.screens():
                # Log no-screen errors so missing display state is preserved for troubleshooting.
                self._log_bmode_event(
                    "error",
                    level="ERROR",
                    message="No screen was detected. Cannot start screen streaming.",
                )
                QMessageBox.warning(
                    self,
                    "Screen Error",
                    "No screen was detected. Cannot start screen streaming.",
                )
            return

        # Log the selected screen and clip details before stream startup.
        geometry = screen.geometry()
        self._log_bmode_event(
            "stream_request",
            level="INFO",
            action="start",
            source="screen",
            screen_name=screen.name(),
            screen_size=f"{geometry.width()}x{geometry.height()}",
            clip_rect=self._screen_rect,
        )
        # Start the screen worker; it runs on the UI thread via QTimer.
        self._screen_worker.start(screen, self._screen_rect, fps=33)

    # Summary: Ask the user which monitor/screen to capture.
    # What it does: Builds a small dropdown list of available screens and returns the chosen QScreen.
    # Input: `self`.
    # Returns: A `QScreen` if the user picks one; otherwise `None`.
    def _select_screen_for_stream(self) -> Optional[QScreen]:

        # Ask Qt for the list of screens currently connected to the machine.
        # Qt will return one QScreen object per monitor.
        screens = QGuiApplication.screens()

        # If Qt returns an empty list, something is wrong (e.g. no display available),
        # so we can't capture anything.
        if not screens:
            return None

        # Prefer the OS "primary" monitor as the default.
        # As a safety fallback, if primaryScreen() is None for some reason,
        # use the first screen in the list.
        primary = QGuiApplication.primaryScreen() or screens[0]

        # If there is only one monitor, there's nothing to choose from.
        # Just return the default (primary) screen.
        if len(screens) == 1:
            return primary

        # Build a list of human-friendly labels to show in the dropdown dialog.
        labels: list[str] = []

        # Keep a mapping from label -> QScreen so when the user selects a label,
        # we can get back the corresponding QScreen object.
        label_to_screen: dict[str, QScreen] = {}

        # Track which dropdown item should be selected by default.
        # We start with 0, then update it when we see the primary screen.
        primary_index = 0

        # Create one dropdown item per screen.
        # enumerate(..., start=1) makes the labels "Screen 1", "Screen 2", etc.
        for idx, screen in enumerate(screens, start=1):

            # Read the screen geometry so we can display the resolution to the user.
            geometry = screen.geometry()

            # Compose a simple label with:
            # - a user-friendly index
            # - the OS/Qt screen name
            # - the screen resolution (width x height)
            label = f"Screen {idx}: {screen.name()} ({geometry.width()}x{geometry.height()})"

            # If this is the primary screen, mark it in the UI and also remember
            # its position so it shows as the default selection in the dialog.
            if screen == primary:
                label = f"{label} [Primary]"
                primary_index = len(labels)

            # Add the label to the options list shown in the dialog.
            labels.append(label)

            # Store the mapping so we can look up the chosen screen later.
            label_to_screen[label] = screen

        # Show a very simple built-in Qt dialog with a dropdown list.
        # It returns:
        # - selection: the chosen label string
        # - accepted: True if user clicked OK, False if they clicked Cancel/closed it
        selection, accepted = QInputDialog.getItem(
            self,  # parent widget (keeps the dialog centered and modal to this window)
            "Select Screen",  # window title
            "Choose the screen to capture:",  # message text
            labels,  # dropdown items
            primary_index,  # default selected item index
            False,  # editable=False (user must pick one of the options)
        )

        # If the user cancels the dialog, treat it as "no screen selected".
        # The caller can decide what to do (e.g., don't start streaming).
        if not accepted:
            return None

        # Convert the selected label back into the QScreen object.
        # If something unexpected happens (e.g. label not found), fall back to primary.
        return label_to_screen.get(selection, primary)

    # Summary: Slot function that caches the latest frame packet from the proxy.
    # What it does: Stores the packet for the display timer without doing heavy work on the signal.
    # Input: `self`, `packet` (FramePacket).
    # Returns: `None`.
    def _on_bmodeStreamProxy_frame_ready(self, packet: FramePacket) -> None:
        # Cache the packet so rendering can happen on the display timer.
        self._latest_frame_packet = packet
        # Count incoming frames so periodic health logs can report real observed cadence.
        self._health_window_frame_count += 1
        self._log_stream_health()
        # Convert the FramePacket fields to the agreed image packet naming.
        image_ts_ms = int(packet.timestamp_ms)
        image_data = packet
        # Cache the latest image packet for consumers that need pull-style access.
        self._latest_image_ts_ms = image_ts_ms
        self._latest_image_data = image_data
        # Emit the packet so the coupling controller can match image+rigidbody samples.
        self.sig_image_packet.emit(image_ts_ms, image_data)
        # Avoid per-frame recording while external coupling is driving snapshots.
        if self._external_recording_active:
            return
        # Enqueue the frame for recording without blocking the UI thread.
        self._image_record_worker.handle_frame(packet)

    # Summary: Slot function that reacts to stream state changes and updates the UI.
    # What it does: Updates buttons/inputs, starts or stops the display timer, and clears the image on stop.
    # Input: `self`, `is_running` (bool), `message` (str).
    # Returns: `None`.
    def _on_bmodeStreamProxy_state_changed(self, is_running: bool, message: str) -> None:
        # Log every stream state transition so start/stop reasons stay visible after dialogs close.
        self._log_bmode_event(
            "stream_state",
            level="INFO" if is_running else "WARN",
            is_running=is_running,
            reason=message,
        )

        if is_running:
            # Update state so the open button toggles to stop.
            self._is_streaming = True
            # Reset health counters and timing gates at stream start for a clean session baseline.
            self._health_window_start_ts = time.monotonic()
            self._health_window_frame_count = 0
            self._last_health_log_ts = 0.0
            # UI state change: show stop intent while streaming.
            self.ui.pushButton_bmode_openStream.setText("Stop Stream")
            # UI state change: lock inputs so settings are stable while streaming.
            self.ui.comboBox_bmode_streamOption.setEnabled(False)
            self.ui.comboBox_bmode_streamPort.setEnabled(False)
            self.ui.pushButton_bmode_calibBrowse.setEnabled(False)
            self.ui.pushButton_bmode_calibClear.setEnabled(False)
            self.ui.lineEdit_bmode_calibPath.setEnabled(False)
            # Start the display timer to render cached frames on the UI thread.
            self._display_timer.start()
            return

        # Stop recording if the stream ended so we do not write stale frames.
        if self._image_record_worker.is_active():
            self._stop_recording("Recording stopped because streaming ended.")

        # Update state to idle so the open button toggles back to start.
        self._is_streaming = False
        # UI state change: restore the button label after streaming stops.
        self.ui.pushButton_bmode_openStream.setText("Open Stream")
        # UI state change: re-enable inputs now that streaming is idle.
        self.ui.comboBox_bmode_streamOption.setEnabled(True)
        # Respect the current option when re-enabling the port selector.
        self.ui.comboBox_bmode_streamPort.setEnabled(
            self.ui.comboBox_bmode_streamOption.currentText()
            != "Stream Screen (This PC)"
        )
        self.ui.pushButton_bmode_calibBrowse.setEnabled(True)
        self.ui.pushButton_bmode_calibClear.setEnabled(True)
        self.ui.lineEdit_bmode_calibPath.setEnabled(True)
        # Stop rendering so the UI stays quiet while idle.
        self._display_timer.stop()
        # Clear cached frame data so we do not display stale images.
        self._latest_frame_packet = None
        self._latest_image_ts_ms = None
        self._latest_image_data = None
        # Reset health counters and timing gate so idle periods do not affect the next stream window.
        self._health_window_start_ts = time.monotonic()
        self._health_window_frame_count = 0
        self._last_health_log_ts = 0.0
        self.ui.label_bmode_image.clear()

    # Summary: Slot function that shows worker errors on the UI thread.
    # What it does: Displays a warning dialog and stops streaming cleanly.
    # Input: `self`, `message` (str).
    # Returns: `None`.
    def _on_bmodeStreamProxy_error_message(self, message: str) -> None:
        # Log stream errors to the text stream so failures remain visible after modal dialogs close.
        if message:
            self._log_bmode_event("error", level="ERROR", message=message)
        # Show the warning on the UI thread so it is safe and visible.
        if message:
            QMessageBox.warning(self, "Stream Error", message)
        # Stop streaming so the UI resets after an error.
        self._stop_stream()

    # Summary: Slot function that stops recording when the worker reports a failure.
    # What it does: Stops the writer thread on the UI thread and surfaces the reason once.
    # Input: `self`, `reason` (str).
    # Returns: `None`.
    def _on_bmodeStreamProxy_record_stop(self, reason: str) -> None:
        # Log record-stop reasons even when recording is already inactive.
        if reason:
            self._log_bmode_event("record_stop", level="WARN", reason=reason)
        # Stop recording safely on the UI thread when an error or overload occurs.
        if self._image_record_worker.is_active():
            self._stop_recording(reason)
        elif reason:
            # Still surface the message even if recording already stopped.
            QMessageBox.warning(self, "Recording Stopped", reason)

    # Summary: Slot function that surfaces non-fatal recording messages on the UI thread.
    # What it does: Stores the latest message as a tooltip so the UI stays non-blocking.
    # Input: `self`, `message` (str).
    # Returns: `None`.
    def _on_bmodeStreamProxy_record_message(self, message: str) -> None:
        # Avoid modal dialogs for transient recording warnings.
        if message:
            # Log non-fatal recording warnings as extra text feedback for the user.
            self._log_bmode_event("warning", level="WARN", message=message)
            self.ui.pushButton_bmode_recordStream.setToolTip(message)

    # Summary: Slot function that renders the latest cached frame on a timer.
    # What it does: Delegates to the render helper so heavy work stays on a controlled interval.
    # Input: `self`.
    # Returns: `None`.
    def _on_bmodeDisplayTimer_timeout(self) -> None:
        # Render the latest cached packet on the UI thread.
        self._render_latest_frame()

    # Summary: Render the cached frame packet into the image label.
    # What it does: Builds a QImage from raw bytes, detaches the memory, and displays it.
    # Input: `self`.
    # Returns: `None`.
    def _render_latest_frame(self) -> None:
        # Skip rendering when no packet has arrived yet.
        if self._latest_frame_packet is None:
            return

        packet = self._latest_frame_packet
        # Default to grayscale rendering because the stream payload is now single-channel uint8.
        qimage_format = QImage.Format_Grayscale8
        bytes_per_line = packet.width
        expected_size = bytes_per_line * packet.height

        # Keep RGB fallback support so older/legacy packets still display correctly.
        if packet.fmt == "rgb888":
            qimage_format = QImage.Format_RGB888
            bytes_per_line = packet.width * 3
            expected_size = bytes_per_line * packet.height

        if len(packet.data) < expected_size:
            # Ignore incomplete packets to avoid rendering garbage.
            return

        # Create a QImage from the raw bytes using the packet format.
        qimage = QImage(
            packet.data,
            packet.width,
            packet.height,
            bytes_per_line,
            qimage_format,
        )
        # Detach the QImage from the raw buffer so it stays valid after this method.
        qimage = qimage.copy()
        pixmap = QPixmap.fromImage(qimage)
        self._display_pixmap(pixmap)

    # Summary: Show a pixmap in the UI label.
    # What it does: Scales the pixmap to fit the label while keeping the aspect ratio.
    # Input: `self`, `pixmap` (the image to show).
    # Returns: `None`.
    def _display_pixmap(self, pixmap: QPixmap) -> None:
        # Use the label content rect (inside stylesheet border) to prevent size-hint feedback loops.
        target_size = self.ui.label_bmode_image.contentsRect().size()
        # Fall back to full label size if content rect is transiently empty during layout updates.
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = self.ui.label_bmode_image.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            # Skip drawing until the widget has a valid render area.
            return

        # Scale the pixmap to the available content area while keeping the image aspect ratio.
        scaled = pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.ui.label_bmode_image.setPixmap(scaled)

    # Summary: Stop streaming and reset the UI.
    # What it does: Stops workers and timers, clears cached frames, and restores UI controls.
    # Input: `self`.
    # Returns: `None`.
    def _stop_stream(self) -> None:
        # Stop the workers so acquisition ends in both camera and screen paths.
        self._camera_worker.stop()
        self._screen_worker.stop()
        # Stop rendering so the UI does not redraw stale frames.
        self._display_timer.stop()
        # Reset the stream state so the open/stop button toggles correctly.
        self._is_streaming = False
        # UI state change: restore the button label after stopping.
        self.ui.pushButton_bmode_openStream.setText("Open Stream")
        # UI state change: re-enable inputs now that streaming has stopped.
        self.ui.comboBox_bmode_streamOption.setEnabled(True)
        self.ui.comboBox_bmode_streamPort.setEnabled(
            self.ui.comboBox_bmode_streamOption.currentText()
            != "Stream Screen (This PC)"
        )
        self.ui.pushButton_bmode_calibBrowse.setEnabled(True)
        self.ui.pushButton_bmode_calibClear.setEnabled(True)
        self.ui.lineEdit_bmode_calibPath.setEnabled(True)
        # Clear cached frame data and the image label.
        self._latest_frame_packet = None
        self._latest_image_ts_ms = None
        self._latest_image_data = None
        # Reset health counters and timing gate so stale state does not leak across stream sessions.
        self._health_window_start_ts = time.monotonic()
        self._health_window_frame_count = 0
        self._last_health_log_ts = 0.0
        self.ui.label_bmode_image.clear()

    # Summary: Handle the window close event.
    # What it does: Stops streaming so workers and timers are cleaned up before the app closes.
    # Input: `self`, `event` (the Qt close event object).
    # Returns: `None`.
    def closeEvent(self, event) -> None:
        # Log window-close cleanup so forced stream/record shutdown has visible feedback.
        self._log_bmode_event(
            "info",
            level="INFO",
            message="Window closing. Cleaning up stream and recording.",
        )
        # Stop recording so the writer thread can close cleanly.
        if self._image_record_worker.is_active():
            self._stop_recording("Recording stopped because the window closed.")
        # Ensure worker threads and timers are stopped when the window closes.
        self._stop_stream()
        # Call the base class close handler after our cleanup.
        super().closeEvent(event)
        event.accept()


# Summary: App entry point for the B-mode streaming UI.
# What it does: Creates the QApplication, shows the main window, and starts the Qt event loop.
# Input: No arguments; reads `sys.argv` for Qt options.
# Returns: `None` (the process exits via `sys.exit(...)`).
def main() -> None:
    """Create the QApplication and show the V2 window."""
    # Create the Qt application and pass through any command-line args.
    app = QApplication(sys.argv)
    # Instantiate and show the main window that renders the V2 form.
    window = BModeWidget()
    window.show()
    # Enter the Qt event loop and exit the process when it ends.
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
