"""Controller for software coupling between image packets and rigid body packets."""

# Standard library helpers for bounded buffering.
from collections import deque
from typing import Any

# Qt base classes and signal/slot primitives.
from PySide6.QtCore import QObject, Signal, Slot


# Summary:
# - Lightweight QObject that couples image packets with mocap packets using sample-and-hold.
# - What it does: buffers recent rigid body packets, receives image packets, and emits a
#   coupled packet only when the newest rigid body sample at-or-before the image timestamp
#   exists and is within the configured stale threshold.
# - Input: constructed with max allowed timestamp difference and buffer length.
# - Returns: QObject instance that emits coupled packet signals for recording.
class CoupledStreamController(QObject):
    # Signal: emits one accepted coupled packet as
    # (image_ts_ms, image_data, rigidbody_ts_ms, rigidbody_data, diff_ms).
    sig_coupled_packet = Signal(int, object, int, object, int)
    # Signal: emits running coupling counters for UI/debug visibility.
    sig_stats = Signal(dict)

    # Summary:
    # - Initialize the coupling controller state.
    # - What it does: validates config values, prepares the mocap ring buffer, and
    #   initializes counters plus recording state.
    # - Input: `self`, `maxdiff_imagepose_ms` (int), `mocap_buffer_maxlen` (int, optional).
    # - Returns: None.
    def __init__(self, maxdiff_imagepose_ms: int, mocap_buffer_maxlen: int = 500) -> None:
        super().__init__()
        # Give the controller a stable objectName so slot names stay convention-friendly.
        self.setObjectName("coupledStreamController")

        # Store coupling thresholds with defensive lower bounds.
        self._maxdiff_imagepose_ms = max(0, int(maxdiff_imagepose_ms))
        buffer_maxlen = max(1, int(mocap_buffer_maxlen))

        # Keep a bounded history of recent rigid body packets for reverse scan matching.
        self._mocap_buf: deque[tuple[int, Any]] = deque(maxlen=buffer_maxlen)

        # Recording gate: when False, image packets are ignored by the coupler.
        self._is_recording = False

        # Counters for coupled and dropped packets, exposed via `sig_stats`.
        self._count_coupled = 0
        self._count_dropped_no_pose = 0
        self._count_dropped_stale = 0

    # Summary:
    # - Return whether the controller is currently accepting image packets for coupling.
    # - Input: `self`.
    # - Returns: True when recording is active (bool).
    def is_recording(self) -> bool:
        return self._is_recording

    # Summary:
    # - Start coupling for recording and reset counters.
    # - What it does: enables the recording gate and resets packet/drop statistics.
    # - Input: `self`.
    # - Returns: None.
    def start_recording(self) -> None:
        # UI-state style change: arm the coupler so incoming image packets are processed.
        # This will allow the gate keeper (see _on_coupledStreamController_image_packet)
        # process the coupled recording.
        self._is_recording = True

        # Reset counters at each session start so stats are session-local.
        self._count_coupled = 0
        self._count_dropped_no_pose = 0
        self._count_dropped_stale = 0
        self._emit_stats()

    # Summary:
    # - Stop coupling for recording.
    # - What it does: disables the recording gate without blocking work.
    # - Input: `self`.
    # - Returns: None.
    def stop_recording(self) -> None:
        # UI-state style change: disarm the coupler so image packets are ignored.
        self._is_recording = False
        self._emit_stats()

    # Summary:
    # - Build and emit current coupling counters.
    # - What it does: sends a small stats dictionary for UI/log visibility.
    # - Input: `self`.
    # - Returns: None.
    def _emit_stats(self) -> None:
        # Emit counters plus session state so UI can render one concise debug line.
        self.sig_stats.emit(
            {
                "is_recording": self._is_recording,
                "count_coupled": self._count_coupled,
                "count_dropped_no_pose": self._count_dropped_no_pose,
                "count_dropped_stale": self._count_dropped_stale,
                "mocap_buffer_size": len(self._mocap_buf),
                "maxdiff_imagepose_ms": self._maxdiff_imagepose_ms,
            }
        )

    # Summary:
    # - Slot function that receives rigid body packets and appends them to the rolling buffer.
    # - What it does: stores recent Qualisys/QTM rigid body samples so later image packets can
    #   find the newest pose at-or-before the image timestamp. Rigid body packets can arrive
    #   faster or more often than ultrasound image packets, so this slot does not perform the
    #   coupling itself; it only keeps a bounded history of mocap data for lookup.
    # - Input: `self`, `rigidbody_ts_ms` (int), `rigidbody_data` (object).
    # - Returns: None.
    @Slot(int, object)
    def _on_coupledStreamController_rigidbody_packet(
        self, rigidbody_ts_ms: int, rigidbody_data: object
    ) -> None:
        # Keep the mocap buffer updated even when recording is not active.
        # The image packet is the event that creates a coupled record (see function below), 
        # so the gate lives in the image slot. Mocap packets are only cached here as recent 
        # lookup data. This keeps the buffer "warm" during live preview, so the first image 
        # after the user presses Record can immediately find a recent pose instead of waiting for
        # a new mocap packet. The deque maxlen still prevents memory from growing forever.
        self._mocap_buf.append((int(rigidbody_ts_ms), rigidbody_data))

    # Summary:
    # - Slot function that receives image packets and performs the actual image-to-mocap coupling.
    # - What it does: treats each image packet as the trigger for one coupling attempt, searches
    #   the rolling rigid body buffer, selects the newest rigid body sample with ts <= image ts,
    #   validates its age via `maxdiff_imagepose_ms`, and emits an accepted coupled packet.
    # - Input: `self`, `image_ts_ms` (int), `image_data` (object).
    # - Returns: None.
    @Slot(int, object)
    def _on_coupledStreamController_image_packet(
        self, image_ts_ms: int, image_data: object
    ) -> None:
        # Gate image-triggered coupling so live preview does not become a recording.
        # The signal connections are active as long as the streams are running, so image
        # packets can arrive before the user presses Record. Without this guard, every
        # preview image would search the mocap buffer, emit coupled packets, update
        # counters, and possibly reach downstream writers. The gate keeps recording
        # session data intentional while still allowing mocap data to stay ready.
        if not self._is_recording:
            return

        selected_rigidbody_ts_ms = None
        selected_rigidbody_data = None

        # Scan newest-to-oldest so we pick the freshest valid sample for sample-and-hold.
        for rigidbody_ts_ms, rigidbody_data in reversed(self._mocap_buf):
            # Keep only rigid body packets at-or-before the image timestamp.
            if rigidbody_ts_ms <= image_ts_ms:
                selected_rigidbody_ts_ms = rigidbody_ts_ms
                selected_rigidbody_data = rigidbody_data
                break

        # Drop when no at-or-before rigid body sample exists.
        if selected_rigidbody_ts_ms is None:
            self._count_dropped_no_pose += 1
            self._emit_stats()
            return

        # Compute signed difference in milliseconds for threshold validation.
        diff_ms = int(image_ts_ms) - int(selected_rigidbody_ts_ms)
        # Defensive guard: negative diffs violate sample-and-hold ordering.
        if diff_ms < 0:
            self._count_dropped_no_pose += 1
            self._emit_stats()
            return

        # Reject stale rigid body packets that are too old for this image.
        if diff_ms > self._maxdiff_imagepose_ms:
            self._count_dropped_stale += 1
            self._emit_stats()
            return

        # Accept and emit the coupled packet for downstream recording.
        self._count_coupled += 1
        self.sig_coupled_packet.emit(
            int(image_ts_ms),
            image_data,
            int(selected_rigidbody_ts_ms),
            selected_rigidbody_data,
            diff_ms,
        )
        self._emit_stats()

    # Public alias: keep requested method names for direct signal wiring call sites.
    on_rigidbody_packet = _on_coupledStreamController_rigidbody_packet
    on_image_packet = _on_coupledStreamController_image_packet
