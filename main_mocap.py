"""PySide6 widget to stream Qualisys 6D residual data without blocking the UI."""

# Standard library imports for async streaming, background threads, and numeric helpers.
import asyncio
import csv
import ipaddress
import math
import os
import queue
import threading
import time
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timezone
from typing import Optional

# Third-party imports for QTM real-time streaming, numeric transforms, plotting, and Qt UI.
import qtm_rt
import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Qt, QTimer, QSize, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

# Generated UI class from Qt Designer (do not edit the _ui.py file).
from ui.mocap_ui import Ui_Form as Ui_Mocap

QTM_PORT = 22223
QTM_PROTOCOL_VERSION = "1.20"

# Recording constants to keep the writer thread responsive and avoid unbounded memory use.
RECORD_QUEUE_MAXSIZE = 500
RECORD_FLUSH_EVERY = 50
RECORD_DROP_WARN_INTERVAL = 1.0
RECORD_DROP_WINDOW_SECONDS = 2.0
RECORD_DROP_RATE_THRESHOLD = 0.05
RECORD_NAN = float("nan")


# Summary:
# - Build a monotonic timestamp in milliseconds for mocap packets.
# - What it does: Uses `time.monotonic()` to keep rigid body timestamps on the same
#   stable clock source as image timestamps used for coupling.
# - Input: None.
# - Returns: Monotonic milliseconds (int).
def _now_ms() -> int:
    # Use a monotonic clock so packet deltas stay valid if wall clock time changes.
    return int(time.monotonic() * 1000)


# Summary:
# - Matplotlib canvas that renders a live 3D scatter plot for mocap body positions.
# - What it does: owns the Figure, a 3D Axes, and a single scatter handle that is updated in place.
class Mocap3DCanvas(FigureCanvas):
    # Base transform to map QTM's Y-up world into Matplotlib's Z-up world.
    T_base = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # Summary:
    # - Initialize the 3D plot with labels and an empty scatter handle.
    # - Input: `self`, `parent` (QWidget | None).
    # - Returns: None.
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        # Build the figure and axes once so we can update without clearing every frame.
        figure = Figure()
        axes = figure.add_subplot(111, projection="3d")
        # UI state change: enable tight layout so axis labels are not clipped in embedded Qt layouts.
        figure.set_tight_layout(True)
        super().__init__(figure)

        # Keep a reference to the axes for fast updates later.
        self._axes = axes

        # Create the scatter handle once so updates only move existing points.
        self._scatter = self._axes.scatter([], [], [], s=20)

        # Track text labels so each point can show its rigid body name.
        self._label_texts: list = []
        # Keep one quiver artist per axis so we can replace orientation arrows per frame.
        self._axis_quivers: list = []
        # Keep one quiver artist per axis for the world origin reference frame.
        self._origin_axis_quivers: list = []
        # Keep local-axis arrows at a fixed world length so orientation remains readable.
        self._axis_length_mm = 50.0
        # Use a compact arrowhead to reduce clutter when many rigid bodies are visible.
        self._axis_arrow_ratio = 0.25
        # Keep axis colors consistent with robotics convention (X=red, Y=green, Z=blue).
        self._axis_colors = {"x": "red", "y": "green", "z": "blue"}
        # Keep a persistent point marker at the world origin for orientation context.
        self._origin_point = self._axes.scatter(
            [0.0], [0.0], [0.0], s=30, c="black", depthshade=False
        )

        # Label axes so users know the coordinate directions (Z is up after the base transform).
        self._axes.set_xlabel("X")
        self._axes.set_ylabel("Y")
        self._axes.set_zlabel("Z (Up)")
        # Hide all 3D panes so only points/labels and axes remain visible.
        self._axes.xaxis.pane.set_visible(False)
        self._axes.yaxis.pane.set_visible(False)
        self._axes.zaxis.pane.set_visible(False)
        # Set the axis to be equal and orthogonal 
        self._axes.set_aspect("equal")
        self._axes.set_proj_type("ortho")

        # UI state change: lock the plot to a fixed world box so live points stay in a stable frame.
        self._apply_fixed_bounds()
        # Draw the global origin frame once so users always have a fixed reference axis.
        self._draw_origin_reference_axes()

        # Ensure the canvas is parented to the widget tree if provided.
        if parent is not None:
            self.setParent(parent)

    # Summary:
    # - Apply fixed world-coordinate bounds to keep the view stable during live streaming.
    # - Input: `self`.
    # - Returns: None.
    def _apply_fixed_bounds(self) -> None:
        # Disable autoscale so incoming points never override the fixed world box.
        self._axes.set_autoscale_on(False)
        # Keep origin centered in X/Y and at the bottom of Z, using QTM millimeter units.
        self._axes.set_xlim(-500.0, 500.0)
        self._axes.set_ylim(-500.0, 500.0)
        self._axes.set_zlim(0.0, 1000.0)

    # Summary:
    # - Ensure the label text artists match the number of rigid body labels.
    # - Input: `self`, `labels` (list[str]).
    # - Returns: None.
    def _sync_label_artists(self, labels: list[str]) -> None:
        # Remove extra label artists so stale labels are not shown.
        while len(self._label_texts) > len(labels):
            text_artist = self._label_texts.pop()
            # Remove from the axes so it is no longer drawn.
            text_artist.remove()

        # Add missing label artists so every body has a text placeholder.
        while len(self._label_texts) < len(labels):
            label = labels[len(self._label_texts)]
            # Create the text in a neutral location; we move it per-frame.
            # Create text without a fixed 3D direction so it stays horizontal.
            text_artist = self._axes.text(0.0, 0.0, 0.0, label, fontsize=9, rotation=0)
            self._label_texts.append(text_artist)

    # Summary:
    # - Remove all current orientation quiver artists from the 3D axes.
    # - Input: `self`.
    # - Returns: None.
    def _clear_axis_quivers(self) -> None:
        # Remove quivers before re-drawing so old orientations never linger on screen.
        while self._axis_quivers:
            quiver_artist = self._axis_quivers.pop()
            # Defensive cleanup: if Matplotlib already detached an artist, ignore the remove failure.
            try:
                quiver_artist.remove()
            except ValueError:
                pass

    # Summary:
    # - Draw a fixed world-origin axis triad that stays visible in idle and streaming states.
    # - Input: `self`.
    # - Returns: None.
    def _draw_origin_reference_axes(self) -> None:
        # Remove old origin-axis artists first in case this helper is called more than once.
        while self._origin_axis_quivers:
            origin_axis_artist = self._origin_axis_quivers.pop()
            # Defensive cleanup: ignore remove errors when Matplotlib already detached the artist.
            try:
                origin_axis_artist.remove()
            except ValueError:
                pass

        # Keep a single static origin point so the world frame anchor is always visible.
        self._origin_point._offsets3d = ([0.0], [0.0], [0.0])
        # Build fixed direction vectors for the world origin frame in the plot coordinate system.
        origin_x_values = [0.0]
        origin_y_values = [0.0]
        origin_z_values = [0.0]
        # Draw the world X axis (red) with the same scale as rigid-body local axes.
        self._origin_axis_quivers.append(
            self._axes.quiver(
                origin_x_values,
                origin_y_values,
                origin_z_values,
                [1.0],
                [0.0],
                [0.0],
                length=self._axis_length_mm,
                normalize=True,
                color=self._axis_colors["x"],
                arrow_length_ratio=self._axis_arrow_ratio,
            )
        )
        # Draw the world Y axis (green) with the same scale as rigid-body local axes.
        self._origin_axis_quivers.append(
            self._axes.quiver(
                origin_x_values,
                origin_y_values,
                origin_z_values,
                [0.0],
                [1.0],
                [0.0],
                length=self._axis_length_mm,
                normalize=True,
                color=self._axis_colors["y"],
                arrow_length_ratio=self._axis_arrow_ratio,
            )
        )
        # Draw the world Z axis (blue) with the same scale as rigid-body local axes.
        self._origin_axis_quivers.append(
            self._axes.quiver(
                origin_x_values,
                origin_y_values,
                origin_z_values,
                [0.0],
                [0.0],
                [1.0],
                length=self._axis_length_mm,
                normalize=True,
                color=self._axis_colors["z"],
                arrow_length_ratio=self._axis_arrow_ratio,
            )
        )

    # Summary:
    # - Convert a transformed 4x4 pose into quiver origin and local X/Y/Z axis vectors.
    # - Input: `self`, `pose_plot` (np.ndarray).
    # - Returns: `(origin, x_axis, y_axis, z_axis)` when finite and drawable, otherwise `None`.
    def _collect_axis_vectors_from_pose(
        self, pose_plot: np.ndarray
    ) -> Optional[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ]:
        # Validation/transforms: only 4x4 finite poses can be safely converted to orientation arrows.
        if not isinstance(pose_plot, np.ndarray) or pose_plot.shape != (4, 4):
            return None

        # Read the point origin and local basis vectors directly from the homogeneous transform.
        origin_vector = pose_plot[:3, 3]
        x_axis_vector = pose_plot[:3, 0]
        y_axis_vector = pose_plot[:3, 1]
        z_axis_vector = pose_plot[:3, 2]

        # Validation/transforms: skip arrows when any origin or axis component is non-finite.
        if not (
            np.isfinite(origin_vector).all()
            and np.isfinite(x_axis_vector).all()
            and np.isfinite(y_axis_vector).all()
            and np.isfinite(z_axis_vector).all()
        ):
            return None

        # Validation/transforms: skip degenerate axes to avoid undefined arrow directions.
        if (
            np.linalg.norm(x_axis_vector) <= 1e-12
            or np.linalg.norm(y_axis_vector) <= 1e-12
            or np.linalg.norm(z_axis_vector) <= 1e-12
        ):
            return None

        return (
            (float(origin_vector[0]), float(origin_vector[1]), float(origin_vector[2])),
            (float(x_axis_vector[0]), float(x_axis_vector[1]), float(x_axis_vector[2])),
            (float(y_axis_vector[0]), float(y_axis_vector[1]), float(y_axis_vector[2])),
            (float(z_axis_vector[0]), float(z_axis_vector[1]), float(z_axis_vector[2])),
        )

    # Summary:
    # - Update the scatter points using the latest QTM poses.
    # - Input: `self`, `poses_qtm` (dict[str, np.ndarray | None]).
    # - Returns: None.
    def update_poses(self, poses_qtm: dict[str, Optional[np.ndarray]]) -> None:
        # Skip updates when no poses have been cached yet.
        if poses_qtm is None:
            return

        # Keep label order aligned with dict insertion order so points and names match.
        labels = list(poses_qtm.keys())
        # Sync label artists with the incoming labels before updating positions.
        self._sync_label_artists(labels)

        # Keep only finite values for scatter updates because mixed NaN/finite arrays
        # can cause unstable 3D rendering on early frames.
        x_values: list[float] = []
        y_values: list[float] = []
        z_values: list[float] = []
        # Collect per-body local axes so we can draw quivers after scatter updates.
        axis_origins: list[tuple[float, float, float]] = []
        x_axis_values: list[tuple[float, float, float]] = []
        y_axis_values: list[tuple[float, float, float]] = []
        z_axis_values: list[tuple[float, float, float]] = []

        # Convert each pose into scatter arrays while keeping insertion order.
        for index, (label, pose_qtm) in enumerate(poses_qtm.items()):
            # Grab the matching text artist so each point can show its label.
            label_artist = self._label_texts[index]

            # Treat missing entries as invalid and hide the label for this body.
            if pose_qtm is None or not isinstance(pose_qtm, np.ndarray):
                # Hide the label because there is no visible point.
                label_artist.set_visible(False)
                continue

            # Ensure we only process valid homogeneous transforms.
            if pose_qtm.shape != (4, 4):
                # Hide the label because the pose is not drawable.
                label_artist.set_visible(False)
                continue

            # Apply the fixed base transform (left-multiplication changes world frame to plot frame).
            pose_plot = self.T_base @ pose_qtm
            x_value, y_value, z_value = pose_plot[:3, 3]

            # Validation/transforms: only use finite coordinates for both scatter and labels
            # so missing bodies never suppress valid bodies in the same frame.
            if (
                math.isfinite(x_value)
                and math.isfinite(y_value)
                and math.isfinite(z_value)
            ):
                x_values.append(float(x_value))
                y_values.append(float(y_value))
                z_values.append(float(z_value))
                # Update the label position so it follows the visible point.
                label_artist.set_text(label)
                label_artist.set_position((float(x_value), float(y_value)))
                # Keep labels horizontal by avoiding a 3D direction alignment.
                label_artist.set_3d_properties(float(z_value), zdir=None)
                label_artist.set_visible(True)
                # Validation/transforms: only store orientation vectors when the full pose is drawable.
                axis_vectors = self._collect_axis_vectors_from_pose(pose_plot)
                if axis_vectors is not None:
                    origin_value, x_axis_value, y_axis_value, z_axis_value = axis_vectors
                    axis_origins.append(origin_value)
                    x_axis_values.append(x_axis_value)
                    y_axis_values.append(y_axis_value)
                    z_axis_values.append(z_axis_value)
            else:
                # Hide the label when the values are not finite.
                label_artist.set_visible(False)

        # Update the existing scatter in place instead of recreating it.
        self._scatter._offsets3d = (x_values, y_values, z_values)
        # Remove old orientation arrows before drawing the latest frame's quivers.
        self._clear_axis_quivers()
        # Draw local X/Y/Z axes for each visible rigid body using one quiver call per axis/color.
        if axis_origins:
            origin_x_values = [origin[0] for origin in axis_origins]
            origin_y_values = [origin[1] for origin in axis_origins]
            origin_z_values = [origin[2] for origin in axis_origins]
            x_u_values = [axis[0] for axis in x_axis_values]
            x_v_values = [axis[1] for axis in x_axis_values]
            x_w_values = [axis[2] for axis in x_axis_values]
            y_u_values = [axis[0] for axis in y_axis_values]
            y_v_values = [axis[1] for axis in y_axis_values]
            y_w_values = [axis[2] for axis in y_axis_values]
            z_u_values = [axis[0] for axis in z_axis_values]
            z_v_values = [axis[1] for axis in z_axis_values]
            z_w_values = [axis[2] for axis in z_axis_values]
            # Orientation quiver: local X (red) axis for all rigid bodies in the current frame.
            self._axis_quivers.append(
                self._axes.quiver(
                    origin_x_values,
                    origin_y_values,
                    origin_z_values,
                    x_u_values,
                    x_v_values,
                    x_w_values,
                    length=self._axis_length_mm,
                    normalize=True,
                    color=self._axis_colors["x"],
                    arrow_length_ratio=self._axis_arrow_ratio,
                )
            )
            # Orientation quiver: local Y (green) axis for all rigid bodies in the current frame.
            self._axis_quivers.append(
                self._axes.quiver(
                    origin_x_values,
                    origin_y_values,
                    origin_z_values,
                    y_u_values,
                    y_v_values,
                    y_w_values,
                    length=self._axis_length_mm,
                    normalize=True,
                    color=self._axis_colors["y"],
                    arrow_length_ratio=self._axis_arrow_ratio,
                )
            )
            # Orientation quiver: local Z (blue) axis for all rigid bodies in the current frame.
            self._axis_quivers.append(
                self._axes.quiver(
                    origin_x_values,
                    origin_y_values,
                    origin_z_values,
                    z_u_values,
                    z_v_values,
                    z_w_values,
                    length=self._axis_length_mm,
                    normalize=True,
                    color=self._axis_colors["z"],
                    arrow_length_ratio=self._axis_arrow_ratio,
                )
            )

        # Request a redraw without blocking the UI thread.
        self.draw_idle()

    # Summary:
    # - Clear the scatter points while keeping the fixed plot bounds.
    # - Input: `self`.
    # - Returns: None.
    def clear_points(self) -> None:
        # Empty the scatter data so the plot visibly clears on stop.
        self._scatter._offsets3d = ([], [], [])
        # Clear local-axis quivers so stop/start cycles never show stale orientation arrows.
        self._clear_axis_quivers()
        # Keep the origin point and global origin axis visible even when stream data is cleared.
        self._draw_origin_reference_axes()
        # Hide labels so the cleared plot has no lingering text.
        for label_artist in self._label_texts:
            label_artist.set_visible(False)
        # UI state change: re-apply fixed bounds so stop/start cycles keep the same world box.
        self._apply_fixed_bounds()
        # Schedule a redraw so the clear is visible to the user.
        self.draw_idle()


# Summary:
# - Qt signal bridge used to safely move data/state from the asyncio streaming thread into the Qt UI thread.
# - Why it exists: Qt widgets must only be updated on the main thread, but QTM packets arrive on a background
#   thread. This proxy provides Qt 'Signals' that the widget connects to, so updates are queued and delivered
#   in a thread-safe way.
class MocapStreamProxy(QObject):
    text_ready = Signal(str)
    poses_ready = Signal(int, object)
    stream_health_ready = Signal(int, int, int, float, bool)
    state_changed = Signal(bool, str)
    record_message = Signal(str)
    record_stop = Signal(str)
    record_row_ts_ms = Signal(object)

    # Summary:
    # - Initialize the proxy and assign a stable objectName for slot naming consistency.
    # - Input: `self`.
    # - Returns: None.
    def __init__(self) -> None:
        super().__init__()
        # Give the proxy a name so slot handlers can follow the naming convention.
        self.setObjectName("mocapStreamProxy")


# Summary:
# - Worker that owns CSV recording state and a background writer thread.
# - What it does: receives 6D residual results, formats rows, and writes them to disk without blocking.
class CsvRecordWorker:
    # Summary:
    # - Build a UTC epoch timestamp in milliseconds for CSV rows.
    # - Input: None.
    # - Returns: Milliseconds since Unix epoch at UTC (int).
    @staticmethod
    def _utc_epoch_ms() -> int:
        # Use UTC time so files are consistent across machines and time zones.
        now_utc = datetime.now(timezone.utc)
        # Convert to integer milliseconds to avoid floating-point timestamps.
        return int(now_utc.timestamp() * 1000)

    # Summary:
    # - Normalize a quaternion to unit length and provide a safe fallback.
    # - Input: `quat` (tuple[float, float, float, float]).
    # - Returns: Normalized quaternion as (w, x, y, z) (tuple[float, float, float, float]).
    @staticmethod
    def _normalize_quaternion(
        quat: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        # Compute the norm so we can avoid division by zero.
        w_value, x_value, y_value, z_value = quat
        norm = math.sqrt(
            (w_value * w_value)
            + (x_value * x_value)
            + (y_value * y_value)
            + (z_value * z_value)
        )
        if norm <= 0.0:
            # Fall back to identity when the norm is invalid.
            return (1.0, 0.0, 0.0, 0.0)
        return (w_value / norm, x_value / norm, y_value / norm, z_value / norm)

    # Summary:
    # - Convert a 3x3 rotation matrix into a stable quaternion (w, x, y, z).
    # - Input: `rotation_3x3` (list[list[float]]).
    # - Returns: Quaternion tuple (w, x, y, z) (tuple[float, float, float, float]).
    @staticmethod
    def _rotation_matrix_to_quaternion(
        rotation_3x3: list[list[float]],
    ) -> tuple[float, float, float, float]:
        # Use the classic stable branch selection based on the matrix trace.
        m00, m01, m02 = rotation_3x3[0]
        m10, m11, m12 = rotation_3x3[1]
        m20, m21, m22 = rotation_3x3[2]

        trace = m00 + m11 + m22
        if trace > 0.0:
            # Trace-positive path minimizes numerical error when the angle is small.
            scale = math.sqrt(trace + 1.0) * 2.0
            w_value = 0.25 * scale
            x_value = (m21 - m12) / scale
            y_value = (m02 - m20) / scale
            z_value = (m10 - m01) / scale
        elif m00 > m11 and m00 > m22:
            # Diagonal-major path for X dominance.
            scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w_value = (m21 - m12) / scale
            x_value = 0.25 * scale
            y_value = (m01 + m10) / scale
            z_value = (m02 + m20) / scale
        elif m11 > m22:
            # Diagonal-major path for Y dominance.
            scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w_value = (m02 - m20) / scale
            x_value = (m01 + m10) / scale
            y_value = 0.25 * scale
            z_value = (m12 + m21) / scale
        else:
            # Diagonal-major path for Z dominance.
            scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w_value = (m10 - m01) / scale
            x_value = (m02 + m20) / scale
            y_value = (m12 + m21) / scale
            z_value = 0.25 * scale

        # Normalize to unit length to keep downstream math stable.
        return CsvRecordWorker._normalize_quaternion(
            (w_value, x_value, y_value, z_value)
        )

    # Summary:
    # - Initialize the CSV recording worker and its state containers.
    # - Input: `self`, `proxy` (MocapStreamProxy).
    # - Returns: None.
    def __init__(self, proxy: MocapStreamProxy) -> None:
        # Store the proxy so background threads can send UI-safe signals.
        self._proxy = proxy

        # Track whether recording is active so we can ignore packets when stopped.
        self._active = False
        # Store thread resources that are created on start and released on stop.
        self._queue: Optional[queue.Queue] = None
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

        # Persist the body ordering for CSV columns and quaternion continuity.
        self._record_body_names: list[str] = []
        self._record_prev_quaternion: dict[str, tuple[float, float, float, float]] = {}

        # Track drop statistics to warn and auto-stop on sustained overload.
        self._record_recent_drop_count = 0
        self._record_recent_drop_rate = 0.0
        self._record_drop_window_start: Optional[float] = None
        self._record_drop_window_total = 0
        self._record_drop_window_dropped = 0
        self._record_drop_total = 0
        self._record_last_drop_warning_time = 0.0
        self._record_overload_triggered = False

        # Guard start/stop so we do not race if the UI toggles quickly.
        self._lock = threading.Lock()

    # Summary:
    # - Start the CSV writer thread and reset recording state for a new session.
    # - Input: `self`, `file_path` (str), `header` (list[str]), `body_names` (list[str]).
    # - Returns: None.
    def start(self, file_path: str, header: list[str], body_names: list[str]) -> None:
        # Ensure only one recording session is active at a time.
        with self._lock:
            if self._active:
                return
            self._active = True

            # Freeze the body names and reset per-session state.
            self._record_body_names = list(body_names)
            self._record_prev_quaternion = {}
            self._record_recent_drop_count = 0
            self._record_recent_drop_rate = 0.0
            self._record_drop_window_start = None
            self._record_drop_window_total = 0
            self._record_drop_window_dropped = 0
            self._record_drop_total = 0
            self._record_last_drop_warning_time = 0.0
            self._record_overload_triggered = False

            # Create fresh queue/stop event per session so writer state is isolated.
            self._queue = queue.Queue(maxsize=RECORD_QUEUE_MAXSIZE)
            self._stop_event = threading.Event()

            # Start the writer thread so file I/O never blocks the stream thread.
            self._thread = threading.Thread(
                target=self._writer_worker,
                args=(file_path, header, self._queue, self._stop_event),
                daemon=True,
            )
            # Thread start: begin draining rows to disk asynchronously.
            self._thread.start()

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
            self._record_body_names = []
            self._record_prev_quaternion = {}

        # Signal the writer thread to drain and exit.
        if stop_event is not None:
            stop_event.set()

        # Join briefly so we do not freeze the UI while final rows flush.
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
    # - Handle a 6D residual result by enqueuing a CSV row when recording is active.
    # - Input: `self`, `result` (tuple | None).
    # - Returns: None.
    def handle_6d_residual_result(self, result) -> None:
        # Snapshot state under lock so stop/start does not race packet handling.
        with self._lock:
            if not self._active:
                return
            record_queue = self._queue
            record_body_names = list(self._record_body_names)
            stop_event = self._stop_event
        if (
            record_queue is None
            or not record_body_names
            or (stop_event is not None and stop_event.is_set())
        ):
            return

        # Capture the arrival time on the stream thread for accurate timestamps.
        row = [self._utc_epoch_ms()]
        # Cache the row timestamp so external image sync can use the same CSV clock.
        ts_ms = row[0]

        # Fill NaNs when the 6D component is missing in the current packet.
        if result is None:
            row.extend([RECORD_NAN] * (len(record_body_names) * 7))
        else:
            _header, bodies = result
            for index, body_name in enumerate(record_body_names):
                # Guard against packets with fewer bodies than expected.
                if index >= len(bodies):
                    row.extend([RECORD_NAN] * 7)
                    continue

                body = bodies[index]
                if body is None:
                    row.extend([RECORD_NAN] * 7)
                    continue

                # Extract position and rotation while ignoring residuals.
                try:
                    position, rotation, _residual = body
                except (TypeError, ValueError):
                    row.extend([RECORD_NAN] * 7)
                    continue

                if position is None or rotation is None:
                    row.extend([RECORD_NAN] * 7)
                    continue

                # Read translation in raw QTM coordinates (no plot transform).
                if hasattr(position, "x") and hasattr(position, "y") and hasattr(
                    position, "z"
                ):
                    x_value, y_value, z_value = position.x, position.y, position.z
                else:
                    try:
                        x_value, y_value, z_value = position
                    except (TypeError, ValueError):
                        row.extend([RECORD_NAN] * 7)
                        continue

                try:
                    x_float = float(x_value)
                    y_float = float(y_value)
                    z_float = float(z_value)
                except (TypeError, ValueError):
                    row.extend([RECORD_NAN] * 7)
                    continue

                # Pull the rotation matrix and convert it to a quaternion.
                rotation_matrix = getattr(rotation, "matrix", None)
                if rotation_matrix is None:
                    row.extend([RECORD_NAN] * 7)
                    continue

                try:
                    rotation_values = [float(value) for value in rotation_matrix]
                except (TypeError, ValueError):
                    row.extend([RECORD_NAN] * 7)
                    continue
                if len(rotation_values) != 9:
                    row.extend([RECORD_NAN] * 7)
                    continue

                # QTM packs Rot[0..8] column-wise, so we must reshape with Fortran order.
                # Reference: https://docs.qualisys.com/qtm/content/processing_measurement/6dof_data_format.htm
                rotation_3x3 = (
                    np.array(rotation_values, dtype=float)
                    .reshape((3, 3), order="F")
                    .tolist()
                )
                quat = self._rotation_matrix_to_quaternion(rotation_3x3)

                # Optional sign continuity to avoid sudden sign flips per body.
                prev_quat = self._record_prev_quaternion.get(body_name)
                if prev_quat is not None:
                    dot_value = (
                        (prev_quat[0] * quat[0])
                        + (prev_quat[1] * quat[1])
                        + (prev_quat[2] * quat[2])
                        + (prev_quat[3] * quat[3])
                    )
                    if dot_value < 0.0:
                        quat = (-quat[0], -quat[1], -quat[2], -quat[3])

                self._record_prev_quaternion[body_name] = quat
                row.extend(
                    [quat[0], quat[1], quat[2], quat[3], x_float, y_float, z_float]
                )

        # Enqueue the row without blocking so the stream thread stays responsive.
        did_drop = False
        try:
            record_queue.put_nowait(row)
        except queue.Full:
            # Drop newest when the queue is full to protect latency.
            did_drop = True
        if not did_drop:
            # Emit the CSV row timestamp so coupled image recording can sync per-row.
            self._safe_emit(self._proxy.record_row_ts_ms.emit, int(ts_ms))

        now_monotonic = time.monotonic()
        overload = self._update_record_drop_stats(did_drop, now_monotonic)

        # Rate-limit UI warnings so we do not spam the text area.
        if did_drop and (
            now_monotonic - self._record_last_drop_warning_time
            >= RECORD_DROP_WARN_INTERVAL
        ):
            self._record_last_drop_warning_time = now_monotonic
            message = (
                f"Recording queue full: dropped {self._record_drop_total} frames total."
            )
            if self._record_recent_drop_rate > 0.0:
                message += (
                    f" Recent drop rate {self._record_recent_drop_rate * 100.0:.1f}%."
                )
            self._safe_emit(self._proxy.record_message.emit, message)

        # Auto-stop recording when sustained overload exceeds the threshold.
        if overload and not self._record_overload_triggered:
            self._record_overload_triggered = True
            self._safe_emit(
                self._proxy.record_stop.emit,
                "Recording stopped: disk too slow or queue overflow "
                f"(dropped {self._record_recent_drop_count} frames in "
                f"{RECORD_DROP_WINDOW_SECONDS:.0f}s; total dropped {self._record_drop_total})."
            )

    # Summary:
    # - Update the drop counters and decide whether sustained overload occurred.
    # - Input: `self`, `did_drop` (bool), `now_monotonic` (float).
    # - Returns: True if overload threshold is exceeded, else False (bool).
    def _update_record_drop_stats(self, did_drop: bool, now_monotonic: float) -> bool:
        # Initialize the drop window on the first packet.
        if self._record_drop_window_start is None:
            self._record_drop_window_start = now_monotonic
            self._record_drop_window_total = 0
            self._record_drop_window_dropped = 0

        # Track total samples for the current window.
        self._record_drop_window_total += 1
        if did_drop:
            # Track dropped samples and overall dropped count.
            self._record_drop_window_dropped += 1
            self._record_drop_total += 1

        # Only evaluate overload once the window duration has elapsed.
        elapsed = now_monotonic - self._record_drop_window_start
        if elapsed < RECORD_DROP_WINDOW_SECONDS:
            return False

        # Compute recent metrics from the finished window.
        if self._record_drop_window_total > 0:
            self._record_recent_drop_count = self._record_drop_window_dropped
            self._record_recent_drop_rate = (
                self._record_drop_window_dropped / self._record_drop_window_total
            )
        else:
            self._record_recent_drop_count = 0
            self._record_recent_drop_rate = 0.0

        # Reset the window counters for the next evaluation cycle.
        self._record_drop_window_start = now_monotonic
        self._record_drop_window_total = 0
        self._record_drop_window_dropped = 0

        # Return overload state so callers can auto-stop when needed.
        return self._record_recent_drop_rate > RECORD_DROP_RATE_THRESHOLD

    # Summary:
    # - Writer thread entry point that owns file I/O for CSV recording.
    # - Input: `self`, `file_path` (str), `header` (list[str]),
    #   `record_queue` (queue.Queue), `stop_event` (threading.Event).
    # - Returns: None.
    def _writer_worker(
        self,
        file_path: str,
        header: list[str],
        record_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        # Summary:
        # - Format numeric values for CSV output with fixed decimal precision.
        # - Input: `value` (object).
        # - Returns: Formatted value for CSV output (object).
        def _format_csv_value(value):
            # Keep integers unchanged so timestamps remain exact.
            if isinstance(value, int):
                return value
            # Format floats with up to 4 decimals and keep NaN readable.
            if isinstance(value, float):
                if math.isnan(value):
                    return "NaN"
                return f"{value:.4f}"
            return value

        # Wrap file I/O in a try/except so writer errors do not crash the app.
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                # Write the header once at the top of the file.
                writer.writerow(header)

                rows_since_flush = 0
                while True:
                    if stop_event.is_set():
                        # Drain any remaining rows before exiting.
                        try:
                            row = record_queue.get_nowait()
                        except queue.Empty:
                            break
                    else:
                        # Wait briefly for new rows so we do not spin.
                        try:
                            row = record_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue

                    # Format floats on the writer thread to keep the stream thread light.
                    formatted_row = [_format_csv_value(value) for value in row]
                    writer.writerow(formatted_row)
                    rows_since_flush += 1
                    # Flush periodically so data hits disk without per-row overhead.
                    if rows_since_flush >= RECORD_FLUSH_EVERY:
                        csv_file.flush()
                        rows_since_flush = 0

                # Flush one last time before closing the file.
                csv_file.flush()
        except Exception as exc:
            # Report writer failures back to the UI without touching widgets directly.
            self._safe_emit(self._proxy.record_stop.emit, f"Recording stopped: {exc}")


# Summary:
# - Worker that owns the asyncio loop and QTM streaming thread.
# - What it does: connects to QTM, streams packets, and forwards data through MocapStreamProxy.
class QtmStreamWorker:
    # Summary:
    # - Initialize the stream worker with its proxy and recorder references.
    # - Input: `self`, `proxy` (MocapStreamProxy), `recorder` (CsvRecordWorker).
    # - Returns: None.
    def __init__(self, proxy: MocapStreamProxy, recorder: CsvRecordWorker) -> None:
        # Store the proxy to emit UI-safe signals from the stream thread.
        self._proxy = proxy
        # Store the recorder so packet callbacks can forward results.
        self._recorder = recorder

        # Track the stream thread and asyncio loop for safe stopping.
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None

        # Cache body names so packet formatting uses stable labels.
        self._body_names: list[str] = []
        # Keep a short rolling window so we can estimate stream cadence cheaply per packet.
        self._health_window_start_ts = time.monotonic()
        self._health_window_frame_count = 0
        self._health_fps_estimate = 0.0

        # Lock protects loop/body name updates across threads.
        self._lock = threading.Lock()

    # Summary:
    # - Parse the 6D parameters XML and return a list of rigid body names.
    # - Input: `xml_string` (str).
    # - Returns: A list of body names (list[str]).
    @staticmethod
    def _parse_6d_body_names(xml_string: str) -> list[str]:
        # Return early when there is no XML, so callers can handle missing data safely.
        if not xml_string:
            return []

        # Parse the XML and guard against malformed data that could crash the stream.
        try:
            root = ElementTree.fromstring(xml_string)
        except ElementTree.ParseError:
            return []

        # Find the 6D section because body names live inside The_6D/Body/Name nodes.
        the_6d = root.find(".//The_6D")
        if the_6d is None:
            return []

        # Collect body names in the order QTM reports them.
        body_names: list[str] = []
        for body in the_6d.findall("Body"):
            name_element = body.find("Name")
            if name_element is not None and name_element.text:
                body_names.append(name_element.text.strip())

        return body_names

    # Summary:
    # - Request 6D parameters from QTM so we can map rigid body indices to names.
    # - Input: `connection` (qtm_rt.QRTConnection).
    # - Returns: A list of rigid body names (list[str]).
    @staticmethod
    async def _load_body_names(connection) -> list[str]:
        # Ask QTM for its 6D parameter XML so the packet formatter can show labels.
        xml_string = await connection.get_parameters(parameters=["6d"])
        # Transform the XML into a simple list of names.
        return QtmStreamWorker._parse_6d_body_names(xml_string)

    # Summary:
    # - Convert a 6D residual packet into a readable multi-line string.
    # - Input: `packet` (qtm_rt.QRTPacket), `body_names` (list[str]), `result` (tuple | None).
    # - Returns: Formatted text for display in the UI (str).
    @staticmethod
    def _format_6d_residual_text(packet, body_names: list[str], result=None) -> str:
        # Use a provided 6D result so callers can avoid duplicate packet reads.
        if result is None:
            # Always check for missing components because QTM can send empty frames.
            result = packet.get_6d_residual()
        if result is None:
            return f"Frame {packet.framenumber} | No 6D residual data"

        # Unpack the header and body list now that we know the component exists.
        header, bodies = result
        lines = [f"Frame {packet.framenumber} | {header}"]

        # Build a line per rigid body with position, rotation matrix, and residual.
        for index, body in enumerate(bodies):
            # Map the body index to a known name when possible.
            label = body_names[index] if index < len(body_names) else f"Body{index + 1}"
            position, rotation, residual = body

            # Extract position coordinates and guard against missing tuples.
            x, y, z = position if position is not None else (0.0, 0.0, 0.0)
            # Pull the rotation matrix if present; otherwise mark it as missing.
            rotation_matrix = getattr(rotation, "matrix", None)
            # Pull the residual value if present; otherwise mark it as missing.
            residual_value = getattr(residual, "residual", None)

            if rotation_matrix is None or residual_value is None:
                # Provide a safe fallback when QTM omits rotation/residual data.
                lines.append(
                    f"{label}: X={x:.2f}, Y={y:.2f}, Z={z:.2f} | R=N/A | Residual=N/A"
                )
                continue

            r11, r12, r13, r21, r22, r23, r31, r32, r33 = rotation_matrix
            lines.append(
                f"{label}:\n"
                f"X = [ {x:.2f}, {y:.2f}, {z:.2f}]\n"
                f"R = [ {r11:.2f}  {r12:.2f}  {r13:.2f};\n"
                f"      {r21:.2f}  {r22:.2f}  {r23:.2f};\n"
                f"      {r31:.2f}  {r32:.2f}  {r33:.2f}]\n"
                f"Res = {residual_value:.2f}"
            )

        return "\n".join(lines)

    # Summary:
    # - Build a dict of 6D rigid body poses aligned to the configured body names.
    # - Input: `result` (tuple | None), `body_names` (list[str]).
    # - Returns: Mapping of body name to 4x4 pose matrix or None (dict[str, np.ndarray | None]).
    @staticmethod
    def _extract_6d_poses(
        result, body_names: list[str]
    ) -> dict[str, Optional[np.ndarray]]:
        # Return an empty mapping when there is no data and no known bodies.
        if result is None and not body_names:
            return {}

        # If there is no 6D component, return None for each known body name.
        if result is None:
            return {name: None for name in body_names}

        # Unpack the 6D result safely; the header is unused for pose extraction.
        _header, bodies = result

        # Build a stable list of names so dict insertion order stays consistent.
        if body_names:
            target_names = body_names
        else:
            # Fall back to generic names when QTM parameters are unavailable.
            target_names = [f"Body{index + 1}" for index in range(len(bodies))]

        poses: dict[str, Optional[np.ndarray]] = {}
        for index, name in enumerate(target_names):
            # Store None when QTM omits a body so downstream plotting can skip it safely.
            if index >= len(bodies):
                poses[name] = None
                continue

            body = bodies[index]
            # Guard against empty body tuples to avoid crashes.
            if body is None:
                poses[name] = None
                continue

            # The SDK returns (position, rotation, residual); we need position + rotation.
            try:
                position, rotation, _residual = body
            except (TypeError, ValueError):
                poses[name] = None
                continue

            # Skip missing components because a full pose needs both translation and rotation.
            if position is None or rotation is None:
                poses[name] = None
                continue

            # Support tuple-style positions; fall back to attributes if provided.
            if hasattr(position, "x") and hasattr(position, "y") and hasattr(
                position, "z"
            ):
                x_value, y_value, z_value = position.x, position.y, position.z
            else:
                try:
                    x_value, y_value, z_value = position
                except (TypeError, ValueError):
                    poses[name] = None
                    continue

            # Convert values to floats; if conversion fails, store None.
            try:
                x_float = float(x_value)
                y_float = float(y_value)
                z_float = float(z_value)
            except (TypeError, ValueError):
                poses[name] = None
                continue

            # Pull the rotation matrix from the SDK rotation object.
            rotation_matrix = getattr(rotation, "matrix", None)
            if rotation_matrix is None:
                poses[name] = None
                continue

            # Convert rotation entries to floats and validate the expected length.
            try:
                rotation_values = [float(value) for value in rotation_matrix]
            except (TypeError, ValueError):
                poses[name] = None
                continue
            if len(rotation_values) != 9:
                poses[name] = None
                continue

            # QTM packs Rot[0..8] column-wise, so reshape with Fortran order.
            # Reference: https://docs.qualisys.com/qtm/content/processing_measurement/6dof_data_format.htm
            rotation_3x3 = np.array(rotation_values, dtype=float).reshape((3, 3), order="F")

            # Assemble the homogeneous transform in QTM's raw coordinate frame.
            pose_matrix = np.eye(4, dtype=float)
            pose_matrix[:3, :3] = rotation_3x3
            pose_matrix[:3, 3] = [x_float, y_float, z_float]

            poses[name] = pose_matrix

        return poses

    # Summary:
    # - Check whether a pose matrix contains finite rigid-body data.
    # - Input: `pose_matrix` (object).
    # - Returns: True when the pose is a finite 4x4 matrix, otherwise False (bool).
    @staticmethod
    def _is_valid_pose_matrix(pose_matrix: object) -> bool:
        # Validation/transforms: only count finite 4x4 poses as detected bodies.
        if not isinstance(pose_matrix, np.ndarray):
            return False
        if pose_matrix.shape != (4, 4):
            return False
        return bool(np.isfinite(pose_matrix).all())

    # Summary:
    # - Start the QTM streaming thread for the given IP address.
    # - Input: `self`, `ip_address` (str).
    # - Returns: None.
    def start(self, ip_address: str) -> None:
        # Avoid launching duplicate stream threads.
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            # Thread start: run the asyncio loop in a background thread.
            self._thread = threading.Thread(
                target=self._run_stream_loop,
                args=(ip_address,),
                daemon=True,
            )
            self._thread.start()

    # Summary:
    # - Signal the asyncio loop to stop streaming safely.
    # - Input: `self`.
    # - Returns: None.
    def stop(self) -> None:
        # Snapshot the loop and stop event under lock for thread-safe access.
        with self._lock:
            loop = self._loop
            stop_event = self._stop_event

        # Exit early if the loop never started or already stopped.
        if loop is None or stop_event is None:
            return

        # Summary:
        # - Thread-safe callback to set the asyncio stop event.
        # - Input: None.
        # - Returns: None.
        def _request_stop() -> None:
            # Set the stop event so the async task can exit gracefully.
            if self._stop_event is not None:
                self._stop_event.set()

        # Ask the asyncio loop to execute the stop request in its own thread.
        loop.call_soon_threadsafe(_request_stop)

    # Summary:
    # - Return a copy of the most recent body names list.
    # - Input: `self`.
    # - Returns: Copy of body names (list[str]).
    def body_names_snapshot(self) -> list[str]:
        # Lock to keep the list consistent while copying.
        with self._lock:
            return list(self._body_names)

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
    # - Run the asyncio event loop for streaming in a background thread.
    # - Input: `self`, `ip_address` (str).
    # - Returns: None.
    def _run_stream_loop(self, ip_address: str) -> None:
        # Create a dedicated asyncio "event loop" for QTM streaming.
        #
        # Asyncio can be confusing at first, so the key idea is:
        # - The event loop is like a tiny scheduler that repeatedly:
        #   1) checks which async tasks are ready to continue,
        #   2) runs them a little bit,
        #   3) then waits (without busy-looping) for I/O or timers.
        #
        # IMPORTANT: event loops are thread-specific.
        # - Qt already owns the *main* thread because `app.exec()` runs the UI loop there.
        # - If we ran an asyncio loop in the main thread, the UI would freeze.
        # - So we start a *second* loop in this background thread dedicated to networking/QTM.
        #
        # `asyncio.new_event_loop()` creates a fresh loop object that is not attached to any thread yet.
        loop = asyncio.new_event_loop()
        # `asyncio.set_event_loop(loop)` registers this loop as "the current loop" for THIS thread.
        # This matters because many asyncio helpers (e.g., `asyncio.Event()`, `asyncio.get_event_loop()`)
        # assume there is a current loop and will bind themselves to it.
        asyncio.set_event_loop(loop)

        # Store the loop and stop event so the UI thread can signal a stop.
        with self._lock:
            self._loop = loop
            self._stop_event = asyncio.Event()

        try:
            # Run the async streaming coroutine until it exits or is stopped.
            loop.run_until_complete(self._stream_main(ip_address))
        finally:
            # Always close the loop to release resources when the thread exits.
            loop.close()
            with self._lock:
                self._loop = None
                self._stop_event = None

    # Summary:
    # - Async coroutine that connects to QTM and streams 6D residual data.
    # - Input: `self`, `ip_address` (str).
    # - Returns: None.
    async def _stream_main(self, ip_address: str) -> None:
        connection = None

        try:
            # Attempt to connect to QTM using the requested protocol version.
            connection = await qtm_rt.connect(
                ip_address,
                port=QTM_PORT,
                version=QTM_PROTOCOL_VERSION,
            )

            # Report a failure if the connection could not be established.
            if connection is None:
                self._safe_emit(
                    self._proxy.state_changed.emit,
                    False, f"Could not connect to QTM at {ip_address}:{QTM_PORT}"
                )
                return

            # If a stop was requested while connecting, close immediately.
            if self._stop_event is not None and self._stop_event.is_set():
                connection.disconnect()
                self._safe_emit(
                    self._proxy.state_changed.emit, False, "Stream stopped before connect."
                )
                return

            # Load body names before streaming so labels are ready for the first packet.
            body_names = await self._load_body_names(connection)
            with self._lock:
                self._body_names = list(body_names)

            # Notify the UI that streaming has started successfully.
            # Reset health counters at stream start so cadence is measured per active stream session.
            self._health_window_start_ts = time.monotonic()
            self._health_window_frame_count = 0
            self._health_fps_estimate = 0.0
            self._safe_emit(
                self._proxy.state_changed.emit,
                True, f"Connected to QTM at {ip_address}:{QTM_PORT}"
            )

            # Start streaming 6D residual frames and route packets to the callback.
            # The SDK returns a status string immediately, so we must keep the loop alive.
            stream_result = await connection.stream_frames(
                components=["6dres"],
                on_packet=self._on_packet,
            )
            # Normalize the SDK return (can be bytes) so comparisons are stable.
            if isinstance(stream_result, bytes):
                stream_result = stream_result.decode(errors="ignore")
            # If QTM did not accept the stream request, stop and notify the UI.
            if stream_result != "Ok":
                self._safe_emit(
                    self._proxy.state_changed.emit,
                    False, f"Stream request failed: {stream_result}"
                )
                return

            # Stay alive until the UI requests a stop or QTM drops the transport.
            if self._stop_event is not None:
                while not self._stop_event.is_set():
                    # If QTM drops the connection, exit so the UI can reset.
                    if not connection.has_transport():
                        self._safe_emit(
                            self._proxy.state_changed.emit,
                            False, "Connection closed by QTM."
                        )
                        return
                    # Sleep briefly to avoid busy-waiting on the asyncio loop.
                    await asyncio.sleep(0.1)

            # Ask QTM to stop streaming before disconnecting.
            if connection.has_transport():
                await connection.stream_frames_stop()

            # Inform the UI that streaming has stopped so it can reset controls.
            self._safe_emit(self._proxy.state_changed.emit, False, "Stream stopped.")
        except Exception as exc:
            # Report unexpected errors so the UI can recover and show feedback.
            self._safe_emit(self._proxy.state_changed.emit, False, f"Stream error: {exc}")
        finally:
            # Disconnect if the transport is still active.
            if connection is not None and connection.has_transport():
                connection.disconnect()

    # Summary:
    # - Handle incoming QTM packets and forward pose/health data to the UI.
    # - Input: `self`, `packet` (qtm_rt.QRTPacket).
    # - Returns: None.
    def _on_packet(self, packet) -> None:
        # Read the 6D residual component once so pose, health, and recording use the same packet view.
        result = packet.get_6d_residual()

        # Snapshot body names so formatting uses stable labels.
        with self._lock:
            body_names = list(self._body_names)

        # Build a stable pose mapping aligned with body_names for the plot.
        poses_qtm = self._extract_6d_poses(result, body_names)
        # Timestamp the rigid body packet at PC-receive time for software coupling.
        rigidbody_ts_ms = _now_ms()
        # Compute compact stream-health metrics so the UI can log one readable summary line periodically.
        n_bodies_total = len(body_names) if body_names else len(poses_qtm)
        n_bodies_detected = sum(
            1
            for pose_matrix in poses_qtm.values()
            if self._is_valid_pose_matrix(pose_matrix)
        )
        # Track packet cadence in a rolling monotonic window to estimate FPS-like health cheaply.
        self._health_window_frame_count += 1
        now_ts = time.monotonic()
        elapsed = now_ts - self._health_window_start_ts
        if elapsed >= 0.5:
            if elapsed > 0.0:
                self._health_fps_estimate = self._health_window_frame_count / elapsed
            # Reset rolling window so the cadence estimate stays responsive.
            self._health_window_start_ts = now_ts
            self._health_window_frame_count = 0
        has_6d_data = result is not None
        # Emit the timestamped rigid body packet from worker thread to UI thread.
        self._safe_emit(self._proxy.poses_ready.emit, rigidbody_ts_ms, poses_qtm)
        # Emit compact health stats using the same safe proxy signal pathway.
        self._safe_emit(
            self._proxy.stream_health_ready.emit,
            int(packet.framenumber),
            int(n_bodies_total),
            int(n_bodies_detected),
            float(self._health_fps_estimate),
            bool(has_6d_data),
        )

        # Forward the result to the recorder without touching any UI state.
        self._recorder.handle_6d_residual_result(result)


# Summary:
# - Main mocap panel widget that renders the Designer UI and orchestrates the streaming lifecycle.
# - What it owns: UI state (enabled/disabled controls, button toggle text), plot timer, and worker instances.
# - What it ensures: the Qt UI never freezes (no blocking calls on the UI thread), and streaming/recording can
#   be started and stopped repeatedly without restarting the application.
class MocapWidget(QWidget):
    # Signal: emit one rigid body packet per mocap update for downstream coupling.
    sig_rigidbody_packet = Signal(int, object)
    # Signal: emit the CSV row timestamp so external image recording can sync to it.
    # Use object to preserve 64-bit UTC epoch values without overflow.
    sig_record_row_ts_ms = Signal(object)

    # Summary:
    # - Initialize the UI and wire signal handlers for streaming control.
    # - Input: `self`.
    # - Returns: None.
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_Mocap()
        self.ui.setupUi(self)
        # UI state change: force stylesheet background/border painting on this QWidget container.
        # WHY: some QWidget styles render inconsistently without WA_StyledBackground enabled.
        self.ui.widget_mocap_matplotlib.setAttribute(Qt.WA_StyledBackground, True)
        # Set a stable objectName so slot naming can follow project conventions.
        self.setObjectName("mocapWidget")
        # Cache the original mocap plot-container stylesheet so recording indicator changes are reversible.
        self._mocap_matplotlib_base_stylesheet = (
            self.ui.widget_mocap_matplotlib.styleSheet()
        )
        # Keep a fixed border thickness for the recording indicator to avoid magic numbers in style updates.
        self._mocap_recording_indicator_border_px = 6
        # Cache the original layout margins so we can preserve any future Designer-configured spacing.
        base_margins = self.ui.verticalLayout_2.contentsMargins()
        self._mocap_matplotlib_base_margins = (
            base_margins.left(),
            base_margins.top(),
            base_margins.right(),
            base_margins.bottom(),
        )
        # UI state change: always reserve enough inset so the red border is never covered by child widgets.
        # WHY: toolbar/canvas fill this layout; without inset, they visually occlude container borders.
        self.ui.verticalLayout_2.setContentsMargins(
            max(
                self._mocap_matplotlib_base_margins[0],
                self._mocap_recording_indicator_border_px,
            ),
            max(
                self._mocap_matplotlib_base_margins[1],
                self._mocap_recording_indicator_border_px,
            ),
            max(
                self._mocap_matplotlib_base_margins[2],
                self._mocap_recording_indicator_border_px,
            ),
            max(
                self._mocap_matplotlib_base_margins[3],
                self._mocap_recording_indicator_border_px,
            ),
        )

        # Keep the text area readable but non-editable for incoming stream data.
        self.ui.plainTextEdit_mocap_textStream.setReadOnly(True)
        # UI state change: disable undo history because the log is append-only and should stay lightweight.
        self.ui.plainTextEdit_mocap_textStream.setUndoRedoEnabled(False)
        # Enable the widget so stream output is visible even though editing is blocked.
        self.ui.plainTextEdit_mocap_textStream.setEnabled(True)

        # Track the streaming state so the button can toggle cleanly.
        self._stream_state = "idle"
        # Keep the mocap text stream bounded so long sessions do not grow widget memory unbounded.
        self._textstream_max_lines = 200
        # Track the most recent raw line so short duplicate bursts can be debounced.
        self._last_textstream_line = ""
        self._last_textstream_line_ts = 0.0
        # Keep the latest stream-health sample so health logs can be rate-limited on the UI thread.
        self._last_health_log_ts = 0.0
        self._health_log_interval_sec = 1.0
        self._latest_health_frame_number: Optional[int] = None
        self._latest_health_n_bodies_total = 0
        self._latest_health_n_bodies_detected = 0
        self._latest_health_fps_estimate = 0.0
        self._latest_health_has_6d_data = False

        # Build the Matplotlib canvas and place it in the prepared UI container.
        self._mocap_canvas = Mocap3DCanvas(self)
        # Build a Matplotlib navigation toolbar so users can pan/zoom/reset and save the 3D view.
        self._mocap_toolbar = NavigationToolbar(self._mocap_canvas, self)
        # UI state change: use compact icon-only controls so the toolbar consumes less vertical space.
        self._mocap_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self._mocap_toolbar.setIconSize(QSize(14, 14))
        self._mocap_toolbar.setFixedHeight(18)
        # UI state change: trim button padding/margins so controls stay usable but denser.
        self._mocap_toolbar.setStyleSheet(
            "QToolBar { spacing: 0px; padding: 0px; margin: 0px; }"
            "QToolButton { padding: 0px; margin: 0px; }"
        )
        # UI state change: place toolbar above the canvas for direct interaction controls.
        self.ui.verticalLayout_mocap_matplotlib.addWidget(self._mocap_toolbar)
        self.ui.verticalLayout_mocap_matplotlib.addWidget(self._mocap_canvas)

        # Cache the latest QTM poses for the throttled plot updates.
        self._latest_poses_qtm: dict[str, Optional[np.ndarray]] = {}
        # Cache the latest rigid body packet fields for coupled stream consumers.
        self._latest_rigidbody_ts_ms: Optional[int] = None
        self._latest_rigidbody_data: Optional[object] = None

        # Create a timer that redraws the plot at a fixed rate on the UI thread.
        self._plot_timer = QTimer(self)
        self._plot_timer.setObjectName("mocapPlotTimer")
        self._plot_timer.setInterval(33)
        # Signal connection: update the plot at a fixed rate to avoid per-packet redraws.
        self._plot_timer.timeout.connect(self._on_mocapPlotTimer_timeout)

        # Create a signal proxy so thread callbacks can safely update the UI.
        self._stream_proxy = MocapStreamProxy()
        # Signal connection: route background text updates into the UI thread.
        self._stream_proxy.text_ready.connect(self._on_mocapStreamProxy_text_ready)
        # Signal connection: route background pose updates into the UI thread.
        self._stream_proxy.poses_ready.connect(self._on_mocapStreamProxy_poses_ready)
        # Signal connection: route worker health snapshots into the UI thread for periodic log output.
        self._stream_proxy.stream_health_ready.connect(
            self._on_mocapStreamProxy_stream_health_ready
        )
        # Signal connection: react to state changes from the stream thread (connected, stopped, errors).
        self._stream_proxy.state_changed.connect(self._on_mocapStreamProxy_state_changed)
        # Signal connection: show recording warnings or info on the UI thread.
        self._stream_proxy.record_message.connect(self._on_mocapStreamProxy_record_message)
        # Signal connection: stop recording safely from background threads.
        self._stream_proxy.record_stop.connect(self._on_mocapStreamProxy_record_stop)
        # Signal connection: forward record row timestamps to external consumers.
        self._stream_proxy.record_row_ts_ms.connect(self.sig_record_row_ts_ms)

        # Create worker instances that own the stream and recording threads.
        self._record_worker = CsvRecordWorker(self._stream_proxy)
        self._stream_worker = QtmStreamWorker(self._stream_proxy, self._record_worker)

        # Connect the open/close button to the toggle handler.
        self.ui.pushButton_mocap_openStream.clicked.connect(
            self._on_pushButton_mocap_openStream_clicked
        )
        # Signal connection: clear the record directory when the clear button is clicked.
        self.ui.pushButton_mocap_recorddirClear.clicked.connect(
            self._on_pushButton_mocap_recorddirClear_clicked
        )
        # Signal connection: open a directory picker for the record directory.
        self.ui.pushButton_mocap_recorddirBrowse.clicked.connect(
            self._on_pushButton_mocap_recorddirBrowse_clicked
        )
        # Signal connection: toggle recording when the record button is clicked.
        self.ui.pushButton_mocap_record.clicked.connect(
            self._on_pushButton_mocap_record_clicked
        )

        # Ensure the system combo starts on Qualisys, since that is the only supported option.
        self.ui.comboBox_mocap_systemSelect.setCurrentText("Qualisys")
        # Disable recording until streaming is active.
        self.ui.pushButton_mocap_record.setEnabled(False)
        # Initial feedback: make the first state explicit so users know how to begin.
        self._log_mocap_event(
            "info",
            level="INFO",
            message="Idle. Enter QTM IP and press Open Stream.",
        )

    # Summary:
    # - Slot function that handles the open/close stream button click.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_mocap_openStream_clicked(self) -> None:
        # Toggle based on current state so the user can start or stop streaming.
        if self._stream_state == "idle":
            self._start_stream()
        else:
            # Log stop intent so request and resulting state transitions are both visible.
            self._log_mocap_event(
                "stream_request",
                level="INFO",
                action="stop",
                source="qtm",
            )
            self._stop_stream()

    # Summary:
    # - Slot function that clears the record directory line edit.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_mocap_recorddirClear_clicked(self) -> None:
        # UI state change: clear the selected record directory text.
        self.ui.lineEdit_mocap_recorddir.setText("")

    # Summary:
    # - Slot function that opens a directory picker for the record directory.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_mocap_recorddirBrowse_clicked(self) -> None:
        # Use the current text as the starting folder when it exists.
        current_path = self.ui.lineEdit_mocap_recorddir.text().strip()
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
        self.ui.lineEdit_mocap_recorddir.setText(absolute_path)

    # Summary:
    # - Slot function that starts or stops CSV recording.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_mocap_record_clicked(self) -> None:
        # Toggle based on current recording state to keep the button behavior predictable.
        if self._record_worker.is_active():
            self._stop_recording("Recording stopped.")
            return
        self._start_recording()

    # Summary:
    # - Resolve the IP address from the line edit (or placeholder).
    # - Input: `self`.
    # - Returns: The IP address string (str), or an empty string if unavailable.
    def _resolve_ip_address(self) -> str:
        # Prefer the explicit user input, but fall back to the placeholder.
        ip_address = self.ui.lineEdit_mocap_ip.text().strip()
        if not ip_address:
            ip_address = self.ui.lineEdit_mocap_ip.placeholderText().strip()
        return ip_address

    # Summary:
    # - Validate that a string looks like a real IP address (IPv4 or IPv6).
    # - Input: `ip_address` (str).
    # - Returns: True when valid, otherwise False (bool).
    def _is_ip_address_valid(self, ip_address: str) -> bool:
        # Use the stdlib parser so we only accept well-formed IP addresses.
        try:
            ipaddress.ip_address(ip_address)
        except ValueError:
            return False
        return True

    # Summary:
    # - Start the Qualisys streaming thread and update the UI for a pending connection.
    # - Input: `self`.
    # - Returns: None.
    def _start_stream(self) -> None:
        # Validate the selected mocap system because only Qualisys is supported.
        system = self.ui.comboBox_mocap_systemSelect.currentText().strip()
        if system != "Qualisys":
            QMessageBox.warning(
                self,
                "Unsupported System",
                "Only the Qualisys system is supported right now.",
            )
            return

        # Resolve the target IP address before attempting a connection.
        ip_address = self._resolve_ip_address()
        if not ip_address:
            QMessageBox.warning(
                self,
                "Missing IP Address",
                "Please enter the Qualisys system IP address.",
            )
            return

        # Confirm the IP address looks valid so we fail fast on obvious typos.
        if not self._is_ip_address_valid(ip_address):
            QMessageBox.warning(
                self,
                "Invalid IP Address",
                "Please enter a valid IP address (for example, 192.168.0.10).",
            )
            return

        # Prevent duplicate start requests while we spin up the background loop.
        if self._stream_state != "idle":
            return

        # Mark the UI as busy so users know we are trying to connect.
        self._stream_state = "connecting"
        # Disable input controls while a connection is pending to avoid mid-connection changes.
        self.ui.comboBox_mocap_systemSelect.setEnabled(False)
        self.ui.lineEdit_mocap_ip.setEnabled(False)
        # UI state change: disable recording while connecting.
        self.ui.pushButton_mocap_record.setEnabled(False)
        # Change the button label so a second click stops the connection attempt.
        self.ui.pushButton_mocap_openStream.setText("Stop Stream")
        # Log the connection attempt so users can see explicit stream lifecycle intent.
        self._log_mocap_event(
            "stream_request",
            level="INFO",
            action="start",
            source="qtm",
            endpoint=f"{ip_address}:{QTM_PORT}",
        )

        # Clear the plot and cache so stale points are not shown during a new connect.
        self._reset_plot_state()

        # Launch the asyncio event loop in a background thread so the UI stays responsive.
        self._stream_worker.start(ip_address)

    # Summary:
    # - Stop the Qualisys stream by signaling the asyncio loop to shut down.
    # - Input: `self`.
    # - Returns: None.
    def _stop_stream(self) -> None:
        # Stop plot updates immediately so the UI does not redraw after the user stops.
        self._plot_timer.stop()
        # Clear the plot and cache so the stopped state looks empty.
        self._reset_plot_state()

        # Signal the streaming worker to stop its asyncio loop safely.
        self._stream_worker.stop()

    # Summary:
    # - Append one formatted line to the mocap log console.
    # - What it does: Adds timestamp and level labels, enforces bounded log size, and keeps the view scrolled to the latest line.
    # - Input: `self`, `level` (str), `message` (str).
    # - Returns: None.
    def _append_mocap_textstream(self, level: str, message: str) -> None:
        # Normalize and ignore empty messages so the console only contains useful entries.
        message_text = str(message).strip()
        if not message_text:
            return

        # Prefix with local wall-clock time and level so both widgets use the same visible structure.
        timestamp_text = datetime.now().strftime("%H:%M:%S")
        level_text = (level or "INFO").upper()
        formatted_line = f"[{timestamp_text}] [{level_text}] {message_text}"
        self.ui.plainTextEdit_mocap_textStream.appendPlainText(formatted_line)

        # Trim old lines from the top so long sessions keep memory usage stable.
        document = self.ui.plainTextEdit_mocap_textStream.document()
        while document.blockCount() > int(self._textstream_max_lines):
            cursor = QTextCursor(document)
            cursor.movePosition(QTextCursor.Start)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            # Remove the trailing newline left after deleting the first block.
            cursor.deleteChar()

        # Keep the latest line visible after append/trim so the console behaves like a live activity feed.
        cursor = self.ui.plainTextEdit_mocap_textStream.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.ui.plainTextEdit_mocap_textStream.setTextCursor(cursor)
        self.ui.plainTextEdit_mocap_textStream.ensureCursorVisible()

    # Summary:
    # - Log one mocap lifecycle event with canonical event-based formatting.
    # - What it does: Converts event/context data into one human-readable line, debounces duplicates, and appends to the bounded text stream.
    # - Input: `self`, `event` (str), `level` (str), `context` (keyword context fields).
    # - Returns: None.
    def _log_mocap_event(self, event: str, *, level: str = "INFO", **context) -> None:
        # Normalize the event key so callers can pass stable, case-insensitive identifiers.
        event_key = str(event).strip().lower()
        message_text = ""

        # Build stream-start/stop request messages with source-specific context.
        if event_key == "stream_request":
            action = str(context.get("action", "start")).strip().lower() or "start"
            source = str(context.get("source", "unknown")).strip().lower() or "unknown"
            detail_parts: list[str] = []
            endpoint_text = str(context.get("endpoint", "")).strip()
            if endpoint_text:
                detail_parts.append(f"endpoint={endpoint_text}")
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

        # Skip blank messages so malformed event context does not spam empty rows.
        if not message_text:
            return

        # Debounce identical consecutive messages so repeated callbacks do not flood the UI.
        now_ts = time.monotonic()
        line_key = f"{(level or 'INFO').upper()}::{message_text}"
        if line_key == self._last_textstream_line and (
            now_ts - self._last_textstream_line_ts
        ) < 1.0:
            return
        self._last_textstream_line = line_key
        self._last_textstream_line_ts = now_ts
        self._append_mocap_textstream(level, message_text)

    # Summary:
    # - Log periodic stream-health snapshots at a fixed cadence.
    # - What it does: Converts the latest worker health metrics into one concise line and rate-limits output.
    # - Input: `self`.
    # - Returns: None.
    def _log_stream_health(self) -> None:
        # Only emit health snapshots while streaming; ignore stale packets when the UI is idle.
        if self._stream_state != "streaming":
            return
        if self._latest_health_frame_number is None:
            return

        # Rate-limit health lines so users get readable status without per-packet spam.
        now_ts = time.monotonic()
        if (now_ts - self._last_health_log_ts) < self._health_log_interval_sec:
            return
        self._last_health_log_ts = now_ts

        # Build a compact line that highlights frame index, detected bodies, and cadence.
        health_line = (
            "Stream OK | "
            f"Frame={self._latest_health_frame_number} | "
            "Bodies: detected "
            f"{self._latest_health_n_bodies_detected}/{self._latest_health_n_bodies_total}"
        )
        if not self._latest_health_has_6d_data:
            health_line += " (no 6D this frame)"
        health_line += f" | FPS~{self._latest_health_fps_estimate:.1f}"
        self._log_mocap_event("stream_health", level="INFO", message=health_line)

    # Summary:
    # - Slot function that updates the activity log from proxy text messages.
    # - Input: `self`, `text` (str).
    # - Returns: None.
    def _on_mocapStreamProxy_text_ready(self, text: str) -> None:
        # Slot function: route proxy text to the bounded log console on the UI thread.
        self._log_mocap_event("info", level="INFO", message=text)

    # Summary:
    # - Slot function that reacts to stream state changes and updates UI controls accordingly.
    # - Input: `self`, `is_running` (bool), `message` (str).
    # - Returns: None.
    def _on_mocapStreamProxy_state_changed(self, is_running: bool, message: str) -> None:
        # Slot function: keep stream lifecycle messages visible in the bounded activity log.
        self._log_mocap_event(
            "stream_state",
            level="INFO" if is_running else "WARN",
            is_running=is_running,
            reason=message,
        )

        if is_running:
            # Mark the UI as streaming so users know the connection is live.
            self._stream_state = "streaming"

            # Reset health timing/state at stream start so periodic logs begin fresh for this session.
            self._last_health_log_ts = 0.0
            self._latest_health_frame_number = None
            self._latest_health_n_bodies_total = 0
            self._latest_health_n_bodies_detected = 0
            self._latest_health_fps_estimate = 0.0
            self._latest_health_has_6d_data = False
            self.ui.pushButton_mocap_openStream.setText("Stop Stream")

            # Keep inputs disabled while streaming to prevent changing live settings.
            self.ui.comboBox_mocap_systemSelect.setEnabled(False)
            self.ui.lineEdit_mocap_ip.setEnabled(False)

            # Clear cached plot state and start the timer for throttled redraws.
            self._reset_plot_state()
            self._plot_timer.start()
            
            # UI state change: enable recording only when streaming is active.
            self.ui.pushButton_mocap_record.setEnabled(True)
            return

        # Reset UI controls after the stream stops or fails.
        self._stream_state = "idle"
        self.ui.pushButton_mocap_openStream.setText("Open Stream")

        # Re-enable inputs so the user can change settings before reconnecting.
        self.ui.comboBox_mocap_systemSelect.setEnabled(True)
        self.ui.lineEdit_mocap_ip.setEnabled(True)

        # Stop plot updates immediately when the stream stops or drops.
        self._plot_timer.stop()

        # Clear plot data so the stopped state is visually empty.
        self._reset_plot_state()

        # Reset health timing/state so idle periods do not leak into the next stream.
        self._last_health_log_ts = 0.0
        self._latest_health_frame_number = None
        self._latest_health_n_bodies_total = 0
        self._latest_health_n_bodies_detected = 0
        self._latest_health_fps_estimate = 0.0
        self._latest_health_has_6d_data = False
        # Stop recording if the stream is no longer running.
        if self._record_worker.is_active():
            self._stop_recording("Recording stopped because streaming ended.")
        # UI state change: disable recording while idle.
        self.ui.pushButton_mocap_record.setEnabled(False)

    # Summary:
    # - Slot function that displays recording-related messages on the UI thread.
    # - Input: `self`, `message` (str).
    # - Returns: None.
    def _on_mocapStreamProxy_record_message(self, message: str) -> None:
        # Slot function: keep non-fatal recording warnings visible in the same bounded activity log.
        if message:
            self._log_mocap_event("warning", level="WARN", message=message)

    # Summary:
    # - Slot function that stops recording when background threads request it.
    # - Input: `self`, `reason` (str).
    # - Returns: None.
    def _on_mocapStreamProxy_record_stop(self, reason: str) -> None:
        # Slot function: surface record-stop reasons in the activity log before changing worker/UI state.
        if reason:
            self._log_mocap_event("record_stop", level="WARN", reason=reason)
        # Stop recording safely on the UI thread when an error or overload occurs.
        if self._record_worker.is_active():
            # Skip duplicate reason logging because this slot already logged the stop cause.
            self._stop_recording(reason, log_reason=False)

    # Summary:
    # - Slot function that receives worker health metrics and triggers periodic log output.
    # - Input: `self`, `frame_number` (int), `n_bodies_total` (int), `n_bodies_detected` (int),
    #   `fps_estimate` (float), `has_6d_data` (bool).
    # - Returns: None.
    def _on_mocapStreamProxy_stream_health_ready(
        self,
        frame_number: int,
        n_bodies_total: int,
        n_bodies_detected: int,
        fps_estimate: float,
        has_6d_data: bool,
    ) -> None:
        # Slot function: cache worker health samples so logging cadence is controlled on the UI thread.
        self._latest_health_frame_number = int(frame_number)
        self._latest_health_n_bodies_total = int(n_bodies_total)
        self._latest_health_n_bodies_detected = int(n_bodies_detected)
        self._latest_health_fps_estimate = float(fps_estimate)
        self._latest_health_has_6d_data = bool(has_6d_data)
        self._log_stream_health()

    # Summary:
    # - Slot function that caches the latest QTM poses for the plot timer.
    # - Input: `self`, `rigidbody_ts_ms` (int), `rigidbody_data` (dict[str, np.ndarray | None]).
    # - Returns: None.
    def _on_mocapStreamProxy_poses_ready(
        self, rigidbody_ts_ms: int, rigidbody_data: dict[str, Optional[np.ndarray]]
    ) -> None:
        # Cache poses so the UI timer can update the plot at a fixed rate.
        self._latest_poses_qtm = rigidbody_data
        # Cache the latest rigid body packet fields for pull-style coupling usage.
        self._latest_rigidbody_ts_ms = int(rigidbody_ts_ms)
        self._latest_rigidbody_data = rigidbody_data
        # Emit the packet outward so the coupled controller can match by timestamp.
        self.sig_rigidbody_packet.emit(
            self._latest_rigidbody_ts_ms, self._latest_rigidbody_data
        )

    # Summary:
    # - Slot function that redraws the plot on a fixed timer interval.
    # - Input: `self`.
    # - Returns: None.
    def _on_mocapPlotTimer_timeout(self) -> None:
        # Update the plot on the UI thread using the cached poses.
        self._mocap_canvas.update_poses(self._latest_poses_qtm)

    # Summary:
    # - Clear cached poses and reset the matplotlib scatter to empty.
    # - Input: `self`.
    # - Returns: None.
    def _reset_plot_state(self) -> None:
        # Clear cached poses so the timer does not redraw stale data.
        self._latest_poses_qtm = {}
        self._latest_rigidbody_ts_ms = None
        self._latest_rigidbody_data = None
        # Clear the scatter so the user sees an empty plot when stopped.
        self._mocap_canvas.clear_points()

    # Summary:
    # - Build the CSV header for a frozen list of rigid body names.
    # - Input: `body_names` (list[str]).
    # - Returns: CSV header row (list[str]).
    @staticmethod
    def _build_record_header(body_names: list[str]) -> list[str]:
        # Start with the timestamp column required by the schema.
        header = ["utc_epoch_ms"]
        for name in body_names:
            # Append the quaternion columns first (w, x, y, z), then translation.
            header.extend(
                [
                    f"{name}_q1",
                    f"{name}_q2",
                    f"{name}_q3",
                    f"{name}_q4",
                    f"{name}_t1",
                    f"{name}_t2",
                    f"{name}_t3",
                ]
            )
        return header

    # Summary:
    # - Build a unique CSV file path for a new recording session.
    # - Input: `record_dir` (str).
    # - Returns: Full file path for the CSV (str).
    @staticmethod
    def _build_record_file_path(record_dir: str) -> str:
        # Use UTC in the filename so sessions are sortable and unambiguous.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        base_name = f"qtm_{timestamp}.csv"
        file_path = os.path.join(record_dir, base_name)

        # Add a numeric suffix when a file with the same name already exists.
        if not os.path.exists(file_path):
            return file_path

        for suffix in range(1, 1000):
            candidate = os.path.join(record_dir, f"qtm_{timestamp}_{suffix:03d}.csv")
            if not os.path.exists(candidate):
                return candidate

        # Fall back to the base name if we somehow exhausted all suffixes.
        return file_path

    # Summary:
    # - Start a CSV recording session requested by an external controller.
    # - What it does: Validates streaming state and record directory, builds the CSV schema,
    #   starts the writer thread, and returns the active CSV file path.
    # - Input: `self`, `record_dir` (str).
    # - Returns: Full CSV file path on success, otherwise an empty string (str).
    def start_external_csv_record(self, record_dir: str) -> str:
        # Validation: require an active stream so recording stays aligned to live packets.
        if self._stream_state != "streaming":
            QMessageBox.warning(
                self,
                "Stream Not Active",
                "Start streaming before recording.",
            )
            return ""

        # Validation: avoid double-starts when a recording is already active.
        if self._record_worker.is_active():
            QMessageBox.warning(
                self,
                "Recording Active",
                "A mocap recording is already active.",
            )
            return ""

        # Validation: ensure a record directory is provided and writable.
        if not record_dir:
            QMessageBox.warning(
                self,
                "Missing Record Directory",
                "Please choose a record directory before recording.",
            )
            return ""
        record_dir = os.path.abspath(record_dir)
        if not os.path.isdir(record_dir):
            QMessageBox.warning(
                self,
                "Invalid Record Directory",
                "The selected record directory does not exist.",
            )
            return ""
        if not os.access(record_dir, os.W_OK):
            QMessageBox.warning(
                self,
                "Record Directory Not Writable",
                "The selected directory is not writable.",
            )
            return ""

        # Freeze the body names at record start so the CSV schema is stable.
        record_body_names = self._stream_worker.body_names_snapshot()
        if not record_body_names:
            # Fall back to the latest pose keys when QTM parameters are unavailable.
            if not self._latest_poses_qtm:
                QMessageBox.warning(
                    self,
                    "Body Names Missing",
                    "Rigid body names are not available yet. Try again once streaming is active.",
                )
                return ""
            record_body_names = list(self._latest_poses_qtm.keys())

        # Build the output path and header before starting the writer thread.
        record_file_path = self._build_record_file_path(record_dir)
        record_header = self._build_record_header(record_body_names)

        # Start the writer thread so file I/O never blocks the stream thread.
        self._record_worker.start(record_file_path, record_header, record_body_names)

        return record_file_path

    # Summary:
    # - Stop an externally-controlled CSV recording session.
    # - What it does: Stops the writer thread so the CSV file can flush and close.
    # - Input: `self`, `reason` (str).
    # - Returns: None.
    def stop_external_csv_record(self, reason: str = "") -> None:
        # Stop the recording worker so it can flush and close cleanly.
        self._record_worker.stop(reason)

    # Summary:
    # - Start a new CSV recording session on the UI thread.
    # - Input: `self`.
    # - Returns: None.
    def _start_recording(self) -> None:
        # Require an active stream so recording stays aligned to live packets.
        if self._stream_state != "streaming":
            QMessageBox.warning(
                self,
                "Stream Not Active",
                "Start streaming before recording.",
            )
            return

        # Validate the record directory so we fail fast on missing paths.
        record_dir = self.ui.lineEdit_mocap_recorddir.text().strip()
        if not record_dir:
            QMessageBox.warning(
                self,
                "Missing Record Directory",
                "Please choose a record directory before recording.",
            )
            return
        if not os.path.isdir(record_dir):
            QMessageBox.warning(
                self,
                "Invalid Record Directory",
                "The selected record directory does not exist.",
            )
            return
        if not os.access(record_dir, os.W_OK):
            QMessageBox.warning(
                self,
                "Record Directory Not Writable",
                "The selected directory is not writable.",
            )
            return

        # Freeze the body names at record start so the CSV schema is stable.
        record_body_names = self._stream_worker.body_names_snapshot()
        if not record_body_names:
            # Fall back to the latest pose keys when QTM parameters are unavailable.
            if not self._latest_poses_qtm:
                QMessageBox.warning(
                    self,
                    "Body Names Missing",
                    "Rigid body names are not available yet. Try again once streaming is active.",
                )
                return
            record_body_names = list(self._latest_poses_qtm.keys())

        # Build the output path and header before starting the writer thread.
        record_file_path = self._build_record_file_path(record_dir)
        record_header = self._build_record_header(record_body_names)

        # Start the writer thread so file I/O never blocks the stream thread.
        self._record_worker.start(record_file_path, record_header, record_body_names)

        # UI state change: lock the record directory while recording is active.
        self.ui.lineEdit_mocap_recorddir.setEnabled(False)
        self.ui.pushButton_mocap_recorddirBrowse.setEnabled(False)
        self.ui.pushButton_mocap_recorddirClear.setEnabled(False)
        # UI state change: update the record button to show stop intent.
        self.ui.pushButton_mocap_record.setText("Stop Recording")
        # UI state change: show a clear visual indicator around the mocap plot while recording is active.
        self._set_mocap_recording_indicator(active=True)
        # Log recording start so the destination path remains visible in the activity log.
        self._log_mocap_event(
            "record_start",
            level="INFO",
            mode="local",
            target=record_file_path,
        )

    # Summary:
    # - Stop the active CSV recording session and release writer resources.
    # - Input: `self`, `reason` (str), `log_reason` (bool).
    # - Returns: None.
    def _stop_recording(self, reason: str = "", *, log_reason: bool = True) -> None:
        # Avoid redundant work when recording is already stopped.
        if not self._record_worker.is_active():
            return

        # Stop the recording worker so it can flush and close cleanly.
        self._record_worker.stop(reason)

        # UI state change: unlock the record directory controls.
        self.ui.lineEdit_mocap_recorddir.setEnabled(True)
        self.ui.pushButton_mocap_recorddirBrowse.setEnabled(True)
        self.ui.pushButton_mocap_recorddirClear.setEnabled(True)
        # UI state change: restore the record button text.
        self.ui.pushButton_mocap_record.setText("Record")
        # UI state change: remove the recording indicator and restore the original plot-container style.
        self._set_mocap_recording_indicator(active=False)
        # Keep the record button enabled only while streaming.
        self.ui.pushButton_mocap_record.setEnabled(self._stream_state == "streaming")

        # Log the stop reason so recording lifecycle events remain visible after UI state changes.
        if reason and log_reason:
            self._log_mocap_event("record_stop", level="INFO", reason=reason)

    # Summary:
    # - Update the mocap plot-container border to indicate whether recording is active.
    # - What it does: Preserves the original container stylesheet and conditionally adds/removes
    #   a thick red border for clear recording status feedback.
    # - Input: `self`, `active` (bool).
    # - Returns: None.
    def _set_mocap_recording_indicator(self, active: bool) -> None:
        # Use the cached base style so we can always restore exactly what existed before recording.
        base_stylesheet = self._mocap_matplotlib_base_stylesheet.strip()
        if active:
            # Scope the border rule to the container objectName so only this widget draws the indicator.
            selector = "#widget_mocap_matplotlib"
            border_rule = (
                f"{selector} "
                f"{{ border: {self._mocap_recording_indicator_border_px}px solid rgb(255, 0, 0); }}"
            )
            # Keep Qt syntax valid for both selector-based and property-only cached styles.
            if "{" in base_stylesheet and "}" in base_stylesheet:
                indicator_stylesheet = f"{base_stylesheet}\n{border_rule}"
            elif base_stylesheet:
                normalized_base = base_stylesheet.rstrip(";")
                indicator_stylesheet = (
                    f"{selector} {{ {normalized_base}; "
                    f"border: {self._mocap_recording_indicator_border_px}px solid rgb(255, 0, 0); }}"
                )
            else:
                indicator_stylesheet = border_rule
            self.ui.widget_mocap_matplotlib.setStyleSheet(indicator_stylesheet)
            return

        # Restore the original container style when recording is not active.
        self.ui.widget_mocap_matplotlib.setStyleSheet(base_stylesheet)

    # Summary:
    # - Set the mocap recording indicator from an external controller.
    # - What it does: Exposes a small public API so parent windows can toggle the same indicator
    #   used by this widget's local record button without duplicating stylesheet logic.
    # - Input: `self`, `active` (bool).
    # - Returns: None.
    def set_recording_indicator(self, active: bool) -> None:
        # WHY: Keep indicator style ownership inside this widget even when recording is started elsewhere.
        self._set_mocap_recording_indicator(active=active)

    # Summary:
    # - Ensure the streaming thread is stopped before the widget closes.
    # - Input: `self`, `event` (Qt close event).
    # - Returns: None.
    def closeEvent(self, event) -> None:
        # Stop recording so the writer thread can close the file cleanly.
        if self._record_worker.is_active():
            self._stop_recording("Recording stopped because the window closed.")
        # Stop streaming so QTM connections are released when the window closes.
        self._stop_stream()
        # Stop the plot timer to avoid redraws after the widget is closed.
        self._plot_timer.stop()
        event.accept()


# Summary:
# - Application entry point that shows the MocapWidget window.
# - Input: None (reads command-line args via sys.argv).
# - Returns: None.
def main() -> None:
    # Import sys here to keep module imports focused for the widget.
    import sys

    # Create and run the Qt application.
    app = QApplication(sys.argv)
    window = MocapWidget()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
