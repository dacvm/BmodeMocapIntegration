"""Main window wrapper that embeds B-mode and Mocap widgets for acquisition."""

# Standard library helpers for filesystem normalization and app exit codes.
from datetime import datetime
import os
import sys
from typing import Optional

# Qt core helpers for queued signal connections and typed slots.
from PySide6.QtCore import Qt, Slot, QTimer
# Qt widgets for main window, dialogs, and message boxes.
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

# Coupler that matches image and rigid body packets by timestamp.
from helpers.coupled_stream_controller import CoupledStreamController
# Sequence writer that exports coupled packets into Plus-style .mha files.
from helpers.mha_writer import MhaWriter
# Custom widgets that will be embedded into the acquisition main window.
from main_bmode import BModeWidget
from main_mocap import CoupledCsvWriter, MocapWidget
# Generated UI class from Qt Designer (do not edit the generated file).
from ui.app_acquisition_ui import Ui_MainWindow

COUPLED_MAXDIFF_IMAGEPOSE_MS = 120
COUPLED_MOCAP_BUFFER_MAXLEN = 500
COUPLED_SNAPSHOT_TIMEOUT_MS = 2000


# Summary:
# - Main window wrapper that hosts the acquisition UI and embeds custom widgets.
# - What it does: Builds the generated UI, inserts B-mode/Mocap widgets, and
#   implements coupled-record directory interactions without the volume widget.
# - Input: `self` (new window instance).
# - Returns: A fully initialized QMainWindow subclass instance.
class AcquisitionMainWindow(QMainWindow):
    # Summary:
    # - Initialize the acquisition main window and wire up the UI behavior.
    # - What it does: Creates the generated UI, embeds custom widgets, adjusts layout
    #   spacing, and connects button signals to slot handlers.
    # - Input: `self` (the new window instance).
    # - Returns: None.
    def __init__(self) -> None:
        super().__init__()

        # Build the generated UI so all Qt Designer widgets are available.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Make embedded layouts tight so custom widgets fill their group boxes.
        self.ui.verticalLayout_bmode.setContentsMargins(0, 0, 0, 0)
        self.ui.verticalLayout_bmode.setSpacing(4)
        self.ui.verticalLayout_mocap.setContentsMargins(0, 0, 0, 0)
        self.ui.verticalLayout_mocap.setSpacing(4)

        # Create custom widgets and keep references to avoid garbage collection.
        self._bmode_widget = BModeWidget()
        self._mocap_widget = MocapWidget()

        # Add custom widgets into their assigned layouts.
        self.ui.verticalLayout_bmode.addWidget(self._bmode_widget)
        self.ui.verticalLayout_mocap.addWidget(self._mocap_widget)

        # UI state change: keep status labels in a known initial state on startup.
        self.ui.label_status_bmode.setText("Disconnected")
        self.ui.label_status_mocap.setText("Disconnected")

        # Track the latest debug values so one status line can show coupling health.
        self._latest_coupled_diff_ms: Optional[int] = None
        self._latest_coupler_stats: dict = {}
        # Track active record mode and writer so all stop paths can finalize safely.
        self._mha_writer: Optional[MhaWriter] = None
        self._active_record_mode: Optional[str] = None
        # Track image+csv coupled recording state so we can stop both recorders together.
        self._coupled_imagecsv_active = False
        self._imagecsv_session_dir = ""
        self._imagecsv_csv_path = ""
        # Track the CSV file owned by `.mha + .csv` mode so MHA and CSV stop together.
        self._mhacsv_csv_path = ""
        self._coupled_csv_writer: Optional[CoupledCsvWriter] = None
        self._mha_recording_error = ""
        # Track stop coordination so queued timestamps can drain before stopping the image writer.
        self._imagecsv_stop_pending = False
        self._imagecsv_stop_reason = ""
        # Track one-shot coupled snapshot state separately from continuous recording state.
        self._coupled_snapshot_pending = False
        self._coupled_snapshot_mode: Optional[str] = None
        self._coupled_snapshot_record_dir = ""
        self._suppress_next_coupler_stats_status_update = False

        # Create the software coupler for image+pose sample-and-hold matching.
        self._coupler = CoupledStreamController(
            maxdiff_imagepose_ms=COUPLED_MAXDIFF_IMAGEPOSE_MS,
            mocap_buffer_maxlen=COUPLED_MOCAP_BUFFER_MAXLEN,
        )

        # Signal connection: mirror B-mode stream state into the acquisition status label.
        self._bmode_widget._proxy.state_changed.connect(
            self._on_bmodeStreamProxy_state_changed
        )
        # Signal connection: mirror Mocap stream state into the acquisition status label.
        self._mocap_widget._stream_proxy.state_changed.connect(
            self._on_mocapStreamProxy_state_changed
        )
        # Signal connection: forward image packets into the coupler on a queued connection.
        self._bmode_widget.sig_image_packet.connect(
            self._coupler.on_image_packet,
            Qt.QueuedConnection,
        )
        # Signal connection: forward rigid body packets into the coupler on a queued connection.
        self._mocap_widget.sig_rigidbody_packet.connect(
            self._coupler.on_rigidbody_packet,
            Qt.QueuedConnection,
        )
        # Signal connection: show coupled packet timing for debug visibility.
        self._coupler.sig_coupled_packet.connect(
            self._on_coupledStreamController_coupled_packet,
            # Both objects live on the UI thread. Finish the pair before Stop can run.
            Qt.DirectConnection,
        )
        # Signal connection: show coupling drop counters for debug visibility.
        self._coupler.sig_stats.connect(
            self._on_coupledStreamController_stats_updated,
            Qt.QueuedConnection,
        )
        # Create a single-shot timer to delay image stop so queued timestamps can flush.
        self._imagecsv_stop_timer = QTimer(self)
        self._imagecsv_stop_timer.setObjectName("coupledImageCsvStopTimer")
        self._imagecsv_stop_timer.setSingleShot(True)
        # Signal connection: finalize the image+csv stop after the drain window.
        self._imagecsv_stop_timer.timeout.connect(
            self._on_coupledImageCsvStopTimer_timeout
        )
        # Create a single-shot timer so one-shot snapshots cannot wait forever.
        self._coupled_snapshot_timer = QTimer(self)
        self._coupled_snapshot_timer.setObjectName("coupledSnapshotTimeoutTimer")
        self._coupled_snapshot_timer.setSingleShot(True)
        # Signal connection: cancel pending snapshot requests when no valid pair arrives.
        self._coupled_snapshot_timer.timeout.connect(
            self._on_coupledSnapshotTimeoutTimer_timeout
        )
        # Signal connection: stop image+csv recording if the image writer reports a failure.
        self._bmode_widget._proxy.record_stop.connect(
            self._on_bmodeStreamProxy_record_stop,
            Qt.QueuedConnection,
        )
        # Signal connection: stop image+csv if its independent CSV writer fails.
        self._mocap_widget._stream_proxy.record_stop.connect(
            self._on_mocapStreamProxy_record_stop,
            Qt.QueuedConnection,
        )

        # Signal connection: browse for an output directory.
        self.ui.pushButton_coupledrecord_recorddirBrowse.clicked.connect(
            self._on_pushButton_coupledrecord_recorddirBrowse_clicked
        )
        # Signal connection: clear the output directory.
        self.ui.pushButton_coupledrecord_recorddirClear.clicked.connect(
            self._on_pushButton_coupledrecord_recorddirClear_clicked
        )
        # Signal connection: run the record stream.
        self.ui.pushButton_coupledrecord_recordStream.clicked.connect(
            self._on_pushButton_coupledrecord_recordStream_clicked
        )
        # Signal connection: request exactly one coupled image+pose snapshot.
        self.ui.pushButton_coupledrecord_snapshot.clicked.connect(
            self._on_pushButton_coupledrecord_snapshot_clicked
        )
        # UI state change: initialize coupled debug line in the status bar.
        self._update_coupled_debug_status()

    # Summary:
    # - Toggle both embedded recording indicators for coupled recording state.
    # - What it does: Routes one active/inactive state to both child widgets so the
    #   acquisition flow has one place to control recording borders.
    # - Input: `self`, `active` (bool).
    # - Returns: None.
    def _set_coupled_recording_indicators(self, active: bool) -> None:
        # WHY: Keep coupled-record indicator control centralized to avoid mismatched UI states.
        self._bmode_widget.set_recording_indicator(active=active)
        self._mocap_widget.set_recording_indicator(active=active)

    # Summary:
    # - Slot function for B-mode stream-state updates from the embedded B-mode widget.
    # - What it does: Keeps the acquisition status label synchronized with stream start/stop.
    # - Input: `self`, `is_running` (bool), `message` (str).
    # - Returns: None.
    def _on_bmodeStreamProxy_state_changed(self, is_running: bool, message: str) -> None:
        # Slot function: react to stream state changes emitted by the B-mode widget proxy.
        _ = message

        # UI state change: show the active stream state in the B-mode status label.
        if is_running:
            self.ui.label_status_bmode.setText("Streaming")
            return

        # UI state change: restore idle state whenever B-mode streaming stops or fails.
        self.ui.label_status_bmode.setText("Disconnected")
        # Stop image+csv recording when B-mode streaming ends so sessions stay aligned.
        if self._coupled_imagecsv_active:
            self._stop_coupled_imagecsv_recording(
                "B-mode stream stopped; image+csv recording was ended."
            )
        # Stop coupled recording when one source drops so sessions stay valid.
        if self._coupler.is_recording():
            final_path, csv_path, had_active_mha_writer = (
                self._stop_active_mha_recording(
                    reason="B-mode stream stopped; coupled recording was ended.",
                    show_error_dialog=True,
                )
            )
            if final_path and csv_path:
                self.statusBar().showMessage(
                    "B-mode stream stopped; partial .mha saved: "
                    f"{final_path} (CSV: {csv_path})",
                    9000,
                )
            elif final_path:
                self.statusBar().showMessage(
                    f"B-mode stream stopped; partial .mha saved: {final_path}",
                    9000,
                )
            elif csv_path:
                self.statusBar().showMessage(
                    f"B-mode stream stopped; CSV saved: {csv_path}",
                    9000,
                )
            elif not had_active_mha_writer:
                self.statusBar().showMessage(
                    "B-mode stream stopped; coupled recording was ended.",
                    9000,
                )

    # Summary:
    # - Slot function for mocap stream-state updates from the embedded mocap widget.
    # - What it does: Keeps the acquisition status label synchronized with QTM stream start/stop.
    # - Input: `self`, `is_running` (bool), `message` (str).
    # - Returns: None.
    def _on_mocapStreamProxy_state_changed(self, is_running: bool, message: str) -> None:
        # Slot function: react to stream state changes emitted by the mocap widget proxy.
        _ = message

        # UI state change: show the active stream state in the mocap status label.
        if is_running:
            self.ui.label_status_mocap.setText("Streaming")
            return

        # UI state change: restore idle state whenever mocap streaming stops or fails.
        self.ui.label_status_mocap.setText("Disconnected")
        # Stop image+csv recording when mocap streaming ends so sessions stay aligned.
        if self._coupled_imagecsv_active:
            self._stop_coupled_imagecsv_recording(
                "Mocap stream stopped; image+csv recording was ended."
            )
        # Stop coupled recording when one source drops so sessions stay valid.
        if self._coupler.is_recording():
            final_path, csv_path, had_active_mha_writer = (
                self._stop_active_mha_recording(
                    reason="Mocap stream stopped; coupled recording was ended.",
                    show_error_dialog=True,
                )
            )
            if final_path and csv_path:
                self.statusBar().showMessage(
                    "Mocap stream stopped; partial .mha saved: "
                    f"{final_path} (CSV: {csv_path})",
                    9000,
                )
            elif final_path:
                self.statusBar().showMessage(
                    f"Mocap stream stopped; partial .mha saved: {final_path}",
                    9000,
                )
            elif csv_path:
                self.statusBar().showMessage(
                    f"Mocap stream stopped; CSV saved: {csv_path}",
                    9000,
                )
            elif not had_active_mha_writer:
                self.statusBar().showMessage(
                    "Mocap stream stopped; coupled recording was ended.",
                    9000,
                )

    # Summary:
    # - Slot function for B-mode recording stop events from the embedded B-mode widget.
    # - What it does: Stops image+csv coupled recording when the image writer reports a failure.
    # - Input: `self`, `reason` (str).
    # - Returns: None.
    def _on_bmodeStreamProxy_record_stop(self, reason: str) -> None:
        # Slot function: react to B-mode record-stop signals.
        if self._coupled_imagecsv_active:
            # Stop both recorders so the coupled session stays consistent.
            self._stop_coupled_imagecsv_recording(reason)

    # Summary:
    # - Slot function for mocap recording stop events from the embedded Mocap widget.
    # - What it does: Stops image+csv when its independent CSV worker reports a failure.
    # - Input: `self`, `reason` (str).
    # - Returns: None.
    def _on_mocapStreamProxy_record_stop(self, reason: str) -> None:
        # Slot function: react to mocap record-stop signals.
        if self._coupled_imagecsv_active:
            # Stop both recorders so the coupled session stays consistent.
            self._stop_coupled_imagecsv_recording(reason)
            return

    # Summary:
    # - Slot function that handles mocap CSV row timestamps during image+csv recording.
    # - What it does: Forwards the timestamp to the B-mode widget to save a single snapshot.
    # - Input: `self`, `ts_ms` (int).
    # - Returns: None.
    def _on_mocapWidget_record_row_ts_ms(self, ts_ms: int) -> None:
        # Slot function: forward mocap timestamps to the B-mode snapshot helper.
        self._bmode_widget.record_latest_frame_with_ts(int(ts_ms))

    # Summary:
    # - Slot function that finalizes a delayed image+csv recording stop.
    # - What it does: Disconnects the timestamp coupling, stops the image writer, resets state,
    #   and updates the UI with saved output paths.
    # - Input: `self`.
    # - Returns: None.
    def _on_coupledImageCsvStopTimer_timeout(self) -> None:
        # Slot function: finalize the delayed stop after queued timestamps drain.
        reason = self._imagecsv_stop_reason
        session_dir = self._imagecsv_session_dir
        csv_path = self._imagecsv_csv_path

        # Disconnect the row-timestamp coupling so no more snapshots are queued.
        try:
            self._mocap_widget.sig_record_row_ts_ms.disconnect(
                self._on_mocapWidget_record_row_ts_ms
            )
        except TypeError:
            # Ignore disconnect failures when the signal is already disconnected.
            pass

        # Stop the image recorder so queued frames flush and close cleanly.
        self._bmode_widget.stop_external_image_record(reason)

        # Reset session state so a new recording can start cleanly.
        self._coupled_imagecsv_active = False
        self._imagecsv_session_dir = ""
        self._imagecsv_csv_path = ""
        self._active_record_mode = None
        self._imagecsv_stop_pending = False
        self._imagecsv_stop_reason = ""

        # UI state change: restore the record button text after stopping.
        self.ui.pushButton_coupledrecord_recordStream.setText("Record")
        # UI state change: clear both embedded recording borders once delayed stop is fully finalized.
        self._set_coupled_recording_indicators(active=False)

        # Build a status message that includes the saved output locations.
        if reason:
            message = f"{reason} Image+csv saved in: {session_dir} (CSV: {csv_path})"
        else:
            message = f"Image+csv saved in: {session_dir} (CSV: {csv_path})"
        self.statusBar().showMessage(message, 9000)

    # Summary:
    # - Slot function for coupled packet events from the CoupledStreamController.
    # - What it does: caches the latest diff and refreshes debug status text.
    # - Input: `self`, `image_ts_ms` (int), `image_data` (object), `rigidbody_ts_ms` (int),
    #   `rigidbody_data` (object), `diff_ms` (int).
    # - Returns: None.
    @Slot(int, object, int, object, int)
    def _on_coupledStreamController_coupled_packet(
        self,
        image_ts_ms: int,
        image_data: object,
        rigidbody_ts_ms: int,
        rigidbody_data: object,
        diff_ms: int,
    ) -> None:
        # Cache the latest diff so status/debug text can show coupling quality.
        self._latest_coupled_diff_ms = int(diff_ms)
        self._update_coupled_debug_status()
        # Snapshot handling: consume exactly one accepted coupled packet and skip
        # continuous recording logic so one-shot output cannot append extra frames.
        if self._coupled_snapshot_pending:
            self._handle_coupled_snapshot_packet(
                image_ts_ms=image_ts_ms,
                image_data=image_data,
                rigidbody_ts_ms=rigidbody_ts_ms,
                rigidbody_data=rigidbody_data,
            )
            return
        # During active MHA-backed recording, append this accepted coupled packet to disk.
        if (
            self._coupler.is_recording()
            and self._active_record_mode in (".mha", ".mha + .csv")
            and self._mha_writer is not None
        ):
            try:
                csv_row = None
                if self._active_record_mode == ".mha + .csv":
                    if self._coupled_csv_writer is None:
                        raise RuntimeError("Coupled CSV writer is not open")
                    csv_row = self._coupled_csv_writer.prepare_row(rigidbody_data)
                stored = self._mha_writer.append_coupled_packet(
                    image_ts_ms=image_ts_ms,
                    image_data=image_data,
                    rigidbody_ts_ms=rigidbody_ts_ms,
                    rigidbody_data=rigidbody_data,
                )
                if stored and csv_row is not None:
                    self._coupled_csv_writer.append_row(csv_row)
            except Exception as exc:
                self._mha_recording_error = str(exc)
                # Stop and finalize on write failure so no partial writer keeps running.
                self._stop_active_mha_recording(
                    reason=f"Coupled recording stopped because writing failed: {exc}",
                    show_error_dialog=True,
                )

    # Summary:
    # - Slot function for coupling stats updates from the CoupledStreamController.
    # - What it does: caches the latest counters and refreshes debug status text.
    # - Input: `self`, `stats` (dict).
    # - Returns: None.
    @Slot(dict)
    def _on_coupledStreamController_stats_updated(self, stats: dict) -> None:
        # Validation/transform: guard against unexpected payloads before caching.
        if not isinstance(stats, dict):
            return
        self._latest_coupler_stats = dict(stats)
        # Queued stats must not hide an incomplete-recording error after shutdown.
        if self._mha_recording_error and not self._coupler.is_recording() and not self._coupled_snapshot_pending:
            return
        # UI state change: keep final snapshot messages visible when the coupler emits
        # its trailing stats update after a successful one-shot packet.
        if self._suppress_next_coupler_stats_status_update:
            self._suppress_next_coupler_stats_status_update = False
            return
        self._update_coupled_debug_status()

    # Summary:
    # - Rebuild the coupled-stream debug text in the status bar.
    # - What it does: combines latest diff and counters so users can quickly verify coupling health.
    # - Input: `self`.
    # - Returns: None.
    def _update_coupled_debug_status(self) -> None:
        # Read stats with safe defaults so the message still renders before first packet.
        count_coupled = int(self._latest_coupler_stats.get("count_coupled", 0))
        count_dropped_no_pose = int(
            self._latest_coupler_stats.get("count_dropped_no_pose", 0)
        )
        count_dropped_stale = int(
            self._latest_coupler_stats.get("count_dropped_stale", 0)
        )
        is_recording = bool(self._latest_coupler_stats.get("is_recording", False))
        is_snapshot_pending = bool(
            self._latest_coupler_stats.get("is_snapshot_pending", False)
        )

        # Render diff as "N/A" until the first coupled packet arrives.
        if self._latest_coupled_diff_ms is None:
            diff_text = "N/A"
        else:
            diff_text = f"{self._latest_coupled_diff_ms} ms"

        # UI state change: show one compact debug line for coupling success and drop reasons.
        if is_recording:
            state_text = "REC"
        elif is_snapshot_pending:
            state_text = "SNAPSHOT"
        else:
            state_text = "IDLE"
        self.statusBar().showMessage(
            "Coupled streaming "
            f"[ {state_text} ] Diff = {diff_text} | "
            f"Data coupled = {count_coupled} | "
            f"Data dropped (no match) = {count_dropped_no_pose} | "
            f"Data dropped (too old) = {count_dropped_stale}"
        )

    # Summary:
    # - Build a unique coupled session directory path.
    # - What it does: Uses the existing `coupled_YYYYMMDD_HHMMSS` naming style and adds
    #   a numeric suffix only when another session already exists for the same second.
    # - Input: `self`, `record_dir` (str).
    # - Returns: Full coupled session directory path (str).
    def _build_coupled_imagecsv_session_dir(self, record_dir: str) -> str:
        # Normalize the parent directory so downstream writers receive an absolute path.
        normalized_dir = os.path.abspath(os.path.expanduser(str(record_dir)))
        timestamp_text = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_session_dir = os.path.join(normalized_dir, f"coupled_{timestamp_text}")
        if not os.path.exists(base_session_dir):
            return base_session_dir

        # Avoid failing when two short sessions are created within the same clock second.
        for suffix in range(1, 1000):
            candidate = os.path.join(
                normalized_dir,
                f"coupled_{timestamp_text}_{suffix:03d}",
            )
            if not os.path.exists(candidate):
                return candidate
        return base_session_dir

    # Summary:
    # - Build a timestamped output path for one coupled snapshot `.mha` file.
    # - What it does: Normalizes the record directory and creates a deterministic snapshot filename.
    # - Input: `self`, `record_dir` (str).
    # - Returns: Full `.mha` output path (str).
    def _build_mha_snapshot_output_path(self, record_dir: str) -> str:
        # Normalize path so the one-frame writer receives a stable absolute destination.
        normalized_dir = os.path.abspath(os.path.expanduser(record_dir))
        # Filename format: keep a compact timestamp so saved MHA files sort chronologically.
        timestamp_text = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"SequenceSnapshot_{timestamp_text}.mha"
        return os.path.join(normalized_dir, filename)

    # Summary:
    # - Reset local UI/state for a completed or canceled coupled snapshot request.
    # - What it does: clears pending state, stops the timeout timer, re-enables the snapshot
    #   button, and hides recording indicators when no continuous recording is active.
    # - Input: `self`.
    # - Returns: None.
    def _reset_coupled_snapshot_state(self) -> None:
        # Stop the timeout timer so a completed snapshot cannot also report a timeout.
        if self._coupled_snapshot_timer.isActive():
            self._coupled_snapshot_timer.stop()

        # Clear one-shot state so the next Snapshot click starts from a clean request.
        self._coupled_snapshot_pending = False
        self._coupled_snapshot_mode = None
        self._coupled_snapshot_record_dir = ""

        # UI state change: allow another snapshot request after this one finishes.
        self.ui.pushButton_coupledrecord_snapshot.setEnabled(True)

        # UI state change: clear snapshot indicators unless a continuous recording owns them.
        if not self._coupled_imagecsv_active and not self._coupler.is_recording():
            self._set_coupled_recording_indicators(active=False)

    # Summary:
    # - Write one coupled packet as a one-frame `.mha` snapshot.
    # - What it does: starts an MHA writer, appends exactly one coupled packet, finalizes the
    #   file immediately, and cleans temporary payload state on append failures.
    # - Input: `self`, coupled packet fields (`image_ts_ms`, `image_data`, `rigidbody_ts_ms`,
    #   `rigidbody_data`), and `record_dir` (str).
    # - Returns: Final `.mha` path (str).
    def _write_coupled_mha_snapshot(
        self,
        image_ts_ms: int,
        image_data: object,
        rigidbody_ts_ms: int,
        rigidbody_data: object,
        record_dir: str,
    ) -> str:
        output_mha_path = self._build_mha_snapshot_output_path(record_dir)
        writer = MhaWriter(invert_transforms=False)
        writer.start(output_mha_path)
        finalize_attempted = False
        try:
            # Store exactly one accepted coupled packet in the one-frame sequence.
            writer.append_coupled_packet(
                image_ts_ms=image_ts_ms,
                image_data=image_data,
                rigidbody_ts_ms=rigidbody_ts_ms,
                rigidbody_data=rigidbody_data,
            )
            finalize_attempted = True
            return writer.finalize()
        except Exception:
            # Cleanup: finalize deletes the temporary payload when append failed before finalize.
            if not finalize_attempted:
                try:
                    writer.finalize()
                except Exception:
                    pass
            raise

    # Summary:
    # - Write one coupled packet as one B-mode JPEG plus one mocap CSV row.
    # - What it does: creates a coupled session directory, writes one image snapshot under a
    #   B-mode subfolder, and writes one CSV row using the same rigid body timestamp.
    # - Input: `self`, coupled packet fields (`image_data`, `rigidbody_ts_ms`, `rigidbody_data`),
    #   and `record_dir` (str).
    # - Returns: tuple of `(session_dir, image_path, csv_path)`.
    def _write_coupled_imagecsv_snapshot(
        self,
        image_data: object,
        rigidbody_ts_ms: int,
        rigidbody_data: object,
        record_dir: str,
    ) -> tuple[str, str, str]:
        # Build/create the coupled session folder before asking child widgets to write files.
        session_dir = self._build_coupled_imagecsv_session_dir(record_dir)
        os.makedirs(session_dir, exist_ok=False)

        # Use the rigid body timestamp for both files so their filenames/row values align.
        snapshot_ts_ms = int(rigidbody_ts_ms)
        image_path = self._bmode_widget.save_external_image_snapshot(
            session_dir,
            snapshot_ts_ms,
            image_data,
        )
        csv_path = self._mocap_widget.save_external_csv_snapshot(
            session_dir,
            snapshot_ts_ms,
            rigidbody_data,
        )
        return session_dir, image_path, csv_path

    # Summary:
    # - Store one accepted coupled packet for the pending snapshot request.
    # - What it does: dispatches to the selected snapshot format, resets snapshot state, and
    #   reports the saved path or write error to the operator.
    # - Input: `self`, coupled packet fields (`image_ts_ms`, `image_data`, `rigidbody_ts_ms`,
    #   `rigidbody_data`).
    # - Returns: None.
    def _handle_coupled_snapshot_packet(
        self,
        image_ts_ms: int,
        image_data: object,
        rigidbody_ts_ms: int,
        rigidbody_data: object,
    ) -> None:
        # Snapshot state is copied before reset so writer errors can still report context.
        snapshot_mode = self._coupled_snapshot_mode
        snapshot_record_dir = self._coupled_snapshot_record_dir
        message = ""

        try:
            # Dispatch to the storage format chosen by the existing radio buttons.
            if snapshot_mode == ".mha":
                final_path = self._write_coupled_mha_snapshot(
                    image_ts_ms=image_ts_ms,
                    image_data=image_data,
                    rigidbody_ts_ms=rigidbody_ts_ms,
                    rigidbody_data=rigidbody_data,
                    record_dir=snapshot_record_dir,
                )
                message = f"Coupled snapshot saved to: {final_path}"
            elif snapshot_mode == ".mha + .csv":
                # Hybrid snapshot: write the exact same one-frame MHA and one-row CSV outputs.
                final_path = self._write_coupled_mha_snapshot(
                    image_ts_ms=image_ts_ms,
                    image_data=image_data,
                    rigidbody_ts_ms=rigidbody_ts_ms,
                    rigidbody_data=rigidbody_data,
                    record_dir=snapshot_record_dir,
                )
                csv_path = self._mocap_widget.save_external_csv_snapshot(
                    snapshot_record_dir,
                    int(rigidbody_ts_ms),
                    rigidbody_data,
                )
                message = (
                    f"Coupled snapshot saved to: {final_path} "
                    f"(CSV: {csv_path})"
                )
            elif snapshot_mode == "image + .csv":
                session_dir, image_path, csv_path = self._write_coupled_imagecsv_snapshot(
                    image_data=image_data,
                    rigidbody_ts_ms=rigidbody_ts_ms,
                    rigidbody_data=rigidbody_data,
                    record_dir=snapshot_record_dir,
                )
                message = (
                    f"Coupled snapshot saved in: {session_dir} "
                    f"(Image: {image_path}, CSV: {csv_path})"
                )
            else:
                raise ValueError("Snapshot record mode is not selected")

            # UI state change: report the final snapshot destination after writing succeeds.
            self.statusBar().showMessage(message, 9000)
        except Exception as exc:
            # UI state change: surface write failures because no continuous stop path will run.
            message = f"Coupled snapshot failed: {exc}"
            self.statusBar().showMessage(message, 9000)
            QMessageBox.warning(
                self,
                "Snapshot Write Error",
                f"Could not write coupled snapshot:\n{exc}",
            )
        finally:
            # Keep the saved/failed snapshot message visible after the coupler's trailing stats.
            self._suppress_next_coupler_stats_status_update = True
            self._reset_coupled_snapshot_state()

    # Summary:
    # - Start an image+csv coupled recording session.
    # - What it does: Creates a session folder, starts B-mode image recording and mocap CSV recording,
    #   connects row timestamps to image snapshots, and updates UI state.
    # - Input: `self`, `record_dir` (str).
    # - Returns: True on success, otherwise False (bool).
    def _start_coupled_imagecsv_recording(self, record_dir: str) -> bool:
        # Build a unique session directory to keep images and CSV together.
        session_dir = self._build_coupled_imagecsv_session_dir(record_dir)
        # Create the session directory so both recorders have a stable base path.
        try:
            os.makedirs(session_dir, exist_ok=False)
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Session Directory Error",
                f"Could not create session directory:\n{exc}",
            )
            return False

        # Start B-mode image recording first so snapshot writes can succeed immediately.
        try:
            self._bmode_widget.start_external_image_record(session_dir)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Image Record Start Failed",
                f"Could not start B-mode image recording:\n{exc}",
            )
            return False

        # Start mocap CSV recording inside the same session directory.
        csv_path = self._mocap_widget.start_external_csv_record(session_dir)
        if not csv_path:
            # Stop the image recorder to avoid a half-started coupled session.
            self._bmode_widget.stop_external_image_record("CSV record start failed.")
            QMessageBox.warning(
                self,
                "CSV Record Start Failed",
                "Could not start mocap CSV recording.",
            )
            return False

        # Signal connection: write one image snapshot per mocap CSV row timestamp.
        self._mocap_widget.sig_record_row_ts_ms.connect(
            self._on_mocapWidget_record_row_ts_ms,
            Qt.QueuedConnection,
        )

        # Cache session state so stop handlers can report paths.
        self._coupled_imagecsv_active = True
        self._imagecsv_session_dir = session_dir
        self._imagecsv_csv_path = csv_path

        # UI state change: show stop intent while image+csv recording is active.
        self.ui.pushButton_coupledrecord_recordStream.setText("Stop Recording")
        # UI state change: show both embedded recording borders only after both recorders started.
        self._set_coupled_recording_indicators(active=True)
        # UI state change: show the active session directory in the status bar.
        self.statusBar().showMessage(
            f"Recording image+csv in session: {session_dir}",
            9000,
        )

        return True

    # Summary:
    # - Stop an image+csv coupled recording session.
    # - What it does: Stops the CSV writer immediately, then delays image stop briefly so
    #   queued timestamps can enqueue their snapshots.
    # - Input: `self`, `reason` (str), `drain_ms` (int).
    # - Returns: None.
    def _stop_coupled_imagecsv_recording(
        self, reason: str = "", drain_ms: int = 250
    ) -> None:
        # Avoid redundant work if the session is already inactive.
        if not self._coupled_imagecsv_active:
            return
        # Avoid double-stop when a delayed stop is already scheduled.
        if self._imagecsv_stop_pending:
            return

        # Store the stop reason so the delayed finalize can show the right message.
        self._imagecsv_stop_pending = True
        self._imagecsv_stop_reason = reason

        # Stop CSV recording first so no new timestamps are emitted.
        self._mocap_widget.stop_external_csv_record(reason)

        # Delay the image stop so queued timestamps can flush before stopping the writer.
        if drain_ms <= 0:
            self._on_coupledImageCsvStopTimer_timeout()
            return

        # Reset any pending timer and start a new drain window.
        if self._imagecsv_stop_timer.isActive():
            self._imagecsv_stop_timer.stop()
        self._imagecsv_stop_timer.start(int(drain_ms))

    # Summary:
    # - Stop the CSV half of an active `.mha + .csv` coupled recording.
    # - What it does: Stops the mocap CSV writer and clears the cached CSV path while
    #   leaving `.mha` finalization to the caller.
    # - Input: `self`, `reason` (str).
    # - Returns: The CSV path that belonged to the hybrid session, or an empty string.
    def _stop_coupled_mhacsv_csv_recording(self, reason: str = "") -> str:
        # Cache the path before stopping so callers can still report the saved CSV.
        csv_path = self._mhacsv_csv_path

        # Release session ownership even if flushing/closing raises an error.
        writer = self._coupled_csv_writer
        self._coupled_csv_writer = None
        self._mhacsv_csv_path = ""
        if writer is not None:
            writer.close()
        return csv_path

    # Summary:
    # - Stop an active `.mha` or `.mha + .csv` coupled recording.
    # - What it does: Disarms the coupler, stops the hybrid CSV writer when needed,
    #   finalizes the active MHA writer, and restores shared recording UI state.
    # - Input: `self`, `reason` (str), `show_error_dialog` (bool).
    # - Returns: Tuple `(final_path, csv_path, had_active_mha_writer)`.
    def _stop_active_mha_recording(
        self,
        reason: str,
        show_error_dialog: bool,
    ) -> tuple[Optional[str], str, bool]:
        # Cache mode before finalization because finalization always clears active mode.
        active_mode = self._active_record_mode
        had_active_mha_writer = (
            active_mode in (".mha", ".mha + .csv") and self._mha_writer is not None
        )

        # Stop the image-triggered coupling gate before closing writers so no new MHA frames append.
        if self._coupler.is_recording():
            self._coupler.stop_recording()

        # Capture counts before either writer is finalized and resets its state.
        errors = [self._mha_recording_error] if self._mha_recording_error else []
        csv_output_path = self._mhacsv_csv_path
        frame_count = self._mha_writer.frame_count if self._mha_writer is not None else 0
        csv_writer = self._coupled_csv_writer
        if active_mode == ".mha + .csv":
            if csv_writer is None:
                errors.append("Coupled CSV writer is missing")
            elif frame_count != csv_writer.row_count:
                errors.append(f"Saved counts differ: {frame_count} MHA frames, {csv_writer.row_count} CSV rows")

        # Always attempt both closes, even when the first one fails.
        csv_path = ""
        if active_mode == ".mha + .csv":
            try:
                csv_path = self._stop_coupled_mhacsv_csv_recording(reason)
            except Exception as exc:
                errors.append(f"CSV close failed: {exc}")

        # Finalize active .mha output so payload and header stay consistent.
        final_path = self._finalize_active_mha_recording(
            show_success_message=False,
            show_error_dialog=False,
        )
        if had_active_mha_writer and final_path is None:
            errors.append(self._mha_recording_error or "MHA finalization failed")

        # UI state change: restore button text when MHA-backed recording stops.
        self.ui.pushButton_coupledrecord_recordStream.setText("Record")
        # UI state change: clear both embedded recording borders after MHA-backed recording stops.
        self._set_coupled_recording_indicators(active=False)

        if errors:
            self._mha_recording_error = "; ".join(dict.fromkeys(errors))
            message = "Recording incomplete: " + self._mha_recording_error
            if final_path:
                message += f"\nRecoverable MHA: {final_path}"
            if csv_output_path:
                message += f"\nCSV output (may be incomplete): {csv_output_path}"
            self.statusBar().showMessage(message, 15000)
            if show_error_dialog:
                QMessageBox.warning(self, "Recording Incomplete", message)
            # Callers must not overwrite the error with a successful-save message.
            return None, "", had_active_mha_writer
        return final_path, csv_path, had_active_mha_writer

    # Summary:
    # - Slot function for the record directory clear button.
    # - What it does: Clears the record directory line edit so the user can start over.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_coupledrecord_recorddirClear_clicked(self) -> None:
        # Slot function: react to the clear button click.
        # UI state change: clear the text to show the directory is unset.
        self.ui.lineEdit_coupledrecord_recorddir.clear()

    # Summary:
    # - Slot function for the record directory browse button.
    # - What it does: Opens a folder picker and normalizes the chosen directory path.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_coupledrecord_recorddirBrowse_clicked(self) -> None:
        # Slot function: react to the browse button click.
        # Ask the user for a directory so recordings go to a known location.
        chosen_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Record Directory",
        )

        # Only update the UI when a folder is actually chosen.
        if chosen_dir:
            # Normalize the path so downstream code gets a clean absolute location.
            path = os.path.abspath(os.path.expanduser(chosen_dir))
            # UI state change: show the normalized path to the user.
            self.ui.lineEdit_coupledrecord_recorddir.setText(path)

    # Summary:
    # - Slot function for the coupled snapshot timeout timer.
    # - What it does: Cancels a pending one-shot snapshot request when no valid coupled
    #   packet arrives within the configured timeout window.
    # - Input: `self`.
    # - Returns: None.
    def _on_coupledSnapshotTimeoutTimer_timeout(self) -> None:
        # Slot function: react to the one-shot snapshot timeout.
        if not self._coupled_snapshot_pending:
            return

        # Keep the timeout message visible after cancel_snapshot emits its stats update.
        self._suppress_next_coupler_stats_status_update = True
        # Snapshot gate: cancel the pending request so live preview images are ignored again.
        self._coupler.cancel_snapshot()
        # UI state change: restore controls and indicators after the timed-out request.
        self._reset_coupled_snapshot_state()
        self.statusBar().showMessage(
            "Coupled snapshot timed out after "
            f"{COUPLED_SNAPSHOT_TIMEOUT_MS} ms; no valid image+pose packet was saved.",
            9000,
        )

    # Summary:
    # - Slot function for the Snapshot button.
    # - What it does: validates snapshot prerequisites, arms one pending coupled packet
    #   request, and starts a timeout so the UI does not wait forever.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_coupledrecord_snapshot_clicked(self) -> None:
        # Slot function: react to the Snapshot button click.
        if self._coupled_snapshot_pending:
            self.statusBar().showMessage("Coupled snapshot is already pending.", 5000)
            return

        # Validation: block one-shot snapshots while a continuous recording owns the writers.
        if self._coupled_imagecsv_active or self._coupler.is_recording():
            QMessageBox.warning(
                self,
                "Recording is Currently Active",
                "Please stop the current coupled recording before taking a snapshot.",
            )
            return

        # Validation: ensure a record directory is set before proceeding.
        record_dir = self.ui.lineEdit_coupledrecord_recorddir.text().strip()
        if not record_dir:
            QMessageBox.warning(
                self,
                "Record Directory Missing",
                "Please choose a record directory before taking a snapshot.",
            )
            return

        # Validation: require both sources to be streaming so a coupled packet can be created.
        bmode_state = self.ui.label_status_bmode.text().strip().lower()
        mocap_state = self.ui.label_status_mocap.text().strip().lower()
        if "streaming" not in bmode_state or "streaming" not in mocap_state:
            QMessageBox.warning(
                self,
                "Streaming Required",
                "Start both B-mode and Mocap streaming before taking a snapshot.",
            )
            return

        # Defensive check: radio buttons might not exist yet in the UI.
        if (
            not hasattr(self.ui, "radioButton_imagecsv")
            or not hasattr(self.ui, "radioButton_mha")
            or not hasattr(self.ui, "radioButton_mhacsv")
        ):
            QMessageBox.warning(
                self,
                "Record Mode Missing",
                "Record mode controls are not available in the current UI.",
            )
            return

        # Normalize/create output directory early so writer errors are explicit.
        normalized_record_dir = os.path.abspath(os.path.expanduser(record_dir))
        try:
            os.makedirs(normalized_record_dir, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Record Directory Error",
                f"Could not prepare record directory:\n{exc}",
            )
            return

        # Derive selected snapshot format from the existing coupled record radio buttons.
        if self.ui.radioButton_mha.isChecked():
            selected_mode = ".mha"
        elif self.ui.radioButton_mhacsv.isChecked():
            selected_mode = ".mha + .csv"
        elif self.ui.radioButton_imagecsv.isChecked():
            selected_mode = "image + .csv"
        else:
            QMessageBox.warning(
                self,
                "Record Mode Missing",
                "Choose .mha, .mha + .csv, or image + .csv before taking a snapshot.",
            )
            return

        # Cache one-shot state before arming the coupler so the accepted packet can be stored.
        self._coupled_snapshot_pending = True
        self._coupled_snapshot_mode = selected_mode
        self._coupled_snapshot_record_dir = normalized_record_dir
        self._suppress_next_coupler_stats_status_update = False
        self._latest_coupled_diff_ms = None

        # UI state change: prevent duplicate requests and show temporary snapshot activity.
        self.ui.pushButton_coupledrecord_snapshot.setEnabled(False)
        self._set_coupled_recording_indicators(active=True)

        # Snapshot gate: let the next valid image-triggered coupled packet pass through.
        self._coupler.request_snapshot()
        if self._coupled_snapshot_timer.isActive():
            self._coupled_snapshot_timer.stop()
        self._coupled_snapshot_timer.start(COUPLED_SNAPSHOT_TIMEOUT_MS)
        self.statusBar().showMessage(
            f"Coupled snapshot requested ({selected_mode}). Waiting for one valid packet...",
            9000,
        )

    # Summary:
    # - Slot function for the Record button.
    # - What it does: Toggles coupled recording, validates prerequisites on start,
    #   and controls the CoupledStreamController recording gate.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_coupledrecord_recordStream_clicked(self) -> None:
        # Slot function: react to the Record button click.
        # Validation: avoid starting/stopping continuous recording while a one-shot request is active.
        if self._coupled_snapshot_pending:
            QMessageBox.warning(
                self,
                "Snapshot Pending",
                "Wait for the coupled snapshot to finish before recording.",
            )
            return
        # Toggle behavior: stop immediately if image+csv recording is currently active.
        if self._coupled_imagecsv_active:
            self._stop_coupled_imagecsv_recording("Recording stopped.")
            return
        # Toggle behavior: stop immediately if coupled recording is currently active.
        if self._coupler.is_recording():
            final_path, csv_path, had_active_mha_writer = (
                self._stop_active_mha_recording(
                    reason="Recording stopped.",
                    show_error_dialog=True,
                )
            )
            if final_path and csv_path:
                self.statusBar().showMessage(
                    f"Coupled recording saved to: {final_path} (CSV: {csv_path})",
                    9000,
                )
            elif final_path:
                self.statusBar().showMessage(
                    f"Coupled recording saved to: {final_path}",
                    9000,
                )
            elif csv_path:
                self.statusBar().showMessage(
                    f"Coupled CSV saved to: {csv_path}",
                    9000,
                )
            elif not had_active_mha_writer:
                self.statusBar().showMessage(
                    "Coupled recording stopped.",
                    5000,
                )
            return

        # Validation: ensure a record directory is set before proceeding.
        record_dir = self.ui.lineEdit_coupledrecord_recorddir.text().strip()
        if not record_dir:
            QMessageBox.warning(
                self,
                "Record Directory Missing",
                "Please choose a record directory before recording.",
            )
            return

        # Validation: require both sources to be streaming before recording.
        bmode_state = self.ui.label_status_bmode.text().strip().lower()
        mocap_state = self.ui.label_status_mocap.text().strip().lower()
        if "streaming" not in bmode_state or "streaming" not in mocap_state:
            QMessageBox.warning(
                self,
                "Streaming Required",
                "Start both B-mode and Mocap streaming before recording.",
            )
            return

        # Defensive check: radio buttons might not exist yet in the UI.
        if (
            not hasattr(self.ui, "radioButton_imagecsv")
            or not hasattr(self.ui, "radioButton_mha")
            or not hasattr(self.ui, "radioButton_mhacsv")
        ):
            QMessageBox.warning(
                self,
                "Record Mode Missing",
                "Record mode controls are not available in the current UI.",
            )
            return

        # Normalize/create output directory early so writer startup errors are explicit.
        normalized_record_dir = os.path.abspath(os.path.expanduser(record_dir))
        try:
            os.makedirs(normalized_record_dir, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Record Directory Error",
                f"Could not prepare record directory:\n{exc}",
            )
            return

        # Derive selected mode and initialize only the writers needed for that mode.
        if self.ui.radioButton_imagecsv.isChecked():
            selected_mode = "image + .csv"
            self._mha_writer = None
            self._active_record_mode = selected_mode
            # Start the image+csv coupled session and return early on failure.
            if not self._start_coupled_imagecsv_recording(normalized_record_dir):
                self._active_record_mode = None
            return
        if self.ui.radioButton_mha.isChecked() or self.ui.radioButton_mhacsv.isChecked():
            # Hybrid mode uses the exact same MHA writer path as the MHA-only option.
            if self.ui.radioButton_mhacsv.isChecked():
                selected_mode = ".mha + .csv"
            else:
                selected_mode = ".mha"
            output_mha_path = self._build_mha_output_path(normalized_record_dir)
            writer = MhaWriter(invert_transforms=False)
            # writer = MhaWriter(invert_transforms=True)
            try:
                writer.start(output_mha_path)
            except Exception as exc:
                # Cleanup state when startup fails before recording can start.
                self._mha_writer = None
                self._active_record_mode = None
                QMessageBox.warning(
                    self,
                    "MHA Start Failed",
                    f"Could not start .mha recording:\n{exc}",
                )
                return
            self._mha_writer = writer
            self._active_record_mode = selected_mode

            self._mha_recording_error = ""
            # Hybrid CSV rows are supplied only by the accepted coupled-packet handler.
            if selected_mode == ".mha + .csv":
                try:
                    self._coupled_csv_writer = self._mocap_widget.start_coupled_csv_record(
                        normalized_record_dir
                    )
                except Exception as exc:
                    # The CSV constructor closes on failure; discard the empty MHA session.
                    try:
                        writer.finalize()
                    except Exception:
                        pass
                    self._mha_writer = None
                    self._active_record_mode = None
                    self._mhacsv_csv_path = ""
                    QMessageBox.warning(
                        self,
                        "CSV Record Start Failed",
                        f"Could not start mocap CSV recording:\n{exc}",
                    )
                    return
                self._mhacsv_csv_path = self._coupled_csv_writer.file_path
        else:
            self._mha_writer = None
            self._active_record_mode = None
            QMessageBox.warning(
                self,
                "Record Mode Missing",
                "Choose .mha, .mha + .csv, or image + .csv before recording.",
            )
            return

        # Start coupling session so incoming image packets produce coupled packets.
        self._coupler.start_recording()
        self._latest_coupled_diff_ms = None
        # UI state change: show stop intent while coupled recording is active.
        self.ui.pushButton_coupledrecord_recordStream.setText("Stop Recording")
        # UI state change: show both embedded recording borders while .mha recording is active.
        self._set_coupled_recording_indicators(active=True)
        # UI state change: provide immediate debug text for active mode/session.
        self.statusBar().showMessage(
            f"Coupled recording started ({selected_mode}). Waiting for coupled packets..."
        )

    # Summary:
    # - Build a timestamped output path for one coupled sequence `.mha` file.
    # - What it does: Normalizes the record directory and creates a deterministic Plus-like filename.
    # - Input: `self`, `record_dir` (str).
    # - Returns: Full `.mha` output path (str).
    def _build_mha_output_path(self, record_dir: str) -> str:
        # Normalize path so all writer calls receive a stable absolute directory.
        normalized_dir = os.path.abspath(os.path.expanduser(record_dir))
        # Filename format: keep a compact timestamp so saved MHA files sort chronologically.
        timestamp_text = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"SequenceRecording_{timestamp_text}.mha"
        output_path = os.path.join(normalized_dir, filename)
        # A quick stop/restart in the same second must not overwrite the previous pair.
        suffix = 1
        while os.path.exists(output_path):
            output_path = os.path.join(normalized_dir, f"SequenceRecording_{timestamp_text}_{suffix:03d}.mha")
            suffix += 1
        return output_path

    # Summary:
    # - Finalize and clear the active MHA-backed writer session if one is running.
    # - What it does: Calls `MhaWriter.finalize()` with error handling, updates status/UI feedback,
    #   and always clears writer state so future recordings start cleanly.
    # - Input: `self`, `show_success_message` (bool), `show_error_dialog` (bool).
    # - Returns: Final path on success, otherwise None (Optional[str]).
    def _finalize_active_mha_recording(
        self,
        show_success_message: bool,
        show_error_dialog: bool,
    ) -> Optional[str]:
        final_path = None

        # No-op when current record mode is not MHA-backed or writer was never created.
        if (
            self._active_record_mode not in (".mha", ".mha + .csv")
            or self._mha_writer is None
        ):
            self._mha_writer = None
            self._active_record_mode = None
            return None

        try:
            final_path = self._mha_writer.finalize()
            if show_success_message:
                # UI state change: confirm final file path for quick operator verification.
                self.statusBar().showMessage(
                    f"Coupled .mha saved: {final_path}",
                    9000,
                )
        except Exception as exc:
            # Surface finalize failures because they can indicate incomplete recording output.
            self._mha_recording_error = f"MHA finalization failed: {exc}"
            self.statusBar().showMessage(
                f"Failed to finalize coupled .mha recording: {exc}",
                9000,
            )
            if show_error_dialog:
                QMessageBox.warning(
                    self,
                    "MHA Finalize Error",
                    f"Could not finalize .mha recording:\n{exc}",
                )
        finally:
            # Always clear mode/writer references to avoid stale state on the next session.
            self._mha_writer = None
            self._active_record_mode = None

        return final_path

    # Summary:
    # - Handle the main-window close event.
    # - What it does: Forces both status labels back to disconnected so UI state is consistent on shutdown.
    # - Input: `self`, `event` (Qt close event object).
    # - Returns: None.
    def closeEvent(self, event) -> None:
        # UI state change: reset both labels during shutdown for deterministic final state.
        self.ui.label_status_bmode.setText("Disconnected")
        self.ui.label_status_mocap.setText("Disconnected")
        # Stop image+csv recording so external writers can flush before shutdown.
        if self._coupled_imagecsv_active:
            self._stop_coupled_imagecsv_recording(
                "Recording stopped because the window closed.",
                drain_ms=0,
            )
        # Stop coupling so queued packets are ignored during window teardown.
        if self._coupler.is_recording():
            # Finalize best-effort on close to keep the current recording recoverable.
            self._stop_active_mha_recording(
                reason="Recording stopped because the window closed.",
                show_error_dialog=False,
            )
        # Cancel a pending snapshot so queued image packets do not write during teardown.
        if self._coupled_snapshot_pending:
            self._coupler.cancel_snapshot()
            self._reset_coupled_snapshot_state()
        # UI state change: restore record button text on shutdown.
        self.ui.pushButton_coupledrecord_recordStream.setText("Record")
        # UI state change: final safety reset for both embedded recording borders on shutdown.
        self._set_coupled_recording_indicators(active=False)

        # Delegate to Qt's default close handling after local cleanup.
        super().closeEvent(event)
        event.accept()


# Summary:
# - Entry point for launching the acquisition main window.
# - What it does: Creates the QApplication, shows the AcquisitionMainWindow, and starts the Qt loop.
# - Input: None.
# - Returns: Process exit code (int).
def main() -> int:
    # Create the Qt application instance so widgets can be shown.
    app = QApplication(sys.argv)
    # Create and show the acquisition main window wrapper.
    window = AcquisitionMainWindow()
    window.show()
    # Run the Qt event loop and return its exit code.
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
