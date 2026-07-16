"""Qt QObject runner for the external volume reconstruction process."""

# Standard library helpers for paths and timestamps.
import os
from datetime import datetime
from typing import Optional

# Qt core types for process control and signals.
from PySide6.QtCore import QObject, QProcess, Signal

DEFAULT_VOLUME_RECONSTRUCTOR_EXE = "C:/Users/DennisChristie/PlusApp-2.9.0.20240320-Win64/bin/VolumeReconstructor.exe"
VOLUME_RECONSTRUCTOR_EXE_ENV = "PLUS_VOLUME_RECONSTRUCTOR_EXE"
DEFAULT_RECON_OUTPUT_TAIL_LIMIT = 8000


# Summary:
# - QObject helper that runs the external volume reconstruction process.
# - What it does: owns the QProcess, builds arguments, streams output, and emits lifecycle signals.
class VolumeReconstructionRunner(QObject):
    sig_started = Signal()
    sig_stopped = Signal()
    sig_succeeded = Signal(str)
    sig_failed = Signal(str)
    sig_stdout = Signal(str)
    sig_stderr = Signal(str)

    # Summary:
    # - Initialize the runner and its internal process state.
    # - Input: `self`, `parent` (QObject | None), `output_tail_limit` (int).
    # - Returns: None.
    def __init__(
        self,
        parent: Optional[QObject] = None,
        output_tail_limit: int = DEFAULT_RECON_OUTPUT_TAIL_LIMIT,
    ) -> None:
        super().__init__(parent)
        # Give the runner a stable object name for slot naming conventions.
        self.setObjectName("volumeReconstructionRunner")

        # Track the active QProcess instance for lifecycle control.
        self._process: Optional[QProcess] = None
        # Track the output path for the current run.
        self._output_path = ""
        # Track a small stderr tail for failure messages.
        self._stderr_tail = ""
        # Track whether the user requested a stop.
        self._stop_requested = False
        # Store the maximum tail length for stderr buffering.
        self._output_tail_limit = output_tail_limit

    # Summary:
    # - Check whether a reconstruction process is currently running.
    # - Input: `self`.
    # - Returns: True when a QProcess exists and is running; otherwise False.
    def is_running(self) -> bool:
        # Guard against missing process handles.
        if self._process is None:
            return False
        return self._process.state() != QProcess.NotRunning

    # Summary:
    # - Start the external reconstruction process asynchronously.
    # - Input: `self`, `config_xml` (str), `sequence_mha` (str), `output_dir` (str).
    # - Returns: None.
    def start_reconstruct(
        self, config_xml: str, sequence_mha: str, output_dir: str
    ) -> None:
        # Guard against starting a second run while one is active.
        if self.is_running():
            self.sig_failed.emit("Reconstruction is already running.")
            return

        # Cleanup any stale process handle from previous runs.
        if self._process is not None:
            self._cleanup_recon_process()

        # Reset state for a fresh run.
        self._stop_requested = False
        self._stderr_tail = ""

        # Prefer an environment override so deployments can configure the executable location.
        env_override = os.getenv(VOLUME_RECONSTRUCTOR_EXE_ENV)
        exe_path = env_override.strip() if env_override else DEFAULT_VOLUME_RECONSTRUCTOR_EXE
        exe_path = os.path.abspath(exe_path)

        # Validation: ensure the executable exists before attempting to start.
        if not os.path.isfile(exe_path):
            self.sig_failed.emit(f"The executable was not found:\n{exe_path}")
            return

        # Normalize inputs so the command line receives stable absolute paths.
        config_path = os.path.abspath(config_xml)
        seq_path = os.path.abspath(sequence_mha)
        out_dir = os.path.abspath(output_dir)

        # Guard against missing output folders so we can surface a clear failure message.
        if not os.path.isdir(out_dir):
            self.sig_failed.emit(f"The output folder was not found:\n{out_dir}")
            return

        # Build a unique output path for this reconstruction run.
        output_path = self._make_unique_output_volume_path(out_dir)

        # Create a fresh QProcess so signals are scoped to this run.
        process = QProcess(self)
        process.setObjectName("volumeReconstructProcess")
        self._process = process

        # Signal connection: announce when the process actually starts.
        process.started.connect(self._on_volumeReconstructProcess_started)
        # Signal connection: stream stdout text chunks.
        process.readyReadStandardOutput.connect(
            self._on_volumeReconstructProcess_stdout_ready
        )
        # Signal connection: stream stderr text chunks.
        process.readyReadStandardError.connect(
            self._on_volumeReconstructProcess_stderr_ready
        )
        # Signal connection: handle process-level errors (failed start or crash).
        process.errorOccurred.connect(
            self._on_volumeReconstructProcess_error_occurred
        )
        # Signal connection: handle process completion.
        process.finished.connect(self._on_volumeReconstructProcess_finished)

        # Build the arguments exactly as expected by the CLI tool.
        arguments = [
            f"--config-file={config_path}",
            f"--source-seq-file={seq_path}",
            f"--output-volume-file={output_path}",
            "--image-to-reference-transform=ImageToReference",
            "--disable-compression",
        ]

        # Async start: launch the process without blocking the UI thread.
        process.start(exe_path, arguments)

    # Summary:
    # - Request the running reconstruction process to stop.
    # - Input: `self`.
    # - Returns: None.
    def stop(self) -> None:
        # Guard against stop requests when no process exists.
        if self._process is None:
            return
        # Guard against stop requests when the process already exited.
        if self._process.state() == QProcess.NotRunning:
            return
        # Track user intent so the finish handler can emit sig_stopped.
        self._stop_requested = True
        # Async stop: request termination without blocking the UI.
        self._process.terminate()

    # Summary:
    # - Build a unique output volume file path inside the output folder.
    # - Input: `self`, `out_dir` (str).
    # - Returns: Unique output file path (str).
    def _make_unique_output_volume_path(self, out_dir: str) -> str:
        # Use a compact timestamped base name so each run keeps its own output.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"VolumeOutput_{timestamp}"
        candidate = os.path.join(out_dir, f"{base_name}.mha")

        # If a file exists, append a numeric suffix until we find a free name.
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(out_dir, f"{base_name}_{counter}.mha")
            counter += 1

        # Track the output path so finish handlers know where to load from.
        self._output_path = candidate
        return candidate

    # Summary:
    # - Slot function that emits when the process successfully starts.
    # - Input: `self`.
    # - Returns: None.
    def _on_volumeReconstructProcess_started(self) -> None:
        # This is a slot function for QProcess started signals.
        self.sig_started.emit()

    # Summary:
    # - Slot function that handles reconstruction stdout output.
    # - Input: `self`.
    # - Returns: None.
    def _on_volumeReconstructProcess_stdout_ready(self) -> None:
        # This is a slot function for QProcess stdout readiness.
        if self._process is None:
            return

        # Read any available stdout bytes from the process.
        data = self._process.readAllStandardOutput()
        if not data:
            return

        # Decode and emit the chunk so the UI can display it.
        text = bytes(data).decode(errors="replace")
        self.sig_stdout.emit(text)

    # Summary:
    # - Slot function that handles reconstruction stderr output.
    # - Input: `self`.
    # - Returns: None.
    def _on_volumeReconstructProcess_stderr_ready(self) -> None:
        # This is a slot function for QProcess stderr readiness.
        if self._process is None:
            return

        # Read any available stderr bytes from the process.
        data = self._process.readAllStandardError()
        if not data:
            return

        # Decode and emit the chunk so the UI can display it.
        text = bytes(data).decode(errors="replace")
        self.sig_stderr.emit(text)

        # Keep a limited tail for failure messages.
        self._stderr_tail += text
        if len(self._stderr_tail) > self._output_tail_limit:
            self._stderr_tail = self._stderr_tail[-self._output_tail_limit :]

    # Summary:
    # - Slot function that handles reconstruction process errors.
    # - Input: `self`, `process_error` (QProcess.ProcessError).
    # - Returns: None.
    def _on_volumeReconstructProcess_error_occurred(
        self, process_error: QProcess.ProcessError
    ) -> None:
        # This is a slot function for QProcess error signals.
        if self._process is None:
            return

        # Collect diagnostics to help the user debug failed starts.
        exe_path = self._process.program()
        error_text = self._process.errorString()
        stderr_tail = self._stderr_tail or "(no stderr captured)"

        # Build a concise error message for the UI to display.
        message = "\n".join(
            [
                f"Executable: {exe_path}",
                f"Error: {process_error} ({error_text})",
                "Stderr (tail):",
                stderr_tail,
            ]
        )

        # Cleanup before emitting so the UI sees a non-running state.
        self._cleanup_recon_process()
        self.sig_failed.emit(message)

    # Summary:
    # - Slot function that handles reconstruction completion.
    # - Input: `self`, `exit_code` (int), `exit_status` (QProcess.ExitStatus).
    # - Returns: None.
    def _on_volumeReconstructProcess_finished(
        self, exit_code: int, exit_status: QProcess.ExitStatus
    ) -> None:
        # This is a slot function for QProcess finished signals.
        if self._process is None:
            return

        # Capture the output path before cleanup.
        output_path = self._output_path

        # Handle user-requested stops separately from failures.
        if self._stop_requested:
            self._cleanup_recon_process()
            self.sig_stopped.emit()
            return

        # Validate normal exit before attempting to load outputs.
        if exit_status != QProcess.NormalExit or exit_code != 0:
            stderr_tail = self._stderr_tail or "(no stderr captured)"
            message = "\n".join(
                [
                    f"Exit status: {exit_status}",
                    f"Exit code: {exit_code}",
                    "Stderr (tail):",
                    stderr_tail,
                ]
            )
            self._cleanup_recon_process()
            self.sig_failed.emit(message)
            return

        # Verify the output file exists before reporting success.
        if not output_path or not os.path.isfile(output_path):
            message = f"The output volume file was not found:\n{output_path}"
            self._cleanup_recon_process()
            self.sig_failed.emit(message)
            return

        # Cleanup before emitting so the UI sees a non-running state.
        self._cleanup_recon_process()
        self.sig_succeeded.emit(output_path)

    # Summary:
    # - Cleanup the reconstruction QProcess to avoid stale signal emissions.
    # - Input: `self`.
    # - Returns: None.
    def _cleanup_recon_process(self) -> None:
        # Guard against cleanup when the process is already cleared.
        if self._process is None:
            return

        # Allow Qt to delete the process safely on the event loop.
        self._process.deleteLater()
        self._process = None

        # Reset run-scoped state so the next run starts clean.
        self._output_path = ""
        self._stderr_tail = ""
        self._stop_requested = False
