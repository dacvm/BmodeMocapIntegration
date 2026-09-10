"""Streaming Plus Sequence Metafile (.mha) writer for coupled image+pose packets."""

# Standard library helpers for filesystem-safe streaming writes.
import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional

# Numeric helpers for matrix validation, finite checks, and inversions.
import numpy as np


# Summary:
# - Lightweight metadata record for one stored sequence frame.
# - What it does: Keeps only per-frame text metadata in RAM while image payload bytes are
#   streamed to a temporary raw file.
# - Input: frame index, timestamp, transforms, and status fields.
# - Returns: Dataclass instance used during final header generation.
@dataclass
class _FrameMetadata:
    frame_index: int
    timestamp_s: float
    probe_transform: np.ndarray
    probe_status: str
    ref_transform: np.ndarray
    ref_status: str
    image_status: str


# Summary:
# - Streaming writer for Plus Toolkit sequence metafiles (.mha) from coupled packets.
# - What it does: Writes frame payload bytes to a temporary raw file during recording and
#   later assembles a final CRLF-terminated sequence header plus local payload on finalize.
# - Input: optional header defaults, transform body names, and invert-transforms flag.
# - Returns: Writer instance with start/append/finalize lifecycle methods.
class MhaWriter:
    # Summary:
    # - Initialize writer configuration and per-session state.
    # - What it does: Stores header defaults, transform options, and prepares empty runtime state.
    # - Input: header fields and behavior flags used during file generation.
    # - Returns: None.
    def __init__(
        self,
        object_type: str = "Image",
        ndims: int = 3,
        binary_data: bool = True,
        binary_data_byte_order_msb: bool = False,
        compressed_data: bool = False,
        transform_matrix: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        offset: tuple[float, ...] = (0.0, 0.0, 0.0),
        center_of_rotation: tuple[float, ...] = (0.0, 0.0, 0.0),
        anatomical_orientation: str = "RAI",
        element_spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
        element_type: str = "MET_UCHAR",
        kinds: str = "domain domain list",
        ultrasound_image_orientation: str = "MFA",
        ultrasound_image_type: str = "BRIGHTNESS",
        invert_transforms: bool = False,
        probe_body_name: str = "B_N_PRB",
        ref_body_name: str = "B_N_REF",
        flush_every_n_frames: int = 100,
    ) -> None:
        # Header defaults follow the Plus sequence metafile field conventions.
        self._object_type = str(object_type)
        self._ndims = int(ndims)
        self._binary_data = bool(binary_data)
        self._binary_data_byte_order_msb = bool(binary_data_byte_order_msb)
        self._compressed_data = bool(compressed_data)
        self._transform_matrix = tuple(float(value) for value in transform_matrix)
        self._offset = tuple(float(value) for value in offset)
        self._center_of_rotation = tuple(float(value) for value in center_of_rotation)
        self._anatomical_orientation = str(anatomical_orientation)
        self._element_spacing = tuple(float(value) for value in element_spacing)
        self._element_type = str(element_type)
        self._kinds = str(kinds)
        self._ultrasound_image_orientation = str(ultrasound_image_orientation)
        self._ultrasound_image_type = str(ultrasound_image_type)

        # Transform extraction config chooses body names and optional inversion.
        self._invert_transforms = bool(invert_transforms)
        self._probe_body_name = str(probe_body_name)
        self._ref_body_name = str(ref_body_name)

        # Flush cadence limits syscall overhead during high-rate appends.
        self._flush_every_n_frames = max(1, int(flush_every_n_frames))

        # Runtime session state is reset on start/finalize.
        self._output_mha_path: Optional[str] = None
        self._temp_payload_path: Optional[str] = None
        self._payload_fp = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._t0_image_ts_ms: Optional[int] = None
        self._payload_bytes_written = 0
        self._frames_since_flush = 0
        self._frame_metadata: list[_FrameMetadata] = []

    # Summary:
    # - Start a new streaming write session.
    # - What it does: Validates output location, opens a temporary raw payload file, and resets counters.
    # - Input: `output_mha_path` (str) final .mha destination path.
    # - Returns: None.
    def start(self, output_mha_path: str) -> None:
        # Guard against overlapping sessions because one writer handles one file at a time.
        if self._payload_fp is not None:
            raise RuntimeError("MhaWriter session already started")

        normalized_output_path = os.path.abspath(os.path.expanduser(str(output_mha_path)))
        output_dir = os.path.dirname(normalized_output_path) or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        # Reset all runtime fields so restarts never leak previous session data.
        self._frame_metadata = []
        self._width = None
        self._height = None
        self._t0_image_ts_ms = None
        self._payload_bytes_written = 0
        self._frames_since_flush = 0

        # Keep final and temporary paths for finalize-time assembly.
        self._output_mha_path = normalized_output_path
        temp_name = f".__seq_payload_{uuid.uuid4().hex}.raw"
        self._temp_payload_path = os.path.join(output_dir, temp_name)
        self._payload_fp = open(self._temp_payload_path, "wb")

    # Summary:
    # - Append one coupled packet into the active sequence write session.
    # - What it does: Validates image payload and rigid body transforms, streams image bytes to temp
    #   storage, and caches small per-frame metadata for final header generation.
    # - Input: coupled packet fields (`image_ts_ms`, `image_data`, `rigidbody_ts_ms`, `rigidbody_data`).
    # - Returns: True for a stored frame, False when image validation skips it.
    def append_coupled_packet(
        self,
        image_ts_ms: int,
        image_data: object,
        rigidbody_ts_ms: int,
        rigidbody_data: object,
    ) -> bool:
        # Ensure append is only used after start() and before finalize().
        if self._payload_fp is None or self._temp_payload_path is None:
            raise RuntimeError("MhaWriter session not started")

        _ = rigidbody_ts_ms
        width, height, image_bytes = self._extract_image_payload(image_data)
        # Skip packets we cannot validate as single-channel uint8 frame payloads.
        if image_bytes is None:
            return False

        if self._width is None or self._height is None:
            # First accepted frame defines sequence image dimensions and local time zero.
            if width is None or height is None or width <= 0 or height <= 0:
                return False
            self._width = int(width)
            self._height = int(height)
            self._t0_image_ts_ms = int(image_ts_ms)
        else:
            # Enforce fixed dimensions for all stored frames to keep DimSize valid.
            if width is not None and int(width) != self._width:
                return False
            if height is not None and int(height) != self._height:
                return False

        expected_size = int(self._width) * int(self._height)
        # Skip incomplete payload packets so sequence payload size stays exact.
        if len(image_bytes) != expected_size:
            return False

        if self._t0_image_ts_ms is None:
            self._t0_image_ts_ms = int(image_ts_ms)
        timestamp_s = (int(image_ts_ms) - int(self._t0_image_ts_ms)) / 1000.0

        # Extract both required body transforms using strict finite 4x4 validation.
        probe_transform, probe_status = self._extract_transform(
            rigidbody_data=rigidbody_data,
            body_name=self._probe_body_name,
        )
        ref_transform, ref_status = self._extract_transform(
            rigidbody_data=rigidbody_data,
            body_name=self._ref_body_name,
        )

        # Stream frame payload now so RAM use stays nearly constant for long captures.
        self._payload_fp.write(image_bytes)
        self._payload_bytes_written += len(image_bytes)
        self._frames_since_flush += 1
        if self._frames_since_flush >= self._flush_every_n_frames:
            self._payload_fp.flush()
            self._frames_since_flush = 0

        frame_index = len(self._frame_metadata)
        self._frame_metadata.append(
            _FrameMetadata(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                probe_transform=probe_transform,
                probe_status=probe_status,
                ref_transform=ref_transform,
                ref_status=ref_status,
                image_status="OK",
            )
        )
        return True

    @property
    def frame_count(self) -> int:
        """Return stored frames before finalize resets this session's metadata."""
        return len(self._frame_metadata)

    # Summary:
    # - Finalize the active write session and build the final `.mha` file.
    # - What it does: Writes CRLF header lines, appends temporary payload bytes, validates output size,
    #   removes temporary files, and resets writer state for reuse.
    # - Input: `self`.
    # - Returns: Final output `.mha` path (str).
    def finalize(self) -> str:
        # Ensure finalize is called only for an active session.
        if self._output_mha_path is None or self._temp_payload_path is None:
            raise RuntimeError("MhaWriter session not started")

        output_path = self._output_mha_path
        temp_payload_path = self._temp_payload_path

        # Close active payload handle before copying so bytes on disk are complete.
        if self._payload_fp is not None:
            self._payload_fp.close()
            self._payload_fp = None

        num_frames = len(self._frame_metadata)
        if num_frames <= 0 or self._width is None or self._height is None:
            self._safe_delete_file(temp_payload_path)
            self._reset_session_state()
            raise ValueError("No valid frames were recorded; .mha file was not created")

        header_lines = self._build_header_lines(num_frames=num_frames)
        # CRLF line endings are required for compatibility with common Plus sequence files.
        header_blob = ("\r\n".join(header_lines) + "\r\n").encode("ascii")

        expected_payload_bytes = int(self._width) * int(self._height) * int(num_frames)

        try:
            with open(output_path, "wb") as out_fp:
                out_fp.write(header_blob)
                with open(temp_payload_path, "rb") as payload_fp:
                    shutil.copyfileobj(payload_fp, out_fp, length=1024 * 1024)

            # Sanity checks: file exists and payload size is not truncated.
            if not os.path.exists(output_path):
                raise IOError(f"Final sequence file was not created: {output_path}")
            actual_size = os.path.getsize(output_path)
            minimum_size = len(header_blob) + expected_payload_bytes
            if actual_size < minimum_size:
                raise IOError(
                    "Final sequence file is smaller than expected "
                    f"(actual={actual_size}, minimum={minimum_size})"
                )
        except Exception:
            # Cleanup partial output so failed finalization does not leave corrupt files.
            self._safe_delete_file(output_path)
            raise
        finally:
            # Always remove temporary payload to avoid littering on success or failure.
            self._safe_delete_file(temp_payload_path)
            self._reset_session_state()

        return output_path

    # Summary:
    # - Convert an arbitrary image packet object into (width, height, payload bytes).
    # - What it does: Supports FramePacket-like objects, dict-like payloads, and 2D uint8 arrays.
    # - Input: `image_data` (object) from coupled packets.
    # - Returns: tuple[Optional[int], Optional[int], Optional[bytes]].
    def _extract_image_payload(
        self, image_data: object
    ) -> tuple[Optional[int], Optional[int], Optional[bytes]]:
        width = None
        height = None
        data_obj = None

        # Prefer attribute-style access used by the existing FramePacket payload.
        if hasattr(image_data, "width"):
            width = getattr(image_data, "width")
        if hasattr(image_data, "height"):
            height = getattr(image_data, "height")
        if hasattr(image_data, "data"):
            data_obj = getattr(image_data, "data")

        # Fall back to mapping-style packets if callers pass dictionaries.
        if isinstance(image_data, Mapping):
            if width is None:
                width = image_data.get("width")
            if height is None:
                height = image_data.get("height")
            if data_obj is None:
                data_obj = image_data.get("data")

        normalized_width = self._to_positive_int_or_none(width)
        normalized_height = self._to_positive_int_or_none(height)

        # Accept raw bytes payloads directly when available.
        if isinstance(data_obj, (bytes, bytearray)):
            return normalized_width, normalized_height, bytes(data_obj)
        if isinstance(data_obj, memoryview):
            return normalized_width, normalized_height, data_obj.tobytes()

        # Accept a 2D uint8 array payload without any type conversion.
        if isinstance(data_obj, np.ndarray):
            if data_obj.ndim != 2 or data_obj.dtype != np.uint8:
                return normalized_width, normalized_height, None
            array_2d = np.ascontiguousarray(data_obj)
            return array_2d.shape[1], array_2d.shape[0], array_2d.tobytes(order="C")

        # If packet itself is a 2D uint8 array, use it as the image payload.
        if isinstance(image_data, np.ndarray):
            if image_data.ndim != 2 or image_data.dtype != np.uint8:
                return normalized_width, normalized_height, None
            array_2d = np.ascontiguousarray(image_data)
            return array_2d.shape[1], array_2d.shape[0], array_2d.tobytes(order="C")

        return normalized_width, normalized_height, None

    # Summary:
    # - Extract one rigid body transform matrix with strict validation and optional inversion.
    # - What it does: Reads `body_name` from rigidbody mapping, validates finite 4x4 shape, and
    #   returns either a valid matrix with `OK` status or identity with `INVALID` status.
    # - Input: `rigidbody_data` (object), `body_name` (str).
    # - Returns: tuple[np.ndarray, str].
    def _extract_transform(self, rigidbody_data: object, body_name: str) -> tuple[np.ndarray, str]:
        identity = np.eye(4, dtype=float)

        # Missing or non-mapping rigidbody payload cannot provide named transforms.
        if not isinstance(rigidbody_data, Mapping):
            return identity, "INVALID"

        matrix_candidate = rigidbody_data.get(body_name)
        if matrix_candidate is None:
            return identity, "INVALID"

        try:
            matrix = np.asarray(matrix_candidate, dtype=float)
        except (TypeError, ValueError):
            return identity, "INVALID"

        # Plus transform fields expect full homogeneous 4x4 matrices.
        if matrix.shape != (4, 4):
            return identity, "INVALID"
        if not np.all(np.isfinite(matrix)):
            return identity, "INVALID"

        if self._invert_transforms:
            # Optional inversion allows callers to export the opposite transform direction.
            try:
                matrix = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                return identity, "INVALID"
            if not np.all(np.isfinite(matrix)):
                return identity, "INVALID"

        return matrix.astype(float, copy=False), "OK"

    # Summary:
    # - Build all text header lines, including per-frame sequence fields.
    # - What it does: Generates stable-order lines that satisfy Plus sequence metafile expectations,
    #   placing all `Seq_Frame...` fields before `ElementDataFile = LOCAL`.
    # - Input: `num_frames` (int).
    # - Returns: list[str].
    def _build_header_lines(self, num_frames: int) -> list[str]:
        if self._width is None or self._height is None:
            raise ValueError("Image width/height must be known before header build")

        lines = [
            f"ObjectType = {self._object_type}",
            f"NDims = {self._ndims}",
            f"BinaryData = {self._bool_to_meta_bool(self._binary_data)}",
            f"BinaryDataByteOrderMSB = {self._bool_to_meta_bool(self._binary_data_byte_order_msb)}",
            f"CompressedData = {self._bool_to_meta_bool(self._compressed_data)}",
            f"TransformMatrix = {self._format_values(self._transform_matrix)}",
            f"Offset = {self._format_values(self._offset)}",
            f"CenterOfRotation = {self._format_values(self._center_of_rotation)}",
            f"AnatomicalOrientation = {self._anatomical_orientation}",
            f"ElementSpacing = {self._format_values(self._element_spacing)}",
            f"DimSize = {int(self._width)} {int(self._height)} {int(num_frames)}",
            f"ElementType = {self._element_type}",
            f"Kinds = {self._kinds}",
            f"UltrasoundImageOrientation = {self._ultrasound_image_orientation}",
            f"UltrasoundImageType = {self._ultrasound_image_type}",
        ]

        # Write frame fields in strict frame order for deterministic output.
        for metadata in self._frame_metadata:
            frame_prefix = f"Seq_Frame{metadata.frame_index:04d}"
            lines.append(
                f"{frame_prefix}_ProbeToTrackerDeviceTransform = "
                f"{self._format_matrix_4x4(metadata.probe_transform)}"
            )
            lines.append(
                f"{frame_prefix}_ProbeToTrackerDeviceTransformStatus = {metadata.probe_status}"
            )
            lines.append(
                f"{frame_prefix}_ReferenceToTrackerDeviceTransform = "
                f"{self._format_matrix_4x4(metadata.ref_transform)}"
            )
            lines.append(
                f"{frame_prefix}_ReferenceToTrackerDeviceTransformStatus = {metadata.ref_status}"
            )
            lines.append(
                f"{frame_prefix}_Timestamp = {self._format_float(metadata.timestamp_s)}"
            )
            lines.append(f"{frame_prefix}_ImageStatus = {metadata.image_status}")

        # This must stay last so payload parsing starts right after it.
        lines.append("ElementDataFile = LOCAL")
        return lines

    # Summary:
    # - Convert float-like values into MetaImage-compatible text tokens.
    # - What it does: Applies compact formatting and strips trailing zeros for cleaner headers.
    # - Input: `values` (iterable of float-like).
    # - Returns: One space-delimited string.
    def _format_values(self, values: tuple[float, ...]) -> str:
        return " ".join(self._format_float(float(value)) for value in values)

    # Summary:
    # - Convert a 4x4 matrix into the required row-major 16-value text string.
    # - What it does: Flattens in C-order to preserve row-major transform order expected by Plus.
    # - Input: `matrix` (np.ndarray shape 4x4).
    # - Returns: Space-delimited string with 16 scalar values.
    def _format_matrix_4x4(self, matrix: np.ndarray) -> str:
        matrix_4x4 = np.asarray(matrix, dtype=float).reshape((4, 4))
        return " ".join(self._format_float(float(value)) for value in matrix_4x4.ravel(order="C"))

    # Summary:
    # - Format one numeric scalar for sequence header output.
    # - What it does: Uses compact precision, strips redundant trailing zeros, and avoids '-0'.
    # - Input: `value` (float).
    # - Returns: String representation suitable for `.mha` text headers.
    def _format_float(self, value: float) -> str:
        text = f"{float(value):.9f}".rstrip("0").rstrip(".")
        if text in ("", "-0"):
            return "0"
        return text

    # Summary:
    # - Convert Python bool to MetaImage boolean text.
    # - What it does: Returns `True` or `False` as expected in standard .mha headers.
    # - Input: `value` (bool).
    # - Returns: `True` or `False` string.
    def _bool_to_meta_bool(self, value: bool) -> str:
        return "True" if bool(value) else "False"

    # Summary:
    # - Convert an arbitrary value into a positive integer when possible.
    # - What it does: Safely parses numeric-like values and rejects non-positive or invalid input.
    # - Input: `value` (object).
    # - Returns: Positive integer or None when parsing fails.
    def _to_positive_int_or_none(self, value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return parsed

    # Summary:
    # - Delete one file path without raising on missing/permission errors.
    # - What it does: Performs best-effort cleanup for temporary and partial-output files.
    # - Input: `path` (str).
    # - Returns: None.
    def _safe_delete_file(self, path: str) -> None:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            return

    # Summary:
    # - Reset runtime session fields after finalize or failed startup.
    # - What it does: Clears file handles, paths, counters, and per-frame metadata.
    # - Input: `self`.
    # - Returns: None.
    def _reset_session_state(self) -> None:
        self._output_mha_path = None
        self._temp_payload_path = None
        self._payload_fp = None
        self._width = None
        self._height = None
        self._t0_image_ts_ms = None
        self._payload_bytes_written = 0
        self._frames_since_flush = 0
        self._frame_metadata = []
