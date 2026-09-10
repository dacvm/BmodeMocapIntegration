"""Exercise real recording paths with synthetic packets; no cameras or QTM needed."""

import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from main_acquisition import AcquisitionMainWindow
from main_bmode import FramePacket
from main_mocap import CoupledCsvWriter


def poses(value):
    """Give each body a distinct, known translation for checking row alignment."""
    result = {}
    for offset, name in enumerate(("B_N_PRB", "B_N_REF", "bone_pin")):
        matrix = np.eye(4)
        matrix[:3, 3] = [value + offset, value * 2, -value]
        result[name] = matrix
    return result


def read_pair(directory):
    """Read the actual output counts, pose metadata and binary frame payload."""
    mha = next(Path(directory).glob("*.mha"))
    header, payload = mha.read_bytes().split(b"ElementDataFile = LOCAL\r\n", 1)
    fields = dict(line.split(" = ", 1) for line in header.decode("ascii").splitlines())
    with next(Path(directory).glob("*.csv")).open(newline="") as file:
        rows = list(csv.DictReader(file))
    return fields, payload, rows


class CoupledRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.warning = self.enterContext(patch("main_acquisition.QMessageBox.warning"))
        # Hardware discovery is unrelated to packet-driven recording tests.
        self.enterContext(patch("main_bmode.BModeWidget._populate_cameras", new=lambda self: None))
        self.window = AcquisitionMainWindow()
        self.addCleanup(self.window.close)
        self.window.ui.label_status_bmode.setText("Streaming")
        self.window.ui.label_status_mocap.setText("Streaming")
        self.window._mocap_widget._stream_state = "streaming"
        self.enterContext(patch.object(
            self.window._mocap_widget._stream_worker, "body_names_snapshot",
            return_value=["B_N_PRB", "B_N_REF", "bone_pin"],
        ))
        self.window.ui.lineEdit_coupledrecord_recorddir.setText(self.temp.name)
        self.window.ui.radioButton_mhacsv.setChecked(True)

    def start(self):
        self.window._on_pushButton_coupledrecord_recordStream_clicked()
        self.assertTrue(self.window._coupler.is_recording())
        # Hybrid mode must never arm the raw QTM recorder.
        self.assertFalse(self.window._mocap_widget._record_worker.is_active())

    def image(self, ts, data=b"\x01\x02\x03\x04", width=2):
        packet = FramePacket(ts, width, 2, "GRAY8", data)
        self.window._coupler.on_image_packet(ts, packet)

    def pose(self, ts, value):
        self.window._coupler.on_rigidbody_packet(ts, poses(value))

    def stop(self):
        self.window._on_pushButton_coupledrecord_recordStream_clicked()
        self.assertFalse(self.window._coupler.is_recording())

    def test_faster_qtm_stream_preserves_selected_poses(self):
        self.start()
        for ts in range(100, 201, 10):
            self.pose(ts, ts)
        self.image(135)  # Must select 130, not the latest available pose at 200.
        self.image(175)
        self.assertEqual(self.window._coupled_csv_writer.row_count, 2)
        self.stop()
        fields, payload, rows = read_pair(self.temp.name)
        self.assertEqual(fields["DimSize"], "2 2 2")
        self.assertEqual(len(payload), 8)
        self.assertEqual(len(rows), 2)
        for index, value in enumerate((130, 170)):
            for body, transform in (("B_N_PRB", "Probe"), ("B_N_REF", "Reference")):
                matrix = np.array([float(x) for x in fields[
                    f"Seq_Frame{index:04d}_{transform}ToTrackerDeviceTransform"
                ].split()]).reshape(4, 4)
                np.testing.assert_allclose(matrix[:3, 3], [float(rows[index][f"{body}_t{i}"]) for i in (1, 2, 3)])
            self.assertEqual(float(rows[index]["bone_pin_t1"]), value + 2)
            self.assertGreater(int(rows[index]["utc_epoch_ms"]), 1_000_000_000_000)
        self.warning.assert_not_called()

    def test_skipped_frames_and_missing_bodies(self):
        self.start()
        self.image(100)  # No pose.
        self.pose(100, 1)
        self.image(221)  # Pose is stale.
        self.image(110, data=b"x")  # Wrong payload length.
        self.image(110)
        self.image(111, width=3, data=b"123456")  # Dimensions changed.
        self.window._coupler.on_rigidbody_packet(120, {"B_N_PRB": np.eye(4)})
        self.image(121)
        self.stop()
        fields, payload, rows = read_pair(self.temp.name)
        self.assertEqual(fields["DimSize"], "2 2 2")
        self.assertEqual(len(payload), 8)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["bone_pin_t1"], "NaN")
        self.assertEqual(rows[1]["B_N_REF_q1"], "NaN")

    def test_stop_restart_has_no_pending_accepted_pairs(self):
        self.start()
        self.pose(100, 1)
        self.image(101)
        # This is already written without pumping Qt's queued events.
        self.assertEqual(self.window._coupled_csv_writer.row_count, 1)
        self.stop()
        self.image(102)  # Gate closed: neither output may receive this packet.
        second = Path(self.temp.name) / "second"
        self.window.ui.lineEdit_coupledrecord_recorddir.setText(str(second))
        self.start()
        self.app.processEvents()
        self.assertEqual(self.window._coupled_csv_writer.row_count, 0)
        self.pose(200, 2)
        self.image(201)
        self.stop()
        self.assertEqual(len(read_pair(self.temp.name)[2]), 1)
        self.assertEqual(len(read_pair(second)[2]), 1)
        self.assertEqual(float(read_pair(second)[2][0]["B_N_PRB_t1"]), 2)

    def test_csv_start_failure_does_not_enable_recording(self):
        with patch.object(self.window._mocap_widget, "start_coupled_csv_record", side_effect=OSError("disk error")):
            self.window._on_pushButton_coupledrecord_recordStream_clicked()
        self.assertFalse(self.window._coupler.is_recording())
        self.assertIsNone(self.window._mha_writer)
        self.assertEqual(list(Path(self.temp.name).glob(".__seq_payload*")), [])
        self.warning.assert_called()

    def test_csv_append_failure_reports_incomplete_and_closes_both(self):
        self.start()
        self.pose(100, 1)
        writer = self.window._coupled_csv_writer
        with patch.object(writer, "append_row", side_effect=OSError("CSV write failed")):
            self.image(101)
        self.assertFalse(self.window._coupler.is_recording())
        self.assertTrue(writer._file.closed)
        self.assertIsNone(self.window._mha_writer)
        self.app.processEvents()
        message = self.window.statusBar().currentMessage()
        self.assertIn("Recording incomplete", message)
        self.assertIn("1 MHA frames, 0 CSV rows", message)

    def test_mha_append_failure_does_not_append_csv(self):
        self.start()
        self.pose(100, 1)
        writer = self.window._coupled_csv_writer
        with patch.object(self.window._mha_writer, "append_coupled_packet", side_effect=OSError("MHA write failed")):
            self.image(101)
        self.assertEqual(writer.row_count, 0)
        self.assertTrue(writer._file.closed)
        self.assertIn("Recording incomplete", self.window.statusBar().currentMessage())

    def test_csv_close_failure_still_finalizes_mha(self):
        self.start()
        self.pose(100, 1)
        self.image(101)
        writer = self.window._coupled_csv_writer
        close = writer.close
        def fail_close():
            close()
            raise OSError("flush failure")
        with patch.object(writer, "close", side_effect=fail_close):
            self.stop()
        self.assertTrue(list(Path(self.temp.name).glob("*.mha")))
        self.assertIn("Recording incomplete", self.window.statusBar().currentMessage())

    def test_mha_finalize_failure_does_not_report_success(self):
        self.start()
        self.pose(100, 1)
        self.image(101)
        writer = self.window._coupled_csv_writer
        finalize = self.window._mha_writer.finalize
        def fail_finalize():
            finalize()
            raise OSError("finalization failure")
        with patch.object(self.window._mha_writer, "finalize", side_effect=fail_finalize):
            self.stop()
        self.assertTrue(writer._file.closed)
        self.assertIn("Recording incomplete", self.window.statusBar().currentMessage())

    def test_disconnect_and_close_finalize_pairs(self):
        for event in ("bmode", "mocap", "close"):
            directory = Path(self.temp.name) / event
            self.window.ui.lineEdit_coupledrecord_recorddir.setText(str(directory))
            self.window.ui.label_status_bmode.setText("Streaming")
            self.window.ui.label_status_mocap.setText("Streaming")
            self.start()
            self.pose(100, 1)
            self.image(101)
            if event == "close":
                self.window.close()
            else:
                getattr(self.window, f"_on_{event}StreamProxy_state_changed")(False, "disconnected")
            self.assertFalse(self.window._coupler.is_recording())
            self.assertEqual(len(read_pair(directory)[2]), 1)

    def test_snapshot_still_writes_one_pair(self):
        self.window._on_pushButton_coupledrecord_snapshot_clicked()
        self.pose(100, 1)
        self.image(101)
        fields, _, rows = read_pair(self.temp.name)
        self.assertEqual(fields["DimSize"], "2 2 1")
        self.assertEqual(len(rows), 1)
        self.assertFalse(self.window._coupled_snapshot_pending)

    def test_mha_only_does_not_open_csv(self):
        self.window.ui.radioButton_mha.setChecked(True)
        self.start()
        self.pose(100, 1)
        self.image(101)
        self.stop()
        self.assertEqual(len(list(Path(self.temp.name).glob("*.mha"))), 1)
        self.assertEqual(list(Path(self.temp.name).glob("*.csv")), [])

    def test_same_second_restart_preserves_both_files(self):
        with patch("main_acquisition.datetime") as clock:
            clock.now.return_value.strftime.return_value = "20260910_120000"
            for value in (1, 2):
                self.start()
                self.pose(100, value)
                self.image(101)
                self.stop()
        self.assertEqual(len(list(Path(self.temp.name).glob("*.mha"))), 2)
        self.assertEqual(len(list(Path(self.temp.name).glob("*.csv"))), 2)

    def test_csv_header_failure_closes_file(self):
        path = Path(self.temp.name) / "header.csv"
        with patch("main_mocap.csv.writer") as make_writer:
            make_writer.return_value.writerow.side_effect = OSError("header error")
            with self.assertRaisesRegex(OSError, "header error"):
                CoupledCsvWriter(str(path), ["B_N_PRB"])
        # On Windows this also verifies the failed constructor released the handle.
        path.unlink()

    def test_independent_csv_recorder_remains_available(self):
        mocap = self.window._mocap_widget
        path = mocap.start_external_csv_record(self.temp.name)
        try:
            self.assertTrue(mocap._record_worker.is_active())
            self.assertIsNone(self.window._coupled_csv_writer)
        finally:
            mocap.stop_external_csv_record()
        self.assertTrue(Path(path).exists())

    def test_image_csv_mode_remains_qtm_driven(self):
        self.window.ui.radioButton_imagecsv.setChecked(True)
        self.window._bmode_widget._latest_frame_packet = FramePacket(100, 2, 2, "GRAY8", b"1234")
        self.window._on_pushButton_coupledrecord_recordStream_clicked()
        self.assertTrue(self.window._coupled_imagecsv_active)
        self.assertFalse(self.window._coupler.is_recording())
        self.assertIsNone(self.window._coupled_csv_writer)
        # Even a QTM packet with no tracked bodies retains its legacy CSV/image row.
        self.window._mocap_widget._record_worker.handle_6d_residual_result(None)
        for _ in range(3):
            self.app.processEvents()
        self.window._stop_coupled_imagecsv_recording(drain_ms=0)
        self.assertEqual(len(list(Path(self.temp.name).rglob("*.jpg"))), 1)
        csv_path = next(Path(self.temp.name).rglob("*.csv"))
        with csv_path.open(newline="") as file:
            self.assertEqual(len(list(csv.DictReader(file))), 1)

    def test_quaternion_sign_continuity_and_frozen_schema(self):
        path = str(Path(self.temp.name) / "sign.csv")
        writer = CoupledCsvWriter(path, ["B_N_PRB"])
        self.addCleanup(writer.close)
        with patch("main_mocap.MocapWidget._pose_matrix_to_record_values", side_effect=[
            [1., 0., 0., 0., 1., 2., 3.], [-1., 0., 0., 0., 4., 5., 6.],
        ]):
            writer.append_row(writer.prepare_row(poses(1)))
            writer.append_row(writer.prepare_row(poses(2)))
        writer.close()
        with open(path, newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]["B_N_PRB_q1"], rows[1]["B_N_PRB_q1"])
        self.assertEqual(len(rows[0]), 8)


if __name__ == "__main__":
    unittest.main()
