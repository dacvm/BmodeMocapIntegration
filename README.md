# B-mode Streaming

PySide6 desktop tools for tracked freehand ultrasound in the SonoSkin setup. They stream B-mode images from a USB frame grabber, stream rigid-body poses from a Qualisys motion capture system, record both streams matched by timestamp, and reconstruct 3D volumes with the PLUS Toolkit.

## Overview

```
Frame grabber (OpenCV) ──► B-mode stream ─┐
                                          ├─► CoupledStreamController ─► .mha (+ .csv) ─► PLUS VolumeReconstructor ─► volume .mha
Qualisys QTM (qtm-rt)  ──► Mocap stream ──┘     (matched by timestamp)
```

`CoupledStreamController` pairs each image with the nearest pose. Pairs further apart than `COUPLED_MAXDIFF_IMAGEPOSE_MS` (default 120 ms) are dropped.

## Applications

| Entry point | Purpose |
|---|---|
| `main_acquisition.py` | B-mode and mocap side by side, with matched recording (`.mha`, `.mha + .csv`, or `image + .csv`) and single-frame snapshots. |
| `main_reconstruction.py` | Acquisition plus the volume reconstruction view. |
| `main_bmode.py` | B-mode live preview and image recording on its own. |
| `main_mocap.py` | Qualisys 6D streaming, 3D pose plot and CSV recording on its own. |
| `main_volume.py` | Runs PLUS volume reconstruction on a recorded sequence and previews the result. |

> [!WARNING]
> `main_volume.py` has not been updated in a long time. It may not match the current recording formats or the other widgets, and its defaults point to sample files (`seqs/...`) that may not exist. Use it with care. `main_reconstruction.py` embeds this widget, so the same applies to its volume view.

## Requirements

- Windows (development and testing were done on Windows 10)
- Python 3.13
- Python packages from [requirements.txt](requirements.txt): `PySide6`, `opencv-python`, `numpy`, `matplotlib`, `qtm-rt`
- External software and hardware:
  - [PLUS Toolkit](https://plustoolkit.github.io/) 2.9, for `VolumeReconstructor.exe`
  - Qualisys Track Manager (QTM) with real-time streaming enabled (port `22223`, protocol 1.20)
  - A USB video frame grabber (for example Epiphan) connected to the ultrasound scanner

## Installation

With conda:

```bash
conda create -n bmode_streaming python=3.13
```

```bash
conda activate bmode_streaming
```

```bash
pip install -r requirements.txt
```

## Usage

Run the app you need from the project root:

```bash
python main_acquisition.py
```

A typical acquisition session:

1. Start QTM and make sure the probe and reference rigid bodies are defined and tracked.
2. In the **B-mode** panel, pick a **Calib File** from `configs/`, choose the source (**Stream Image** from a USB port, or **Stream Screen (This PC)**), and click **Open Stream**.
3. In the **Motion Capture** panel, select **Qualisys**, enter the QTM host IP (default `127.0.0.1`), and connect.
4. Under **Coupled Recording**, set the **Record Dir** and **Record Type**, then click **Record**. **Snapshot** saves a single matched frame instead.
5. Stop recording. The writer finalizes the output files.

## Configuration

- **PLUS device-set configs** (`configs/*.xml`): fCal calibration files with the image-to-probe transforms and the video device settings. `PlusDeviceSet_fCal_..._Example.xml` is a template.
- **VolumeReconstructor path**: set the `PLUS_VOLUME_RECONSTRUCTOR_EXE` environment variable to your `VolumeReconstructor.exe`. Without it, the hardcoded default in `helpers/volume_reconstruction_runner.py` is used, which only exists on the original development machine.
- **Matching and buffer constants**: `COUPLED_MAXDIFF_IMAGEPOSE_MS`, `COUPLED_MOCAP_BUFFER_MAXLEN` and `COUPLED_SNAPSHOT_TIMEOUT_MS` at the top of `main_acquisition.py` and `main_reconstruction.py`.

## Output formats

| Record type | Output |
|---|---|
| `.mha` | One Plus Sequence Metafile. |
| `.mha + .csv` | The `.mha` plus a pose CSV. |
| `image + .csv` | A `coupled_<YYYYMMDD_HHMMSS>/` session folder with image files and a pose CSV. |

- **`.mha` (Plus Sequence Metafile)**: written by `helpers/mha_writer.py`. Each frame stores the image plus `ProbeToTrackerDeviceTransform` and `ReferenceToTrackerDeviceTransform` with their status fields. These files can be passed straight to PLUS tools. Snapshots are saved as `SequenceSnapshot_<timestamp>.mha`.
- **`.csv`**: columns are `utc_epoch_ms`, then for each rigid body `<name>_q1..q4` (quaternion w, x, y, z) and `<name>_t1..t3` (translation).

## Project structure

```
main_*.py          Application entry points and their widgets
helpers/           Stream coupling, .mha read/write, XML config parsing, reconstruction runner
ui/                Qt Designer forms (*.ui) and generated Python classes (*_ui.py)
configs/           PLUS device-set / fCal calibration XML files
docs/qualisys/     QTM real-time API notes and example scripts
```

Don't edit the files in `ui/*_ui.py` by hand. Change the `.ui` file in Qt Designer and regenerate it, for example:

```bash
pyside6-uic ui/app_acquisition.ui -o ui/app_acquisition_ui.py
```

Form to generated file mapping: `app_acquisition.ui` → `app_acquisition_ui.py`, `app_reconstruction.ui` → `app_reconstruction_ui.py`, `bmode_v2.ui` → `bmode_ui.py`, `mocap_v2.ui` → `mocap_ui.py`, `volume.ui` → `volume_ui.py`.
