#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂操作员界面 (Operator GUI)
目标: 在不改变原有推理核心逻辑的前提下，提供更贴近一线操作员的交互流程。
"""

import sys
import os

# ===== Qt Plugin Conflict Resolution =====
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
try:
    import PyQt5 as _pyqt5
    _qt5_plugins = os.path.join(os.path.dirname(_pyqt5.__file__), "Qt5", "plugins")
    if os.path.isdir(_qt5_plugins):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt5_plugins
        os.environ["QT_PLUGIN_PATH"] = _qt5_plugins
except Exception:
    pass
os.environ["OPENCV_VIDEOIO_PRIORITY_QT"] = "0"

# ===== Skip pip requirements checks (Jetson uses custom NVIDIA wheels) =====
os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"

# ===== Suppress ONNX Runtime warning on Jetson =====
def _preload_onnxruntime_quiet():
    """Import onnxruntime while suppressing C++ stderr warnings."""
    try:
        _fd = os.dup(2)
        with open(os.devnull, "wb") as _devnull:
            os.dup2(_devnull.fileno(), 2)
        try:
            import onnxruntime  # noqa: F401
        finally:
            os.dup2(_fd, 2)
            os.close(_fd)
    except Exception:
        pass


_preload_onnxruntime_quiet()

import time
import logging
import threading
import ast
import json
import platform
from contextlib import contextmanager
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QSlider,
    QGroupBox,
    QMessageBox,
    QSplitter,
    QFrame,
    QStyleFactory,
    QGridLayout,
    QSizePolicy,
    QComboBox,
    QSpinBox,
    QListWidget,
    QDoubleSpinBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

import cv2


@contextmanager
def _silence_stderr():
    """Temporarily silence native stderr output from noisy SDK probes."""
    fd = None
    try:
        fd = os.dup(2)
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    except Exception:
        yield
    finally:
        if fd is not None:
            try:
                os.dup2(fd, 2)
            finally:
                os.close(fd)


@contextmanager
def _silence_stdio():
    """Temporarily silence native stdout/stderr output from noisy SDK probes."""
    fd_out = None
    fd_err = None
    try:
        fd_out = os.dup(1)
        fd_err = os.dup(2)
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    except Exception:
        yield
    finally:
        if fd_out is not None:
            try:
                os.dup2(fd_out, 1)
            finally:
                os.close(fd_out)
        if fd_err is not None:
            try:
                os.dup2(fd_err, 2)
            finally:
                os.close(fd_err)


def _sdk_call_quiet(func, *args, **kwargs):
    with _silence_stdio():
        return func(*args, **kwargs)


def _safe_imread(path):
    """Read image robustly; fallback helps with special filenames/extensions."""
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size > 0:
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None


def _guess_media_type(path):
    """Classify source as image/video by extension first, then content probe."""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".jfif"}
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".m4v", ".ts", ".mts"}

    ext = os.path.splitext(path)[1].lower()
    if ext in image_exts:
        return "image"
    if ext in video_exts:
        return "video"

    if _safe_imread(path) is not None:
        return "image"

    cap = cv2.VideoCapture(path)
    try:
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return "video"
    finally:
        cap.release()

    return None


def _detect_platform():
    """Detect whether running on Jetson."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            if "jetson" in model:
                return "jetson"
    except FileNotFoundError:
        pass
    return "desktop"


PLATFORM = _detect_platform()


def _resolve_runtime_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_project_dir(runtime_dir):
    candidates = [runtime_dir, os.path.dirname(runtime_dir)]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "ultralytics")):
            return path
        if os.path.isfile(os.path.join(path, "pyproject.toml")):
            return path
    return runtime_dir


_SCRIPT_DIR = _resolve_runtime_dir()
_PROJECT_DIR = _resolve_project_dir(_SCRIPT_DIR)

_MINDVISION_PY_DEMO_DIR = os.path.join(_PROJECT_DIR, "mindvision", "demo", "python_demo")
if os.path.isdir(_MINDVISION_PY_DEMO_DIR) and _MINDVISION_PY_DEMO_DIR not in sys.path:
    sys.path.append(_MINDVISION_PY_DEMO_DIR)

try:
    import mvsdk
except Exception:
    mvsdk = None

_WEIGHT_CANDIDATES = [
    "/home/ros/proj/ultralytics/packages/results/runs/detect/results/runs/self499_aug/SCAW/weights/best.engine",
    os.path.join(_PROJECT_DIR, "results", "migration", "final_weights.pt"),
    os.path.join(_PROJECT_DIR, "ptdoc", "selfbest_baseline.pt"),
    os.path.join(_PROJECT_DIR, "yolov8n.pt"),
    os.path.join(_PROJECT_DIR, "yolo11n.pt"),
]
DEFAULT_WEIGHTS = next((p for p in _WEIGHT_CANDIDATES if os.path.isfile(p)), _WEIGHT_CANDIDATES[0])

DEFAULT_CAMERA_INDEX = 0
INITIAL_CONF = 0.75
INITIAL_IOU = 0.45

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("OperatorGUI")

try:
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Colors
except ImportError:
    logger.critical("ultralytics not found – pip install ultralytics")
    sys.exit(1)


class DetectionThread(QThread):
    """QThread for YOLO inference, separated from UI event loop."""

    frame_ready = pyqtSignal(QImage)
    data_ready = pyqtSignal(list)
    stats_ready = pyqtSignal(float, float, int, float, float)  # infer_fps, latency_ms, count, gpu_mem_mb, stream_fps
    status_msg = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished_work = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.weights_path = DEFAULT_WEIGHTS
        self.source_type = None
        self.source_path = None
        self.model = None
        self.class_names = {}
        self._running = False
        self.preview_only = False

        self.conf_thres = INITIAL_CONF
        self.iou_thres = INITIAL_IOU
        self.mm_per_pixel = 1.0
        self.camera_index = DEFAULT_CAMERA_INDEX
        self.use_csi = False
        self.camera_backend = "opencv"

        self.model_format = None
        self.use_fp16 = False
        self.dataset_yaml_path = None
        self._palette = Colors()
        self._gpu_mem_baseline_free = None

        self._last_signal_time = {"frame": 0, "stats": 0, "data": 0}
        self.signal_interval = 0.1 if PLATFORM == "jetson" else 0.05

    @staticmethod
    def _normalize_names(names):
        """Normalize names to dict[int, str] from list/dict/other."""
        if isinstance(names, dict):
            out = {}
            for k, v in names.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    continue
            return out
        if isinstance(names, (list, tuple)):
            return {i: str(v) for i, v in enumerate(names)}
        return {}

    @staticmethod
    def _looks_generic_names(name_map):
        """Return True when names are placeholder labels like class0/class1."""
        if not name_map:
            return True
        vals = [str(v).strip().lower() for _, v in sorted(name_map.items())]
        generic_hits = 0
        for v in vals:
            if v == "class" or v.startswith("class"):
                generic_hits += 1
        return generic_hits >= max(1, int(len(vals) * 0.6))

    def _load_names_from_yaml(self, yaml_path):
        """Load class names from dataset yaml names field."""
        if not yaml_path or not os.path.isfile(yaml_path):
            return {}
        if yaml is None:
            # Minimal fallback parser for simple data.yaml formats.
            try:
                text = Path(yaml_path).read_text(encoding="utf-8")
                for ln in text.splitlines():
                    s = ln.strip()
                    if s.startswith("names:") and "[" in s and "]" in s:
                        _, rhs = s.split(":", 1)
                        parsed = ast.literal_eval(rhs.strip())
                        return self._normalize_names(parsed)
            except Exception:
                return {}
            return {}
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return self._normalize_names(data.get("names", {}))
        except Exception as e:
            logger.warning("Failed to parse dataset yaml %s: %s", yaml_path, e)
            return {}

    def _find_nearby_dataset_yaml(self, model_path):
        """Best-effort search of dataset yaml near exported model artifacts."""
        p = Path(model_path).resolve()
        candidates = []

        # Same dir and several parent levels (covers weights/best.engine -> ../../data.yaml).
        for base in [p.parent, p.parent.parent, p.parent.parent.parent]:
            if str(base) == ".":
                continue
            candidates.append(base / "data.yaml")
            candidates.append(base / "_data_fixed.yaml")

        # Also search under known workspace model directories.
        candidates.append(Path(_PROJECT_DIR) / "models" / "self" / "selfpart499_augmented" / "data.yaml")
        candidates.append(Path(_PROJECT_DIR) / "models" / "open" / "openv2" / "data.yaml")

        for c in candidates:
            if c.is_file():
                return str(c)
        return None

    def load_model(self, path):
        """Load YOLO model with support for .pt/.onnx/.engine."""
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pt":
                self.model_format = "pt"
            elif ext == ".onnx":
                self.model_format = "onnx"
            elif ext == ".engine":
                self.model_format = "engine"
            else:
                self.model_format = "pt"

            self.status_msg.emit("正在加载模型: {}".format(os.path.basename(path)))

            try:
                import torch

                if torch.cuda.is_available():
                    free, _ = torch.cuda.mem_get_info(0)
                    self._gpu_mem_baseline_free = free
            except Exception:
                pass

            self.model = YOLO(path)
            model_names = self._normalize_names(getattr(self.model, "names", {}))
            self.class_names = model_names
            self.weights_path = path

            # Some exported onnx/engine models lose custom labels and only keep class0/class1.
            if self._looks_generic_names(self.class_names):
                yaml_path = self._find_nearby_dataset_yaml(path)
                yaml_names = self._load_names_from_yaml(yaml_path)
                if yaml_names:
                    self.class_names = yaml_names
                    self.dataset_yaml_path = yaml_path
                    logger.info("Class names loaded from dataset yaml: %s", yaml_path)
                else:
                    logger.warning("Using generic class names from model metadata: %s", self.class_names)
            else:
                self.dataset_yaml_path = None

            self.use_fp16 = PLATFORM == "jetson" and self.model_format == "engine"
            self.status_msg.emit(
                "模型就绪: {} [{}{}]".format(
                    os.path.basename(path),
                    self.model_format.upper(),
                    ", FP16" if self.use_fp16 else "",
                )
            )
            logger.info("Model loaded: %s", path)
            return True
        except Exception as e:
            self.error_occurred.emit("模型加载失败: {}".format(e))
            logger.error("load_model failed: %s", e, exc_info=True)
            return False

    def set_source(self, stype, spath=None):
        self.source_type = stype
        self.source_path = spath

    def update_params(self, conf, iou):
        self.conf_thres = conf
        self.iou_thres = iou

    def stop(self):
        self._running = False

    def run(self):
        if self.source_type is None:
            self.error_occurred.emit("请先选择输入源。")
            return
        if not self.preview_only and self.model is None:
            self.error_occurred.emit("请先加载模型。")
            return

        self._running = True

        # Re-capture baseline if thread was recreated before start.
        if self._gpu_mem_baseline_free is None:
            try:
                import torch

                if torch.cuda.is_available():
                    free, _ = torch.cuda.mem_get_info(0)
                    self._gpu_mem_baseline_free = free
            except Exception:
                pass

        self.status_msg.emit("摄像头预览中..." if self.preview_only else "检测运行中...")

        try:
            if self.preview_only:
                self._preview_loop()
            else:
                self._inference_loop()
        except Exception as e:
            self.error_occurred.emit("运行时错误: {}".format(e))
            logger.error("inference loop error: %s", e, exc_info=True)
        finally:
            self._running = False
            self.finished_work.emit()
            self.status_msg.emit("预览已停止" if self.preview_only else "检测已停止")

    def _preview_loop(self):
        prev_time = time.perf_counter()
        for frame in self._frame_generator():
            if not self._running:
                break
            now = time.perf_counter()
            stream_fps = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now
            self._emit_frame_throttled(frame)
            self._emit_stats_throttled(0.0, 0.0, 0, 0.0, stream_fps)

    def _inference_loop(self):
        prev_time = time.perf_counter()

        for frame in self._frame_generator():
            if not self._running:
                break

            t0 = time.perf_counter()
            predict_kwargs = {
                "conf": self.conf_thres,
                "iou": self.iou_thres,
                "verbose": False,
                "device": 0,
            }
            if self.use_fp16:
                predict_kwargs["half"] = True

            results = self.model.predict(frame, **predict_kwargs)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            infer_fps = 1000.0 / max(latency_ms, 1e-6)

            detections = []
            annotated = frame.copy()
            if results:
                res = results[0]
                if res.boxes is not None:
                    for idx, box in enumerate(res.boxes):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        real_x = cx * self.mm_per_pixel
                        real_y = cy * self.mm_per_pixel

                        det = {
                            "id": idx + 1,
                            "label": self.class_names.get(cls_id, str(cls_id)),
                            "conf": conf,
                            "cx": cx,
                            "cy": cy,
                            "real_x": real_x,
                            "real_y": real_y,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "cls_id": cls_id,
                        }
                        detections.append(det)
                self._draw_yolo_style_overlay(annotated, detections)

            now = time.perf_counter()
            stream_fps = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            gpu_mem_mb = self._get_gpu_memory_usage()

            self._emit_frame_throttled(annotated)
            self._emit_data_throttled(detections)
            self._emit_stats_throttled(infer_fps, latency_ms, len(detections), gpu_mem_mb, stream_fps)

            if self.source_type in ("image", "folder"):
                time.sleep(0.5)

    def _draw_yolo_style_overlay(self, image, detections):
        """Draw class-colored boxes and readable labels with anti-overlap placement."""
        if not detections:
            return

        h, w = image.shape[:2]
        lw = max(2, int(round(min(h, w) / 320)))
        font_scale = max(0.65, min(h, w) / 1200.0)
        font_thick = max(1, lw - 1)
        pad = max(3, lw)
        occupied = []

        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            cls_id = int(det.get("cls_id", 0))
            color = self._palette(cls_id, bgr=True)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, lw)

            label = "ID:{} {} {:.2f}".format(det["id"], det["label"], det["conf"])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            rect = self._pick_label_rect(x1, y1, x2, y2, tw, th, pad, w, h, occupied)
            rx1, ry1, rx2, ry2, tx, ty = rect

            # Filled colored label with dark stroke text for readability.
            cv2.rectangle(image, (rx1, ry1), (rx2, ry2), color, -1)
            cv2.putText(
                image,
                label,
                (tx + 1, ty + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thick + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thick,
                cv2.LINE_AA,
            )

            occupied.append((rx1, ry1, rx2, ry2))

            # Red crosshair at center point (same as main_gui_jetson.py)
            s, c, t = 15, (0, 0, 255), 2
            cx, cy = det["cx"], det["cy"]
            cv2.line(image, (cx - s, cy), (cx + s, cy), c, t)
            cv2.line(image, (cx, cy - s), (cx, cy + s), c, t)

    @staticmethod
    def _rect_intersects(r1, r2):
        return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

    def _pick_label_rect(self, x1, y1, x2, y2, tw, th, pad, img_w, img_h, occupied):
        """Pick a label box location around bbox while avoiding overlaps when possible."""
        bw = tw + pad * 2
        bh = th + pad * 2

        candidates = [
            (x1, y1 - bh - 2),
            (x2 - bw, y1 - bh - 2),
            (x1, y2 + 2),
            (x2 - bw, y2 + 2),
            (x1, y1 + 2),
        ]

        for cx, cy in candidates:
            rx1 = int(max(1, min(cx, img_w - bw - 1)))
            ry1 = int(max(1, min(cy, img_h - bh - 1)))
            rx2, ry2 = rx1 + bw, ry1 + bh
            rect = (rx1, ry1, rx2, ry2)
            if not any(self._rect_intersects(rect, o) for o in occupied):
                tx = rx1 + pad
                ty = ry2 - pad
                return rx1, ry1, rx2, ry2, tx, ty

        # Fallback: return first candidate clamped to image.
        cx, cy = candidates[0]
        rx1 = int(max(1, min(cx, img_w - bw - 1)))
        ry1 = int(max(1, min(cy, img_h - bh - 1)))
        rx2, ry2 = rx1 + bw, ry1 + bh
        tx = rx1 + pad
        ty = ry2 - pad
        return rx1, ry1, rx2, ry2, tx, ty

    def _emit_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888).copy()
        self.frame_ready.emit(qimg)

    def _emit_frame_throttled(self, bgr):
        now = time.time()
        if now - self._last_signal_time["frame"] > self.signal_interval:
            self._emit_frame(bgr)
            self._last_signal_time["frame"] = now

    def _emit_data_throttled(self, detections):
        now = time.time()
        if now - self._last_signal_time["data"] > self.signal_interval * 2:
            self.data_ready.emit(detections)
            self._last_signal_time["data"] = now

    def _emit_stats_throttled(self, infer_fps, latency_ms, count, gpu_mem_mb, stream_fps):
        now = time.time()
        if now - self._last_signal_time["stats"] > self.signal_interval:
            self.stats_ready.emit(infer_fps, latency_ms, count, gpu_mem_mb, stream_fps)
            self._last_signal_time["stats"] = now

    def _get_gpu_memory_usage(self):
        """Get GPU memory usage delta in MB from load-time baseline."""
        try:
            import torch

            if torch.cuda.is_available() and self._gpu_mem_baseline_free is not None:
                free, _ = torch.cuda.mem_get_info(0)
                used_delta = (self._gpu_mem_baseline_free - free) / 1024**2
                return round(max(used_delta, 0.0), 1)
        except Exception:
            pass
        return 0.0

    def _buffered_reader(self, cap, pacing_interval=None):
        """Read frames in a background thread to smooth frame delivery."""
        latest = [None]
        lock = threading.Lock()
        new_frame_evt = threading.Event()
        reader_eof = [False]
        stop_evt = threading.Event()

        def _reader():
            while cap.isOpened() and not stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    reader_eof[0] = True
                    new_frame_evt.set()
                    return
                with lock:
                    latest[0] = frame
                new_frame_evt.set()
                if pacing_interval:
                    stop_evt.wait(timeout=pacing_interval)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        try:
            while not reader_eof[0]:
                if not self._running:
                    break
                if not new_frame_evt.wait(timeout=1.0):
                    continue
                new_frame_evt.clear()
                with lock:
                    frame = latest[0]
                    latest[0] = None
                if frame is not None:
                    yield frame
        finally:
            stop_evt.set()
            t.join(timeout=3.0)

    def _frame_generator(self):
        if self.source_type == "camera":
            yield from self._gen_camera()
        elif self.source_type == "video":
            yield from self._gen_video()
        elif self.source_type == "image":
            yield from self._gen_image()
        elif self.source_type == "folder":
            yield from self._gen_folder()

    @staticmethod
    def _gstreamer_pipeline(sensor_id=0, width=1280, height=720, fps=30):
        return (
            "nvarguscamerasrc sensor-id={sid} ! "
            "video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, "
            "framerate=(int){fps}/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=(int){w}, height=(int){h}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
        ).format(sid=sensor_id, w=width, h=height, fps=fps)

    def _gen_camera(self):
        if self.camera_backend == "mindvision":
            yield from self._gen_mindvision_camera()
            return

        idx = self.camera_index
        if self.use_csi:
            backend_name = "gstreamer"
            pipeline = self._gstreamer_pipeline(sensor_id=idx)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            backend_name = "v4l2"
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)

        if not cap.isOpened():
            self.error_occurred.emit("摄像头未检测到，请检查连接。")
            return

        if not self.use_csi:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                if PLATFORM == "jetson":
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_FPS, 60)
            except Exception:
                pass

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            self.error_occurred.emit("摄像头打开成功但无法读取图像。")
            return

        logger.info("Camera opened (backend=%s, index=%s).", backend_name, idx)

        try:
            yield frame
            yield from self._buffered_reader(cap)
        finally:
            cap.release()

    def _gen_mindvision_camera(self):
        if mvsdk is None:
            self.error_occurred.emit("未找到 mvsdk，请确认 mindvision/demo/python_demo/mvsdk.py 可用。")
            return

        try:
            dev_list = _sdk_call_quiet(mvsdk.CameraEnumerateDevice)
        except Exception as e:
            self.error_occurred.emit("工业相机枚举失败: {}".format(e))
            return

        if not dev_list:
            self.error_occurred.emit("未发现工业相机，请检查网口/IP 配置。")
            return

        idx = max(0, min(int(self.camera_index), len(dev_list) - 1))
        dev_info = dev_list[idx]
        try:
            self.status_msg.emit("工业相机已连接: {}".format(dev_info.GetFriendlyName()))
        except Exception:
            self.status_msg.emit("工业相机已连接")

        h_camera = 0
        p_frame_buffer = 0
        try:
            h_camera = _sdk_call_quiet(mvsdk.CameraInit, dev_info, -1, -1)
            cap = mvsdk.CameraGetCapability(h_camera)
            mono_camera = cap.sIspCapacity.bMonoSensor != 0

            if mono_camera:
                mvsdk.CameraSetIspOutFormat(h_camera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
            else:
                mvsdk.CameraSetIspOutFormat(h_camera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

            mvsdk.CameraSetTriggerMode(h_camera, 0)
            mvsdk.CameraSetAeState(h_camera, 0)
            mvsdk.CameraSetExposureTime(h_camera, 30 * 1000)
            mvsdk.CameraPlay(h_camera)
            logger.info("Camera opened (backend=%s, index=%s).", "mindvision", idx)

            frame_buffer_size = (
                cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if mono_camera else 3)
            )
            p_frame_buffer = mvsdk.CameraAlignMalloc(frame_buffer_size, 16)

            while self._running:
                try:
                    p_raw_data, frame_head = mvsdk.CameraGetImageBuffer(h_camera, 200)
                except mvsdk.CameraException as e:
                    if e.error_code == mvsdk.CAMERA_STATUS_TIME_OUT:
                        continue
                    self.error_occurred.emit("工业相机取流失败({}): {}".format(e.error_code, e.message))
                    break

                try:
                    mvsdk.CameraImageProcess(h_camera, p_raw_data, p_frame_buffer, frame_head)
                finally:
                    mvsdk.CameraReleaseImageBuffer(h_camera, p_raw_data)

                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(p_frame_buffer, frame_head, 1)

                frame_data = (mvsdk.c_ubyte * frame_head.uBytes).from_address(p_frame_buffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape(
                    (
                        frame_head.iHeight,
                        frame_head.iWidth,
                        1 if frame_head.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3,
                    )
                )

                if frame.ndim == 3 and frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                yield frame

        except mvsdk.CameraException as e:
            self.error_occurred.emit("工业相机初始化失败({}): {}".format(e.error_code, e.message))
            return
        except Exception as e:
            self.error_occurred.emit("工业相机异常: {}".format(e))
            return
        finally:
            if h_camera:
                try:
                    mvsdk.CameraUnInit(h_camera)
                except Exception:
                    pass
            if p_frame_buffer:
                try:
                    mvsdk.CameraAlignFree(p_frame_buffer)
                except Exception:
                    pass

    def _gen_video(self):
        if not self.source_path or not os.path.isfile(self.source_path):
            self.error_occurred.emit("视频文件不存在: {}".format(self.source_path))
            return

        cap = cv2.VideoCapture(self.source_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = 1.0 / fps if fps and fps > 1e-6 else None
        try:
            yield from self._buffered_reader(cap, pacing_interval=interval)
        finally:
            cap.release()

    def _gen_image(self):
        if not self.source_path or not os.path.isfile(self.source_path):
            self.error_occurred.emit("图片不存在: {}".format(self.source_path))
            return
        img = _safe_imread(self.source_path)
        if img is None:
            self.error_occurred.emit("图片解码失败。")
            return
        yield img

    def _gen_folder(self):
        if not self.source_path or not os.path.isdir(self.source_path):
            self.error_occurred.emit("目录不存在: {}".format(self.source_path))
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".jfif"}
        files = sorted(p for p in Path(self.source_path).iterdir() if p.suffix.lower() in exts)
        if not files:
            self.error_occurred.emit("目录中没有可用图片。")
            return

        for f in files:
            if not self._running:
                break
            img = _safe_imread(str(f))
            if img is not None:
                yield img


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLO的小型零件识别与定位系统")
        self.resize(1480, 920)

        self.thread = DetectionThread()
        self.current_data = []

        self._fps_filter = []
        self._latency_filter = []
        self._gpu_mem_filter = []
        self._filter_size = 10
        self._last_auto_send_signature = None
        self._last_auto_send_ts = 0.0
        self._auto_send_min_interval = 5.0
        self._op_log_max_rows = 300

        self._build_ui()
        self._connect_signals()
        self._on_grab_mode_changed(self.cmb_grab_mode.currentIndex())

        if os.path.exists(DEFAULT_WEIGHTS):
            self.thread.load_model(DEFAULT_WEIGHTS)
            self.lbl_model_name.setText(os.path.basename(DEFAULT_WEIGHTS))
        else:
            self.statusBar().showMessage("未找到默认模型，请手动选择。")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)

        self.lbl_video = QLabel("等待输入 ...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(720, 520)
        self.lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_video.setStyleSheet("background:#111827; color:#9CA3AF; font-size:20px; border:1px solid #374151;")
        splitter.addWidget(self.lbl_video)

        right = QWidget()
        rl = QVBoxLayout(right)

        gb_kpi = QGroupBox("实时运行状态")
        kpi_grid = QGridLayout()

        self.lbl_fps = QLabel("—")
        self.lbl_latency = QLabel("—")
        self.lbl_count = QLabel("0")
        self.lbl_gpu_mem = QLabel("—")

        for w in (self.lbl_fps, self.lbl_latency, self.lbl_count, self.lbl_gpu_mem):
            w.setStyleSheet("font-size:24px; font-weight:bold; color:#0B66C3;")
            w.setAlignment(Qt.AlignCenter)

        titles = ["FPS(推理/流)", "延迟(ms)", "识别数量", "显存(MB)"]
        vals = [self.lbl_fps, self.lbl_latency, self.lbl_count, self.lbl_gpu_mem]
        for i, (t, v) in enumerate(zip(titles, vals)):
            tl = QLabel(t)
            tl.setStyleSheet("font-size:12px; color:#4B5563;")
            tl.setAlignment(Qt.AlignCenter)
            kpi_grid.addWidget(tl, 0, i)
            kpi_grid.addWidget(v, 1, i)

        gb_kpi.setLayout(kpi_grid)
        rl.addWidget(gb_kpi)

        rl.addWidget(QLabel("识别结果列表"))
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["编号", "类别", "置信度", "X(px)", "Y(px)", "X(mm)", "Y(mm)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        rl.addWidget(self.table)

        gb_grab = QGroupBox("坐标发送")
        gl = QVBoxLayout()

        line1 = QHBoxLayout()
        line1.addWidget(QLabel("发送模式"))
        self.cmb_grab_mode = QComboBox()
        self.cmb_grab_mode.addItems(["自动(发送全部目标)", "手动(指定编号)"])
        self.cmb_grab_mode.currentIndexChanged.connect(self._on_grab_mode_changed)
        line1.addWidget(self.cmb_grab_mode)

        line1.addWidget(QLabel("目标编号"))
        self.spin_target_id = QSpinBox()
        self.spin_target_id.setRange(1, 9999)
        self.spin_target_id.setEnabled(False)
        self.spin_target_id.valueChanged.connect(self._update_current_target_hint)
        line1.addWidget(self.spin_target_id)

        self.lbl_current_target = QLabel("当前发送: 无")
        self.lbl_current_target.setStyleSheet("font-size:12px; color:#6B7280;")

        self.btn_grab = QPushButton("发送坐标")
        self.btn_grab.setStyleSheet(
            "QPushButton{background:#2563EB; color:white; font-weight:bold; padding:8px 16px;}"
            "QPushButton:disabled{background:#9CA3AF; color:#E5E7EB;}"
        )
        self.btn_grab.clicked.connect(self._on_grab)

        self.op_log = QListWidget()
        self.op_log.setMinimumHeight(130)

        gl.addLayout(line1)
        gl.addWidget(self.lbl_current_target)
        gl.addWidget(self.btn_grab)
        gl.addWidget(QLabel("发送日志"))
        gl.addWidget(self.op_log)
        gb_grab.setLayout(gl)
        rl.addWidget(gb_grab)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, stretch=1)

        ctrl = QFrame()
        ctrl.setStyleSheet("QFrame{background:#F3F4F6; border-top:1px solid #D1D5DB;}")
        cl = QHBoxLayout(ctrl)

        gb_setup = QGroupBox("系统设置")
        sl = QHBoxLayout()
        btn_wt = QPushButton("选择模型")
        btn_wt.clicked.connect(self._on_select_weights)
        self.lbl_model_name = QLabel("—")
        self.lbl_model_name.setStyleSheet("font-weight:bold;")

        self.btn_cam = QPushButton("开启摄像头")
        self.btn_cam.clicked.connect(self._on_select_camera)

        sl.addWidget(btn_wt)
        sl.addWidget(self.lbl_model_name)
        sl.addWidget(self.btn_cam)
        gb_setup.setLayout(sl)

        gb_param = QGroupBox("参数调节")
        pl = QHBoxLayout()
        self.slider_conf, conf_w = self._make_slider("Conf", INITIAL_CONF)
        self.slider_iou, iou_w = self._make_slider("IoU", INITIAL_IOU)
        self.slider_conf.valueChanged.connect(self._on_params_changed)
        self.slider_iou.valueChanged.connect(self._on_params_changed)

        mm_box = QWidget()
        mm_lo = QVBoxLayout(mm_box)
        mm_lo.setContentsMargins(0, 0, 0, 0)
        mm_lo.addWidget(QLabel("mm/px"))
        self.spin_mm_per_px = QDoubleSpinBox()
        self.spin_mm_per_px.setRange(0.01, 100.0)
        self.spin_mm_per_px.setValue(1.0)
        self.spin_mm_per_px.setSingleStep(0.1)
        self.spin_mm_per_px.setDecimals(2)
        self.spin_mm_per_px.valueChanged.connect(self._on_calib_changed)
        mm_lo.addWidget(self.spin_mm_per_px)

        pl.addWidget(conf_w)
        pl.addWidget(iou_w)
        pl.addWidget(mm_box)
        gb_param.setLayout(pl)

        gb_act = QGroupBox("操作")
        al = QHBoxLayout()
        self.btn_start = QPushButton("开始检测")
        self.btn_start.setStyleSheet("background:#16A34A; color:white; font-weight:bold; padding:6px 14px;")
        self.btn_start.clicked.connect(self._on_toggle)
        al.addWidget(self.btn_start)
        gb_act.setLayout(al)

        cl.addWidget(gb_setup)
        cl.addWidget(gb_param)
        cl.addWidget(gb_act)
        root.addWidget(ctrl)

    @staticmethod
    def _make_slider(label, init):
        container = QWidget()
        lo = QVBoxLayout(container)
        lo.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("{}: {:.2f}".format(label, init))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(init * 100))

        def _update(v, _lbl=lbl, _label=label):
            _lbl.setText("{}: {:.2f}".format(_label, v / 100.0))

        slider.valueChanged.connect(_update)
        lo.addWidget(lbl)
        lo.addWidget(slider)
        return slider, container

    def _connect_signals(self):
        for sig in [
            self.thread.frame_ready,
            self.thread.data_ready,
            self.thread.stats_ready,
            self.thread.status_msg,
            self.thread.error_occurred,
            self.thread.finished_work,
        ]:
            try:
                sig.disconnect()
            except TypeError:
                pass

        self.thread.frame_ready.connect(self._update_frame)
        self.thread.data_ready.connect(self._update_table)
        self.thread.stats_ready.connect(self._update_kpis)
        self.thread.status_msg.connect(lambda m: self.statusBar().showMessage(m))
        self.thread.error_occurred.connect(self._on_error)
        self.thread.finished_work.connect(self._on_finished)

    def _on_select_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            os.path.dirname(DEFAULT_WEIGHTS),
            "Model Weights (*.pt *.onnx *.engine);;PyTorch (*.pt);;ONNX (*.onnx);;TensorRT (*.engine)",
        )
        if path and self.thread.load_model(path):
            self.lbl_model_name.setText(os.path.basename(path))

    def _on_select_file(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("选择输入类型")
        dlg.setText("请选择输入源：")
        btn_media = dlg.addButton("图片 / 视频文件", QMessageBox.ActionRole)
        btn_folder = dlg.addButton("图片文件夹", QMessageBox.ActionRole)
        dlg.addButton(QMessageBox.Cancel)
        dlg.exec_()

        clicked = dlg.clickedButton()
        if clicked == btn_media:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择文件",
                "",
                "所有文件 (*.*);;"
                "智能识别(图片/视频) (*.*);;"
                "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.gif *.jfif *.JPG *.JPEG *.PNG *.BMP *.WEBP *.TIF *.TIFF *.GIF *.JFIF);;"
                "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.m4v *.ts *.mts *.MP4 *.AVI *.MKV *.MOV *.FLV *.WMV *.M4V *.TS *.MTS)",
            )
            if path:
                stype = _guess_media_type(path)
                if stype is None:
                    QMessageBox.warning(self, "提示", "无法识别该文件类型，请选择图片或视频文件。")
                    return
                self.thread.set_source(stype, path)
                self.statusBar().showMessage("已选择: {}".format(os.path.basename(path)))
                self._preview_source(stype, path)
        elif clicked == btn_folder:
            path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
            if path:
                self.thread.set_source("folder", path)
                self.statusBar().showMessage("已选择目录: {}".format(path))
                self._preview_source("folder", path)

    def _preview_source(self, stype, path):
        frame = None
        try:
            if stype == "image":
                frame = _safe_imread(path)
            elif stype == "video":
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        frame = None
                    cap.release()
            elif stype == "folder":
                exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".jfif"}
                files = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in exts)
                if files:
                    frame = _safe_imread(str(files[0]))
        except Exception as e:
            logger.warning("Preview failed: %s", e)

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_video.setPixmap(pix)

    def _on_select_camera(self):
        if self.thread.source_type == "camera":
            self._close_camera()
            return

        use_csi = False
        cam_idx = DEFAULT_CAMERA_INDEX
        camera_backend = "opencv"

        dlg = QMessageBox(self)
        dlg.setWindowTitle("选择摄像头类型")
        dlg.setText("请选择摄像头接口：")
        btn_usb = dlg.addButton("USB 摄像头", QMessageBox.ActionRole)
        btn_industrial = dlg.addButton("工业相机(MindVision)", QMessageBox.ActionRole)
        dlg.addButton(QMessageBox.Cancel)
        dlg.exec_()
        clicked = dlg.clickedButton()
        if clicked == btn_usb:
            camera_backend = "opencv"
        elif clicked == btn_industrial:
            camera_backend = "mindvision"
        else:
            return

        self.thread.use_csi = use_csi
        self.thread.camera_index = cam_idx
        self.thread.camera_backend = camera_backend
        self.thread.set_source("camera", cam_idx)
        if camera_backend == "mindvision":
            self.statusBar().showMessage("已选择摄像头: 工业相机(MindVision)")
        else:
            self.statusBar().showMessage("已选择摄像头: USB")

        self._start_preview()
        self.btn_cam.setText("关闭摄像头")

    def _close_camera(self):
        if self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)

        self.thread.set_source(None, None)
        self.thread.preview_only = False
        self.btn_cam.setText("开启摄像头")
        self.btn_start.setText("开始检测")
        self.btn_start.setStyleSheet("background:#16A34A; color:white; font-weight:bold; padding:6px 14px;")
        self.statusBar().showMessage("摄像头已关闭")

    def _start_preview(self):
        if self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)

        old = self.thread
        self.thread = DetectionThread()
        self.thread.model = old.model
        self.thread.class_names = old.class_names
        self.thread.weights_path = old.weights_path
        self.thread.source_type = old.source_type
        self.thread.source_path = old.source_path
        self.thread.conf_thres = old.conf_thres
        self.thread.iou_thres = old.iou_thres
        self.thread.mm_per_pixel = old.mm_per_pixel
        self.thread.camera_index = old.camera_index
        self.thread.use_csi = old.use_csi
        self.thread.camera_backend = old.camera_backend
        self.thread.model_format = old.model_format
        self.thread.use_fp16 = old.use_fp16
        self.thread.dataset_yaml_path = old.dataset_yaml_path
        self.thread._gpu_mem_baseline_free = old._gpu_mem_baseline_free
        self.thread.preview_only = True

        self._connect_signals()
        self.current_data = []
        self.table.setRowCount(0)
        self.lbl_count.setText("0")
        self.lbl_current_target.setText("当前发送: 无")

        self.thread.start()
        self.btn_start.setText("开始检测")
        self.btn_start.setStyleSheet("background:#16A34A; color:white; font-weight:bold; padding:6px 14px;")

    def _on_toggle(self):
        if self.thread.isRunning() and not self.thread.preview_only:
            self.thread.stop()
            self.thread.wait(3000)
            return

        if self.thread.isRunning() and self.thread.preview_only:
            self._start_detection()
            return

        if self.thread.model is None:
            QMessageBox.warning(self, "提示", "请先加载模型。")
            return
        if self.thread.source_type is None:
            QMessageBox.warning(self, "提示", "请先选择输入源。")
            return

        self._start_detection()

    def _start_detection(self):
        if self.thread.model is None:
            QMessageBox.warning(self, "提示", "请先加载模型。")
            return

        if self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)

        old = self.thread
        self.thread = DetectionThread()
        self.thread.model = old.model
        self.thread.class_names = old.class_names
        self.thread.weights_path = old.weights_path
        self.thread.source_type = old.source_type
        self.thread.source_path = old.source_path
        self.thread.conf_thres = old.conf_thres
        self.thread.iou_thres = old.iou_thres
        self.thread.mm_per_pixel = old.mm_per_pixel
        self.thread.camera_index = old.camera_index
        self.thread.use_csi = old.use_csi
        self.thread.camera_backend = old.camera_backend
        self.thread.model_format = old.model_format
        self.thread.use_fp16 = old.use_fp16
        self.thread.dataset_yaml_path = old.dataset_yaml_path
        self.thread._gpu_mem_baseline_free = old._gpu_mem_baseline_free
        self.thread.preview_only = False

        self._connect_signals()
        self.current_data = []
        self.table.setRowCount(0)
        self.lbl_current_target.setText("当前发送: 无")

        self.thread.start()
        self.btn_start.setText("停止检测")
        self.btn_start.setStyleSheet("background:#DC2626; color:white; font-weight:bold; padding:6px 14px;")

    def _on_finished(self):
        # Ignore finished signals from old threads during preview->detect thread swap.
        if self.sender() is not self.thread:
            return
        self.btn_start.setText("开始检测")
        self.btn_start.setStyleSheet("background:#16A34A; color:white; font-weight:bold; padding:6px 14px;")
        if self.thread.source_type != "camera":
            self.btn_cam.setText("开启摄像头")

    def _on_params_changed(self):
        self.thread.update_params(self.slider_conf.value() / 100.0, self.slider_iou.value() / 100.0)

    def _on_calib_changed(self, value):
        self.thread.mm_per_pixel = value

    def _on_grab_mode_changed(self, idx):
        manual = idx == 1
        self.spin_target_id.setEnabled(manual)
        self.btn_grab.setEnabled(manual)
        self.btn_grab.setToolTip("手动模式下发送指定编号目标" if manual else "自动模式下由系统自动发送")
        self._update_current_target_hint()

    def _resolve_send_targets(self):
        """Resolve targets to send: default all, or one selected id in manual mode."""
        if not self.current_data:
            return []

        mode = self.cmb_grab_mode.currentIndex()
        if mode == 0:
            return sorted(self.current_data, key=lambda x: x["id"])

        target_id = self.spin_target_id.value()
        for det in self.current_data:
            if det["id"] == target_id:
                return [det]
        return []

    def _update_current_target_hint(self):
        targets = self._resolve_send_targets()
        if not targets:
            if self.current_data:
                ids = [str(d["id"]) for d in self.current_data]
                self.lbl_current_target.setText("当前发送: 未找到指定编号，可用编号 [{}]".format(",".join(ids)))
            else:
                self.lbl_current_target.setText("当前发送: 无")
            return

        if self.cmb_grab_mode.currentIndex() == 0:
            ids = [str(t["id"]) for t in targets[:5]]
            suffix = "..." if len(targets) > 5 else ""
            self.lbl_current_target.setText(
                "当前发送: 全部目标 共{}个 [{}{}]".format(len(targets), ",".join(ids), suffix)
            )
            return

        target = targets[0]
        self.lbl_current_target.setText(
            "当前发送: #{} {} ({:.1f},{:.1f})mm".format(
                target["id"], target["label"], target["real_x"], target["real_y"]
            )
        )

    def _build_coordinate_payload(self, targets):
        payload = []
        for t in targets:
            payload.append(
                {
                    "id": t["id"],
                    "label": t["label"],
                    "confidence": round(float(t["conf"]), 4),
                    "x_mm": round(float(t["real_x"]), 3),
                    "y_mm": round(float(t["real_y"]), 3),
                    "x_px": int(t["cx"]),
                    "y_px": int(t["cy"]),
                }
            )
        return payload

    def _to_robot_pose(self, item):
        """Convert center coordinates to a common 6D robot pose representation."""
        return {
            "id": item["id"],
            "label": item["label"],
            "confidence": item["confidence"],
            "pose": {
                "x": round(item["x_mm"], 3),
                "y": round(item["y_mm"], 3),
                "z": 0.0,
                "rx": 180.0,
                "ry": 0.0,
                "rz": 0.0,
                "unit": "mm_deg",
                "frame": "robot_base",
            },
        }

    def _build_robot_payload(self, targets, send_mode=None):
        coords = self._build_coordinate_payload(targets)
        mode = send_mode or ("auto_all" if self.cmb_grab_mode.currentIndex() == 0 else "manual_single")
        return {
            "timestamp": int(time.time() * 1000),
            "mode": mode,
            "targets": [self._to_robot_pose(item) for item in coords],
        }

    def _send_coordinates(self, targets, send_mode=None):
        """Mock sender: replace this with TCP/串口/ROS2 publisher in real deployment."""
        payload = self._build_robot_payload(targets, send_mode=send_mode)
        return True, payload

    @staticmethod
    def _robot_payload_lines(payload):
        lines = []
        for t in payload.get("targets", []):
            pose = t.get("pose", {})
            lines.append(
                "ROBOT_PICK id={id} cls={cls} conf={conf:.3f} X={x:.3f} Y={y:.3f} Z={z:.3f} RX={rx:.3f} RY={ry:.3f} RZ={rz:.3f} {unit} frame={frame}".format(
                    id=t.get("id"),
                    cls=t.get("label", ""),
                    conf=float(t.get("confidence", 0.0)),
                    x=float(pose.get("x", 0.0)),
                    y=float(pose.get("y", 0.0)),
                    z=float(pose.get("z", 0.0)),
                    rx=float(pose.get("rx", 0.0)),
                    ry=float(pose.get("ry", 0.0)),
                    rz=float(pose.get("rz", 0.0)),
                    unit=pose.get("unit", "mm_deg"),
                    frame=pose.get("frame", "robot_base"),
                )
            )
        return lines

    def _append_op_log(self, text):
        ts = time.strftime("%H:%M:%S")
        self.op_log.addItem("{} | {}".format(ts, text))
        while self.op_log.count() > self._op_log_max_rows:
            self.op_log.takeItem(0)
        self.op_log.scrollToBottom()

    def _try_auto_send(self):
        if self.cmb_grab_mode.currentIndex() != 0:
            return
        targets = self._resolve_send_targets()
        if not targets:
            return

        signature = tuple(
            (t["id"], round(float(t["real_x"]), 2), round(float(t["real_y"]), 2), round(float(t["conf"]), 3))
            for t in targets
        )
        now = time.time()
        # Strict throttle: never auto-send faster than configured interval.
        if (now - self._last_auto_send_ts) < self._auto_send_min_interval:
            return

        ok, payload = self._send_coordinates(targets, send_mode="auto_all")
        if not ok:
            return

        count = len(payload.get("targets", []))
        msg = "自动发送完成: 共{}个目标".format(count)
        self.statusBar().showMessage(msg)
        self._append_op_log(msg)
        for line in self._robot_payload_lines(payload):
            self._append_op_log(line)
        self._append_op_log("JSON {}".format(json.dumps(payload, ensure_ascii=False)))

        self._last_auto_send_signature = signature
        self._last_auto_send_ts = now

    def _on_grab(self):
        if not self.btn_grab.isEnabled():
            self.statusBar().showMessage("自动模式下将随检测结果自动发送，无需手动点击。")
            return

        if not self.current_data:
            QMessageBox.warning(self, "发送提示", "当前无可发送目标，请先启动检测。")
            return

        target_id = int(self.spin_target_id.value())
        target = next((det for det in self.current_data if int(det.get("id", -1)) == target_id), None)
        if target is None:
            QMessageBox.warning(self, "发送提示", "未找到指定编号的目标，请检查手动编号。")
            self._update_current_target_hint()
            return
        targets = [target]

        ok, payload = self._send_coordinates(targets, send_mode="manual_single")
        if not ok:
            QMessageBox.warning(self, "发送提示", "坐标发送失败，请检查通讯连接。")
            return

        rows = payload.get("targets", [])
        if len(rows) == 1:
            p = rows[0]
            pose = p.get("pose", {})
            msg = "坐标已发送: 编号#{id}, 类别={label}, 坐标=({x:.1f}, {y:.1f})mm".format(
                id=p.get("id"), label=p.get("label"), x=float(pose.get("x", 0.0)), y=float(pose.get("y", 0.0))
            )
        else:
            msg = "坐标已发送: 共{}个目标".format(len(rows))

        self.statusBar().showMessage(msg)
        self._append_op_log(msg)
        for line in self._robot_payload_lines(payload):
            self._append_op_log(line)
        self._append_op_log("JSON {}".format(json.dumps(payload, ensure_ascii=False)))
        QMessageBox.information(self, "坐标发送结果", msg)

    def _on_error(self, msg):
        self.statusBar().showMessage("Error: {}".format(msg))
        QMessageBox.critical(self, "Error", msg)

    @pyqtSlot(QImage)
    def _update_frame(self, qimg):
        transform = Qt.FastTransformation if PLATFORM == "jetson" else Qt.SmoothTransformation
        pix = QPixmap.fromImage(qimg).scaled(self.lbl_video.size(), Qt.KeepAspectRatio, transform)
        self.lbl_video.setPixmap(pix)

    @pyqtSlot(list)
    def _update_table(self, dets):
        self.current_data = dets
        self.table.setRowCount(len(dets))
        for r, d in enumerate(dets):
            self.table.setItem(r, 0, QTableWidgetItem(str(d["id"])))
            self.table.setItem(r, 1, QTableWidgetItem(d["label"]))
            self.table.setItem(r, 2, QTableWidgetItem("{:.1f}%".format(d["conf"] * 100)))
            self.table.setItem(r, 3, QTableWidgetItem(str(d["cx"])))
            self.table.setItem(r, 4, QTableWidgetItem(str(d["cy"])))
            self.table.setItem(r, 5, QTableWidgetItem("{:.1f}".format(d["real_x"])))
            self.table.setItem(r, 6, QTableWidgetItem("{:.1f}".format(d["real_y"])))

        self._update_current_target_hint()
        self._try_auto_send()

    @pyqtSlot(float, float, int, float, float)
    def _update_kpis(self, infer_fps, latency_ms, count, gpu_mem, stream_fps):
        self._fps_filter.append(infer_fps)
        self._latency_filter.append(latency_ms)
        self._gpu_mem_filter.append(gpu_mem)
        if len(self._fps_filter) > self._filter_size:
            self._fps_filter.pop(0)
        if len(self._latency_filter) > self._filter_size:
            self._latency_filter.pop(0)
        if len(self._gpu_mem_filter) > self._filter_size:
            self._gpu_mem_filter.pop(0)

        avg_infer_fps = sum(self._fps_filter) / len(self._fps_filter)
        avg_latency = sum(self._latency_filter) / len(self._latency_filter)
        avg_gpu_mem = sum(self._gpu_mem_filter) / len(self._gpu_mem_filter)
        infer_text = "{:.1f}".format(avg_infer_fps) if avg_infer_fps > 0 else "-"
        self.lbl_fps.setText("{}/ {:.1f}".format(infer_text, stream_fps))
        self.lbl_latency.setText("{:.1f}".format(avg_latency))
        self.lbl_count.setText(str(count))
        self.lbl_gpu_mem.setText("{:.1f}".format(avg_gpu_mem))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)
        event.accept()


def main():
    def _excepthook(exc_type, exc_value, exc_tb):
        import traceback

        traceback.print_exception(exc_type, exc_value, exc_tb)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
