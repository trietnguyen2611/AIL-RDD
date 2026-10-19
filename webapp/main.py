#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Auto-switch to project virtualenv if invoked via system Python
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    if os.environ.get("_AIL_RDD_VENV_EXEC") != "1":
        os.environ["_AIL_RDD_VENV_EXEC"] = "1"
        try:
            os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)
        except Exception:
            pass

import base64
import io
import json
import mimetypes
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

SCRIPT_DIR = _SCRIPT_DIR
PROJECT_ROOT = _PROJECT_ROOT
WEBAPP_DIR = SCRIPT_DIR
DATASET_DIR = (PROJECT_ROOT / "dataset" / "test" / "images").resolve()

CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
CLASS_LABELS = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}

# Hardware and engine initialization
TORCH_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False
DEVICE = "cpu"
MODEL_CACHE = {}

try:
    import torch
    import torchvision
    import torch.nn as nn
    from typing import Tuple, List, Optional, Dict, Any

    TORCH_AVAILABLE = True

    if torch.cuda.is_available():
        DEVICE = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    from ultralytics import YOLO
    from ultralytics.nn import tasks, modules
    from ultralytics.nn.modules import Conv, Detect

    ULTRALYTICS_AVAILABLE = True

    # P26 Custom Architectural Modules
    class DeformConv2d(nn.Module):
        def __init__(self, inc: int, outc: int, kernel_size: int = 3, padding: int = 1, stride: int = 1, bias: Optional[bool] = None):
            super().__init__()
            self.stride = stride
            self.padding = padding
            self.conv_offset = nn.Conv2d(inc, 3 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
            nn.init.constant_(self.conv_offset.weight, 0.0)
            nn.init.constant_(self.conv_offset.bias, 0.0)
            self.weight = nn.Parameter(torch.empty(outc, inc, kernel_size, kernel_size))
            nn.init.kaiming_uniform_(self.weight, a=1)
            if bias:
                self.bias = nn.Parameter(torch.zeros(outc))
            else:
                self.register_parameter("bias", None)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_dtype = x.dtype
            x_f = x.float().contiguous(memory_format=torch.contiguous_format)
            out = self.conv_offset(x_f)
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1).contiguous(memory_format=torch.contiguous_format)
            offset = torch.nan_to_num(offset, nan=0.0, posinf=10.0, neginf=-10.0)
            mask = torch.nan_to_num(torch.sigmoid(mask), nan=0.5).contiguous(memory_format=torch.contiguous_format)
            w_f = self.weight.float().contiguous(memory_format=torch.contiguous_format)
            b_f = self.bias.float().contiguous(memory_format=torch.contiguous_format) if self.bias is not None else None
            try:
                res = torchvision.ops.deform_conv2d(
                    input=x_f, offset=offset, weight=w_f, bias=b_f,
                    stride=(self.stride, self.stride), padding=(self.padding, self.padding), mask=mask
                )
            except Exception:
                res = nn.functional.conv2d(x_f, w_f, b_f, stride=self.stride, padding=self.padding)
            return res.to(orig_dtype)

    class Bottleneck_DCNv2(nn.Module):
        def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5):
            super().__init__()
            c_ = int(c2 * e)
            self.cv1 = Conv(c1, c_, k[0], 1)
            self.dcn = DeformConv2d(c_, c2, k[1], 1, padding=k[1] // 2)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU()
            self.add = shortcut and c1 == c2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.act(self.bn(self.dcn(self.cv1(x))))
            return x + out if self.add else out

    class C2f_DCNv2(nn.Module):
        def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
            super().__init__()
            self.c = int(c2 * e)
            self.cv1 = Conv(c1, 2 * self.c, 1, 1)
            self.cv2 = Conv((2 + n) * self.c, c2, 1)
            self.m = nn.ModuleList(Bottleneck_DCNv2(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = list(self.cv1(x).chunk(2, 1))
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))

    class GSConv(nn.Module):
        def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True):
            super().__init__()
            c_ = c2 // 2
            self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
            self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.cv1(x)
            x2 = self.cv2(x1)
            b, c, h, w = x1.shape
            out = torch.cat((x1, x2), dim=1)
            out = out.view(b, 2, c, h, w).transpose(1, 2).contiguous().view(b, c * 2, h, w)
            return out

    class GSBottleneck(nn.Module):
        def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, e: float = 0.5):
            super().__init__()
            c_ = int(c2 * e)
            self.conv1 = GSConv(c1, c_, 1, 1)
            self.conv2 = GSConv(c_, c2, 3, 1)
            self.shortcut = c1 == c2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))

    class VoVGSCSP(nn.Module):
        def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
            super().__init__()
            c_ = int(c2 * e)
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(2 * c_, c2, 1, 1)
            self.m = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.cv1(x)
            return self.cv2(torch.cat((self.m(x1), x1), dim=1))

    class GCHDetect(Detect):
        def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: Tuple[int, ...] = ()):
            super().__init__(nc, reg_max, end2end, ch)
            c_mid = max(16, ch[0] // 4, self.reg_max * 4)
            g = 4
            self.shared = nn.ModuleList(
                nn.Sequential(
                    Conv(x, c_mid, 3, g=min(g, x, c_mid)),
                    Conv(c_mid, c_mid, 3, g=min(g, c_mid)),
                ) for x in ch
            )
            self.cv2 = nn.ModuleList(nn.Sequential(nn.Conv2d(c_mid, 4 * self.reg_max, 1)) for _ in ch)
            self.cv3 = nn.ModuleList(nn.Sequential(nn.Conv2d(c_mid, self.nc, 1)) for _ in ch)

        def forward_head(self, x: List[torch.Tensor], box_head=None, cls_head=None) -> Dict[str, Any]:
            bs = x[0].shape[0]
            feats = [self.shared[i](x[i]) for i in range(self.nl)]
            boxes = torch.cat([self.cv2[i](feats[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
            scores = torch.cat([self.cv3[i](feats[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
            return {"boxes": boxes, "scores": scores, "feats": x}

    # Register P26 classes
    p26_classes = [DeformConv2d, Bottleneck_DCNv2, C2f_DCNv2, GSConv, GSBottleneck, VoVGSCSP, GCHDetect]
    main_mod = sys.modules.get("__main__")
    for cls in p26_classes:
        if main_mod:
            setattr(main_mod, cls.__name__, cls)
        setattr(tasks, cls.__name__, cls)
        setattr(modules, cls.__name__, cls)

    if hasattr(torch.serialization, "add_safe_globals"):
        try:
            torch.serialization.add_safe_globals(p26_classes)
        except Exception:
            pass

except Exception as err:
    sys.stderr.write(f"Inference engine notice: {err}\n")


MODEL_ALIASES = {
    "yolo11s": "yolo11s_best",
    "yolov11s": "yolo11s_best",
    "yolo11s_best": "yolo11s_best",
    "yolov10s": "yolov10s_best",
    "yolov10s_best": "yolov10s_best",
    "yolov26n": "yolov26n_best",
    "yolo26n": "yolov26n_best",
    "yolov26n_best": "yolov26n_best",
    "yolov8s": "yolov8s_best",
    "yolov8s_best": "yolov8s_best",
}

ALLOWED_MODELS = {
    "yolo11s_best",
    "yolov10s_best",
    "yolov26n_best",
    "yolov8s_best",
}
MAX_BODY_SIZE = 25 * 1024 * 1024  # 25 MB limit


def get_model(model_name: str):
    """Retrieve or dynamically load model with caching and allowlist enforcement."""
    if not ULTRALYTICS_AVAILABLE:
        return None

    canonical_name = MODEL_ALIASES.get(model_name, model_name)
    if canonical_name not in ALLOWED_MODELS:
        sys.stderr.write(f"Rejected unallowed model request: {model_name}\n")
        return None

    if canonical_name in MODEL_CACHE:
        return MODEL_CACHE[canonical_name]

    # Resolve weight path safely across known naming conventions
    candidates = [
        (PROJECT_ROOT / "model" / f"{canonical_name}.pt").resolve(),
        (PROJECT_ROOT / "model" / f"{model_name}.pt").resolve(),
        (WEBAPP_DIR / "model" / f"{canonical_name}.pt").resolve(),
        (WEBAPP_DIR / "model" / f"{model_name}.pt").resolve(),
    ]

    if "yolov26" in canonical_name:
        candidates.insert(0, (PROJECT_ROOT / "model" / "yolov26n (Ours).pt").resolve())
    elif "yolo11" in canonical_name:
        candidates.insert(1, (PROJECT_ROOT / "model" / "yolov11s.pt").resolve())
    elif "yolov10" in canonical_name:
        candidates.insert(1, (PROJECT_ROOT / "model" / "yolov10s.pt").resolve())
    elif "yolov8" in canonical_name:
        candidates.insert(1, (PROJECT_ROOT / "model" / "yolov8s.pt").resolve())

    target_path = None
    for c in candidates:
        if c.exists():
            target_path = c
            break

    if not target_path:
        sys.stderr.write(f"Model file not found for: {model_name} (canonical: {canonical_name})\n")
        return None

    try:
        model = YOLO(str(target_path))
        MODEL_CACHE[canonical_name] = model
        return model
    except Exception as e:
        sys.stderr.write(f"Failed to load model {model_name}: {e}\n")
        return None


class RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEBAPP_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/models":
            self.handle_api_models()
        elif path == "/api/samples":
            self.handle_api_samples()
        elif path == "/api/benchmark":
            self.handle_api_benchmark()
        elif path == "/api/dataset-images":
            self.handle_api_dataset_images(parsed)
        elif path == "/api/dataset-image":
            self.handle_api_dataset_image(parsed)
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/detect":
            self.handle_api_detect()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def handle_api_models(self):
        benchmark_file = WEBAPP_DIR / "data" / "benchmark.json"
        models_data = []
        if benchmark_file.exists():
            with open(benchmark_file, "r") as f:
                models_data = json.load(f)

        status_str = "online" if (TORCH_AVAILABLE and ULTRALYTICS_AVAILABLE) else "standalone"
        response = {
            "status": status_str,
            "device": DEVICE,
            "torch_available": TORCH_AVAILABLE,
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
            "models": models_data,
        }
        self.send_json(response)

    def handle_api_samples(self):
        samples_file = WEBAPP_DIR / "data" / "samples.json"
        if samples_file.exists():
            with open(samples_file, "r") as f:
                samples_data = json.load(f)
            self.send_json(samples_data)
        else:
            self.send_json([])

    def handle_api_benchmark(self):
        benchmark_file = WEBAPP_DIR / "data" / "benchmark.json"
        if benchmark_file.exists():
            with open(benchmark_file, "r") as f:
                data = json.load(f)
            self.send_json(data)
        else:
            self.send_json([])

    def handle_api_dataset_images(self, parsed):
        query = parse_qs(parsed.query)
        try:
            offset = int(query.get("offset", [0])[0])
            limit = int(query.get("limit", [10])[0])
        except (ValueError, IndexError):
            offset = 0
            limit = 10

        offset = max(0, offset)
        limit = max(1, min(limit, 50))

        all_files = []
        if DATASET_DIR.exists():
            all_files = sorted([f.name for f in DATASET_DIR.glob("*.jpg")])

        total = len(all_files)
        batch = all_files[offset : offset + limit]

        images = []
        for f_name in batch:
            images.append({
                "id": Path(f_name).stem,
                "filename": f_name,
                "url": f"/api/dataset-image?file={f_name}",
                "title": Path(f_name).stem.replace("_", " "),
            })

        response = {
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": (offset + limit) < total,
            "images": images,
        }
        self.send_json(response)

    def handle_api_dataset_image(self, parsed):
        query = parse_qs(parsed.query)
        file_param = query.get("file", [""])[0]
        if not file_param or "/" in file_param or "\\" in file_param or ".." in file_param:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid file parameter")
            return

        safe_name = Path(file_param).name
        target_path = None
        cand_dataset = (DATASET_DIR / safe_name).resolve()
        if cand_dataset.is_relative_to(DATASET_DIR) and cand_dataset.exists():
            target_path = cand_dataset
        else:
            cand_samples = (WEBAPP_DIR / "samples" / safe_name).resolve()
            if cand_samples.is_relative_to(WEBAPP_DIR / "samples") and cand_samples.exists():
                target_path = cand_samples

        if not target_path:
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return

        try:
            with open(target_path, "rb") as f:
                img_bytes = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(img_bytes)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(img_bytes)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Error reading image: {e}")

    def handle_api_detect(self):
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length == 0:
            self.send_json({"error": "Empty request body"}, HTTPStatus.BAD_REQUEST)
            return

        if content_length > MAX_BODY_SIZE:
            self.send_json({"error": "Payload exceeds 25MB limit"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return

        body = self.rfile.read(content_length)

        # Parse JSON payload
        model_name = "yolo11s_best"
        conf_thresh = 0.25
        iou_thresh = 0.45
        image_bytes = None
        sample_id = None

        if "application/json" in content_type:
            try:
                data = json.loads(body.decode("utf-8"))
                model_name = str(data.get("model", "yolo11s_best"))
                conf_thresh = float(data.get("conf", 0.25))
                iou_thresh = float(data.get("iou", 0.45))
                raw_sample = data.get("sample_id")
                if raw_sample and isinstance(raw_sample, str):
                    sample_id = raw_sample
                b64_val = data.get("image_base64") or data.get("image")
                if b64_val:
                    b64_str = str(b64_val)
                    if "," in b64_str:
                        b64_str = b64_str.split(",", 1)[1]
                    image_bytes = base64.b64decode(b64_str)
            except Exception as e:
                self.send_json({"error": f"Invalid JSON payload: {e}"}, HTTPStatus.BAD_REQUEST)
                return
        elif "multipart/form-data" in content_type:
            try:
                # Minimal multipart parser
                boundary = content_type.split("boundary=")[1].encode()
                parts = body.split(b"--" + boundary)
                for part in parts:
                    if b"Content-Disposition" in part:
                        headers, part_body = part.split(b"\r\n\r\n", 1)
                        part_body = part_body.rstrip(b"\r\n--")
                        headers_str = headers.decode("latin1", errors="ignore")
                        if 'name="model"' in headers_str:
                            model_name = part_body.decode("utf-8", errors="ignore").strip()
                        elif 'name="conf"' in headers_str:
                            conf_thresh = float(part_body.decode("utf-8", errors="ignore").strip())
                        elif 'name="iou"' in headers_str:
                            iou_thresh = float(part_body.decode("utf-8", errors="ignore").strip())
                        elif 'name="image"' in headers_str:
                            image_bytes = part_body
            except Exception as e:
                self.send_json({"error": f"Multipart parsing error: {e}"}, HTTPStatus.BAD_REQUEST)
                return

        # Model allowlist validation with alias normalization
        canonical_model = MODEL_ALIASES.get(model_name, model_name)
        if canonical_model not in ALLOWED_MODELS:
            self.send_json({"error": f"Unrecognized or forbidden model: {model_name}"}, HTTPStatus.BAD_REQUEST)
            return
        model_name = canonical_model

        # Safe sample_id handling (resolves from webapp/samples or dataset directory)
        if not image_bytes and sample_id:
            safe_name = Path(sample_id).name
            if not safe_name.endswith((".jpg", ".png", ".jpeg")):
                safe_name = f"{safe_name}.jpg"

            cand_samples = (WEBAPP_DIR / "samples" / safe_name).resolve()
            cand_dataset = (DATASET_DIR / safe_name).resolve()

            if cand_samples.is_relative_to(WEBAPP_DIR / "samples") and cand_samples.exists():
                with open(cand_samples, "rb") as f:
                    image_bytes = f.read()
            elif cand_dataset.is_relative_to(DATASET_DIR) and cand_dataset.exists():
                with open(cand_dataset, "rb") as f:
                    image_bytes = f.read()

        if not image_bytes:
            self.send_json({"error": "No image data supplied"}, HTTPStatus.BAD_REQUEST)
            return

        # Execute detection with PyTorch/Ultralytics
        model = get_model(model_name)
        if not model:
            # Fallback check in precomputed sample data
            fallback_res = self.find_sample_fallback(sample_id, model_name, conf_thresh)
            if fallback_res:
                self.send_json(fallback_res)
                return

            self.send_json({
                "error": f"Model {model_name} could not be loaded into memory on {DEVICE}.",
                "model": model_name,
            }, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        from PIL import Image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_w, img_h = pil_img.size

            t0 = time.perf_counter()
            results = model(
                pil_img,
                conf=conf_thresh,
                iou=iou_thresh,
                device=DEVICE,
                verbose=False
            )[0]
            total_latency = (time.perf_counter() - t0) * 1000.0

            speed_dict = getattr(results, "speed", {})
            preprocess_ms = speed_dict.get("preprocess", 0.0)
            inference_ms = speed_dict.get("inference", total_latency)
            postprocess_ms = speed_dict.get("postprocess", 0.0)

            boxes_output = []
            for box in results.boxes:
                coords = [round(float(c), 1) for c in box.xyxy[0]]
                conf = round(float(box.conf[0]), 3)
                cls_idx = int(box.cls[0])
                cname = CLASS_NAMES.get(cls_idx, f"D{cls_idx}")
                boxes_output.append({
                    "class_id": cname,
                    "class_name": CLASS_LABELS.get(cname, cname),
                    "confidence": conf,
                    "bbox": coords,
                })

            response = {
                "model": model_name,
                "device": DEVICE,
                "width": img_w,
                "height": img_h,
                "count": len(boxes_output),
                "detections": boxes_output,
                "latency_ms": round(total_latency, 1),
                "speed": {
                    "preprocess_ms": round(preprocess_ms, 1),
                    "inference_ms": round(inference_ms, 1),
                    "postprocess_ms": round(postprocess_ms, 1),
                },
            }
            self.send_json(response)

        except Exception as e:
            self.send_json({"error": f"Inference execution failed: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def find_sample_fallback(self, sample_id, model_name, conf_thresh):
        """Lookup precomputed sample detection when live engine is offline."""
        if not sample_id:
            return None
        samples_file = WEBAPP_DIR / "data" / "samples.json"
        if not samples_file.exists():
            return None
        try:
            with open(samples_file, "r") as f:
                data = json.load(f)
            for s in data:
                if s.get("id") == sample_id or s.get("file", "").endswith(sample_id):
                    preds = s.get("predictions", {}).get(model_name, {})
                    raw_dets = preds.get("detections", [])
                    filtered = [d for d in raw_dets if d.get("confidence", 0) >= conf_thresh]
                    return {
                        "model": model_name,
                        "device": "precomputed",
                        "width": s.get("width", 800),
                        "height": s.get("height", 600),
                        "count": len(filtered),
                        "detections": filtered,
                        "latency_ms": preds.get("latency_ms", 25.0),
                        "speed": {
                            "preprocess_ms": 1.2,
                            "inference_ms": preds.get("latency_ms", 25.0),
                            "postprocess_ms": 0.8,
                        },
                    }
        except Exception:
            pass
        return None

    def send_json(self, data: Any, status: int = HTTPStatus.OK):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def run_server(preferred_port: int = 8000):
    port = preferred_port
    httpd = None
    max_retries = 20

    for attempt in range(max_retries):
        try:
            server_address = ("", port)
            httpd = ReusableThreadingHTTPServer(server_address, RequestHandler)
            break
        except OSError as e:
            if e.errno in (48, 98):  # 48 on macOS (EADDRINUSE), 98 on Linux
                sys.stdout.write(f"[Notice] Port {port} is in use. Trying port {port + 1}...\n")
                port += 1
            else:
                raise e

    if httpd is None:
        sys.stderr.write("Error: Could not bind to any open port.\n")
        sys.exit(1)

    sys.stdout.write(f"AIL-RDD web application available at http://localhost:{port}\n")
    sys.stdout.write(f"Inference device: {DEVICE} (Torch: {TORCH_AVAILABLE}, Ultralytics: {ULTRALYTICS_AVAILABLE})\n")
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stdout.write("\nShutting down server.\n")
        httpd.server_close()


if __name__ == "__main__":
    env_port = os.environ.get("PORT")
    port = int(env_port) if env_port and env_port.isdigit() else 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
