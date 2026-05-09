import asyncio
import argparse
import json
import os
import pathlib
from collections import deque
import warnings
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
import supervision as sv
import websockets
from PIL import Image, ImageDraw
from rfdetr import RFDETRMedium
from trackers import ByteTrackTracker
from turbojpeg import TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE, TJPF_RGB, TurboJPEG
from torch.jit import TracerWarning

warnings.filterwarnings("ignore", category=TracerWarning)

# --- RF-DETR Model ---

MODEL_WEIGHTS = pathlib.Path(__file__).parent / "model" / "checkpoint_best_total.pth"
model_h = 576
model_w = 576


def load_model():
    # kwargs = {
    #     "num_classes": 60,
    #     "resolution": model_w,
    # }
    # return RFDETRMedium(pretrain_weights=str(MODEL_WEIGHTS), **kwargs)
    return RFDETRMedium(pretrain_weights=str(MODEL_WEIGHTS))


print(f"Loading {MODEL_WEIGHTS}...")
model = load_model()
model.optimize_for_inference()

# --- TurboJPEG ---

if os.name == "nt":
    jpeg = TurboJPEG(r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll")
else:
    jpeg = TurboJPEG()

# --- Thread pool ---

executor = ThreadPoolExecutor(max_workers=1)
save_history_dir = None
save_history_slots = deque()

# --- Warmup ---

dummy_input = np.zeros((1, 3, model_h, model_w), dtype=np.float32)
try:
    model.predict(Image.fromarray(np.zeros((model_h, model_w, 3), dtype=np.uint8)))
    print("Model warmup complete.")
except Exception as e:
    print(f"Warmup failed: {e}")

# --- Class & health data ---

CLASS_MAPPING = {
    1: "AH Biologisch Penne",
    2: "AH Biologisch Smeuïge pindakaas met stukjes pinda",
    3: "AH Biologisch Stevige crackers kaas",
    4: "AH Cranberry notenmix ongezouten",
    5: "AH Dunne crackers kaas",
    6: "AH Dunne crackers volkoren",
    7: "AH Rode linzen penne",
    8: "AH Stevige crackers waldkorn",
    9: "AH Terra Elitehaver ongebrand",
    10: "AH Terra Jonge kapucijners",
    11: "AH Terra Noten cranberry rozijn mix ongebrand",
    12: "AH Terra Plantaardig biologisch 100% pindakaas",
    13: "AH Terra Plantaardige erwtendrink ongezoet",
    14: "AH Terra Plantaardige haverdrink ongezoet",
    15: "AH Terra Plantaardige sojadrink",
    16: "Alesto Noten mix",
    17: "Alpro Barista soja",
    18: "Alpro Haverdrink zonder suikers",
    19: "Alpro Sojadrink Houdbaar",
    20: "BioToday Peanutbutter crunchy bio",
    21: "Bolletje Low carb groentecrackers paprika",
    22: "Bolletje Oerknack Waldkorn",
    23: "Bolletje Ontbijt Crackers Spelt Volkoren",
    24: "Bonduelle Chilibonen in saus",
    25: "Bonduelle Zwarte bonen",
    26: "Bonne Maman Pindapasta romig",
    27: "Calvé Pindakaas Original",
    28: "De Cecco Penne Rigate Nr. 41",
    29: "Duyvis Oven baked peanuts honey salt",
    30: "Duyvis Oven roasted pinda's original",
    31: "Ekoplaza Volkoren penne",
    32: "Freshona Linzen Lentils lidl",
    33: "Grand' Italia Penne tradizionali",
    34: "Hak Bonen Bruine",
    35: "Hak Kapucijners",
    36: "Hak witte bonen in tomatensaus",
    37: "Heinz Beanz",
    38: "Jumbo's 100% Biologische Pindakaas Naturel",
    39: "Jumbo Groentecrackers Wortel & Zoete Aardappel",
    40: "Jumbo Haverdrink Naturel",
    41: "Jumbo Hollandse Bruine Bonen",
    42: "Jumbo Organic Whole Wheat Rigatoni",
    43: "Jumbo Penne Rigate Whole Wheat",
    44: "Jumbo's 100% Natural Peanut Butter",
    45: "Jumbo Sojadrink Naturel",
    46: "Jumbo Stevige Crackers Spelt 8",
    47: "Jumbo Studentenhaver Ongezouten",
    48: "La bioidea Penne",
    49: "La Molisana Penne rigate",
    50: "Lidl cashew cranberrymix mix alesto",
    51: "Luna E Terra Pindakaas Smooth Bio",
    52: "Natural Happiness nut mix Raw",
    53: "Natural Happiness proteïnemix",
    54: "Oatly! Organic oat drink",
    55: "Rude Health Barista soja",
    56: "Rummo Volkoren Biologische Penne Rigate",
    57: "Skippy Creamy peanut butter",
    58: "Smaakt Bio Zwarte Bonen",
    59: "TastyBasics Low carb-high protein cracker meerzaden",
    60: "Whole earth Drizzler golden roasted peanut butter",
}

SCHIJF_VAN_VIJF = {
    # Peanut Butter
    "Calvé Pindakaas Original": False,
    "Skippy Creamy peanut butter": False,
    "Bonne Maman Pindapasta romig": False,
    "AH Biologisch Smeuïge pindakaas met stukjes pinda": False,
    "BioToday Peanutbutter crunchy bio": False,
    "Jumbo's 100% Natural Peanut Butter": True,
    "Whole earth Drizzler golden roasted peanut butter": True,
    "AH Terra Plantaardig biologisch 100% pindakaas": True,
    "Jumbo's 100% Biologische Pindakaas Naturel": True,
    "Luna E Terra Pindakaas Smooth Bio": True,
    # Pasta
    "De Cecco Penne Rigate Nr. 41": False,
    "La Molisana Penne rigate": False,
    "AH Biologisch Penne": False,
    "Grand' Italia Penne tradizionali": False,
    "La bioidea Penne": False,
    "Jumbo Penne Rigate Whole Wheat": True,
    "Jumbo Organic Whole Wheat Rigatoni": True,
    "Rummo Volkoren Biologische Penne Rigate": True,
    "Ekoplaza Volkoren penne": True,
    "AH Rode linzen penne": True,
    # Crackers
    "AH Biologisch Stevige crackers kaas": False,
    "AH Dunne crackers volkoren": False,
    "Jumbo Stevige Crackers Spelt 8": False,
    "Jumbo Groentecrackers Wortel & Zoete Aardappel": False,
    "Bolletje Oerknack Waldkorn": False,
    "AH Dunne crackers kaas": True,
    "AH Stevige crackers waldkorn": True,
    "Bolletje Low carb groentecrackers paprika": True,
    "Bolletje Ontbijt Crackers Spelt Volkoren": True,
    "TastyBasics Low carb-high protein cracker meerzaden": True,
    # Nuts
    "AH Cranberry notenmix ongezouten": False,
    "AH Terra Noten cranberry rozijn mix ongebrand": False,
    "Lidl cashew cranberrymix mix alesto": False,
    "Duyvis Oven baked peanuts honey salt": False,
    "Duyvis Oven roasted pinda's original": False,
    "Natural Happiness proteïnemix": True,
    "AH Terra Elitehaver ongebrand": True,
    "Alesto Noten mix": True,
    "Jumbo Studentenhaver Ongezouten": True,
    "Natural Happiness nut mix Raw": True,
    # Beans
    "Bonduelle Chilibonen in saus": False,
    "Hak witte bonen in tomatensaus": False,
    "Heinz Beanz": False,
    "Smaakt Bio Zwarte Bonen": False,
    "Hak Bonen Bruine": False,
    "Jumbo Hollandse Bruine Bonen": True,
    "Bonduelle Zwarte bonen": True,
    "Freshona Linzen Lentils lidl": True,
    "Hak Kapucijners": True,
    "AH Terra Jonge kapucijners": True,
    # Plant-based Drinks
    "AH Terra Plantaardige haverdrink ongezoet": False,
    "Alpro Haverdrink zonder suikers": False,
    "Jumbo Haverdrink Naturel": False,
    "Oatly! Organic oat drink": False,
    "Rude Health Barista soja": False,
    "AH Terra Plantaardige erwtendrink ongezoet": True,
    "Alpro Sojadrink Houdbaar": True,
    "Alpro Barista soja": True,
    "AH Terra Plantaardige sojadrink": True,
    "Jumbo Sojadrink Naturel": True,
    # Pasta Sauce
    "Spagheroni Tradizionale": False,
    "Heinz Traditional pasta sauce": False,
    "La Dolce Vita Traditional Pastasaus": False,
    "AH Biologisch Pastasaus tradizionale": False,
    "Fertilia Pastasaus traditionale": False,
    "Ecoplaza Pastasaus sugo tradizionale": False,
}

# --- Shared settings (overrides persisted to disk) ---

SETTINGS_FILE = pathlib.Path(__file__).parent / "settings.json"


def load_overrides():
    try:
        data = json.loads(SETTINGS_FILE.read_text())
        return data.get("overrides", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_overrides(overrides):
    SETTINGS_FILE.write_text(json.dumps({"overrides": overrides}, indent=2))


# Global overrides dict (product_name -> bool), applied on top of SCHIJF_VAN_VIJF
health_overrides = load_overrides()

NUM_PRODUCT_CLASSES = max(CLASS_MAPPING) + 1
MIN_TRACKING_SCORE = 0.5
TRACK_ACTIVATION_SCORE = 0.7
TRACK_HIGH_CONF_SCORE = 0.9

# --- Coordinate math ---


def get_real_coordinates(boxes_norm, orig_w, orig_h, model_dim=560):
    """
    boxes_norm: np.array of shape (N, 4) -> [cx, cy, w, h] (0-1 normalized)
    Returns: np.array of shape (N, 4) -> [x1, y1, x2, y2] in original image coords
    """
    boxes_px = boxes_norm * model_dim

    scale = min(model_dim / orig_w, model_dim / orig_h)
    pad_x = (model_dim - orig_w * scale) / 2
    pad_y = (model_dim - orig_h * scale) / 2

    cx_real = (boxes_px[:, 0] - pad_x) / scale
    cy_real = (boxes_px[:, 1] - pad_y) / scale
    w_real = boxes_px[:, 2] / scale
    h_real = boxes_px[:, 3] / scale

    x1 = cx_real - (w_real / 2)
    y1 = cy_real - (h_real / 2)
    x2 = cx_real + (w_real / 2)
    y2 = cy_real + (h_real / 2)

    return np.stack([x1, y1, x2, y2], axis=1)


# --- Inference ---


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def build_detection(xyxy, class_id, confidence, tracker_id=None):
    class_name = CLASS_MAPPING[int(class_id)]
    x1, y1, x2, y2 = xyxy

    # Apply user overrides on top of default health data
    if class_name in health_overrides:
        healthy = health_overrides[class_name]
    else:
        healthy = SCHIJF_VAN_VIJF.get(class_name, True)

    det = {
        "class": class_name,
        "confidence": float(confidence),
        "in_schijf_van_vijf": healthy,
        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
    }
    if tracker_id is not None:
        det["tracker_id"] = int(tracker_id)
    return det


def clamp(n, lo, hi):
    if n < lo:
        return lo
    if n > hi:
        return hi
    return n


def draw_detections(image, detections):
    draw = ImageDraw.Draw(image)
    w, h = image.size

    for det in detections:
        x, y, bw, bh = det["bbox"]
        x2 = x + bw
        y2 = y + bh

        x = clamp(x, 0, w - 1)
        y = clamp(y, 0, h - 1)
        x2 = clamp(x2, 0, w - 1)
        y2 = clamp(y2, 0, h - 1)
        if x2 <= x or y2 <= y:
            continue

        is_healthy = det.get("in_schijf_van_vijf", True)
        color = "green" if is_healthy else "red"
        draw.rectangle([x, y, x2, y2], outline=color, width=3)
        label = f"{det['class']} | {det['confidence']:.2f}"
        if "tracker_id" in det:
            label += f" | id {det['tracker_id']}"
        draw.text((x + 2, y + 2), label, fill=color)


def run_model_inference(image, min_score):
    raw_detections = model.predict(image, threshold=min_score)

    if len(raw_detections) == 0:
        return sv.Detections.empty()

    class_ids = raw_detections.class_id.astype(int)
    valid_mask = np.array(
        [int(class_id) in CLASS_MAPPING for class_id in class_ids],
        dtype=bool,
    )

    if not np.any(valid_mask):
        return sv.Detections.empty()

    confidence = raw_detections.confidence
    if confidence is None:
        confidence = np.ones(len(raw_detections), dtype=np.float32)

    return sv.Detections(
        xyxy=raw_detections.xyxy[valid_mask].astype(np.float32),
        confidence=confidence[valid_mask].astype(np.float32),
        class_id=class_ids[valid_mask].astype(int),
    )


def build_tracked_detections(sv_dets):
    detections = []

    for i in range(len(sv_dets)):
        track_id = None
        if sv_dets.tracker_id is not None:
            track_id = int(sv_dets.tracker_id[i])

        # Ignore tentative tracks until ByteTrack confirms them with a real ID.
        if track_id is None or track_id < 0:
            continue

        class_id = int(sv_dets.class_id[i])
        confidence = float(sv_dets.confidence[i])
        bbox = sv_dets.xyxy[i].astype(np.float32).copy()

        detections.append(
            build_detection(
                bbox,
                class_id,
                confidence,
                tracker_id=track_id,
            )
        )

    return detections


def build_raw_detections(sv_dets):
    detections = []
    for i in range(len(sv_dets)):
        detections.append(
            build_detection(
                sv_dets.xyxy[i].astype(np.float32).copy(),
                int(sv_dets.class_id[i]),
                float(sv_dets.confidence[i]),
            )
        )
    return detections


def save_input_frame(image, request_id, detections=None):
    if save_history_dir is None:
        return

    if len(save_history_slots) >= 10:
        old_raw_path, old_annotated_path = save_history_slots.popleft()
        for old_path in (old_raw_path, old_annotated_path):
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass

    raw_path = save_history_dir / f"frame_{request_id:010d}.jpg"
    image.save(raw_path, format="JPEG", quality=95)

    annotated_path = save_history_dir / f"frame_{request_id:010d}_annotated.jpg"
    annotated_image = image.copy()
    draw_detections(annotated_image, detections or [])
    annotated_image.save(annotated_path, format="JPEG", quality=95)

    save_history_slots.append((raw_path, annotated_path))


def run_inference_sync(image_bytes, request_id, tracker, no_track):
    try:
        # Decode
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = jpeg.decode(
            npimg,
            pixel_format=TJPF_RGB,
            flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT,
        )
        if img is None:
            return {"error": "Failed to decode image"}
        img = Image.fromarray(img)

        # Inference
        start = perf_counter()
        sv_dets = run_model_inference(img, MIN_TRACKING_SCORE)
        inference_ms = (perf_counter() - start) * 1000
        print(f"inference {inference_ms:.1f}ms")

        if no_track:
            detections = build_raw_detections(sv_dets)
        else:
            if len(sv_dets) == 0:
                sv_dets = tracker.update(sv.Detections.empty())
            else:
                sv_dets = tracker.update(sv_dets)
            detections = build_tracked_detections(sv_dets)

        save_input_frame(img, request_id, detections)

        return {"detections": detections, "requestId": request_id}

    except Exception as e:
        print(f"Inference error: {e}")
        return {"error": str(e)}


# --- WebSocket server ---


async def inference_worker(websocket, queue, tracker, no_track):
    while True:
        message = await queue.get()
        try:
            # Binary protocol: [4 bytes uint32 requestId BE] + [JPEG bytes]
            request_id = int.from_bytes(message[:4], byteorder="big")
            image_bytes = message[4:]

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                executor,
                run_inference_sync,
                image_bytes,
                request_id,
                tracker,
                no_track,
            )
            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"Worker error: {e}")
        finally:
            queue.task_done()


async def handler(websocket, no_track):
    queue = asyncio.Queue(maxsize=1)
    tracker = ByteTrackTracker(
        track_activation_threshold=TRACK_ACTIVATION_SCORE,
        high_conf_det_threshold=TRACK_HIGH_CONF_SCORE,
        minimum_consecutive_frames=1,
    )
    worker_task = asyncio.create_task(
        inference_worker(websocket, queue, tracker, no_track)
    )
    client = getattr(websocket, "remote_address", None)
    print(f"Client connected: {client}")

    try:
        async for message in websocket:
            # Text messages are settings commands
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    if data.get("type") == "save_settings":
                        overrides = data.get("overrides", {})
                        health_overrides.clear()
                        health_overrides.update(overrides)
                        save_overrides(overrides)
                        print(f"Settings saved: {len(overrides)} overrides")
                    elif data.get("type") == "get_settings":
                        await websocket.send(
                            json.dumps(
                                {"type": "settings", "overrides": health_overrides}
                            )
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Settings error: {e}")
                continue

            # Binary messages are image frames
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await queue.put(message)
    except websockets.exceptions.ConnectionClosed as e:
        print(
            f"Client disconnected: {client}, code={e.code}, reason={e.reason or 'none'}"
        )
    finally:
        worker_task.cancel()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable ByteTrack and return raw model detections directly.",
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save the last 10 decoded input frames before inference.",
    )
    return parser.parse_args()


async def main():
    global save_history_dir
    args = parse_args()
    if args.save_history:
        save_history_dir = pathlib.Path(__file__).parent / "inference-history"
        save_history_dir.mkdir(parents=True, exist_ok=True)
        for path in save_history_dir.glob("frame_*.jpg"):
            path.unlink()
        print(f"Saving last 10 input frames to {save_history_dir}")
    print("Starting WebSocket server on 0.0.0.0:5174...")
    async with websockets.serve(
        lambda websocket: handler(websocket, args.no_track), "0.0.0.0", 5174
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
