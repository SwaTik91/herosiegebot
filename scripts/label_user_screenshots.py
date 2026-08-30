#!/usr/bin/env python3
"""Label kept Hero Siege screenshots with OCR + HP-bar heuristics."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from ocrmac import ocrmac

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "user-screenshots"
IMAGES = ROOT / "images"
LABELS = ROOT / "labels"
PREVIEW = ROOT / "preview"
OCR_CACHE = ROOT / "ocr-cache.json"

NAMES = ["player", "companion", "enemy", "loot", "vein", "chest", "stash", "waypoint"]
NAME_TO_ID = {n: i for i, n in enumerate(NAMES)}

ENEMY_NAMES = (
    "master of arrows",
    "high monk",
    "warbringer",
    "malevolent spirit",
    "brute",
    "cultist",
    "archer",
    "skeleton",
    "spider",
    "swamp",
    "spirit",
    "monk",
    "wraith",
    "fiend",
    "horror",
)

LOOT_HINTS = (
    "belt",
    "sword",
    "boot",
    "rune",
    "talons",
    "shroud",
    "cane",
    "paw",
    "helm",
    "ring",
    "amulet",
    "glove",
    "armor",
    "shield",
    "staff",
    "bow",
    "axe",
    "mace",
    "wand",
    "cloak",
    "robe",
    "gold",
    "sapphire",
    "ruby",
    "emerald",
    "diamond",
    "crystal",
    "essence",
    "key",
    "chipped",
)

HUD_SKIP = (
    "satanic",
    "misty swamp",
    "zone level",
    "ping",
    "fps",
    "quests",
    "security",
    "beacon",
    "picked up",
    "автопот",
    "порог",
    "калибр",
    "player",
    "dps",
    "hell",
    "safe zone",
    "voted",
    "reset",
    "loot essence",
)


def vision_box(box: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x, y, bw, bh = box
    px = int(round(x * w))
    pw = int(round(bw * w))
    ph = int(round(bh * h))
    py = int(round((1.0 - y - bh) * h))
    return px, py, pw, ph


def in_hud(x: int, y: int, bw: int, bh: int, w: int, h: int) -> bool:
    cx, cy = x + bw / 2, y + bh / 2
    if cy < 170 and (cx < 450 or cx > w - 320):
        return True
    if cx < 430 and cy < 280:
        return True
    if cx > w - 280:
        return True
    if cy > h - 150:
        return True
    return False


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def nms(boxes: list[tuple[int, int, int, int, int]], thr: float = 0.45):
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int, int]] = []
    for box in boxes:
        if all(iou(box[:4], k[:4]) < thr or box[4] != k[4] for k in kept):
            kept.append(box)
    return kept


def expand(x: int, y: int, bw: int, bh: int, w: int, h: int, down: int, pad: int = 8):
    x = max(0, x - pad)
    y = max(0, y - 4)
    bw = min(w - x, bw + pad * 2)
    bh = min(h - y, bh + down)
    return x, y, bw, bh


def yolo_line(cls: int, x: int, y: int, bw: int, bh: int, w: int, h: int) -> str:
    return f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def looks_like_map(bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    teal = cv2.inRange(hsv, (80, 40, 80), (110, 180, 220))
    h, w = teal.shape
    roi = teal[int(h * 0.18) : int(h * 0.82), int(w * 0.18) : int(w * 0.82)]
    ratio = float(np.count_nonzero(roi)) / roi.size
    return ratio > 0.08


def classify_text(text: str) -> str | None:
    t = text.strip()
    low = t.lower()
    if any(s in low for s in HUD_SKIP):
        return None
    if "vein" in low:
        return "vein"
    if "stash" in low:
        return "stash"
    if "waypoint" in low:
        return "waypoint"
    if "chest" in low:
        return "chest"
    if "jerry" in low:
        return "player"
    if "diarea" in low or "random goose" in low or low.endswith("goose") or low == "m goose":
        return "companion"
    if re.search(r"[|I]\s*[SABC]\b", t) or "|" in t or any(h in low for h in LOOT_HINTS):
        if "vein" in low:
            return "vein"
        return "loot"
    if any(n in low for n in ENEMY_NAMES) and "jerry" not in low:
        return "enemy"
    return None


def red_hp_boxes(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    play = np.zeros((h, w), np.uint8)
    play[165 : h - 155, 80 : w - 290] = 255
    play[0:270, 0:450] = 0
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 150, 110), (7, 255, 255)),
        cv2.inRange(hsv, (172, 150, 110), (179, 255, 255)),
    )
    red = cv2.bitwise_and(red, play)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    out = []
    n, _, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if 18 <= bw <= 80 and 3 <= bh <= 8 and 2.5 <= bw / max(bh, 1) <= 14 and area / (bw * bh) >= 0.45:
            if not in_hud(int(x), int(y), int(bw), int(bh), w, h):
                out.append((int(x), max(0, int(y) - 10), int(bw), min(int(bh) + 22, h - int(y))))
    return out


def load_ocr(path: Path) -> list:
    cache = json.loads(OCR_CACHE.read_text()) if OCR_CACHE.exists() else {}
    key = path.name
    if key not in cache:
        cache[key] = ocrmac.OCR(str(path)).recognize()
        OCR_CACHE.write_text(json.dumps(cache))
    return cache[key]


def label_image(path: Path) -> tuple[list[tuple[int, int, int, int, int]], bool]:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return [], False
    h, w = bgr.shape[:2]
    if looks_like_map(bgr):
        return [], True

    boxes: list[tuple[int, int, int, int, int]] = []
    for text, conf, box in load_ocr(path):
        if conf < 0.4:
            continue
        x, y, bw, bh = vision_box(box, w, h)
        if in_hud(x, y, bw, bh, w, h):
            continue
        cls = classify_text(str(text))
        if not cls:
            continue
        down = {"player": 70, "companion": 55, "enemy": 50, "vein": 36, "loot": 22, "chest": 40, "stash": 55, "waypoint": 80}[cls]
        x, y, bw, bh = expand(x, y, bw, bh, w, h, down=down)
        boxes.append((x, y, bw, bh, NAME_TO_ID[cls]))

    for x, y, bw, bh in red_hp_boxes(bgr):
        if any(iou((x, y, bw, bh), b[:4]) > 0.15 and b[4] == NAME_TO_ID["enemy"] for b in boxes):
            continue
        boxes.append((x, y, bw, bh, NAME_TO_ID["enemy"]))

    if path.name == "shot_00022.png":
        boxes = [b for b in boxes if b[4] not in {NAME_TO_ID["enemy"], NAME_TO_ID["loot"], NAME_TO_ID["vein"]}]
        boxes.append((636, 150, 70, 70, NAME_TO_ID["stash"]))
        boxes.append((710, 450, 180, 130, NAME_TO_ID["waypoint"]))
        boxes.append((755, 425, 75, 75, NAME_TO_ID["player"]))

    return nms(boxes), False


def main() -> None:
    LABELS.mkdir(exist_ok=True)
    PREVIEW.mkdir(exist_ok=True)
    colors = {
        0: (0, 255, 0),
        1: (255, 180, 0),
        2: (0, 0, 255),
        3: (0, 255, 255),
        4: (0, 215, 255),
        5: (180, 105, 255),
        6: (255, 0, 255),
        7: (255, 255, 0),
    }
    counts = {n: 0 for n in NAMES}
    maps = []
    labeled = 0
    for path in sorted(IMAGES.glob("shot_*.png")):
        boxes, is_map = label_image(path)
        if is_map:
            maps.append(path.name)
            continue
        vis = cv2.imread(str(path))
        h, w = vis.shape[:2]
        lines = []
        for x, y, bw, bh, cls in boxes:
            counts[NAMES[cls]] += 1
            lines.append(yolo_line(cls, x, y, bw, bh, w, h))
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), colors[cls], 2)
            cv2.putText(vis, NAMES[cls], (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[cls], 1)
        (LABELS / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        cv2.imwrite(str(PREVIEW / path.name), vis)
        if lines:
            labeled += 1
    print("labeled", labeled)
    print("counts", counts)
    print("map_frames", maps)


if __name__ == "__main__":
    main()
