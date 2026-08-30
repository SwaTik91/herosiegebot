from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YoloBox:
    cls: int
    x: float
    y: float
    w: float
    h: float


def parse_yolo_text(text: str, class_count: int) -> tuple[YoloBox, ...]:
    boxes: list[YoloBox] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"invalid YOLO line: {raw!r}")
        cls = int(parts[0])
        if cls < 0 or cls >= class_count:
            raise ValueError(f"class id out of range: {cls}")
        x, y, w, h = (float(value) for value in parts[1:])
        x, y, w, h = (min(1.0, max(0.0, value)) for value in (x, y, w, h))
        if w == 0 or h == 0:
            continue
        boxes.append(YoloBox(cls, x, y, w, h))
    return tuple(boxes)


def format_yolo_text(boxes: tuple[YoloBox, ...]) -> str:
    if not boxes:
        return ""
    return (
        "\n".join(
            f"{box.cls} {box.x:.6f} {box.y:.6f} {box.w:.6f} {box.h:.6f}" for box in boxes
        )
        + "\n"
    )
