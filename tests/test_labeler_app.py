from __future__ import annotations

from pathlib import Path

from hero_siege_bot.labeler_app import boxes_payload, save_boxes


def _dataset(tmp_path: Path) -> Path:
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "data.yaml").write_text(
        "names: [player, companion, enemy, loot]\n",
        encoding="utf-8",
    )
    (tmp_path / "images" / "shot_00001.png").write_bytes(
        Path(__file__).parent.joinpath("fixtures").joinpath("tiny.png").read_bytes()
        if Path(__file__).parent.joinpath("fixtures").joinpath("tiny.png").is_file()
        else _tiny_png()
    )
    (tmp_path / "labels" / "shot_00001.txt").write_text(
        "2 0.500000 0.400000 0.200000 0.100000\n",
        encoding="utf-8",
    )
    return tmp_path


def _tiny_png() -> bytes:
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")),
            chunk(b"IEND", b""),
        ]
    )


def test_boxes_payload_reads_existing_yolo_label(tmp_path: Path) -> None:
    data = _dataset(tmp_path)
    payload = boxes_payload("shot_00001.png", data, ["player", "companion", "enemy", "loot"])
    assert payload["boxes"] == [{"cls": 2, "x": 0.5, "y": 0.4, "w": 0.2, "h": 0.1}]


def test_save_boxes_round_trip(tmp_path: Path) -> None:
    data = _dataset(tmp_path)
    names = ["player", "companion", "enemy", "loot"]
    save_boxes(
        "shot_00001.png",
        [{"cls": 3, "x": 0.25, "y": 0.75, "w": 0.1, "h": 0.2}],
        data,
        names,
    )
    assert boxes_payload("shot_00001.png", data, names)["boxes"] == [
        {"cls": 3, "x": 0.25, "y": 0.75, "w": 0.1, "h": 0.2}
    ]
