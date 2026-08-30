from hero_siege_bot.yolo_labels import YoloBox, format_yolo_text, parse_yolo_text


def test_parse_and_format_round_trip_yolo_boxes() -> None:
    text = "2 0.500000 0.400000 0.100000 0.200000\n0 0.250000 0.750000 0.050000 0.080000\n"
    boxes = parse_yolo_text(text, class_count=8)

    assert boxes == (
        YoloBox(2, 0.5, 0.4, 0.1, 0.2),
        YoloBox(0, 0.25, 0.75, 0.05, 0.08),
    )
    assert parse_yolo_text(format_yolo_text(boxes), class_count=8) == boxes


def test_parse_yolo_text_skips_blank_and_comments() -> None:
    assert parse_yolo_text("# keep\n\n3 0.1 0.2 0.3 0.4\n", class_count=8) == (
        YoloBox(3, 0.1, 0.2, 0.3, 0.4),
    )
