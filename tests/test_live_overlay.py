import numpy as np

from hero_siege_bot.live_overlay import (
    OVERLAY_CLASS_NAME,
    is_game_focused,
    pack_dib_bgra,
)


def test_game_stays_focused_when_overlay_is_foreground() -> None:
    assert is_game_focused(10, 10, "Hero Siege")
    assert is_game_focused(10, 99, OVERLAY_CLASS_NAME)
    assert is_game_focused(10, 99, "Unknown", "Hero Siege Bot Overlay")
    assert not is_game_focused(10, 99, "chrome")


def test_pack_dib_bgra_is_32bit_bottom_up() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)
    image[1, 2] = (40, 50, 60)

    packed = np.frombuffer(pack_dib_bgra(image), dtype=np.uint8).reshape(2, 3, 4)

    assert packed[0, 2, :3].tolist() == [40, 50, 60]
    assert packed[1, 0, :3].tolist() == [10, 20, 30]
    assert packed[:, :, 3].tolist() == [[255, 255, 255], [255, 255, 255]]


def test_pack_dib_bgra_makes_chroma_key_transparent() -> None:
    from hero_siege_bot.diagnostics import CHROMA_KEY_BGR

    image = np.full((1, 2, 3), CHROMA_KEY_BGR, dtype=np.uint8)
    image[0, 1] = (0, 255, 0)

    packed = np.frombuffer(pack_dib_bgra(image), dtype=np.uint8).reshape(1, 2, 4)

    assert packed[0, 0, 3] == 0
    assert packed[0, 1, :4].tolist() == [0, 255, 0, 255]
