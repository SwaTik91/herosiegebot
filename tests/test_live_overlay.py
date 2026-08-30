import numpy as np

from hero_siege_bot.live_overlay import pack_dib_bgra


def test_pack_dib_bgra_is_32bit_bottom_up() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)
    image[1, 2] = (40, 50, 60)

    packed = np.frombuffer(pack_dib_bgra(image), dtype=np.uint8).reshape(2, 3, 4)

    assert packed[0, 2, :3].tolist() == [40, 50, 60]
    assert packed[1, 0, :3].tolist() == [10, 20, 30]
    assert packed[:, :, 3].tolist() == [[255, 255, 255], [255, 255, 255]]
