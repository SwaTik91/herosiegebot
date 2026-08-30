from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

from hero_siege_bot.yolo_labels import YoloBox, format_yolo_text, parse_yolo_text

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "user-screenshots"
STATIC = ROOT / "scripts" / "labeler"
HOST = "127.0.0.1"
PORT = 8765


def class_names(data_dir: Path = DATA) -> list[str]:
    payload = yaml.safe_load((data_dir / "data.yaml").read_text(encoding="utf-8"))
    names = payload["names"]
    if not isinstance(names, list) or not names:
        raise ValueError("data.yaml must list class names")
    return [str(name) for name in names]


def image_names(data_dir: Path = DATA) -> list[str]:
    return [path.name for path in sorted((data_dir / "images").glob("shot_*.png"))]


def label_path(name: str, data_dir: Path = DATA) -> Path:
    if name != Path(name).name or not name.endswith(".png"):
        raise ValueError("invalid image name")
    return data_dir / "labels" / f"{Path(name).stem}.txt"


def boxes_payload(name: str, data_dir: Path, names: list[str]) -> dict[str, object]:
    path = label_path(name, data_dir)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    boxes = parse_yolo_text(text, class_count=len(names))
    return {
        "boxes": [
            {"cls": box.cls, "x": box.x, "y": box.y, "w": box.w, "h": box.h}
            for box in boxes
        ]
    }


def save_boxes(
    name: str,
    raw_boxes: list[dict[str, object]],
    data_dir: Path,
    names: list[str],
) -> int:
    boxes = tuple(
        YoloBox(
            int(item["cls"]),
            float(item["x"]),
            float(item["y"]),
            float(item["w"]),
            float(item["h"]),
        )
        for item in raw_boxes
    )
    parse_yolo_text(format_yolo_text(boxes), class_count=len(names))
    path = label_path(name, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_yolo_text(boxes), encoding="utf-8")
    return len(boxes)


class LabelHandler(BaseHTTPRequestHandler):
    data_dir = DATA
    static_dir = STATIC
    names: list[str] = []

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path in {"/", "/index.html"}:
            self._send_file(self.static_dir / "index.html", "text/html; charset=utf-8")
            return
        if path == "/api/meta":
            self._send_json(
                {"classes": self.names, "images": image_names(self.data_dir)}
            )
            return
        if path.startswith("/api/label/"):
            name = path.removeprefix("/api/label/")
            try:
                payload = boxes_payload(name, self.data_dir, self.names)
            except ValueError as error:
                self._send_status(400, str(error))
                return
            self._send_json(payload)
            return
        if path.startswith("/images/"):
            name = Path(path.removeprefix("/images/")).name
            image = self.data_dir / "images" / name
            if not image.is_file():
                self._send_status(404, "missing image")
                return
            content_type = mimetypes.guess_type(image.name)[0] or "image/png"
            self._send_file(image, content_type)
            return
        self._send_status(404, "not found")

    def do_PUT(self) -> None:
        path = unquote(urlparse(self.path).path)
        if not path.startswith("/api/label/"):
            self._send_status(404, "not found")
            return
        name = path.removeprefix("/api/label/")
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        try:
            count = save_boxes(
                name,
                list(payload.get("boxes", [])),
                self.data_dir,
                self.names,
            )
        except (ValueError, KeyError, TypeError) as error:
            self._send_status(400, str(error))
            return
        self._send_json({"ok": True, "count": count})

    def _send_json(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_status(self, code: int, message: str) -> None:
        body = message.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(
    *,
    host: str = HOST,
    port: int = PORT,
    open_browser: bool = True,
    data_dir: Path = DATA,
) -> ThreadingHTTPServer:
    names = class_names(data_dir)
    handler = type(
        "BoundLabelHandler",
        (LabelHandler,),
        {"data_dir": data_dir, "static_dir": STATIC, "names": names},
    )
    class ReusableServer(HTTPServer):
        allow_reuse_address = True

    server = None
    error: OSError | None = None
    for try_port in range(port, port + 10):
        try:
            server = ReusableServer((host, try_port), handler)
            port = try_port
            break
        except OSError as exc:
            error = exc
    if server is None:
        raise SystemExit(f"could not bind {host}:{port}: {error}")
    url = f"http://{host}:{port}/"
    print(f"labeler {url}", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    return server


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local YOLO screenshot labeler")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    print("starting labeler", flush=True)
    if not (DATA / "images").is_dir():
        raise SystemExit(f"no images in {DATA / 'images'}")
    server = serve(port=args.port, open_browser=not args.no_browser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
