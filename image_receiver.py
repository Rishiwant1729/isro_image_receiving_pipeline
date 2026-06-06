import argparse
import json
import re
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "test"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")


def json_response(handler, status_code, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def validate_filename(filename):
    name = Path(filename).name
    suffix = Path(name).suffix.lower()

    if not name or name != filename:
        return None, "invalid filename"
    if not SAFE_FILENAME.match(name):
        return None, "filename can only contain letters, numbers, dots, dashes, and underscores"
    if suffix not in ALLOWED_EXTENSIONS:
        return None, "unsupported image extension"

    return name, None


class ImageUploadHandler(BaseHTTPRequestHandler):
    server_version = "ISROTestImageReceiver/1.0"

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            json_response(
                self,
                200,
                {
                    "status": "ok",
                    "test_dir": str(TEST_DIR),
                },
            )
            return

        json_response(
            self,
            404,
            {
                "status": "error",
                "message": "use POST /upload?filename=test.png",
            },
        )

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/upload":
            json_response(self, 404, {"status": "error", "message": "unknown endpoint"})
            return

        query = parse_qs(parsed.query)
        filename_values = query.get("filename", [])
        if not filename_values:
            json_response(self, 400, {"status": "error", "message": "missing filename"})
            return

        filename, error = validate_filename(filename_values[0])
        if error:
            json_response(self, 400, {"status": "error", "message": error})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            json_response(self, 400, {"status": "error", "message": "invalid content length"})
            return

        if content_length <= 0:
            json_response(self, 400, {"status": "error", "message": "empty upload"})
            return
        if content_length > MAX_UPLOAD_BYTES:
            json_response(self, 413, {"status": "error", "message": "image too large"})
            return

        TEST_DIR.mkdir(exist_ok=True)
        final_path = TEST_DIR / filename

        with tempfile.NamedTemporaryFile(
            dir=TEST_DIR,
            prefix=f".{filename}.",
            suffix=".part",
            delete=False,
        ) as temp_file:
            remaining = content_length
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    temp_file.close()
                    Path(temp_file.name).unlink(missing_ok=True)
                    json_response(
                        self,
                        400,
                        {"status": "error", "message": "upload ended early"},
                    )
                    return
                temp_file.write(chunk)
                remaining -= len(chunk)

            temp_path = Path(temp_file.name)

        temp_path.replace(final_path)
        print(f"Received test image: {final_path}")
        json_response(
            self,
            200,
            {
                "status": "saved",
                "filename": filename,
                "path": str(final_path),
                "bytes": content_length,
            },
        )


def run_receiver(host, port):
    TEST_DIR.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((host, port), ImageUploadHandler)
    print(f"Image receiver listening on http://{host}:{port}")
    print(f"Saving uploaded test images to {TEST_DIR}")
    print("Health check: /health")
    print("Upload endpoint: POST /upload?filename=test.png")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nImage receiver stopped.")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(
        description="Receive uploaded test images and save them into the test folder."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to listen on.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    args = parser.parse_args()

    run_receiver(args.host, args.port)


if __name__ == "__main__":
    main()
