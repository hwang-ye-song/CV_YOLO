"""노트북 출력 이미지를 가볍게 만든다 (GitHub 웹에서 노트북이 열리도록).

    python tools/shrink_notebook.py notebooks/yolov8_manufacturing_practice.ipynb

- 가로가 MAX_W 보다 큰 이미지는 비율을 유지해 줄인다
- PNG 를 JPEG(품질 85)로 바꿨을 때 더 작아지는 경우에만 바꾼다 (사진이 들어간 그림)
글자 출력과 코드는 건드리지 않는다.
"""
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

MAX_W = 1400
QUALITY = 85


def shrink(b64: str) -> tuple[str, str]:
    """(mime, base64) 를 돌려준다."""
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    png = io.BytesIO(); im.save(png, "PNG", optimize=True)
    jpg = io.BytesIO(); im.convert("RGB").save(jpg, "JPEG", quality=QUALITY, optimize=True)
    best_mime, best = ("image/jpeg", jpg) if jpg.tell() < png.tell() * 0.8 else ("image/png", png)
    return best_mime, base64.b64encode(best.getvalue()).decode()


def main(path: str):
    p = Path(path)
    before = p.stat().st_size
    nb = json.loads(p.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" in data:
                mime, b64 = shrink("".join(data.pop("image/png")))
                data[mime] = b64
                out.get("metadata", {}).pop("image/png", None)
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{before / 1e6:.1f} MB → {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
