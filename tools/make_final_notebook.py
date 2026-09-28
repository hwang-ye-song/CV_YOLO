"""최종 확인 노트북: 실습 노트북(1부) + 틀린그림찾기 노트북(2부)을 하나로 묶고 PDF도 만든다.

    python tools/make_final_notebook.py

결과
- 최종확인_노트북.ipynb : 두 노트북을 합친 것 (GitHub에서 열리도록 이미지 압축)
- 최종확인_노트북.pdf   : 표지 + 1부 + 2부

각 부의 실행 결과는 원래 노트북을 실행한 결과를 그대로 옮긴 것이다.
(1부는 학습이 여러 번 들어 있어 다시 실행하면 오래 걸린다.)
"""
import copy
import shutil
import sys
from pathlib import Path

import nbformat
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import make_pdf  # noqa: E402
import shrink_notebook  # noqa: E402

PRACTICE = ROOT / "notebooks" / "yolov8_manufacturing_practice_original.ipynb"
SPOTDIFF = ROOT / "spotdiff" / "spotdiff_yolo.ipynb"
OUT_NB = ROOT / "최종확인_노트북.ipynb"
OUT_PDF = ROOT / "최종확인_노트북.pdf"
PART1 = "1부. YOLOv8 제조 데이터 객체 탐지 실습"
PART2 = "2부. YOLO로 틀린그림찾기"

INTRO = f"""# 최종 확인 노트북

과제 실습과 직접 만든 틀린그림찾기를 **한 노트북에 모은 최종 확인용 노트북**이다.

| 부 | 내용 | 원래 노트북 |
|---|---|---|
| {PART1} | 과제 기본 코드(버스 이미지 추론, stamp 데이터 학습·평가)를 원본 그대로 따라가고, 파트마다 개인 실험(모델 비교, 신뢰도 최적화)과 회고(KPT/AAR)를 덧붙였다 | `notebooks/yolov8_manufacturing_practice.ipynb` |
| {PART2} | Gemini로 문제 그림 5장을 만들고, YOLO로 두 그림의 물체를 찾아 비교하는 탐지기를 만들어 채점했다 (재현율 65.7%, 정밀도 100%) | `spotdiff/spotdiff_yolo.ipynb` |

각 부의 실행 결과는 원래 노트북을 실행한 결과를 그대로 옮긴 것이다. 1부는 학습이 여러 번 들어 있어 처음부터 다시 실행하면 시간이 오래 걸린다.
각 부 앞의 작업 폴더 이동 칸은 이 노트북을 레포 맨 위에서 다시 실행할 때를 위한 것이다.
틀린그림찾기를 만들며 겪은 시행착오는 `spotdiff/report/작업기록.md`에 따로 정리했다.
"""


def part_cells(src: Path, title: str, workdir: str):
    nb = nbformat.read(src, as_version=4)
    cells = [copy.deepcopy(c) for c in nb.cells]
    # 원래 제목 줄을 'n부' 제목으로 바꾼다
    first = cells[0]
    lines = first.source.splitlines()
    first.source = "\n".join([f"# {title}"] + lines[1:])
    chdir = nbformat.v4.new_code_cell(
        f"# 이 부의 코드는 {workdir}/ 폴더 기준 경로를 쓴다\n"
        "import os\n"
        "from pathlib import Path\n"
        "REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'spotdiff').is_dir())   # 레포 맨 위\n"
        f"os.chdir(REPO / '{workdir}')")
    if workdir == "spotdiff":   # 최종 노트북은 레포 맨 위에 있으므로 링크 경로를 맞춘다
        for c in cells:
            if c.cell_type == "markdown":
                c.source = c.source.replace("](report/", "](spotdiff/report/")
    return [first, chdir] + cells[1:]


def renumber(nb):
    """두 노트북의 셀 id가 겹치지 않게 새로 붙인다."""
    for i, c in enumerate(nb.cells):
        c["id"] = f"cell-{i:03d}"


def build():
    p1 = part_cells(PRACTICE, PART1, "notebooks")
    p2 = part_cells(SPOTDIFF, PART2, "spotdiff")
    base = nbformat.read(SPOTDIFF, as_version=4)
    full = nbformat.v4.new_notebook(metadata=base.metadata)
    full.cells = [nbformat.v4.new_markdown_cell(INTRO)] + p1 + p2
    renumber(full)
    return full, p1, p2


def make_pdf_file(p1, p2):
    tmp = ROOT / "build" / "final_pdf"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    make_pdf.TITLE = "최종 확인 노트북<br>YOLO 실습 + 틀린그림찾기"
    make_pdf.SUBTITLE = "과제 실습(제조 데이터 탐지·개인 실험·회고) → 직접 만든 YOLO 틀린그림찾기 탐지기"
    sections = []
    for i, (title, cells) in enumerate([(PART1, [nbformat.v4.new_markdown_cell(INTRO)] + p1), (PART2, p2)]):
        nb = nbformat.v4.new_notebook(cells=[copy.deepcopy(c) for c in cells])
        renumber(nb)
        path = tmp / f"part{i + 1}.ipynb"
        nbformat.write(nb, path)
        sections.append((title, make_pdf.notebook_html(path)))
    edge = next(p for p in make_pdf.EDGE_CANDIDATES if p.exists())
    pages = [("표지", make_pdf.cover_html([t for t, _ in sections]))] + sections
    writer = PdfWriter()
    for i, (title, doc) in enumerate(pages):
        h, p = tmp / f"{i:02d}.html", tmp / f"{i:02d}.pdf"
        h.write_text(doc, encoding="utf-8")
        make_pdf.print_pdf(edge, h, p)
        start = len(writer.pages)
        writer.append(PdfReader(p))
        writer.add_outline_item(title, start)
        print(f"{title}: {len(writer.pages) - start}쪽")
    writer.add_metadata({"/Title": "최종 확인 노트북 — YOLO 실습 + 틀린그림찾기", "/Author": make_pdf.AUTHOR})
    with open(OUT_PDF, "wb") as f:
        writer.write(f)
    print(f"PDF → {OUT_PDF} (총 {len(writer.pages)}쪽)")


def main():
    full, p1, p2 = build()
    make_pdf_file(p1, p2)
    nbformat.write(full, OUT_NB)
    # GitHub 웹에서 열리도록 그림을 줄인다 (두 노트북 분량이라 기존보다 더 작게)
    shrink_notebook.MAX_W, shrink_notebook.QUALITY = 720, 60
    shrink_notebook.main(str(OUT_NB))
    print(f"노트북 → {OUT_NB}")


if __name__ == "__main__":
    main()
