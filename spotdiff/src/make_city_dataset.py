"""COCO 도시 사진으로 틀린그림찾기 데이터셋을 만든다 (문제 사진 A/B + 정답).

    python spotdiff/src/make_city_dataset.py --n 20 --diffs 3

- 도시 장면(자동차·버스·트럭·신호등이 충분히 보이는 사진)만 고른다.
- 사진 A는 원본, 사진 B는 물체를 지우거나(remove) 옆으로 옮기거나(move) 하나 더 놓아(copy) 만든다.
- 박스 안에서 물체 모양만 따내(GrabCut) 지우고 붙이므로 배경이 같이 따라오지 않는다.
- 옮기거나 붙일 때는 땅 위에 놓이도록 발밑 높이를 맞춘다.
- 바꾼 위치는 COCO 정답 박스를 그대로 쓰므로, 정답이 자동으로 기록된다.

결과: spotdiff/dataset_city/
    A/xxx.jpg, B/xxx.jpg          문제 사진 한 쌍
    answer/xxx.jpg                정답 표시 (A | B 나란히, 바뀐 곳에 번호)
    answers.json                  사진별 정답 목록 (종류, 물체, A에서의 박스, B에서의 박스)
    answer_sheet.jpg              전체 정답을 한 장에 모은 확인용 이미지
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COCO = ROOT / "data" / "coco128"
OUT = ROOT / "dataset_city"

NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "traffic light",
         10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 24: "backpack", 26: "handbag",
         28: "suitcase", 25: "umbrella", 16: "dog"}
VEHICLE_SIGN = {2, 3, 5, 7, 9, 10, 11, 12}
GROUND = {0, 1, 2, 3, 5, 7, 10, 12, 13, 16}          # 땅 위에 서 있는 물체 (신호등·표지판은 제외)
KIND_KO = {"remove": "사라짐", "move": "이동", "copy": "추가"}


def load_boxes(img_path, w, h):
    lbl = COCO / "labels" / "train2017" / (img_path.stem + ".txt")
    out = []
    for line in lbl.read_text().splitlines() if lbl.exists() else []:
        c, cx, cy, bw, bh = map(float, line.split()[:5])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        out.append((int(c), max(0, x1), max(0, y1), min(w, x1 + int(bw * w)), min(h, y1 + int(bh * h))))
    return out


def is_city(objs, w, h):
    """차·버스·트럭·신호등 등이 하나라도 사진의 1% 이상 크기로 보이면 도시 장면"""
    return any(o[0] in VEHICLE_SIGN and (o[3] - o[1]) * (o[4] - o[2]) >= w * h * 0.01 for o in objs)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0


def usable(obj, others, w, h):
    """지우거나 옮기기 좋은 물체: 너무 크거나 작지 않고, 가장자리에 붙지 않고, 다른 물체와 거의 안 겹침"""
    c, x1, y1, x2, y2 = obj
    area = (x2 - x1) * (y2 - y1) / (w * h)
    if not (0.003 <= area <= 0.08) or c not in NAMES or (x2 - x1) < 15 or (y2 - y1) < 15:
        return False
    if x1 < 2 or y1 < 2 or x2 > w - 2 or y2 > h - 2:
        return False
    return all(iou(obj[1:], o[1:]) < 0.15 for o in others if o is not obj)


def object_mask(img, box, fallback=True):
    """박스 안에서 물체 모양만 따낸다 (GrabCut). 실패하면 박스 안쪽 타원으로 대신한다.
    fallback=False 이면 실패했을 때 None 을 돌려준다 (붙여 넣을 조각에는 타원을 쓰지 않는다)."""
    x1, y1, x2, y2 = box
    mask = np.zeros(img.shape[:2], np.uint8)
    try:
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, (x1, y1, x2 - x1, y2 - y1), bg, fg, 4, cv2.GC_INIT_WITH_RECT)
        m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        m = np.zeros(img.shape[:2], np.uint8)
    ratio = m[y1:y2, x1:x2].mean() / 255
    if not fallback and not (0.2 <= ratio <= 0.9):   # 너무 적거나 박스를 거의 다 채우면 실패로 본다
        return None
    if ratio < 0.15:                              # 너무 조금 잡히면 타원으로 대신
        m[:] = 0
        cv2.ellipse(m, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 255, -1)
    return m


def erase(img, obj_mask, grow=7):
    """물체 모양(조금 넓혀서)만 주변 배경으로 메운다."""
    m = cv2.dilate(obj_mask, np.ones((grow, grow), np.uint8), iterations=2)
    return cv2.inpaint(img, m, 9, cv2.INPAINT_TELEA)


def paste(img, patch, patch_mask, dst_xy):
    """물체 조각을 부드러운 가장자리로 합성한다."""
    out = img.copy()
    ph, pw = patch.shape[:2]
    x, y = dst_xy
    alpha = cv2.GaussianBlur(patch_mask.astype(np.float32) / 255, (7, 7), 0)[..., None]
    roi = out[y:y + ph, x:x + pw].astype(np.float32)
    out[y:y + ph, x:x + pw] = (alpha * patch + (1 - alpha) * roi).astype(np.uint8)
    return out


def ground_spot(size, bottom_y, taken, w, h, rng, tries=200):
    """발밑 높이(bottom_y)를 맞춘 채 옆으로만 움직여, 겹치지 않는 자리를 찾는다."""
    pw, ph = size
    y = int(bottom_y - ph)
    if y < 0 or bottom_y > h:
        return None
    for _ in range(tries):
        x = rng.randint(3, max(3, w - pw - 3))
        cand = (x, y, x + pw, y + ph)
        if all(iou(cand, t) == 0 for t in taken):
            return cand
    return None


def build_bank():
    """다른 사진에서 가져와 붙일 물체 조각 (땅 위에 서는 물체만)"""
    bank = []
    for p in sorted((COCO / "images" / "train2017").glob("*.jpg")):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        objs = load_boxes(p, w, h)
        for o in objs:
            if o[0] in GROUND and usable(o, objs, w, h):
                x1, y1, x2, y2 = o[1:]
                m = object_mask(img, (x1, y1, x2, y2), fallback=False)
                if m is not None:
                    bank.append((o[0], img[y1:y2, x1:x2].copy(), m[y1:y2, x1:x2], p.name))
    return bank


def make_pair(img_path, n_diffs, rng, bank):
    a = cv2.imread(str(img_path))
    h, w = a.shape[:2]
    objs = load_boxes(img_path, w, h)
    if not objs or not is_city(objs, w, h):
        return None
    cands = [o for o in objs if usable(o, objs, w, h)]
    rng.shuffle(cands)

    b = a.copy()
    taken = [o[1:] for o in objs]
    diffs = []
    for obj in cands[:n_diffs]:
        c, x1, y1, x2, y2 = obj
        box = (x1, y1, x2, y2)
        m = object_mask(a, box)
        kind = rng.choice(["remove", "remove", "move"]) if c in GROUND else "remove"
        if kind == "move":
            spot = ground_spot((x2 - x1, y2 - y1), y2, taken + [box], w, h, rng)
            if spot is not None and abs(spot[0] - x1) > (x2 - x1):
                b = erase(b, m)
                b = paste(b, a[y1:y2, x1:x2], m[y1:y2, x1:x2], spot[:2])
                taken.append(spot)
                diffs.append({"kind": "move", "class": NAMES[c], "box_a": box, "box_b": spot})
                continue
        b = erase(b, m)
        diffs.append({"kind": "remove", "class": NAMES[c], "box_a": box, "box_b": None})

    # 바꿀 물체가 모자라면 다른 사진의 물체를 이 사진 속 비슷한 물체 크기·발밑 높이에 맞춰 놓는다
    anchors = [o for o in objs if o[0] in GROUND]
    tries = 0
    while len(diffs) < n_diffs and bank and anchors and tries < 60:
        tries += 1
        c, patch, pm, src_name = bank[rng.randrange(len(bank))]
        if src_name == img_path.name:
            continue
        ref = [o for o in anchors if o[0] == c] or anchors
        _, rx1, ry1, rx2, ry2 = rng.choice(ref)
        scale = (ry2 - ry1) / patch.shape[0] * (1.0 if c == ref[0][0] else 0.8)
        pw, ph = int(patch.shape[1] * scale), int(patch.shape[0] * scale)
        if pw < 12 or ph < 12 or pw > w * 0.4 or ph > h * 0.6:
            continue
        spot = ground_spot((pw, ph), ry2, taken, w, h, rng)
        if spot is None:
            continue
        b = paste(b, cv2.resize(patch, (pw, ph)), cv2.resize(pm, (pw, ph)), spot[:2])
        taken.append(spot)
        diffs.append({"kind": "copy", "class": NAMES[c], "box_a": None, "box_b": spot})
    if len(diffs) < n_diffs:
        return None
    return a, b, diffs


def draw_answer(a, b, diffs):
    va, vb = a.copy(), b.copy()
    for i, d in enumerate(diffs, 1):
        for im, box, thick in ((va, d["box_a"], 2), (vb, d["box_b"], 2), (vb, d["box_a"] if d["kind"] == "remove" else None, 1)):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cv2.rectangle(im, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), thick)
            cv2.putText(im, str(i), (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    gap = np.full((a.shape[0], 12, 3), 255, np.uint8)
    return cv2.hconcat([va, gap, vb])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="만들 문제 수")
    ap.add_argument("--diffs", type=int, default=3, help="문제당 차이 개수")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for sub in ("A", "B", "answer"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    bank = build_bank()
    print("붙여 넣을 물체 조각", len(bank), "개")
    answers, thumbs = {}, []
    for p in sorted((COCO / "images" / "train2017").glob("*.jpg")):
        res = make_pair(p, args.diffs, rng, bank)
        if res is None:
            continue
        a, b, diffs = res
        qid = f"city_{len(answers) + 1:02d}"
        cv2.imwrite(str(OUT / "A" / f"{qid}.jpg"), a)
        cv2.imwrite(str(OUT / "B" / f"{qid}.jpg"), b)
        ans = draw_answer(a, b, diffs)
        cv2.imwrite(str(OUT / "answer" / f"{qid}.jpg"), ans)
        answers[qid] = {"source": p.name, "size": [a.shape[1], a.shape[0]],
                        "differences": [{"no": i, **d} for i, d in enumerate(diffs, 1)]}
        t = cv2.resize(ans, (640, int(ans.shape[0] * 640 / ans.shape[1])))
        cv2.putText(t, qid, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        thumbs.append(t)
        if len(answers) == args.n:
            break

    (OUT / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = []
    for i in range(0, len(thumbs), 2):
        pair = thumbs[i:i + 2]
        hmax = max(t.shape[0] for t in pair)
        pair = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0] + 8, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in pair]
        if len(pair) == 1:
            pair.append(np.full_like(pair[0], 255))
        rows.append(cv2.hconcat(pair))
    if rows:
        cv2.imwrite(str(OUT / "answer_sheet.jpg"), cv2.vconcat(rows))
    # 사람이 확인하기 쉬운 정답표
    lines = ["# 도시 틀린그림찾기 정답표", "",
             "사진 A(원본)와 B(바뀐 사진)를 비교했을 때의 정답이다. `answer/` 폴더의 이미지에 같은 번호로 표시돼 있다.",
             "위치는 사진을 가로·세로 3칸씩 나눴을 때 어느 칸인지로 적었다.", ""]
    def where(box, w, h):
        cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
        return ["위", "가운데", "아래"][min(2, int(cy * 3))] + " " + ["왼쪽", "가운데", "오른쪽"][min(2, int(cx * 3))]
    for qid, v in answers.items():
        w, h = v["size"]
        lines += [f"## {qid} (원본 {v['source']})", "", "| 번호 | 종류 | 물체 | 위치 |", "|---|---|---|---|"]
        for d in v["differences"]:
            if d["kind"] == "remove":
                pos = f"A의 {where(d['box_a'], w, h)}에 있던 것이 B에서 없어짐"
            elif d["kind"] == "move":
                pos = f"A의 {where(d['box_a'], w, h)} → B의 {where(d['box_b'], w, h)}"
            else:
                pos = f"B의 {where(d['box_b'], w, h)}에 새로 생김"
            lines.append(f"| {d['no']} | {KIND_KO[d['kind']]} | {d['class']} | {pos} |")
        lines.append("")
    (OUT / "정답표.md").write_text("\n".join(lines), encoding="utf-8")

    kinds = [d["kind"] for v in answers.values() for d in v["differences"]]
    print(f"문제 {len(answers)}개 | 차이 {len(kinds)}개 (사라짐 {kinds.count('remove')}, 이동 {kinds.count('move')}, 추가 {kinds.count('copy')})")


if __name__ == "__main__":
    main()
