# CV-YOLO — YOLOv8 기반 제조 데이터 객체 탐지

제조 공정 영상(작업대 위 **신발 + 도장**)에서 YOLO로 물체를 탐지하고, 모델·데이터·촬영 조건에 따른 성능 범위를 실험한 프로젝트입니다.

- 📓 **실습 노트북**: [notebooks/yolov8_manufacturing_practice.ipynb](notebooks/yolov8_manufacturing_practice.ipynb) (GitHub 표시용: 이미지 압축·출력 정리)
  - 원본(실행 직후 그대로, 10.7MB): [yolov8_manufacturing_practice_original.ipynb](notebooks/yolov8_manufacturing_practice_original.ipynb) — 용량 때문에 GitHub 웹에서는 안 열릴 수 있어 내려받아 Jupyter로 열어 주세요
- 📄 **제출용 PDF** (노트북 전체, 50쪽): [CV-YOLO_제출본.pdf](CV-YOLO_제출본.pdf)

## 노트북 구성

**과제 기본 코드를 원본 순서·코드 그대로** 따라가며 코드마다 아래에 의미를 정리하고, 파트가 끝날 때마다 **개인 실험**을 덧붙였습니다.
(원본과 다른 곳은 문법 오류였던 `import requests=` 수정, 평가표에 이번 실행 값을 넣은 것 두 곳뿐입니다.)

| 파트 | 내용 |
|---|---|
| Part 1. YOLOv8 확인하기 | 버스 이미지 추론, `Results`/`Boxes` 구조, xywh·xyxy·정규화 좌표 직접 계산, 시각화, 클래스 필터링, crop |
| 🔬 개인 실험 1 | 신뢰도 임계값, 추가 샘플(zidane)로 모델 비교(버전: v8n vs **11n**, 크기: v8n vs v8s), IoU·NMS |
| Part 2. YOLOv8 학습 | Roboflow stamp v10 학습(20 epoch) → 검증 → 평가표 → test 이미지 예측 |
| 🔬 개인 실험 2 | 2-1 모델 비교(버전·크기·학습 길이) / 2-2 신뢰도 최적화 — 각각 가설 → 결과 → 이유 순서 |
| 정리 | 목표 기준 평가, **막혔던 부분과 해결**, 개인 회고(KPT / AAR) |

## 결과 요약

**기본 학습 (YOLOv8n, 20 epoch, valid 273장)**

| Class | Precision | Recall | mAP@.5 | mAP@.5:.95 |
|---|---|---|---|---|
| all | 0.953 | 0.987 | 0.985 | 0.640 |
| shoes | 0.928 | 0.974 | 0.975 | 0.717 |
| stamp | 0.978 | 1.000 | 0.995 | 0.562 |

**모델 비교 (test 193장)** — 같은 조건끼리 비교

| 모델 | 비교 | mAP50 | mAP50-95 | 파라미터 |
|---|---|---|---|---|
| YOLOv8n · 20에폭 | 기준 | 0.985 | 0.627 | 3.0M |
| YOLO11n · 20에폭 | 버전 (nano끼리) | 0.982 | **0.634** | **2.6M** |
| YOLOv8s · 20에폭 | 크기 (v8끼리) | 0.975 | 0.628 | 11.1M |
| YOLOv8n · 50에폭 | 학습 길이 | 0.979 | 0.626 | 3.0M |

![](report/images/stamp_experiments.png)

**신뢰도 최적화 (YOLO11n, test 193장)**

| | 신뢰도 | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|
| 기본값 | 0.25 | 0.961 | 0.946 | 0.954 |
| 최적값 (valid에서 선택) | 0.55 | **0.980** | 0.935 | **0.957** |

**핵심 발견**
- 최신 모델(YOLO11n)이 v8n보다 조금 우수하고 더 가볍지만, 모델 크기·버전·학습 길이를 바꿔도 성능 차이는 0.01 이내였습니다. 한계는 모델보다 **데이터**(가장자리·손에 든 신발 같은 드문 장면)에 있습니다.
- 신뢰도를 올리면 오탐이 줄어 정밀도가 0.98까지 오르지만, 전체 성능(F1)은 거의 같습니다.
- 도장은 검증 재현율 1.000으로, "도장이 찍혔는지" 확인하는 공정 목적에는 충분합니다.

## 레포 구조

```
CV-YOLO/
├── notebooks/
│   ├── yolov8_manufacturing_practice.ipynb            # 실습 + 개인 실험 + 회고 (제출 본문, GitHub 표시용)
│   ├── yolov8_manufacturing_practice_original.ipynb   # 위와 같은 내용의 원본 (이미지·출력 그대로)
│   └── 03_my_defect_detection.ipynb          # (진행 예정) 직접 촬영한 정상/불량 제품 탐지
├── src/                  # 노트북 흐름을 재사용 가능한 스크립트로 리팩터링
│   ├── common.py         # 경로·클래스·디바이스 설정
│   ├── capture.py        # 웹캠 촬영
│   ├── label.py          # OpenCV 박스 라벨링 도구 → YOLO txt
│   ├── split.py          # train/val 분할 + data.yaml 생성
│   ├── train.py          # 파인튜닝
│   ├── evaluate.py       # mAP + 판 단위 OK/NG 판정, 촬영 조건 강건성
│   └── live.py           # 웹캠 실시간 정상/불량 표시
├── tools/
│   ├── make_pdf.py           # 노트북 → 제출용 PDF
│   └── shrink_notebook.py    # 노트북 이미지 용량 줄이기 (GitHub 표시용)
├── report/images/        # 그래프
├── results/              # 실험 결과 csv
└── CV-YOLO_제출본.pdf
```

`data/`, `datasets/`, `runs/`, `*.pt`는 용량이 커서 커밋하지 않습니다.

## 재현 방법

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50 시리즈
pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name cv-yolo --display-name "CV-YOLO (.venv)"
```

1. [Roboflow stamp 데이터셋](https://universe.roboflow.com/warisara-kaewsuwan-cf2hs/stamp-bcrhe) **v10**을 YOLOv8 형식으로 받아 `datasets/stamp/`에 압축 해제 (노트북의 `../datasets/stamp/...` 경로와 맞춤)
2. `Jupyter_실행.bat` 실행 → `yolov8_manufacturing_practice.ipynb`를 **CV-YOLO (.venv)** 커널로 실행
3. 이미지 용량 줄이기: `python tools/shrink_notebook.py notebooks/yolov8_manufacturing_practice.ipynb`
4. PDF 생성: `python tools/make_pdf.py`

## 환경
Windows 11 · Python 3.11 · PyTorch 2.11 (CUDA 12.8) · Ultralytics 8.4.154 · RTX 5060 Laptop 8GB
