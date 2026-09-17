# CV-YOLO — YOLOv8 기반 제조 데이터 객체 탐지

제조 공정 영상(작업대 위 **신발 + 도장**)에서 YOLO로 물체를 탐지하고, 모델·데이터·촬영 조건에 따른 성능 범위를 실험한 프로젝트입니다.

- 📓 **실습 노트북**: [notebooks/yolov8_manufacturing_practice.ipynb](notebooks/yolov8_manufacturing_practice.ipynb)
- 📄 **제출용 PDF** (노트북 전체, 59쪽): [CV-YOLO_제출본.pdf](CV-YOLO_제출본.pdf)

## 노트북 구성

과제 기본 코드를 순서대로 따라가며 **코드마다 아래에 의미를 정리**하고, 파트가 끝날 때마다 **개인 실험**을 덧붙였습니다.

| 파트 | 내용 |
|---|---|
| Part 1. YOLOv8 확인하기 | 버스 이미지 추론, `Results`/`Boxes` 구조, xywh·xyxy·정규화 좌표 직접 계산, 시각화, 클래스 필터링, crop |
| 🔬 개인 실험 1 | 신뢰도 임계값, 추가 샘플(zidane) × 모델 3종(YOLOv8n/s, **YOLO11n**), NMS IoU |
| Part 2. YOLOv8 학습 | Roboflow stamp v10 데이터 분석 → 학습(20 epoch) → 평가표 자동 생성 → test 예측 |
| 🔬 개인 실험 2 | 모델·에폭 비교, 클래스별 임계값 곡선, **촬영 조건 5종**(원본/어둡게/밝게/흐림/노이즈), 오류 사례 분석 |
| 정리 | 목표 기준 평가, 개인 회고(KPT / AAR) |

## 결과 요약

**기본 학습 (YOLOv8n, 20 epoch, valid 273장)**

| Class | Precision | Recall | mAP@.5 | mAP@.5:.95 |
|---|---|---|---|---|
| all | 0.942 | 0.971 | 0.984 | 0.642 |
| shoes | 0.944 | 0.944 | 0.972 | 0.710 |
| stamp | 0.941 | 0.998 | 0.995 | 0.574 |

**모델·에폭 비교 (test 193장)**

| 실험 | 파라미터 | mAP50 | mAP50-95 | 추론(ms/장) |
|---|---|---|---|---|
| YOLOv8n · 20ep | 3.01M | 0.984 | **0.637** | **2.8** |
| YOLOv8n · 50ep | 3.01M | 0.979 | 0.626 | 2.9 |
| YOLO11n · 20ep | 2.59M | 0.982 | 0.634 | 3.1 |
| YOLOv8s · 20ep | 11.14M | 0.975 | 0.628 | 6.6 |

![](report/images/stamp_experiments.png)

**촬영 조건 변화 (test 193장)**

| 조건 | mAP50 | mAP50-95 | shoes Recall | stamp Recall |
|---|---|---|---|---|
| original | 0.984 | 0.635 | 0.952 | 0.995 |
| dark | 0.983 | 0.637 | 0.938 | 0.989 |
| bright | 0.980 | 0.640 | 0.922 | 0.987 |
| blur | 0.976 | 0.548 | 0.844 | 0.987 |
| noise | 0.777 | 0.478 | **0.223** | 0.950 |

**핵심 발견**
- 더 오래 학습하거나(50 epoch) 큰 모델(v8s)을 써도 성능이 오르지 않았습니다. 한계는 모델보다 **데이터**에 있습니다. 작은 도장은 박스 경계를 정확히 맞추기 어렵고, 손에 가려진 신발은 놓칩니다.
- 조명 변화에는 강하지만 **센서 노이즈에서 신발 탐지가 급락**합니다. 현장에서는 조명 확보와 노이즈 증강이 필요합니다.
- 도장은 Recall 0.998로, "도장이 찍혔는지" 확인하는 공정 목적에는 충분합니다.

## 레포 구조

```
CV-YOLO/
├── notebooks/
│   ├── yolov8_manufacturing_practice.ipynb   # 실습 + 개인 실험 + 회고 (제출 본문)
│   └── 03_my_defect_detection.ipynb          # (진행 예정) 직접 촬영한 정상/불량 제품 탐지
├── src/                  # 노트북 흐름을 재사용 가능한 스크립트로 리팩터링
│   ├── common.py         # 경로·클래스·디바이스 설정
│   ├── capture.py        # 웹캠 촬영
│   ├── label.py          # OpenCV 박스 라벨링 도구 → YOLO txt
│   ├── split.py          # train/val 분할 + data.yaml 생성
│   ├── train.py          # 파인튜닝
│   ├── evaluate.py       # mAP + 판 단위 OK/NG 판정, 촬영 조건 강건성
│   └── live.py           # 웹캠 실시간 정상/불량 표시
├── tools/make_pdf.py     # 노트북 → 제출용 PDF
├── report/images/        # 그래프
├── results/              # 실험 결과 csv
└── CV-YOLO_제출본.pdf
```

`data/`, `runs/`, `*.pt`는 용량이 커서 커밋하지 않습니다.

## 재현 방법

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50 시리즈
pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name cv-yolo --display-name "CV-YOLO (.venv)"
```

1. [Roboflow stamp 데이터셋](https://universe.roboflow.com/warisara-kaewsuwan-cf2hs/stamp-bcrhe) **v10**을 YOLOv8 형식으로 받아 `data/stamp/`에 압축 해제
2. `Jupyter_실행.bat` 실행 → `yolov8_manufacturing_practice.ipynb`를 **CV-YOLO (.venv)** 커널로 실행
3. PDF 생성: `python tools/make_pdf.py`

## 환경
Windows 11 · Python 3.11 · PyTorch 2.11 (CUDA 12.8) · Ultralytics 8.4.154 · RTX 5060 Laptop 8GB
