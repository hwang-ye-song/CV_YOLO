# CV-YOLO — YOLO 기반 제조 제품 정상/불량 영역 탐지

검사대 위 여러 제품을 카메라로 비추면 **제품마다 박스(영역)와 정상/불량 상태를 동시에** 표시하고,
한 판에 불량이 하나라도 있으면 **NG**로 판정하는 프로젝트입니다.
직접 촬영·라벨링한 데이터로 YOLO를 학습해 제조 현장 적용 가능성을 확인했습니다.

| 노트북 | 내용 |
|---|---|
| [01_yolov8_basics](notebooks/01_yolov8_basics.ipynb) | YOLOv8 추론, 결과 구조(xywh/xyxy/정규화 좌표) 확인, 클래스 필터링, crop, conf 임계값 실험 |
| [02_stamp_training](notebooks/02_stamp_training.ipynb) | Roboflow `stamp` 데이터셋 학습·평가, 모델(YOLOv8n/s, YOLO11n)·에폭 비교 |
| [03_my_defect_detection](notebooks/03_my_defect_detection.ipynb) | 직접 찍은 사진으로 정상/불량 탐지 학습, test1(기본 성능)·test2(조건 변화), 실시간 판정 |

보고서: [report/REPORT.md](report/REPORT.md)

## 폴더 구조

```
CV-YOLO/
├── src/                  # 노트북 코드를 재사용 가능한 스크립트로 리팩터링
│   ├── common.py         # 경로·클래스·디바이스 설정
│   ├── capture.py        # 웹캠 촬영 (space=저장)
│   ├── label.py          # OpenCV 박스 라벨링 도구 → YOLO txt
│   ├── split.py          # train/val 분할 + data.yaml 생성
│   ├── train.py          # YOLO11n 파인튜닝
│   ├── evaluate.py       # test1(mAP + OK/NG 판정), test2(조명/흐림/노이즈)
│   └── live.py           # 웹캠 실시간 정상/불량 표시
├── notebooks/            # 실험 기록 (.ipynb)
├── report/               # 보고서 + 그래프
├── results/              # 실험 결과 csv/json
└── requirements.txt
```

`data/`, `runs/`, `*.pt`는 용량 문제로 커밋하지 않습니다.

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50 시리즈
pip install -r requirements.txt
python -m ipykernel install --user --name cv-yolo --display-name "CV-YOLO (.venv)"
jupyter lab
```

## 사용법 (스크립트)

```bash
python src/capture.py                # 1) 촬영 → data/raw/images
python src/label.py                  # 2) 박스 라벨링 → data/raw/labels
python src/split.py                  # 3) train/val 분할
python src/train.py --epochs 50      # 4) 학습
python src/evaluate.py test1         # 5) 기본 성능
python src/evaluate.py test2         # 6) 조건 변화 강건성
python src/live.py --conf 0.5        # 7) 실시간 판정
```

라벨링 도구 조작: 드래그=박스, `1`/`2`=normal/defect, 우클릭=클래스 변경, `z`=되돌리기, `d`/`a`=다음/이전, `q`=종료

## 결과 요약

(실험 완료 후 작성)

## 환경
Windows 11 · Python 3.11 · PyTorch 2.11 (CUDA 12.8) · Ultralytics 8.4 · RTX 5060 Laptop 8GB
