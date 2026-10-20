# gpu-training

Persona-based Interview AI 서비스를 위한 **GPU 학습·평가·추론 모듈**입니다.
웹/UI/인증/RAG/PDF 파싱 등 서비스 본체는 포함하지 않으며, GPU가 실제로 필요한 ML 부분만 독립적으로 다룹니다.

## 1. 목적

| 영역 | 내용 | 상태 |
|---|---|---|
| A. Speech AI | 관측 가능한 음성 특성(속도·휴지·피치·에너지·필러) → 행동 점수 | 파이프라인 구현 |
| B. Video Behavior AI | MediaPipe landmark 시퀀스 `[T, F]` → Temporal 모델 | 파이프라인 구현 |
| C. Answer Evaluation AI | 질문+답변 → rubric 항목별 multi-head 회귀 | 파이프라인 구현 |
| D. Multimodal Fusion | speech/video/text embedding → Fusion Transformer | skeleton (smoke test만) |

설계 원칙

* 모델은 **수치만** 반환합니다 (`speech_rate_wpm`, `gaze_center_ratio`, `relevance` …). 사용자에게 보여줄 문장은 서비스/LLM 레이어의 몫입니다.
* 자신감·불안·성격·거짓말·정신건강·채용 적합성 같은 **심리 상태/적합성은 예측하지 않습니다.** (MIT 데이터의 `RecommendHiring`, ChaLearn의 personality/interview 라벨은 adapter에서 의도적으로 제외)
* 데이터가 없으면 학습하지 않습니다. 가짜 데이터로 성능을 보고하지 않습니다. 테스트의 dummy 데이터는 파이프라인 연결만 검증합니다.

```
raw dataset ─ Adapter ─▶ InterviewSample ─ prepare_*.py (CPU) ─▶ data/processed/<task>
                                                                      │
                     configs/<task>.yaml ─▶ common.runner ─▶ common.Trainer (GPU, AMP)
                                                                      │
                 checkpoints/<task>/<task>_vNNN  ◀── best model ──────┘──▶ outputs/<experiment>
                                │
        SpeechAnalyzer / VideoBehaviorModel / AnswerEvaluator .predict()  →  JSON
```

```
gpu-training/
  configs/            base.yaml + speech / video / answer / multimodal
  src/interview_gpu/
    common/           config, device, seed, logging, checkpoint, metrics, tasks, trainer, runner
    schemas/          InterviewSample, BeforeAfterSample, feature/prediction schema
    datasets/         adapter 8종, group split, inspection
    speech/ video/ answer/ multimodal/
  scripts/            check_gpu, inspect_dataset, prepare_*, train_*, evaluate_all
  tests/
  data/ checkpoints/ outputs/ models/   (git 제외)
```

> 제안된 `src/datasets/` 구조는 HuggingFace `datasets` 패키지와 import 이름이 충돌하므로
> 모든 코드를 `src/interview_gpu/` 네임스페이스 아래에 두었습니다.

## 2. 설치

[uv](https://docs.astral.sh/uv/) 사용을 권장합니다 (`uv.lock`으로 버전 고정).

```bash
cd gpu-training
uv sync                    # .venv 생성 + Python 3.12 + CUDA 12.8용 PyTorch 설치
uv sync --extra lora       # (선택) answer 모델 LoRA
uv sync --extra wandb      # (선택) Weights & Biases
```

이후 명령은 `uv run python scripts/...` 또는 `.venv` 활성화 후 `python scripts/...`로 실행합니다.

pip을 쓰는 경우: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` 후 `pip install -e .`

## 3. Python version

Python **3.11 이상** (3.11 – 3.13). 기본값은 `.python-version`의 3.12입니다.

## 4. CUDA 설치 요구사항

* NVIDIA 드라이버만 있으면 됩니다. PyTorch wheel이 CUDA/cuDNN 런타임을 포함하므로 **CUDA Toolkit을 따로 설치할 필요가 없습니다.**
* `pyproject.toml`은 **CUDA 12.8 (cu128)** wheel을 사용합니다 → 드라이버 **570 이상** 필요. RTX 50 시리즈는 cu128 이상이 필수이며, RTX 20/30/40 시리즈도 동작합니다.
* 다른 CUDA 빌드가 필요하면 `pyproject.toml`의 `[[tool.uv.index]]` URL(`cu128`)과 `Dockerfile.gpu`의 `CUDA_VERSION`을 함께 바꾸세요.
* macOS는 PyPI 기본 wheel(MPS/CPU)을 사용합니다. 디바이스 우선순위: **CUDA > MPS > CPU**.

## 5. GPU 확인

```bash
python scripts/check_gpu.py          # --json 지원
```

```
PyTorch: 2.x
CUDA Available: True
CUDA Version: 12.8
GPU Count: 1
GPU 0:
  NVIDIA GeForce RTX XXXX
  VRAM: XX GB
BF16 Support: True
FP16 Support: True
Selected Device: cuda
Recommended Precision: bf16
Mixed Precision: bf16
```

Mixed precision: Ampere 이상 `bf16`, 그 외 CUDA GPU `fp16`(+GradScaler), CPU/MPS `fp32`.
학습 시작 시 같은 정보가 `train.log`에 기록됩니다.

## 6. 데이터 배치 방법

데이터는 **자동으로 다운로드되지 않습니다.** 각 라이선스에 따라 직접 받아 `data/raw/<폴더>/`에 두세요.
데이터가 없으면 모든 명령이 다음과 같이 멈춥니다.

```
ERROR: Dataset not found.

Expected path:
.../gpu-training/data/raw/aihub-interview/

Please place the downloaded dataset in this directory.
```

배치 후 반드시 구조를 확인하세요 (JSON 리포트: `outputs/inspection/<dataset>.json`).

```bash
python scripts/inspect_dataset.py --dataset aihub_interview --path /path/to/data
```

출력: 샘플 수, audio/video/text 수, 누락·손상 파일, label 분포, speaker 수, 평균 길이, 디스크 용량.

## 7. Dataset folder 구조

| # | 데이터 | `--dataset` | `data/raw/` 폴더 | scope |
|---|---|---|---|---|
| 01 | AI Hub 채용면접 인터뷰 (핵심) | `aihub_interview` | `aihub-interview/` | research |
| 02 | AI Hub 감정 분류 대화 음성 | `aihub_speech` | `aihub-emotion-speech/` | research |
| 03 | AI Hub 멀티모달 영상 | `aihub_multimodal` | `aihub-multimodal-video/` | research |
| 04 | AI Hub 안내 로봇 인터랙션 | `aihub_robot_interaction` | `aihub-robot-interaction/` | research |
| 05 | AI Hub 한국인 감정인식 복합 영상 | `aihub_korean_face` | `aihub-korean-emotion-video/` | research |
| 06 | MIT Interview Dataset | `mit_interview` | `mit-interview/` | research |
| 07 | ChaLearn First Impressions V2 | `chalearn` | `chalearn-first-impressions/` | research |
| 08 | 서비스 자체 데이터 (Before/After 포함) | `internal` | `internal/*.jsonl` | production |

세부 구조와 규칙은 [`data/README.md`](data/README.md), 서비스 데이터 JSONL 형식은
`src/interview_gpu/datasets/internal.py` 상단 docstring을 참고하세요.

모든 adapter는 공통 schema `InterviewSample`로 변환합니다
(`sample_id, question, answer, audio_path, video_path, transcript, company, position, job_family, labels, metadata`
+ `speaker_id, session_id, source_dataset, usage_scope, consent`).
새 데이터셋은 `BaseDatasetAdapter`를 상속해 `_iter_samples()`만 구현하고 `datasets/__init__.py`에 등록하면 됩니다.

**Split**: `group_id`(speaker > session > sample) 기준 group split으로 같은 사람이 train/val/test에 섞이지 않습니다.

## 8. Preprocessing (CPU)

```bash
# Speech: handcrafted feature → data/processed/speech/features.csv
python scripts/prepare_speech.py --input /path/to/raw --output data/processed/speech --dataset aihub_interview
#   Stage 2 (frozen encoder embedding, GPU 권장)
python scripts/prepare_speech.py --input ... --dataset mit_interview --encoder microsoft/wavlm-base-plus

# Video: MediaPipe landmark → data/processed/video/sequences/*.npy + index.csv
python scripts/prepare_video.py --input /path/to/video --output data/processed/video --dataset mit_interview --fps 5

# Answer: 질문/답변 → data/processed/answer/answers.jsonl
python scripts/prepare_answer.py --input /path/to/raw --output data/processed/answer --dataset aihub_interview
```

**MediaPipe 모델 파일**: MediaPipe Tasks API는 모델 파일이 필요하며 자동 다운로드하지 않습니다.
한 번만 받아서 `models/mediapipe/`에 두세요 (또는 `--face-model/--pose-model`로 경로 지정).

* `models/mediapipe/face_landmarker.task` ← <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>
* `models/mediapipe/pose_landmarker.task` ← <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task>

**Video feature schema** (프레임당): `timestamp, head_yaw, head_pitch, head_roll, face_visible, face_center_x/y,
gaze_proxy_x/y, left/right_shoulder_x/y, shoulder_angle, upper_body_movement, gesture_motion, landmark_confidence`.
프레임이나 얼굴 이미지는 저장하지 않고 landmark에서 계산한 수치만 저장합니다.

**Speech feature**: `utterance_duration, response_latency, pause_count/total/mean/max, long_pause_count,
pause_rate_per_min, speech_ratio, pitch_mean/std/cv, energy_mean/std_db, voice_stability_proxy,
speech_rate_syllables_per_sec, articulation_rate, speech_rate_wpm, filler_count, filler_per_min`.

## 9. Speech training

```bash
python scripts/train_speech.py --config configs/speech.yaml
python scripts/train_speech.py --config configs/speech.yaml --device cuda --set training.batch_size=64
```

| Stage | 설정 |
|---|---|
| 1. handcrafted → MLP | `model.name: speech_mlp`, `data.params.input: handcrafted` (기본) |
| 2. frozen encoder embedding → MLP | `prepare_speech.py --encoder ...` 후 `data.params.input: embedding` 또는 `both` |
| 3. partial fine-tuning | `model.name: speech_encoder_finetune`, `data.params.input: waveform`, `model.params.trainable_layers: 2` |

`data.target_columns`에 `출력이름: label 컬럼`을 지정합니다. 기본값은 MIT Interview의 행동 평정
(`SpeakingRate, Paused, NoFillers, EngagingTone`)에 맞춰져 있습니다. 다른 데이터에는 해당 라벨이 없을 수 있으며,
그 경우 학습기는 사용 가능한 label 컬럼 목록과 함께 중단합니다.

## 10. Video training

```bash
python scripts/train_video.py --config configs/video.yaml
python scripts/train_video.py --config configs/video.yaml --set model.params.backbone=gru
```

`model.params.backbone`: `gru | lstm | tcn | transformer` (기본 transformer). Raw RGB 모델은 초기 버전에 포함하지 않습니다.

## 11. Answer training

```bash
python scripts/train_answer.py --config configs/answer.yaml
```

Shared encoder(기본 `intfloat/multilingual-e5-small`) + rubric 항목별 head
(`relevance, structure, specificity, evidence, technical_depth, ownership, logical_consistency`, 0–4점: 0 매우 부족 · 2 보통 · 4 매우 좋음).
라벨이 일부만 있는 샘플은 mask로 처리됩니다.

| Stage | 방법 |
|---|---|
| A. 라벨 생성/검증 | `label_answers.py`로 **로컬 LLM judge**(기본 `Qwen/Qwen3-8B`)가 채점 → `prepare_answer.py --labels`로 병합 (범위·항목 검증). **외부 LLM API 호출은 이 프로젝트에 없습니다.** 아래 "Stage A 로컬 채점" 참고 |
| B. encoder baseline | 기본 설정. `model.params.encoder_name`을 `multilingual-e5-base`, `BAAI/bge-m3`(+`input_style: plain`) 등으로 교체 |
| C. LoRA | `uv sync --extra lora` 후 `--set model.params.lora.enabled=true` |
| D. custom evaluator | 자체 Before→After 데이터가 충분히 쌓인 뒤 |

> AI Hub 채용면접 데이터에는 rubric 점수가 없습니다. Stage A 라벨이 준비되기 전에는
> `No labeled samples` 오류로 학습이 중단되는 것이 정상입니다.

**Stage A 로컬 채점** (`scripts/label_answers.py`, 설정 `configs/labeling.yaml`, 4bit/8bit 로딩은 `uv sync --extra labeling`)

```bash
python scripts/label_answers.py --dry-run                           # 200개 → data/labeled/dry_run/ (사람 검수용)
python scripts/label_answers.py --limit 2000 --consistency-runs 2   # rubric 확정 후 검증용 2,000개
python scripts/label_answers.py --full --consistency-runs 2         # 전체. --full 없이는 전체가 실행되지 않음
python scripts/prepare_answer.py --labels data/labeled/qwen3_answers.jsonl
```

* **Rubric V2** (`src/interview_gpu/answer/prompts.py`): 점수는 0~4 정수(0 매우 부족 · 2 보통 · 4 매우 좋음)입니다.
  judge가 먼저 질문 유형(`question_type`, 10종)을 분류하고 항목별 평가 대상 여부(`applicability`)를 정합니다.
  `technical_depth`는 기술·직무 지식을 묻는 질문에서만, `ownership`은 본인 경험을 묻는 질문에서만 채점하며,
  평가 대상이 아닌 항목은 0이 아니라 `null`입니다. `null`은 `--labels` 병합 때 라벨 없음으로 처리되어 loss에서 mask됩니다.
* 출력 레코드: `question_type`, `applicability`, `scores`, `score_evidence`(항목별 인용), `missing_points`,
  `judge_confidence`(HIGH/MEDIUM/LOW), `needs_review`, `review_reason`.
* 출력 JSONL이 곧 checkpoint입니다. 중단 후 같은 명령을 다시 실행하면 이미 채점한 `sample_id`는 건너뜁니다.
  샘플 순서는 seed로 고정되어 200개 ⊂ 2,000개 ⊂ 전체입니다.
* 함께 생성되는 파일: `.review.csv`(질문·답변·점수·근거), `.stats.json`(점수 분포·null 비율·질문 유형 분포·
  Pearson/Spearman 상관·`needs_review` 비율·confidence 분포·JSON 파싱 실패율), `.failed_samples.jsonl`, `.meta.json`.
* `needs_review=true`가 되는 경우: `judge_confidence`가 LOW, `--consistency-runs 2`의 두 채점이 한 항목에서 2점 이상 차이,
  두 채점이 평가 대상 여부를 다르게 판단. 라벨은 첫 번째(greedy) 채점 값이며, `needs_review` 라벨은 `--labels` 병합에서 제외됩니다.
* backend·모델·rubric 프롬프트·채점 설정이 바뀌면 기존 출력에 이어 쓰기를 거부합니다. rubric을 고쳤다면 `--output`을 바꾸세요.
* 두 채점 결과 비교(예: rubric V1 vs V2): `python scripts/compare_labels.py --old <v1.jsonl> --old-scale 10 --new <v2.jsonl>`
* **Backend**: 기본은 `transformers`(Windows·Linux). 16GB VRAM에서는 Qwen3-8B가 자동으로 4bit 로딩됩니다.
  `--backend vllm`은 Linux/WSL2 전용 선택 사항이며(`pip install vllm`), Windows에서는 자동으로 transformers를 씁니다.
  16GB VRAM에서 vLLM을 쓰려면 `--vllm-model`로 사전 양자화 모델을 지정하세요. 24GB 이상이면 `--model Qwen/Qwen3-14B`를 권장합니다.

**Before → After 데이터**: `internal` adapter가 `before_answer/feedback/after_answer/before_scores/after_scores`를 읽어
before·after 두 샘플로 펼치고(같은 group에 묶어 split 누수 방지), `BeforeAfterSample.deltas()`로
`delta_relevance … overall_delta`를 계산합니다.

### 공통 학습 설정 (`configs/base.yaml`)

`batch_size, epochs, lr, weight_decay, gradient_accumulation_steps, mixed_precision(auto|bf16|fp16|fp32),
grad_clip_norm, scheduler, warmup_ratio, early_stopping_patience, num_workers, pin_memory, persistent_workers,
prefetch_factor, compile(torch.compile, 기본 false)`, `experiment.seed / deterministic`,
`training.imbalance.{class_weights, weighted_sampler, focal_loss}`(classification 전용, 기본 off),
`logging.wandb`(없어도 동작). 모든 값은 `--set section.key=value`로 덮어쓸 수 있습니다.

**Metrics** — regression: MAE, RMSE, R², Spearman (head별 + 평균) / classification: Accuracy, Balanced Accuracy,
Macro F1, Precision, Recall, AUROC, AUPRC / calibration: ECE, Brier, 잔차 분포·경험적 예측구간(`error_distribution`).

## 12. Checkpoint

```
checkpoints/<task>/<task>_v001/
  model.pt        weights, optimizer/scheduler/scaler state, epoch, metric, config,
                  label mapping, normalization, dataset metadata, timestamp, git commit
  config.yaml     metrics.json     metadata.json
outputs/<experiment>/
  train.log  metrics.json  history.csv  config.yaml  loss_curve.png  predictions.csv  (confusion_matrix.png)
```

학습할 때마다 새 버전 폴더가 생깁니다. `checkpoint.metric`(예: `val_loss`, `val_mean_mae`, `val_macro_f1`) 기준 best만 저장하며,
`save_best_only: false`면 epoch별 파일도 남깁니다. `history.csv`에는 epoch별 `gpu_max_allocated_mb / gpu_max_reserved_mb`가 기록됩니다.

```bash
python scripts/evaluate_all.py --split test     # task별 최신 checkpoint 재평가
```

## 13. Inference

FastAPI 등에서 바로 불러 쓸 수 있는 공통 인터페이스입니다. 반환값은 모두 JSON serializable `dict`입니다.

```python
from interview_gpu.speech.inference import SpeechAnalyzer
from interview_gpu.video.inference import VideoBehaviorModel
from interview_gpu.answer.inference import AnswerEvaluator

SpeechAnalyzer("checkpoints/speech/speech_v001").predict("answer.wav", transcript="...")
# {"metrics": {"speech_rate_wpm": 196.0, "long_pause_count": 2.0, ...},
#  "scores": {"pace_score": ..., "pause_score": ...}, "model_version": "speech_v001"}

VideoBehaviorModel("checkpoints/video/video_v001").predict("answer.mp4")   # 또는 landmark feature DataFrame
# {"metrics": {"gaze_center_ratio": 0.61, "head_yaw_std": ..., ...}, "scores": {...}, "model_version": "video_v001"}

AnswerEvaluator("checkpoints/answer/answer_v001").predict(question="...", answer="...", job="Backend Engineer")
# {"scores": {"relevance": 3.2, "structure": 2.6, ...}, "model_version": "answer_v001"}
```

`SpeechAnalyzer()` / `VideoBehaviorModel()`는 checkpoint 없이도 raw metric(근거 데이터)을 반환합니다.

## 14. Troubleshooting

| 증상 | 원인 / 해결 |
|---|---|
| `OSError: [WinError 4551] 애플리케이션 제어 정책에서 이 파일을 차단했습니다` (torch import) | Windows **Smart App Control / WDAC**가 서명되지 않은 DLL을 차단. 코드 문제가 아님. ① Docker/WSL2(Linux)에서 실행 ② 다른 학습용 머신 사용 ③ Windows 보안 → 앱 및 브라우저 컨트롤 → 스마트 앱 컨트롤 끄기(**되돌리려면 Windows 초기화 필요** — 신중히 결정) |
| `CUDA Available: False` | 드라이버 570 미만이거나 CPU용 torch 설치됨. `nvidia-smi` 확인 후 `uv sync` 재실행 |
| `Device 'cuda' was requested but CUDA is not available` | 명시적 `--device cuda`는 CPU로 조용히 넘어가지 않음. `check_gpu.py`로 진단 |
| `Dataset not found` | `data/raw/<폴더>` 배치 또는 `prepare_*.py` 미실행 |
| `No usable samples found` | 폴더 구조가 adapter 가정과 다름. `inspect_dataset.py`로 확인 후 adapter의 key path 수정 |
| `No target columns configured` / `Target column(s) ... not found` | `data.target_columns`를 메시지에 표시된 label 컬럼으로 지정 |
| `MediaPipe model file not found` | 8장의 `.task` 파일을 `models/mediapipe/`에 배치 |
| `Loss became nan` | 라벨/feature의 NaN·inf 확인, `lr` 낮추기, `mixed_precision=fp32` 시도 |
| `Checkpoint does not match the model architecture` | checkpoint 옆 `config.yaml`과 다른 설정으로 모델 생성 |
| Windows에서 DataLoader가 느리게 시작 | worker가 spawn 방식. `--set training.num_workers=0`으로 비교 |

## 15. OOM 대응

CUDA OOM 시 학습은 즉시 중단되고 안내 메시지를 출력합니다 (**CPU로 자동 전환하지 않음**).

1. `--set training.batch_size=<절반>`
2. `--set training.gradient_accumulation_steps=2` 로 유효 batch 유지
3. 시퀀스 축소: `data.params.max_length`(answer), `max_frames`(video), `max_seconds`(speech waveform)
4. answer: `model.params.freeze_encoder=true` / `trainable_layers=N` / `lora.enabled=true` / `gradient_checkpointing=true`, 더 작은 encoder
5. `training.mixed_precision`이 `fp32`로 고정돼 있지 않은지 확인

학습 시작 로그의 `estimated minimum training VRAM`은 weight+gradient+AdamW state만 계산한 **하한**이며 activation은 포함하지 않습니다.

## 16. Dataset license 주의사항

* 원본 데이터, 가공 데이터, checkpoint, output은 `.gitignore`로 제외됩니다. **재배포 금지 데이터가 repository나 Docker 이미지에 들어가지 않게 하세요.**
* 각 adapter에 license metadata가 있고 `manifest.json`으로 전파됩니다. `data.usage_scope: production` 학습은 `research` 데이터를 거부합니다.
* AI Hub 데이터는 승인 목적·이용약관이 데이터셋/신청자마다 다릅니다. 기본값은 보수적으로 `research`이며, 상업적 사용 가능 여부를 확인한 뒤에만 adapter의 `usage_scope`를 바꾸세요.
* MIT Interview, ChaLearn은 연구 전용 benchmark입니다. production 모델 학습에 쓰지 않습니다.
* 연구 데이터로 학습한 checkpoint의 서비스 배포 가능 여부도 원 라이선스의 영향을 받습니다.
* 서비스 자체 데이터는 `consent.opt_in == true`인 레코드만 읽습니다. Raw resume/audio/video를 동의 없이 학습에 포함하지 않습니다.

## Docker (GPU)

```bash
docker build -f Dockerfile.gpu -t interview-gpu-training .
docker run --rm --gpus all \
  -v "$PWD/data:/app/data" -v "$PWD/checkpoints:/app/checkpoints" \
  -v "$PWD/outputs:/app/outputs" -v "$PWD/models:/app/models" \
  interview-gpu-training python scripts/train_speech.py --config configs/speech.yaml
```

호스트에 NVIDIA 드라이버 + NVIDIA Container Toolkit이 필요합니다. 이미지(`nvidia/cuda:12.8.1`)와 PyTorch wheel(cu128)의 CUDA 버전을 맞춰 두었습니다.

## Tests

```bash
uv run pytest
```

CPU에서 대부분 실행되며 CUDA 전용 테스트는 GPU가 없으면 skip됩니다. PyTorch를 로드할 수 없는 환경에서는 해당 테스트가 사유와 함께 skip됩니다.
