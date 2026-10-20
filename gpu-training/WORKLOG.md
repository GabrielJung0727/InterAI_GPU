# 작업 기록 (2026-10-13)

AI Hub 데이터 배치, 어댑터 검증, parquet → CSV 전환까지의 작업 내용과 남은 일을 정리한 문서입니다.
변경 사항은 **아직 커밋하지 않았습니다.**

## 1. 현재 상태 요약

| 항목 | 상태 |
|---|---|
| GPU | RTX 5070 Ti 16GB, torch 2.11.0+cu128, CUDA 사용 가능 (`scripts/check_gpu.py` 통과, bf16 권장) |
| Smart App Control | 켜짐(끄지 않기로 결정). `pyarrow.parquet`, `torchaudio` DLL을 차단함 |
| 저장 형식 | parquet → CSV 전환 완료 |
| 테스트 | 91개 중 90개 통과, 1개 실패 (기존 문제, 5장 참고) |
| WSL / Docker | WSL 3.0.3 + Docker Desktop 29.8.2 설치 완료. `interview-gpu-training` 이미지 빌드, 컨테이너 안 `check_gpu.py` 통과 (8장 참고) |

## 2. 데이터 배치 현황 (`data/raw/`)

| # | 데이터셋 | 폴더 | 받은 것 | 상태 |
|---|---|---|---|---|
| 01 | 채용면접 인터뷰 | `aihub-interview/` | Training + Validation, 원천 + 라벨링 (zip 112개, 162GB) | **전체 압축 해제·검증 완료** (8장 참고) |
| 02 | 감정 분류 대화 음성 | `aihub-emotion-speech/` | zip 3개(WAV) + CSV 3개 (14GB) | **압축 해제·검증 완료** |
| 03 | 멀티모달 영상 | `aihub-multimodal-video/` | zip 14개 (89GB), `clip_N.mp4` + `clip_N.json` | **압축 해제·검증 완료** |
| 04 | 안내 로봇 인터랙션 | `aihub-robot-interaction/` | Validation 원천 zip 9개 (160GB) | zip 안에 또 zip(이중 압축). 안쪽 내용 미확인 |
| 05 | 한국인 감정인식 복합 영상 | `aihub-korean-emotion-video/` | Validation 원천 zip 6개 (79GB) | **영상이 아니라 JPG 이미지** → 현재 어댑터로는 샘플 0개. **2026-10-13 삭제함** |

* 압축 해제는 각 zip 옆에 같은 이름의 폴더로 풀었고, 원본 zip은 그대로 두었습니다.
* 코드는 AI Hub의 Training/Validation 구분을 쓰지 않고, 폴더 안 전체를 모아 직접 다시 split합니다.

### 01 Validation 검증 결과 (`scripts/inspect_dataset.py --dataset aihub_interview`)

| 항목 | 결과 |
|---|---|
| 샘플 수 | 8,026개 (라벨 JSON 8,028개 중) |
| 음성 매칭 | 8,026개 전부, 누락·손상 0개 |
| 평균 답변 길이 | 약 70초 |
| 답변 의도 라벨 | attitude 5,505 / background 996 / personality 770 / technology 423 / etc 332 |
| 화자 수 | 0 |

알아둘 점:

* **화자 ID가 라벨에 없습니다.** split이 샘플 단위로 떨어져 같은 사람이 train/val/test에 섞일 수 있습니다.
* **질문 의도 라벨은 비어 있습니다.** 의도 라벨은 답변 쪽만 잡힙니다.
* **rubric 점수가 없습니다.** 답변 모델 학습 전에 Stage A(외부 LLM 채점 → `--labels` 병합)가 필요합니다.
* **읽지 못한 라벨 2개**: `ckmk_d_bm_f_n_161564.json`, `ckmk_d_ard_f_e_66661.json`. 문자열 안에 제어 문자가 들어 있는 원본 결함이라 건너뛰게 두었습니다.

## 3. 코드 변경 내용

### 3.1 AI Hub 채용면접 어댑터

* `src/interview_gpu/datasets/aihub_interview.py`: `FIELD_PATHS["audio_file"]`에 `rawDataInfo.answer.audioPath` 추가.
  실제 라벨의 키가 `fileName`이 아니라 `audioPath`이고, 라벨 파일명(`ckmk_d_…`)과 음성 파일명(`ckmk_a_…`)이 달라 수정 전에는 음성이 전부 매칭되지 않았습니다.

### 3.2 parquet → CSV

Smart App Control이 `pyarrow.parquet`을 막아서 전처리·학습이 중간에 실패하던 문제를 해결했습니다.

| 이전 | 이후 |
|---|---|
| `data/processed/speech/features.parquet` | `features.csv` |
| `data/processed/video/index.parquet` | `index.csv` |
| `data/processed/multimodal/index.parquet` | `index.csv` |
| `<output>/predictions.parquet` | `predictions.csv` |

* `src/interview_gpu/common/io.py`에 `write_table` / `read_table` 추가. 모든 표 입출력이 이 두 함수를 거칩니다.
* `read_table` 규칙: `sample_id`·`group_id`·`true`·`pred`·`*_path` 컬럼은 문자열로 읽고, 빈 칸만 결측으로 처리하며, 실수는 오차 없이 복원합니다.
* 수정 파일: `speech/preprocess.py`, `speech/dataset.py`, `video/preprocess.py`, `video/dataset.py`,
  `video/temporal_features.py`, `multimodal/dataset.py`, `common/runner.py`
* 테스트: `needs_parquet` skip 장치 제거, 파일 이름을 CSV로 변경.
* 문서: `README.md`, `data/README.md`, `configs/speech.yaml` 주석.

### 3.3 이 PC에서 학습을 막던 다른 문제 (추가 수정)

* **체크포인트 로드 실패**: `common/checkpoint.py`에서 `pytorch_version`을 `TorchVersion` 객체로 저장해
  `weights_only=True` 로드가 거부됐습니다. `str(torch.__version__)`로 수정.
* **torchaudio 차단**: transformers가 설치된 torchaudio를 자동 import하다가 차단에 걸려 답변 모델·음성 인코더가 로드되지 않았습니다.
  프로젝트 코드는 torchaudio를 쓰지 않으므로 `pyproject.toml`에서 제거하고 가상환경에서도 삭제했습니다.

### 3.4 의존성

* `pyproject.toml`: `pyarrow`, `torchaudio` 제거 (pyarrow는 HuggingFace `datasets`의 간접 의존성으로 여전히 설치됨).
* `uv.lock` 갱신.
* `README.md`의 pip 설치 안내에서 `torchaudio` 제거.

## 4. 변경된 파일 목록

```
README.md
configs/speech.yaml
data/README.md
pyproject.toml
uv.lock
src/interview_gpu/common/checkpoint.py
src/interview_gpu/common/io.py
src/interview_gpu/common/runner.py
src/interview_gpu/datasets/aihub_interview.py
src/interview_gpu/multimodal/dataset.py
src/interview_gpu/speech/dataset.py
src/interview_gpu/speech/preprocess.py
src/interview_gpu/video/dataset.py
src/interview_gpu/video/preprocess.py
src/interview_gpu/video/temporal_features.py
tests/conftest.py
tests/test_adapters.py
tests/test_training.py
tests/test_video_features.py
```

## 5. 남은 문제

* **`tests/test_adapters.py::test_prepare_speech_dataset` 실패 (기존 문제)**
  음성 feature 추출이 무음 구간을 실제보다 약 0.13초(프레임 길이 2048 샘플) 짧게 잽니다.
  테스트 데이터의 0.35초 무음이 0.224초로 측정되어 `min_pause_sec` 0.25에 못 미치고 pause가 0개로 나옵니다.
  실제 데이터에서도 0.25~0.38초 정도의 짧은 pause가 집계에서 빠집니다.
  테스트만 고칠지, 측정 방식을 보정할지 결정이 필요합니다.
* **실제 데이터로 전처리·학습을 돌려 본 적은 아직 없습니다.** 지금까지의 확인은 테스트용 더미 데이터 기준입니다.
* speech·video 설정의 기본 타깃은 MIT Interview의 행동 평정입니다. AI Hub 02~05만으로는 지도학습이 안 됩니다.

## 6. 다음 단계

1. 01 Training(147GB) 압축 해제 → `inspect_dataset.py`로 재확인
2. 01 데이터로 실제 전처리 실행
   ```bash
   python scripts/prepare_answer.py --dataset aihub_interview --output data/processed/answer
   python scripts/prepare_speech.py --dataset aihub_interview --output data/processed/speech
   ```
3. 답변 모델 라벨 준비 (Stage A): `prepare_answer.py --export-prompts` → 외부 LLM 채점 → `--labels` 병합
4. 02, 03 압축 해제 및 `inspect_dataset.py` 검증 (02는 CSV 컬럼명 확인)
5. 05 삭제, 04는 필요해질 때 이중 압축 해제
6. 변경 사항 커밋

## 7. Docker로 옮길 때 참고

* 컨테이너 안은 Linux라서 Smart App Control 차단이 적용되지 않습니다. GPU는 `--gpus all`로 같은 RTX 5070 Ti를 씁니다.
* CSV 전환과 torchaudio 제거는 Docker에서도 그대로 동작하므로 되돌릴 필요가 없습니다.
* 데이터가 Windows 쪽 C 드라이브에 있으면 컨테이너에서 읽는 속도가 느립니다.
  작은 파일이 많은 01 데이터는 전처리 시간이 늘어날 수 있으니, 전처리 결과(`data/processed/`)는 Linux 쪽 저장소에 두는 편이 좋습니다.
* WSL 기본 설정은 메모리를 전체의 절반 정도만 씁니다. 부족하면 `%UserProfile%\.wslconfig`에서 늘립니다.
* 이미지는 `Dockerfile.gpu`로 빌드합니다. `pyproject.toml`/`uv.lock`이 바뀌었으므로 새로 빌드해야 합니다.
* 재부팅 후 현재 Windows 가상환경(`.venv`)은 그대로 쓸 수 있습니다. 압축 해제 스크립트는 임시 폴더에 있었으므로 다시 만들어야 할 수 있습니다.

## 8. 재부팅 이후 진행 (2026-10-13 오후)

### Docker

* Docker Desktop은 로그인 시 자동 실행이 꺼져 있어 직접 실행해야 합니다.
* `docker run --rm --gpus all ubuntu:24.04 nvidia-smi`로 컨테이너에서 RTX 5070 Ti 인식 확인.
* `docker build -f Dockerfile.gpu -t interview-gpu-training .` 성공 (디스크 26.8GB). 컨테이너 안 `check_gpu.py` 통과.
* 이미지는 `--no-dev`로 빌드되어 pytest가 없습니다. 컨테이너에서 테스트하려면 `uv pip install pytest` 후 실행합니다.
  `test_prepare_speech_dataset`는 컨테이너에서도 같은 이유(5장)로 실패합니다.
* Docker에 잡힌 자원: CPU 16개, 메모리 약 31GB (호스트 64GB의 절반, `.wslconfig` 없음).

### 압축 해제

01 Training, 02, 03의 zip 73개를 각 zip 옆 같은 이름 폴더에 풀었습니다 (실패 0개, 원본 zip 유지).
04와 05는 그대로 zip 상태입니다. 해제 후 C 드라이브 여유 공간은 약 824GB입니다.

### 검증 결과 (`scripts/inspect_dataset.py`, 리포트는 `outputs/inspection/`)

| 데이터셋 | 샘플 수 | 평균 길이 | 누락·손상 | 건너뛴 레코드 | 해제 후 폴더 크기(zip 포함) |
|---|---|---|---|---|---|
| 01 `aihub_interview` | 76,100 | 음성 80.3초 | 0 | 읽지 못한 JSON 6개 | 375GB |
| 02 `aihub_speech` | 43,991 | 음성 6.0초 | 0 | 전사 없음 16개 | 38GB |
| 03 `aihub_multimodal` | 5,600 | 영상 66.7초 | 0 | 없음 | 205GB |

* 01 답변 의도 분포: attitude 59,422 / background 7,895 / personality 4,005 / technology 3,100 / etc 1,678.
  attitude가 78%로 Validation만 볼 때(69%)보다 더 치우쳐 있습니다.
* 01·02·03 모두 화자 수가 0으로 잡힙니다. 화자 단위 split은 여전히 불가능합니다.
* 02, 03은 `label_distribution`이 비어 있습니다 (어댑터가 타깃 라벨을 읽지 않음).

### 전처리

* `prepare_answer.py --dataset aihub_interview` 완료: `data/processed/answer/answers.jsonl`에 76,100개, rubric 라벨은 0개 (Stage A 필요).
* `prepare_speech.py --dataset aihub_interview` 실행 중: 100개 시험 실행이 27초였고, 전체는 4~5시간으로 추정합니다.
  로그는 `outputs/logs/prepare_speech.*.log`, 결과는 `data/processed/speech/`.

## 9. pause 측정 보정, Stage A 로컬 채점 (2026-10-13 오후)

### pause 측정 보정 (5장의 실패 테스트 해결)

* `speech/features.py`: 무음 구간 검출만 25ms 창 / 10ms hop(`SILENCE_FRAME_LENGTH=400`, `SILENCE_HOP_LENGTH=160`)으로 바꿨습니다.
  기존 2048 샘플 창은 pause를 약 0.13초 짧게 쟀습니다. 테스트는 고치지 않았고 전체 통과합니다.
* 실제 01 음성 12개 비교: pause 수가 같거나 늘고(예: 12→23, 23→31), pause 총 길이는 2~6초 늘며, `speech_ratio`는 0.03~0.12 낮아집니다.
* **보정 전에 시작한 `prepare_speech.py` 실행(PID 4844)의 결과는 옛 측정값입니다.** 끝난 뒤 다시 실행해야 합니다.

### Stage A 로컬 채점

* 추가: `src/interview_gpu/answer/labeling.py`, `scripts/label_answers.py`, `configs/labeling.yaml`, `tests/test_labeling.py`.
  사용법은 README "Stage A 로컬 채점" 참고. 외부 API는 쓰지 않습니다.
* `pyproject.toml`에 `labeling` extra(bitsandbytes) 추가, `Dockerfile.gpu`가 이 extra를 설치합니다. `data/labeled/`는 gitignore.
* `prepare_answer.py --labels`는 `needs_review=true` 라벨을 병합에서 제외합니다.
* Windows 쪽 Claude 세션에서는 HuggingFace 다운로드가 막혀 있어 채점은 Docker에서 실행했습니다. 모델 캐시는 `.cache/huggingface/`.
  ```powershell
  docker run --rm --gpus all -v "${PWD}\data:/app/data" -v "${PWD}\.cache\huggingface:/app/.cache/huggingface" `
    interview-gpu-training python scripts/label_answers.py --dry-run --consistency-runs 2
  ```

### 200개 dry-run 결과 (Qwen3-8B, 4bit, 2회 채점, batch 17)

| 항목 | 값 |
|---|---|
| 채점 성공 | 200 / 200 (첫 응답 파싱 실패 1%, 재시도로 해결) |
| needs_review | 10% |
| 속도 | 0.46개/초 (2회 채점 기준) → 전체 76,100개는 약 46시간, 1회 채점이면 약 23시간 |
| 평균 점수 | relevance 6.7 / structure 4.8 / specificity 4.7 / evidence 4.3 / technical_depth 2.6 / ownership 3.7 / logical_consistency 6.4 |
| 항목 간 상관 | 0.70 ~ 0.94 |

* 검수 파일: `data/labeled/dry_run/qwen3_answers.review.csv`
* 첫 시도(`data/labeled/dry_run_v1/`)는 근거 인용이 길어 JSON이 잘린 실패가 6건 있었습니다.
  프롬프트에 "한 줄 출력, 인용 30자 이내"를 넣고 `max_new_tokens`를 320으로 올려 해결했습니다.
