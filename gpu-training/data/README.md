# data/

Nothing in this directory is committed except this file and `.gitkeep` placeholders.
Datasets are **never downloaded automatically** — download them yourself under each
dataset's license and place them as shown below.

```
data/
  raw/                              original downloads (read-only input)
    aihub-interview/                01 AI Hub 채용면접 인터뷰 데이터         (core)
    aihub-emotion-speech/           02 AI Hub 감정 분류를 위한 대화 음성
    aihub-multimodal-video/         03 AI Hub 멀티모달 영상
    aihub-robot-interaction/        04 AI Hub 안내 로봇 인터랙션 멀티모달
    aihub-korean-emotion-video/     05 AI Hub 한국인 감정인식을 위한 복합 영상
    mit-interview/                  06 MIT Interview Dataset
    chalearn-first-impressions/     07 ChaLearn First Impressions V2
    internal/                       08 service data (*.jsonl, opt-in only)
  interim/                          scratch space for manual conversions
  processed/                        written by scripts/prepare_*.py
    speech/   features.parquet, embeddings/*.npy, manifest.json
    video/    index.parquet, sequences/*.npy, manifest.json
    answer/   answers.jsonl, manifest.json
    multimodal/                     (future) aligned embeddings
```

| Dataset | Adapter name (`--dataset`) | Expected content | Usage scope |
|---|---|---|---|
| 01 AI Hub 채용면접 인터뷰 | `aihub_interview` | `**/*.json` labels + `**/*.wav`, matched by file name | research |
| 02 AI Hub 감정 분류 대화 음성 | `aihub_speech` | `**/*.wav` + optional `*.csv` (wav id, 발화문) | research |
| 03 AI Hub 멀티모달 영상 | `aihub_multimodal` | `**/*.mp4` (+ same-name `.json`) | research |
| 04 AI Hub 안내 로봇 인터랙션 | `aihub_robot_interaction` | `**/*.mp4` (+ same-name `.json`) | research |
| 05 AI Hub 한국인 감정인식 복합 영상 | `aihub_korean_face` | `**/*.mp4` (+ same-name `.json`) | research |
| 06 MIT Interview | `mit_interview` | `Audio/P*.wav`, `Videos/P*.avi`, `Labels/turker_scores_full_interview.csv` | research |
| 07 ChaLearn First Impressions V2 | `chalearn` | `**/*.mp4` | research |
| 08 Service data | `internal` | `*.jsonl` (see `src/interview_gpu/datasets/internal.py`) | production |

## Rules

* **Usage scope.** Every adapter carries license metadata (`dataset_name`, `source`,
  `license`, `commercial_use`, `research_only`, `redistribution_allowed`, `usage_scope`,
  `notes`). Preprocessing copies it into `manifest.json`, and training with
  `data.usage_scope: production` refuses any dataset marked `research`.
  The AI Hub entries default to `research` conservatively — confirm your approved purpose
  on AI Hub before changing an adapter to `production`.
* **Privacy.** Service records without `"consent": {"opt_in": true}` are skipped. Raw
  resumes are never read. Video preprocessing stores landmark-derived numbers only, never
  frames or face crops.
* **Verify the layout first.** The AI Hub adapters were written from the public dataset
  descriptions and have not been run against real downloads. After placing data, run

  ```
  python scripts/inspect_dataset.py --dataset aihub_interview
  ```

  and check that questions, answers, audio files and labels are actually found.
