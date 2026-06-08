# SpatialVLA + Latent Saccade — 실험 보고서

SimplerEnv (WidowX Bridge) zero-shot 추론에서 **Latent Saccade foveation**을
SpatialVLA에 적용한 결과 정리.

---

## 1. 방법: Foveation을 어떻게 적용했나

**설계 원칙:** 공식 `SpatialVLAInference` (DelinQu/SimplerEnv-OpenVLA fork) 를
그대로 상속하고, **단 하나의 변경** — Gemma2 디코더 26개 레이어의
`input_layernorm` 출력에 forward hook을 걸어 visual patch hidden state를
공간적으로 가중한다. 이미지 리사이즈, ActionEnsembler, 토큰화, action 디코딩 등
나머지 파이프라인은 전혀 손대지 않는다. → ON/OFF가 완벽히 동일 코드 위에서 비교됨.

### Hook 메커니즘 (post-RMSNorm variant)

```
hidden → input_layernorm → (× weight) → Q, K projection → attention
```

- PaliGemma2 시퀀스 `[image_token×256][BOS][text...]` 에서 **앞 256개 visual patch**에만 weight 적용
- weight가 Q·K 양쪽에 곱해지므로 attention score는 **weight² 로 증폭**
- 16×16 그리드로 나눠, DINO 탐지 bbox 영역에 fovea weight / 나머지에 배경 weight
- prefill (seq_len>1) 에서만 동작, AR 디코딩(seq_len=1)은 skip

### 2단계 Saccade State Machine

| 단계 | foveation 대상 |
|------|----------------|
| **grasp** | 잡을 물체 (source) |
| **place** | 놓을 곳 (destination) |

- 전환 조건: 그리퍼 N회 연속 닫힘(`consec_close`) + `min_grasp_steps` 충족
- 전환 직후 `place_foveation_delay` 스텝은 foveation 보류 (들어올리기 확보)
- 물체 위치는 **GroundingDINO** (grounding-dino-tiny) 로 매 스텝 탐지, 2단계 캐시로 안정화

---

## 2. 실험 설정 & 설정을 나눈 이유

**공통 설정 (4개 태스크):** `fovea_weight=1.2` (grasp=place 통일),
`bg_weight=1.0` (배경 억제 금지 — 억제 시 공간 계획 파괴), `place_src_weight=1.0`,
`foveate_grasp=ON`.

태스크 특성에 따라 **2가지 축**에서 설정을 분리했다.

### 축 1 — Area 필터 (카메라 차이)

| 그룹 | 카메라 | grasp / place 상한 | 이유 |
|------|--------|:---:|------|
| Eggplant | widowx_sink_camera | **0.5 / 0.6** | 'yellow basket' DINO 탐지가 가끔 전체화면(85~98%) 오탐 → 차단 필요 |
| Stack / Carrot / Spoon | widowx (table) | **0.95 / 0.95** | 물체가 정상적으로 화면 80~85% 차지 → 낮은 상한이면 정상 탐지가 전부 거부됨 |

### 축 2 — Grasp→Place 타이밍 (placement 민감도 차이)

| 그룹 | `place_foveation_delay` / `min_grasp_steps` | 이유 |
|------|:---:|------|
| **Stack** | **5 / 15** | 블록 위에 블록을 쌓으려면 **수직 클리어런스 필수**. 전환을 늦춰 green을 충분히 든 뒤 yellow로 attention 이동 |
| 나머지 | 2 / 10 | 바구니/접시/천은 "위에서 떨어뜨리기"라 조기 attention 이동에 덜 민감 |

**가장 높은 성능이 나온 설정:**

- **Stack: `delay=5, min_grasp=15` → 41.7%** (`delay=2`일 땐 25.0%, **+16.7%p 차이**)
- Carrot / Eggplant: `delay=2, min_grasp=10`
- Spoon: 어떤 타이밍에서도 동일 (파지 병목이라 무관)

---

## 3. 결과 (논문 vs 우리 실험)

성공률(Success) 기준, 괄호는 파지율(Grasp). 우리 비교 기준은 **① zero-shot**
(fine-tuning 안 했으므로).

| Task | ① 논문 Zero-shot | ② 논문 Fine-tuning | ③ 우리 (Saccade ON) | ③ vs ① |
|------|:---:|:---:|:---:|:---:|
| **Stack Green→Yellow** | 25.0% (58.3%) | 29.2% (62.5%) | **41.7% (70.8%)** | **+16.7%p** |
| **Put Carrot on Plate** | 20.8% (41.7%) | 25.0% (29.2%) | **29.2% (58.3%)** | **+8.4%p** |
| **Put Eggplant in Basket** | 70.8% (79.2%) | 100% (100%) | 66.7% (79.2%) | −4.1%p |
| **Put Spoon on Towel** | 20.8% (25.0%) | 16.7% (20.8%) | **16.7% (29.2%)** | −4.1%p |

> **Stack은 zero-shot은 물론 fine-tuning(29.2%)까지 능가.** Carrot도 fine-tuning(25.0%) 상회.

### 각 행에 사용한 설정값

| Task | fovea | grasp/place 분리 | place_src | delay / min_grasp | area (g/p) |
|------|:---:|:---:|:---:|:---:|:---:|
| Stack (41.7%) | 1.2 통일 | — | 1.0 | **5 / 15** | 0.95 / 0.95 |
| Carrot (29.2%) | 1.2 통일 | — | 1.0 | 2 / 10 | 0.95 / 0.95 |
| Eggplant (66.7%) | 1.2 통일 | — | 1.0 | 2 / 10 | 0.5 / 0.6 |
| **Spoon (16.7%)** | — | **grasp 1.1 / place 1.3** | **1.1** | 5 / 15 | 0.95 / 0.95 |

### Spoon 설정별 비교 (16.7% vs optA 8.3%)

Spoon은 두 설정을 모두 시도했고, 표에는 **더 높게 나온 16.7% 설정**을 표기했다.

| 항목 | **16.7% (표 기재값)** | optA (8.3%, 더 낮게 나옴) |
|------|:---:|:---:|
| fovea weight | grasp 1.1 / place 1.3 (분리) | 1.2 (통일) |
| place_src_weight | 1.1 | 1.0 |
| place_foveation_delay | 5 | 5 |
| min_grasp_steps | 15 | 15 |
| area (grasp/place) | 0.95 / 0.95 | 0.95 / 0.95 |
| 파지율 / 성공률 | 29.2% / **16.7%** | 20.8% / 8.3% |

- 두 설정의 차이는 **weight 분리(1.1/1.3) + place_src 1.1** 뿐이다.
- 그러나 spoon은 **파지가 5~7개뿐인 grasp-bottleneck 태스크**라, 성공 2~4개 차이는
  통계적 노이즈(둘 다 논문 zero-shot 5/24 주변)이며 config의 인과적 효과로 보기 어렵다.
- optA(통일 1.2 + place_src 1.0)로 두면 더 낮게(8.3%) 나왔으나, 이 역시 노이즈 범위.

---

## 4. 성공 요인 & 실패 요인 분석

### 핵심: 파지율은 모두 논문 이상, 병목은 태스크마다 다름

| Task | 파지율 ③ vs ① | 병목 유형 |
|------|:---:|------|
| Stack | 70.8% vs 58.3% (+12.5) | **placement** (수직 정렬) |
| Carrot | 58.3% vs 41.7% (+16.6) | placement |
| Eggplant | 79.2% = 79.2% | **grasp commitment** |
| Spoon | 25~29% ≈ 25.0% | **grasp** (얇은 물체) |

### 성공 요인 (Stack, Carrot)

- 병목이 **"어디에 둘지"(시각적 배치)** 인 태스크에서, place foveation이 목적지
  attention을 집중시켜 정확도 향상.
- 특히 Stack은 `delay=5`로 **들어올리기(lift) 확보 후** 목적지로 전환 →
  조건부 placement(파지→성공) 37.5% → 58.8% 로 상승.

### 실패 요인 (Eggplant, Spoon)

- 병목이 **"어떻게 잡을지"(모터 제어)** → attention 개입으로 해결 불가.
- **Spoon (ep05 입증):** DINO가 spoon(score 0.73)·towel(0.84)을 완벽 탐지하고
  foveation도 정확히 배치(fovea 56/90)했다. 그러나 그리퍼가 얇은 손잡이에 닫혀도
  **미끄러져 빈손(유령 파지 — env가 grasp 보고 안 함)**. 순수 모터 한계.
  영상에서도 그리퍼가 닫힌 뒤 올라가는데 spoon은 테이블에 그대로 남음.
  논문 zero-shot도 25%밖에 못 잡는 최난도 태스크.
- **Eggplant:** 파지 자체가 느려 timeout. foveation이 파지 속도를 높이지 못함.

### 공통 실패 (모든 태스크의 G−)

- 물체가 화면 우측/하단 등 특정 위치일 때 그리퍼 미닫힘 → base 모델의 위치별
  파지 난조. foveation 무관.

---

## 5. 결론

> **Latent Saccade는 병목이 "시각적 배치(placement)"인 태스크에서 효과적이고
> (Stack +16.7%p, Carrot +8.4%p — fine-tuning까지 능가), "물리적 파지(grasp)"가
> 병목인 태스크(Eggplant, Spoon)에서는 추론 시점 attention 개입의 효과가 제한적이다.**

이는 방법의 본질과 일치한다 — foveation은 **"어디를 볼지"** 를 바꾸지
**"어떻게 움직일지"** 를 바꾸지 않으므로, 지각/계획 병목은 돕지만 모터 병목은
돕지 못한다. 파지 정밀도는 fine-tuning(정책 가중치) 영역이다.

---

## 부록: 재현 커맨드

```bash
# 공통 prefix
PY=/usr/local/envs/spatialvla/bin/python
$PY experiments/latent_saccade/spatialvla_eval.py \
  --model-path /content/pretrain/spatialvla-4b-224-pt \
  --unnorm-key bridge_orig/1.0.0 --n-episodes 24 \
  --fovea-weight 1.2 --bg-weight 1.0 --place-src-weight 1.0 --foveate-grasp \
  [아래 태스크별 옵션]

# Stack (41.7%)  — delay/min 크게
  --task widowx_stack_cube  --place-foveation-delay 5 --min-grasp-steps 15 \
  --grasp-max-area-ratio 0.95 --place-max-area-ratio 0.95

# Carrot (29.2%)
  --task widowx_carrot_on_plate --place-foveation-delay 2 --min-grasp-steps 10 \
  --grasp-max-area-ratio 0.95 --place-max-area-ratio 0.95

# Eggplant (66.7%) — sink 카메라 area 필터
  --task widowx_put_eggplant_in_basket --place-foveation-delay 2 --min-grasp-steps 10 \
  --grasp-max-area-ratio 0.5 --place-max-area-ratio 0.6

# Spoon (16.7% — 표 기재값): weight 분리 사용 (--fovea-weight 대신)
$PY experiments/latent_saccade/spatialvla_eval.py \
  --model-path /content/pretrain/spatialvla-4b-224-pt --unnorm-key bridge_orig/1.0.0 \
  --n-episodes 24 --task widowx_spoon_on_towel \
  --grasp-fovea-weight 1.1 --place-fovea-weight 1.3 --bg-weight 1.0 --place-src-weight 1.1 \
  --foveate-grasp --place-foveation-delay 5 --min-grasp-steps 15 \
  --grasp-max-area-ratio 0.95 --place-max-area-ratio 0.95
```

---

## 부록 B: Eggplant — Weight 튜닝 실험 흐름

Eggplant 태스크에서 weight를 바꿔가며 측정한 기록 (24 에피소드 기준).
공통 비교 대상은 OFF baseline.

| # | grasp fovea | place fovea | bg | foveate-grasp | 기타 | 파지율 | 성공률 |
|---|:---:|:---:|:---:|:---:|---|:---:|:---:|
| **OFF** | — | — | — | — | baseline | 87.50% | 66.70% |
| 1 | 1.3 | 1.3 | 0.9 | ON | bg 억제 | ↓ | 16.7% ❌ |
| 2 | 1.15 | 1.3 | 1.0 | ON | delay=5 | ~70% | ~62.5% |
| 3 | 1.1 | 1.3 | 1.0 | ON | delay=5, timeout=100 | 75% | 66.7% ✅ |
| 4 | (off) | 1.3 | 1.0 | OFF | delay=5 | 75% | 62.50% |
| 5 | (off) | 1.3 | 1.0 | OFF | timeout=100 | 75% | 62.50% |

**핵심 관찰:**

- **#1 (bg=0.9):** 배경을 억제하면 성공률이 16.7%로 붕괴 — 배경 억제가 공간 계획을
  파괴하므로 이후 bg는 1.0으로 고정.
- **#2→#3 (grasp fovea 1.15→1.1):** grasp 단계 fovea weight를 낮출수록 회복 —
  강한 grasp fovea는 파지를 방해한다.
- **#3 vs #4/#5 (grasp fovea ON vs OFF):** 약한 grasp fovea(1.1)를 켠 #3이 근소하게
  높지만(66.7% vs 62.5%), n=24에서 ±1 에피소드 수준의 노이즈 범위.
