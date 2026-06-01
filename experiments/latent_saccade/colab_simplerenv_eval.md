# SpatialVLA + Latent Saccade — SimplerEnv Colab 평가 노트북

UniVLA용 Colab 셋업을 SpatialVLA에 맞게 포팅한 버전.

## 설계 원칙

공정한 대조 실험을 위해 `LatentSaccadeSpatialVLAInference` 는 공식
`SpatialVLAInference` (DelinQu/SimplerEnv-OpenVLA fork) 를 **상속** 합니다.
ActionEnsembler, image history, do_normalize=False, cv2 resize, raw prompt 등
공식 파이프라인은 전혀 변경되지 않고, hook 만 추가됩니다.
`--no-latent-mask` 플래그 하나로 ON / OFF 를 동일 코드에서 실험합니다.

## UniVLA 대비 핵심 차이

| 항목 | UniVLA | SpatialVLA |
|---|---|---|
| 백본 | Emu3 + VQ-VAE | PaliGemma2 (SigLiP + Gemma2) |
| 모델 로드 | 커스텀 emu3 코드 | `AutoModel(trust_remote_code=True)` |
| transformers | 4.44.0 | **4.47.0** (PaliGemma2 필수) |
| VQ-VAE 다운로드 | 필요 (Emu3-VisionTokenizer) | **불필요** |
| fast tokenizer | 필요 | **불필요** |
| tiktoken | 필요 | **불필요** |
| 모델 가중치 | Yuqi1997/UniVLA (~14GB) | IPEC-COMMUNITY/spatialvla-4b-224-pt (~8.5GB) |
| SimplerEnv | simpler-env/SimplerEnv | **DelinQu/SimplerEnv-OpenVLA** (SpatialVLA 지원 fork) |
| conda env | univla | spatialvla |

> 각 코드 블록은 Colab 셀 하나에 대응합니다. `%%bash`로 표시된 블록은 bash 셀, 그 외는 Python 셀입니다.

---

## 1. Miniconda 설치

```bash
%%bash
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -f -p /usr/local
conda --version
```

## 2. conda TOS 동의

```bash
%%bash
source /usr/local/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 3. conda 환경 생성 (spatialvla, python 3.10)

```bash
%%bash
source /usr/local/etc/profile.d/conda.sh
conda create -n spatialvla python=3.10.12 -y
conda run -n spatialvla python --version
```

## 4. 시스템 렌더링 의존성 (EGL / Vulkan / ffmpeg)

```bash
%%bash
apt-get update -yqq
apt-get -yqq install libegl1-mesa libegl1 libgl1 libosmesa6-dev
apt-get install -yqq --no-install-recommends libvulkan-dev vulkan-tools
apt-get install -yqq ffmpeg
```

## 5. 컴파일러 + 기본 파이썬 패키지

```bash
%%bash
source /usr/local/etc/profile.d/conda.sh
conda run -n spatialvla conda install -c conda-forge gcc=12.1.0 gxx_linux-64 -y
conda run -n spatialvla pip install mediapy
# SpatialVLA는 PaliGemma2 백본 → transformers 4.47.0 필수
conda run -n spatialvla pip install transformers==4.47.0 tokenizers==0.21.0 pillow
conda run -n spatialvla pip install matplotlib
```

## 6. SpatialVLA + SimplerEnv 클론 및 시뮬레이터 의존성

```bash
%%bash
source /usr/local/etc/profile.d/conda.sh

# SpatialVLA — Latent Saccade 실험 코드가 있는 브랜치
if [ ! -d /content/SpatialVLA ]; then
  git clone --depth 1 -b claude/happy-hypatia-y9BTO \
    https://github.com/trillion-boy/spatialvla.git /content/SpatialVLA
fi

# SimplerEnv (SpatialVLA/OpenVLA 지원 fork, allenzren/ManiSkill2_real2sim 서브모듈 포함)
if [ ! -d /content/SimplerEnv ]; then
  git clone https://github.com/DelinQu/SimplerEnv-OpenVLA \
    --recurse-submodules -q /content/SimplerEnv
fi

# SAPIEN 2.2.2
conda run -n spatialvla pip install sapien==2.2.2

# ManiSkill2_real2sim
conda run -n spatialvla pip install --no-deps -e /content/SimplerEnv/ManiSkill2_real2sim

# gym
conda run -n spatialvla conda install -c conda-forge gym=0.21.0 -y

# ruckig
conda run -n spatialvla pip install ruckig

# opencv + 기타 시뮬 의존성
conda run -n spatialvla pip install \
  transforms3d "opencv-python-headless==4.8.1.78" \
  "trimesh==3.22.5" "open3d==0.17.0" "mplib==0.0.9" "gymnasium==0.29.1"

# mani-skill2 누락 deps
conda run -n spatialvla pip install \
  gdown GitPython h5py \
  imageio "imageio[ffmpeg]" \
  rtree tabulate

# numpy 고정 (SAPIEN/ManiSkill 호환을 위해 1.24.4)
conda run -n spatialvla pip install "numpy==1.24.4"

# SimplerEnv
conda run -n spatialvla pip install --no-deps -e /content/SimplerEnv
```

## 7. opencv / numpy 버전 재고정

```bash
%%bash
source /usr/local/etc/profile.d/conda.sh
conda run -n spatialvla pip install "opencv-python==4.8.1.78"
conda run -n spatialvla pip install "numpy==1.24.4"
```

## 8. xvfb (헤드리스 디스플레이)

```bash
%%bash
apt-get install -yqq xvfb
echo "xvfb done"
```

## 9. Vulkan ICD 설정

```bash
%%bash
mkdir -p /etc/vulkan/icd.d
cat > /etc/vulkan/icd.d/nvidia_icd.json << 'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.2.155"
    }
}
EOF
mkdir -p /usr/share/vulkan/implicit_layer.d
echo "Vulkan config done"
```

## 10. SAPIEN renderer_config 확인 (필요 시 패치)

```python
# Python 셀
import os, sys

rc_path = None
search_dirs = ["/usr/local/envs/spatialvla/lib/python3.10/site-packages"]
for sp in search_dirs:
    p = os.path.join(sp, 'sapien', 'core', 'renderer_config.py')
    if os.path.exists(p):
        rc_path = p
        break

if rc_path is None:
    print("WARN: renderer_config.py 못 찾음 (SAPIEN 버전에 따라 없을 수 있음 — 건너뜀)")
else:
    print(f"패치 대상: {rc_path}")
    with open(rc_path, 'r') as f:
        content = f.read()
    print("=== 원본 (앞 300자) ===")
    print(content[:300])
```

## 11. pkg_resources shim (setuptools 충돌 회피)

```python
# Python 셀 — sapien/maniskill이 구버전 pkg_resources를 찾을 때 대비
import os

site_packages = "/usr/local/envs/spatialvla/lib/python3.10/site-packages"
pr_dir = os.path.join(site_packages, "pkg_resources")

# 이미 정상 pkg_resources가 있으면 건너뜀
if os.path.exists(os.path.join(pr_dir, "__init__.py")):
    try:
        import importlib, subprocess
        out = subprocess.run(
            ["/usr/local/envs/spatialvla/bin/python", "-c", "import pkg_resources; print('ok')"],
            capture_output=True, text=True
        )
        if "ok" in out.stdout:
            print("pkg_resources 정상 작동 — shim 불필요")
            raise SystemExit
    except SystemExit:
        pass

os.makedirs(pr_dir, exist_ok=True)
shim = '''"""pkg_resources shim — importlib.metadata 기반"""
import importlib
import importlib.metadata as _meta
import os

def get_distribution(name):
    class _Dist:
        try:
            version = _meta.version(name)
        except Exception:
            version = "0.0.0"
    return _Dist()

def require(requirements):
    pass

def resource_filename(package_name, resource_name):
    try:
        mod = importlib.import_module(package_name)
        return os.path.join(os.path.dirname(mod.__file__), resource_name)
    except Exception:
        return resource_name

def resource_string(package_name, resource_name):
    path = resource_filename(package_name, resource_name)
    with open(path, "rb") as f:
        return f.read()

def resource_exists(package_name, resource_name):
    path = resource_filename(package_name, resource_name)
    return os.path.exists(path)

def resource_stream(package_name, resource_name):
    path = resource_filename(package_name, resource_name)
    return open(path, "rb")

class WorkingSet:
    def __iter__(self): return iter([])
    def __contains__(self, item): return False
    def require(self, *a, **kw): pass

working_set = WorkingSet()
'''
with open(os.path.join(pr_dir, "__init__.py"), "w") as f:
    f.write(shim)
print(f"pkg_resources shim 생성: {pr_dir}/__init__.py")
```

## 12. 시뮬레이터 import 검증

```bash
%%bash
source /usr/local/etc/profile.d/conda.sh
conda run -n spatialvla python -c "import pkg_resources; print('pkg_resources OK')"
conda run -n spatialvla python -c "import sapien.core; print('sapien OK')"
conda run -n spatialvla python -c "import simpler_env; print('simpler_env OK')"
```

## 13. GroundingDINO 의존성 (Latent Saccade의 bbox 탐지)

```bash
%%bash
source /usr/local/etc/profile.d/conda.sh
# transformers 4.47.0은 GroundingDINO를 이미 지원 — 버전 변경 금지
conda run -n spatialvla pip install torchvision --quiet
conda run -n spatialvla python -c "
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision
print('transformers DINO 지원: OK')
print('torchvision:', torchvision.__version__)
"
```

## 14. SpatialVLA 모델 다운로드 (HF hub, ~8.5GB)

```bash
%%bash
source /usr/local/etc/profile.d/conda.sh
# modeling_*.py / configuration_*.py / processing_*.py 가 포함되어
# trust_remote_code=True 로 자동 로드됨. VQ-VAE / fast tokenizer 불필요.
conda run -n spatialvla --no-capture-output python -c "
from huggingface_hub import snapshot_download
print('Downloading SpatialVLA spatialvla-4b-224-pt ...')
snapshot_download(
    repo_id='IPEC-COMMUNITY/spatialvla-4b-224-pt',
    local_dir='/content/pretrain/spatialvla-4b-224-pt',
    ignore_patterns=['*.git*', '*.bin'],   # safetensors 우선
)
print('Done: SpatialVLA weights')
"
```

## 15. 경로 검증

```python
# Python 셀
import os

model_path = "/content/pretrain/spatialvla-4b-224-pt"

print(f"{'OK' if os.path.isdir(model_path) else 'MISSING'} SpatialVLA model: {model_path}")
if os.path.isdir(model_path):
    files = sorted(os.listdir(model_path))
    print("   files:", files[:20])
    for need in ["config.json", "modeling_spatialvla.py", "processing_spatialvla.py"]:
        mark = "OK" if need in files else "MISSING"
        print(f"   {mark}: {need}")
```

## 16. torch CUDA 12.1 재설치 (Colab GPU 매칭)

```bash
%%bash
source /usr/local/etc/profile.d/conda.sh
# SpatialVLA requirements: torch 2.5.1 + cu121
conda run -n spatialvla pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121 -q
conda run -n spatialvla python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

## 17. 평가 실행 (Latent Saccade ON)

셀 1 — 스크립트 작성:
```python
%%writefile /tmp/run_saccade_on.sh
#!/bin/bash
set -e
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SIMPLER_ENV_ROOT=/content/SimplerEnv
export PYTHONPATH=/content/SpatialVLA:$PYTHONPATH

# headless display
Xvfb :99 -screen 0 1280x1024x24 &
XVFB_PID=$!
sleep 2
export DISPLAY=:99

cd /content/SpatialVLA
/usr/local/envs/spatialvla/bin/python \
  experiments/latent_saccade/spatialvla_eval.py \
    --model-path /content/pretrain/spatialvla-4b-224-pt \
    --unnorm-key bridge_orig/1.0.0 \
    --task widowx_put_eggplant_in_basket \
    --n-episodes 24 \
    --output-dir /content/saccade_on_results \
    --grasp-fovea-weight 1.1 --place-fovea-weight 1.3 --bg-weight 1.0 --place-src-weight 1.1 \
    --foveate-grasp --place-foveation-delay 5 --max-grasp-steps 100 \
    --save-video

kill $XVFB_PID 2>/dev/null || true
```

셀 2 — 실행:
```python
!bash /tmp/run_saccade_on.sh
```

## 18. 베이스라인 실행 (Latent Saccade OFF, 대조군)

셀 1 — 스크립트 작성:
```python
%%writefile /tmp/run_saccade_off.sh
#!/bin/bash
set -e
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SIMPLER_ENV_ROOT=/content/SimplerEnv
export PYTHONPATH=/content/SpatialVLA:$PYTHONPATH

# headless display
Xvfb :99 -screen 0 1280x1024x24 &
XVFB_PID=$!
sleep 2
export DISPLAY=:99

cd /content/SpatialVLA
/usr/local/envs/spatialvla/bin/python \
  experiments/latent_saccade/spatialvla_eval.py \
    --model-path /content/pretrain/spatialvla-4b-224-pt \
    --unnorm-key bridge_orig/1.0.0 \
    --task widowx_put_eggplant_in_basket \
    --n-episodes 24 \
    --output-dir /content/saccade_off_results \
    --no-latent-mask \
    --save-video

kill $XVFB_PID 2>/dev/null || true
```

셀 2 — 실행:
```python
!bash /tmp/run_saccade_off.sh
```

## 19. 결과 비교 (eggplant → basket)

```python
# Python 셀
import json, os

def load(p):
    fp = os.path.join(p, "results_widowx_put_eggplant_in_basket.json")
    if not os.path.exists(fp):
        print(f"결과 없음: {fp}")
        return None
    with open(fp) as f:
        return json.load(f)

on  = load("/content/saccade_on_results")
off = load("/content/saccade_off_results")

print(f"{'='*50}")
print(f"{'설정':<24}{'파지율':>12}{'성공률':>12}")
print(f"{'-'*50}")
for name, r in [("Latent Saccade ON", on), ("Baseline (OFF)", off)]:
    if r:
        print(f"{name:<24}{r['grasp_rate']:>11.1%}{r['success_rate']:>12.1%}")
print(f"{'='*50}")
if on and off:
    d = on['success_rate'] - off['success_rate']
    print(f"성공률 차이 (ON - OFF): {d:+.1%}")
```

---

## 20. PutSpoonOnTableCloth — Baseline (OFF)

Instruction 파싱: `src='spoon'  dst='table cloth'`

셀 1 — 스크립트 작성:
```python
%%writefile /tmp/run_spoon_off.sh
#!/bin/bash
set -e
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SIMPLER_ENV_ROOT=/content/SimplerEnv
export PYTHONPATH=/content/SpatialVLA:$PYTHONPATH

Xvfb :99 -screen 0 1280x1024x24 &
XVFB_PID=$!
sleep 2
export DISPLAY=:99

cd /content/SpatialVLA
/usr/local/envs/spatialvla/bin/python \
  experiments/latent_saccade/spatialvla_eval.py \
    --model-path /content/pretrain/spatialvla-4b-224-pt \
    --unnorm-key bridge_orig/1.0.0 \
    --task widowx_spoon_on_towel \
    --n-episodes 24 \
    --output-dir /content/results/spoon_off \
    --no-latent-mask \
    --save-video

kill $XVFB_PID 2>/dev/null || true
```

셀 2 — 실행:
```python
!bash /tmp/run_spoon_off.sh
```

## 21. PutSpoonOnTableCloth — Latent Saccade (ON)

셀 1 — 스크립트 작성:
```python
%%writefile /tmp/run_spoon_on.sh
#!/bin/bash
set -e
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SIMPLER_ENV_ROOT=/content/SimplerEnv
export PYTHONPATH=/content/SpatialVLA:$PYTHONPATH

Xvfb :99 -screen 0 1280x1024x24 &
XVFB_PID=$!
sleep 2
export DISPLAY=:99

cd /content/SpatialVLA
/usr/local/envs/spatialvla/bin/python \
  experiments/latent_saccade/spatialvla_eval.py \
    --model-path /content/pretrain/spatialvla-4b-224-pt \
    --unnorm-key bridge_orig/1.0.0 \
    --task widowx_spoon_on_towel \
    --n-episodes 24 \
    --output-dir /content/results/spoon_on \
    --grasp-fovea-weight 1.1 --place-fovea-weight 1.3 --bg-weight 1.0 --place-src-weight 1.1 \
    --foveate-grasp --place-foveation-delay 2 \
    --grasp-max-area-ratio 0.95 --place-max-area-ratio 0.95 \
    --save-video

kill $XVFB_PID 2>/dev/null || true
```

셀 2 — 실행:
```python
!bash /tmp/run_spoon_on.sh
```

---

## 22. PutCarrotOnPlate — Baseline (OFF)

Instruction 파싱: `src='carrot'  dst='plate'`

셀 1 — 스크립트 작성:
```python
%%writefile /tmp/run_carrot_off.sh
#!/bin/bash
set -e
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SIMPLER_ENV_ROOT=/content/SimplerEnv
export PYTHONPATH=/content/SpatialVLA:$PYTHONPATH

Xvfb :99 -screen 0 1280x1024x24 &
XVFB_PID=$!
sleep 2
export DISPLAY=:99

cd /content/SpatialVLA
/usr/local/envs/spatialvla/bin/python \
  experiments/latent_saccade/spatialvla_eval.py \
    --model-path /content/pretrain/spatialvla-4b-224-pt \
    --unnorm-key bridge_orig/1.0.0 \
    --task widowx_carrot_on_plate \
    --n-episodes 24 \
    --output-dir /content/results/carrot_off \
    --no-latent-mask \
    --save-video

kill $XVFB_PID 2>/dev/null || true
```

셀 2 — 실행:
```python
!bash /tmp/run_carrot_off.sh
```

## 23. PutCarrotOnPlate — Latent Saccade (ON)

셀 1 — 스크립트 작성:
```python
%%writefile /tmp/run_carrot_on.sh
#!/bin/bash
set -e
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SIMPLER_ENV_ROOT=/content/SimplerEnv
export PYTHONPATH=/content/SpatialVLA:$PYTHONPATH

Xvfb :99 -screen 0 1280x1024x24 &
XVFB_PID=$!
sleep 2
export DISPLAY=:99

cd /content/SpatialVLA
/usr/local/envs/spatialvla/bin/python \
  experiments/latent_saccade/spatialvla_eval.py \
    --model-path /content/pretrain/spatialvla-4b-224-pt \
    --unnorm-key bridge_orig/1.0.0 \
    --task widowx_carrot_on_plate \
    --n-episodes 24 \
    --output-dir /content/results/carrot_on \
    --grasp-fovea-weight 1.1 --place-fovea-weight 1.3 --bg-weight 1.0 --place-src-weight 1.1 \
    --foveate-grasp --place-foveation-delay 2 \
    --grasp-max-area-ratio 0.95 --place-max-area-ratio 0.95 \
    --save-video

kill $XVFB_PID 2>/dev/null || true
```

셀 2 — 실행:
```python
!bash /tmp/run_carrot_on.sh
```

---

## 24. StackGreenCubeOnYellowCube — Baseline (OFF)

Instruction 파싱: `src='green cube'  dst='yellow cube'`

셀 1 — 스크립트 작성:
```python
%%writefile /tmp/run_stack_off.sh
#!/bin/bash
set -e
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SIMPLER_ENV_ROOT=/content/SimplerEnv
export PYTHONPATH=/content/SpatialVLA:$PYTHONPATH

Xvfb :99 -screen 0 1280x1024x24 &
XVFB_PID=$!
sleep 2
export DISPLAY=:99

cd /content/SpatialVLA
/usr/local/envs/spatialvla/bin/python \
  experiments/latent_saccade/spatialvla_eval.py \
    --model-path /content/pretrain/spatialvla-4b-224-pt \
    --unnorm-key bridge_orig/1.0.0 \
    --task widowx_stack_cube \
    --n-episodes 24 \
    --output-dir /content/results/stack_off \
    --no-latent-mask \
    --save-video

kill $XVFB_PID 2>/dev/null || true
```

셀 2 — 실행:
```python
!bash /tmp/run_stack_off.sh
```

## 25. StackGreenCubeOnYellowCube — Latent Saccade (ON)

셀 1 — 스크립트 작성:
```python
%%writefile /tmp/run_stack_on.sh
#!/bin/bash
set -e
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SIMPLER_ENV_ROOT=/content/SimplerEnv
export PYTHONPATH=/content/SpatialVLA:$PYTHONPATH

Xvfb :99 -screen 0 1280x1024x24 &
XVFB_PID=$!
sleep 2
export DISPLAY=:99

cd /content/SpatialVLA
/usr/local/envs/spatialvla/bin/python \
  experiments/latent_saccade/spatialvla_eval.py \
    --model-path /content/pretrain/spatialvla-4b-224-pt \
    --unnorm-key bridge_orig/1.0.0 \
    --task widowx_stack_cube \
    --n-episodes 24 \
    --output-dir /content/results/stack_on \
    --grasp-fovea-weight 1.1 --place-fovea-weight 1.3 --bg-weight 1.0 --place-src-weight 1.1 \
    --foveate-grasp --place-foveation-delay 2 \
    --grasp-max-area-ratio 0.95 --place-max-area-ratio 0.95 \
    --save-video

kill $XVFB_PID 2>/dev/null || true
```

셀 2 — 실행:
```python
!bash /tmp/run_stack_on.sh
```

---

## 26. 전체 결과 비교 (4개 task)

```python
# Python 셀
import json, os

RESULTS = [
    ("PutEggplant→Basket",  "widowx_put_eggplant_in_basket",
     "/content/saccade_off_results", "/content/saccade_on_results"),
    ("PutSpoon→Towel",       "widowx_spoon_on_towel",
     "/content/results/spoon_off",   "/content/results/spoon_on"),
    ("PutCarrot→Plate",      "widowx_carrot_on_plate",
     "/content/results/carrot_off",  "/content/results/carrot_on"),
    ("StackGreen→Yellow",    "widowx_stack_cube",
     "/content/results/stack_off",   "/content/results/stack_on"),
]

def load_result(dir_path, task_key):
    fp = os.path.join(dir_path, f"results_{task_key}.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)

print(f"{'Task':<24} {'OFF 파지':>8} {'OFF 성공':>8}  {'ON 파지':>8} {'ON 성공':>8}  {'Δ성공':>7}")
print("─" * 72)
for label, task_key, off_dir, on_dir in RESULTS:
    off = load_result(off_dir, task_key)
    on  = load_result(on_dir,  task_key)
    off_g = f"{off['grasp_rate']:.1%}"  if off else "N/A"
    off_s = f"{off['success_rate']:.1%}" if off else "N/A"
    on_g  = f"{on['grasp_rate']:.1%}"   if on  else "N/A"
    on_s  = f"{on['success_rate']:.1%}"  if on  else "N/A"
    delta = ""
    if off and on:
        d = on["success_rate"] - off["success_rate"]
        delta = f"{d:+.1%}"
    print(f"{label:<24} {off_g:>8} {off_s:>8}  {on_g:>8} {on_s:>8}  {delta:>7}")
print("─" * 72)
```

---

## 트러블슈팅

- **Vulkan segfault (`[svulkan2] Vulkan is incompatible with your driver`)**:
  `xvfb-run ... conda run ...` 패턴은 `conda run`이 새로운 서브프로세스를 생성하면서
  `DISPLAY` 환경 변수가 전달되지 않아 Vulkan 초기화에 실패합니다.
  셀 17/18은 이를 피하기 위해 Xvfb를 직접 백그라운드로 실행한 뒤
  `/usr/local/envs/spatialvla/bin/python`을 직접 호출합니다.
  (conda activate 없이도 해당 환경의 Python/패키지를 직접 사용합니다.)

- **`get_robot_control_mode(robot, "spatialvla")` KeyError**:
  `spatialvla_eval.py`는 이미 `try/except`로 `"spatialvla"` 실패 시 `"openvla"` 제어 모드로 fallback합니다. WidowX의 경우 둘 다 `arm_pd_ee_delta_pose`로 매핑되어 동일합니다.

- **flash-attn 미설치 경고**:
  SpatialVLA의 Gemma2 백본은 SDPA를 자동으로 eager로 전환합니다(logit softcapping). flash-attn 없이도 작동합니다. Colab에서 flash-attn 빌드는 불필요합니다.

- **numpy 버전 충돌**:
  SAPIEN/ManiSkill은 `numpy==1.24.4`를 요구합니다. transformers 4.47.0과 호환되므로 1.24.4로 고정하세요. (SpatialVLA requirements의 1.26.4는 학습용이며 추론엔 불필요)

- **DINO weight 자동 다운로드**:
  첫 step 실행 시 `IDEA-Research/grounding-dino-tiny`가 HF hub에서 자동 다운로드됩니다. 네트워크가 막힌 환경이면 사전에 `snapshot_download`로 받아두세요.

- **OOM (T4 15GB 등)**:
  SpatialVLA 4B(bf16) + GroundingDINO + SimplerEnv 렌더링은 약 10~12GB를 사용합니다. T4에서도 동작하지만 여유가 적으면 `--dino-cache-steps`를 늘려 DINO 호출 빈도를 줄이세요.

- **공정한 대조 실험 보장**:
  ON/OFF 두 실행 모두 동일한 `spatialvla_eval.py` 를 사용합니다. OFF 시 (`--no-latent-mask`) hook 은 등록되어 있지만 `_current_weight_1d is None` 조건에서 즉시 return 하므로 아무 영향이 없습니다. ActionEnsembler, image history, do_normalize, resize 등 공식 파이프라인은 두 조건에서 완전히 동일합니다.
