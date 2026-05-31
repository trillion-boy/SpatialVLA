"""
latent_saccade_spatialvla.py

SpatialVLA용 Latent Saccade (Post-RMSNorm variant).

공식 SpatialVLAInference (DelinQu/SimplerEnv-OpenVLA fork) 를 상속하여
latent saccade foveation hook 만 추가합니다.
ActionEnsembler, image history, do_normalize=False, cv2 resize, raw prompt 등
공식 파이프라인은 모두 super().step() 이 그대로 처리합니다.
단 하나의 차이: predict_action() 호출 중 input_layernorm hook 이 visual
patch 위치의 hidden state 를 공간적으로 가중합니다.

아키텍처
--------
SpatialVLA = PaliGemma2 (SigLiP + Gemma2 18층)
  시퀀스:   [image_token × num_patches][BOS][text...]
  visual:  첫 num_patches 위치 (PaliGemma 고정 레이아웃 — positional assumption)
  레이어:  model.language_model.model.layers  (Gemma2ForCausalLM)

Hook 위치 (post-RMSNorm variant)
--------------------------------
  token_emb → RMSNorm → *weight* → Q, K, V
  weight 가 Q, K 양쪽에 곱해지므로 attention score 는 weight² 로 증폭.
  UniVLA / OpenVLA postnorm 버전과 동일한 효과.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
except ImportError as exc:
    raise ImportError(
        "simpler_env 를 찾을 수 없습니다. DelinQu/SimplerEnv-OpenVLA fork 를 설치하세요:\n"
        "  git clone https://github.com/DelinQu/SimplerEnv-OpenVLA --recurse-submodules\n"
        "  pip install -e SimplerEnv-OpenVLA"
    ) from exc


# ---------------------------------------------------------------------------
# Saccade State Machine  (OpenVLA 버전과 동일)
# ---------------------------------------------------------------------------

class SaccadeStateMachine:
    """
    Grasp / Place 2-phase state machine.

    state='grasp'  → fovea on source object
    state='place'  → fovea on destination object
    Transition: gripper close count >= consecutive_close_required
                AND grasp_steps >= min_grasp_steps
    """

    def __init__(
        self,
        min_grasp_steps: int = 15,
        consecutive_close_required: int = 3,
        min_place_steps: int = 8,
        max_grasp_steps: int = 60,
        close_thresh: float = 0.5,
    ):
        self.min_grasp_steps = min_grasp_steps
        self.consecutive_close_required = consecutive_close_required
        self.min_place_steps = min_place_steps
        self.max_grasp_steps = max_grasp_steps
        self.close_thresh = close_thresh

        self.source_noun: str = ""
        self.dest_noun: str = ""
        self.state: str = "grasp"
        self._close_count: int = 0
        self._grasp_steps: int = 0

    @property
    def current_target(self) -> str:
        return self.source_noun if self.state == "grasp" else self.dest_noun

    def update(self, gripper_norm: float) -> bool:
        """
        Update state from gripper action value.
        SpatialVLA: g=1.0=open, g=0.0=close → close when g <= close_thresh (0.5).
        Returns True if state just transitioned grasp→place.
        """
        if self.state == "grasp":
            self._grasp_steps += 1
            if gripper_norm <= self.close_thresh:
                self._close_count += 1
            else:
                self._close_count = 0

            if (
                self._grasp_steps >= self.min_grasp_steps
                and self._close_count >= self.consecutive_close_required
            ):
                steps = self._grasp_steps
                self.state = "place"
                self._grasp_steps = 0
                self._close_count = 0
                print(f"[LatentSaccade] grasp→place  (gripper_close trigger, steps={steps})", flush=True)
                return True

            if self.max_grasp_steps > 0 and self._grasp_steps >= self.max_grasp_steps:
                steps = self._grasp_steps
                self.state = "place"
                self._grasp_steps = 0
                self._close_count = 0
                print(f"[LatentSaccade] grasp→place  (timeout at {steps} steps)", flush=True)
                return True
        return False

    def reset(self):
        self.state = "grasp"
        self._close_count = 0
        self._grasp_steps = 0


# ---------------------------------------------------------------------------
# GroundingDINO Detector  (OpenVLA 버전과 동일)
# ---------------------------------------------------------------------------

class GroundingDINODetector:
    """GroundingDINO wrapper using HuggingFace transformers."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        device: str = "cuda",
    ):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from PIL import Image as PIL_Image

        self._PIL_Image = PIL_Image
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

    def detect(
        self, image_np: np.ndarray, text: str
    ) -> List[Tuple[np.ndarray, float]]:
        """Returns [(bbox_xyxy_pixels, score), ...] sorted by score descending."""
        if not text:
            return []
        if not text.endswith("."):
            text = text + "."
        pil_image = self._PIL_Image.fromarray(image_np)
        inputs = self.processor(
            images=pil_image, text=text, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        detections = sorted(zip(boxes, scores), key=lambda x: -x[1])
        if detections:
            best_box, best_score = detections[0]
            print(f"[DINO] '{text.rstrip('.')}' score={best_score:.3f} → {best_box.astype(int).tolist()}")
        return detections

    _NOUN_REMAP: dict = {
        "towel": "tablecloth",
    }

    @staticmethod
    def extract_source_dest_nouns(instruction: str) -> Tuple[str, str]:
        """
        Regex-based extraction for standard manipulation instructions.
        e.g. 'put the eggplant in the basket' → ('eggplant', 'basket')
        """
        inst = instruction.lower().strip()
        pattern = (
            r"(?:put|place|move|stack|push)\s+"
            r"(?:the\s+)?(.+?)\s+"
            r"(?:on(?:\s+top\s+of)?|in(?:to)?|onto|inside)\s+"
            r"(?:the\s+)?(.+)"
        )
        remap = GroundingDINODetector._NOUN_REMAP
        m = re.match(pattern, inst)
        if m:
            src = m.group(1).strip().rstrip(".,")
            dst = m.group(2).strip().rstrip(".,")
            return remap.get(src, src), remap.get(dst, dst)
        return "", ""


# ---------------------------------------------------------------------------
# Main inference class  — inherits from official SpatialVLAInference
# ---------------------------------------------------------------------------

class LatentSaccadeSpatialVLAInference(SpatialVLAInference):
    """
    SpatialVLA + Latent Saccade foveation.

    상속 전략
    ---------
    SpatialVLAInference (공식 파이프라인) 를 그대로 상속하고, step() 에서
    super().step() 을 호출하기 전에 _current_weight_1d 를 세팅합니다.
    super().step() 내부에서 predict_action() 이 호출될 때 prefill forward
    pass 에 걸린 hook 이 해당 weight 를 hidden state 에 곱합니다.
    공식 파이프라인 (ActionEnsembler, image history, do_normalize=False,
    cv2 resize, raw task_description prompt) 은 전혀 변경되지 않습니다.

    Hook 동작
    ---------
    _current_weight_1d:  (num_patches,) float32 텐서
    hook: seq_len > 1 인 prefill 단계에서만 동작.
          첫 num_patches 위치 = visual patch → weight 적용
          나머지 위치 (BOS + text) → 1.0 (변경 없음)
          PaliGemma2 시퀀스는 항상 [image_tokens × N][BOS][text...] 이므로
          positional assumption 이 항상 성립함.
    """

    def __init__(
        self,
        # ── Official SpatialVLAInference params ────────────────────────────
        saved_model_path: str = "IPEC-COMMUNITY/spatialvla-4b-224-pt",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        exec_horizon: int = 1,
        image_size: list = None,
        action_scale: float = 1.0,
        action_ensemble_temp: float = -0.8,
        # ── Latent Saccade params ──────────────────────────────────────────
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_cache_steps: int = 5,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        bbox_margin: int = 2,
        # ── UniVLA fovea-only boost recipe (검증된 기본값) ──────────────────
        #   bg_weight=1.0      배경 절대 억제 안 함 (억제 시 공간 계획 파괴)
        #   place_src_weight   source/dest 영역 약한 boost
        #   fovea_weight       target 영역 boost → attention score = w² 증폭
        bg_weight: float = 1.0,
        place_src_weight: float = 1.1,
        fovea_weight: float = 1.3,
        min_grasp_steps: int = 15,
        consecutive_close_required: int = 3,
        min_place_steps: int = 8,
        max_grasp_steps: int = 60,
        enable_latent_mask: bool = True,
        # SpatialVLA 는 256패치뿐이라 fovea 가 20~25% 를 차지 → grasp 단계에
        # foveation 을 걸면 그리퍼 미세 제어 신호가 묻혀 파지가 망가짐.
        # (UniVLA 는 visual token 이 수천 개라 fovea 비율이 작아 괜찮았음)
        # 따라서 SpatialVLA 는 grasp 는 끄고 place 단계에만 foveation 적용.
        foveate_grasp: bool = False,
        # area 필터: SpatialVLA sink 카메라에서 'yellow basket' DINO 탐지가
        # 가끔 전체화면([1,70,638,478]≈85%)으로 잡힘 → fovea=256(전부) 가 되어
        # foveation 무의미. 정상 basket 은 화면의 ~20% 이므로 상한 0.6 으로
        # 전체화면 오탐만 차단.
        enable_area_filter: bool = True,
        place_max_area_ratio: float = 0.6,
        dino_debug_dir: Optional[str] = None,
    ):
        if image_size is None:
            image_size = [224, 224]

        # Initialise the official SpatialVLAInference (loads model, processor,
        # ActionEnsembler, image history deque, gripper state, etc.)
        super().__init__(
            saved_model_path=saved_model_path,
            unnorm_key=unnorm_key,
            policy_setup=policy_setup,
            exec_horizon=exec_horizon,
            image_size=image_size,
            action_scale=action_scale,
            action_ensemble_temp=action_ensemble_temp,
        )

        # ── Saccade weights ────────────────────────────────────────────────
        self._bg_weight = bg_weight
        self._place_src_weight = place_src_weight
        self._fovea_weight = fovea_weight
        self._enable_latent_mask = enable_latent_mask
        # foveate_grasp=True → UniVLA 처럼 grasp/place 양쪽 모두 foveation.
        self._foveate_grasp = foveate_grasp
        # area 필터: 전체화면 오탐 차단 (기본 활성, place 상한 0.6)
        self._enable_area_filter = enable_area_filter
        self._place_max_area_ratio = place_max_area_ratio
        self._dino_cache_steps = dino_cache_steps
        self._bbox_margin = bbox_margin
        self._dino_debug_dir = dino_debug_dir

        # ── Visual patch config (after super().__init__ so self.vla / processor exist) ──
        # processor.image_seq_length == num visual tokens injected into sequence
        self.num_patches: int = self.processor.image_seq_length
        self._grid_size: int = int(round(self.num_patches ** 0.5))
        assert self._grid_size ** 2 == self.num_patches, (
            f"num_patches={self.num_patches} is not a perfect square."
        )

        # ── Saccade state machine ──────────────────────────────────────────
        self.saccade = SaccadeStateMachine(
            min_grasp_steps=min_grasp_steps,
            consecutive_close_required=consecutive_close_required,
            min_place_steps=min_place_steps,
            max_grasp_steps=max_grasp_steps,
        )

        # ── GroundingDINO detector ─────────────────────────────────────────
        self.detector = GroundingDINODetector(
            model_id=dino_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device="cuda",
        )

        # ── Internal state ─────────────────────────────────────────────────
        # _saccade_instruction: tracks last instruction for saccade noun extraction
        self._saccade_instruction: Optional[str] = None
        # (num_patches,) weight set before super().step(); read by hook during predict_action()
        self._current_weight_1d: Optional[torch.Tensor] = None
        self._ln_hook_handles: List = []
        self._bbox_confidence_threshold: float = 0.3
        self._fovea_bbox_cache = None
        self._secondary_bbox_cache = None
        self._last_good_fovea = None
        self._last_good_secondary = None
        self._cache_step: int = 0

        # ── Register post-RMSNorm hooks ────────────────────────────────────
        self._register_postnorm_hooks()

    # ── Layer / norm discovery ─────────────────────────────────────────────

    def _find_decoder_layers(self):
        """
        Return the decoder layer ModuleList from self.vla.
        SpatialVLA (PaliGemma2 / Gemma2): model.language_model.model.layers
        """
        candidates = [
            lambda m: m.language_model.model.layers,   # SpatialVLA (Gemma2)
            lambda m: m.llm_backbone.llm.model.layers,  # OpenVLA (LLaMA)
            lambda m: m.llm_backbone.model.layers,
            lambda m: m.model.layers,
        ]
        for fn in candidates:
            try:
                layers = fn(self.vla)
                if layers is not None and len(layers) > 0:
                    return layers
            except AttributeError:
                continue
        raise RuntimeError(
            "[LatentSaccade] Cannot find decoder layers. "
            "Tried: language_model.model.layers, llm_backbone.llm.model.layers, "
            "llm_backbone.model.layers, model.layers"
        )

    def _find_layernorm(self, layer):
        """Return the pre-attention RMSNorm of a decoder layer."""
        for attr in ("input_layernorm", "ln_1", "layer_norm_1", "norm1"):
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise RuntimeError(
            f"[LatentSaccade] Cannot find input_layernorm in {type(layer).__name__}. "
            f"Norm-like attrs: {[a for a in dir(layer) if 'norm' in a.lower() or 'ln' in a.lower()]}"
        )

    # ── Hook registration ──────────────────────────────────────────────────

    def _register_postnorm_hooks(self):
        """
        Register persistent forward hooks on input_layernorm of every Gemma2
        decoder layer.

        Hook behaviour (post-RMSNorm variant, identical to UniVLA):
          Prefill (seq_len > 1): multiply hidden_states by (seq_len,) weight.
            - First num_patches positions → spatial weight from DINO bbox
            - Remaining positions (BOS + text) → 1.0
          AR steps (seq_len == 1): skip (KV cache active, position is fixed).

        Positional assumption:
          PaliGemma2 sequence layout is ALWAYS [img_tokens × N][BOS][text...].
          Image tokens occupy positions 0 .. num_patches-1 in every forward pass.
          This allows building the weight vector without scanning input_ids.
        """
        layers = self._find_decoder_layers()

        for layer in layers:
            ln = self._find_layernorm(layer)

            def _make_hook(self_ref):
                def _hook(module, inp, output):
                    if not self_ref._enable_latent_mask:
                        return output
                    if self_ref._current_weight_1d is None:
                        return output
                    if output.shape[1] <= 1:   # skip AR generation steps
                        return output

                    seq_len = output.shape[1]
                    n_vis = self_ref.num_patches

                    # Build (seq_len,) weight: visual = weight_1d, rest = 1.0
                    w_1d = self_ref._current_weight_1d.to(
                        dtype=output.dtype, device=output.device
                    )
                    w = torch.ones(seq_len, dtype=output.dtype, device=output.device)
                    n = min(n_vis, seq_len)
                    w[:n] = w_1d[:n]

                    return output * w.view(1, seq_len, 1)
                return _hook

            handle = ln.register_forward_hook(_make_hook(self))
            self._ln_hook_handles.append(handle)

        print(
            f"[LatentSaccade] Registered post-RMSNorm hooks on "
            f"{len(self._ln_hook_handles)} Gemma2 decoder layers  "
            f"(num_patches={self.num_patches}, grid={self._grid_size}×{self._grid_size})"
        )

    # ── DINO detection ─────────────────────────────────────────────────────

    def _get_bboxes(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (fovea_bbox, secondary_bbox) with two-tier DINO cache."""
        if self._cache_step % self._dino_cache_steps == 0:
            target = self.saccade.current_target
            secondary = self.saccade.source_noun if self.saccade.state == "place" else None
            thr = self._bbox_confidence_threshold

            H, W = image.shape[:2]

            # place 단계 상한. grasp 단계는 foveate_grasp=False 면 _get_bboxes 가
            # 애초에 호출되지 않으므로 사실상 place 전용 필터.
            max_area_ratio = self._place_max_area_ratio if self.saccade.state == "place" else 0.5

            def _area_ok(bbox):
                # 전체화면 오탐 차단. 정상 basket 은 화면의 ~20% 이므로 통과,
                # 전체화면 오탐(~85%) 은 거부.
                if not self._enable_area_filter:
                    return True
                x1, y1, x2, y2 = bbox
                ratio = ((x2 - x1) * (y2 - y1)) / (W * H)
                if ratio > max_area_ratio:
                    print(f"[DINO] bbox area {ratio:.1%} > {max_area_ratio:.0%} → rejected as false positive")
                    return False
                return True

            if target:
                dets = self.detector.detect(image, target)
                dets = [(b, s) for b, s in dets if _area_ok(b)]
                if dets and dets[0][1] >= thr:
                    self._fovea_bbox_cache = dets[0][0]
                    self._last_good_fovea = dets[0][0]
                elif dets:
                    print(f"[DINO] low-conf ({dets[0][1]:.3f} < {thr}) → using cached bbox")
                    self._fovea_bbox_cache = self._last_good_fovea
                else:
                    self._fovea_bbox_cache = self._last_good_fovea

            if secondary and secondary != target:
                dets = self.detector.detect(image, secondary)
                dets = [(b, s) for b, s in dets if _area_ok(b)]
                if dets and dets[0][1] >= thr:
                    self._secondary_bbox_cache = dets[0][0]
                    self._last_good_secondary = dets[0][0]
                else:
                    self._secondary_bbox_cache = self._last_good_secondary
            elif not secondary:
                self._secondary_bbox_cache = None

        self._cache_step += 1
        return self._fovea_bbox_cache, self._secondary_bbox_cache

    # ── Spatial weight map ─────────────────────────────────────────────────

    def _build_weight_map(
        self,
        image: np.ndarray,
        fovea_bbox: Optional[np.ndarray],
        secondary_bbox: Optional[np.ndarray],
    ) -> Optional[torch.Tensor]:
        """
        Build (num_patches,) spatial weight vector from detected bboxes.
        image is the original (unresized) observation — used for H/W only.
        Returns None if no bbox detected (disables masking for this step).
        """
        if fovea_bbox is None and secondary_bbox is None:
            return None

        H, W = image.shape[:2]
        g = self._grid_size

        grid = torch.full((g, g), self._bg_weight, dtype=torch.float32)

        def _bbox_to_grid(bbox):
            if bbox is None:
                return None
            x1, y1, x2, y2 = bbox
            c1 = max(0, int(x1 / W * g) - self._bbox_margin)
            r1 = max(0, int(y1 / H * g) - self._bbox_margin)
            c2 = min(g, int(np.ceil(x2 / W * g)) + self._bbox_margin)
            r2 = min(g, int(np.ceil(y2 / H * g)) + self._bbox_margin)
            if r2 <= r1 or c2 <= c1:
                return None
            return r1, c1, r2, c2

        sec_region = _bbox_to_grid(secondary_bbox)
        if sec_region:
            r1, c1, r2, c2 = sec_region
            grid[r1:r2, c1:c2] = self._place_src_weight

        fov_region = _bbox_to_grid(fovea_bbox)
        if fov_region:
            r1, c1, r2, c2 = fov_region
            grid[r1:r2, c1:c2] = self._fovea_weight

        if self._dino_debug_dir is not None:
            self._save_debug_image(image, fovea_bbox, secondary_bbox, grid)

        return grid.view(-1)   # (num_patches,)

    # ── Overridden step ────────────────────────────────────────────────────

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict, dict]:
        """
        Latent Saccade step — identical to official SpatialVLAInference.step()
        except that visual patch hidden states are spatially weighted during
        the prefill forward pass inside predict_action().

        Foveation is added via hooks registered in __init__; everything else
        (image resize, image history, processor call with do_normalize=False,
        predict_action + decode_actions, ActionEnsembler, gripper conversion)
        is handled by super().step() without modification.

        Returns:
            raw_action: dict with 'world_vector', 'rotation_delta', 'open_gripper'
            action:     dict with 'world_vector', 'rot_axangle', 'gripper',
                        'terminate_episode'  — pass directly to env.step()
        """
        # ── 1. Sync saccade nouns when instruction changes ─────────────────
        if task_description is not None and task_description != self._saccade_instruction:
            self._saccade_instruction = task_description
            src, dst = GroundingDINODetector.extract_source_dest_nouns(task_description)
            self.saccade.source_noun = src
            self.saccade.dest_noun = dst
            print(f"[LatentSaccade] Instruction → src='{src}'  dst='{dst}'")

        # ── 2. DINO detection on original image (before resize) ────────────
        # SpatialVLA 는 grasp 단계 foveation 이 파지를 망치므로(256패치 한계)
        # 기본적으로 place 단계에만 foveation 적용 (foveate_grasp=False).
        # foveate_grasp=True 로 두면 grasp 단계에도 적용(실험용).
        foveate_now = self._enable_latent_mask and (
            self._foveate_grasp or self.saccade.state == "place"
        )
        if foveate_now:
            fovea_bbox, secondary_bbox = self._get_bboxes(image)
            weight_1d = self._build_weight_map(image, fovea_bbox, secondary_bbox)
        else:
            fovea_bbox = secondary_bbox = None
            weight_1d = None

        n_fovea = int((weight_1d >= self._fovea_weight).sum()) if weight_1d is not None else 0
        n_src = (
            int(((weight_1d >= self._place_src_weight) & (weight_1d < self._fovea_weight)).sum())
            if weight_1d is not None else 0
        )
        n_bg = int((weight_1d < self._place_src_weight).sum()) if weight_1d is not None else 0
        print(
            f"[LatentSaccade] phase={self.saccade.state}  "
            f"target='{self.saccade.current_target}'  "
            f"fovea={n_fovea}  src={n_src}  bg={n_bg}  "
            f"fovea_bbox={fovea_bbox}"
        )

        # ── 3. Activate hook weight, run official pipeline ─────────────────
        # Note: super().step() may call self.reset(task_description) internally
        # if the instruction changed (first call of each episode).  Our reset()
        # does NOT clear _current_weight_1d so the hook remains active.
        self._current_weight_1d = weight_1d
        try:
            raw_action, action = super().step(image, task_description, *args, **kwargs)
        finally:
            self._current_weight_1d = None   # always clear after generate()

        # ── 4. Update saccade state from raw gripper output ────────────────
        # raw_action["open_gripper"]: 0.0 = close, 1.0 = open (from tokenizer)
        g = float(raw_action["open_gripper"])
        print(
            f"[LatentSaccade-dbg] g={g:.2f}  "
            f"close_count={self.saccade._close_count}  "
            f"grasp_steps={self.saccade._grasp_steps}",
            flush=True,
        )
        transitioned = self.saccade.update(g)
        if transitioned:
            self._fovea_bbox_cache = None
            self._secondary_bbox_cache = None
            self._cache_step = 0
            print("[LatentSaccade] State transition: grasp → place", flush=True)

        return raw_action, action

    # ── Overridden reset ───────────────────────────────────────────────────

    def reset(self, task_description: str) -> None:
        """Reset per-episode state. Delegates to official SpatialVLAInference.reset()."""
        super().reset(task_description)
        self.saccade.reset()
        # _saccade_instruction reset to None so saccade nouns are re-extracted next step
        self._saccade_instruction = None
        # Do NOT clear _current_weight_1d here — managed by step()'s finally block.
        # (super().step() may call this reset() mid-step; the weight must remain active.)
        self._fovea_bbox_cache = None
        self._secondary_bbox_cache = None
        self._last_good_fovea = None
        self._last_good_secondary = None
        self._cache_step = 0

    def __del__(self):
        for handle in getattr(self, "_ln_hook_handles", []):
            handle.remove()

    # ── Debug helpers ──────────────────────────────────────────────────────

    def _save_debug_image(self, image, fovea_bbox, secondary_bbox, grid):
        """Save annotated debug image showing detected bboxes and weight grid."""
        import os
        from PIL import Image as PIL_Image, ImageDraw
        os.makedirs(self._dino_debug_dir, exist_ok=True)
        pil = PIL_Image.fromarray(image).copy()
        draw = ImageDraw.Draw(pil)
        if fovea_bbox is not None:
            x1, y1, x2, y2 = fovea_bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 12), f"fovea ({self._fovea_weight})", fill="red")
        if secondary_bbox is not None:
            x1, y1, x2, y2 = secondary_bbox
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            draw.text((x1, y1 - 12), f"secondary ({self._place_src_weight})", fill="blue")
        step_idx = self._cache_step
        save_path = os.path.join(self._dino_debug_dir, f"step_{step_idx:05d}.png")
        pil.save(save_path)
