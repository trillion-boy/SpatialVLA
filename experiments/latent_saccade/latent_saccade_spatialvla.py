"""
latent_saccade_spatialvla.py

SpatialVLA용 Latent Saccade (Post-RMSNorm variant).

OpenVLA 포팅 버전과 메커니즘 동일:
  token embeddings → RMSNorm × weight → Q,K,V
                              ↑ weight survives into attention

OpenVLA와의 핵심 차이
--------------------
OpenVLA (LLaMA):
  시퀀스:   [BOS(pos 0)] [patch_0..patch_255(pos 1..256)] [text(pos 257..)]
  visual:  positions [1, 1+num_patches)
  레이어:   model.llm_backbone.llm.model.layers

SpatialVLA (Gemma2, PaliGemma-style):
  시퀀스:   [patch_0..patch_255(pos 0..255)] [BOS(pos 256)] [text(pos 257..)]
  visual:  input_ids 에서 id == image_token_index 인 위치를 스캔
           (UniVLA 의 vis_start≤id≤vis_end 스캔과 동일한 방식;
            SpatialVLA 는 전용 image token id 가 하나라 정확 일치로 탐지)
  레이어:   model.language_model.model.layers  ← 핵심 변경
  num_patches: 256 (SigLiP 224/14 → 16×16)

UniVLA 와의 일치
----------------
  UniVLA:     _build_seq_weight 가 input_ids 를 스캔해 visual 위치를 찾고
              (seq_len,) seq_weight 를 step() 에서 만든 뒤 hook 이 읽음
  SpatialVLA: 완전히 동일. id == image_token_index 스캔만 다름.

변경된 부분 (vs OpenVLA 포팅본)
------------------------------
  _find_decoder_layers : 레이어 경로 변경 (Gemma2)
  _build_seq_weight    : input_ids 스캔으로 visual 위치 탐지 (UniVLA 방식)
  __init__             : processor.image_seq_length 로 num_patches 탐지
  step()               : processor 입력 구성 + decode_actions 로 연속 action 획득

변경되지 않은 부분
-----------------
  SaccadeStateMachine (완전 동일)
  GroundingDINODetector (완전 동일)
  hook 핸들러 구조 (output * w.view(1, seq_len, 1))
  _register_postnorm_hooks 구조
  _current_seq_weight 메커니즘
  _build_weight_map (grid 기반, 동일)
  _get_bboxes, 2-tier cache 로직 (동일)

Usage
-----
  model, processor = load_spatialvla(model_path)
  saccade = LatentSaccadeSpatialVLAInference(
      model=model, processor=processor,
      unnorm_key="bridge_orig/1.0.0",
      bg_weight=1.0, place_src_weight=1.1, fovea_weight=1.3,
  )
  saccade.reset()
  action = saccade.step(image_np, instruction)   # np.ndarray (7,)
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image as PIL_Image


# ---------------------------------------------------------------------------
# Saccade State Machine  (OpenVLA 버전과 동일)
# ---------------------------------------------------------------------------

class SaccadeStateMachine:
    """
    Grasp / Place 2-phase state machine.

    state="grasp"  → fovea on source object
    state="place"  → fovea on destination object
    Transition: gripper close count ≥ consecutive_close_required
                AND grasp_steps ≥ min_grasp_steps
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

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

    def detect(
        self, image_np: np.ndarray, text: str
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Returns [(bbox_xyxy_pixels, score), ...] sorted by score descending.
        """
        if not text:
            return []
        if not text.endswith("."):
            text = text + "."
        pil_image = PIL_Image.fromarray(image_np)
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
        e.g. "put the eggplant in the basket" → ("eggplant", "basket")
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
# Main inference class
# ---------------------------------------------------------------------------

class LatentSaccadeSpatialVLAInference:
    """
    Latent Saccade for SpatialVLA (post-RMSNorm variant).

    Mechanism (identical to UniVLA / OpenVLA postnorm):
      Registers persistent forward hooks on input_layernorm of every Gemma2
      decoder layer.  During the prefill forward pass (seq_len > 1), the hook
      multiplies hidden_states by a (seq_len,) weight tensor.

      Weight tensor (SpatialVLA / PaliGemma-style sequence):
        visual patch positions → spatial weight from DINO detection
        BOS + text + padding   → 1.0

    Visual position identification (matches UniVLA's input_ids scan):
      UniVLA:     scan input_ids for vis_start ≤ id ≤ vis_end
      OpenVLA:    fixed positional slice [1, 1+num_patches)
      SpatialVLA: scan input_ids for id == image_token_index  ← exact match
                  (PaliGemma uses one dedicated image token id, so the scan is
                  exact rather than a range; positions need not be assumed.)
    """

    def __init__(
        self,
        model,
        processor=None,
        unnorm_key: Optional[str] = None,
        device: str = "cuda",
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_cache_steps: int = 5,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        bbox_margin: int = 2,
        bg_weight: float = 1.0,
        place_src_weight: float = 1.1,
        fovea_weight: float = 1.3,
        min_grasp_steps: int = 15,
        consecutive_close_required: int = 3,
        min_place_steps: int = 8,
        max_grasp_steps: int = 60,
        enable_latent_mask: bool = True,
        dino_debug_dir: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self._unnorm_key = unnorm_key
        self._bg_weight = bg_weight
        self._place_src_weight = place_src_weight
        self._fovea_weight = fovea_weight
        self._enable_latent_mask = enable_latent_mask
        self._dino_cache_steps = dino_cache_steps
        self._bbox_margin = bbox_margin
        self._dino_debug_dir = dino_debug_dir

        # ── Visual patch config ───────────────────────────────────────────
        # SpatialVLA uses SigLiP vision tower (not Prismatic vision_backbone).
        # Prefer processor.image_seq_length (set from image_processor.image_seq_length).
        # Fallback: compute from vision_config (image_size / patch_size)^2.
        if processor is not None and hasattr(processor, "image_seq_length"):
            self.num_patches: int = processor.image_seq_length
        else:
            try:
                cfg = model.config.vision_config
                self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
            except AttributeError:
                self.num_patches = 256  # SigLiP 224/14 default
        self._grid_size: int = int(round(self.num_patches ** 0.5))
        assert self._grid_size ** 2 == self.num_patches, (
            f"num_patches={self.num_patches} is not a perfect square; "
            "update _build_weight_map if using non-square patch grids."
        )

        # ── Saccade state machine ─────────────────────────────────────────
        self.saccade = SaccadeStateMachine(
            min_grasp_steps=min_grasp_steps,
            consecutive_close_required=consecutive_close_required,
            min_place_steps=min_place_steps,
            max_grasp_steps=max_grasp_steps,
        )

        # ── GroundingDINO detector ────────────────────────────────────────
        self.detector = GroundingDINODetector(
            model_id=dino_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

        # ── Image token id (for input_ids scan, mirrors UniVLA vis_start/vis_end) ──
        # SpatialVLA has a single dedicated image token (PaliGemma image_token_index),
        # so an exact equality scan is more precise than UniVLA's range check.
        self._image_token_index: int = getattr(
            getattr(model, "config", None), "image_token_index", 256000
        )

        # ── Internal state ────────────────────────────────────────────────
        self._current_instruction: Optional[str] = None
        # Full (seq_len,) weight tensor built in step() from input_ids scan.
        # (UniVLA stores _current_seq_weight the same way.)
        self._current_seq_weight: Optional[torch.Tensor] = None
        self._ln_hook_handles: List = []
        self._bbox_confidence_threshold: float = 0.3
        self._fovea_bbox_cache = None
        self._secondary_bbox_cache = None
        self._last_good_fovea = None
        self._last_good_secondary = None
        self._cache_step: int = 0

        # ── Register post-RMSNorm hooks ───────────────────────────────────
        self._register_postnorm_hooks()

    # ── Layer discovery ───────────────────────────────────────────────────

    def _find_decoder_layers(self):
        """
        SpatialVLA layer path (changed from OpenVLA):
          OpenVLA:     model.llm_backbone.llm.model.layers  (LLaMA)
          SpatialVLA:  model.language_model.model.layers    (Gemma2)
            language_model : Gemma2ForCausalLM
            model          : Gemma2Model
            layers         : ModuleList[Gemma2DecoderLayer]
        """
        candidates = [
            lambda m: m.language_model.model.layers,    # SpatialVLA (Gemma2)
            lambda m: m.llm_backbone.llm.model.layers,  # OpenVLA (LLaMA)
            lambda m: m.llm_backbone.model.layers,
            lambda m: m.model.layers,
        ]
        for fn in candidates:
            try:
                layers = fn(self.model)
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
        """
        Gemma2DecoderLayer has input_layernorm — same attribute name as LLaMA.
        No change needed vs OpenVLA version.
        """
        for attr in ("input_layernorm", "ln_1", "layer_norm_1", "norm1"):
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise RuntimeError(
            f"[LatentSaccade] Cannot find input_layernorm in {type(layer).__name__}. "
            f"Norm-like attrs: {[a for a in dir(layer) if 'norm' in a.lower() or 'ln' in a.lower()]}"
        )

    # ── Hook registration ─────────────────────────────────────────────────

    def _register_postnorm_hooks(self):
        """
        Hook logic identical to UniVLA postnorm.

        UniVLA pre-builds a (seq_len,) tensor in step() via _build_seq_weight
        (scanning input_ids for visual positions) and stores it in
        _current_seq_weight.  The hook just reads it, clamps to the current
        seq_len for safety, and multiplies.  We do exactly the same here —
        the visual positions are located by an input_ids scan, NOT by a
        fixed positional assumption.
        """
        layers = self._find_decoder_layers()

        for layer in layers:
            ln = self._find_layernorm(layer)

            def _make_hook(self_ref):
                def _hook(module, inp, output):
                    if not self_ref._enable_latent_mask:
                        return output
                    if self_ref._current_seq_weight is None:
                        return output
                    # Skip single-token autoregressive steps (KV cache active)
                    if output.shape[1] <= 1:
                        return output

                    seq_len = output.shape[1]
                    w = self_ref._current_seq_weight.to(
                        dtype=output.dtype, device=output.device
                    )
                    # Clamp in case seq lengths differ (safety) — same as UniVLA
                    w = w[:seq_len]
                    out = output.clone()
                    out = out * w.view(1, seq_len, 1)
                    return out
                return _hook

            handle = ln.register_forward_hook(_make_hook(self))
            self._ln_hook_handles.append(handle)

        print(
            f"[LatentSaccade] Registered post-RMSNorm hooks on "
            f"{len(self._ln_hook_handles)} Gemma2 decoder layers  "
            f"(num_patches={self.num_patches}, grid={self._grid_size}×{self._grid_size})"
        )

    # ── DINO detection ────────────────────────────────────────────────────

    def _get_bboxes(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (fovea_bbox, secondary_bbox) with two-tier DINO cache."""
        if self._cache_step % self._dino_cache_steps == 0:
            target = self.saccade.current_target
            secondary = self.saccade.source_noun if self.saccade.state == "place" else None
            thr = self._bbox_confidence_threshold

            H, W = image.shape[:2]
            max_area_ratio = 0.5

            def _area_ok(bbox):
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

    # ── Spatial weight map ────────────────────────────────────────────────

    def _build_weight_map(
        self,
        image: np.ndarray,
        fovea_bbox: Optional[np.ndarray],
        secondary_bbox: Optional[np.ndarray],
    ) -> torch.Tensor:
        """
        Build (num_patches,) spatial weight vector.
        Identical logic to OpenVLA version; grid_size adapts to ViT config.
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

    # ── Sequence weight builder (mirrors UniVLA _build_seq_weight) ─────────

    def _build_seq_weight(
        self,
        input_ids: torch.Tensor,            # (1, seq_len)
        weight_1d: Optional[torch.Tensor],  # (num_patches,) or None
    ) -> Optional[torch.Tensor]:
        """
        Build a (seq_len,) weight tensor:
          1.0   for all non-visual positions (BOS, text, padding)
          w_i   for visual token positions, in input_ids order

        Visual positions are found by scanning input_ids for image_token_index
        — the exact analogue of UniVLA's
            is_visual = (frame_ids >= vis_start) & (frame_ids <= vis_end)
        but using SpatialVLA's single dedicated image token id, so the match
        is exact rather than a range.

        Returns None if weight_1d is None (disables masking).
        """
        if weight_1d is None:
            return None

        seq_len = input_ids.shape[1]
        seq_weight = torch.ones(seq_len, dtype=torch.float32)

        is_visual = (input_ids[0] == self._image_token_index)
        vis_idx = is_visual.nonzero(as_tuple=True)[0]   # absolute positions
        n_vis = vis_idx.numel()

        if n_vis > 0:
            # weight_1d is the flattened patch grid in row-major order, which
            # matches the order image tokens appear in input_ids.
            w = weight_1d[:n_vis]
            seq_weight[vis_idx[:w.numel()]] = w
        else:
            print(
                "[LatentSaccade][warn] no image tokens found in input_ids "
                f"(image_token_index={self._image_token_index}); mask is a no-op"
            )

        return seq_weight

    # ── step: main inference with latent saccade ──────────────────────────

    def step(self, image: np.ndarray, goal: str) -> np.ndarray:
        """
        Run one inference step.

        Changes vs OpenVLA step():
          - processor builds inputs (images + text + unnorm_key)
          - model.predict_action() returns discrete token IDs
          - processor.decode_actions() converts to continuous 7D action
          - gripper signal from decoded action[-1] (same 0=close/1=open convention)

        Returns:
          action (np.ndarray, shape (7,)): unnormalized continuous action
            [dx, dy, dz, drx, dry, drz, gripper]
        """
        # ── 1. Sync instruction → saccade nouns ──────────────────────────
        if goal != self._current_instruction:
            self._current_instruction = goal
            src, dst = GroundingDINODetector.extract_source_dest_nouns(goal)
            self.saccade.source_noun = src
            self.saccade.dest_noun = dst
            print(f"[LatentSaccade] Instruction → src='{src}'  dst='{dst}'")

        # ── 2. DINO detection → spatial weight map ────────────────────────
        if self._enable_latent_mask:
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

        # ── 3. Store weight_1d → hooks read during generate() ────────────
        # ── 3. Build processor inputs ─────────────────────────────────────
        pil_image = PIL_Image.fromarray(image)
        prompt = f"What action should the robot take to {goal.lower()}?"

        inputs = self.processor(
            images=[pil_image],
            text=prompt,
            unnorm_key=self._unnorm_key,
            return_tensors="pt",
        )

        # ── 4. Build full seq_weight by scanning input_ids → hooks read it ─
        # (mirrors UniVLA: locate visual tokens in the actual sequence, then
        #  place the spatial weights at exactly those positions)
        self._current_seq_weight = self._build_seq_weight(
            inputs["input_ids"], weight_1d
        )

        try:
            # predict_action internally calls .to(bfloat16).to(device)
            generation_outputs = self.model.predict_action(inputs)
        finally:
            self._current_seq_weight = None   # always clear after generate

        # ── 5. Decode token IDs → continuous action ───────────────────────
        # generation_outputs: (1, max_new_tokens) token ID tensor
        # decode_actions returns {"actions": (chunk, 7), "action_ids": (chunk, 3)}
        decoded = self.processor.decode_actions(
            generation_outputs, unnorm_key=self._unnorm_key
        )
        action = decoded["actions"][0]   # (7,) for action_chunk_size=1

        # ── 6. Update saccade state from gripper output ───────────────────
        # SpatialVLA: gripper token 0 → 0.0 (close), token 1 → 1.0 (open)
        # Same convention as OpenVLA → close_thresh=0.5 works identically
        g = float(action[-1])
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

        return action

    # ── Reset / Cleanup ───────────────────────────────────────────────────

    def reset(self):
        """Reset per-episode state (call at episode start)."""
        self.saccade.reset()
        self._current_instruction = None
        self._current_seq_weight = None
        self._fovea_bbox_cache = None
        self._secondary_bbox_cache = None
        self._last_good_fovea = None
        self._last_good_secondary = None
        self._cache_step = 0

    def __del__(self):
        for handle in getattr(self, "_ln_hook_handles", []):
            handle.remove()

    # ── Debug helpers ─────────────────────────────────────────────────────

    def _save_debug_image(self, image, fovea_bbox, secondary_bbox, grid):
        """Save annotated debug image showing detected bboxes and weight grid."""
        import os
        from PIL import ImageDraw
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
