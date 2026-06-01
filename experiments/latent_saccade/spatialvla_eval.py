#!/usr/bin/env python3
"""
spatialvla_eval.py

SpatialVLA + Latent Saccade SimplerEnv 평가 스크립트.

LatentSaccadeSpatialVLAInference 가 공식 SpatialVLAInference 를 상속하므로
ActionEnsembler, image history, do_normalize=False, cv2 resize, raw prompt 등
공식 파이프라인이 완벽히 동일하게 유지됩니다.
--no-latent-mask 플래그로 ON / OFF 를 동일 코드에서 대조 실험합니다.

사용법:
  # Latent Saccade ON
  python experiments/latent_saccade/spatialvla_eval.py \\
    --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \\
    --task widowx_put_eggplant_in_basket --n-episodes 24

  # Baseline (OFF)
  python experiments/latent_saccade/spatialvla_eval.py \\
    --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \\
    --task widowx_put_eggplant_in_basket --n-episodes 24 --no-latent-mask
"""

import sys
import os
import argparse
import json
import time

import numpy as np
from PIL import Image as PIL_Image

# ── SimplerEnv 경로 (환경에 맞게 수정) ────────────────────────────────────
SIMPLER_ENV_ROOT = os.environ.get("SIMPLER_ENV_ROOT", "/content/SimplerEnv")
if os.path.exists(SIMPLER_ENV_ROOT):
    sys.path.insert(0, SIMPLER_ENV_ROOT)
    sys.path.insert(0, os.path.join(SIMPLER_ENV_ROOT, "ManiSkill2_real2sim"))


# ── Task configs ───────────────────────────────────────────────────────────
TASK_CONFIGS = {
    "widowx_put_eggplant_in_basket": {
        "env_name": "PutEggplantInBasketScene-v0",
        "robot": "widowx_sink_camera_setup",
        "scene_name": "bridge_table_1_v2",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 5,
        "sim_freq": 500,
        "max_episode_steps": 120,
        # robot init: from scripts/bridge.sh (widowx_sink_camera_setup scene)
        "robot_init_x": 0.127,
        "robot_init_y": 0.06,
    },
    "widowx_carrot_on_plate": {
        "env_name": "PutCarrotOnPlateInScene-v0",
        "robot": "widowx",
        "scene_name": "bridge_table_1_v1",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 5,
        "sim_freq": 500,
        "max_episode_steps": 60,
        # robot init: from scripts/bridge.sh (widowx bridge_table_1_v1 scene)
        "robot_init_x": 0.147,
        "robot_init_y": 0.028,
    },
    "widowx_stack_cube": {
        "env_name": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "robot": "widowx",
        "scene_name": "bridge_table_1_v1",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 5,
        "sim_freq": 500,
        "max_episode_steps": 60,
        # robot init: from scripts/bridge.sh (widowx bridge_table_1_v1 scene)
        "robot_init_x": 0.147,
        "robot_init_y": 0.028,
    },
    "widowx_spoon_on_towel": {
        "env_name": "PutSpoonOnTableClothInScene-v0",
        "robot": "widowx",
        "scene_name": "bridge_table_1_v1",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 5,
        "sim_freq": 500,
        "max_episode_steps": 60,
        # robot init: from scripts/bridge.sh (widowx bridge_table_1_v1 scene)
        "robot_init_x": 0.147,
        "robot_init_y": 0.028,
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="HF hub 경로 또는 로컬 체크포인트 디렉토리")
    p.add_argument("--unnorm-key", default="bridge_orig/1.0.0",
                   help="action un-normalization 통계 키")
    p.add_argument("--policy-setup", default="widowx_bridge",
                   choices=["widowx_bridge", "google_robot"])
    p.add_argument("--task", default="widowx_put_eggplant_in_basket",
                   choices=list(TASK_CONFIGS.keys()))
    p.add_argument("--n-episodes", type=int, default=24)
    p.add_argument("--output-dir", default="./latent_saccade_spatialvla_results")
    # Saccade weights (fovea-only boost, grasp/place 분리)
    p.add_argument("--bg-weight",        type=float, default=1.0,
                   help="배경 weight. 1.0=억제 안 함. 낮추면 공간 계획 파괴")
    p.add_argument("--place-src-weight", type=float, default=1.1)
    p.add_argument("--grasp-fovea-weight", type=float, default=1.15,
                   help="grasp 단계 target fovea weight (약하게 — 파지 방해 최소화)")
    p.add_argument("--place-fovea-weight", type=float, default=1.3,
                   help="place 단계 target fovea weight (강하게 — placement 정밀도)")
    p.add_argument("--fovea-weight",     type=float, default=None,
                   help="주면 grasp/place 둘 다 이 값으로 덮어씀 (하위호환)")
    # Saccade timing
    p.add_argument("--min-grasp-steps",  type=int, default=10)
    p.add_argument("--max-grasp-steps",  type=int, default=60,
                   help="Force grasp→place after this many steps (0=disabled)")
    p.add_argument("--consec-close",     type=int, default=3)
    p.add_argument("--min-place-steps",  type=int, default=8)
    # DINO
    p.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--dino-cache-steps", type=int, default=5)
    p.add_argument("--box-threshold",   type=float, default=0.15)
    p.add_argument("--text-threshold",  type=float, default=0.15)
    p.add_argument("--dino-debug-dir",  default=None)
    # Toggle foveation
    p.add_argument("--enable-latent-mask", action="store_true", default=True)
    p.add_argument("--no-latent-mask",     dest="enable_latent_mask", action="store_false")
    # grasp 단계 foveation (SpatialVLA 는 기본 OFF — place 단계만 foveate)
    p.add_argument("--foveate-grasp", action="store_true", default=False,
                   help="grasp 단계에도 foveation 적용 (실험용, 기본 OFF)")
    p.add_argument("--no-foveate-grasp", dest="foveate_grasp", action="store_false")
    # place 전환 후 foveation 지연 (lift 확보 → 파지 마무리 방해 방지)
    p.add_argument("--place-foveation-delay", type=int, default=2,
                   help="place 전환 후 N 스텝 foveation 보류 (물체 lift 확보용)")
    # bbox area 필터 (전체화면 오탐 차단, 기본 활성)
    p.add_argument("--enable-area-filter", action="store_true", default=True,
                   help="bbox area 필터 활성화 (전체화면 오탐 차단)")
    p.add_argument("--no-area-filter", dest="enable_area_filter", action="store_false")
    p.add_argument("--grasp-max-area-ratio", type=float, default=0.5,
                   help="grasp 단계 bbox 최대 면적 비율. eggplant(sink cam)=0.5, bridge_table_1_v1=0.95")
    p.add_argument("--place-max-area-ratio", type=float, default=0.6,
                   help="place 단계 bbox 최대 면적 비율. eggplant(sink cam)=0.6, bridge_table_1_v1=0.95")
    # Video / OOD
    p.add_argument("--save-video",  action="store_true")
    p.add_argument("--no-overlay",  action="store_true",
                   help="OOD: rgb_overlay 제거")
    p.add_argument("--overlay-path", default=None,
                   help="OOD: 다른 overlay 이미지 경로")
    p.add_argument("--brightness",   type=float, default=1.0,
                   help="OOD: 이미지 밝기 스케일 (1.0=정상)")
    return p.parse_args()


def build_env(cfg, ep_id, no_overlay=False, overlay_path=None):
    from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
    robot = cfg["robot"]
    try:
        control_mode = get_robot_control_mode(robot, "spatialvla")
    except Exception:
        control_mode = get_robot_control_mode(robot, "openvla")
    kw = dict(
        obs_mode="rgbd",
        robot=robot,
        sim_freq=cfg["sim_freq"],
        control_mode=control_mode,
        control_freq=cfg["control_freq"],
        max_episode_steps=cfg["max_episode_steps"],
        scene_name=cfg["scene_name"],
        camera_cfgs={"add_segmentation": True},
    )
    if not no_overlay:
        cand = None
        if overlay_path and os.path.exists(overlay_path):
            cand = overlay_path
        else:
            for base in [SIMPLER_ENV_ROOT, os.path.join(SIMPLER_ENV_ROOT, "ManiSkill2_real2sim")]:
                p = os.path.join(base, cfg["rgb_overlay_path"])
                if os.path.exists(p):
                    cand = p
                    break
        if cand:
            kw["rgb_overlay_path"] = cand
            kw["rgb_overlay_cameras"] = cfg["rgb_overlay_cameras"]
    env = build_maniskill2_env(cfg["env_name"], **kw)
    reset_options = {"obj_init_options": {"episode_id": ep_id}}
    if "robot_init_x" in cfg:
        reset_options["robot_init_options"] = {
            "init_xy": np.array([cfg["robot_init_x"], cfg["robot_init_y"]]),
            "init_rot_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }
    obs, _ = env.reset(options=reset_options)
    return env, obs


def get_image(env, obs, cam_name):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    return get_image_from_maniskill2_obs_dict(env, obs, camera_name=cam_name)


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return image
    from PIL import ImageEnhance
    pil = PIL_Image.fromarray(image)
    return np.array(ImageEnhance.Brightness(pil).enhance(factor))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    task_cfg = TASK_CONFIGS[args.task]
    cam_name = task_cfg["obs_camera_name"]

    # ── SpatialVLA path for latent_saccade module ──────────────────────────
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from experiments.latent_saccade.latent_saccade_spatialvla import LatentSaccadeSpatialVLAInference

    # ── Instantiate model (model loading happens inside __init__) ──────────
    # All official SpatialVLAInference params are passed through so the
    # pipeline is exactly equivalent to running the vanilla policy.
    policy = LatentSaccadeSpatialVLAInference(
        saved_model_path=args.model_path,
        unnorm_key=args.unnorm_key,
        policy_setup=args.policy_setup,
        # Latent Saccade params
        dino_model=args.dino_model,
        dino_cache_steps=args.dino_cache_steps,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        bg_weight=args.bg_weight,
        place_src_weight=args.place_src_weight,
        grasp_fovea_weight=args.grasp_fovea_weight,
        place_fovea_weight=args.place_fovea_weight,
        fovea_weight=args.fovea_weight,
        min_grasp_steps=args.min_grasp_steps,
        max_grasp_steps=args.max_grasp_steps,
        consecutive_close_required=args.consec_close,
        min_place_steps=args.min_place_steps,
        enable_latent_mask=args.enable_latent_mask,
        foveate_grasp=args.foveate_grasp,
        place_foveation_delay=args.place_foveation_delay,
        enable_area_filter=args.enable_area_filter,
        grasp_max_area_ratio=args.grasp_max_area_ratio,
        place_max_area_ratio=args.place_max_area_ratio,
        dino_debug_dir=args.dino_debug_dir,
    )
    print(
        f"[OK] num_patches={policy.num_patches}  "
        f"enable_latent_mask={args.enable_latent_mask}",
        flush=True,
    )

    # ── Episode loop ───────────────────────────────────────────────────────
    base_ids = list(range(*task_cfg["obj_episode_range"]))
    ep_ids   = [base_ids[i % len(base_ids)] for i in range(args.n_episodes)]
    results  = []

    for ep_count, ep_id in enumerate(ep_ids):
        print(f"\n── ep {ep_count:02d} (env_id={ep_id}) ──────────────────────────", flush=True)
        env, obs    = build_env(task_cfg, ep_id, no_overlay=args.no_overlay, overlay_path=args.overlay_path)
        instruction = env.get_language_instruction()
        image       = get_image(env, obs, cam_name)
        image       = apply_brightness(image, args.brightness)
        print(f"   instruction: {instruction}", flush=True)

        # reset() takes task_description (matches official SpatialVLAInference API)
        policy.reset(instruction)

        frames    = [image.copy()] if args.save_video else []
        done      = truncated = False
        grasped   = False
        step      = 0
        t0        = time.time()

        while not (done or truncated) and step < task_cfg["max_episode_steps"]:
            # step() returns (raw_action, action) — identical to official policy
            raw_action, action = policy.step(image, instruction)

            # official maniskill2_evaluator concatenates the dict into a flat array
            env_action = np.concatenate([
                action["world_vector"],
                action["rot_axangle"],
                np.atleast_1d(action["gripper"]),
            ])
            obs, _, done, truncated, info = env.step(env_action)
            image = apply_brightness(get_image(env, obs, cam_name), args.brightness)

            if not grasped and isinstance(info, dict):
                if info.get("is_src_obj_grasped", False):
                    grasped = True
                    print(f"[Grasp] env-reported grasp at step={step}", flush=True)

            if args.save_video and step % 4 == 0:
                frames.append(image.copy())

            # If environment changes the instruction (multi-stage tasks), sync policy
            new_instr = env.get_language_instruction()
            if new_instr != instruction:
                instruction = new_instr
                policy.reset(instruction)

            step += 1

        elapsed   = time.time() - t0
        grasp_str = "G+" if grasped else "G-"
        status    = "SUCCESS" if done else "FAIL"
        print(f"   → {grasp_str} {status}  ({step} steps, {elapsed:.1f}s)", flush=True)
        env.close()

        if args.save_video and frames:
            vpath = os.path.join(args.output_dir, f"ep{ep_count:02d}_{status.lower()}.gif")
            pils  = [PIL_Image.fromarray(f) for f in frames]
            pils[0].save(vpath, save_all=True, append_images=pils[1:], loop=0, duration=100)
            print(f"   GIF: {vpath}", flush=True)

        results.append({
            "ep": ep_count, "ep_id": ep_id,
            "grasped": grasped, "success": bool(done),
            "steps": step, "elapsed": elapsed,
        })

    # ── Summary ────────────────────────────────────────────────────────────
    n_grasp = sum(r["grasped"] for r in results)
    n_ok    = sum(r["success"] for r in results)
    gr      = n_grasp / len(results)
    sr      = n_ok    / len(results)
    tag     = "ON" if args.enable_latent_mask else "OFF (baseline)"
    print(f"\n{'='*50}", flush=True)
    print(f"  model:     SpatialVLA + LatentSaccade [{tag}]", flush=True)
    print(f"  task:      {args.task}", flush=True)
    print(f"  파지율:    {n_grasp}/{len(results)} = {gr:.1%}", flush=True)
    print(f"  성공률:    {n_ok}/{len(results)} = {sr:.1%}", flush=True)
    print(f"  평균 스텝: {np.mean([r['steps'] for r in results]):.0f}", flush=True)
    print(f"{'='*50}", flush=True)
    for r in results:
        g_mark = "G+" if r["grasped"] else "G-"
        s_mark = "OK" if r["success"] else "--"
        print(f"  {s_mark}{g_mark} ep{r['ep']:02d} (id={r['ep_id']}): {r['steps']} steps", flush=True)

    summary = {
        "model": f"SpatialVLA+LatentSaccade[{tag}]",
        "task": args.task,
        "enable_latent_mask": args.enable_latent_mask,
        "ood_no_overlay": args.no_overlay,
        "ood_overlay_path": args.overlay_path,
        "ood_brightness": args.brightness,
        "grasp_rate": gr,
        "success_rate": sr,
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "config": {
            "grasp_fovea_weight": args.grasp_fovea_weight,
            "place_fovea_weight": args.place_fovea_weight,
            "fovea_weight_override": args.fovea_weight,
            "bg_weight": args.bg_weight,
            "place_src_weight": args.place_src_weight,
            "foveate_grasp": args.foveate_grasp,
            "place_foveation_delay": args.place_foveation_delay,
            "min_grasp_steps": args.min_grasp_steps,
            "consec_close": args.consec_close,
            "dino_cache_steps": args.dino_cache_steps,
            "unnorm_key": args.unnorm_key,
            "policy_setup": args.policy_setup,
        },
        "episodes": results,
    }
    save_path = os.path.join(args.output_dir, f"results_{args.task}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n결과 저장: {save_path}", flush=True)


if __name__ == "__main__":
    main()
