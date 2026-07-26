from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_out_path: str = "data/libero/results.jsonl"  # Path to save structured rollout results
    action_trace_out_path: str | None = None  # Optional per-step state and action JSONL path
    save_videos: bool = True

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    video_out_path = pathlib.Path(args.video_out_path)
    results_out_path = pathlib.Path(args.results_out_path)
    action_trace_out_path = pathlib.Path(args.action_trace_out_path) if args.action_trace_out_path else None
    if results_out_path.exists():
        raise FileExistsError(f"Results file already exists: {results_out_path}. Choose a new --args.results-out-path.")
    if action_trace_out_path is not None and action_trace_out_path.exists():
        raise FileExistsError(
            f"Action trace file already exists: {action_trace_out_path}. Choose a new --args.action-trace-out-path."
        )
    results_out_path.parent.mkdir(parents=True, exist_ok=True)
    if action_trace_out_path is not None:
        action_trace_out_path.parent.mkdir(parents=True, exist_ok=True)
    action_trace_file = (
        action_trace_out_path.open("a", encoding="utf-8", buffering=1) if action_trace_out_path else None
    )
    if args.save_videos:
        video_out_path.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            episode_error = None
            replay_images = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    state = np.concatenate(
                        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    )
                    replanned = False
                    planned_actions = None
                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state,
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        planned_actions = np.asarray(action_chunk[: args.replan_steps])
                        action_plan.extend(planned_actions)
                        replanned = True

                    action = action_plan.popleft()
                    if action_trace_file is not None:
                        action_trace_record = {
                            "record_type": "action",
                            "task_suite": args.task_suite_name,
                            "task_id": task_id,
                            "task_description": task_description,
                            "episode_idx": episode_idx,
                            "seed": args.seed,
                            "t": t,
                            "state": np.asarray(state).tolist(),
                            "action": np.asarray(action).tolist(),
                            "replanned": replanned,
                            "planned_actions": planned_actions.tolist() if planned_actions is not None else None,
                        }
                        action_trace_file.write(json.dumps(action_trace_record) + "\n")

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    episode_error = str(e)
                    break

            task_episodes += 1
            total_episodes += 1

            if args.save_videos and replay_images:
                suffix = "success" if done else "failure"
                video_path = video_out_path / (
                    f"{args.task_suite_name}_task_{task_id:02d}_episode_{episode_idx:02d}_{suffix}.mp4"
                )
                if video_path.exists():
                    raise FileExistsError(f"Video already exists: {video_path}")
                imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

            episode_result = {
                "record_type": "episode",
                "task_suite": args.task_suite_name,
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": episode_idx,
                "seed": args.seed,
                "success": bool(done),
                "steps": t,
                "error": episode_error,
            }
            with results_out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(episode_result) + "\n")
            if action_trace_file is not None:
                action_trace_file.write(
                    json.dumps(
                        {
                            "record_type": "episode_end",
                            "task_suite": args.task_suite_name,
                            "task_id": task_id,
                            "episode_idx": episode_idx,
                            "seed": args.seed,
                            "success": bool(done),
                            "steps": t,
                            "error": episode_error,
                        }
                    )
                    + "\n"
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    summary_result = {
        "record_type": "summary",
        "task_suite": args.task_suite_name,
        "seed": args.seed,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "success_rate": float(total_successes) / float(total_episodes),
    }
    with results_out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary_result) + "\n")
    if action_trace_file is not None:
        action_trace_file.close()


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
