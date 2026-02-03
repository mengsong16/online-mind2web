import argparse
import os
from methods.agenttrek_eval import *
from methods.automomous_eval import *
from methods.webjudge_general_eval import *
from methods.webjudge_online_mind2web import *
from methods.webvoyager_eval import *
from utils import OpenaiEngine, extract_predication, reset_eval_stats, read_eval_stats, log_eval_stat, set_eval_stats_path, set_eval_stats_shared, append_eval_log_lines
import json
import copy
import asyncio
import multiprocessing
from eval_config import JUDGE_MAX_TOKENS


def auto_eval(args, task_subset, final_predicted_labels, lock, model):

    ################## get the already done task id ###############
    output_json_path = os.path.join(args.output_path, f"{args.mode}_{args.model}_score_threshold_{args.score_threshold}_auto_eval_results.json")
    already_ids = []
    if os.path.exists(output_json_path):
        with open(output_json_path,"r") as f:
            already_data = f.read()
        already_tasks = already_data.splitlines()
        for item in already_tasks:
            item = json.loads(item)
            already_ids.append(item["task_id"])

    print(f"The number of already done tasks: {len(already_ids)}")

    for task_id in task_subset:
        #Skip already done task
        if task_id in already_ids:
            continue

        trajectory_images_path = os.path.join(args.trajectories_dir, task_id, "trajectory")
        screenshot_paths = []
        thoughts = None
        action_history = None
        final_result_response = None
        input_image_paths = None
        task_description = None
        # Load results
        with open(os.path.join(args.trajectories_dir, task_id, "result.json")) as f:
            result = json.load(f)
            output_results = copy.deepcopy(result)
            task_description = result["task"]
            if "action_history" in result:
                action_history = result["action_history"]
            if "thoughts" in result:
                thoughts = result["thoughts"]
            if "final_result_response" in result:
                final_result_response = result["final_result_response"]
            if "input_image_paths" in result:
                input_image_paths = result["input_image_paths"]

        print(f"Start evaluation for {task_description}")
        # Do the auto-eval
        if args.mode == "Autonomous_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                    screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg = Autonomous_eval(task_description, action_history, screenshot_paths[-1])
        
        elif args.mode == "AgentTrek_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                    screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg = AgentTrek_eval(task_description, action_history, thoughts, screenshot_paths[-1])
        
        elif args.mode == "WebVoyager_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg = WebVoyager_eval(task_description, screenshot_paths, final_result_response)
        
        elif args.mode == "WebJudge_Online_Mind2Web_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg, record, key_points = asyncio.run(WebJudge_Online_Mind2Web_eval(task_description, action_history, screenshot_paths, model, args.score_threshold))
            output_results["image_judge_record"] = record
            output_results["key_points"] = key_points

        elif args.mode == "WebJudge_general_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg, record, key_points = asyncio.run(WebJudge_general_eval(task_description, input_image_paths, thoughts, action_history, screenshot_paths, model, args.score_threshold))
            output_results["image_judge_record"] = record
            output_results["key_points"] = key_points

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        #response = model.generate(messages)[0] # default max_completion_tokens=512 
        response = model.generate(messages, max_new_tokens=JUDGE_MAX_TOKENS)[0]

        
        if response is None or len(str(response)) == 0:
            log_eval_stat("empty_judge", 1)
            print(f"[JUDGE MODEL RESPONSE RAW LEN] {0 if response is None else len(str(response))} task_id={task_id}", flush=True)
# =============== record debug info ===================
        lower = (response or "").lower()
        has_status = "status:" in lower
        output_results["debug_has_status"] = has_status
        # record has_status stats (cross-process)
        log_eval_stat("has_status_missing", 1 if not has_status else 0)
        log_eval_stat("has_status_present", 1 if has_status else 0)
        output_results["debug_response_len"] = 0 if response is None else len(response)
        output_results["debug_response_head"] = (response or "")[:300]
        output_results["debug_response_tail"] = (response or "")[-300:]
        # ==========================================
                
        predicted_label = extract_predication(response, args.mode)
        
        #Store evaluation details
        evaluation_results = {"response": response, "predicted_label": predicted_label}
        output_results["task_id"] = task_id
        output_results["input_text"] = text
        output_results["system_msg"] = system_msg
        output_results["evaluation_details"] = evaluation_results
        output_results["predicted_label"] = predicted_label

        with lock:
            final_predicted_labels.append(predicted_label)

        print(f"Finish evaluation for {task_description}")
        print("="*20)
        os.makedirs(args.output_path, exist_ok=True)
        with lock:
            with open(os.path.join(args.output_path, f"{args.mode}_{args.model}_score_threshold_{args.score_threshold}_auto_eval_results.json"), "a+") as f_out:
                f_out.write(json.dumps(output_results) + "\n")
        
        # --- DEBUG PRINTS (for parallel_eval) ---
        task_id = output_results.get("task_id", output_results.get("id", "UNKNOWN_TASK"))
        
        print("\n" + "=" * 90, flush=True)
        print(f"[EVAL DEBUG] task_id={task_id} mode={args.mode}", flush=True)
        print("-" * 90, flush=True)

        print("debug_has_status:", output_results.get("debug_has_status"), flush=True)
        print("debug_response_len:", output_results.get("debug_response_len"), flush=True)

        # head = output_results.get("debug_response_head") or ""
        # tail = output_results.get("debug_response_tail") or ""

        # print("debug_response_head:\n" + head, flush=True)
        # print("-" * 90, flush=True)
        # print("debug_response_tail:\n" + tail, flush=True)

        print("-" * 90, flush=True)
        print("predicted_label:", output_results.get("predicted_label"), flush=True)
        print("=" * 90 + "\n", flush=True)
        # --- END DEBUG PRINTS ---
        


def process_subset(task_subset, args, final_predicted_labels, lock, model, eval_stats):

    # Ensure child processes (including spawn) write stats to the same log path.
    set_eval_stats_path(os.path.join(args.output_path, 'open2mind_eval_stats.log'))

    set_eval_stats_shared(eval_stats, lock)

    auto_eval(args, task_subset, final_predicted_labels, lock, model)

def parallel_eval(args, num_workers=60):

    #Evaluate in parallel based on num of works
    task_dirs = [
        d for d in sorted(os.listdir(args.trajectories_dir)) 
        if os.path.isdir(os.path.join(args.trajectories_dir, d))
    ]
    os.makedirs(args.output_path, exist_ok=True)
    set_eval_stats_path(os.path.join(args.output_path, 'open2mind_eval_stats.log'))
    reset_eval_stats()  # clear shared eval stats log
    import time, datetime
    _t0 = time.perf_counter()
    print(f"Evaluating {len(task_dirs)} tasks in total.")
    chunk_size = len(task_dirs) // num_workers
    task_subsets = [task_dirs[i:i + chunk_size] for i in range(0, len(task_dirs), chunk_size)]

    #Load model
    model = OpenaiEngine(
        model=args.model,
        api_key=args.api_key
    )

    lock = multiprocessing.Lock()
    with multiprocessing.Manager() as manager:
        final_predicted_labels = manager.list()
        eval_stats = manager.dict()
        set_eval_stats_shared(eval_stats, lock)
        processes = []
        for subset in task_subsets:
            p = multiprocessing.Process(target=process_subset, args=(subset, args, final_predicted_labels, lock, model, eval_stats))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        success_num = sum(final_predicted_labels) 

        print("Evaluation complete.")
        print(f"The success rate is {(success_num / len(task_dirs)) * 100}.")
        stats = read_eval_stats()
        print(f"[EMPTY RESPONSE COUNTS] key_point={stats.get('empty_key_point', 0)} score={stats.get('empty_score', 0)} judge={stats.get('empty_judge', 0)}", flush=True)
        # Status presence stats (should be consistent with status_parse_error)
        has_missing = stats.get("has_status_missing", 0)
        has_present = stats.get("has_status_present", 0)
        status_parse_err = stats.get("status_parse_error", 0)
        print(f"[STATUS STATS] has_status_missing={has_missing} has_status_present={has_present} status_parse_error={status_parse_err}", flush=True)
        print(f"[SCORE STATS] score_parse_error={stats.get('score_parse_error', 0)} score_parse_ok={stats.get('score_parse_ok', 0)}", flush=True)
        _elapsed = datetime.timedelta(seconds=(time.perf_counter() - _t0))
        print(f"[EVAL TIMER] Total evaluation time: {_elapsed}", flush=True)
        # Also write a concise summary into the shared eval log file (stored under auto_eval_directory).
        append_eval_log_lines([
            f"The success rate is {(success_num / len(task_dirs)) * 100}.",
            f"[EMPTY RESPONSE COUNTS] key_point={stats.get('empty_key_point', 0)} score={stats.get('empty_score', 0)} judge={stats.get('empty_judge', 0)}",
            f"[STATUS STATS] has_status_missing={has_missing} has_status_present={has_present} status_parse_error={status_parse_err}",
            f"[SCORE STATS] score_parse_error={stats.get('score_parse_error', 0)} score_parse_ok={stats.get('score_parse_ok', 0)}",
            f"[EVAL TIMER] Total evaluation time: {_elapsed}",
        ])
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto evaluation of web navigation tasks.")
    parser.add_argument('--mode', type=str, default='Online_Mind2Web_eval', help='the mode of evaluation')
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument("--trajectories_dir", type=str, required=True, help="Path to trajectories directory")
    parser.add_argument("--api_key", type=str, required=True, help="The api key")
    parser.add_argument("--output_path", type=str, required=True, help="The output path")
    parser.add_argument('--score_threshold', type=int, default=3)
    parser.add_argument('--num_worker', type=int, default=60)
    args = parser.parse_args()

    parallel_eval(args, args.num_worker)