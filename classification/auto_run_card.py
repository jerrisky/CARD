import os
import yaml
import argparse
import itertools
import numpy as np
import sys
import shutil
import time

# 引入 main 脚本作为模块
import main as main_script 

# ================= 配置区域 =================
RESULT_ROOT = "result"
MODEL_ROOT = "model"
TEMP_MODEL_ROOT = "temp_model_search"
TEMP_CONFIG_DIR = "temp_configs"
CONFIG_DIR = "configs"

# 搜索空间
SEARCH_SPACE = {
    "lr": [1e-3, 5e-4, 1e-4],
    "batch_size": [64, 128],
    "hidden_dim": [256, 512],
    "feature_dim": [64,256, 512,1024]
}

METRICS_KEYS = ['Cheby', 'Clark', 'Canbe', 'KL', 'Cosine', 'Inter']

# ================= 工具函数 (保持不变) =================

def check_base_config(dataset):
    yml_path = os.path.join(CONFIG_DIR, f"{dataset}.yml")
    if not os.path.exists(yml_path):
        print(f"❌ Config not found: {yml_path}")
        sys.exit(1)
    with open(yml_path, 'r') as f:
        return yaml.safe_load(f)

def update_config(dataset, params, run_idx, is_search=False):
    base_yml = os.path.join(CONFIG_DIR, f"{dataset}.yml")
    with open(base_yml, 'r') as f:
        config = yaml.safe_load(f)

    # 1. 注入参数
    config.setdefault('optim', {})['lr'] = params['lr']
    config.setdefault('training', {})['batch_size'] = params['batch_size']
    config.setdefault('model', {})['hidden_dim'] = params['hidden_dim']
    config.setdefault('model', {})['feature_dim'] = params['feature_dim']
    
    # 2. 注入 Split/Run
    config.setdefault('data', {})['run_idx'] = 0 if is_search else run_idx
    config['data']['dataset'] = dataset

    # 3. Epoch 控制
    if is_search:
        config.setdefault('training', {})['n_epochs'] = 200
    else:
        config.setdefault('training', {})['n_epochs'] = 3000

    # 4. 生成临时文件
    prefix = "search" if is_search else "eval"
    temp_name = f"temp_{dataset}_{prefix}_{run_idx}_{int(time.time())}.yml"
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)
    temp_path = os.path.join(TEMP_CONFIG_DIR, temp_name)
    
    with open(temp_path, 'w') as f:
        yaml.dump(config, f)
    return temp_path

def run_task(dataset, params, run_idx, device, is_search_phase=False):
    # 1. 生成配置
    config_path = update_config(dataset, params, run_idx, is_search_phase)
    
    # 2. 确定路径和文档名
    if is_search_phase:
        model_save_dir = TEMP_MODEL_ROOT 
        doc_name = f"search/search_{run_idx}" 
    else:
        model_save_dir = MODEL_ROOT
        doc_name = f"run_{run_idx}"

    # 3. 构造命令参数列表
    cmd_args = [
        "--config", config_path,
        "--doc", doc_name,
        "--exp", os.path.join(RESULT_ROOT, dataset), 
        "--device", str(device),
        "--loss", "card_onehot_conditional",
        "--model_dir", model_save_dir,
        "--split", str(run_idx),
        "--ni",              
        "--verbose", "info"
    ]
    
    if is_search_phase:
        cmd_args.append("--tune")
        
    print(f"🚀 Running: {doc_name} | Params: {params}")

    try:
        # 解析参数
        task_args = main_script.parser.parse_args(cmd_args)
        
        # 4. 调用 main 函数
        result = main_script.main(task_args) 
        
    except Exception as e:
        print(f"❌ Error in run_task: {e}")
        import traceback
        traceback.print_exc()
        result = None

    # 5. 清理临时配置文件
    if os.path.exists(config_path):
        os.remove(config_path)
    
    # 清理搜索阶段产生的临时模型文件 (temp_model_search 文件夹)
    if is_search_phase:
        temp_run_dir = os.path.join(TEMP_MODEL_ROOT, dataset, f"run_0")
        if os.path.exists(temp_run_dir):
            shutil.rmtree(temp_run_dir, ignore_errors=True)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    # --- Phase 1: Grid Search ---
    print(f"\n🔍 Phase 1: Search (Metric: AvgImp, Epochs: 200)...")
    keys, values = zip(*SEARCH_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    search_results = [] 

    for i, params in enumerate(combinations):
        # 直接拿到 float
        avg_imp = run_task(args.dataset, params, i, args.device, is_search_phase=True)
        
        # 过滤无效结果
        if avg_imp is not None and isinstance(avg_imp, float):
            print(f"👉 Trial {i}: AvgImp = {avg_imp:.4%}")
            search_results.append({
                "params": params,
                "imp": avg_imp,
                "id": i
            })
        else:
            print(f"❌ Trial {i} Failed.")

    if not search_results:
        print("❌ All search trials failed!")
        sys.exit(1)

    # 排序找最大
    search_results.sort(key=lambda x: x['imp'], reverse=True)
    best_record = search_results[0]
    best_params = best_record['params']
    
    print(f"\n✅ Search Finished.")
    print(f"🏆 Best Params: {best_params} (Imp: {best_record['imp']:.4%})")

    # --- Phase 2: Evaluation ---
    print(f"\n🏃 Phase 2: Running 10-Fold Evaluation (Epochs=3000)...")
    
    all_results = []
    summary_txt_path = os.path.join(RESULT_ROOT, args.dataset, "result.txt")
    os.makedirs(os.path.dirname(summary_txt_path), exist_ok=True)
    
    # 先写入表头
    with open(summary_txt_path, "w") as f:
        f.write(f"Experiment Report: {args.dataset}\n")
        f.write(f"Best Params: {best_params}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Run':<5} | {'Cheby':<8} | {'Clark':<8} | {'Canbe':<8} | {'KL':<8} | {'Cosine':<8} | {'Inter':<8}\n")

    for run_idx in range(10):
        # 直接拿到 list
        metrics = run_task(args.dataset, best_params, run_idx, args.device, is_search_phase=False)
        
        if metrics is not None and isinstance(metrics, (list, tuple)) and len(metrics) == len(METRICS_KEYS):
            all_results.append(metrics)
            # 【修改点 1】: .4f 改为 .8f (保留8位小数)
            res_str = f"{run_idx:<5} | " + " | ".join([f"{v:.8f}" for v in metrics])
            print(f"✅ Run {run_idx} Done: {res_str}")
            with open(summary_txt_path, "a") as f:
                f.write(res_str + "\n")
        else:
            print(f"❌ Run {run_idx} Failed.")

    # --- Phase 3: Statistics ---
    if all_results:
        all_results = np.array(all_results)
        means = np.mean(all_results, axis=0)
        stds = np.std(all_results, axis=0)
        
        summary = []
        summary.append("\n" + "="*50)
        summary.append(f"Final Average Results (Mean ± Std)")
        summary.append("-" * 50)
        for i, key in enumerate(METRICS_KEYS):
            # 【修改点 2】: .4f 改为 .8f (保留8位小数)
            summary.append(f"{key:<10}: {means[i]:.8f} ± {stds[i]:.8f}")
        summary.append("="*50)
        
        summary_str = "\n".join(summary)
        print(summary_str)
        with open(summary_txt_path, "a") as f:
            f.write(summary_str)
            
        print(f"\n🎉 All Done! Result saved to: {summary_txt_path}")