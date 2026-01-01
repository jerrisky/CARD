import os
import yaml

# 12 个 LDL 数据集列表
datasets = [
    "Flickr_LDL", "SBU_3DFE", "Scene", "Gene", "Movie", "RAF_ML", 
    "Ren_Cecps", "SJAFFE", "M2B", "SCUT_FBP5500", "Twitter_LDL", "SCUT_FBP"
]

# 严谨对齐原始 CARD cifar10.yml 的模板
# 同时保留了 LDL 任务所需的 MLP 结构 (simple/linear)
# 修正了 validation_freq = 10 (对齐原始)
# 修正了 snapshot_freq = 1e9 (对齐原始)
template = """data:
  dataset: "{dataset_name}"
  seed: 2000
  label_min_max: [0.001, 0.999]  # 对齐原始：保留此参数
  num_classes: 0                 # 代码会自动覆盖
  num_workers: 4
  dataroot: '../Data/feature/{dataset_name}' # 自动匹配路径
  run_idx: 0                     # 你的新增参数

model:
  type: "simple"
  data_dim: 0                    # 代码会自动覆盖
  n_input_channels: 3            # 对齐原始：保留结构
  n_input_padding: 0             # 对齐原始：保留结构
  feature_dim: 512               # MLP 宽度 (可按需统一修改)
  hidden_dim: 512                # MLP 宽度
  cat_x: True
  cat_y_pred: True
  arch: linear                   # LDL 任务特定
  var_type: fixedlarge
  ema_rate: 0.9999
  ema: True

diffusion:
  beta_schedule: linear
  beta_start: 0.0001
  beta_end: 0.02
  timesteps: 1000
  vis_step: 100
  num_figs: 10
  include_guidance: True
  apply_aux_cls: True
  trained_aux_cls_ckpt_path: ''
  trained_aux_cls_ckpt_name: ''
  aux_cls:
    arch: linear                 # LDL 任务特定
    pre_train: True
    joint_train: False
    n_pretrain_epochs: 100
    logging_interval: 10

training:
  batch_size: 128
  n_epochs: 5000                 
  warmup_epochs: 40
  add_t0_loss: False
  n_steps_req_grad: 100
  n_minibatches_add_ce: 20
  n_ce_epochs_warmup: 10
  n_ce_epochs_interval: 50
  n_sanity_check_epochs_freq: 500
  snapshot_freq: 1000000000      # 对齐原始：禁用按 Step 保存
  logging_freq: 100              # 建议：原始是1200(针对大图集)，LDL数据少，设100更合理
  validation_freq: 10            # 对齐原始：每 10 个 Epoch 验证一次
  image_folder: 'training_image_samples'

sampling:
  batch_size: 256
  sampling_size: 1000
  last_only: True
  image_folder: 'sampling_image_samples'

testing:
  batch_size: 256
  sampling_size: 1000
  last_only: True
  plot_freq: 200
  image_folder: 'testing_image_samples'
  n_samples: 100
  n_bins: 10
  compute_metric_all_steps: False
  metrics_t: 0
  ttest_alpha: 0.05
  trimmed_mean_range: [0.0, 100.0]
  PICP_range: [2.5, 97.5]
  make_plot: False
  squared_plot: False
  plot_true: False
  plot_gen: False
  fig_size: [8, 5]

optim:
  weight_decay: 0.000
  optimizer: "Adam"
  lr: 0.001
  beta1: 0.9
  amsgrad: False
  eps: 0.00000001
  grad_clip: 1.0
  lr_schedule: True
  min_lr: 0.0

aux_optim:
  weight_decay: 0.000
  optimizer: "Adam"
  lr: 0.001
  beta1: 0.9
  amsgrad: True
  eps: 0.00000001
  grad_clip: 1.0
"""

# 确保输出目录存在
output_dir = "configs" # 或者 "config"，根据你的文件夹名修改
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"🚀 Generating 12 aligned config files in '{output_dir}/'...")

for ds in datasets:
    # 填入数据集名称
    content = template.format(dataset_name=ds)
    
    file_path = os.path.join(output_dir, f"{ds}.yml")
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Generated: {file_path}")

print("\n🎉 All 12 configs are strictly aligned with CARD-original settings.")