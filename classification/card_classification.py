import logging
import time
import gc
import matplotlib.pyplot as plt
# import statsmodels.api as sm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.stats import ttest_rel
from tqdm import tqdm
from data_loader import *
from ema import EMA
from model import *
from pretraining.encoder import Model as AuxCls
from pretraining.resnet import ResNet18
from utils import *
from diffusion_utils import *
import json  # 新增
import metrics
plt.style.use('ggplot')
import torch
import torch.multiprocessing
from sklearn.model_selection import KFold
from torch.utils.data import Subset
torch.multiprocessing.set_sharing_strategy('file_system')
def calc_avg_improvement(current_scores, sota_scores):
    improvements = []
    # 0-3: 越小越好 (SOTA - Ours) / SOTA
    for i in range(4):
        val = current_scores[i]
        ref = sota_scores[i]
        imp = (ref - val) / (ref + 1e-8)
        improvements.append(imp)
    
    # 4-5: 越大越好 (Ours - SOTA) / SOTA
    for i in range(4, 6):
        val = current_scores[i]
        ref = sota_scores[i]
        imp = (val - ref) / (ref + 1e-8)
        improvements.append(imp)
        
    return np.mean(improvements)
class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.ldl_datasets = ["SBU_3DFE", "Scene", "Gene", "Movie", "RAF_ML", "Ren_Cecps", 
                             "SJAFFE", "M2B", "SCUT_FBP5500", "Twitter_LDL", "Flickr_LDL", "SCUT_FBP"]
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.model_save_root = getattr(args, 'model_dir', 'model')
        self.model_var_type = config.model.var_type
        self.num_timesteps = config.diffusion.timesteps
        self.vis_step = config.diffusion.vis_step
        self.num_figs = config.diffusion.num_figs

        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=config.diffusion.beta_start, end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # initial prediction model as guided condition
        if config.diffusion.apply_aux_cls:
            if config.data.dataset == "gaussian_mixture":
                self.cond_pred_model = nn.Sequential(
                    nn.Linear(1, 100),
                    nn.ReLU(),
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 1)
                ).to(self.device)
            # elif config.data.dataset == "MNIST" and config.model.arch == "simple":
            elif config.data.dataset == "MNIST":
                self.cond_pred_model = nn.Sequential(
                    nn.Linear(config.model.data_dim, 300),
                    nn.BatchNorm1d(300),
                    nn.ReLU(),
                    nn.Linear(300, 100),
                    nn.BatchNorm1d(100),
                    nn.ReLU(),
                    nn.Linear(100, config.data.num_classes)
                ).to(self.device)
            elif config.data.dataset in ["FashionMNIST", "CIFAR10", "CIFAR100"]:
                if config.diffusion.aux_cls.arch == "lenet":
                    self.cond_pred_model = LeNet(config.data.num_classes,
                                                 config.model.n_input_channels,
                                                 config.model.n_input_padding).to(self.device)
                elif config.diffusion.aux_cls.arch == "lenet5":
                    self.cond_pred_model = LeNet5(config.data.num_classes,
                                                  config.model.n_input_channels,
                                                  config.model.n_input_padding).to(self.device)
                elif config.diffusion.aux_cls.arch == "resnet18_ckpt":
                    # self.cond_pred_model = resnet18(pretrained=False).to(self.device)
                    self.cond_pred_model = ResNet18(num_classes=config.data.num_classes).to(self.device)
                elif config.data.dataset in self.ldl_datasets:
                    self.cond_pred_model = nn.Sequential(
                        nn.Linear(config.model.data_dim, config.model.hidden_dim),
                        nn.BatchNorm1d(config.model.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(config.model.hidden_dim, config.model.hidden_dim),
                        nn.BatchNorm1d(config.model.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(config.model.hidden_dim, config.data.num_classes)
                    ).to(self.device)
                    logging.info(f"Initialized MLP Guidance Model for LDL: {config.model.data_dim} -> {config.model.hidden_dim} -> {config.data.num_classes}")
                else:
                    self.cond_pred_model = AuxCls(config).to(self.device)
            elif config.data.dataset in self.ldl_datasets:
                self.cond_pred_model = nn.Sequential(
                    nn.Linear(config.model.data_dim, config.model.hidden_dim),
                    nn.BatchNorm1d(config.model.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.model.hidden_dim, config.model.hidden_dim),
                    nn.BatchNorm1d(config.model.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.model.hidden_dim, config.data.num_classes)
                ).to(self.device)
                logging.info(f"Initialized MLP Guidance Model for LDL Task.")
            else:
                self.cond_pred_model = AuxCls(config).to(self.device)
            if config.data.dataset in self.ldl_datasets:
                logging.info("LDL Task: Using Raw MSELoss (No Softmax constraints)")
                self.aux_cost_function = nn.MSELoss()
            else:
                self.aux_cost_function = nn.CrossEntropyLoss()
        else:
            pass

        # scaling temperature for NLL and ECE computation
        self.sota_values = None
        sota_json_path = '../../Data/sota.json'
        self.metrics_keys = ['Cheby', 'Clark', 'Canbe', 'KL', 'Cosine', 'Inter']

        if os.path.exists(sota_json_path):
            try:
                with open(sota_json_path, 'r', encoding='utf-8') as f:
                    full_json = json.load(f)
                    sota_data = full_json.get('data', {})
                
                key = config.data.dataset
                if key in sota_data:
                    # 按照 metrics_keys 的顺序提取 mean 值
                    self.sota_values = [sota_data[key][k]['mean'] for k in self.metrics_keys]
                    logging.info(f"Loaded SOTA for {key}: {self.sota_values}")
                else:
                    logging.warning(f"Dataset {key} not found in sota.json")
            except Exception as e:
                logging.warning(f"Failed to load SOTA: {e}")
        else:
            logging.warning(f"SOTA file not found at {sota_json_path}")
        self.tuned_scale_T = None
    # Compute guiding prediction as diffusion condition
    def compute_guiding_prediction(self, x):
        """
        Compute y_0_hat, to be used as the Gaussian mean at time step T.
        """
        if self.config.model.arch == "simple" or \
                (self.config.model.arch == "linear" and self.config.data.dataset == "MNIST"):
            x = torch.flatten(x, 1)
        y_pred = self.cond_pred_model(x)
        return y_pred

    def evaluate_guidance_model(self, dataset_loader):
        """
        Evaluate guidance model by reporting train or test set accuracy.
        """
        y_acc_list = []
        for step, feature_label_set in tqdm(enumerate(dataset_loader)):
            # logging.info("\nEvaluating test Minibatch {}...\n".format(step))
            # minibatch_start = time.time()
            x_batch, y_labels_batch = feature_label_set
            y_labels_batch = y_labels_batch.reshape(-1, 1)
            y_pred_prob = self.compute_guiding_prediction(
                x_batch.to(self.device)).softmax(dim=1)  # (batch_size, n_classes)
            y_pred_label = torch.argmax(y_pred_prob, 1, keepdim=True).cpu().detach().numpy()  # (batch_size, 1)
            y_labels_batch = y_labels_batch.cpu().detach().numpy()
            y_acc = y_pred_label == y_labels_batch  # (batch_size, 1)
            if len(y_acc_list) == 0:
                y_acc_list = y_acc
            else:
                y_acc_list = np.concatenate([y_acc_list, y_acc], axis=0)
        y_acc_all = np.mean(y_acc_list)
        return y_acc_all

    def nonlinear_guidance_model_train_step(self, x_batch, y_batch, aux_optimizer):
        """
        One optimization step of the non-linear guidance model that predicts y_0_hat.
        """
        y_batch_pred = self.compute_guiding_prediction(x_batch)
        aux_cost = self.aux_cost_function(y_batch_pred, y_batch)
        # update non-linear guidance model
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        return aux_cost.cpu().item()
    def tune(self):
        logging.info(">>> Starting Internal 5-Fold Tuning (Minimal Mode)...")
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        config.data.run_idx = 0 
        data_object, full_train_dataset, _ = get_dataset(args, config)
        real_data_dim = getattr(full_train_dataset, 'feature_dim', config.model.data_dim)
        real_num_classes = getattr(full_train_dataset, 'label_dim', config.data.num_classes)
        train_size = len(full_train_dataset)
        kfold = KFold(n_splits=5, shuffle=True, random_state=config.data.seed)
        fold_imps = []

        # 强制更新 Config，确保模型使用真实的数据维度
        config.model.data_dim = real_data_dim
        config.data.num_classes = real_num_classes
        
        logging.info(f"Dataset Loaded. Samples: {train_size}. data_dim: {real_data_dim}")
        for fold, (train_ids, val_ids) in enumerate(kfold.split(full_train_dataset)):
            logging.info(f"\n--- Tuning Fold {fold+1} / 5 ---")
            # (A) 准备数据 (切分)
            train_sub = Subset(full_train_dataset, train_ids)
            val_sub = Subset(full_train_dataset, val_ids)
            train_loader = data.DataLoader(train_sub, batch_size=config.training.batch_size, shuffle=True, num_workers=config.data.num_workers)
            test_loader = data.DataLoader(val_sub, batch_size=config.testing.batch_size, shuffle=False, num_workers=config.data.num_workers)
            if config.diffusion.apply_aux_cls and config.data.dataset in self.ldl_datasets:
                self.cond_pred_model = nn.Sequential(
                    nn.Linear(config.model.data_dim, config.model.hidden_dim),
                    nn.BatchNorm1d(config.model.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.model.hidden_dim, config.model.hidden_dim),
                    nn.BatchNorm1d(config.model.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.model.hidden_dim, config.data.num_classes)
                ).to(self.device)
                logging.info(f"Guidance Model (MLP) Re-initialized: [{real_data_dim}] -> [{config.model.hidden_dim}] -> [{real_num_classes}]")
            # ================================================================
            model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
            model = model.to(self.device)
            
            if self.config.data.dataset not in self.ldl_datasets:
                # 原版逻辑 (CIFAR/MNIST 等) 保留
                y_acc_aux_model = self.evaluate_guidance_model(test_loader)
                logging.info("\nBefore training, the guidance classifier accuracy on the test set is {:.8f}.\n\n".format(
                    y_acc_aux_model))
            else:
                # LDL 逻辑：跳过计算，打印一条提示即可
                logging.info("\n[LDL Task] Skipping initial accuracy check (Metric incompatibility).\n")

            optimizer = get_optimizer(self.config.optim, model.parameters())
            criterion = nn.CrossEntropyLoss()
            brier_score = nn.MSELoss()

            # apply an auxiliary optimizer for the guidance classifier
            if config.diffusion.apply_aux_cls:
                aux_optimizer = get_optimizer(self.config.aux_optim,
                                            self.cond_pred_model.parameters())

            if self.config.model.ema:
                ema_helper = EMA(mu=self.config.model.ema_rate)
                ema_helper.register(model)
            else:
                ema_helper = None

            if config.diffusion.apply_aux_cls:
                if hasattr(config.diffusion, "trained_aux_cls_ckpt_path") and config.diffusion.trained_aux_cls_ckpt_path:
                    aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                                        config.diffusion.trained_aux_cls_ckpt_name),
                                            map_location=self.device)
                    self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
                    self.cond_pred_model.eval()
                elif hasattr(config.diffusion, "trained_aux_cls_log_path"):
                    aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_log_path, "aux_ckpt.pth"),
                                            map_location=self.device)
                    self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
                    self.cond_pred_model.eval()
                else:  # pre-train the guidance auxiliary classifier
                    assert config.diffusion.aux_cls.pre_train
                    self.cond_pred_model.train()
                    pretrain_start_time = time.time()
                    for epoch in range(config.diffusion.aux_cls.n_pretrain_epochs):
                        for feature_label_set in train_loader:
                            if config.data.dataset == "gaussian_mixture":
                                x_batch, y_one_hot_batch, y_logits_batch, y_labels_batch = feature_label_set
                            # [新增] LDL 分支：直接读取，不做 One-hot 转换
                            elif config.data.dataset in self.ldl_datasets:
                                x_batch, y_labels_batch = feature_label_set
                                    # 这一步很关键：直接把真实的分布标签赋值给 y_one_hot_batch
                                y_one_hot_batch = y_labels_batch.to(self.device)
                                y_logits_batch = None # LDL 不需要这个
                            else:
                                x_batch, y_labels_batch = feature_label_set
                                y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch,
                                                                                                    config)
                            aux_loss = self.nonlinear_guidance_model_train_step(x_batch.to(self.device),
                                                                                y_one_hot_batch.to(self.device),
                                                                                aux_optimizer)
                        if epoch % config.diffusion.aux_cls.logging_interval == 0:
                            logging.info(
                                f"epoch: {epoch}, guidance auxiliary classifier pre-training loss: {aux_loss}"
                            )
                    pretrain_end_time = time.time()
                    logging.info("\nPre-training of guidance auxiliary classifier took {:.4f} minutes.\n".format(
                        (pretrain_end_time - pretrain_start_time) / 60))
                    # save auxiliary model after pre-training
                    aux_states = [
                        self.cond_pred_model.state_dict(),
                        aux_optimizer.state_dict(),
                    ]
                    torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
                # report accuracy on both training and test set for the pre-trained auxiliary classifier
                if config.data.dataset not in self.ldl_datasets:
                    y_acc_aux_model = self.evaluate_guidance_model(train_loader)
                    logging.info("\nAfter pre-training, accuracy on training set: {:.8f}.".format(y_acc_aux_model))
                    y_acc_aux_model = self.evaluate_guidance_model(test_loader)
                    logging.info("\nAfter pre-training, accuracy on test set: {:.8f}.\n".format(y_acc_aux_model))
                else:
                    logging.info("\n[LDL Task] Pre-training finished. Skipping accuracy report.\n")

            if not self.args.train_guidance_only:
                start_epoch, step = 0, 0
                if self.args.resume_training:
                    states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                                        map_location=self.device)
                    model.load_state_dict(states[0])

                    states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                    optimizer.load_state_dict(states[1])
                    start_epoch = states[2]
                    step = states[3]
                    if self.config.model.ema:
                        ema_helper.load_state_dict(states[4])
                    # load auxiliary model
                    has_ckpt = hasattr(config.diffusion, "trained_aux_cls_ckpt_path") and config.diffusion.trained_aux_cls_ckpt_path
                    has_log = hasattr(config.diffusion, "trained_aux_cls_log_path") and config.diffusion.trained_aux_cls_log_path
                    
                    if config.diffusion.apply_aux_cls and (not has_ckpt) and (not has_log):
                        aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
                                                map_location=self.device)
                        self.cond_pred_model.load_state_dict(aux_states[0])
                        aux_optimizer.load_state_dict(aux_states[1])

                max_accuracy = 0.0
                best_avg_imp = -float('inf')
                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    logging.info("Prior distribution at timestep T has a mean of 0.")
                if args.add_ce_loss:
                    logging.info("Apply cross entropy as an auxiliary loss during training.")
                for epoch in range(start_epoch, self.config.training.n_epochs):
                    data_start = time.time()
                    data_time = 0
                    for i, feature_label_set in enumerate(train_loader):
                        if config.data.dataset == "gaussian_mixture":
                            x_batch, y_one_hot_batch, y_logits_batch, y_labels_batch = feature_label_set
                        elif config.data.dataset in self.ldl_datasets:
                            x_batch, y_labels_batch = feature_label_set
                            y_one_hot_batch = y_labels_batch.to(self.device) # 分布本身就是目标
                            y_logits_batch = None # 不需要 Logits
                        else:
                            x_batch, y_labels_batch = feature_label_set
                            y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch, config)
                            # y_labels_batch = y_labels_batch.reshape(-1, 1)
                        if config.optim.lr_schedule:
                            adjust_learning_rate(optimizer, i / len(train_loader) + epoch, config)
                        n = x_batch.size(0)
                        # record unflattened x as input to guidance aux classifier
                        x_unflat_batch = x_batch.to(self.device)
                        if config.data.dataset == "toy" or config.model.arch in ["simple", "linear"]:
                            x_batch = torch.flatten(x_batch, 1)
                        data_time += time.time() - data_start
                        model.train()
                        self.cond_pred_model.eval()
                        step += 1

                        # antithetic sampling
                        t = torch.randint(
                            low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                        ).to(self.device)
                        t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                        # noise estimation loss
                        x_batch = x_batch.to(self.device)
                        # y_0_batch = y_logits_batch.to(self.device)
                        if config.data.dataset in self.ldl_datasets:
                            y_0_hat_batch = self.compute_guiding_prediction(x_unflat_batch)
                        else:
                            y_0_hat_batch = self.compute_guiding_prediction(x_unflat_batch).softmax(dim=1)
                        y_T_mean = y_0_hat_batch
                        if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                            y_T_mean = torch.zeros(y_0_hat_batch.shape).to(y_0_hat_batch.device)
                        y_0_batch = y_one_hot_batch.to(self.device)
                        e = torch.randn_like(y_0_batch).to(y_0_batch.device)
                        y_t_batch = q_sample(y_0_batch, y_T_mean,
                                            self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
                        # output = model(x_batch, y_t_batch, t, y_T_mean)
                        output = model(x_batch, y_t_batch, t, y_0_hat_batch)
                        loss = (e - output).square().mean()  # use the same noise sample e during training to compute loss

                        # cross-entropy for y_0 reparameterization
                        loss0 = torch.tensor([0])
                        if args.add_ce_loss:
                            y_0_reparam_batch = y_0_reparam(model, x_batch, y_t_batch, y_0_hat_batch, y_T_mean, t,
                                                            self.one_minus_alphas_bar_sqrt)
                            raw_prob_batch = -(y_0_reparam_batch - 1) ** 2
                            loss0 = criterion(raw_prob_batch, y_labels_batch.to(self.device))
                            loss += config.training.lambda_ce * loss0

                        if not tb_logger is None:
                            tb_logger.add_scalar("loss", loss, global_step=step)

                        if step % self.config.training.logging_freq == 0 or step == 1:
                            logging.info(
                                (
                                        f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, "
                                        f"Noise Estimation loss: {loss.item()}, " +
                                        f"data time: {data_time / (i + 1)}"
                                )
                            )

                        # optimize diffusion model that predicts eps_theta
                        optimizer.zero_grad()
                        loss.backward()
                        try:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), config.optim.grad_clip
                            )
                        except Exception:
                            pass
                        optimizer.step()
                        if self.config.model.ema:
                            ema_helper.update(model)

                        # joint train aux classifier along with diffusion model
                        if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                            self.cond_pred_model.train()
                            aux_loss = self.nonlinear_guidance_model_train_step(x_unflat_batch, y_one_hot_batch,
                                                                                aux_optimizer)
                            if step % self.config.training.logging_freq == 0 or step == 1:
                                logging.info(
                                    f"meanwhile, guidance auxiliary classifier joint-training loss: {aux_loss}"
                                )

                        # save diffusion model
                        if step % self.config.training.snapshot_freq == 0 or step == 1:
                            states = [
                                model.state_dict(),
                                optimizer.state_dict(),
                                epoch,
                                step,
                            ]
                            if self.config.model.ema:
                                states.append(ema_helper.state_dict())

                            if step > 1:  # skip saving the initial ckpt
                                torch.save(
                                    states,
                                    os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                                )
                            # save current states
                            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                            # save auxiliary model
                            if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                                aux_states = [
                                    self.cond_pred_model.state_dict(),
                                    aux_optimizer.state_dict(),
                                ]
                                if step > 1:  # skip saving the initial ckpt
                                    torch.save(
                                        aux_states,
                                        os.path.join(self.args.log_path, "aux_ckpt_{}.pth".format(step)),
                                    )
                                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

                        data_start = time.time()

                    logging.info(
                        (f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, Noise Estimation loss: {loss.item()}, " +
                        f"data time: {data_time / (i + 1)}")
                    )

                    # Evaluate
                    if epoch % self.config.training.validation_freq == 0 \
                            or epoch + 1 == self.config.training.n_epochs:
                        if config.data.dataset in self.ldl_datasets:
                            # 验证阶段只跑 1 次，为了快
                            current_scores = self.test_ldl_task(model, test_loader, n_repeat=1)
                            current_imp = calc_avg_improvement(current_scores, self.sota_values)
                            
                            logging.info(f"Ep: {epoch}, AvgImp: {current_imp:.2%}, KL: {current_scores[3]:.4f}")
                            
                            if current_imp > best_avg_imp:
                                best_avg_imp = current_imp
                        elif config.data.dataset == "toy":
                            with torch.no_grad():
                                model.eval()
                                label_vec = nn.functional.one_hot(test_dataset[:][1]).float().to(self.device)
                                # prior mean at timestep T
                                test_y_0_hat = self.compute_guiding_prediction(
                                    test_dataset[:][0].to(self.device)).softmax(dim=1)
                                y_T_mean = test_y_0_hat
                                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                                    y_T_mean = torch.zeros(test_y_0_hat.shape).to(test_y_0_hat.device)
                                if epoch == start_epoch:
                                    fig, axs = plt.subplots(1, self.num_figs,
                                                            figsize=(self.num_figs * 8.5, 8.5), clear=True)
                                    for i in range(self.num_figs - 1):
                                        cur_y = q_sample(label_vec.cpu(), y_T_mean.cpu(),
                                                        self.alphas_bar_sqrt.cpu(),
                                                        self.one_minus_alphas_bar_sqrt.cpu(),
                                                        torch.tensor([i * self.vis_step])).detach().cpu()
                                        axs[i].scatter(cur_y[:, 0], cur_y[:, 1], s=10, c=test_dataset[:][1]);
                                        axs[i].set_title('$q(\mathbf{y}_{' + str(i * self.vis_step) + '})$', fontsize=25)
                                    cur_y = q_sample(label_vec.cpu(), y_T_mean.cpu(),
                                                    self.alphas_bar_sqrt.cpu(),
                                                    self.one_minus_alphas_bar_sqrt.cpu(),
                                                    torch.tensor([self.num_timesteps - 1])).detach().cpu()
                                    axs[self.num_figs - 1].scatter(cur_y[:, 0], cur_y[:, 1], s=10, c=test_dataset[:][1]);
                                    axs[self.num_figs - 1].set_title(
                                        '$q(\mathbf{y}_{' + str(self.num_timesteps - 1) + '})$', fontsize=25)
                                    if not tb_logger is None:
                                        tb_logger.add_figure('data', fig, step)
                                y_seq = p_sample_loop(model, test_dataset[:][0].to(self.device),
                                                    test_y_0_hat, y_T_mean,
                                                    self.num_timesteps, self.alphas, self.one_minus_alphas_bar_sqrt,
                                                    only_last_sample=False)
                                fig, axs = plt.subplots(1, self.num_figs,
                                                        figsize=(self.num_figs * 8.5, 8.5), clear=True)
                                cur_y = y_seq[0].detach().cpu()
                                axs[self.num_figs - 1].scatter(cur_y[:, 0], cur_y[:, 1], s=10, c=test_dataset[:][1]);
                                axs[self.num_figs - 1].set_title('$p({y}_\mathbf{prior})$', fontsize=25)
                                for i in range(self.num_figs - 1):
                                    cur_y = y_seq[self.num_timesteps - i * self.vis_step - 1].detach().cpu()
                                    axs[i].scatter(cur_y[:, 0], cur_y[:, 1], s=10, c=test_dataset[:][1]);
                                    axs[i].set_title('$p(\mathbf{x}_{' + str(self.vis_step * i) + '})$', fontsize=25)
                                acc_avg = accuracy(y_seq[-1].detach().cpu(), test_dataset[:][1].cpu())[0]
                                logging.info(
                                    (f"epoch: {epoch}, step: {step}, Average accuracy: {acc_avg}%")
                                )
                                if not tb_logger is None:
                                    tb_logger.add_figure('samples', fig, step)
                                    tb_logger.add_scalar('accuracy', acc_avg.item(), global_step=step)
                                fig.savefig(
                                    os.path.join(args.im_path, 'samples_T{}_{}.pdf'.format(self.num_timesteps, step)))
                                plt.close()
                        else:
                            model.eval()
                            self.cond_pred_model.eval()
                            acc_avg = 0.
                            for test_batch_idx, (images, target) in enumerate(test_loader):
                                images_unflat = images.to(self.device)
                                if config.data.dataset == "toy" \
                                        or config.model.arch == "simple" \
                                        or config.model.arch == "linear":
                                    images = torch.flatten(images, 1)
                                images = images.to(self.device)
                                target = target.to(self.device)
                                # target_vec = nn.functional.one_hot(target).float().to(self.device)
                                with torch.no_grad():
                                    target_pred = self.compute_guiding_prediction(images_unflat).softmax(dim=1)
                                    # prior mean at timestep T
                                    y_T_mean = target_pred
                                    if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                                        y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)
                                    if not config.diffusion.noise_prior:  # apply f_phi(x) instead of 0 as prior mean
                                        target_pred = self.compute_guiding_prediction(images_unflat).softmax(dim=1)
                                    label_t_0 = p_sample_loop(model, images, target_pred, y_T_mean,
                                                            self.num_timesteps, self.alphas,
                                                            self.one_minus_alphas_bar_sqrt,
                                                            only_last_sample=True)
                                    acc_avg += accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
                            acc_avg /= (test_batch_idx + 1)
                            if acc_avg > max_accuracy:
                                logging.info("Update best accuracy at Epoch {}.".format(epoch))
                                torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))
                            max_accuracy = max(max_accuracy, acc_avg)
                            if not tb_logger is None:
                                tb_logger.add_scalar('accuracy', acc_avg, global_step=step)
                            logging.info(
                                (
                                        f"epoch: {epoch}, step: {step}, " +
                                        f"Average accuracy: {acc_avg}, " +
                                        f"Max accuracy: {max_accuracy:.2f}%"
                                )
                            )
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())
                torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                # save auxiliary model after training is finished
                if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                    aux_states = [
                        self.cond_pred_model.state_dict(),
                        aux_optimizer.state_dict(),
                    ]
                    torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
                    # report training set accuracy if applied joint training
                    if config.data.dataset not in self.ldl_datasets:
                        y_acc_aux_model = self.evaluate_guidance_model(train_loader)
                        logging.info("After joint-training, acc: {:.8f}.".format(y_acc_aux_model))
                        y_acc_aux_model = self.evaluate_guidance_model(test_loader)
                        logging.info("After joint-training, test acc: {:.8f}.".format(y_acc_aux_model))
                final_ensemble_imp = best_avg_imp 
            logging.info(f"Fold {fold+1} Finished. Best Imp: {final_ensemble_imp:.2%}")
            fold_imps.append(final_ensemble_imp)
        avg_imp = np.mean(fold_imps)
        print(f"[SEARCH_RESULT_START]")
        print(f"AVG_IMP: {avg_imp}")
        print(f"[SEARCH_RESULT_END]")
        return avg_imp
    def train(self):
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        data_object, train_dataset, test_dataset = get_dataset(args, config)
        
        # ================= [新增] 详细的数据维度打印逻辑 =================
        # 获取维度信息 (优先从 dataset 属性读取，否则用 config 默认值)
        real_data_dim = getattr(train_dataset, 'feature_dim', config.model.data_dim)
        real_num_classes = getattr(train_dataset, 'label_dim', config.data.num_classes)
        train_size = len(train_dataset)
        test_size = len(test_dataset)

        # 强制更新 Config，确保模型使用真实的数据维度
        config.model.data_dim = real_data_dim
        config.data.num_classes = real_num_classes
        
        # 组装要打印的报告信息
        data_report = (
            f"\n{'='*20} DATASET INTEGRITY CHECK {'='*20}\n"
            f"Dataset      : {config.data.dataset}\n"
            f"Data Source  : .npy feature file\n"
            f"--------------------------------------------\n"
            f"Train Samples: {train_size}\n"
            f"Test Samples : {test_size}\n"
            f"Feature Dim  : {real_data_dim} (Input)\n"
            f"Label Dim    : {real_num_classes} (Output)\n"
            f"{'='*63}\n"
        )
        # 打印到日志文件开头
        logging.info(data_report)
        
        # 重新初始化引导模型 (MLP)，确保它匹配新的维度
        if config.diffusion.apply_aux_cls and config.data.dataset in self.ldl_datasets:
            self.cond_pred_model = nn.Sequential(
                nn.Linear(config.model.data_dim, config.model.hidden_dim),
                nn.BatchNorm1d(config.model.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.model.hidden_dim, config.model.hidden_dim),
                nn.BatchNorm1d(config.model.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.model.hidden_dim, config.data.num_classes)
            ).to(self.device)
            logging.info(f"Guidance Model (MLP) Re-initialized: [{real_data_dim}] -> [{config.model.hidden_dim}] -> [{real_num_classes}]")
        # ================================================================
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        model = model.to(self.device)
        if self.config.data.dataset not in self.ldl_datasets:
            # 原版逻辑 (CIFAR/MNIST 等) 保留
            y_acc_aux_model = self.evaluate_guidance_model(test_loader)
            logging.info("\nBefore training, the guidance classifier accuracy on the test set is {:.8f}.\n\n".format(
                y_acc_aux_model))
        else:
            # LDL 逻辑：跳过计算，打印一条提示即可
            logging.info("\n[LDL Task] Skipping initial accuracy check (Metric incompatibility).\n")

        optimizer = get_optimizer(self.config.optim, model.parameters())
        criterion = nn.CrossEntropyLoss()
        brier_score = nn.MSELoss()

        # apply an auxiliary optimizer for the guidance classifier
        if config.diffusion.apply_aux_cls:
            aux_optimizer = get_optimizer(self.config.aux_optim,
                                          self.cond_pred_model.parameters())

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path") and config.diffusion.trained_aux_cls_ckpt_path:
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                                     config.diffusion.trained_aux_cls_ckpt_name),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
                self.cond_pred_model.eval()
            elif hasattr(config.diffusion, "trained_aux_cls_log_path"):
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_log_path, "aux_ckpt.pth"),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
                self.cond_pred_model.eval()
            else:  # pre-train the guidance auxiliary classifier
                assert config.diffusion.aux_cls.pre_train
                self.cond_pred_model.train()
                pretrain_start_time = time.time()
                for epoch in range(config.diffusion.aux_cls.n_pretrain_epochs):
                    for feature_label_set in train_loader:
                        if config.data.dataset == "gaussian_mixture":
                            x_batch, y_one_hot_batch, y_logits_batch, y_labels_batch = feature_label_set
                        # [新增] LDL 分支：直接读取，不做 One-hot 转换
                        elif config.data.dataset in self.ldl_datasets:
                            x_batch, y_labels_batch = feature_label_set
                                # 这一步很关键：直接把真实的分布标签赋值给 y_one_hot_batch
                            y_one_hot_batch = y_labels_batch.to(self.device)
                            y_logits_batch = None # LDL 不需要这个
                        else:
                            x_batch, y_labels_batch = feature_label_set
                            y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch,
                                                                                                  config)
                        aux_loss = self.nonlinear_guidance_model_train_step(x_batch.to(self.device),
                                                                            y_one_hot_batch.to(self.device),
                                                                            aux_optimizer)
                    if epoch % config.diffusion.aux_cls.logging_interval == 0:
                        logging.info(
                            f"epoch: {epoch}, guidance auxiliary classifier pre-training loss: {aux_loss}"
                        )
                pretrain_end_time = time.time()
                logging.info("\nPre-training of guidance auxiliary classifier took {:.4f} minutes.\n".format(
                    (pretrain_end_time - pretrain_start_time) / 60))
                # save auxiliary model after pre-training
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
            # report accuracy on both training and test set for the pre-trained auxiliary classifier
            if config.data.dataset not in self.ldl_datasets:
                y_acc_aux_model = self.evaluate_guidance_model(train_loader)
                logging.info("\nAfter pre-training, accuracy on training set: {:.8f}.".format(y_acc_aux_model))
                y_acc_aux_model = self.evaluate_guidance_model(test_loader)
                logging.info("\nAfter pre-training, accuracy on test set: {:.8f}.\n".format(y_acc_aux_model))
            else:
                logging.info("\n[LDL Task] Pre-training finished. Skipping accuracy report.\n")

        if not self.args.train_guidance_only:
            start_epoch, step = 0, 0
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                                    map_location=self.device)
                model.load_state_dict(states[0])

                states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                start_epoch = states[2]
                step = states[3]
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[4])
                # load auxiliary model
                has_ckpt = hasattr(config.diffusion, "trained_aux_cls_ckpt_path") and config.diffusion.trained_aux_cls_ckpt_path
                has_log = hasattr(config.diffusion, "trained_aux_cls_log_path") and config.diffusion.trained_aux_cls_log_path
                
                if config.diffusion.apply_aux_cls and (not has_ckpt) and (not has_log):
                    aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
                                            map_location=self.device)
                    self.cond_pred_model.load_state_dict(aux_states[0])
                    aux_optimizer.load_state_dict(aux_states[1])

            max_accuracy = 0.0
            best_avg_imp = -float('inf')
            if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                logging.info("Prior distribution at timestep T has a mean of 0.")
            if args.add_ce_loss:
                logging.info("Apply cross entropy as an auxiliary loss during training.")
            loss_history = []
            for epoch in range(start_epoch, self.config.training.n_epochs):
                data_start = time.time()
                data_time = 0
                for i, feature_label_set in enumerate(train_loader):
                    if config.data.dataset == "gaussian_mixture":
                        x_batch, y_one_hot_batch, y_logits_batch, y_labels_batch = feature_label_set
                    elif config.data.dataset in self.ldl_datasets:
                        x_batch, y_labels_batch = feature_label_set
                        y_one_hot_batch = y_labels_batch.to(self.device) # 分布本身就是目标
                        y_logits_batch = None # 不需要 Logits
                    else:
                        x_batch, y_labels_batch = feature_label_set
                        y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch, config)
                        # y_labels_batch = y_labels_batch.reshape(-1, 1)
                    if config.optim.lr_schedule:
                        adjust_learning_rate(optimizer, i / len(train_loader) + epoch, config)
                    n = x_batch.size(0)
                    # record unflattened x as input to guidance aux classifier
                    x_unflat_batch = x_batch.to(self.device)
                    if config.data.dataset == "toy" or config.model.arch in ["simple", "linear"]:
                        x_batch = torch.flatten(x_batch, 1)
                    data_time += time.time() - data_start
                    model.train()
                    self.cond_pred_model.eval()
                    step += 1

                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)
                    t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                    # noise estimation loss
                    x_batch = x_batch.to(self.device)
                    # y_0_batch = y_logits_batch.to(self.device)
                    if config.data.dataset in self.ldl_datasets:
                        y_0_hat_batch = self.compute_guiding_prediction(x_unflat_batch)
                    else:
                        y_0_hat_batch = self.compute_guiding_prediction(x_unflat_batch).softmax(dim=1)
                    y_T_mean = y_0_hat_batch
                    if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                        y_T_mean = torch.zeros(y_0_hat_batch.shape).to(y_0_hat_batch.device)
                    y_0_batch = y_one_hot_batch.to(self.device)
                    e = torch.randn_like(y_0_batch).to(y_0_batch.device)
                    y_t_batch = q_sample(y_0_batch, y_T_mean,
                                         self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
                    # output = model(x_batch, y_t_batch, t, y_T_mean)
                    output = model(x_batch, y_t_batch, t, y_0_hat_batch)
                    loss = (e - output).square().mean()  # use the same noise sample e during training to compute loss

                    # cross-entropy for y_0 reparameterization
                    loss0 = torch.tensor([0])
                    if args.add_ce_loss:
                        y_0_reparam_batch = y_0_reparam(model, x_batch, y_t_batch, y_0_hat_batch, y_T_mean, t,
                                                        self.one_minus_alphas_bar_sqrt)
                        raw_prob_batch = -(y_0_reparam_batch - 1) ** 2
                        loss0 = criterion(raw_prob_batch, y_labels_batch.to(self.device))
                        loss += config.training.lambda_ce * loss0

                    if not tb_logger is None:
                        tb_logger.add_scalar("loss", loss, global_step=step)

                    if step % self.config.training.logging_freq == 0 or step == 1:
                        logging.info(
                            (
                                    f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, "
                                    f"Noise Estimation loss: {loss.item()}, " +
                                    f"data time: {data_time / (i + 1)}"
                            )
                        )

                    # optimize diffusion model that predicts eps_theta
                    optimizer.zero_grad()
                    loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()
                    if self.config.model.ema:
                        ema_helper.update(model)

                    # joint train aux classifier along with diffusion model
                    if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                        self.cond_pred_model.train()
                        aux_loss = self.nonlinear_guidance_model_train_step(x_unflat_batch, y_one_hot_batch,
                                                                            aux_optimizer)
                        if step % self.config.training.logging_freq == 0 or step == 1:
                            logging.info(
                                f"meanwhile, guidance auxiliary classifier joint-training loss: {aux_loss}"
                            )

                    # save diffusion model
                    if step % self.config.training.snapshot_freq == 0 or step == 1:
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        if step > 1:  # skip saving the initial ckpt
                            torch.save(
                                states,
                                os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                            )
                        # save current states
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                        # save auxiliary model
                        if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                            aux_states = [
                                self.cond_pred_model.state_dict(),
                                aux_optimizer.state_dict(),
                            ]
                            if step > 1:  # skip saving the initial ckpt
                                torch.save(
                                    aux_states,
                                    os.path.join(self.args.log_path, "aux_ckpt_{}.pth".format(step)),
                                )
                            torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

                    data_start = time.time()
                loss_history.append(loss.item())
                logging.info(
                    (f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, Noise Estimation loss: {loss.item()}, " +
                     f"data time: {data_time / (i + 1)}")
                )

                # Evaluate
                if epoch % self.config.training.validation_freq == 0 \
                        or epoch + 1 == self.config.training.n_epochs:
                    if config.data.dataset in self.ldl_datasets:
                        # 验证阶段只跑 1 次，为了快
                        current_scores = self.test_ldl_task(model, test_loader, n_repeat=1)
                        current_imp = calc_avg_improvement(current_scores, self.sota_values)
                        
                        logging.info(f"Ep: {epoch}, AvgImp: {current_imp:.2%}, KL: {current_scores[3]:.4f}")
                        
                        if current_imp > best_avg_imp:
                            best_avg_imp = current_imp
                            # 构建保存路径: model/数据集/run_X/
                            run_idx = getattr(config.data, "run_idx", 0)
                            save_dir = os.path.join(self.model_save_root, config.data.dataset, f'run_{run_idx}')
                            os.makedirs(save_dir, exist_ok=True)
                            
                            # 保存 Best 模型
                            states = [model.state_dict(), optimizer.state_dict(), epoch, step]
                            torch.save(states, os.path.join(save_dir, "ckpt_best.pth"))
                            
                            # 保存辅助模型
                            if config.diffusion.apply_aux_cls:
                                aux_states = [self.cond_pred_model.state_dict(), aux_optimizer.state_dict()]
                                torch.save(aux_states, os.path.join(save_dir, "aux_best.pth"))
                                
                            logging.info(f"💾 Saved Best Model to: {save_dir}")
                    elif config.data.dataset == "toy":
                        with torch.no_grad():
                            model.eval()
                            label_vec = nn.functional.one_hot(test_dataset[:][1]).float().to(self.device)
                            # prior mean at timestep T
                            test_y_0_hat = self.compute_guiding_prediction(
                                test_dataset[:][0].to(self.device)).softmax(dim=1)
                            y_T_mean = test_y_0_hat
                            if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                                y_T_mean = torch.zeros(test_y_0_hat.shape).to(test_y_0_hat.device)
                            if epoch == start_epoch:
                                fig, axs = plt.subplots(1, self.num_figs,
                                                        figsize=(self.num_figs * 8.5, 8.5), clear=True)
                                for i in range(self.num_figs - 1):
                                    cur_y = q_sample(label_vec.cpu(), y_T_mean.cpu(),
                                                     self.alphas_bar_sqrt.cpu(),
                                                     self.one_minus_alphas_bar_sqrt.cpu(),
                                                     torch.tensor([i * self.vis_step])).detach().cpu()
                                    axs[i].scatter(cur_y[:, 0], cur_y[:, 1], s=10, c=test_dataset[:][1]);
                                    axs[i].set_title('$q(\mathbf{y}_{' + str(i * self.vis_step) + '})$', fontsize=25)
                                cur_y = q_sample(label_vec.cpu(), y_T_mean.cpu(),
                                                 self.alphas_bar_sqrt.cpu(),
                                                 self.one_minus_alphas_bar_sqrt.cpu(),
                                                 torch.tensor([self.num_timesteps - 1])).detach().cpu()
                                axs[self.num_figs - 1].scatter(cur_y[:, 0], cur_y[:, 1], s=10, c=test_dataset[:][1]);
                                axs[self.num_figs - 1].set_title(
                                    '$q(\mathbf{y}_{' + str(self.num_timesteps - 1) + '})$', fontsize=25)
                                if not tb_logger is None:
                                    tb_logger.add_figure('data', fig, step)
                            y_seq = p_sample_loop(model, test_dataset[:][0].to(self.device),
                                                  test_y_0_hat, y_T_mean,
                                                  self.num_timesteps, self.alphas, self.one_minus_alphas_bar_sqrt,
                                                  only_last_sample=False)
                            fig, axs = plt.subplots(1, self.num_figs,
                                                    figsize=(self.num_figs * 8.5, 8.5), clear=True)
                            cur_y = y_seq[0].detach().cpu()
                            axs[self.num_figs - 1].scatter(cur_y[:, 0], cur_y[:, 1], s=10, c=test_dataset[:][1]);
                            axs[self.num_figs - 1].set_title('$p({y}_\mathbf{prior})$', fontsize=25)
                            for i in range(self.num_figs - 1):
                                cur_y = y_seq[self.num_timesteps - i * self.vis_step - 1].detach().cpu()
                                axs[i].scatter(cur_y[:, 0], cur_y[:, 1], s=10, c=test_dataset[:][1]);
                                axs[i].set_title('$p(\mathbf{x}_{' + str(self.vis_step * i) + '})$', fontsize=25)
                            acc_avg = accuracy(y_seq[-1].detach().cpu(), test_dataset[:][1].cpu())[0]
                            logging.info(
                                (f"epoch: {epoch}, step: {step}, Average accuracy: {acc_avg}%")
                            )
                            if not tb_logger is None:
                                tb_logger.add_figure('samples', fig, step)
                                tb_logger.add_scalar('accuracy', acc_avg.item(), global_step=step)
                            fig.savefig(
                                os.path.join(args.im_path, 'samples_T{}_{}.pdf'.format(self.num_timesteps, step)))
                            plt.close()
                    else:
                        model.eval()
                        self.cond_pred_model.eval()
                        acc_avg = 0.
                        for test_batch_idx, (images, target) in enumerate(test_loader):
                            images_unflat = images.to(self.device)
                            if config.data.dataset == "toy" \
                                    or config.model.arch == "simple" \
                                    or config.model.arch == "linear":
                                images = torch.flatten(images, 1)
                            images = images.to(self.device)
                            target = target.to(self.device)
                            # target_vec = nn.functional.one_hot(target).float().to(self.device)
                            with torch.no_grad():
                                target_pred = self.compute_guiding_prediction(images_unflat).softmax(dim=1)
                                # prior mean at timestep T
                                y_T_mean = target_pred
                                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                                    y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)
                                if not config.diffusion.noise_prior:  # apply f_phi(x) instead of 0 as prior mean
                                    target_pred = self.compute_guiding_prediction(images_unflat).softmax(dim=1)
                                label_t_0 = p_sample_loop(model, images, target_pred, y_T_mean,
                                                          self.num_timesteps, self.alphas,
                                                          self.one_minus_alphas_bar_sqrt,
                                                          only_last_sample=True)
                                acc_avg += accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
                        acc_avg /= (test_batch_idx + 1)
                        if acc_avg > max_accuracy:
                            logging.info("Update best accuracy at Epoch {}.".format(epoch))
                            torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))
                        max_accuracy = max(max_accuracy, acc_avg)
                        if not tb_logger is None:
                            tb_logger.add_scalar('accuracy', acc_avg, global_step=step)
                        logging.info(
                            (
                                    f"epoch: {epoch}, step: {step}, " +
                                    f"Average accuracy: {acc_avg}, " +
                                    f"Max accuracy: {max_accuracy:.2f}%"
                            )
                        )

            # save the model after training is finished
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            # save auxiliary model after training is finished
            if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
                # report training set accuracy if applied joint training
                if config.data.dataset not in self.ldl_datasets:
                    y_acc_aux_model = self.evaluate_guidance_model(train_loader)
                    logging.info("After joint-training, acc: {:.8f}.".format(y_acc_aux_model))
                    y_acc_aux_model = self.evaluate_guidance_model(test_loader)
                    logging.info("After joint-training, test acc: {:.8f}.".format(y_acc_aux_model))
            final_ensemble_imp = best_avg_imp 
            try:
                curve_dir = "curve"
                os.makedirs(curve_dir, exist_ok=True)
                
                # 构建文件名：数据集_RunID_时间戳.png
                run_idx = getattr(config.data, "run_idx", 0)
                timestamp = int(time.time())
                plot_path = os.path.join(curve_dir, f"{config.data.dataset}_run{run_idx}_loss.png")
                
                plt.figure(figsize=(10, 6))
                plt.plot(loss_history, label='Training Loss')
                plt.title(f'Training Loss Curve - {config.data.dataset} (Run {run_idx})')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.savefig(plot_path)
                plt.close() # 关闭画布释放内存
                logging.info(f" Training Loss curve saved to: {plot_path}")
            except Exception as e:
                logging.warning(f"Failed to plot loss curve: {e}")
            # ====================================================================
            if config.data.dataset in self.ldl_datasets:
                logging.info("\n>>> Training Finished. Loading Best Model for Final Ensemble Test...")
                
                run_idx = getattr(config.data, "run_idx", 0)
                save_dir = os.path.join(self.model_save_root, config.data.dataset, f'run_{run_idx}')
                best_ckpt_path = os.path.join(save_dir, "ckpt_best.pth")
                
                if os.path.exists(best_ckpt_path):
                    states = torch.load(best_ckpt_path, map_location=self.device)
                    model.load_state_dict(states[0])
                    logging.info("Loaded diffusion model.")
                    
                    if config.diffusion.apply_aux_cls:
                        aux_path = os.path.join(save_dir, "aux_best.pth")
                        if os.path.exists(aux_path):
                            aux_states = torch.load(aux_path, map_location=self.device)
                            self.cond_pred_model.load_state_dict(aux_states[0])
                    final_scores = self.test_ldl_task(model, test_loader, n_repeat=10)
                    final_ensemble_imp = calc_avg_improvement(final_scores, self.sota_values)
                else:
                    logging.error(f"Best checkpoint not found at {best_ckpt_path}, skipping final test.")
                    
        # 4. 打印对比日志
        logging.info("\n" + "="*50)
        logging.info(f"🔹 Best Single-Shot (Training Phase) : {best_avg_imp:.4f}") 
        logging.info(f"🔹 Final Ensemble  (Testing Phase)  : {final_ensemble_imp:.4f}") 
        logging.info("=" * 50 + "\n")

        # 5. 返回正确的值给 main.py
        return final_scores
        
    def test_image_task(self):
        """
        Evaluate model performance on image classification tasks.
        """

        def compute_val_before_softmax(gen_y_0_val):
            """
            Compute raw value before applying Softmax function to obtain prediction in probability scale.
            Corresponding to the part inside the Softmax function of Eq. (10) in paper.
            """
            # TODO: add other ways of computing such raw prob value
            raw_prob_val = -(gen_y_0_val - 1) ** 2
            return raw_prob_val

        #####################################################################################################
        ########################## local functions within the class function scope ##########################
        def compute_and_store_cls_metrics(config, y_labels_batch, generated_y, batch_size, num_t):
            """
            generated_y: y_t in logit, has a shape of (batch_size x n_samples, n_classes)

            For each instance, compute probabilities of prediction of each label, majority voted label and
                its correctness, as well as accuracy of all samples given the instance.
            """
            current_t = self.num_timesteps + 1 - num_t
            gen_y_all_class_raw_probs = generated_y.reshape(batch_size,
                                                            config.testing.n_samples,
                                                            config.data.num_classes).cpu()  # .numpy()
            # compute softmax probabilities of all classes for each sample
            raw_prob_val = compute_val_before_softmax(gen_y_all_class_raw_probs)
            gen_y_all_class_probs = torch.softmax(raw_prob_val / self.tuned_scale_T,
                                                  dim=-1)  # (batch_size, n_samples, n_classes)
            # obtain credible interval of probability predictions in each class for the samples given the same x
            low, high = config.testing.PICP_range
            # use raw predicted probability (right before temperature scaling and Softmax) width
            # to construct prediction interval
            CI_y_pred = raw_prob_val.nanquantile(q=torch.tensor([low / 100, high / 100]),
                                                 dim=1).swapaxes(0, 1)  # (batch_size, 2, n_classes)
            # obtain the predicted label with the largest probability for each sample
            gen_y_labels = torch.argmax(gen_y_all_class_probs, 2, keepdim=True)  # (batch_size, n_samples, 1)
            # convert the predicted label to one-hot format
            gen_y_one_hot = torch.zeros_like(gen_y_all_class_probs).scatter_(
                dim=2, index=gen_y_labels,
                src=torch.ones_like(gen_y_labels.float()))  # (batch_size, n_samples, n_classes)
            # compute proportion of each class as the prediction given the same x
            gen_y_label_probs = gen_y_one_hot.sum(1) / config.testing.n_samples  # (batch_size, n_classes)
            gen_y_all_class_mean_prob = gen_y_all_class_probs.mean(1)  # (batch_size, n_classes)
            # obtain the class being predicted the most given the same x
            gen_y_majority_vote = torch.argmax(gen_y_label_probs, 1, keepdim=True)  # (batch_size, 1)
            # compute the proportion of predictions being the correct label for each x
            gen_y_instance_accuracy = (gen_y_labels == y_labels_batch[:, None]).float().mean(1)  # (batch_size, 1)
            # conduct paired two-sided t-test for the two most predicted classes for each instance
            two_most_probable_classes_idx = gen_y_label_probs.argsort(dim=1, descending=True)[:, :2]
            two_most_probable_classes_idx = torch.repeat_interleave(
                two_most_probable_classes_idx[:, None],
                repeats=config.testing.n_samples, dim=1)  # (batch_size, n_samples, 2)
            gen_y_2_class_probs = torch.gather(gen_y_all_class_probs, dim=2,
                                               index=two_most_probable_classes_idx)  # (batch_size, n_samples, 2)
            # make plots to check normality assumptions (differences btw two most probable classes) for the t-test
            # (check https://www.researchgate.net/post/Paired_t-test_and_normality_test_question)
            if num_t == (self.num_timesteps + 1) and step == 0:
                gen_y_2_class_prob_diff = gen_y_2_class_probs[:, :, 1] \
                                          - gen_y_2_class_probs[:, :, 0]  # (batch_size, n_samples)
                plt.style.use('classic')
                for instance_idx in range(24):
                    fig = sm.qqplot(gen_y_2_class_prob_diff[instance_idx, :],
                                    fit=True, line='45')
                    fig.savefig(os.path.join(args.im_path,
                                             f'qq_plot_instance_{instance_idx}.png'))
                    plt.close()
                plt.style.use('ggplot')
            ttest_pvalues = (ttest_rel(gen_y_2_class_probs[:, :, 0],
                                       gen_y_2_class_probs[:, :, 1],
                                       axis=1, alternative='two-sided')).pvalue  # (batch_size, )
            ttest_reject = (ttest_pvalues < config.testing.ttest_alpha)  # (batch_size, )

            if len(majority_vote_by_batch_list[current_t]) == 0:
                majority_vote_by_batch_list[current_t] = gen_y_majority_vote
            else:
                majority_vote_by_batch_list[current_t] = np.concatenate(
                    [majority_vote_by_batch_list[current_t], gen_y_majority_vote], axis=0)
            if current_t == config.testing.metrics_t:
                gen_y_all_class_probs = gen_y_all_class_probs.reshape(
                    y_labels_batch.shape[0] * config.testing.n_samples, config.data.num_classes)
                if len(label_probs_by_batch_list[current_t]) == 0:
                    all_class_probs_by_batch_list[current_t] = gen_y_all_class_probs
                    label_probs_by_batch_list[current_t] = gen_y_label_probs
                    label_mean_probs_by_batch_list[current_t] = gen_y_all_class_mean_prob
                    instance_accuracy_by_batch_list[current_t] = gen_y_instance_accuracy
                    CI_by_batch_list[current_t] = CI_y_pred
                    ttest_reject_by_batch_list[current_t] = ttest_reject
                else:
                    all_class_probs_by_batch_list[current_t] = np.concatenate(
                        [all_class_probs_by_batch_list[current_t], gen_y_all_class_probs], axis=0)
                    label_probs_by_batch_list[current_t] = np.concatenate(
                        [label_probs_by_batch_list[current_t], gen_y_label_probs], axis=0)
                    label_mean_probs_by_batch_list[current_t] = np.concatenate(
                        [label_mean_probs_by_batch_list[current_t], gen_y_all_class_mean_prob], axis=0)
                    instance_accuracy_by_batch_list[current_t] = np.concatenate(
                        [instance_accuracy_by_batch_list[current_t], gen_y_instance_accuracy], axis=0)
                    CI_by_batch_list[current_t] = np.concatenate(
                        [CI_by_batch_list[current_t], CI_y_pred], axis=0)
                    ttest_reject_by_batch_list[current_t] = np.concatenate(
                        [ttest_reject_by_batch_list[current_t], ttest_reject], axis=0)

        def p_sample_loop_with_eval(model, x, y_0_hat, y_T_mean, n_steps,
                                    alphas, one_minus_alphas_bar_sqrt,
                                    batch_size, config):
            """
            Sample y_{t-1} given y_t on the fly, and evaluate model immediately, to avoid OOM.
            """

            def optional_metric_compute(cur_y, num_t):
                if config.testing.compute_metric_all_steps or \
                        (self.num_timesteps + 1 - num_t == config.testing.metrics_t):
                    compute_and_store_cls_metrics(config, y_labels_batch, cur_y, batch_size, num_t)

            device = next(model.parameters()).device
            z = torch.randn_like(y_T_mean).to(device)  # standard Gaussian
            cur_y = z + y_T_mean  # sampled y_T
            num_t = 1
            optional_metric_compute(cur_y, num_t)
            for i in reversed(range(1, n_steps)):
                y_t = cur_y
                cur_y = p_sample(model, x, y_t, y_0_hat, y_T_mean, i, alphas, one_minus_alphas_bar_sqrt)  # y_{t-1}
                num_t += 1
                optional_metric_compute(cur_y, num_t)
            assert num_t == n_steps
            # obtain y_0 given y_1
            num_t += 1
            y_0 = p_sample_t_1to0(model, x, cur_y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt)
            optional_metric_compute(y_0, num_t)

        def p_sample_loop_only_y_0(model, x, y_0_hat, y_T_mean, n_steps,
                                   alphas, one_minus_alphas_bar_sqrt):
            """
            Only compute y_0 -- no metric evaluation.
            """
            device = next(model.parameters()).device
            z = torch.randn_like(y_T_mean).to(device)  # standard Gaussian
            cur_y = z + y_T_mean  # sampled y_T
            num_t = 1
            for i in reversed(range(1, n_steps)):
                y_t = cur_y
                cur_y = p_sample(model, x, y_t, y_0_hat, y_T_mean, i, alphas, one_minus_alphas_bar_sqrt)  # y_{t-1}
                num_t += 1
            assert num_t == n_steps
            y_0 = p_sample_t_1to0(model, x, cur_y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt)
            return y_0

        def compute_quantile_metrics(config, CI_all_classes, majority_voted_class, true_y_label):
            """
            CI_all_classes: (n_test, 2, n_classes), contains quantiles at (low, high) in config.testing.PICP_range
                for the predicted probabilities of all classes for each test instance.

            majority_voted_class: (n_test, ) contains whether the majority-voted label is correct or not
                for each test instance.

            true_y_label: (n_test, ) contains true label for each test instance.
            """
            # obtain credible interval width by computing high - low
            CI_width_all_classes = torch.tensor(
                CI_all_classes[:, 1] - CI_all_classes[:, 0]).squeeze()  # (n_test, n_classes)
            # predict by the k-th smallest CI width
            for kth_smallest in range(1, config.data.num_classes + 1):
                pred_by_narrowest_CI_width = torch.kthvalue(
                    CI_width_all_classes, k=kth_smallest, dim=1, keepdim=False).indices.numpy()  # (n_test, )
                # logging.info(pred_by_narrowest_CI_width)  #@#
                narrowest_CI_pred_correctness = (pred_by_narrowest_CI_width == true_y_label)  # (n_test, )
                logging.info(("We predict the label by the class with the {}-th narrowest CI width, \n" +
                              "and obtain a test accuracy of {:.4f}% through the entire test set.").format(
                    kth_smallest, np.mean(narrowest_CI_pred_correctness) * 100))
            # check whether the most predicted class is the correct label for each x
            majority_vote_correctness = (majority_voted_class == true_y_label)  # (n_test, )
            # obtain one-hot label
            true_y_one_hot = cast_label_to_one_hot_and_prototype(torch.tensor(true_y_label), config,
                                                                 return_prototype=False)  # (n_test, n_classes)
            # obtain predicted CI width only for the true class
            CI_width_true_class = (CI_width_all_classes * true_y_one_hot).sum(dim=1, keepdim=True)  # (n_test, 1)
            # sanity check
            nan_idx = torch.arange(true_y_label.shape[0])[CI_width_true_class.flatten().isnan()]
            if nan_idx.shape[0] > 0:
                logging.info(("Sanity check: model prediction contains nan " +
                              "for test instances with index {}.").format(nan_idx.numpy()))
            CI_width_true_class_correct_pred = CI_width_true_class[majority_vote_correctness]  # (n_correct, 1)
            CI_width_true_class_incorrect_pred = CI_width_true_class[~majority_vote_correctness]  # (n_incorrect, 1)
            logging.info(("\n\nWe apply the majority-voted class label as our prediction, and achieve {:.4f}% " +
                          "accuracy through the entire test set.\n" +
                          "Out of {} test instances, we made {} correct predictions, with a " +
                          "mean credible interval width of {:.4f} in predicted probability of the true class; \n" +
                          "the remaining {} instances are classified incorrectly, " +
                          "with a mean CI width of {:.4f}.\n").format(
                np.mean(majority_vote_correctness) * 100,
                true_y_label.shape[0],
                majority_vote_correctness.sum(),
                CI_width_true_class_correct_pred.mean().item(),
                true_y_label.shape[0] - majority_vote_correctness.sum(),
                CI_width_true_class_incorrect_pred.mean().item()))

            maj_vote_acc_by_class = []
            CI_w_cor_pred_by_class = []
            CI_w_incor_pred_by_class = []
            # report metrics within each class
            for c in range(config.data.num_classes):
                maj_vote_cor_class_c = majority_vote_correctness[true_y_label == c]  # (n_class_c, 1)
                CI_width_true_class_c = CI_width_true_class[true_y_label == c]  # (n_class_c, 1)
                CI_w_cor_class_c = CI_width_true_class_c[maj_vote_cor_class_c]  # (n_correct_class_c, 1)
                CI_w_incor_class_c = CI_width_true_class_c[~maj_vote_cor_class_c]  # (n_incorrect_class_c, 1)
                logging.info(("\n\tClass {} ({} total instances, {:.4f}% accuracy):" +
                              "\n\t\t{} correct predictions, mean CI width {:.4f}" +
                              "\n\t\t{} incorrect predictions, mean CI width {:.4f}").format(
                    c, maj_vote_cor_class_c.shape[0], np.mean(maj_vote_cor_class_c) * 100,
                    CI_w_cor_class_c.shape[0], CI_w_cor_class_c.mean().item(),
                    CI_w_incor_class_c.shape[0], CI_w_incor_class_c.mean().item()))
                maj_vote_acc_by_class.append(np.mean(maj_vote_cor_class_c))
                CI_w_cor_pred_by_class.append(CI_w_cor_class_c.mean().item())
                CI_w_incor_pred_by_class.append(CI_w_incor_class_c.mean().item())

            return maj_vote_acc_by_class, CI_w_cor_pred_by_class, CI_w_incor_pred_by_class

        def compute_ttest_metrics(config, ttest_reject, majority_voted_class, true_y_label):
            """
            ttest_reject: (n_test, ) contains whether to reject the paired two-sided t-test between
                the two most probable predicted classes for each test instance.

            majority_voted_class: (n_test, ) contains whether the majority-voted label is correct or not
                for each test instance.

            true_y_label: (n_test, ) contains true label for each test instance.
            """
            # check whether the most predicted class is the correct label for each x
            majority_vote_correctness = (majority_voted_class == true_y_label)  # (n_test, )
            # split test instances into correct and incorrect predictions
            ttest_reject_correct_pred = ttest_reject[majority_vote_correctness]  # (n_correct, )
            ttest_reject_incorrect_pred = ttest_reject[~majority_vote_correctness]  # (n_incorrect, )
            logging.info(("\n\nWe apply the majority-voted class label as our prediction, and achieve {:.4f}% " +
                          "accuracy through the entire test set.\n" +
                          "Out of {} test instances, we made {} correct predictions, with a " +
                          "mean rejection rate of {:.4f}% for the paired two-sided t-test; \n" +
                          "the remaining {} instances are classified incorrectly, " +
                          "with a mean rejection rate of {:.4f}%.\n").format(
                np.mean(majority_vote_correctness) * 100,
                true_y_label.shape[0],
                ttest_reject_correct_pred.shape[0],
                ttest_reject_correct_pred.mean().item() * 100,
                ttest_reject_incorrect_pred.shape[0],
                ttest_reject_incorrect_pred.mean().item() * 100))

            # rejection rate by prediction correctness within each class
            maj_vote_acc_by_class = []
            ttest_rej_cor_pred_by_class = []
            ttest_rej_incor_pred_by_class = []
            for c in range(config.data.num_classes):
                maj_vote_cor_class_c = majority_vote_correctness[true_y_label == c]  # (n_class_c, 1)
                ttest_reject_class_c = ttest_reject[true_y_label == c]  # (n_class_c, 1)
                ttest_rej_cor_class_c = ttest_reject_class_c[maj_vote_cor_class_c]  # (n_correct_class_c, 1)
                ttest_rej_incor_class_c = ttest_reject_class_c[~maj_vote_cor_class_c]  # (n_incorrect_class_c, 1)
                logging.info(("\n\tClass {} ({} total instances, {:.4f}% accuracy):" +
                              "\n\t\t{} correct predictions, mean rejection rate {:.4f}%" +
                              "\n\t\t{} incorrect predictions, mean rejection rate {:.4f}%").format(
                    c, maj_vote_cor_class_c.shape[0], np.mean(maj_vote_cor_class_c) * 100,
                    ttest_rej_cor_class_c.shape[0], ttest_rej_cor_class_c.mean().item() * 100,
                    ttest_rej_incor_class_c.shape[0], ttest_rej_incor_class_c.mean().item() * 100))
                maj_vote_acc_by_class.append(np.mean(maj_vote_cor_class_c))
                ttest_rej_cor_pred_by_class.append(ttest_rej_cor_class_c.mean().item())
                ttest_rej_incor_pred_by_class.append(ttest_rej_incor_class_c.mean().item())

            # split test instances into rejected and not-rejected t-tests
            maj_vote_cor_reject = majority_vote_correctness[ttest_reject]  # (n_reject, )
            maj_vote_cor_not_reject = majority_vote_correctness[~ttest_reject]  # (n_not_reject, )
            logging.info(("\n\nFurthermore, among all test instances, " +
                          "we reject {} t-tests, with a " +
                          "mean accuracy of {:.4f}%; \n" +
                          "the remaining {} t-tests are not rejected, " +
                          "with a mean accuracy of {:.4f}%.\n").format(
                maj_vote_cor_reject.shape[0],
                maj_vote_cor_reject.mean().item() * 100,
                maj_vote_cor_not_reject.shape[0],
                maj_vote_cor_not_reject.mean().item() * 100))

            # accuracy by rejection status within each class
            ttest_reject_by_class = []
            maj_vote_cor_reject_by_class = []
            maj_vote_cor_not_reject_by_class = []
            for c in range(config.data.num_classes):
                maj_vote_cor_class_c = majority_vote_correctness[true_y_label == c]  # (n_class_c, 1)
                ttest_reject_class_c = ttest_reject[true_y_label == c]  # (n_class_c, 1)
                maj_vote_cor_rej_class_c = maj_vote_cor_class_c[ttest_reject_class_c]  # (n_rej_class_c, 1)
                maj_vote_cor_not_rej_class_c = maj_vote_cor_class_c[~ttest_reject_class_c]  # (n_not_rej_class_c, 1)
                logging.info(("\n\tClass {} ({} total instances, {:.4f}% rejection rate):" +
                              "\n\t\t{} rejected t-tests, mean accuracy {:.4f}%" +
                              "\n\t\t{} not-rejected t-tests, mean accuracy {:.4f}%").format(
                    c, ttest_reject_class_c.shape[0], np.mean(ttest_reject_class_c) * 100,
                    maj_vote_cor_rej_class_c.shape[0], maj_vote_cor_rej_class_c.mean().item() * 100,
                    maj_vote_cor_not_rej_class_c.shape[0], maj_vote_cor_not_rej_class_c.mean().item() * 100))
                ttest_reject_by_class.append(np.mean(ttest_reject_class_c))
                maj_vote_cor_reject_by_class.append(maj_vote_cor_rej_class_c.mean().item())
                maj_vote_cor_not_reject_by_class.append(maj_vote_cor_not_rej_class_c.mean().item())

            return maj_vote_acc_by_class, ttest_rej_cor_pred_by_class, ttest_rej_incor_pred_by_class, \
                   ttest_reject_by_class, maj_vote_cor_reject_by_class, maj_vote_cor_not_reject_by_class

        #####################################################################################################
        #####################################################################################################

        args = self.args
        config = self.config
        split = args.split
        log_path = os.path.join(self.args.log_path)
        dataset_object, train_dataset, test_dataset = get_dataset(args, config)
        # use test batch size for training set during inference
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        if getattr(self.config.testing, "ckpt_id", None) is None:
            if args.eval_best:
                ckpt_id = 'best'
                states = torch.load(os.path.join(log_path, f"ckpt_{ckpt_id}.pth"),
                                    map_location=self.device)
            else:
                ckpt_id = 'last'
                states = torch.load(os.path.join(log_path, "ckpt.pth"),
                                    map_location=self.device)
        else:
            states = torch.load(os.path.join(log_path, f"ckpt_{self.config.testing.ckpt_id}.pth"),
                                map_location=self.device)
            ckpt_id = self.config.testing.ckpt_id
        logging.info(f"Loading from: {log_path}/ckpt_{ckpt_id}.pth")
        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None

        model.eval()
        if args.sanity_check:
            logging.info("Evaluation function implementation sanity check...")
            config.testing.n_samples = 10
        if args.test_sample_seed >= 0:
            logging.info(f"Manually setting seed {args.test_sample_seed} for test time sampling of class prototype...")
            set_random_seed(args.test_sample_seed)

        # load auxiliary model
        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path") and config.diffusion.trained_aux_cls_ckpt_path:
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                                     config.diffusion.trained_aux_cls_ckpt_name),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
                self.cond_pred_model.eval()
            else:
                aux_cls_path = log_path
                if hasattr(config.diffusion, "trained_aux_cls_log_path"):
                    aux_cls_path = config.diffusion.trained_aux_cls_log_path
                aux_states = torch.load(os.path.join(aux_cls_path, "aux_ckpt.pth"),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
            self.cond_pred_model.eval()
        # report test set RMSE if applied joint training
        y_acc_aux_model = self.evaluate_guidance_model(test_loader)
        logging.info("After training, guidance classifier accuracy on the test set is {:.8f}.".format(
            y_acc_aux_model))

        if config.testing.compute_metric_all_steps:
            logging.info("\nWe compute classification task metrics for all steps.\n")
        else:
            logging.info("\nWe pick samples at timestep t={} to compute evaluation metrics.\n".format(
                config.testing.metrics_t))

        # tune the scaling temperature parameter with training set
        T_description = "default"
        if args.tune_T:
            y_0_one_sample_all = []
            n_tune_T_samples = 5  # config.testing.n_samples; 25; 10
            logging.info("Begin generating {} samples for tuning temperature scaling parameter...".format(
                n_tune_T_samples))
            for idx, feature_label_set in tqdm(enumerate(train_loader)):  # test_loader would give oracle hyperparameter
                x_batch, y_labels_batch = feature_label_set
                x_batch = x_batch.to(self.device)
                # compute y_0_hat as the initial prediction to guide the reverse diffusion process
                y_0_hat_batch = self.compute_guiding_prediction(x_batch).softmax(dim=1)
                if config.data.dataset == "toy" or config.model.arch in ["simple", "linear"]:
                    x_batch = torch.flatten(x_batch, 1)
                batch_size = x_batch.shape[0]
                if len(x_batch.shape) == 2:
                    # x_batch with shape (batch_size, flattened_image_dim)
                    x_tile = (x_batch.repeat(n_tune_T_samples, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1)
                else:
                    # x_batch with shape (batch_size, 3, 32, 32) for CIFAR10 and CIFAR100 dataset
                    x_tile = (x_batch.repeat(n_tune_T_samples, 1, 1, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1)
                y_0_hat_tile = (y_0_hat_batch.repeat(n_tune_T_samples, 1, 1).transpose(0, 1)).flatten(0, 1)
                y_T_mean_tile = y_0_hat_tile
                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    y_T_mean_tile = torch.zeros(y_0_hat_tile.shape).to(self.device)
                minibatch_sample_start = time.time()
                y_0_sample_batch = p_sample_loop_only_y_0(model, x_tile, y_0_hat_tile, y_T_mean_tile,
                                                          self.num_timesteps,
                                                          self.alphas, self.one_minus_alphas_bar_sqrt)  # .cpu().numpy()
                # take the mean of n_tune_T_samples reconstructed y prototypes as the raw predicted y for T tuning
                y_0_sample_batch = y_0_sample_batch.reshape(batch_size,
                                                            n_tune_T_samples,
                                                            config.data.num_classes).mean(1)  # (batch_size, n_classes)
                minibatch_sample_end = time.time()
                logging.info("Minibatch {} sampling took {:.4f} seconds.".format(
                    idx, (minibatch_sample_end - minibatch_sample_start)))
                y_0_one_sample_all.append(y_0_sample_batch.detach())
                # only generate a few batches for sanity checking
                if args.sanity_check:
                    if idx > 2:
                        break
            print(len(y_0_one_sample_all), y_0_one_sample_all[0].shape)

            logging.info("Begin optimizing temperature scaling parameter...")
            scale_T_raw = nn.Parameter(torch.zeros(1))
            scale_T_lr = 0.01
            scale_T_optimizer = torch.optim.Adam([scale_T_raw], lr=scale_T_lr)
            nll_losses = []
            scale_T_n_epochs = 10 if args.sanity_check else 1
            for _ in range(scale_T_n_epochs):
                for idx, feature_label_set in tqdm(enumerate(train_loader)):  # test_loader would give oracle value
                    _, y_labels_batch = feature_label_set
                    y_one_hot_batch, _ = cast_label_to_one_hot_and_prototype(y_labels_batch, config)
                    y_one_hot_batch = y_one_hot_batch.to(self.device)
                    scale_T = nn.functional.softplus(scale_T_raw).to(self.device)
                    y_0_sample_batch = y_0_one_sample_all[idx]
                    raw_p_val = compute_val_before_softmax(y_0_sample_batch)
                    # Eq. (9) of the Calibration paper (Guo et al. 2017)
                    y_0_scaled_batch = (raw_p_val / scale_T).softmax(1)
                    y_0_prob_batch = (y_0_scaled_batch * y_one_hot_batch).sum(1)  # instance likelihood
                    nll_loss = -torch.log(y_0_prob_batch).mean()
                    nll_losses.append(nll_loss.cpu().item())
                    # optimize scaling temperature T parameter
                    scale_T_optimizer.zero_grad()
                    nll_loss.backward()
                    scale_T_optimizer.step()
                    # only tune a few batches for sanity checking
                    if args.sanity_check:
                        if idx > 2:
                            break
                logging.info("NLL of the last mini-batch: {:.8f}".format(nll_losses[-1]))
            self.tuned_scale_T = nn.functional.softplus(scale_T_raw).detach().item()
            T_description = "tuned"
        else:
            self.tuned_scale_T = 1
        logging.info("Apply {} temperature scaling parameter T with a value of {:.4f}".format(
            T_description, self.tuned_scale_T))

        with torch.no_grad():
            true_y_label_by_batch_list = []
            majority_vote_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            all_class_probs_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            label_probs_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            label_mean_probs_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            instance_accuracy_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            CI_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            ttest_reject_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]

            for step, feature_label_set in tqdm(enumerate(test_loader)):
                x_batch, y_labels_batch = feature_label_set
                # compute y_0_hat as the initial prediction to guide the reverse diffusion process
                y_0_hat_batch = self.compute_guiding_prediction(x_batch.to(self.device)).softmax(dim=1)
                true_y_label_by_batch_list.append(y_labels_batch.numpy())
                # # record unflattened x as input to guidance aux classifier
                # x_unflat_batch = x_batch.to(self.device)
                if config.data.dataset == "toy" \
                        or config.model.arch == "simple" \
                        or config.model.arch == "linear":
                    x_batch = torch.flatten(x_batch, 1)
                # y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch, config)
                y_labels_batch = y_labels_batch.reshape(-1, 1)
                batch_size = x_batch.shape[0]
                if len(x_batch.shape) == 2:
                    # x_batch with shape (batch_size, flattened_image_dim)
                    x_tile = (x_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1)
                else:
                    # x_batch with shape (batch_size, 3, 32, 32) for CIFAR10 and CIFAR100 dataset
                    x_tile = (x_batch.repeat(config.testing.n_samples, 1, 1, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1)
                y_0_hat_tile = (y_0_hat_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).flatten(0, 1)
                y_T_mean_tile = y_0_hat_tile
                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    y_T_mean_tile = torch.zeros(y_0_hat_tile.shape).to(self.device)
                # generate samples from all time steps for the current mini-batch
                p_sample_loop_with_eval(model, x_tile, y_0_hat_tile, y_T_mean_tile, self.num_timesteps,
                                        self.alphas, self.one_minus_alphas_bar_sqrt,
                                        batch_size, config)
                # only compute on a few batches for sanity checking
                if args.sanity_check:
                    if step > 0:  # first two mini-batches
                        break

        ################## compute metrics on test set ##################
        all_true_y_labels = np.concatenate(true_y_label_by_batch_list, axis=0).reshape(-1, 1)
        y_majority_vote_accuracy_all_steps_list = []

        if config.testing.compute_metric_all_steps:
            for idx in range(self.num_timesteps + 1):
                current_t = self.num_timesteps - idx
                # compute accuracy at time step t
                majority_voted_class_t = majority_vote_by_batch_list[current_t]
                majority_vote_correctness = (majority_voted_class_t == all_true_y_labels)  # (n_test, 1)
                y_majority_vote_accuracy = np.mean(majority_vote_correctness)
                y_majority_vote_accuracy_all_steps_list.append(y_majority_vote_accuracy)
            logging.info(
                f"Majority Vote Accuracy across all steps: {y_majority_vote_accuracy_all_steps_list}.\n")
        else:
            # compute accuracy at time step metrics_t
            majority_voted_class_t = majority_vote_by_batch_list[config.testing.metrics_t]
            majority_vote_correctness = (majority_voted_class_t == all_true_y_labels)  # (n_test, 1)
            y_majority_vote_accuracy = np.mean(majority_vote_correctness)
            y_majority_vote_accuracy_all_steps_list.append(y_majority_vote_accuracy)
        instance_accuracy_t = instance_accuracy_by_batch_list[config.testing.metrics_t]
        logging.info("Mean accuracy of all samples at test instance level is {:.4f}%.\n".format(
            np.mean(instance_accuracy_t) * 100))
        logging.info("\nNow we compute metrics related to predicted probability quantiles for all classes...")
        CI_all_classes_t = CI_by_batch_list[config.testing.metrics_t]  # (n_test, 2, n_classes)
        majority_vote_t = majority_vote_by_batch_list[config.testing.metrics_t]  # (n_test, 1)
        majority_vote_accuracy_by_class, \
        CI_width_correct_pred_by_class, \
        CI_width_incorrect_pred_by_class = compute_quantile_metrics(
            config, CI_all_classes_t, majority_vote_t.flatten(), all_true_y_labels.flatten())
        # print(CI_all_classes_t[159])  #@#
        # print(majority_vote_t[159])  #@#
        # print(all_true_y_labels[159])  #@#

        logging.info("\nNow we compute metrics related to paired two sample t-test for all classes...")
        ttest_reject_t = ttest_reject_by_batch_list[config.testing.metrics_t]  # (n_test, )
        _ = compute_ttest_metrics(config, ttest_reject_t, majority_vote_t.flatten(), all_true_y_labels.flatten())

        logging.info("\nNow we compute PAvPU based on paired two sample t-test results...")
        if config.testing.compute_metric_all_steps:
            # compute accuracy at time step metrics_t
            majority_voted_class_t = majority_vote_by_batch_list[config.testing.metrics_t]
            majority_vote_correctness = (majority_voted_class_t == all_true_y_labels)  # (n_test, 1)
        majority_vote_incorrectness = ~majority_vote_correctness  # (n_test, 1)
        n_ac = majority_vote_correctness[ttest_reject_t].sum()
        n_au = majority_vote_correctness[~ttest_reject_t].sum()
        n_ic = majority_vote_incorrectness[ttest_reject_t].sum()
        n_iu = majority_vote_incorrectness[~ttest_reject_t].sum()
        PAvPU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)
        logging.info("\n\tCount of accurate and certain: {}".format(n_ac))
        logging.info("\n\tCount of accurate and uncertain: {}".format(n_au))
        logging.info("\n\tCount of inaccurate and certain: {}".format(n_ic))
        logging.info("\n\tCount of inaccurate and uncertain: {}".format(n_iu))
        logging.info(("\nWe obtain a PAvPU of {:.8f} with an alpha level of {:.4f} " +
                      "for the test set with size {}\n\n").format(
            PAvPU, config.testing.ttest_alpha, majority_vote_correctness.shape[0]))

        logging.info("\nNow we compute NLL and ECE for the test set...")
        label_probs_t = label_probs_by_batch_list[config.testing.metrics_t]  # (n_test, n_classes)
        label_mean_prob_t = label_mean_probs_by_batch_list[config.testing.metrics_t]  # (n_test, n_classes)
        n_test = label_probs_t.shape[0]
        logging.info("\nTest set size: {}".format(n_test))
        # predicted probability corresponding to the true class
        gen_y_true_label_pred_prob_t = torch.tensor(label_mean_prob_t).gather(
            dim=1, index=torch.tensor(all_true_y_labels))  # label_probs_t could result in inf
        NLL_t = - torch.log(gen_y_true_label_pred_prob_t).mean()
        logging.info("\nWe obtain an NLL of {:.8f} for the test set with size {}\n".format(NLL_t, n_test))
        # confidence (predicted probability corresponding to the predicted class)
        gen_y_max_label_prob_t = torch.tensor(label_mean_prob_t).gather(dim=1,
                                                                        index=torch.tensor(majority_vote_t))
        # gen_y_max_label_prob_t = torch.tensor(label_probs_t).gather(dim=1,
        #                                                             index=torch.tensor(majority_vote_t))
        hist_t = torch.histogram(gen_y_max_label_prob_t.flatten(), bins=10, range=(0, 1))
        bin_weights_t = hist_t.hist / n_test
        bin_edges = hist_t.bin_edges[1:]
        # bin membership based on confidence
        bin_membership_t = ((gen_y_max_label_prob_t - bin_edges) >= 0).sum(dim=1)  # (n_test, )
        # accuracy
        # gen_y_majority_vote_correct = torch.tensor(majority_vote_correctness).float()  # (n_test, 1)
        gen_y_majority_vote_correct = torch.tensor(majority_vote_t == all_true_y_labels).float()  # (n_test, 1)
        # compute ECE (Expected Calibration Error)
        calibration_error_t = []
        for bin_idx in range(config.testing.n_bins):
            confidence_bin_t = (bin_membership_t == bin_idx)
            if confidence_bin_t.sum() > 0:
                bin_accuracy = gen_y_majority_vote_correct[confidence_bin_t].mean()
                bin_confidence = gen_y_max_label_prob_t[confidence_bin_t].mean()
                calibration_error_t.append(torch.abs(bin_accuracy - bin_confidence))
            else:
                calibration_error_t.append(0)
        calibration_error_t = torch.tensor(calibration_error_t)
        ECE_t = (bin_weights_t * calibration_error_t).sum()
        # @#
        # print(gen_y_max_label_prob_t.shape, bin_membership_t.shape,
        #       gen_y_majority_vote_correct.shape, calibration_error_t.shape, bin_weights_t.shape)
        # print(calibration_error_t)
        # print(bin_weights_t)
        # @#
        logging.info("\nWe obtain an ECE of {:.8f} for the test set with size {}\n\n".format(ECE_t, n_test))

        # make plot
        if config.testing.compute_metric_all_steps:
            n_metrics = 1
            fig, axs = plt.subplots(n_metrics, 1, figsize=(8.5, n_metrics * 3.5), clear=True)  # W x H
            plt.subplots_adjust(hspace=0.5)
            # majority vote accuracy
            axs.plot(y_majority_vote_accuracy_all_steps_list)
            axs.set_title('Majority Vote Top 1 Accuracy across All Timesteps (Reversed)', fontsize=14)
            axs.set_xlabel('timestep (reversed)', fontsize=12)
            axs.set_ylabel('majority vote accuracy', fontsize=12)
            fig.savefig(os.path.join(args.im_path,
                                     'top_1_test_accuracy_all_timesteps.pdf'))

        n_metrics = 1
        all_classes = np.arange(config.data.num_classes)
        fig, axs = plt.subplots(n_metrics, 1, figsize=(8, n_metrics * 3.6), clear=True)  # W x H
        plt.subplots_adjust(hspace=0.5)
        # majority vote accuracy for each class
        axs.plot(all_classes, majority_vote_accuracy_by_class, label="class_acc")
        axs.plot(all_classes, CI_width_correct_pred_by_class, label="CI_w_cor")
        axs.plot(all_classes, CI_width_incorrect_pred_by_class, label="CI_w_incor")
        axs.set_title('Majority Vote Top 1 Accuracy with \nCredible Interval Width ' +
                      'of Correct and Incorrect Predictions', fontsize=9)
        axs.set_xlabel('Class Label', fontsize=8)
        axs.set_ylabel('Class Probability', fontsize=8)
        axs.set_ylim([0, 1])
        axs.legend(loc='best')
        fig.savefig(os.path.join(args.im_path,
                                 'accuracy_and_CI_width_by_class.pdf'))

        # clear the memory
        plt.close('all')
        del label_probs_by_batch_list
        del majority_vote_by_batch_list
        del instance_accuracy_by_batch_list
        gc.collect()

        return y_majority_vote_accuracy_all_steps_list
    
    def test_ldl_task(self, model, test_loader, n_repeat=10):
        """
        [逻辑修正版] 
        遇到 NaN 不是填假数据，而是【重新采样】，保证数据的真实性。
        """
        model.eval()
        self.cond_pred_model.eval()
        
        all_targets = []
        all_sample_preds = [] 
        
        for _ in range(n_repeat):
            all_sample_preds.append([])

        with torch.no_grad():
            # 1. 收集真实标签
            for _, y_true in test_loader:
                all_targets.append(y_true.numpy())
            Y_true = np.concatenate(all_targets, axis=0)

            # 2. 循环推理 n_repeat 次
            iter_range = range(n_repeat)
            if n_repeat > 1:
                logging.info(f"Running Ensemble Test ({n_repeat} rounds)...")

            for i in iter_range:
                for x_batch, _ in test_loader:
                    x_batch = x_batch.to(self.device)
                    
                    # Guidance
                    y_0_hat = self.compute_guiding_prediction(x_batch)
                    
                    # Diffusion Sampling
                    y_T_mean = y_0_hat
                    if self.config.diffusion.noise_prior:
                        y_T_mean = torch.zeros_like(y_0_hat)
                    max_retries = 20 
                    success = False
                    for attempt in range(max_retries):
                        # 采样
                        y_gen_seq = p_sample_loop(
                            model, x_batch, y_0_hat, y_T_mean, 
                            self.num_timesteps, self.alphas, 
                            self.one_minus_alphas_bar_sqrt, only_last_sample=True
                        )
                        # 检查结果是否有效
                        if isinstance(y_gen_seq, list): 
                            y_check = y_gen_seq[0]
                        else:
                            y_check = y_gen_seq
                        # 如果没有 NaN 且没有 Inf，视为成功
                        if not (torch.isnan(y_check).any() or torch.isinf(y_check).any()):
                            if isinstance(y_gen_seq, list): y_gen_seq = y_gen_seq[0]
                            all_sample_preds[i].append(y_gen_seq.cpu().numpy())
                            success = True
                            break # 跳出重试循环，继续下一个 batch
                        else:
                            logging.warning(f"⚠️ NaN detected in Round {i}, Batch attempt {attempt+1}/{max_retries}. Resampling...")
                    if not success:
                        raise ValueError(f"❌ Error: Model output NaN for {max_retries} consecutive attempts. Training Diverged.")
                    # =================================================================
        full_preds_per_repeat = []
        for i in range(n_repeat):
            full_preds_per_repeat.append(np.concatenate(all_sample_preds[i], axis=0))
        
        stacked_preds = np.stack(full_preds_per_repeat, axis=0) # (n_repeat, n_samples, n_classes)
        Y_hat_avg = np.mean(stacked_preds, axis=0)
        Y_hat = metrics.proj(Y_hat_avg)
        scores = metrics.score(Y_true, Y_hat)
        if n_repeat > 1:
            logging.info("\n" + "="*60)
            logging.info(f"FINAL ENSEMBLE RESULT ({n_repeat} rounds) | Dataset: {self.config.data.dataset}")
            logging.info("-" * 60)
            metrics_keys = ['Cheby', 'Clark', 'Canbe', 'KL', 'Cosine', 'Inter']
            for i, name in enumerate(metrics_keys):
                logging.info(f"{name:<10} | {scores[i]:.4f}")
            logging.info("="*60 + "\n")

        return scores