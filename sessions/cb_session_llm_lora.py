# LORA SFT MODE: Full supervised fine-tuning of the decision maker.
# The LLM backbone (Qwen2.5-0.5B) is fine-tuned with LoRA (r=16 default).
# The adapters and cls_head are also trained end-to-end.
# Only the LoRA A/B matrices and the adapters/cls_head have requires_grad=True.
# Base model weights are frozen by PEFT automatically via get_peft_model().
# Training signal: Best-Action Cross-Entropy from FAISS + S_SimDec simulator.
# To save ONLY the LoRA delta weights: value_network.save_lora(path)

import sys
import os
import time
import random
from turtle import mode
import faiss
import torch
import wandb
import numpy as np
from torch import autograd
from tqdm import tqdm
from tools.utils import AverageMeter
from tools import feature_list
from evaluations.metric import compute_rec_loss, compute_error_rates, weighted_label_smoothing_loss, loss_function, focal_loss
from torch.utils.data import DataLoader
from tools.logger import info
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from models.llm_model_lora import LLMLoRAValueNetwork
from sklearn.utils.validation import check_is_fitted
import torch.nn.functional as F
from tools import feature_list

class CB_Session(object):
    def __init__(self, env, model, dataset):
        self.env = env
        self.model = model
        self.value_network = None
        self.optimizer_dm = None
        self.loader = DataLoader(dataset, batch_size=self.env.args.batch_size, shuffle=True)
        self.train_inputs = dataset.train_inputs
        self.val_inputs = dataset.val_inputs
        self.test_inputs = dataset.test_inputs
        self.optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.model.parameters()), 'lr': self.env.args.lr}], weight_decay=self.env.args.decay_coeff)


        self.action_dim = 4
        self.epsilon = 0.1

        self.early_stop = 0
        self.best_epoch = 0
        self.best_dm_epoch = 0
        self.total_epoch = 0
        self.best_overall_accuracy = 0
        self.best_acc1, self.best_acc2, self.best_acc3 = 0, 0, 0
        self.best_dm_accuracy = 0
        self.cost_dic = dataset.cost_mrp
        self.avg_profit = dataset.avg_profit
        self.test_rec_loss = 99999
        self.scaler = StandardScaler()
        # self.init_value_network()
        self._ensure_scaler_fitted()
        self.best_p = 0
        self.best_o = 0
        self.best_pmp1 = 0
        self.best_pmp2 = 0
        self.best_pmp3 = 0

        self.min_profit, self.max_profit = float('inf'), float('-inf')
        self.min_on_time, self.max_on_time = float('inf'), float('-inf')

    def _ensure_scaler_fitted(self):
        # pick the same feature slice that VN uses
        feature_dim = getattr(self.model, "feature_dim", self.train_inputs.shape[1])
        try:
            check_is_fitted(self.scaler)
        except Exception:
            # train_inputs is typically a numpy array already; if torch, convert
            import numpy as np, torch
            X = self.train_inputs
            if isinstance(X, torch.Tensor):
                X = X.detach().cpu().numpy()
            self.scaler.fit(X[:, :feature_dim])

    def init_value_network(self, value_network=None):
        if value_network is None:
            self.value_network = LLMLoRAValueNetwork(
                self.env,
                model_name=getattr(self.env.args, "hf_model_name",
                                   "Qwen/Qwen2.5-0.5B-Instruct"),
                batch_size=getattr(self.env.args, "batch_size", 64),
                lora_r=getattr(self.env.args, "lora_r", 16),
                lora_alpha=getattr(self.env.args, "lora_alpha", 32),
                lora_dropout=getattr(self.env.args, "lora_dropout", 0.05),
                lora_target_modules=getattr(self.env.args, "lora_target_modules", None),
            )
            info(f"[LoRA] Initialized LLMLoRAValueNetwork: "
                 f"{getattr(self.env.args, 'hf_model_name', 'Qwen/Qwen2.5-0.5B-Instruct')}")
        else:
            self.value_network = value_network
            name = getattr(self.value_network.backbone.config, "_name_or_path", "unknown")
            info(f"[LoRA] Using provided LLMLoRAValueNetwork: {name}")

        # Three separate param groups: LoRA matrices (lower lr), adapters, cls_head
        lora_params = [p for n, p in self.value_network.backbone.named_parameters()
                       if p.requires_grad]
        adapter_params = list(self.value_network.adapters.parameters())
        head_params = list(self.value_network.cls_head.parameters())

        # Store for dm_train_epoch gradient norm logging
        self.lora_params = lora_params

        if not lora_params:
            info("WARNING: No trainable LoRA parameters found. Check PEFT configuration.")

        self.optimizer_dm = torch.optim.AdamW([
            {'params': lora_params,    'lr': self.env.args.dm_lr * 0.1,
             'weight_decay': 0.01},    # LoRA layers: 10x lower lr, small WD
            {'params': adapter_params, 'lr': self.env.args.dm_lr,
             'weight_decay': self.env.args.dm_decay_coeff},
            {'params': head_params,    'lr': self.env.args.dm_lr,
             'weight_decay': self.env.args.dm_decay_coeff},
        ])

        self.scheduler_dm = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_dm,
            T_max=self.env.args.dm_epochs,
            eta_min=1e-6,
        )

        n_lora = sum(p.numel() for p in lora_params)
        n_adapter = sum(p.numel() for p in adapter_params)
        n_head = sum(p.numel() for p in head_params)
        info(f"[LoRA SFT] Trainable params — LoRA: {n_lora:,} | "
             f"Adapters: {n_adapter:,} | cls_head: {n_head:,} | "
             f"Total: {n_lora+n_adapter+n_head:,}")

        self.value_network.train()

    def precompute_best_actions(self):
        """Generate per-action rewards, hard labels, soft labels, and class weights
        for all training samples.

        Stores:
        self.best_action_rewards [N, 4]  — normalized combined rewards (fix H)
        self.best_action_labels  [N]     — argmax (hard) labels
        self.soft_labels         [N, 4]  — softmax over rewards (fix D, for KL-div)
        self.class_weights       [4]     — inverse-frequency weights (fix G, for CE)
        """
        self.model.eval()

        with torch.no_grad():
            self._ensure_scaler_fitted()
            X = self.train_inputs
            if isinstance(X, torch.Tensor):
                X_np = X.detach().cpu().numpy().astype(np.float32)
            else:
                X_np = np.asarray(X, dtype=np.float32)

            N = X_np.shape[0]
            feature_dim = len(
                feature_list.product_info[self.env.args.dataset]
                + feature_list.order_info[self.env.args.dataset]
                + feature_list.customer_info[self.env.args.dataset]
                + feature_list.shipping_info[self.env.args.dataset]
            )

            Xs = self.scaler.transform(X_np).astype(np.float32)
            Xs_t = torch.from_numpy(Xs).to(self.env.device)
            ori_t = torch.from_numpy(X_np).to(self.env.device)

            # build FAISS index
            if isinstance(self.cost_dic, torch.Tensor):
                cost_dic_np = self.cost_dic.detach().cpu().numpy()
            else:
                cost_dic_np = np.asarray(self.cost_dic)
            cost_data = np.ascontiguousarray(cost_dic_np[:, :-1].astype("float32"))
            cost_y = cost_dic_np[:, -1]
            index = faiss.IndexFlatL2(cost_data.shape[1])
            index.add(cost_data)

            ridx0, ridx1 = feature_list.retrieva_index[self.env.args.dataset]
            B = int(self.env.args.batch_size)
            n_batches = (N + B - 1) // B
            log_every = max(1, n_batches // 10)

            # Store profit and on-time SEPARATELY — needed for fix H (per-stream normalization)
            all_profits = torch.zeros(N, 4, dtype=torch.float32)
            all_on_time = torch.zeros(N, 4, dtype=torch.float32)

            for batch_idx in range(n_batches):
                start = batch_idx * B
                end = min(start + B, N)
                bs = end - start

                state_b = Xs_t[start:end, :feature_dim]
                ori_b = ori_t[start:end]

                if batch_idx % log_every == 0:
                    info(f"[SFT] precompute_best_actions: batch {batch_idx}/{n_batches}")

                for a in range(4):
                    # profit via FAISS lookup, fall back to avg_profit on miss
                    q_np = np.stack([
                        ori_b[:, ridx0].cpu().numpy(),
                        ori_b[:, ridx1].cpu().numpy(),
                        np.full(bs, float(a), dtype=np.float32),
                    ], axis=1).astype("float32")

                    _, nn_idx = index.search(q_np, 1)
                    nn_idx = nn_idx.flatten()

                    profits = []
                    for i in range(bs):
                        if np.array_equal(q_np[i], cost_data[nn_idx[i]]):
                            profits.append(float(cost_y[nn_idx[i]]))
                        else:
                            if isinstance(self.avg_profit, torch.Tensor):
                                profits.append(float(self.avg_profit[a].detach().cpu().item()))
                            else:
                                profits.append(float(np.asarray(self.avg_profit)[a]))
                    profit_t = torch.tensor(profits, dtype=torch.float32, device=self.env.device)

                    # on-time bonus via frozen S_SimDec simulator
                    a_batch = torch.full((bs,), a, dtype=torch.long, device=self.env.device)
                    onehot = F.one_hot(a_batch, num_classes=4).float()
                    selected_emb = torch.sum(
                        onehot.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1
                    )
                    pred_tokens = self.model(state_b, selected_emb, ori_b[:, feature_dim + 1:])
                    on_time = pred_tokens[-1].argmax(dim=1).float()

                    # Store raw streams, don't combine yet (fix H)
                    all_profits[start:end, a] = profit_t.cpu()
                    all_on_time[start:end, a] = on_time.cpu()

            # ===== normalize each reward stream by its global std before combining =====
            profit_std = all_profits.std().clamp_min(1e-6)
            on_time_std = all_on_time.std().clamp_min(1e-6)
            info(f"[SFT] reward stats — profit_std={profit_std:.4f} "
                f"on_time_std={on_time_std:.4f}")

            all_rewards = (all_profits / profit_std) + \
                        self.env.args.otr_reward_coeff * (all_on_time / on_time_std)

            # Guard: NaN in rewards (e.g. simulator crash on a sample) would silently
            # corrupt soft labels and propagate into every batch — replace with zeros.
            if torch.isnan(all_rewards).any():
                n_nan = torch.isnan(all_rewards).sum().item()
                info(f"[SFT] WARNING: {n_nan} NaN values in reward matrix — replacing with 0")
                all_rewards = torch.nan_to_num(all_rewards, nan=0.0)

            # Hard labels for logging / fallback CE path
            self.best_action_rewards = all_rewards                          # [N, 4]
            self.best_action_labels = all_rewards.argmax(dim=1).long()      # [N]

            # ===== soft reward targets via temperature-scaled softmax =====
            # Store as log-probabilities so KL can use log_target=True and avoid
            # the 0 * log(0) = nan that fires when softmax underflows to 0.0
            temperature = getattr(self.env.args, "soft_label_temp", 1.0)
            self.log_soft_labels = F.log_softmax(all_rewards / temperature, dim=1)  # [N, 4]
            self.soft_labels = self.log_soft_labels.exp()                            # [N, 4]

            # ===== inverse-frequency class weights from hard labels =====
            counts = torch.bincount(self.best_action_labels, minlength=4).float()
            self.class_weights = (counts.sum() / (4 * counts.clamp_min(1.0))).to(self.env.device)

        torch.cuda.empty_cache()
        info(f"[SFT] Label distribution: {torch.bincount(self.best_action_labels).tolist()}")
        info(f"[SFT] Class weights: {self.class_weights.detach().cpu().tolist()}")
        soft_entropy = -(self.soft_labels * self.soft_labels.clamp_min(1e-9).log()).sum(dim=1).mean().item()
        info(f"[SFT] Soft label mean entropy: {soft_entropy:.4f} (max for 4 classes = 1.386)")
    
    def train_epoch(self):
        t = time.time()
        self.model.train()
        self.total_epoch += 1

        all_rec_loss = AverageMeter()
        all_kl_loss = AverageMeter()
        all_classification_loss = AverageMeter()



        feature_dim = len(feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] +\
                           feature_list.customer_info[self.env.args.dataset] + feature_list.shipping_info[self.env.args.dataset] )

        label_dim =  len(feature_list.label[self.env.args.dataset])

        for input_id in tqdm(self.loader):


            ori_input = input_id.to(self.env.device)
            input_id = self.scaler.transform(input_id)
            input_id = torch.FloatTensor(input_id).to(self.env.device)


            predicted_tokens = self.model(input_id[:,:feature_dim], ori_input[:,feature_dim].long(), ori_input[:,feature_dim+1:].long())
            total_loss = 0
            for i in range(label_dim):
                classification_loss = torch.nn.CrossEntropyLoss()(predicted_tokens[i], ori_input[:, feature_dim+1 + i].long())
                total_loss += classification_loss

            loss = total_loss / label_dim

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
        return all_classification_loss.avg, time.time() - t



    def train(self):
        self.early_stop = 0
        for epoch in range(self.env.args.ckpt_start_epoch, self.env.args.epochs):
            classification_loss, train_time = self.train_epoch()
            info('-' * 50)
            info(
                f'TRAIN:epoch = {epoch}/{self.env.args.epochs} classification_loss = {classification_loss:.5f} train_time = {train_time:.2f}')
            self.test('val')
            if epoch % self.env.args.eva_interval == 0:
                self.early_stop += 1
                accuracies, val_time = self.test('val')

                info('-' * 10)

                epoch_metrics = {"loss/classification_loss": classification_loss}
                for i, accuracy in enumerate(accuracies):
                    info(f"{feature_list.label[self.env.args.dataset][i]} Accuracy: {accuracy * 100:.2f}% val_time = {val_time:.2f}")
                    epoch_metrics[f"eval/{feature_list.label[self.env.args.dataset][i]}"] = accuracy

                if (sum(accuracies) / len(accuracies) if accuracies else 0) > self.best_overall_accuracy:
                    info('-' * 10)
                    self.best_overall_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
                    self.best_acc1 = accuracies[0]
                    self.best_acc2 = accuracies[1]
                    self.best_acc3 = accuracies[2]
                    epoch_metrics["eval/best_overall_accuracy"] = self.best_overall_accuracy
                    info(f"best_overall_accuracy: {self.best_overall_accuracy * 100:.2f}% ")

                if self.env.args.wandb:
                    wandb.log(epoch_metrics, step=epoch)
                    self.early_stop = 0
                    if self.env.args.save:
                        self.save_model(epoch, 'sim')
                    self.best_epoch = epoch

            if self.early_stop > self.env.args.early_stop:
                break


    def dm_train_epoch(self):
        """One LoRA SFT epoch: full shuffled sweep over training data."""
        from tools import feature_list

        assert hasattr(self, "best_action_labels") and self.best_action_labels is not None, \
            "best_action_labels not found — call precompute_best_actions() before dm_train()."

        # Frozen simulator stays in eval
        self.model.eval()
        # LoRA backbone, adapters, and cls_head all in training mode
        self.value_network.backbone.train()
        self.value_network.adapters.train()
        self.value_network.cls_head.train()

        self._ensure_scaler_fitted()
        X = self.train_inputs
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=np.float32)

        N = X_np.shape[0]
        B = int(min(self.env.args.batch_size, N))

        feature_dim = len(
            feature_list.product_info[self.env.args.dataset]
            + feature_list.order_info[self.env.args.dataset]
            + feature_list.customer_info[self.env.args.dataset]
            + feature_list.shipping_info[self.env.args.dataset]
        )

        # Shuffle indices once per epoch and sweep through the whole dataset
        perm = torch.randperm(N)

        total_loss = 0.0
        total_loss_ce = 0.0
        total_loss_kl = 0.0
        total_reward = 0.0
        total_correct = 0
        total_samples = 0
        n_batches = 0

        nan_detected = False

        for start in range(0, N, B):
            indices = perm[start:start + B]
            idx_np = indices.numpy()

            Xb = self.scaler.transform(X_np[idx_np]).astype(np.float32)
            state = torch.from_numpy(Xb).to(self.env.device)[:, :feature_dim]
            a_star_batch = self.best_action_labels[indices].to(self.env.device)
            log_soft_batch = self.log_soft_labels[indices].to(self.env.device)

            logits = self.value_network(state)  # [B, 4]

            # Weighted CE on hard labels corrects for class imbalance
            loss_ce = F.cross_entropy(logits, a_star_batch, weight=self.class_weights)
            # KL against log-soft targets; log_target=True avoids 0*log(0)=nan when
            # softmax underflows to 0 for dominated actions
            loss_kl = F.kl_div(F.log_softmax(logits, dim=1), log_soft_batch,
                                reduction='batchmean', log_target=True)
            loss = 0.5 * loss_ce + 0.5 * loss_kl

            # ---- NaN diagnostics: log offending records before breaking ----
            if not torch.isfinite(loss):
                info(f"[NaN] NaN/Inf loss detected — ce={loss_ce.item():.4f} kl={loss_kl.item():.4f} "
                     f"batch_start={start} batch_size={len(indices)}")
                info(f"[NaN] state — has_nan={torch.isnan(state).any().item()} "
                     f"has_inf={torch.isinf(state).any().item()} "
                     f"min={state.min().item():.4f} max={state.max().item():.4f} "
                     f"mean={state.mean().item():.4f}")
                info(f"[NaN] logits — has_nan={torch.isnan(logits).any().item()} "
                     f"has_inf={torch.isinf(logits).any().item()} "
                     f"min={logits.min().item():.4f} max={logits.max().item():.4f}")
                info(f"[NaN] hard labels: {a_star_batch.tolist()}")
                info(f"[NaN] log_soft_labels — has_nan={torch.isnan(log_soft_batch).any().item()} "
                     f"has_inf={torch.isinf(log_soft_batch).any().item()}")
                info(f"[NaN] class_weights: {self.class_weights.tolist()}")
                # Per-sample breakdown: flag any sample whose state, logits, or soft labels have NaN/Inf
                bad_state   = torch.isnan(state).any(dim=1) | torch.isinf(state).any(dim=1)
                bad_logits  = torch.isnan(logits).any(dim=1) | torch.isinf(logits).any(dim=1)
                bad_soft    = torch.isnan(log_soft_batch).any(dim=1) | torch.isinf(log_soft_batch).any(dim=1)
                bad_mask    = bad_state | bad_logits | bad_soft
                if bad_mask.any():
                    for i in bad_mask.nonzero(as_tuple=True)[0].tolist():
                        info(f"[NaN]   record idx={idx_np[i]} "
                             f"state={state[i].tolist()} "
                             f"logits={logits[i].tolist()} "
                             f"label={a_star_batch[i].item()} "
                             f"log_soft={log_soft_batch[i].tolist()} "
                             f"rewards={self.best_action_rewards[indices[i]].tolist()}")
                else:
                    info("[NaN] No individual bad records found — NaN likely from loss aggregation or class_weights.")
                nan_detected = True
                break

            self.optimizer_dm.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
            self.optimizer_dm.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                total_correct += (preds == a_star_batch).sum().item()
                total_samples += a_star_batch.size(0)

            batch_reward = float(self.best_action_rewards[indices].max(dim=1).values.mean().item())
            total_loss += float(loss.item())
            total_loss_ce += float(loss_ce.item())
            total_loss_kl += float(loss_kl.item())
            total_reward += batch_reward
            n_batches += 1

        torch.cuda.empty_cache()

        avg_loss    = total_loss    / max(1, n_batches)
        avg_loss_ce = total_loss_ce / max(1, n_batches)
        avg_loss_kl = total_loss_kl / max(1, n_batches)
        avg_reward  = total_reward  / max(1, n_batches)
        train_acc   = total_correct / max(1, total_samples)
        # Signal NaN to dm_train by returning nan for loss
        if nan_detected:
            avg_loss = float('nan')
        return avg_reward, avg_loss_ce, avg_loss_kl, avg_loss, train_acc

    def dm_train(self):
        """
        LoRA SFT training loop for the decision-maker.
        Validates periodically, tracks best val metrics, saves best checkpoint.
        """
        import time
        from tools.logger import info

        info("[LoRA SFT] Pre-computing best-action labels from FAISS + simulator...")
        self.precompute_best_actions()
        info(f"[LoRA SFT] Label pre-computation complete. Starting SFT for {self.env.args.dm_epochs} epochs.")

        best_val_score = float("-inf")
        early_stop_counter = 0
        eval_every = getattr(self.env.args, "eva_interval", 1)
        patience = getattr(self.env.args, "early_stop", 10)

        lora_save_path = os.path.join(
            self.env.CKPT_PATH, f"{self.env.suffix}_lora_best"
        ) if getattr(self.env.args, "save", False) else None

        # Pre-training baseline so epoch 0 comparison is meaningful
        pre_profit, pre_on_time, _, _, pre_acc = self.dm_test("val")
        pre_score = pre_profit + pre_on_time
        info(f"[LoRA SFT] Pre-training baseline — profit={pre_profit:.4f} "
             f"on_time={pre_on_time:.4f} score={pre_score:.4f}")
        best_val_score = pre_score
        if lora_save_path is not None:
            self.value_network.save_lora(lora_save_path)
            info(f"[LoRA SFT] Saved pre-training baseline as initial best checkpoint")

        loss_history = []

        for epoch in range(self.env.args.ckpt_start_epoch, self.env.args.dm_epochs):
            t0 = time.time()
            avg_r, loss_ce, loss_kl, loss, train_acc = self.dm_train_epoch()
            train_time = time.time() - t0
            loss_history.append(loss)

            info(f"[DM] epoch {epoch}/{self.env.args.dm_epochs} "
                 f"loss={loss:.4f} (ce={loss_ce:.4f} kl={loss_kl:.4f}) "
                 f"train_acc={train_acc:.4f} train_time={train_time:.1f}s")

            self.scheduler_dm.step()

            current_lr = self.scheduler_dm.get_last_lr()[0]
            epoch_metrics = {
                "dm/loss": loss, "dm/loss_ce": loss_ce, "dm/loss_kl": loss_kl,
                "dm/train_acc": train_acc, "dm/train_time_sec": train_time, "dm/lr": current_lr,
            }

            # Abort on NaN loss — detailed diagnostics already printed inside dm_train_epoch
            if loss != loss:
                info(f"[LoRA SFT] Stopping at epoch {epoch} due to NaN loss.")
                break

            if epoch % eval_every == 0:
                val_profit, val_on_time, val_pmp, val_time, val_acc = self.dm_test("val")
                val_score = val_profit + val_on_time

                info(f"[DM] val epoch {epoch}: profit={val_profit:.4f} "
                    f"on_time={val_on_time:.4f} score={val_score:.4f} val_time={val_time:.1f}s")

                epoch_metrics.update({
                    "val/profit": val_profit, "val/on_time": val_on_time,
                    "val/score": val_score, "val/pmp_0.1": val_pmp[0.1],
                    "val/acc": val_acc if val_acc is not None else 0.0,
                })

                if val_score > best_val_score:
                    best_val_score = val_score
                    self.best_dm_epoch = epoch
                    self.best_p = max(self.best_p, val_profit)
                    self.best_o = max(self.best_o, val_on_time)
                    if val_acc is not None:
                        self.best_dm_accuracy = max(self.best_dm_accuracy, val_acc)
                    early_stop_counter = 0

                    if lora_save_path is not None:
                        self.value_network.save_lora(lora_save_path)
                        info(f"[LoRA SFT] New best val_score={val_score:.4f}, saved to {lora_save_path}")
                else:
                    early_stop_counter += 1
                    info(f"[DM] no improvement ({early_stop_counter}/{patience})")

                if early_stop_counter >= patience:
                    info(f"[LoRA SFT] Early stopping at epoch {epoch}. "
                        f"Best val_score={best_val_score:.4f} at epoch {self.best_dm_epoch}.")
                    break

            if getattr(self.env.args, "wandb", False):
                try:
                    import wandb
                    wandb.log(epoch_metrics, step=epoch)
                except Exception:
                    pass

        if loss_history:
            lh = np.array(loss_history)
            finite = lh[np.isfinite(lh)]
            if finite.size > 0:
                info(f"[LoRA SFT] Loss summary over {len(lh)} epochs — "
                     f"initial={lh[0]:.4f}  final={lh[-1]:.4f}  "
                     f"min={finite.min():.4f} (epoch {int(np.where(np.isfinite(lh))[0][finite.argmin()])})  "
                     f"max={finite.max():.4f}  mean={finite.mean():.4f}"
                     + (f"  nan_epochs={int((~np.isfinite(lh)).sum())}" if not np.all(np.isfinite(lh)) else ""))
            else:
                info(f"[LoRA SFT] Loss summary: all {len(lh)} epochs produced NaN/Inf loss.")
        info(f"[LoRA SFT] Training complete. Best val_score={best_val_score:.4f} at epoch {self.best_dm_epoch}.")

    def dm_test(self, mode="test"):

        self._ensure_scaler_fitted()
        self.model.eval()
        self.value_network.eval()
        t = time.time()

        # pick inputs
        if mode in ("val", "ori"):
            input_id = self.val_inputs
        else:
            input_id = self.test_inputs

        if not isinstance(input_id, torch.Tensor):
            input_id = torch.tensor(input_id, dtype=torch.float32)
        ori_input = input_id.to(self.env.device)

        # Optional: limit evaluation size for faster LLM runs
        limit = getattr(self.env.args, "dm_eval_limit", None)
        if limit:
            input_id = input_id[:int(limit)]
            ori_input = ori_input[:int(limit)]

        feature_dim = len(
            feature_list.product_info[self.env.args.dataset]
            + feature_list.order_info[self.env.args.dataset]
            + feature_list.customer_info[self.env.args.dataset]
            + feature_list.shipping_info[self.env.args.dataset]
        )

        if mode != "ori":
            X = input_id.detach().cpu().numpy()
            X = self.scaler.transform(X)
            input_id = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.env.device)
        else:
            input_id = ori_input

        state = input_id[:, :feature_dim]

        # ---- FAISS setup — build index once and cache ----
        if not hasattr(self, "_faiss_index") or self._faiss_index is None:
            if isinstance(self.cost_dic, torch.Tensor):
                cost_dic_np = self.cost_dic.detach().cpu().numpy()
            else:
                cost_dic_np = np.asarray(self.cost_dic)
            self._cost_data = np.ascontiguousarray(cost_dic_np[:, :-1].astype('float32'))
            self._cost_y = cost_dic_np[:, -1].astype('float32')
            self._faiss_index = faiss.IndexFlatL2(self._cost_data.shape[1])
            self._faiss_index.add(self._cost_data)

        with torch.no_grad():
            if mode == "ori":
                decision_indices = input_id[:, feature_dim].long()
                decision_prob = F.one_hot(decision_indices, num_classes=4).float().to(self.env.device)
                decision_prob_value = decision_prob
            else:
                _infer_bs = int(getattr(self.env.args, "batch_size", 64))
                _chunks = [
                    self.value_network(state[_s:_s + _infer_bs])
                    for _s in range(0, state.shape[0], _infer_bs)
                ]
                value_network_output = torch.cat(_chunks, dim=0)
                # Clean one-hot from argmax — no float-equality trick
                actions_tensor = value_network_output.argmax(dim=1)
                decision_prob = F.one_hot(actions_tensor, num_classes=4).float()
                decision_prob_value = decision_prob

            action = decision_prob_value.argmax(dim=1).view(-1)  # view(-1) safe for batch=1

            # Decision accuracy (logged only, no mutation of best_*)
            dm_acc = None
            if mode != "ori":
                gt_actions = ori_input[:, feature_dim].long()
                dm_acc = (action == gt_actions).float().mean().item()
                info(f"[DM] {mode} decision accuracy: {dm_acc:.4f}")

            # ---- Vectorized FAISS lookup ----
            ridx0, ridx1 = feature_list.retrieva_index[self.env.args.dataset]
            ori_cpu = ori_input.detach().cpu()
            action_np = action.detach().cpu().numpy().astype('float32')

            query_vectors = np.empty((state.shape[0], 3), dtype='float32')
            query_vectors[:, 0] = ori_cpu[:, ridx0].numpy()
            query_vectors[:, 1] = ori_cpu[:, ridx1].numpy()
            query_vectors[:, 2] = action_np

            _, nearest_indices = self._faiss_index.search(query_vectors, 1)
            nearest_indices = nearest_indices.flatten()

            # Vectorized: check exact match for each row, fall back to avg_profit
            matched = self._cost_data[nearest_indices]                    # [N, 3]
            is_exact = np.all(matched == query_vectors, axis=1)           # [N] bool
            matched_profits = self._cost_y[nearest_indices]               # [N]

            if isinstance(self.avg_profit, torch.Tensor):
                avg_profit_np = self.avg_profit.detach().cpu().numpy().astype('float32')
            else:
                avg_profit_np = np.asarray(self.avg_profit, dtype='float32')
            fallback_profits = avg_profit_np[action_np.astype(np.int64)]  # [N]

            local_profits = np.where(is_exact, matched_profits, fallback_profits)
            profit_sum = float(local_profits.sum())
            profit_count = int(local_profits.shape[0])

            # ---- on-time prediction via simulator ----
            selected_embedding = torch.sum(
                decision_prob.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1
            )
            predicted_tokens = self.model(
                input_id[:, :feature_dim],
                selected_embedding,
                ori_input[:, feature_dim + 1:]
            )
            time_sum = predicted_tokens[-1].argmax(dim=1).sum().item()
            time_count = predicted_tokens[-1].shape[0]

        # aggregate metrics
        profit = (profit_sum / profit_count) if profit_count > 0 else 0.0
        on_time_ratio = (time_sum / time_count) if time_count > 0 else 0.0

        # percentiles on profits
        if local_profits.size > 0:
            sorted_profits = np.sort(local_profits)
            profit_min_percent = {}
            for thr in [0.1, 0.2, 0.3]:
                idx = max(0, min(len(sorted_profits) - 1, int(thr * len(sorted_profits))))
                profit_min_percent[thr] = float(sorted_profits[idx])
        else:
            profit_min_percent = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0}

        # Callers (dm_train, main) log these via the return values so they can
        # include all metrics in a single wandb.log() call and avoid step conflicts.
        return profit, on_time_ratio, profit_min_percent, time.time() - t, dm_acc


    def test(self, mode):

        chunk_size = int(self.env.args.batch_size // 1.5)
        self.model.eval()
        t = time.time()

        if mode == 'val':
            input_id = self.val_inputs
        else:
            input_id = self.test_inputs



        correct_preds = 0
        total_samples = 0

        ori_input = input_id.to(self.env.device)
        input_id = self.scaler.transform(input_id)
        input_id = torch.FloatTensor(input_id).to(self.env.device)

        feature_dim = len(feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] +\
                           feature_list.customer_info[self.env.args.dataset] + feature_list.shipping_info[self.env.args.dataset] )
        label_dim =  len(feature_list.label[self.env.args.dataset])

        ori_label_value_counts = {j: {} for j in range(label_dim)}


        label_value_counts = {j: {} for j in range(label_dim)}
        with torch.no_grad():

            correct_preds = [0] * label_dim
            total_samples = [0] * label_dim


            for i in range(0, len(input_id), chunk_size):

                input_chunk = input_id[i:i + chunk_size]
                ori_chunk = ori_input[i:i + chunk_size]


                for j in range(label_dim):
                    for value in ori_chunk[:, feature_dim + j + 1].cpu().numpy():
                        if value not in ori_label_value_counts[j]:
                            ori_label_value_counts[j][value] = 0
                        ori_label_value_counts[j][value] += 1



                predicted_tokens = self.model(input_chunk[:,:feature_dim], ori_chunk[:,feature_dim].long(), ori_chunk[:,feature_dim+1:])


                class_labels = ori_chunk[:, -label_dim:].long().to(self.env.device)


                for j in range(label_dim):
                    predicted = torch.argmax(predicted_tokens[j], dim=1)
                    correct_preds[j] += (predicted == class_labels[:, j]).sum().item()
                    total_samples[j] += len(class_labels[:, j])
                    for value in predicted.cpu().numpy():
                        if value not in label_value_counts[j]:
                            label_value_counts[j][value] = 0
                        label_value_counts[j][value] += 1


        accuracies = [correct_preds[j] / total_samples[j] for j in range(label_dim)]

        return accuracies, time.time() - t

    def save_ckpt(self, path):
        torch.save(self.model.state_dict(), path)

    def save_model(self, current_epoch, mode):
        model_state_file = os.path.join(
            self.env.CKPT_PATH, f'{self.env.suffix}_epoch{current_epoch}.pth')
        self.save_ckpt(model_state_file)
        if mode == 'sim':
            best_epoch = self.best_epoch
        else:
            best_epoch = self.best_dm_epoch
        if current_epoch != best_epoch:
            old_model_state_file = os.path.join(
                self.env.CKPT_PATH, f'{self.env.suffix}_epoch{best_epoch}.pth')
            if os.path.exists(old_model_state_file):
                os.system('rm {}'.format(old_model_state_file))

    def step(self, state, action):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.env.device)
            action_tensor = torch.LongTensor([action]).to(self.env.device)


            predicted_tokens = self.forward(state_tensor, action_tensor, state_tensor)
            profit = predicted_tokens[0, 0].item()
            on_time_ratio = predicted_tokens[0, 1].item()


        reward = profit + self.env.beta * on_time_ratio


        next_state = None


        self.env.index += 1
        done = self.env.index >= len(self.env.loader.test_inputs)

        info = {"profit": profit, "on_time": on_time_ratio}
        return next_state, reward, done, info

    def step(self):
        self.model.eval()
        self.value_network.eval()
        t = time.time()
        self.env.index = 0
        input_id = self.train_inputs

        ori_input = input_id.to(self.env.device)


        input_id = torch.FloatTensor(input_id).to(self.env.device)
        feature_dim = len(feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] +\
                        feature_list.customer_info[self.env.args.dataset] + feature_list.shipping_info[self.env.args.dataset])


        self.cost_dic_data = self.cost_dic[:, :-1]
        self.cost_dic_y = self.cost_dic[:, -1]


        index = faiss.IndexFlatL2(self.cost_dic_data.shape[1])
        index.add(self.cost_dic_data)

        profit_sum = 0
        profit_count = 0
        time_sum = 0
        time_count = 0
        local_profits = []

        state = input_id[:, :feature_dim]

        with torch.no_grad():
            value_network_output = self.value_network(state)
            decision_prob_value =(F.softmax(value_network_output, dim=1) == F.softmax(value_network_output, dim=1).max(dim=1, keepdim=True).values).float()
            decision_prob = decision_prob_value


            action = decision_prob_value.argmax(dim=1).squeeze()



            query_vectors = np.array([
                [
                    ori_input[i, feature_list.retrieva_index[self.env.args.dataset][0]].cpu().item(),
                    ori_input[i, feature_list.retrieva_index[self.env.args.dataset][1]].cpu().item(),
                    action[i].item()
                ]
                for i in range(len(state))
            ], dtype='float32')

            _, nearest_indices = index.search(query_vectors, 1)
            nearest_samples = self.cost_dic_data[nearest_indices.flatten()].cpu().numpy()

            profit, on_time_ratio, reward = [], [], []


            for idx, (query, nearest) in enumerate(zip(query_vectors, nearest_samples)):
                if np.array_equal(query, nearest):
                    selected_y = self.cost_dic_y[nearest_indices[idx, 0]]
                else:
                    selected_y = torch.tensor(self.avg_profit)[action[idx].cpu()]

                profit.append(selected_y)



            selected_embedding = torch.sum(decision_prob.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1)

            predicted_tokens = self.model(input_id[:, :feature_dim], selected_embedding, ori_input[:, feature_dim + 1:])


            time_sum += predicted_tokens[-1].argmax(dim=1).sum().item()
            time_count += len(predicted_tokens[-1])





        on_time_ratio = predicted_tokens[-1].argmax(dim=1).tolist()

        for i in range(len(on_time_ratio)):
            reward.append(on_time_ratio[i] + profit[i])


        next_state = None

        done = [False * len(on_time_ratio)]
        done[-1] = True

        info = {"profit": profit, "on_time": on_time_ratio}
        return next_state, reward, done, info
