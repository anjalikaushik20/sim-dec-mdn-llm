# Attention-pooling LLM decision maker.
# Training: frozen backbone; pool_attn + cls_head fine-tuned with CE + KL on best-action labels.
# Inference: forward() → argmax (consistent with training, no generate_action() gap).

import sys
import os
import time
import random
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
from models.llm_model_attnpool import LLMAttnPoolNetwork
from sklearn.utils.validation import check_is_fitted

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
        self.best_dm_accuracy_true = 0
        self.cost_dic = dataset.cost_mrp
        self.avg_profit = dataset.avg_profit
        self.test_rec_loss = 99999
        self.scaler = StandardScaler()
        self._ensure_scaler_fitted()
        self.best_p = 0
        self.best_o = 0
        self.best_pmp1 = 0
        self.best_pmp2 = 0
        self.best_pmp3 = 0

        self.min_profit, self.max_profit = float('inf'), float('-inf')
        self.min_on_time, self.max_on_time = float('inf'), float('-inf')

    def _ensure_scaler_fitted(self):
        feature_dim = getattr(self.model, "feature_dim", self.train_inputs.shape[1])
        try:
            check_is_fitted(self.scaler)
        except Exception:
            import numpy as np, torch
            X = self.train_inputs
            if isinstance(X, torch.Tensor):
                X = X.detach().cpu().numpy()
            self.scaler.fit(X[:, :feature_dim])

    def init_value_network(self, value_network=None):
        if value_network is None:
            model_name = getattr(self.env.args, "hf_model_name", None)
            batch_size = getattr(self.env.args, "batch_size", 32)
            self.value_network = LLMAttnPoolNetwork(self.env, model_name=model_name, batch_size=batch_size)
            info(f"Initialized LLMAttnPoolNetwork: {model_name} (batch={batch_size})")
        else:
            self.value_network = value_network
            name = getattr(self.value_network.backbone.config, "_name_or_path", "unknown")
            info(f"Initialized value network from provided instance: {name}")

        trainable_params = [p for p in self.value_network.parameters() if p.requires_grad]
        if not trainable_params:
            info("WARNING: No trainable parameters found in value_network.")
        self.optimizer_dm = torch.optim.Adam(
            trainable_params, lr=self.env.args.dm_lr, weight_decay=self.env.args.dm_decay_coeff
        )
        self.value_network.train()

    def precompute_best_actions(self):
        """Generate per-action rewards, hard labels, soft labels, and class weights
        for all training samples.

        Stores:
            self.best_action_rewards  [N, 4] — normalized combined rewards
            self.best_action_labels   [N]    — argmax hard labels
            self.log_soft_labels      [N, 4] — log-softmax over rewards (for KL-div)
            self.soft_labels          [N, 4] — softmax over rewards
            self.class_weights        [4]    — inverse-frequency weights (for CE)
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

            all_profits = torch.zeros(N, 4, dtype=torch.float32)
            all_on_time = torch.zeros(N, 4, dtype=torch.float32)

            for batch_idx in range(n_batches):
                start = batch_idx * B
                end = min(start + B, N)
                bs = end - start

                state_b = Xs_t[start:end, :feature_dim]
                ori_b = ori_t[start:end]

                if batch_idx % log_every == 0:
                    info(f"[LABEL] precompute_best_actions: batch {batch_idx}/{n_batches}")

                for a in range(4):
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

                    a_batch = torch.full((bs,), a, dtype=torch.long, device=self.env.device)
                    onehot = F.one_hot(a_batch, num_classes=4).float()
                    selected_emb = torch.sum(
                        onehot.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1
                    )
                    pred_tokens = self.model(state_b, selected_emb, ori_b[:, feature_dim + 1:])
                    on_time = pred_tokens[-1].argmax(dim=1).float()

                    all_profits[start:end, a] = profit_t.cpu()
                    all_on_time[start:end, a] = on_time.cpu()

            profit_std = all_profits.std().clamp_min(1e-6)
            on_time_std = all_on_time.std().clamp_min(1e-6)
            info(f"[LABEL] reward stats — profit_std={profit_std:.4f} on_time_std={on_time_std:.4f}")

            all_rewards = (all_profits / profit_std) + \
                          self.env.args.otr_reward_coeff * (all_on_time / on_time_std)

            if torch.isnan(all_rewards).any():
                n_nan = torch.isnan(all_rewards).sum().item()
                info(f"[LABEL] WARNING: {n_nan} NaN values in reward matrix — replacing with 0")
                all_rewards = torch.nan_to_num(all_rewards, nan=0.0)

            self.best_action_rewards = all_rewards
            self.best_action_labels = all_rewards.argmax(dim=1).long()

            temperature = getattr(self.env.args, "soft_label_temp", 1.0)
            self.log_soft_labels = F.log_softmax(all_rewards / temperature, dim=1)
            self.soft_labels = self.log_soft_labels.exp()

            counts = torch.bincount(self.best_action_labels, minlength=4).float()
            self.class_weights = (counts.sum() / (4 * counts.clamp_min(1.0))).to(self.env.device)

            # Store for _compute_split_labels() calls below
            self._label_faiss_index = index
            self._label_cost_data = cost_data
            self._label_cost_y = cost_y
            self._label_profit_std = profit_std
            self._label_on_time_std = on_time_std

        torch.cuda.empty_cache()
        info(f"[LABEL] Label distribution: {torch.bincount(self.best_action_labels).tolist()}")
        info(f"[LABEL] Class weights: {self.class_weights.detach().cpu().tolist()}")
        soft_ent = -(self.soft_labels * self.soft_labels.clamp_min(1e-9).log()).sum(dim=1).mean().item()
        info(f"[LABEL] Soft label mean entropy: {soft_ent:.4f} (max for 4 classes = 1.386)")

        # Compute optimal labels for val and test splits (used in dm_test dm_acc_true)
        info("[LABEL] Pre-computing optimal action labels for val/test splits...")
        for split_name, split_inputs in [("val", self.val_inputs), ("test", self.test_inputs)]:
            if isinstance(split_inputs, torch.Tensor):
                sp_np = split_inputs.detach().cpu().numpy().astype(np.float32)
            else:
                sp_np = np.asarray(split_inputs, dtype=np.float32)
            labels = self._compute_split_labels(sp_np)
            setattr(self, f"{split_name}_action_labels", labels)
            info(f"[LABEL] {split_name} label dist: {torch.bincount(labels).tolist()}")

    def _compute_split_labels(self, X_np):
        """Compute FAISS-optimal action labels for any input split.

        precompute_best_actions() must have been called first — reuses the FAISS
        index and normalization stats stored on self._label_*.
        Returns best_action_labels [N] (LongTensor, CPU).
        """
        self.model.eval()
        feature_dim = len(
            feature_list.product_info[self.env.args.dataset]
            + feature_list.order_info[self.env.args.dataset]
            + feature_list.customer_info[self.env.args.dataset]
            + feature_list.shipping_info[self.env.args.dataset]
        )
        N = X_np.shape[0]
        Xs = self.scaler.transform(X_np).astype(np.float32)
        Xs_t = torch.from_numpy(Xs).to(self.env.device)
        ori_t = torch.from_numpy(X_np).to(self.env.device)
        B = int(self.env.args.batch_size)
        ridx0, ridx1 = feature_list.retrieva_index[self.env.args.dataset]

        all_profits = torch.zeros(N, 4, dtype=torch.float32)
        all_on_time = torch.zeros(N, 4, dtype=torch.float32)

        with torch.no_grad():
            for batch_idx in range((N + B - 1) // B):
                start = batch_idx * B
                end = min(start + B, N)
                bs = end - start
                state_b = Xs_t[start:end, :feature_dim]
                ori_b = ori_t[start:end]

                for a in range(4):
                    q_np = np.stack([
                        ori_b[:, ridx0].cpu().numpy(),
                        ori_b[:, ridx1].cpu().numpy(),
                        np.full(bs, float(a), dtype=np.float32),
                    ], axis=1).astype("float32")
                    _, nn_idx = self._label_faiss_index.search(q_np, 1)
                    nn_idx = nn_idx.flatten()

                    profits = []
                    for i in range(bs):
                        if np.array_equal(q_np[i], self._label_cost_data[nn_idx[i]]):
                            profits.append(float(self._label_cost_y[nn_idx[i]]))
                        else:
                            if isinstance(self.avg_profit, torch.Tensor):
                                profits.append(float(self.avg_profit[a].detach().cpu().item()))
                            else:
                                profits.append(float(np.asarray(self.avg_profit)[a]))
                    profit_t = torch.tensor(profits, dtype=torch.float32, device=self.env.device)

                    a_batch = torch.full((bs,), a, dtype=torch.long, device=self.env.device)
                    onehot = F.one_hot(a_batch, num_classes=4).float()
                    selected_emb = torch.sum(
                        onehot.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1
                    )
                    pred_tokens = self.model(state_b, selected_emb, ori_b[:, feature_dim + 1:])
                    on_time = pred_tokens[-1].argmax(dim=1).float()

                    all_profits[start:end, a] = profit_t.cpu()
                    all_on_time[start:end, a] = on_time.cpu()

        all_rewards = (all_profits / self._label_profit_std) + \
                      self.env.args.otr_reward_coeff * (all_on_time / self._label_on_time_std)
        all_rewards = torch.nan_to_num(all_rewards, nan=0.0)
        return all_rewards.argmax(dim=1).long()

    def precompute_hidden_states(self):
        """Run the frozen backbone once over all splits and cache the results on CPU.

        After this returns, dm_train_epoch() and dm_test() never call the backbone
        again — they index into these caches and run only pool_attn + cls_head.

        Cache layout per split:
            {split}_hidden_cache : [N, T_max, H]  float16  (CPU)
            {split}_mask_cache   : [N, T_max]      bool     (CPU)
        """
        self.value_network.eval()

        feature_dim = len(
            feature_list.product_info[self.env.args.dataset]
            + feature_list.order_info[self.env.args.dataset]
            + feature_list.customer_info[self.env.args.dataset]
            + feature_list.shipping_info[self.env.args.dataset]
        )
        B = int(self.env.args.batch_size)

        for split_name, inputs in [
            ('train', self.train_inputs),
            ('val',   self.val_inputs),
            ('test',  self.test_inputs),
        ]:
            if isinstance(inputs, torch.Tensor):
                X_np = inputs.detach().cpu().numpy().astype(np.float32)
            else:
                X_np = np.asarray(inputs, dtype=np.float32)

            N = X_np.shape[0]
            all_hidden = []
            all_masks  = []

            info(f"[CACHE] Encoding {split_name} split ({N} samples)...")
            with torch.no_grad():
                for start in range(0, N, B):
                    end = min(start + B, N)
                    raw = torch.from_numpy(X_np[start:end, :feature_dim]).to(self.env.device)
                    h, m = self.value_network.encode_batch(raw)   # [bs,T,H], [bs,T]
                    all_hidden.append(h.cpu().half())              # float16 to save RAM
                    all_masks.append(m.cpu())

            T_max = max(h.shape[1] for h in all_hidden)
            H     = all_hidden[0].shape[2]
            hidden_cache = torch.zeros(N, T_max, H, dtype=torch.float16)
            mask_cache   = torch.zeros(N, T_max, dtype=torch.bool)

            idx = 0
            for h, m in zip(all_hidden, all_masks):
                bs, T, _ = h.shape
                hidden_cache[idx:idx + bs, :T, :] = h
                mask_cache[idx:idx + bs, :T]       = m.bool()
                idx += bs

            mb = hidden_cache.numel() * 2 / 1e6
            info(f"[CACHE] {split_name}: {N}×{T_max}×{H} = {mb:.0f} MB (float16, CPU)")
            setattr(self, f'{split_name}_hidden_cache', hidden_cache)
            setattr(self, f'{split_name}_mask_cache',   mask_cache)

        torch.cuda.empty_cache()
        info("[CACHE] Hidden-state caching complete.")

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
            all_classification_loss.update(loss.item())

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
            if self.env.args.wandb:
                wandb.log({"loss/classification_loss":classification_loss}, epoch)
            if epoch % self.env.args.eva_interval == 0:
                self.early_stop += 1
                accuracies, val_time = self.test('val')

                info('-' * 10)
                for i, accuracy in enumerate(accuracies):
                    info(f"{feature_list.label[self.env.args.dataset][i]} Accuracy: {accuracy * 100:.2f}% val_time = {val_time:.2f}")
                    if self.env.args.wandb:
                        wandb.log({f"eval/{feature_list.label[self.env.args.dataset][i]}":accuracy}, epoch)

                if (sum(accuracies) / len(accuracies) if accuracies else 0) > self.best_overall_accuracy:
                    info('-' * 10)
                    self.best_overall_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
                    self.best_acc1 = accuracies[0]
                    self.best_acc2 = accuracies[1]
                    self.best_acc3 = accuracies[2]
                    if self.env.args.wandb:
                        wandb.log({f"eval/best_overall_accuracy":self.best_overall_accuracy}, epoch)
                    info(f"best_overall_accuracy: {self.best_overall_accuracy * 100:.2f}% ")
                    self.early_stop = 0
                    if self.env.args.save:
                        self.save_model(epoch, 'sim')
                    self.best_epoch = epoch

            if self.early_stop > self.env.args.early_stop:
                break


    def dm_train_epoch(self):
        """One attnpool training epoch: full shuffled sweep over all training data.

        pool_attn and cls_head are trained with class-weighted CE on hard best-action
        labels and KL-div on soft reward targets.  Backbone stays fully frozen.
        Returns (avg_loss, avg_loss_ce, avg_loss_kl, train_acc).
        """
        assert hasattr(self, "best_action_labels") and self.best_action_labels is not None, \
            "best_action_labels not found — call precompute_best_actions() before dm_train()."

        self.model.eval()
        self.value_network.train()

        N = self.train_hidden_cache.shape[0]
        B = int(min(self.env.args.batch_size, N))

        perm = torch.randperm(N)
        total_loss = 0.0
        total_loss_ce = 0.0
        total_loss_kl = 0.0
        total_correct = 0
        total_samples = 0
        n_batches = 0
        nan_detected = False

        for start in range(0, N, B):
            indices = perm[start:start + B]

            # Fetch cached hidden states — backbone never runs during training
            hidden_b = self.train_hidden_cache[indices].to(self.env.device).float()
            mask_b   = self.train_mask_cache[indices].to(self.env.device)
            a_star_batch   = self.best_action_labels[indices].to(self.env.device)
            log_soft_batch = self.log_soft_labels[indices].to(self.env.device)

            logits = self.value_network.forward_from_hidden(hidden_b, mask_b)  # [B, 4]

            loss_ce = F.cross_entropy(logits, a_star_batch, weight=self.class_weights)
            loss_kl = F.kl_div(F.log_softmax(logits, dim=1), log_soft_batch,
                               reduction='batchmean', log_target=True)
            loss = 0.5 * loss_ce + 0.5 * loss_kl

            if not torch.isfinite(loss):
                info(f"[NaN] loss={loss.item():.4f} ce={loss_ce.item():.4f} "
                     f"kl={loss_kl.item():.4f} at batch start={start}")
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

            total_loss += float(loss.item())
            total_loss_ce += float(loss_ce.item())
            total_loss_kl += float(loss_kl.item())
            n_batches += 1

        torch.cuda.empty_cache()
        avg_loss    = float('nan') if nan_detected else total_loss    / max(1, n_batches)
        avg_loss_ce = total_loss_ce / max(1, n_batches)
        avg_loss_kl = total_loss_kl / max(1, n_batches)
        train_acc   = total_correct / max(1, total_samples)
        return avg_loss, avg_loss_ce, avg_loss_kl, train_acc

    def dm_train_epoch_reward(self):
        """One reward-based training epoch using Gumbel-softmax end-to-end.

        Mirrors cb_session.dm_train_epoch() but replaces value_network(state) with
        forward_from_hidden(cached_hidden, mask). The on-time signal flows directly
        through the loss (no proxy-label collapse).
        Returns (avg_loss, avg_profit_loss, avg_late_loss, avg_mi_loss, avg_ma_loss, train_time).
        """
        t = time.time()
        self.model.eval()
        self.value_network.train()

        feature_dim = len(
            feature_list.product_info[self.env.args.dataset]
            + feature_list.order_info[self.env.args.dataset]
            + feature_list.customer_info[self.env.args.dataset]
            + feature_list.shipping_info[self.env.args.dataset]
        )

        if isinstance(self.train_inputs, torch.Tensor):
            train_X_np = self.train_inputs.detach().cpu().numpy().astype(np.float32)
        else:
            train_X_np = np.asarray(self.train_inputs, dtype=np.float32)
        train_Xs = self.scaler.transform(train_X_np).astype(np.float32)

        N = train_X_np.shape[0]
        B = int(min(self.env.args.batch_size, N))

        if isinstance(self.cost_dic, torch.Tensor):
            cost_dic_np = self.cost_dic.detach().cpu().numpy()
        else:
            cost_dic_np = np.asarray(self.cost_dic)
        cost_data = np.ascontiguousarray(cost_dic_np[:, :-1].astype("float32"))
        cost_y = cost_dic_np[:, -1]
        index = faiss.IndexFlatL2(cost_data.shape[1])
        index.add(cost_data)

        ridx0, ridx1 = feature_list.retrieva_index[self.env.args.dataset]

        if isinstance(self.avg_profit, torch.Tensor):
            avg_profit_arr = self.avg_profit.detach().cpu().numpy().astype(np.float32)
        else:
            avg_profit_arr = np.asarray(self.avg_profit, dtype=np.float32)
        profits_t = torch.tensor(avg_profit_arr, dtype=torch.float32)
        profit_weights = (profits_t / profits_t.max()).to(self.env.device)
        target_class = profits_t.argmax().long()

        perm = torch.randperm(N)

        all_mi_loss = AverageMeter()
        all_ma_loss = AverageMeter()
        all_profit_loss = AverageMeter()
        all_late_loss = AverageMeter()
        all_loss = AverageMeter()

        for start in range(0, N, B):
            indices = perm[start:start + B]
            bs = len(indices)
            idx_np = indices.numpy()

            hidden_b = self.train_hidden_cache[indices].to(self.env.device).float()
            mask_b   = self.train_mask_cache[indices].to(self.env.device)

            ori_b   = torch.from_numpy(train_X_np[idx_np]).to(self.env.device)
            state_b = torch.from_numpy(train_Xs[idx_np, :feature_dim]).to(self.env.device)

            logits = self.value_network.forward_from_hidden(hidden_b, mask_b)  # [B, 4]
            decision_prob_value = F.softmax(logits, dim=1)
            decision_prob = F.gumbel_softmax(decision_prob_value, tau=1, hard=True)

            selected_emb = torch.sum(
                decision_prob.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1
            )
            predicted_tokens = self.model(state_b, selected_emb, ori_b[:, feature_dim + 1:])

            profit_loss = F.cross_entropy(
                decision_prob,
                target_class.expand(bs).to(self.env.device),
                weight=profit_weights,
            )
            target_on_time = torch.ones(bs, dtype=torch.long, device=self.env.device)
            late_loss = F.cross_entropy(predicted_tokens[-1], target_on_time)
            mi_loss = self.env.args.mip_coeff * profit_loss + self.env.args.mil_coeff * late_loss

            action = decision_prob.argmax(dim=1)
            q_np = np.stack([
                ori_b[:, ridx0].cpu().numpy(),
                ori_b[:, ridx1].cpu().numpy(),
                action.cpu().numpy().astype("float32"),
            ], axis=1).astype("float32")
            _, nn_idx = index.search(q_np, 1)
            nn_idx = nn_idx.flatten()
            nn_samples = cost_data[nn_idx]
            matches = torch.tensor(
                np.all(nn_samples == q_np, axis=1), device=self.env.device
            )
            matched_y = torch.tensor(cost_y[nn_idx], dtype=torch.float32, device=self.env.device)
            avg_profit_dev = torch.tensor(avg_profit_arr, device=self.env.device)
            selected_y = torch.where(matches, matched_y, avg_profit_dev[action])

            one_hot = F.one_hot(action, num_classes=4).float()
            action_profit_sum  = torch.matmul(one_hot.T, selected_y.unsqueeze(1)).squeeze(1)
            action_profit_count = one_hot.sum(dim=0)
            on_time = predicted_tokens[-1].argmax(dim=1)
            action_on_time_sum = torch.matmul(one_hot.T, on_time.unsqueeze(1).float()).squeeze(1)

            avg_profit_per_action  = action_profit_sum  / (action_profit_count + 1e-8)
            avg_on_time_per_action = action_on_time_sum / (action_profit_count + 1e-8)
            reward_per_action = (avg_profit_per_action
                                 + self.env.args.otr_reward_coeff * avg_on_time_per_action)

            if not hasattr(self, "smoothed_reward"):
                self.smoothed_reward = reward_per_action.detach()
            self.smoothed_reward = (
                self.env.args.reward_smoothing_factor * reward_per_action.detach()
                + (1 - self.env.args.reward_smoothing_factor) * self.smoothed_reward
            )

            predicted_rewards = logits.mean(dim=0)  # [4] — reuse existing forward
            ma_loss = F.mse_loss(predicted_rewards, self.smoothed_reward)

            loss = self.env.args.mi_coeff * mi_loss + self.env.args.ma_coeff * ma_loss

            self.optimizer_dm.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
            self.optimizer_dm.step()

            all_profit_loss.update(self.env.args.mip_coeff * profit_loss.item())
            all_late_loss.update(self.env.args.mil_coeff * late_loss.item())
            all_mi_loss.update(self.env.args.mi_coeff * mi_loss.item())
            all_ma_loss.update(self.env.args.ma_coeff * ma_loss.item())
            all_loss.update(loss.item(), bs)

        torch.cuda.empty_cache()
        return (all_loss.avg, all_profit_loss.avg, all_late_loss.avg,
                all_mi_loss.avg, all_ma_loss.avg, time.time() - t)

    def dm_train(self):
        """Attention-pool head fine-tuning with periodic val evaluation and early stopping."""
        info("[LABEL] Pre-computing best-action labels via FAISS profit lookup + simulator...")
        self.precompute_best_actions()
        info("[CACHE] Pre-computing frozen backbone hidden states for all splits...")
        self.precompute_hidden_states()
        info(f"[DM TRAIN] Precomputation complete. Fine-tuning attnpool head for {self.env.args.dm_epochs} epochs.")

        scheduler_dm = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_dm, T_max=self.env.args.dm_epochs, eta_min=1e-6
        )

        best_val_score = float("-inf")
        early_stop_counter = 0
        eval_every = getattr(self.env.args, "eva_interval", 1)
        patience = getattr(self.env.args, "early_stop", 10)

        adapter_save_path = None
        if getattr(self.env.args, "save", False):
            adapter_save_path = os.path.join(
                self.env.CKPT_PATH, f"{self.env.suffix}_attnpool_best.pth"
            )

        pre_profit, pre_on_time, _, pre_time, pre_acc_hist, pre_acc_true = self.dm_test("val")
        pre_score = pre_profit + pre_on_time
        _pre_acc_str = (f"hist={pre_acc_hist:.4f} true={pre_acc_true:.4f}"
                        if pre_acc_true is not None else f"hist={pre_acc_hist:.4f}")
        info(f"[DM EVAL] pre-training baseline — val_profit={pre_profit:.4f} "
             f"val_on_time={pre_on_time:.4f} val_score={pre_score:.4f} val_acc={_pre_acc_str}")
        best_val_score = pre_score

        loss_history = []

        for epoch in range(self.env.args.ckpt_start_epoch, self.env.args.dm_epochs):
            avg_loss, avg_profit_loss, avg_late_loss, avg_mi_loss, avg_ma_loss, train_time = self.dm_train_epoch_reward()
            loss_history.append(avg_loss)

            current_lr = scheduler_dm.get_last_lr()[0]
            info(f"[DM TRAIN] epoch {epoch}/{self.env.args.dm_epochs} "
                 f"loss={avg_loss:.4f} (profit={avg_profit_loss:.4f} late={avg_late_loss:.4f} "
                 f"mi={avg_mi_loss:.4f} ma={avg_ma_loss:.4f}) "
                 f"lr={current_lr:.2e} time={train_time:.1f}s")
            scheduler_dm.step()

            if avg_loss != avg_loss:
                info(f"[DM] Stopping at epoch {epoch} due to NaN loss.")
                break

            epoch_metrics = {
                "dm/loss": avg_loss, "dm/loss_profit": avg_profit_loss, "dm/loss_late": avg_late_loss,
                "dm/loss_mi": avg_mi_loss, "dm/loss_ma": avg_ma_loss, "dm/lr": current_lr,
            }

            if epoch % eval_every == 0:
                val_profit, val_on_time, val_pmp, val_time, val_acc_hist, val_acc_true = self.dm_test("val")
                val_score = val_profit + val_on_time
                _val_acc_str = (f"hist={val_acc_hist:.4f} true={val_acc_true:.4f}"
                                if val_acc_true is not None else f"hist={val_acc_hist:.4f}")
                info(f"[DM EVAL] epoch {epoch}: val_profit={val_profit:.4f} "
                     f"val_on_time={val_on_time:.4f} val_score={val_score:.4f} "
                     f"val_acc={_val_acc_str} ({val_time:.1f}s)")

                epoch_metrics.update({
                    "val/profit": val_profit, "val/on_time": val_on_time,
                    "val/score": val_score,
                    "val/acc_hist": val_acc_hist if val_acc_hist is not None else 0.0,
                    "val/acc_true": val_acc_true if val_acc_true is not None else 0.0,
                    "val/pmp_0.1": val_pmp[0.1],
                })

                if val_score > best_val_score:
                    best_val_score = val_score
                    self.best_dm_epoch = epoch
                    early_stop_counter = 0
                    if adapter_save_path is not None:
                        adapter_state = {
                            k: v for k, v in self.value_network.state_dict().items()
                            if "pool_attn" in k or "cls_head" in k
                        }
                        torch.save(adapter_state, adapter_save_path)
                        info(f"[DM TRAIN] New best val_score={val_score:.4f}, saved attnpool head → {adapter_save_path}")
                else:
                    early_stop_counter += 1
                    info(f"[DM] no improvement ({early_stop_counter}/{patience})")

                if early_stop_counter >= patience:
                    info(f"[DM] Early stopping at epoch {epoch}. "
                         f"Best val_score={best_val_score:.4f} at epoch {self.best_dm_epoch}.")
                    break

            if getattr(self.env.args, "wandb", False):
                try:
                    wandb.log(epoch_metrics, step=epoch)
                except Exception:
                    pass

        if loss_history:
            lh = np.array(loss_history)
            finite = lh[np.isfinite(lh)]
            if finite.size > 0:
                info(f"[DM] Loss summary over {len(lh)} epochs — "
                     f"initial={lh[0]:.4f} final={lh[-1]:.4f} "
                     f"min={finite.min():.4f} max={finite.max():.4f} mean={finite.mean():.4f}")
        info(f"[DM TRAIN] Attnpool fine-tuning complete. Best val_score={best_val_score:.4f} at epoch {self.best_dm_epoch}.")


    def dm_test(self, mode="test"):
        self._ensure_scaler_fitted()
        self.model.eval()
        self.value_network.eval()
        t = time.time()

        if mode in ("val", "ori"):
            input_id = self.val_inputs
        else:
            input_id = self.test_inputs

        if not isinstance(input_id, torch.Tensor):
            input_id = torch.tensor(input_id, dtype=torch.float32)
        ori_input = input_id.to(self.env.device)

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

        # Raw (unscaled) features for LLM text serialization
        raw_state = ori_input[:, :feature_dim]

        if mode != "ori":
            X = input_id.detach().cpu().numpy()
            X = self.scaler.transform(X)
            input_id = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.env.device)
        else:
            input_id = ori_input

        state = input_id[:, :feature_dim]

        if not hasattr(self, "_faiss_index") or self._faiss_index is None:
            if isinstance(self.cost_dic, torch.Tensor):
                cost_dic_np = self.cost_dic.detach().cpu().numpy()
            else:
                cost_dic_np = np.asarray(self.cost_dic)
            self._cost_dic_data = np.ascontiguousarray(cost_dic_np[:, :-1].astype('float32'))
            self._cost_dic_y = cost_dic_np[:, -1]
            self._faiss_index = faiss.IndexFlatL2(self._cost_dic_data.shape[1])
            self._faiss_index.add(self._cost_dic_data)

        dm_acc_historical = None
        dm_acc_true = None
        local_profits = []
        profit_sum = 0.0
        profit_count = 0
        time_sum = 0
        time_count = 0

        hidden_cache_s = self.val_hidden_cache  if mode in ("val", "ori") else self.test_hidden_cache
        mask_cache_s   = self.val_mask_cache    if mode in ("val", "ori") else self.test_mask_cache
        N_eval = ori_input.shape[0]  # already limited by dm_eval_limit above

        with torch.no_grad():
            if mode == "ori":
                decision_indices = input_id[:, feature_dim].long()
                decision_prob = F.one_hot(decision_indices, num_classes=4).float().to(self.env.device)
            else:
                # Fetch cached hidden states — backbone never runs during eval
                _bs = int(getattr(self.env.args, "batch_size", 64))
                chunks = []
                for s in range(0, N_eval, _bs):
                    h = hidden_cache_s[s:s + _bs].to(self.env.device).float()
                    m = mask_cache_s[s:s + _bs].to(self.env.device)
                    chunks.append(self.value_network.forward_from_hidden(h, m).argmax(dim=1))
                action = torch.cat(chunks, dim=0)
                decision_prob = F.one_hot(action, num_classes=4).float()

            action = decision_prob.argmax(dim=1).view(-1)

            if mode != "ori":
                gt_actions = ori_input[:, feature_dim].long()
                dm_acc_historical = (action == gt_actions).float().mean().item()
                self.best_dm_accuracy = max(self.best_dm_accuracy, dm_acc_historical)
                info(f"[DM] {mode} dm_acc_historical (vs dataset): {dm_acc_historical:.4f}")

                if hasattr(self, "val_action_labels") and hasattr(self, "test_action_labels"):
                    opt_labels = self.val_action_labels if mode in ("val", "ori") else self.test_action_labels
                    opt_labels_eval = opt_labels[:N_eval].to(self.env.device)
                    dm_acc_true = (action == opt_labels_eval).float().mean().item()
                    self.best_dm_accuracy_true = max(self.best_dm_accuracy_true, dm_acc_true)
                    info(f"[DM] {mode} dm_acc_true (vs FAISS-optimal): {dm_acc_true:.4f}")

            ridx0, ridx1 = feature_list.retrieva_index[self.env.args.dataset]
            ori_cpu = ori_input.detach().cpu()
            action_np = action.detach().cpu().numpy().astype('float32')
            query_vectors = np.empty((state.shape[0], 3), dtype='float32')
            query_vectors[:, 0] = ori_cpu[:, ridx0].numpy()
            query_vectors[:, 1] = ori_cpu[:, ridx1].numpy()
            query_vectors[:, 2] = action_np

            _, nearest_indices = self._faiss_index.search(query_vectors, 1)
            nearest_indices = nearest_indices.flatten()

            matched = self._cost_dic_data[nearest_indices]
            is_exact = np.all(matched == query_vectors, axis=1)
            matched_profits = self._cost_dic_y[nearest_indices]

            if isinstance(self.avg_profit, torch.Tensor):
                avg_profit_np = self.avg_profit.detach().cpu().numpy().astype('float32')
            else:
                avg_profit_np = np.asarray(self.avg_profit, dtype='float32')
            fallback_profits = avg_profit_np[action_np.astype(np.int64)]
            local_profits_np = np.where(is_exact, matched_profits, fallback_profits)
            profit_sum = float(local_profits_np.sum())
            profit_count = int(local_profits_np.shape[0])
            local_profits = local_profits_np.tolist()

            selected_embedding = torch.sum(
                decision_prob.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1
            )
            predicted_tokens = self.model(
                input_id[:, :feature_dim],
                selected_embedding,
                ori_input[:, feature_dim + 1:]
            )
            time_sum += predicted_tokens[-1].argmax(dim=1).sum().item()
            time_count += predicted_tokens[-1].shape[0]

        profit = (profit_sum / profit_count) if profit_count > 0 else 0.0
        on_time_ratio = (time_sum / time_count) if time_count > 0 else 0.0

        if local_profits:
            sorted_profits = np.sort(np.asarray(local_profits, dtype=np.float32))
            profit_min_percent = {}
            for thr in [0.1, 0.2, 0.3]:
                idx = max(0, min(len(sorted_profits) - 1, int(thr * len(sorted_profits))))
                profit_min_percent[thr] = float(sorted_profits[idx])
        else:
            profit_min_percent = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0}

        if profit > self.best_p:
            self.best_p = profit
        if on_time_ratio > self.best_o:
            self.best_o = on_time_ratio
        self.best_pmp1 = max(self.best_pmp1, profit_min_percent[0.1])
        self.best_pmp2 = max(self.best_pmp2, profit_min_percent[0.2])
        self.best_pmp3 = max(self.best_pmp3, profit_min_percent[0.3])

        return profit, on_time_ratio, profit_min_percent, time.time() - t, dm_acc_historical, dm_acc_true


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
        label_dim = len(feature_list.label[self.env.args.dataset])

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
