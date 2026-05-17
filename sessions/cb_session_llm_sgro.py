# SGRO: Simulator-Guided Reward Optimization for the decision-maker.
# Pre-computes a [N, 4] reward matrix (FAISS profit + soft p_ontime from simulator),
# then trains with a combined loss: alpha_sft * CE(best-action) + alpha_reward * KL_AWR.
# Checkpoint selection is by estimated val reward (argmax Q -> lookup in val reward matrix).
# Training: lm_head fine-tuned with SGRO loss — transformer body fully frozen.
# Inference: generate_action() drives autoregressive decoding; forward() is used for training only.

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
from models.llm_model_sgro import LLMValueNetwork
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
        # Use provided VN or build one from the new class (which already freezes the backbone
        # and leaves the small head trainable).
        if value_network is None:
            model_name = getattr(self.env.args, "hf_model_name", None)
            batch_size = getattr(self.env.args, "batch_size", 32)
            self.value_network = LLMValueNetwork(self.env, model_name=model_name, batch_size=batch_size)
            info(f"Initialized LLMValueNetwork: {model_name} (batch={batch_size})")
        else:
            self.value_network = value_network
            name = getattr(self.value_network.backbone.config, "_name_or_path", "unknown")
            info(f"Initialized value network from provided instance: {name}")
        
        # Create optimizer for the trainable head only
        trainable_params = [p for p in self.value_network.parameters() if p.requires_grad]
        if not trainable_params:
            info("WARNING: No trainable parameters found in value_network. Check lm_head requires_grad.")
        self.optimizer_dm = torch.optim.Adam(
            trainable_params, lr=self.env.args.dm_lr, weight_decay=self.env.args.dm_decay_coeff
        )
        self.value_network.train()

    def precompute_reward_matrix(self, split="train"):
        """Build [N, 4] reward matrix for `split` using FAISS profit + soft simulator p_ontime.

        Soft p_ontime = softmax(pred_logits)[:, 1] gives richer gradient signal than hard argmax.
        Results stored in self.train_reward_matrix or self.val_reward_matrix.
        Saved/loaded from disk if --reward_path is set; re-computed if --precompute_rewards is set.
        """
        dataset = self.env.args.dataset
        reward_path = getattr(self.env.args, "reward_path", None)
        force_recompute = getattr(self.env.args, "precompute_rewards", False)

        # Attempt to load from cache
        cache_file = None
        if reward_path is not None:
            os.makedirs(reward_path, exist_ok=True)
            cache_file = os.path.join(reward_path, f"{dataset}_{split}_rewards.pt")
            if not force_recompute and os.path.exists(cache_file):
                loaded = torch.load(cache_file, map_location="cpu")
                setattr(self, f"{split}_reward_matrix", loaded)
                info(f"[SGRO] Loaded {split} reward matrix from {cache_file} — shape {loaded.shape}")
                return

        self.model.eval()
        self._ensure_scaler_fitted()

        inputs = self.train_inputs if split == "train" else self.val_inputs
        if isinstance(inputs, torch.Tensor):
            X_np = inputs.detach().cpu().numpy().astype(np.float32)
        else:
            X_np = np.asarray(inputs, dtype=np.float32)

        N = X_np.shape[0]
        feature_dim = len(
            feature_list.product_info[dataset]
            + feature_list.order_info[dataset]
            + feature_list.customer_info[dataset]
            + feature_list.shipping_info[dataset]
        )

        Xs_t = torch.from_numpy(self.scaler.transform(X_np).astype(np.float32)).to(self.env.device)
        ori_t = torch.from_numpy(X_np).to(self.env.device)

        if isinstance(self.cost_dic, torch.Tensor):
            cost_dic_np = self.cost_dic.detach().cpu().numpy()
        else:
            cost_dic_np = np.asarray(self.cost_dic)
        cost_data = np.ascontiguousarray(cost_dic_np[:, :-1].astype("float32"))
        cost_y = cost_dic_np[:, -1]
        index = faiss.IndexFlatL2(cost_data.shape[1])
        index.add(cost_data)

        ridx0, ridx1 = feature_list.retrieva_index[dataset]
        B = int(self.env.args.batch_size)
        n_batches = (N + B - 1) // B
        log_every = max(1, n_batches // 10)

        all_rewards = torch.zeros(N, 4, dtype=torch.float32)

        with torch.no_grad():
            for batch_idx in range(n_batches):
                start = batch_idx * B
                end = min(start + B, N)
                bs = end - start

                state_b = Xs_t[start:end, :feature_dim]
                ori_b = ori_t[start:end]

                if batch_idx % log_every == 0:
                    info(f"[SGRO] precompute_reward_matrix ({split}): batch {batch_idx}/{n_batches}")

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

                    # Soft on-time probability from simulator (richer than hard argmax)
                    a_batch = torch.full((bs,), a, dtype=torch.long, device=self.env.device)
                    onehot = F.one_hot(a_batch, num_classes=4).float()
                    selected_emb = torch.sum(
                        onehot.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1
                    )
                    pred_tokens = self.model(state_b, selected_emb, ori_b[:, feature_dim + 1:])
                    p_ontime = F.softmax(pred_tokens[-1], dim=1)[:, 1]  # soft probability

                    total_reward = profit_t + self.env.args.otr_reward_coeff * p_ontime
                    all_rewards[start:end, a] = total_reward.cpu()

        setattr(self, f"{split}_reward_matrix", all_rewards)
        info(f"[SGRO] {split} reward matrix computed — shape {all_rewards.shape}, "
             f"mean {all_rewards.mean():.4f}")

        if cache_file is not None:
            torch.save(all_rewards, cache_file)
            info(f"[SGRO] Saved {split} reward matrix to {cache_file}")

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
            if self.env.args.wandb:
                wandb.log({"loss/classification_loss":classification_loss}, epoch)
            self.test('val')
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
        """SGRO training step: alpha_sft * CE(best-action) + alpha_reward * KL_AWR."""
        assert hasattr(self, "train_reward_matrix"), \
            "train_reward_matrix not found — call precompute_reward_matrix('train') first."

        self.model.eval()
        self.value_network.train()

        self._ensure_scaler_fitted()
        X = self.train_inputs
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=np.float32)

        N = X_np.shape[0]
        B = int(min(self.env.args.batch_size, N))
        indices = torch.randint(0, N, (B,))

        feature_dim = len(
            feature_list.product_info[self.env.args.dataset]
            + feature_list.order_info[self.env.args.dataset]
            + feature_list.customer_info[self.env.args.dataset]
            + feature_list.shipping_info[self.env.args.dataset]
        )

        # Raw (unscaled) features — LLM serializes them as text by feature name
        raw_state = torch.from_numpy(X_np[indices.numpy()].astype(np.float32)).to(self.env.device)[:, :feature_dim]

        rewards_b = self.train_reward_matrix[indices].to(self.env.device)  # [B, 4]

        q_values = self.value_network(raw_state)  # [B, 4]

        # SFT term: CE toward the highest-reward action label
        best_actions = rewards_b.argmax(dim=1)  # [B]
        ce_loss = F.cross_entropy(q_values, best_actions, label_smoothing=0.05)

        # AWR term: KL divergence toward softmax-normalized advantage distribution
        baseline = rewards_b.mean(dim=-1, keepdim=True)
        advantages = rewards_b - baseline
        tau = float(getattr(self.env.args, "tau", 1.0))
        target_dist = F.softmax(advantages / tau, dim=-1).detach()  # [B, 4]
        log_probs = F.log_softmax(q_values, dim=-1)
        reward_loss = F.kl_div(log_probs, target_dist, reduction="batchmean")

        alpha_sft = float(getattr(self.env.args, "alpha_sft", 1.0))
        alpha_reward = float(getattr(self.env.args, "alpha_reward", 1.0))
        total_loss = alpha_sft * ce_loss + alpha_reward * reward_loss

        self.optimizer_dm.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        self.optimizer_dm.step()

        avg_reward = float(rewards_b.max(dim=1).values.mean().item())
        return avg_reward, 0.0, avg_reward, float(total_loss.item())


    def _estimate_val_reward(self):
        """Estimate avg reward on val set: argmax Q-values -> lookup in val_reward_matrix."""
        assert hasattr(self, "val_reward_matrix"), "val_reward_matrix not precomputed."
        self.value_network.eval()
        X = self.val_inputs
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
        B = int(self.env.args.batch_size)
        best_actions_list = []
        with torch.no_grad():
            for i in range(0, N, B):
                # Raw (unscaled) features for LLM text serialization
                raw_state = torch.from_numpy(X_np[i:i+B, :feature_dim]).to(self.env.device)
                q = self.value_network(raw_state)  # [b, 4]
                best_actions_list.append(q.argmax(dim=1).cpu())
        best_actions = torch.cat(best_actions_list)  # [N]

        val_rewards = self.val_reward_matrix  # [N, 4]
        chosen = val_rewards[torch.arange(N), best_actions]
        return float(chosen.mean().item())

    def save_dm_model(self, epoch):
        """Save value_network state; remove the previous best checkpoint."""
        path = os.path.join(self.env.CKPT_PATH, f"{self.env.suffix}_dm_epoch{epoch}.pth")
        torch.save(self.value_network.state_dict(), path)
        if self.best_dm_epoch != epoch:
            old_path = os.path.join(
                self.env.CKPT_PATH, f"{self.env.suffix}_dm_epoch{self.best_dm_epoch}.pth"
            )
            if os.path.exists(old_path):
                os.system(f"rm {old_path}")

    def dm_train(self):
        """SGRO training loop with val-reward-based checkpoint selection."""
        info("[SGRO] Pre-computing train reward matrix (FAISS profit + soft p_ontime)...")
        self.precompute_reward_matrix("train")
        info("[SGRO] Pre-computing val reward matrix for checkpoint selection...")
        self.precompute_reward_matrix("val")
        info(f"[SGRO] Reward matrices ready. Starting training for {self.env.args.dm_epochs} epochs.")

        best_val_reward = float("-inf")
        t0 = time.time()

        for epoch in range(self.env.args.ckpt_start_epoch, self.env.args.dm_epochs):
            avg_r, on_time, prof, loss = self.dm_train_epoch()

            # Val reward estimate for checkpoint selection
            self.value_network.eval()
            val_reward = self._estimate_val_reward()
            self.value_network.train()

            info(f"[DM] epoch {epoch}/{self.env.args.dm_epochs} "
                 f"train_reward={avg_r:.4f} val_reward={val_reward:.4f} loss={loss:.4f}")

            if getattr(self.env.args, "wandb", False):
                try:
                    wandb.log(
                        {"dm/train_reward": avg_r, "dm/val_reward": val_reward,
                         "dm/loss": loss, "dm/epoch_time_sec": time.time() - t0},
                        step=epoch
                    )
                except Exception:
                    pass

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                self.best_dm_epoch = epoch
                if getattr(self.env.args, "save", False):
                    self.save_dm_model(epoch)
                info(f"[DM] New best val_reward={val_reward:.4f} at epoch {epoch}")

            t0 = time.time()



    def dm_test(self, mode="test"):
        import numpy as np
        import torch
        import torch.nn.functional as F
        import faiss
        from tools.logger import info

        self._ensure_scaler_fitted()
        self.model.eval()
        self.value_network.eval()
        t = time.time()

        # pick inputs
        if mode in ("val", "ori"):
            input_id = self.val_inputs
        else:
            input_id = self.test_inputs

        # ensure torch tensor for downstream ops
        if not isinstance(input_id, torch.Tensor):
            input_id = torch.tensor(input_id, dtype=torch.float32)
        ori_input = input_id.to(self.env.device)
        
        # Optional: limit evaluation size for faster LLM runs
        limit = getattr(self.env.args, "dm_eval_limit", None)
        if limit:
            input_id = input_id[:int(limit)]
            ori_input = ori_input[:int(limit)]


        # feature dimension
        feature_dim = len(
            feature_list.product_info[self.env.args.dataset]
            + feature_list.order_info[self.env.args.dataset]
            + feature_list.customer_info[self.env.args.dataset]
            + feature_list.shipping_info[self.env.args.dataset]
        )

        # Raw (unscaled) features for LLM text serialization — must be set before input_id is rescaled
        raw_state = ori_input[:, :feature_dim]

        # scale (on CPU, NumPy), then back to torch on device
        if mode != "ori":
            X = input_id.detach().cpu().numpy()
            X = self.scaler.transform(X)  # <-- sklearn wants numpy
            input_id = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.env.device)
        else:
            input_id = ori_input  # already on device

        # state that goes into the simulator (scaled)
        state = input_id[:, :feature_dim]

        # ---- FAISS setup ----
        # cost_dic_data: (N, D) float32 numpy contiguous
        # cost_dic_y: (N,) values (torch or numpy both ok; we'll convert to float when used)
        if isinstance(self.cost_dic, torch.Tensor):
            cost_dic_np = self.cost_dic.detach().cpu().numpy()
        else:
            cost_dic_np = np.asarray(self.cost_dic)
        self.cost_dic_data = np.ascontiguousarray(cost_dic_np[:, :-1].astype('float32'))
        self.cost_dic_y    = cost_dic_np[:, -1]  # keep as numpy for simple indexing

        index = faiss.IndexFlatL2(self.cost_dic_data.shape[1])
        index.add(self.cost_dic_data)

        profit_sum = 0.0
        profit_count = 0
        time_sum = 0
        time_count = 0
        local_profits = []

        with torch.no_grad():
            if mode == "ori":
                decision_indices = input_id[:, feature_dim].long()
                decision_prob = F.one_hot(decision_indices, num_classes=4).float().to(self.env.device)
                decision_prob_value = decision_prob
            else:
                # generate_action() runs the LLM autoregressively and parses the output digit
                _bs = int(getattr(self.env.args, "batch_size", 64))
                chunks = [
                    self.value_network.generate_action(raw_state[s:s + _bs])
                    for s in range(0, raw_state.shape[0], _bs)
                ]
                action = torch.cat(chunks, dim=0)
                decision_prob = F.one_hot(action, num_classes=4).float()
                decision_prob_value = decision_prob

            action = decision_prob_value.argmax(dim=1)

            dm_acc = None
            if mode != "ori":
                gt_actions = ori_input[:, feature_dim].long()
                dm_acc = (action == gt_actions).float().mean().item()
                self.best_dm_accuracy = max(self.best_dm_accuracy, dm_acc)
                info(f"[GEN] {mode} generation accuracy (LLM vs ground truth): {dm_acc:.4f}")

            # ---- FAISS queries ----
            # Build (num_samples, 3) float32 numpy query vectors: [feat_i, feat_j, action]
            ridx0, ridx1 = feature_list.retrieva_index[self.env.args.dataset]
            ori_cpu = ori_input.detach().cpu()
            query_vectors = np.empty((state.shape[0], 3), dtype='float32')
            query_vectors[:, 0] = ori_cpu[:, ridx0].numpy()
            query_vectors[:, 1] = ori_cpu[:, ridx1].numpy()
            query_vectors[:, 2] = action.detach().cpu().numpy().astype('float32')

            # search
            _, nearest_indices = index.search(query_vectors, 1)  # (B,1)
            nearest_indices = nearest_indices.flatten()

            for i in range(len(query_vectors)):
                q = query_vectors[i]
                n = self.cost_dic_data[nearest_indices[i]]
                if np.array_equal(q, n):
                    selected_y = self.cost_dic_y[nearest_indices[i]]
                    # selected_y might be numpy scalar -> convert to float
                    selected_y = float(selected_y)
                else:
                    # avg_profit is per-action; ensure it’s indexable and numeric
                    # action[i] is a tensor on device -> move to cpu int
                    ai = int(action[i].detach().cpu().item())
                    # self.avg_profit can be list/np/torch; normalize to float
                    if isinstance(self.avg_profit, torch.Tensor):
                        selected_y = float(self.avg_profit[ai].detach().cpu().item())
                    else:
                        selected_y = float(self.avg_profit[ai])

                profit_sum += selected_y
                profit_count += 1
                local_profits.append(selected_y)

            # ---- time/on-time prediction via your model ----
            selected_embedding = torch.sum(
                decision_prob.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1
            )
            predicted_tokens = self.model(
                input_id[:, :feature_dim],
                selected_embedding,
                ori_input[:, feature_dim + 1:]
            )
            # Your original logic: last token's argmax==on-time
            time_sum += predicted_tokens[-1].argmax(dim=1).sum().item()
            time_count += predicted_tokens[-1].shape[0]

        # aggregate metrics
        profit = (profit_sum / profit_count) if profit_count > 0 else 0.0

        # percentiles on profits
        if local_profits:
            sorted_profits = np.sort(np.asarray(local_profits, dtype=np.float32))
            thresholds = [0.1, 0.2, 0.3]
            profit_min_percent = {}
            for thr in thresholds:
                idx = max(0, min(len(sorted_profits)-1, int(thr * len(sorted_profits))))
                profit_min_percent[thr] = float(sorted_profits[idx])
        else:
            profit_min_percent = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0}

        on_time_ratio = (time_sum / time_count) if time_count > 0 else 0.0
        
        if profit > self.best_p:
            self.best_p = profit
        if on_time_ratio > self.best_o:
            self.best_o = on_time_ratio
        self.best_pmp1 = max(self.best_pmp1, profit_min_percent[0.1])
        self.best_pmp2 = max(self.best_pmp2, profit_min_percent[0.2])
        self.best_pmp3 = max(self.best_pmp3, profit_min_percent[0.3])

        # (optional) log to wandb if enabled
        if self.env.args.wandb:
            import wandb
            wandb.log({
                "dm/profit": profit,
                "dm/on_time": on_time_ratio,
                "dm/pmp_0.1": profit_min_percent[0.1],
                "dm/pmp_0.2": profit_min_percent[0.2],
                "dm/pmp_0.3": profit_min_percent[0.3],
            })

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
            value_network_output = self.value_network.score_all_actions(state)  # [N, 4]
            decision_prob_value = (value_network_output
                                   == value_network_output.max(dim=1, keepdim=True).values).float()
            decision_prob = decision_prob_value

            action = value_network_output.argmax(dim=1).squeeze()  
            
            
            

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