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
from models.llm_model import LLMValueNetwork
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
        self.best_p = 0
        self.best_o = 0
        self.best_pmp1 = 0
        self.best_pmp2 = 0
        self.best_pmp3 = 0

        self.min_profit, self.max_profit = float('inf'), float('-inf')
        self.min_on_time, self.max_on_time = float('inf'), float('-inf')

    def _ensure_faiss(self):
        if hasattr(self, "_faiss_ready") and self._faiss_ready:
            return
        import numpy as np, faiss, torch
        # normalize to numpy float32 contiguous once
        if isinstance(self.cost_dic, torch.Tensor):
            cost_dic_np = self.cost_dic.detach().cpu().numpy()
        else:
            cost_dic_np = np.asarray(self.cost_dic)
        self.cost_dic_data = np.ascontiguousarray(cost_dic_np[:, :-1].astype("float32"))
        self.cost_dic_y    = cost_dic_np[:, -1]

        self.faiss_index = faiss.IndexFlatL2(self.cost_dic_data.shape[1])
        self.faiss_index.add(self.cost_dic_data)
        self._faiss_ready = True
        
    def _ensure_scaler_fitted(self):
        feature_dim = getattr(self.model, "feature_dim", self.train_inputs.shape[1])
        X = self.train_inputs
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        # fit once if not already done
        from sklearn.utils.validation import check_is_fitted
        try:
            check_is_fitted(self.scaler)
        except Exception:
            self.scaler.fit(X[:, :feature_dim])
        Xs = self.scaler.transform(X).astype(np.float32)
        self.train_inputs_scaled = torch.from_numpy(Xs).to(self.env.device)

            
    def init_value_network(self, value_network=None):
        # Use provided VN or build one from the new class (which already freezes the backbone
        # and leaves the small head trainable).
        if value_network is None:
            model_name = getattr(self.env.args, "llm_name", "google/gemma-3-270m-it")
            batch_size = getattr(self.env.args, "batch_size", 32)
            self.value_network = LLMValueNetwork(self.env, model_name=model_name, batch_size=batch_size)
            info(f"Initialized LLMValueNetwork: {model_name} (batch={batch_size})")
        else:
            self.value_network = value_network
            info("Initialized value network from provided instance.")
        
        # Create optimizer for the trainable head only
        trainable_params = [p for p in self.value_network.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters in value_network. Ensure adapter/cls_head are requires_grad=True.")
        self.optimizer_dm = torch.optim.Adam(
            trainable_params, lr=self.env.args.dm_lr, weight_decay=self.env.args.dm_decay_coeff
        )
        self.value_network.train()


    def train_epoch(self):
        t = time.time()
        self.model.train()
        self.total_epoch += 1
        all_classification_loss = AverageMeter()

        feature_dim = len(
            feature_list.product_info[self.env.args.dataset]
            + feature_list.order_info[self.env.args.dataset]
            + feature_list.customer_info[self.env.args.dataset]
            + feature_list.shipping_info[self.env.args.dataset]
        )
        label_dim = len(feature_list.label[self.env.args.dataset])  # sequence length T
        output_dim = 1                                              # MDN predicts 1D per step

        self._ensure_scaler_fitted()

        for input_id in tqdm(self.loader):
            ori_input = input_id.to(self.env.device)
            input_id = self.scaler.transform(input_id)
            input_id = torch.as_tensor(input_id, dtype=torch.float32, device=self.env.device)

            # ---- forward through MDN ----
            generated_mdn_params = self.model(
                input_id[:, :feature_dim],
                ori_input[:, feature_dim].long(),
                ori_input[:, feature_dim + 1 : feature_dim + 1 + label_dim]
            )

            # ---- targets must be [B, T, D] for compute_loss ----
            # Get the same slice that defines T (label_dim)
            targets = (
                ori_input[:, feature_dim + 1 : feature_dim + 1 + label_dim]  # [B, T]
                .float()
                .unsqueeze(-1)  # -> [B, T, 1]
                .to(self.env.device)
            )

            loss = self.model.compute_loss(targets, generated_mdn_params)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            all_classification_loss.update(loss.item())

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

    def _mdn_point_predict(self, mus, logpi):
        """
        Convert MDN params to a point estimate by selecting the MAP component.
        mus:   [B, K, D]
        logpi: [B, K]
        Returns: y_hat [B, D]
        """
        comp = logpi.argmax(dim=-1)  # [B]
        idx = torch.arange(mus.size(0), device=mus.device)
        y_hat = mus[idx, comp]       # [B, D]
        return y_hat

    def dm_train_epoch(self):
        """
        One REINFORCE step for the (trainable) LLM head.
        Returns: avg_reward, on_time_mean, profit_mean, loss
        """
        import torch
        from torch.distributions import Categorical
        from tools import feature_list

        assert self.optimizer_dm is not None, "optimizer_dm is None — create it from the value head's trainable params."

        # --- setup ---
        self._ensure_faiss()
        self._ensure_scaler_fitted()
        self.model.eval()           # simulator frozen during policy update
        self.value_network.train()  # train policy head

        # --- dims ---
        dataset = self.env.args.dataset
        feature_dim = len(
            feature_list.product_info[dataset]
            + feature_list.order_info[dataset]
            + feature_list.customer_info[dataset]
            + feature_list.shipping_info[dataset]
        )
        label_dim = len(feature_list.label[dataset])

        # --- sample one mini-batch ---
        B = int(min(self.env.args.batch_size, self.train_inputs_scaled.size(0)))
        idx = torch.randint(0, self.train_inputs_scaled.size(0), (B,), device=self.env.device)

        # scaled state for policy
        state_scaled = self.train_inputs_scaled
        if not isinstance(state_scaled, torch.Tensor):
            state_scaled = torch.tensor(state_scaled, dtype=torch.float32, device=self.env.device)
        else:
            state_scaled = state_scaled.to(self.env.device)
        state = state_scaled[idx, :feature_dim]  # [B, feature_dim]

        # original (for labels/targets if needed)
        if isinstance(self.train_inputs, torch.Tensor):
            ori_full = self.train_inputs.to(self.env.device)
        else:
            ori_full = torch.tensor(self.train_inputs, dtype=torch.float32, device=self.env.device)
        ori_b = ori_full[idx]

        # --- policy forward (micro-batched if large) ---
        mb = 128
        logits_chunks = [self.value_network(state[i:i + mb]) for i in range(0, state.size(0), mb)]
        logits = torch.cat(logits_chunks, dim=0)  # [B, num_actions]

        dist   = Categorical(logits=logits)
        action = dist.sample()                    # [B]
        logp   = dist.log_prob(action)            # [B]

        # feed the CHOSEN action into the simulator
        shipping_mode = action.long()
        tgt = ori_b[:, feature_dim + 1 : feature_dim + 1 + label_dim]

        # --- run simulator (MDN) ---
        with torch.no_grad():
            out = self.model(state, shipping_mode, tgt)

        # Robustly unpack MDN outputs (supports (mus,sig,logpi) or [ ... , (mus,sig,logpi) ])
        def _unpack_mdn(o):
            import torch as _torch
            if isinstance(o, tuple) and len(o) == 3 and all(isinstance(t, _torch.Tensor) for t in o):
                return o
            if isinstance(o, (list, tuple)) and len(o) > 0:
                last = o[-1]
                if isinstance(last, tuple) and len(last) == 3 and all(isinstance(t, _torch.Tensor) for t in last):
                    return last
            raise RuntimeError(f"Unexpected simulator output format: {type(o)}")

        mus_t, sigmas_t, logpi_t = _unpack_mdn(out)  # [B,K,D], [B,K,D], [B,K]
        D = mus_t.size(-1)

        # --- expected prediction from mixture ---
        p = torch.softmax(logpi_t, dim=-1)               # [B, K]
        y_exp = (p.unsqueeze(-1) * mus_t).sum(dim=1)     # [B, D]

        if D == 1:
            # Single scalar output per sample: treat as "on-time score" after sigmoid
            y_scalar = y_exp.squeeze(-1)                # [B]
            reward = torch.sigmoid(y_scalar)            # [B] in (0,1)
            on_time_hat = (reward >= 0.5).float()       # threshold the probability
        else:
            # Multi-D fallback: use last dimension as on-time score
            y_on = y_exp[..., -1]                       # [B]
            reward = torch.sigmoid(y_on)
            on_time_hat = (reward >= 0.5).float()

        profit = on_time_hat  # if profit == on_time in your setup

        # --- REINFORCE objective ---
        policy_loss = -(reward.detach() * logp).mean()

        self.optimizer_dm.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        self.optimizer_dm.step()

        return (
            float(reward.mean().item()),
            float(on_time_hat.mean().item()),
            float(profit.mean().item()),
            float(policy_loss.item()),
        )


    def dm_train(self):
        """
        Simple training loop for the decision-maker head.
        Tracks best avg_reward and updates best_* fields for logging.
        """
        import time
        from tools.logger import info

        best_reward = float("-inf")
        t0 = time.time()

        for epoch in range(self.env.args.ckpt_start_epoch, self.env.args.dm_epochs):
            avg_r, on_time, prof, loss = self.dm_train_epoch()
            info(f"[DM] epoch {epoch}/{self.env.args.dm_epochs} "
                f"reward={avg_r:.4f} profit={prof:.4f} on_time={on_time:.4f} loss={loss:.4f}")

            # optional: wandb
            if getattr(self.env.args, "wandb", False):
                try:
                    import wandb
                    wandb.log(
                        {"dm/avg_reward": avg_r, "dm/profit": prof, "dm/on_time": on_time, "dm/loss": loss,
                        "dm/epoch_time_sec": time.time() - t0},
                        step=epoch
                    )
                except Exception:
                    pass

            # track bests
            if avg_r > best_reward:
                best_reward = avg_r
                # these fields exist in your logger paths; keep them updated
                self.best_o = max(getattr(self, "best_o", 0.0), on_time)
                self.best_p = max(getattr(self, "best_p", 0.0), prof)
                self.best_dm_epoch = epoch

            t0 = time.time()


    def dm_test(self, mode="test"):
        import numpy as np
        import torch
        import torch.nn.functional as F
        import faiss
        from tools.logger import info
        import time
        from tools import feature_list

        self._ensure_scaler_fitted()
        self.model.eval()
        self.value_network.eval()
        t = time.time()

        # pick inputs
        if mode in ("val", "ori"):
            input_id = self.val_inputs
        else:
            input_id = self.test_inputs

        # ensure torch tensor on device
        if not isinstance(input_id, torch.Tensor):
            input_id = torch.tensor(input_id, dtype=torch.float32)
        ori_input = input_id.to(self.env.device)

        # Optional limit
        limit = getattr(self.env.args, "dm_eval_limit", None)
        if limit:
            input_id = input_id[:int(limit)]
            ori_input = ori_input[:int(limit)]

        # feature/label dims
        feature_dim = len(
            feature_list.product_info[self.env.args.dataset]
            + feature_list.order_info[self.env.args.dataset]
            + feature_list.customer_info[self.env.args.dataset]
            + feature_list.shipping_info[self.env.args.dataset]
        )
        label_dim = len(feature_list.label[self.env.args.dataset])

        # scale on CPU, back to device
        X_cpu = input_id.detach().cpu().numpy()
        X_np = self.scaler.transform(X_cpu)
        X = torch.as_tensor(X_np, dtype=torch.float32, device=self.env.device)
        state = X[:, :feature_dim]

        # LLM head -> greedy actions
        with torch.no_grad():
            # micro-batch in case state is big
            mb = 128
            outs = []
            for i in range(0, state.size(0), mb):
                outs.append(self.value_network(state[i:i+mb]))
            logits = torch.cat(outs, dim=0)                # [B, A]
            actions = logits.argmax(dim=-1)                # [B]

            # prepare simulator inputs
            shipping_mode = actions.long()
            tgt = ori_input[:, feature_dim + 1 : feature_dim + 1 + label_dim]

            # MDN forward (correct signature)
            # mus, sigmas, logpi = self.model(state, shipping_mode, tgt)
            generated = self.model(state, shipping_mode, tgt)
            mus, sigmas, logpi = generated[-1]  # take the last step’s (mus, sigmas, logpi)

            # point prediction from mixtures
            comp = logpi.argmax(dim=-1)                    # [B]
            idx = torch.arange(mus.size(0), device=mus.device)
            y_hat = mus[idx, comp]                         # [B, D]

            on_time_hat = (y_hat[..., 0] >= 0.5).float()
            avg_on_time = on_time_hat.mean().item()

        return {
            "on_time_mean": avg_on_time,
            "time": time.time() - t,
            "num_samples": state.size(0),
        }



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
                    

                
                # MDN inference: take mean of most likely Gaussian per timestep and round to nearest int
                generated_mdn_params = self.model(input_chunk[:, :feature_dim], ori_chunk[:, feature_dim].long(), ori_chunk[:, feature_dim + 1:])
                predicted_values = []
                for t, (mus, sigmas, logpi) in enumerate(generated_mdn_params):
                    selected_gauss = logpi.argmax(dim=-1)
                    idx = selected_gauss.view(-1, 1, 1).expand(-1, 1, mus.size(-1))
                    pred_t = mus.gather(1, idx).squeeze(1)
                    if pred_t.size(-1) == 1:
                        pred_t = pred_t.squeeze(-1)
                    predicted_values.append(pred_t)
                predicted_tokens = torch.stack(predicted_values, dim=1)

                
                class_labels = ori_chunk[:, -label_dim:].long().to(self.env.device)

                
                
                




                
                for j in range(label_dim):
                    # predicted_tokens: [B, seq_len] or [B, seq_len, D]
                    if predicted_tokens.dim() == 3:
                        pred_cont = predicted_tokens[:, j, 0]
                    else:
                        pred_cont = predicted_tokens[:, j]
                    predicted = pred_cont.round().long()
                    correct_preds[j] += (predicted == class_labels[:, j]).sum().item()
                    total_samples[j] += class_labels.size(0)
                    for value in predicted.cpu().numpy():
                        v = int(value)
                        label_value_counts[j][v] = label_value_counts[j].get(v, 0) + 1

        
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
    
