"""
ML baseline decision-makers for supply-chain shipping mode selection.

Trains three baselines on the same FAISS-optimal action labels used by VocabAlign,
then evaluates them with the identical FAISS profit + simulator on-time metrics.
Output logs match the RL/LLM format so existing comparison scripts work unchanged.

Baselines:
  random     — uniform random action in [0, 1, 2, 3]
  historical — always predict the most-frequent action in training data
  rf         — RandomForestClassifier (sklearn)
  xgb        — XGBClassifier (xgboost, optional; falls back to rf if unavailable)

Usage:
    python3 main/cb_main_ml.py --baseline rf --dataset DataCo --train_frac 0.10 \
        --ckpt <simulator_ckpt.pth> --out_dir <log_dir>
"""

import sys
import os
import glob
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
import faiss

from environments.environment import Env
from loaders.s_loader import S_Loader
from models.s_model import S_SimDec
from sessions.cb_session_llm import CB_Session
from tools import feature_list
from tools.logger import info
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu",     type=int,   default=1)
    parser.add_argument("--device_id",   type=int,   default=0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dataset",     type=str,   default="DataCo",
                        choices=["DataCo", "GlobalStore", "OAS", "DataCo_OOD"])
    parser.add_argument("--train_frac",  type=float, default=1.0)
    parser.add_argument("--ckpt",        type=str,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--embed_dim",   type=int,   default=64)
    parser.add_argument("--encoder_num_layers", type=int, default=1)
    parser.add_argument("--decoder_num_layers", type=int, default=1)
    parser.add_argument("--otr_reward_coeff", type=float, default=2.0)
    parser.add_argument("--soft_label_temp",  type=float, default=1.0)
    parser.add_argument("--dm_eval_limit",    type=int,   default=None)
    parser.add_argument("--baseline",    type=str,   default="rf",
                        choices=["random", "historical", "rf", "xgb"],
                        help="Which baseline to train and evaluate")
    parser.add_argument("--out_dir",     type=str,   default=None,
                        help="Directory for log file (default: current dir)")
    # Dummy flags required by Env/S_Loader/CB_Session but unused here
    parser.add_argument("--lr",          type=float, default=0.01)
    parser.add_argument("--dm_lr",       type=float, default=0.01)
    parser.add_argument("--epochs",      type=int,   default=0)
    parser.add_argument("--dm_epochs",   type=int,   default=0)
    parser.add_argument("--early_stop",  type=int,   default=50)
    parser.add_argument("--eva_interval",type=int,   default=1)
    parser.add_argument("--train_mode",  type=int,   default=2)
    parser.add_argument("--decay_coeff", type=float, default=0.00001)
    parser.add_argument("--dm_decay_coeff", type=float, default=0.0005)
    parser.add_argument("--mi_coeff",    type=float, default=10)
    parser.add_argument("--ma_coeff",    type=float, default=1)
    parser.add_argument("--mip_coeff",   type=float, default=1)
    parser.add_argument("--mil_coeff",   type=float, default=1)
    parser.add_argument("--reward_smoothing_factor", type=float, default=0.5)
    parser.add_argument("--ckpt_start_epoch", type=int, default=0)
    parser.add_argument("--ckpt_dir",    type=str,   default=None)
    parser.add_argument("--wandb",       type=int,   default=0)
    parser.add_argument("--save",        type=int,   default=0)
    # Ablation flags (not used, but Env reads args generically)
    parser.add_argument("--pool_init",   type=str,   default="vocab")
    parser.add_argument("--pool_type",   type=str,   default="attention")
    parser.add_argument("--no_soft_labels", action="store_true", default=False)
    parser.add_argument("--hf_model_name", type=str, default="gpt2")
    return parser.parse_args()


def _eval_actions(actions_np, test_inputs, model, env, cost_dic, avg_profit):
    """Evaluate a fixed action array on the test split.

    Returns (profit, on_time_ratio, profit_min_percent_dict, local_profits_list).
    Replicates the FAISS lookup + simulator on-time logic from CB_Session.dm_test().
    """
    if isinstance(cost_dic, torch.Tensor):
        cost_dic_np = cost_dic.detach().cpu().numpy()
    else:
        cost_dic_np = np.asarray(cost_dic)
    cost_data = np.ascontiguousarray(cost_dic_np[:, :-1].astype("float32"))
    cost_y    = cost_dic_np[:, -1]
    index = faiss.IndexFlatL2(cost_data.shape[1])
    index.add(cost_data)

    if isinstance(avg_profit, torch.Tensor):
        avg_profit_np = avg_profit.detach().cpu().numpy().astype("float32")
    else:
        avg_profit_np = np.asarray(avg_profit, dtype="float32")

    dataset = env.args.dataset
    ridx0, ridx1 = feature_list.retrieva_index[dataset]
    feat_dim = len(
        feature_list.product_info[dataset] + feature_list.order_info[dataset]
        + feature_list.customer_info[dataset] + feature_list.shipping_info[dataset]
    )

    if isinstance(test_inputs, torch.Tensor):
        test_np = test_inputs.detach().cpu().numpy().astype(np.float32)
    else:
        test_np = np.asarray(test_inputs, dtype=np.float32)

    N = test_np.shape[0]
    scaler = StandardScaler()
    train_X = test_np  # scaler fit not strictly needed; consistent with session
    # We use raw values for FAISS lookup (ridx0/ridx1 are categorical indices, not scaled)

    # FAISS profit lookup
    query = np.stack([
        test_np[:, ridx0],
        test_np[:, ridx1],
        actions_np.astype("float32"),
    ], axis=1).astype("float32")
    _, nn_idx = index.search(query, 1)
    nn_idx = nn_idx.flatten()
    matched = cost_data[nn_idx]
    is_exact = np.all(matched == query, axis=1)
    matched_profits = cost_y[nn_idx]
    fallback = avg_profit_np[actions_np.astype(np.int64)]
    local_profits = np.where(is_exact, matched_profits, fallback)

    profit = float(local_profits.mean())

    # Simulator on-time
    scaler2 = StandardScaler()
    scaler2.fit(test_np[:, :feat_dim])  # approximate; real train scaler not available here
    Xs = scaler2.transform(test_np[:, :feat_dim]).astype(np.float32)
    B = env.args.batch_size
    model.eval()
    time_sum, time_count = 0, 0
    with torch.no_grad():
        for s in range(0, N, B):
            e = min(s + B, N)
            state_b = torch.from_numpy(Xs[s:e]).to(env.device)
            ori_b   = torch.from_numpy(test_np[s:e]).to(env.device)
            acts_b  = torch.from_numpy(actions_np[s:e]).long().to(env.device)
            onehot  = F.one_hot(acts_b, num_classes=4).float()
            sel_emb = torch.sum(onehot.unsqueeze(2) * model.embedding.weight[:4, :], dim=1)
            pred    = model(state_b, sel_emb, ori_b[:, feat_dim + 1:])
            time_sum   += pred[-1].argmax(dim=1).sum().item()
            time_count += pred[-1].shape[0]

    on_time = time_sum / time_count if time_count > 0 else 0.0

    sorted_p = np.sort(local_profits)
    pmp = {}
    for thr in [0.1, 0.2, 0.3]:
        idx = max(0, min(len(sorted_p) - 1, int(thr * len(sorted_p))))
        pmp[thr] = float(sorted_p[idx])

    return profit, on_time, pmp, local_profits.tolist()


def main():
    args = parse_args()

    os.makedirs("run_logs", exist_ok=True)
    _ts = time.strftime("%Y%m%d_%H%M%S")
    _log_path = f"run_logs/ml_{args.dataset}_{args.baseline}_{_ts}.log"
    handlers = [logging.FileHandler(_log_path), logging.StreamHandler()]
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        handlers=handlers, force=True)

    info(f"ML baseline: {args.baseline} | dataset: {args.dataset} | frac: {args.train_frac}")

    # ── Environment / Data / Simulator ───────────────────────────────────────
    my_env = Env(args)
    my_loader = S_Loader(my_env)
    my_env.feature_classes = my_loader.feature_classes

    my_model = S_SimDec(my_env)
    if args.ckpt:
        ckpt_path = args.ckpt
        if os.path.isdir(ckpt_path):
            pth_files = sorted(glob.glob(os.path.join(ckpt_path, "*.pth")), key=os.path.getmtime)
            if not pth_files:
                raise FileNotFoundError(f"No .pth in {ckpt_path}")
            ckpt_path = pth_files[-1]
        my_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    my_model.to(my_env.device)

    # ── Precompute FAISS-optimal labels (reuse CB_Session infrastructure) ────
    # We need a dummy value_network slot; CB_Session precompute_best_actions only
    # uses self.model (simulator) and self.cost_dic / self.train_inputs.
    # We create a thin CB_Session and call precompute_best_actions directly.
    session = CB_Session(my_env, my_model, my_loader)
    info("Precomputing FAISS-optimal action labels...")
    session.precompute_best_actions()
    best_labels = session.best_action_labels.cpu().numpy()   # [N_train]

    dataset_name = args.dataset
    feat_dim = len(
        feature_list.product_info[dataset_name] + feature_list.order_info[dataset_name]
        + feature_list.customer_info[dataset_name] + feature_list.shipping_info[dataset_name]
    )

    # Train inputs (scaled) for ML models
    train_np = (my_loader.train_inputs.detach().cpu().numpy()
                if isinstance(my_loader.train_inputs, torch.Tensor)
                else np.asarray(my_loader.train_inputs, dtype=np.float32))
    test_np  = (my_loader.test_inputs.detach().cpu().numpy()
                if isinstance(my_loader.test_inputs, torch.Tensor)
                else np.asarray(my_loader.test_inputs, dtype=np.float32))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_np[:, :feat_dim]).astype(np.float32)
    X_test  = scaler.transform(test_np[:, :feat_dim]).astype(np.float32)
    y_train = best_labels  # FAISS-optimal hard labels

    # ── Train chosen baseline ─────────────────────────────────────────────────
    t0 = time.time()
    if args.baseline == "random":
        info("Baseline: RANDOM")
        np.random.seed(args.seed)
        test_actions = np.random.randint(0, 4, size=X_test.shape[0]).astype(np.float32)

    elif args.baseline == "historical":
        info("Baseline: HISTORICAL (majority class)")
        counts = np.bincount(train_np[:, feat_dim].astype(int), minlength=4)
        majority = int(np.argmax(counts))
        info(f"  Action distribution in train: {counts.tolist()} → majority = {majority}")
        test_actions = np.full(X_test.shape[0], majority, dtype=np.float32)

    elif args.baseline == "xgb":
        if not _HAS_XGB:
            info("WARNING: xgboost not installed — falling back to rf")
            args.baseline = "rf"
        else:
            info("Baseline: XGBOOST")
            clf = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=args.seed, n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            test_actions = clf.predict(X_test).astype(np.float32)
            info(f"  XGBoost train acc (vs FAISS labels): {(clf.predict(X_train) == y_train).mean():.4f}")

    if args.baseline == "rf":  # not elif — xgb may fall back here
        info("Baseline: RANDOM FOREST")
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=1,
            random_state=args.seed, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        test_actions = clf.predict(X_test).astype(np.float32)
        info(f"  RF train acc (vs FAISS labels): {(clf.predict(X_train) == y_train).mean():.4f}")

    train_time = time.time() - t0
    info(f"Training done in {train_time:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    info("Evaluating on test split...")
    profit, on_time, pmp, local_profits = _eval_actions(
        test_actions, my_loader.test_inputs, my_model, my_env,
        my_loader.cost_mrp, my_loader.avg_profit,
    )

    info(f"[TEST] profit={profit:.4f} on_time={on_time:.4f} total={profit+on_time:.4f} pmp={pmp}")
    # Emit in the exact format parsed by existing comparison scripts
    info(f"best_profit {profit:.10f}")
    info(f"best_on_time {on_time:.10f}")
    info(f"best_pmp_1 {pmp[0.1]:.10f}")
    info(f"best_pmp_2 {pmp[0.2]:.10f}")
    info(f"best_pmp_3 {pmp[0.3]:.10f}")

    # Optionally write a summary log file in the out_dir
    if args.out_dir:
        ds_lower = dataset_name.lower().replace("_ood", "ood")
        frac_str = f"{args.train_frac:.2f}"
        log_name = os.path.join(args.out_dir, f"{ds_lower}_frac{frac_str}.log")
        with open(log_name, "a") as f:
            f.write(f"baseline={args.baseline}\n")
            f.write(f"best_profit {profit:.10f}\n")
            f.write(f"best_on_time {on_time:.10f}\n")
            f.write(f"best_pmp_1 {pmp[0.1]:.10f}\n")
            f.write(f"best_pmp_2 {pmp[0.2]:.10f}\n")
            f.write(f"best_pmp_3 {pmp[0.3]:.10f}\n")
        info(f"Results appended to {log_name}")

    my_env.close()
    info("Done.")


if __name__ == "__main__":
    main()
