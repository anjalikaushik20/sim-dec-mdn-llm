import os
import re
import glob
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt

def load_logpi_files(base_dir: str):
    # Supports both exp_report/<dataset>/ and exp_report/<dataset>/logpi_results/
    search_dirs = [base_dir, os.path.join(base_dir, "logpi_results")]
    pattern = re.compile(r"logpi_epoch_(\d+)\.pt")

    files = []
    for d in search_dirs:
        if os.path.isdir(d):
            files.extend(glob.glob(os.path.join(d, "logpi_epoch_*.pt")))
    files = [f for f in files if pattern.search(os.path.basename(f))]
    if not files:
        raise FileNotFoundError(f"No logpi_epoch_*.pt files found under: {search_dirs}")

    def epoch_num(path):
        m = pattern.search(os.path.basename(path))
        return int(m.group(1)) if m else -1

    files.sort(key=epoch_num)
    return files, pattern

def summarize(files, pattern):
    records_logpi = []
    records_weights = []

    for fp in files:
        epoch = int(pattern.search(os.path.basename(fp)).group(1))
        obj = torch.load(fp, map_location="cpu")

        # Normalize to 2D tensor [N, K] where N aggregates all captured samples
        if isinstance(obj, list):
            if not obj:
                continue
            t = torch.cat([x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in obj], dim=0)  # [sum_B, K]
        elif isinstance(obj, torch.Tensor):
            if obj.dim() == 3:
                t = obj.reshape(-1, obj.size(-1))  # [sum_B, K]
            elif obj.dim() == 2:
                t = obj  # [sum_B, K]
            else:
                raise ValueError(f"Unexpected tensor shape {obj.shape} in {fp}")
        else:
            raise TypeError(f"Unsupported saved type {type(obj)} in {fp}")

        mean_logpi = t.mean(dim=0)                   # [K]
        weights = t.softmax(dim=-1).mean(dim=0)      # [K]

        for k, v in enumerate(mean_logpi.tolist()):
            records_logpi.append({"epoch": epoch, "gaussian": k, "mean_logpi": v})
        for k, v in enumerate(weights.tolist()):
            records_weights.append({"epoch": epoch, "gaussian": k, "mean_weight": v})

    df_logpi = pd.DataFrame(records_logpi)
    df_weights = pd.DataFrame(records_weights)
    return df_logpi, df_weights

def save_csvs_and_plots(df_logpi, df_weights, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Save CSVs
    df_logpi.to_csv(os.path.join(out_dir, "mean_logpi.csv"), index=False)
    df_weights.to_csv(os.path.join(out_dir, "mean_weights.csv"), index=False)
    summary = df_logpi.merge(df_weights, on=["epoch", "gaussian"])
    summary_path = os.path.join(out_dir, "logpi_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Pivot for plotting
    pivot_logpi = df_logpi.pivot(index="epoch", columns="gaussian", values="mean_logpi").sort_index()
    pivot_weights = df_weights.pivot(index="epoch", columns="gaussian", values="mean_weight").sort_index()

    # Plot mean logpi
    plt.figure(figsize=(8, 5))
    for k in pivot_logpi.columns:
        plt.plot(pivot_logpi.index, pivot_logpi[k], marker="o", label=f"g{k}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean logpi")
    plt.title("Mean logpi per Gaussian over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_logpi.png"), dpi=150)
    plt.close()

    # Plot mean mixture weights
    plt.figure(figsize=(8, 5))
    for k in pivot_weights.columns:
        plt.plot(pivot_weights.index, pivot_weights[k], marker="o", label=f"g{k}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean mixture weight")
    plt.title("Mean mixture weights over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_weights.png"), dpi=150)
    plt.close()

    print(f"Saved:\n  {summary_path}\n  plots in {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Summarize and plot MDN logpi over epochs.")
    parser.add_argument("--dataset", type=str, default="OAS", help="Dataset name (folder under exp_report).")
    parser.add_argument("--exp_root", type=str, default="exp_report", help="Experiment root directory.")
    args = parser.parse_args()

    base_dir = os.path.join(args.exp_root, args.dataset)
    files, pattern = load_logpi_files(base_dir)
    print(f"Found {len(files)} files under {base_dir} (and logpi_results).")

    df_logpi, df_weights = summarize(files, pattern)
    out_dir = os.path.join(base_dir, "logpi_results")
    save_csvs_and_plots(df_logpi, df_weights, out_dir)

if __name__ == "__main__":
    main()