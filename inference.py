# evaluate_with_dataloader.py
import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

# --- 导入您自定义的模块 ---
from AGIQA_1k import AGIQA1kDataset, custom_collate
from net.DiffAGIQA1k import DiffAGIQA

def compute_metrics(pred_scores, true_scores):
    """计算 LCC(PLCC) 和 SRCC，与训练脚本保持一致。"""
    pred_scores = np.array(pred_scores).flatten()
    true_scores = np.array(true_scores).flatten()
    
    valid_indices = ~np.isnan(pred_scores) & ~np.isnan(true_scores)
    if np.sum(valid_indices) < 2:
        return 0.0, 0.0

    plcc = pearsonr(pred_scores[valid_indices], true_scores[valid_indices])[0]
    srcc = spearmanr(pred_scores[valid_indices], true_scores[valid_indices])[0]
    return plcc, srcc

def main(args):
    """使用AGIQA1kDataset和Dataloader高效地进行模型评估。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DiffAGIQA()
    print(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    model.to(device)

    print("Initializing dataset and dataloader...")
    dataset = AGIQA1kDataset(
        csv_file=args.csv_path,
        img_dir=args.img_dir
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate
    )

    all_predicted_scores = []
    all_true_scores = []

    print("Starting evaluation using DataLoader...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            prompts = batch['prompt']
            true_scores = batch['score']
            
            # --- 这里是关键改动 ---
            # 不再使用 .squeeze(1)。我们直接将 Dataloader 产生的
            # (B, 1, 3, 512, 512) 的 5D 张量传递给模型。
            # 这样模型内部循环取出的 i 将是 (1, 3, 512, 512) 的 4D 张量，问题解决。
            
            quality_norm = model(images, prompts)
            
            predicted_scores = quality_norm.cpu() * 5.0
            
            all_predicted_scores.extend(predicted_scores.flatten().tolist())
            all_true_scores.extend(true_scores.flatten().tolist())

    original_df = pd.read_csv(args.csv_path)
    if len(original_df) == len(all_predicted_scores):
        original_df['predicted_score'] = all_predicted_scores
        final_df = original_df
    else:
        print("Warning: Length mismatch. Saving scores in a new file.")
        final_df = pd.DataFrame({
            'true_score': all_true_scores,
            'predicted_score': all_predicted_scores
        })

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(args.output_path, index=False, float_format='%.4f')
    print(f"\nDetailed evaluation results saved to: {args.output_path}")

    plcc, srcc = compute_metrics(all_predicted_scores, all_true_scores)
    
    print("\n" + "="*40)
    print("--- Model Performance Validation ---")
    print(f"Dataset evaluated: {os.path.basename(args.csv_path)}")
    print(f"Samples evaluated: {len(all_true_scores)}")
    print("-" * 20)
    print(f"SRCC (Spearman): {srcc:.4f}")
    print(f"PLCC (Pearson):  {plcc:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficiently evaluate a trained model using AGIQA1kDataset and DataLoader.")
    
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model .pth file.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the CSV file for evaluation.")
    parser.add_argument('--img_dir', type=str, required=True, help="The base directory where images are stored.")
    parser.add_argument('--output_path', type=str, default='exp_results/dataloader_validation_report.csv', help="Path to save the detailed CSV report.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker processes for data loading.")

    args = parser.parse_args()
    main(args)