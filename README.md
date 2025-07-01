# Diff-AGIQA
[ICME 2025] Official code release of our paper "Text-to-Image Diffusion Models are AI-Generated Image Quality Scorers"
## Environment
```bash
pip install -r requirements.txt
```
## Testing

After preparing the code environment and downloading the weights, run the following codes to train and test model.

```bash
python evaluate_with_dataloader.py \
    --model_path <path> \
    --csv_path "./dataset/dval.csv" \
    --img_dir <AGIQA-1k_path> \
    --output_path "exp_results/full_dataset_report.csv" \
    --batch_size 64 \
    --num_workers 8
```
