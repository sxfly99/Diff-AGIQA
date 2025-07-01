# Diff-AGIQA
[ICME 2025] Official code release of our paper "Text-to-Image Diffusion Models are AI-Generated Image Quality Scorers"
## Motivation
<iframe src="https://docs.google.com/viewer?url=https://github.com/your_username/your_repository/raw/branch/path/to/your_pdf.pdf&embedded=true" style="width:100%; height:600px;" frameborder="0"></iframe>
## Method
<iframe src="https://docs.google.com/viewer?url=https://github.com/your_username/your_repository/raw/branch/path/to/your_pdf.pdf&embedded=true" style="width:100%; height:600px;" frameborder="0"></iframe>

## News
📌 **TO DO**
- ✅ Inference code release
- [ ] training code release

## Environment
```bash
pip install -r requirements.txt
```
Downloading the [weights](https://pan.baidu.com/s/11nYAQO_bouD22rjCpKT32A?pwd=ncju)
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
