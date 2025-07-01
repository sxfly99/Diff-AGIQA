# Diff-AGIQA
[ICME 2025] Official code release of our paper "Text-to-Image Diffusion Models are AI-Generated Image Quality Scorers"
## Motivation
![image](https://github.com/sxfly99/Diff-AGIQA/blob/main/Figs/Figure2_01.png)
#### 🧠 1. Rich Prior Knowledge & High-Quality Benchmark

> Pre-trained diffusion models encapsulate rich prior knowledge about both **high-quality images** and **text-image alignment**. This knowledge is derived from carefully curated, large-scale training data and is implicitly encoded through the text-to-image generation pre-training objective.

#### 🎯 2. Natural Alignment with the AGI Domain

> As generative models, they are inherently aligned with the AGI (Artificial General Intelligence) domain, which helps **mitigate the domain gap issue** prevalent in previous methods.

#### 🔗 3. Fine-Grained Multimodal Interaction via U-Net

> The **U-Net architecture**, central to diffusion models, facilitates **multi-scale interaction** between visual and textual features. This design naturally produces multi-modal features with fine-grained interactions during the feature extraction process.
## Method
![image](https://github.com/sxfly99/Diff-AGIQA/blob/main/Figs/Figure3_01.png)
💡 1. Generation of Specialized Visual Prompts
To better leverage the model's pre-trained knowledge, lightweight visual prompters are introduced. Their specific function is to generate two types of crucial guidance cues: 'perception' and 'alignment' prompts.

🖼️ 2. Input Formulation for the Diffusion Model
These newly generated prompts are then superimposed directly onto the original image. This composite image—combining the source visual with the new prompts—serves as the complete input for the pre-trained Diffusion model.

⚙️ 3. Systematic Feature Integration for Prediction
Following the model's processing, a carefully designed feature selection process is employed. This process systematically identifies and integrates the most discriminative features, which are then used to predict the final AGI quality score.

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
