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
#### 🎨 1. Prompt-Augmented Input Generation

> We first employ **lightweight visual prompters** to generate specialized 'perception' and 'alignment' prompts. These are then **superimposed onto the original image**, creating a composite input designed to effectively leverage the prior knowledge within the pre-trained Diffusion model.

#### ⚙️ 2. Feature Integration for Quality Prediction

> Once the model processes this augmented input, a **carefully designed feature selection process** systematically integrates the most **discriminative features**. This final, integrated feature set is then used to predict the AGI quality score.

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
