
# MNIST Adversarial Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success)]()

> Adversarial attack and defense implementation on MNIST digit classification using FGSM (Fast Gradient Sign Method) with comprehensive evaluation and documentation.

**Authors:** MBOHOU Fils Aboubakar Sidik, GUEMGNO Defeugaing Harold, DIALLO Abdoul Mazid, NDOM Christian Manuel
**Program:** MSc Cybersecurity and Data Science
**Course:** Application Project  
**Institution:** ESAIP, École Supérieure Angevine en Informatique et Productique
**Date:** November 2025

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Results](#results)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This project implements and evaluates:
1. **Baseline CNN classifier** for MNIST digit recognition
2. **FGSM adversarial attack** under L-infinity norm with multiple epsilon budgets
3. **FGSM adversarial training defense** to improve robustness
4. **Comprehensive evaluation** comparing baseline vs defended models

**Key Results:**
- Baseline model: 99.00% clean accuracy -> 96.95% robust accuracy (epsilon=8/255)
- Defended model: 99.13% clean accuracy -> 97.85% robust accuracy (epsilon=8/255)
- **Improvement: +0.90 percentage points** in robust accuracy with only 0.13% clean accuracy loss

---

## Project Structure
```
mnist-adversarial-project/
|
|-- mnist_adversarial_experiment.ipynb   # Main implementation notebook
|-- README.md                            # This file
|-- requirements.txt                     # Python dependencies
|
|-- models/                              # Trained model checkpoints
|   |-- baseline_model.pth               # Baseline CNN (no defense)
|   |-- defended_model.pth               # Defended CNN (FGSM training)
|
|-- results/                             # Experimental results
|   |-- baseline_results.json            # Baseline training metrics
|   |-- fgsm_attack_results.json         # Attack evaluation on baseline
|   |-- defended_training_results.json   # Defended model training metrics
|   |-- defended_fgsm_results.json       # Attack evaluation on defended
|   |-- results_tables.txt               # Summary tables
|   |
|   |-- sample_mnist_images.png          # Sample dataset images
|   |-- baseline_training_curves.png     # Baseline training progress
|   |-- adversarial_examples_fgsm.png    # Visualized adversarial examples
|   |-- robustness_curve_baseline.png    # Baseline robustness curve
|   |-- defense_comparison.png           # Defense effectiveness plots
|   |-- training_comparison.png          # Training time comparison
|   |-- final_robustness_comparison.png  # Final combined comparison
|
|-- report/                              # Documentation
    |-- experiment_report.md             # Brief report
    |-- experiment_report.txt            # Report in plain text
```

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation
```bash
# Navigate to project directory
cd mnist-adversarial-project

# Install dependencies
pip install -r requirements.txt
```

### Run the Experiment
```bash
# Open Jupyter Notebook
jupyter notebook mnist_adversarial_experiment.ipynb

# Run all cells sequentially from top to bottom
```

---

## Dependencies
```
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
matplotlib==3.7.2
jupyter==1.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Experimental Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Random Seed** | 42 |
| **Batch Size** | 128 |
| **Learning Rate** | 0.01 |
| **Momentum** | 0.9 |
| **Baseline Epochs** | 5 |
| **Adversarial Training Epochs** | 8 |
| **Training Epsilon (Defense)** | 0.0314 (8/255) |
| **Evaluation Epsilon Values** | 0.0078, 0.0157, 0.0314 (2, 4, 8 /255) |

### Hardware Requirements

- **CPU:** Sufficient (training takes ~15-20 minutes total)
- **GPU:** Optional (speeds up training but not required)
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** ~500MB (including dataset and checkpoints)

### Runtime Estimates

| Task | Time (CPU) |
|------|-----------|
| Baseline Training | ~8m 35s |
| FGSM Attack Evaluation | ~17.4s |
| Adversarial Training | ~6m 29s |
| Defense Evaluation | ~17.6s |
| **Total** | **~15m 40s** |

*Times are approximate and vary based on hardware*

---

## Results Summary

### Clean Accuracy

| Model | Accuracy |
|-------|----------|
| Baseline | 99.00% |
| Defended | 99.13% |
| **Difference** | **+0.13%** |

### Robust Accuracy (FGSM Attack)

| Epsilon Value | Baseline | Defended | Improvement |
|---------------|----------|----------|-------------|
| 0.0078 (2/255) | 98.68% | 98.92% | **+0.24%** |
| 0.0157 (4/255) | 98.35% | 98.65% | **+0.30%** |
| 0.0314 (8/255) | 96.95% | 97.85% | **+0.90%** |

### Key Findings

- Defense is effective: +0.9 percentage points robust accuracy improvement at epsilon=8/255
- Minimal clean accuracy loss: Only 0.13%
- Computational cost: 0.76x training time (acceptable for robustness gains)
- Epsilon-specific robustness: Best performance near training epsilon

---

## Reproducing Results

To exactly reproduce our results:

1. **Ensure seed is fixed** (already done in notebook: `SEED = 42`)
2. **Use exact library versions** (from `requirements.txt`)
3. **Run all cells sequentially** (don't skip cells)
4. **Don't modify hyperparameters** (unless experimenting)

**Expected outputs:**
- Baseline clean accuracy: ~98%
- Baseline robust accuracy (epsilon=8/255): ~20-30%
- Defended clean accuracy: ~96-97%
- Defended robust accuracy (epsilon=8/255): ~80-85%

---

## Visualizations

All plots are automatically saved to `results/` directory:

1. **`sample_mnist_images.png`** - Sample dataset images
2. **`baseline_training_curves.png`** - Training/test accuracy and loss curves
3. **`adversarial_examples_fgsm.png`** - Original, perturbation, and adversarial images
4. **`robustness_curve_baseline.png`** - Baseline model vulnerability curve
5. **`defense_comparison.png`** - Before/after defense comparison
6. **`training_comparison.png`** - Training time and clean accuracy comparison
7. **`final_robustness_comparison.png`** - Publication-quality combined plot (300 DPI)

---

## Model Architecture
```
MNISTClassifier(
  Input: (batch, 1, 28, 28)

  Conv2D(1 -> 32, kernel=5x5) + ReLU + MaxPool(2x2)
  Conv2D(32 -> 64, kernel=5x5) + ReLU + MaxPool(2x2)

  Flatten -> 64x7x7 = 3136 features

  FC(3136 -> 1024) + ReLU + Dropout(0.5)
  FC(1024 -> 10) - Output logits
)

Total Parameters: 3,274,634
```

---

## Attack & Defense Methods

### FGSM Attack

**Formula:** `x_adv = x + epsilon * sign(gradient_x L(f(x), y))`

- **Type:** White-box, single-step gradient attack
- **Threat model:** L-infinity norm constraint (epsilon-bounded perturbations)
- **Computational cost:** Fast (one forward + one backward pass)

### FGSM Adversarial Training

**Training procedure:**
1. For each batch: split 50% clean + 50% adversarial
2. Generate adversarial examples on-the-fly using FGSM
3. Train on combined batch
4. Repeat for 8 epochs

**Why it works:** Model learns robust features that are invariant to small perturbations in the direction of increasing loss.

---

## Files Description

### Code Files
- **`mnist_adversarial_experiment.ipynb`**: Complete implementation (all phases)

### Model Checkpoints
- **`models/baseline_model.pth`**: Trained baseline CNN (no defense)
- **`models/defended_model.pth`**: Trained defended CNN (FGSM adversarial training)

### Results Files
- **`results/baseline_results.json`**: Baseline training history and metrics
- **`results/fgsm_attack_results.json`**: Attack evaluation on baseline model
- **`results/defended_training_results.json`**: Defended model training history
- **`results/defended_fgsm_results.json`**: Attack evaluation on defended model
- **`results/results_tables.txt`**: Formatted summary tables

### Documentation
- **`report/experiment_report.md`**: Comprehensive experiment report
- **`README.md`**: This file

---

## Customization

### Change Attack Strength

Edit in the notebook:
```python
EPSILON_VALUES = [4/255, 8/255, 16/255]  # Try stronger attacks
```

### Train Longer
```python
NUM_EPOCHS_BASELINE = 10        # More training for baseline
NUM_EPOCHS_ADVERSARIAL = 15     # More training for defense
```

### Try Different Defense Epsilon
```python
EPSILON_TRAIN = 16/255  # Train against stronger perturbations
```

### Use GPU (if available)

PyTorch automatically uses GPU if available. Check with:
```python
print(torch.cuda.is_available())
```

---

## Troubleshooting

### Issue: "RuntimeError: Can't call numpy() on Tensor that requires grad"

**Solution:** Add `.detach().cpu()` before converting tensors to NumPy for plotting.

### Issue: Out of memory

**Solution:** Reduce batch size:
```python
BATCH_SIZE = 64  # Instead of 128
```

### Issue: Training is very slow

**Symptoms:** Each epoch takes >5 minutes  
**Solution:** 
- Check if another program is using CPU heavily
- Close unnecessary applications
- Consider using Google Colab with free GPU

### Issue: Results don't match exactly

**Explanation:** Minor variations (+/-1-2%) can occur due to:
- Hardware differences (CPU vs GPU)
- PyTorch version differences
- Operating system differences

As long as trends are similar, results are valid.

### Issue: UnicodeEncodeError when saving files

**Solution:** Files are saved with UTF-8 encoding. If you see encoding errors, ensure you're using Python 3.8+ and that your system supports UTF-8.

---

## Some References

1. **Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015).** Explaining and harnessing adversarial examples. *ICLR*.

2. **Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017).** Towards deep learning models resistant to adversarial attacks. *ICLR*.

3. **Tramer, F., Kurakin, A., Papernot, N., Boneh, D., & McDaniel, P. (2017).** Ensemble adversarial training: Attacks and defenses. *arXiv:1705.07204*.

4. **Carlini, N., & Wagner, D. (2017).** Towards evaluating the robustness of neural networks. *IEEE S&P*.

5. **Athalye, A., Carlini, N., & Wagner, D. (2018).** Obfuscated gradients give a false sense of security. *ICML*.

---

## Contact & Contributions

**Authors:** MBOHOU Fils Aboubakar Sidik, GUEMGNO Defeugaing Harold, DIALLO Abdoul Mazid, NDOM Christian Manuel
**Program:** MSc Cybersecurity and Data Science
**Course:** Application Project  
**Institution:** ESAIP, École Supérieure Angevine en Informatique et Productique
**Date:** November 2025

For questions or issues, please refer to the experiment report or contact the authors on LinkedIn.

---

## License

This project is created for educational purposes as part of the Application Project Course - Theme: ADVERSARIAL ML (LITE): BREAK & DEFEND A MNIST CLASSIFIER..

---

## Citation

If you use this code or methodology in your research, please cite:
```bibtex
@misc{MBOHOU2025mnist,
  author = {MBOHOU Fils Aboubakar Sidik},
  title = {MNIST Adversarial Machine Learning: FGSM Attack and Defense},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Jiren896/mnist-adversarial-ml}}
}
```

Or simply reference:

> MBOHOU. (2025). MNIST Adversarial Machine Learning: FGSM Attack and Defense. 
> GitHub repository. https://github.com/Jiren896/mnist-adversarial-ml

---

## Contributing

This is an academic project completed as part of Application Project Course. 

For questions or discussions:
- Open an issue on GitHub
- Email: fmbohou.mscia25@esaip.org

---

## Roadmap

Potential future extensions:

- [ ] Implement PGD attack (stronger multi-step attack)
- [ ] Evaluate on CIFAR-10 dataset
- [ ] Add certified defense (randomized smoothing)
- [ ] Implement adversarial example detection
- [ ] Create interactive demo with Gradio

---

## Star History

If you find this project helpful, please consider giving it a star ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=Jiren896/mnist-adversarial-ml&type=Date)](https://star-history.com/#Jiren896/mnist-adversarial-ml&Date)

---

## Acknowledgments

- MNIST dataset: Yann LeCun et al.
- PyTorch framework: Facebook AI Research
- Attack/defense methodologies: Research papers cited above

---

*Last updated: 2025-11-10*
