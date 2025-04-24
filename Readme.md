# Sclera Segmentation with UI

<div align="center">
   <br><br>
   <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;">
      <img src="/assets/training_tab.png" alt="Training Interface" width="400"/>
      <img src="/assets/inference_tab.png" alt="Inference Interface" width="400"/>
      <img src="/assets/evaluation_tab.png" alt="Evaluation Interface" width="400"/>
      <img src="/assets/info_tab.png" alt="Information Interface" width="400"/>
      <img src="/assets/results.png" alt="Results" width="400"/>
      <img src="/assets/about.png" alt="About" width="400"/>
   </div>
   <p><em>Explore the key interfaces of our Sclera Segmentation tool</em></p>
</div>

An interactive application for training, evaluating, and running inference on sclera segmentation models using PyTorch and Gradio.

## Overview

This project provides a complete end-to-end solution for sclera segmentation in eye images, with a user-friendly graphical interface built using Gradio. The application allows researchers and practitioners to train custom models on their own datasets, evaluate model performance, and run inference on new images without requiring programming knowledge.

## Features

- **Interactive web-based UI** with Gradio for easy use without coding
- **End-to-end workflow** from training to inference
- **Real-time training visualization** with progress tracking
- **Model evaluation** with comprehensive metrics
- **One-click inference** on new images
- **GPU acceleration** support for faster processing
- **Support for custom datasets** with flexible directory structures

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tayden1990/Sclera-Segmentation-with-UI.git
   cd Sclera-Segmentation-with-UI
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

```bash
python app.py
```

This will start the Gradio interface, which you can access through your web browser (typically at http://127.0.0.1:7860).

### Application Tabs

The application consists of four main tabs:

1. **Train Model** - Train new sclera segmentation models
2. **Inference** - Run inference on new images using trained models
3. **Evaluation** - Evaluate model performance on test datasets
4. **Info** - View system information and dataset statistics

### Screenshots

#### Training Tab
![Training Tab](/assets/training_tab.png)
*The training interface allows you to configure model parameters and monitor training progress*

#### Inference Tab
![Inference Tab](/assets/inference_tab.png)
*Upload images and run inference with trained models*

#### Evaluation Tab
![Evaluation Tab](/assets/evaluation_tab.png)
*Comprehensive model evaluation with detailed metrics*

#### Info Tab
![Info Tab](/assets/info_tab.png)
*System information and dataset statistics*

### Dataset Structure

The application expects your data to be organized as follows:

```
sclera_dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

- Images should be in JPG or PNG format
- Masks should be binary images (white for sclera, black for background)

## Training

The training tab allows you to:

1. Set the number of epochs
2. Adjust batch size
3. Configure image size for training
4. Use a subset of the training data (useful for quick experiments)
5. Name your model
6. Start training with a single click
7. Monitor training progress in real-time

Training is performed using PyTorch's implementation of Mask R-CNN with a ResNet-50 backbone. The model is fine-tuned for sclera segmentation specifically.

## Inference

The inference tab allows you to:

1. Upload a new image
2. Select a trained model
3. Adjust the detection threshold
4. Run inference with a single click
5. View and download the segmentation results

## Evaluation

The evaluation tab allows you to:

1. Select a model to evaluate
2. Choose a test dataset
3. Set evaluation parameters
4. Run comprehensive evaluation
5. View detailed performance metrics including IoU, precision, recall, and F1 score

## Command Line Usage

For advanced users, you can also use the command-line scripts directly:

```bash
# Training a new model
python train.py --epochs 20 --batch-size 2 --image-size 256

# Evaluating models
python evaluate_torch_models.py --checkpoint-dir logs/your_model_dir --test-dir sclera_dataset/test

# Running inference
python inference_torch.py --weights logs/your_model_dir/checkpoint_019.pth --image your_image.jpg
```

## Project Structure

```
Sclera-Segmentation-with-UI/
├── app.py                   # Main application with Gradio UI
├── train.py                 # Training script
├── inference_torch.py       # Inference script
├── evaluate_torch_models.py # Evaluation script
├── sclera_dataset_torch.py  # Dataset handling
├── utils.py                 # Utility functions
├── requirements.txt         # Dependencies
├── logs/                    # Directory for trained models
├── sclera_dataset/          # Dataset directory
├── temp/                    # Temporary files
└── results/                 # Evaluation results
```

## Technical Details

### Model Architecture

The segmentation model is based on Mask R-CNN with a ResNet-50 backbone. The model is trained to segment the sclera (white part of the eye) in images.

### Performance Considerations

- Training on a GPU is highly recommended
- For larger datasets, increase the batch size as your GPU memory allows
- Image size significantly affects both training time and segmentation quality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author & Contact

**Taher Akbari Saeed**  
Postgraduate Student in Hematology and Blood Transfusion  
Department of Oncology, Hematology, and Radiotherapy  
Institute of Postgraduate Education,  
Pirogov Russian National Research Medical University (RNRMU), Russia

- Email: taherakbarisaeed@gmail.com
- GitHub: [tayden1990](https://github.com/tayden1990)
- Telegram: [tayden2023](https://t.me/tayden2023)
- ORCID: 0000-0002-9517-9773

## Acknowledgments

- PyTorch and TorchVision teams for providing the core deep learning frameworks
- Gradio team for simplifying the creation of web interfaces for ML models