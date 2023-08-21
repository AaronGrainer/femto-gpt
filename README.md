# FemtoGPT

A lightweight and minimalistic implementation of the GPT (Generative Pre-trained Transformer) model, a simple implementation maintaining the core functionality of training and generating text. It's built using PyTorch Lightning, a powerful library that simplifies the training process for PyTorch models.

## Features

- **Compact Implementation**: femtoGPT offers a minimalistic yet functional GPT implementation.
- **PyTorch Lightning**: Utilize the simplicity and flexibility of PyTorch Lightning for training.
- **Text Generation**: Generate creative and coherent text using the trained femtoGPT model.
- **Easy to Use**: Get started quickly with straightforward setup and usage.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/AaronGrainer/femto-gpt.git
cd femto-gpt
```
   
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Dataset - Download and prepare training data:

```bash
make download-dataset
```

2. Training - Run the training script:

```bash
python -m main train
```

3. Text Generation - After training, use the trained model to generate text:

```bash
python -m main generate
```
