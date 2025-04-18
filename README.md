# LoRA: Low-Rank Adaptation for Neural Networks 🧠

LoRA (Low-Rank Adaptation) is an efficient fine-tuning technique for neural networks that dramatically reduces the number of trainable parameters by using low-rank decomposition methods.

![Image](https://github.com/user-attachments/assets/a9e9ff8d-1763-4615-9a90-f8bbdb55694e)

## 📊 Overview

Instead of updating all parameters during fine-tuning, LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the neural network. This approach:

- Significantly reduces memory requirements
- Speeds up training time
- Maintains model performance comparable to full fine-tuning

## ⚙️ How LoRA Works

### The Core Idea

Traditionally, when fine-tuning a model, we update the original weights W directly:

```
W_updated = W + ΔW
```

LoRA approximates the weight updates (ΔW) using the product of two low-rank matrices A and B:

```
W_updated = W + A·B
```

Where:
- W is the frozen pre-trained weight matrix
- A and B are smaller matrices with a bottleneck dimension r
- r is a hyperparameter controlling the rank of the decomposition

### Mathematical Example

Let's illustrate with concrete numbers:

1. Suppose a layer has a weight matrix W of size 5,000×10,000 (50M parameters)
2. With LoRA using rank r=8:
   - Matrix A has dimensions 5,000×8 (40,000 parameters)
   - Matrix B has dimensions 8×10,000 (80,000 parameters)
   
Total trainable parameters: 40,000 + 80,000 = 120,000  
**That's 400× fewer parameters than full fine-tuning!**

### Forward Pass

During the forward pass of input x through a LoRA-adapted layer:

```
output = x·W + α·(x·A·B)
```

Where α is a scaling factor to control the magnitude of the LoRA update.

## 🚀 LoRA Benefits and Applications

- **Efficiency**: Fine-tune large models on consumer hardware
- **Parameter Sharing**: Multiple LoRA adapters can be trained for different tasks while sharing the base model
- **Quick Adaptation**: Train specialized versions of a model for different domains or tasks
- **Reduced Storage**: Store only the small adapter weights (A and B matrices) instead of full model copies

## 🔧 Implementation Details

### LoRA Layer Structure

A basic LoRA layer adds low-rank updates to the original layer output:

![LoRA Layer Diagram](https://github.com/user-attachments/assets/f52ecf42-5044-4717-828f-cc28c54d2fe0)

### Scaling Factor (α)

The α hyperparameter controls the magnitude of the LoRA adaptation:
- Higher α = larger modifications to model behavior
- Lower α = more subtle changes

### Training Process

1. Freeze weights of the pre-trained model
2. Initialize LoRA matrices (A with random small weights, B with zeros)
3. Train only the LoRA matrices A and B
4. At inference time, you can either:
   - Continue computing W + A·B separately
   - Merge the weights: W_merged = W + α·(A·B)

## 💡 Practical Considerations

- **Rank Selection**: Lower ranks save memory but provide less modeling capacity
- **Which Layers to Adapt**: Commonly applied to attention layers in transformers, but can be used on any linear layers
- **Initialization Strategy**: Proper initialization of A and B matrices is important for stable training

## 🔄 Applications Beyond LLMs

While LoRA was initially developed for large language models, the technique works for any neural network with linear layers, including:

- Computer vision models
- Audio processing networks
- Multimodal architectures
- Reinforcement learning policies

## 🏁 Getting Started

The simplest way to use LoRA is to replace standard linear layers with LoRA-augmented versions:

```python
# Instead of:
layer = nn.Linear(input_dim, output_dim)

# Use:
layer = LinearWithLoRA(nn.Linear(input_dim, output_dim), rank=4, alpha=8)
```

Then freeze the original weights and train only the LoRA parameters.


## 📚 Learn More

For a deeper dive into LoRA, check the original paper:  
["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) by Hu et al.



## 👤 Author

For any questions or issues, please open an issue on GitHub: [@Siddharth Mishra](https://github.com/Sid3503)

---

<p align="center">
  Made with ❤️ and lots of ☕
</p>
