# GenAi
A comprehensive compilation of my Generative AI experiments, projects, and learning resources. This repository documents my progress through various GenAI concepts.

## 🌟 Core Focus Areas

### 1. Large Language Models (LLMs)
- **Foundation Models**: GPT-3/4, Llama 2, Claude
- **Specialized Implementations**:
  - Code generation models

### 2. Diffusion Models
- **Image Generation**:
  - Stable Diffusion (SDXL, 1.5, 2.1)
  - DALL-E implementations
  - Imagen concepts
- **Video Generation**:
  - AnimateDiff
  - Sora architecture analysis
- **Optimization Techniques**:
  - LCM-LoRA
  - ControlNet adaptations

### 3. GANs & VAEs
- **Generative Adversarial Networks**:
  - DCGAN, StyleGAN variants
- **Variational Autoencoders**:
  - Basic implementations

### 4. Prompt Engineering
- **Techniques**:
  - Chain-of-Thought
  - Tree-of-Thought
  - Self-consistency methods

### 5. Fine-tuning Techniques
- **Parameter-Efficient Methods**:
  - LoRA (Low-Rank Adaptation)
  - QLoRA (Quantized LoRA)
  - Adapter modules
- **Full Fine-tuning**:
  - Domain adaptation
  - Instruction tuning
- **RLHF Pipeline**:
  - Reward modeling

### 6. RAG Implementations
- **Retrieval Systems**:
  - Dense retrieval (ANCE, DPR)
  - Sparse retrieval (BM25)
- **Augmentation Methods**:
  - Basic RAG pipelines
  - Advanced re-ranking
  - Hypothetical Document Embeddings (HyDE)

### 7. AI Agent Frameworks
- **Single Agent Systems**:
  - ReAct implementations
  - Plan-and-execute patterns
- **Multi-Agent Systems**:
  - AutoGen configurations
  - ChatDev simulations
- **Tool Usage**:
  - Function calling
  - Code interpreter integration
  - 
- **Novel Approaches**:
  - Diffusion Transformers
  - Neural Symbolic systems
  - Energy-based models

## � Projects Showcase

| Project | Description | Technologies | Notebook |
|---------|-------------|--------------|----------|
| Medical RAG Pipeline | Healthcare Q&A system | Llama2, LangChain, FAISS | [Open](projects/medical_rag.ipynb) |
| LoRA Fine-tuning | Customizing Mistral | PyTorch, HuggingFace | [Open](finetuning/mistral-lora.ipynb) |
| SDXL ControlNet | Architecture sketches | Stable Diffusion, ControlNet | [Open](diffusion/sdxl-controlnet.ipynb) |

## 🛠️ Technical Stack

```mermaid
graph TD
    A[Core Frameworks] --> B[HuggingFace]
    A --> C[PyTorch]
    A --> D[TensorFlow]
    E[Specialized Tools] --> F[LangChain]
    E --> G[LlamaIndex]
    E --> H[AutoGen]
