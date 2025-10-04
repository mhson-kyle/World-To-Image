# World2Image

## Component Details

1. **Prompt Optimizer**: LLM-based prompt optimization
2. **Image Retriever**: LLM-based image retrieval
3. **Scorer**: Scoring of the generated image
4. **Orchestrator**: Orchestrates the entire optimization workflow
5. **Pipeline**: Orchestrates the entire optimization workflow

## Installation

### Prerequisites
- Python 3.10

### Install with uv (Recommended)
```bash
# Clone the repository
git clone https://github.com/anonym-code996/world2image.git
cd world2image

# Install with uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Configuration
### OmniGen2
To use OmniGen2,
please follow the instructions in https://github.com/VectorSpaceLab/OmniGen2/ to install OmniGen2.

### Environment Variables
```bash
# Azure OpenAI
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-endpoint.openai.azure.com/"
export AZURE_API_VERSION="2024-12-01-preview"
export RAPIDAPI_KEY="your-rapidapi-key"
```

## Quick Start

### Basic Optimization
```bash
# Single prompt optimization
python run_single.py 'dr strange' --iterations 3

# Multiple prompts optimization
python run.py \
  --config configs/config_base.yaml \
```