# Gigabytes to Guidance: Efficiently Adapting a Compact LLM for STEM Education

## Overview
Co-developed a STEM AI tutor by fine-tuning the compact Qwen3-0.6B model. Engineered a multi-stage pipeline involving general Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and task-specific MCQA instruction tuning via LoRA. Implemented 4-bit NF4 Post-Training Quantization (PTQ) using bitsandbytes, successfully reducing VRAM usage by 62% and disk size by 77% without accuracy degradation. Integrated a dense passage Retrieval-Augmented Generation (RAG) framework, boosting final accuracy on the ARC-Easy benchmark from 27% to 72%.

## Project Architecture & Methodology
This repository contains the training scripts, data, and configurations used to adapt `Qwen/Qwen3-0.6B-Base` into a highly efficient, specialized STEM tutor. The training pipeline was divided into distinct, modular phases:

### 1. General Supervised Fine-Tuning (SFT)
To establish a foundation in scientific language and reasoning, the base model was initially fine-tuned on a diverse mixture of 12,000 open-domain STEM samples (from FLAN-v2, OpenMathInstruct, and NuminaMath-1.5). This provided the model with a broad capability to handle complex conceptual explanations and problem-solving processes.

### 2. Task-Specific Alignment
From the common SFT base, the model was specialized via two parallel tracks using Low-Rank Adaptation (LoRA):
* **Direct Preference Optimization (DPO):** To create a preference-aligned conversational agent, the model was optimized using DPO on curated STEM datasets. This reinforced adherence to formal problem constraints (like identifying subtle code bugs) and improved the model's preference accuracy.
* **MCQA Instruction Tuning:** The model was fine-tuned strictly on multiple-choice STEM questions. This aggressive optimization strategy sharpened the model's classification behavior, teaching it to output confident and consistent answer keys. 

### 3. Post-Training Quantization (PTQ)
To ensure the model could be deployed efficiently on consumer hardware, we applied weight-only quantization using the `bitsandbytes` library. After evaluating multiple configurations, we implemented **4-bit NormalFloat4 (NF4) with Double Quantization**. This dramatically lowered the computational barriers without compromising the reasoning accuracy learned during the SFT and MCQA stages.

### 4. Retrieval-Augmented Generation (RAG)
To address factual knowledge gaps and reduce hallucinations, the final generator model was augmented with a retrieval-then-generate architecture. Using a frozen `sentence-transformers/all-MiniLM-L6-v2` encoder, the pipeline performs dense passage retrieval (via cosine similarity) across a 100k-chunk external knowledge base composed of STEM Wikipedia articles and Camel-AI Physics data. 

## Key Performance Metrics

**1. Reasoning & Task Accuracy**
The multi-phase curriculum training resulted in massive gains in the model's ability to answer structured STEM questions.
* **Base Qwen Model:** 27% (ARC-Easy)
* **Post-General SFT:** 59% (ARC-Easy)
* **Final MCQA Model:** 72% (ARC-Easy)
* *Note: The addition of the RAG pipeline provided a further consistent accuracy boost of 2-5% across multiple evaluations by supplying directly relevant external context.*

**2. Quantization Efficiency**
The 4-bit NF4 configuration provided an exceptional trade-off between resource cost and model performance:
* **Original VRAM:** 8.24 GB ➔ **Quantized VRAM:** 3.10 GB *(62% Reduction)*
* **Original Disk Size:** 2.22 GB ➔ **Quantized Disk:** 0.50 GB *(77% Reduction)*
* **Accuracy Retention:** Maintained ~66% accuracy under the Lighteval framework, proving high resilience to aggressive compression.

## Repository Structure
* `/code`: Contains the executable bash scripts (`train_dpo.sh`, `train_mcqa.sh`, `train_quantized.sh`, `train_rag.sh`) and the specific training logic for each model variation.
* `/data`: Points to the datasets used for the fine-tuning pipeline. 
* `/model_configs`: Contains the YAML configuration files for the HuggingFace Hub deployments.
* `/pdf`: Contains the comprehensive research report detailing the ablation studies, mathematical formulations, and qualitative analyses of the model outputs.

## Contributors
* Alix Papadatos
* Inés Altemir Marinas
* Nizar Ben Mohamed
* Florian Tanguy
