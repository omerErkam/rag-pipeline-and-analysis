# Retrieval-Augmented Generation (RAG) & NLP Architecture Benchmarks

## Project Overview
This repository explores advanced Natural Language Processing (NLP) and Information Retrieval (IR) paradigms[cite: 4]. It evaluates and benchmarks the performance of various neural architectures across Sentiment Analysis and Neural Machine Translation, culminating in the construction and diagnostic evaluation of a Retrieval-Augmented Generation (RAG) pipeline[cite: 4]. The project provides empirical insights into how structural choices—such as embedding types, attention mechanisms, and retrieval confidence—affect model accuracy, convergence speed, and generation reliability[cite: 4].

## Tech Stack & Tools
*   **Deep Learning Frameworks:** PyTorch (torch)[cite: 4].
*   **NLP & Transformer Libraries:** transformers (Hugging Face), datasets, spaCy, nltk[cite: 4].
*   **Information Retrieval:** rank_bm25 (Okapi BM25)[cite: 4].
*   **Data Processing & Utilities:** pandas, numpy, scikit-learn, tqdm[cite: 4].
*   **Evaluation Metrics:** bert_score, rouge_score[cite: 4].
*   **Visualization:** matplotlib[cite: 4].

## Methodology

### 1. Sentiment Analysis: Recurrent Architectures & Embeddings
*   **Task:** Binary classification on the 50,000-review IMDb dataset[cite: 4].
*   **Architecture:** Compared Bidirectional LSTM (BiLSTM) and Bidirectional GRU (BiGRU) networks (hidden dim: 256, 2 layers)[cite: 4].
*   **Embeddings:** Evaluated the impact of static embeddings (frozen 300d GloVe vectors) versus contextual embeddings (dynamic 768d feature extraction via DistilBERT-base-uncased)[cite: 4].

### 2. Neural Machine Translation: Attention Mechanisms
*   **Task:** German-to-English translation using the Multi30k dataset[cite: 4].
*   **Seq2Seq Attention:** Engineered a unified Encoder-Decoder BiGRU baseline to isolate and compare Additive (Bahdanau), Multiplicative (Luong), and Scaled Dot-Product attention mechanisms[cite: 4].

### 3. Transformer Architectures & Ablation Studies
*   **Implementation:** Engineered a standard Transformer (3 layers, 8 heads) from scratch for the NMT task[cite: 4].
*   **Ablation:** Conducted studies on depth (reducing to 1 layer) and multi-head attention (reducing to 2 heads) to evaluate architectural generalization and overfitting on smaller datasets[cite: 4].

### 4. Retrieval-Augmented Generation (RAG) Pipeline
*   **System Architecture:** Constructed a dual-component RAG system answering queries based on a 200-document knowledge base curated from the SQuAD validation set[cite: 4].
*   **Retrieval & Generation:** Utilized BM25 for sparse document retrieval (top-k=3) and a pre-trained T5-Small model as the generative reader[cite: 4].

### 5. Failure Analysis & Hallucination Diagnostics
*   **Uncertainty Quantification:** Diagnosed system failure modes by measuring the generative model's prediction uncertainty using Shannon Entropy[cite: 4]. 
*   **Evaluation:** Correlated retrieval accuracy with generation confidence to identify cases of "Confident Hallucination," specifically analyzing model behavior when conditioned on factually irrelevant but topically related context[cite: 4].

## Results & Evaluation
*   **Recurrent Benchmarks:** The BiGRU paired with DistilBERT contextual embeddings achieved the highest overall performance with a **92.007% Accuracy** and rapid convergence[cite: 4].
*   **Attention Alignment:** Multiplicative attention produced the highest BLEU score (**6.97**) and generated the sharpest word-to-word mapping[cite: 4].
*   **Transformer Performance:** The standard Transformer architecture massively reduced perplexity compared to the best RNN baseline (from 33.07 to **2.42**)[cite: 4]. The shallow 1-layer Transformer achieved the best BLEU score (**7.46**) due to better generalization on the limited dataset[cite: 4].
*   **RAG System Calibration:** Generated answers maintained high semantic validity (**92.06 BERTScore**) despite a 34.00% top-3 retrieval bottleneck[cite: 4]. Entropy analysis revealed the critical finding that the generator often outputs incorrect answers with *higher* confidence (lower entropy) when conditioned on misaligned retrieval contexts[cite: 4].

## Repository Structure
```text
rag-pipeline-and-analysis/
│
├── notebooks/
│   └── nlp_architectures_and_rag_pipeline.ipynb # Executable notebook containing the full pipeline
│
├── docs/                               # Documentation and reports
│   ├── project_report.pdf
│   └── images/                         # Folder for README assets (attention heatmaps, convergence plots)
│
├── requirements.txt                    # Project dependencies
└── README.md                           # This file
