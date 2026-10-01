# Advanced NLP & Information Retrieval: Comparative Architectures and RAG Systems

## Project Overview
This repository explores advanced Natural Language Processing (NLP) and Information Retrieval (IR) paradigms[cite: 1]. It evaluates and benchmarks the performance of various neural architectures across three primary tasks: Binary Sentiment Analysis, Neural Machine Translation (NMT), and Question Answering via a Retrieval-Augmented Generation (RAG) pipeline[cite: 1]. The project provides empirical insights into how structural choices—such as embedding types, attention mechanisms, and retrieval confidence—affect model accuracy, convergence speed, and generation reliability[cite: 1].

## Tech Stack & Tools
*   **Deep Learning Frameworks:** PyTorch (torch)[cite: 1].
*   **NLP & Transformer Libraries:** transformers (Hugging Face), datasets, spaCy, nltk[cite: 1].
*   **Information Retrieval:** rank_bm25 (Okapi BM25)[cite: 1].
*   **Data Processing & Utilities:** pandas, numpy, scikit-learn, tqdm[cite: 1].
*   **Evaluation Metrics:** bert_score, rouge_score[cite: 1].
*   **Visualization:** matplotlib[cite: 1].

## Methodology

### 1. Sentiment Analysis: Recurrent Architectures & Embedding Paradigms
*   **Task:** Binary classification on the 50,000-review IMDb dataset[cite: 1].
*   **Architecture:** Compared Bidirectional LSTM (BiLSTM) and Bidirectional GRU (BiGRU) networks (hidden dim: 256, 2 layers)[cite: 1].
*   **Embeddings:** Evaluated the impact of static embeddings (frozen 300d GloVe vectors) versus contextual embeddings (dynamic 768d feature extraction via DistilBERT-base-uncased)[cite: 1].

### 2. Neural Machine Translation: Attention Mechanisms & Transformers
*   **Task:** German-to-English translation using the Multi30k dataset[cite: 1].
*   **Seq2Seq Attention:** Engineered a unified Encoder-Decoder BiGRU baseline to compare Additive (Bahdanau), Multiplicative (Luong), and Scaled Dot-Product attention mechanisms[cite: 1].
*   **Transformer Ablation:** Implemented a standard Transformer (3 layers, 8 heads) and conducted ablation studies on depth (1 layer) and multi-head attention (2 heads) to evaluate generalization on smaller datasets[cite: 1].

### 3. Retrieval-Augmented Generation (RAG) System
*   **Pipeline:** Constructed a dual-component RAG system answering queries based on a 200-document knowledge base from the SQuAD validation set[cite: 1].
*   **Retrieval & Generation:** Utilized BM25 for sparse document retrieval (top-k=3) and a pre-trained T5-Small model as the reading generator[cite: 1].
*   **Failure & Uncertainty Analysis:** Diagnosed system failure modes by measuring prediction uncertainty (Shannon Entropy) to identify cases of "Confident Hallucination" when the retriever supplied factually irrelevant but topically related context[cite: 1].
