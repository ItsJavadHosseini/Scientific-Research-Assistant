from pathlib import Path


PAPER_CATALOG = {
    # Large Language Models
    "1706.03762v7": {
        "paper_id": "attention",
        "title": "Attention Is All You Need",
        "category": "llm",
        "year": 2017,
    },
    "1810.04805v2": {
        "paper_id": "bert",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "category": "llm",
        "year": 2018,
    },
    "2005.11401v4": {
        "paper_id": "rag",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "category": "llm",
        "year": 2020,
    },
    "2106.09685v2": {
        "paper_id": "lora",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "category": "llm",
        "year": 2021,
    },
    "2201.11903v6": {
        "paper_id": "cot",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "category": "llm",
        "year": 2022,
    },
    "2307.03172v3": {
        "paper_id": "lost-middle",
        "title": "Lost in the Middle: How Language Models Use Long Contexts",
        "category": "llm",
        "year": 2023,
    },

    # Machine Learning
    "srivastava14a": {
        "paper_id": "dropout",
        "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
        "category": "machine_learning",
        "year": 2014,
    },
    "1412.6980v9": {
        "paper_id": "adam",
        "title": "Adam: A Method for Stochastic Optimization",
        "category": "machine_learning",
        "year": 2014,
    },
    "1502.03167v3": {
        "paper_id": "batchnorm",
        "title": "Batch Normalization: Accelerating Deep Network Training",
        "category": "machine_learning",
        "year": 2015,
    },
    "1603.02754v3": {
        "paper_id": "xgboost",
        "title": "XGBoost: A Scalable Tree Boosting System",
        "category": "machine_learning",
        "year": 2016,
    },
    "1705.07874v2": {
        "paper_id": "shap",
        "title": "A Unified Approach to Interpreting Model Predictions",
        "category": "machine_learning",
        "year": 2017,
    },
    "1802.03426v3": {
        "paper_id": "umap",
        "title": "UMAP: Uniform Manifold Approximation and Projection",
        "category": "machine_learning",
        "year": 2018,
    },

    # Computer Vision
    "1406.2661v1": {
        "paper_id": "gan",
        "title": "Generative Adversarial Nets",
        "category": "computer_vision",
        "year": 2014,
    },
    "1505.04597v1": {
        "paper_id": "unet",
        "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "category": "computer_vision",
        "year": 2015,
    },
    "1506.01497v3": {
        "paper_id": "faster-rcnn",
        "title": "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
        "category": "computer_vision",
        "year": 2015,
    },
    "1506.02640v5": {
        "paper_id": "yolo",
        "title": "You Only Look Once: Unified, Real-Time Object Detection",
        "category": "computer_vision",
        "year": 2016,
    },
    "1512.03385v1": {
        "paper_id": "resnet",
        "title": "Deep Residual Learning for Image Recognition",
        "category": "computer_vision",
        "year": 2015,
    },
    "2010.11929v2": {
        "paper_id": "vit",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "category": "computer_vision",
        "year": 2020,
    },
}


def get_paper_metadata(pdf_path: Path) -> dict:
    """Get paper metadata using the original PDF filename."""

    file_id = pdf_path.stem

    if file_id not in PAPER_CATALOG:
        raise ValueError(
            f"Unknown PDF file: '{pdf_path.name}'. "
            "Add its metadata to PAPER_CATALOG."
        )

    return PAPER_CATALOG[file_id].copy()