EVALUATION_DATASET = [

    {
        "question": "What is LoRA?",
        "expected_source": "lora.pdf",
        "ground_truth": (
            "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning "
            "method that freezes the pretrained model weights and trains only "
            "small low-rank adaptation matrices, significantly reducing the "
            "number of trainable parameters."
        )
    },

    {
        "question": "Why is LoRA memory efficient?",
        "expected_source": "lora.pdf",
        "ground_truth": (
            "LoRA is memory efficient because it keeps the original model "
            "weights frozen and trains only a small set of low-rank matrices, "
            "reducing GPU memory usage and storage requirements."
        )
    },

    {
        "question": "Compare LoRA with full fine-tuning.",
        "expected_source": "lora.pdf",
        "ground_truth": (
            "LoRA fine-tunes only low-rank adapter matrices while keeping the "
            "original model weights frozen, making it more memory and "
            "compute efficient. Full fine-tuning updates all model parameters, "
            "requiring significantly more memory, computation, and storage."
        )
    },

    {
        "question": "What is GPT-3?",
        "expected_source": "gpt3.pdf",
        "ground_truth": (
            "GPT-3 is a large autoregressive language model developed by "
            "OpenAI. It uses the Transformer architecture and contains "
            "175 billion parameters, enabling it to perform a wide range of "
            "natural language processing tasks with few-shot or zero-shot learning."
        )
    },

    {
        "question": "What is LLaMA?",
        "expected_source": "llama.pdf",
        "ground_truth": (
            "LLaMA (Large Language Model Meta AI) is a family of open-weight "
            "large language models developed by Meta, designed to achieve "
            "strong performance while using fewer parameters than many "
            "previous large language models."
        )
    },

    {
        "question": "Compare GPT-3 and LLaMA.",
        "expected_source": ["gpt3.pdf", "llama.pdf"],
        "ground_truth": (
            "GPT-3 is a proprietary large language model developed by OpenAI "
            "with 175 billion parameters, whereas LLaMA is an open-weight "
            "family of language models developed by Meta. GPT-3 emphasizes "
            "few-shot learning, while LLaMA focuses on achieving strong "
            "performance with relatively smaller and more efficient models."
        )
    }

]