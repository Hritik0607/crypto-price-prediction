import os
from typing import Literal, Optional, Tuple

import comet_ml
from datasets import Dataset, load_dataset
from loguru import logger
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# chat template we use to format the data we feed to the model
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

instruction = '\nYou are an expert crypto financial analyst with deep knowledge of market dynamics and sentiment analysis.\nAnalyze the following news story and determine its potential impact on crypto asset prices.\nFocus on both direct mentions and indirect implications for each asset.\n\nDo not output data for a given coin if the news is not relevant to it.\n\n## Example input news story\n"Goldman Sachs wants to invest in Bitcoin and Ethereum, but not in XRP"\n\n## Example output\n[\n    {"coin": "BTC", "signal": 1},\n    {"coin": "ETH", "signal": 1},\n    {"coin": "XRP", "signal": -1},\n]\n'


def load_base_llm_and_tokenizer(
    base_llm_name: str,
    max_seq_length: Optional[int] = 2048,
    dtype: Optional[str] = None,
    load_in_4bit: Optional[bool] = True,
) -> Tuple[FastLanguageModel, AutoTokenizer]:
    """
    Loads and returns the base LLM and its tokenizer

    Args:
        base_llm_name: The name of the base LLM to load
        max_seq_length: The maximum sequence length to use
        dtype: The data type to use -> None means auto-detect
        load_in_4bit: Whether to load the model in 4-bit

    Returns:
        The base LLM and its tokenizer
    """
    logger.info(f'Loading base LLM and tokenizer: {base_llm_name}')
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_llm_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    return model, tokenizer


def add_lora_adapters(
    model: FastLanguageModel,
) -> FastLanguageModel:
    """
    Adds LoRA adapters to the base model

    TODO: it would be good to expose these parameters as function arguments
    so you can tune them for your use case

    - lora_alpha: the alpha value for the LoRA adapters
    - lora_dropout: the dropout rate for the LoRA adapters
    - bias: the bias for the LoRA adapters
    - use_gradient_checkpointing: whether to use gradient checkpointing
    - random_state: the random state for the LoRA adapters
    - use_rslora: whether to use rank stabilized LoRA
    - loftq_config: the LoftQ configuration for the LoRA adapters
    """
    logger.info('Adding LoRA adapters to the base model')
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'gate_proj',
            'up_proj',
            'down_proj',
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias='none',  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing='unsloth',  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    return model


def load_and_split_dataset(
    dataset_path: str, eos_token: str
) -> Tuple[Dataset, Dataset]:
    """
    Loads and preprocesses the dataset
    """
    # load the dataset form JSONL file into a HuggingFace Dataset object
    logger.info(f'Loading and preprocessing dataset from {dataset_path}')
    dataset = load_dataset('json', data_files=dataset_path)

    # Access the default split (usually named "train")
    dataset = dataset['train']

    def format_prompts(examples):
        # chat template we use to format the data we feed to the model
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""

        instructions = examples['instruction']
        inputs = examples['input']
        outputs = examples['output']
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add eos_token, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + eos_token
            texts.append(text)
        return {
            'text': texts,
        }

    dataset = dataset.map(
        format_prompts,
        batched=True,
    )

    # split the dataset into train and test, with a fix seed to ensure reproducibility
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    return dataset['train'], dataset['test']


def fine_tune(
    model: FastLanguageModel,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    test_dataset: Dataset,
    max_seq_length: int,
):
    """
    Fine-tunes the model using supervised fine tuning.
    """
    # trainer with hyper-parameters
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field='text',
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=10,  # Set this for 1 full training run.
            max_steps=120,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim='adamw_8bit',
            weight_decay=0.01,
            lr_scheduler_type='linear',
            seed=3407,
            output_dir='outputs',
            report_to='comet_ml',  # Use this for WandB etc
            eval_strategy='epoch',
        ),
    )

    # start training
    trainer.train()


def sanity_check(
    model: FastLanguageModel,
    tokenizer: AutoTokenizer,
    input_example: str,
):
    """
    Just checking if the trained model works on a simple example
    """
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction,  # instruction
                input_example,  # input
                '',  # output - leave this blank for generation!
            )
        ],
        return_tensors='pt',
    ).to('cuda')

    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    output = tokenizer.batch_decode(outputs)
    logger.info('Inference: {}', output)


def export_model_to_ollama_format(
    model: FastLanguageModel,
    tokenizer: AutoTokenizer,
    quantization_method: Optional[Literal['q4_k_m', 'f16', 'q8_0']] = 'q8_0',
    # Optionally you can push it to HugginFace model registry
    hf_username: Optional[str] = None,
    hf_token: Optional[str] = None,
):
    """
    Saves the model and the tokenizer to disk locally and pushes them
    to the HF model registry

    Args:
        model:
        tokenizer:
        quantization_method:

        hf_username
        hf_token
    """
    # export the quantized model
    logger.info('Saving model locally to disk')
    model.save_pretrained_gguf(
        'model', tokenizer, quantization_method=quantization_method
    )
    logger.info('Model saved to disk!')

    # export the Ollama Modelfile

    breakpoint()

    # TODO: if you want to push to HF you need to get your HF username and generate
    # a token at https://huggingface.co/settings/tokens
    # model.push_to_hub_gguf(
    #     "hf/model", # Change hf to your username!
    #     tokenizer,
    #     quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
    #     token = "", # Get a token at https://huggingface.co/settings/tokens
    # )


def run(
    base_llm_name: str,
    dataset_path: str,
    max_seq_length: Optional[int] = 2048,
    max_steps: Optional[int] = 100,
    comet_ml_api_key: Optional[str] = None,
    comet_ml_project_name: Optional[str] = None,
):
    """
    Fine-tunes a base LLM using supervised fine tuning.
    The training results are logged to CometML
    The final artifact is saved as an Ollama model, so we can use it to generate signals
    locally.

    Args:
        base_llm_name: The name of the base LLM to fine-tune
        dataset_path: The path to the dataset to use for fine-tuning
        max_seq_length: The maximum sequence length to use
        max_steps: The maximum number of steps to train for
            We set it to a small number by default to debug faster.
            Once we know the fine tuning is working, we can set it to a larger number.

        comet_ml_api_key: The API key to use for CometML
        comet_ml_project_name: The name of the CometML project to use
    """
    # 0. Login to CometML so we log training run metrics that we can see on CometML dashboard
    os.environ['COMET_LOG_ASSETS'] = 'True'
    comet_ml.login(
        # api_key=comet_ml_api_key, # I previously ran $uv run comet login on the terminal to paste my API key
        project_name=comet_ml_project_name
    )
    logger.info(f'Logged in to CometML with project name: {comet_ml_project_name}')

    # 1. Load the base LLM and tokenizer
    model, tokenizer = load_base_llm_and_tokenizer(
        base_llm_name, max_seq_length=max_seq_length
    )

    # 2. Add LoRA adapters to the base model
    model = add_lora_adapters(model)

    # 3. Load the dataset with (instruction, input, output) tuples into a HuggingFace Dataset object
    # with alpaca prompt format
    train_dataset, test_dataset = load_and_split_dataset(
        dataset_path, eos_token=tokenizer.eos_token
    )

    # 4. Fine-tune the base LLM
    fine_tune(
        model,
        tokenizer,
        train_dataset,
        test_dataset,
        max_seq_length=max_seq_length,
        max_steps=max_steps,
    )

    # 5. Inference on a few examples - sanity check
    sanity_check(
        model,
        tokenizer,
        input_example='Goldman Sachs considers doubling exposure on BTC and ETH. Remains skeptical about XRP',
    )

    # 6. Save model
    export_model_to_ollama_format(model, tokenizer, quantization_method='q8_0')


if __name__ == '__main__':
    from fire import Fire

    Fire(run)
