from django.shortcuts import render
from django.http import JsonResponse
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from datasets import load_dataset, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
import os
import sys
import wandb
import gc
import torch
from decouple import config
from huggingface_hub import login
import subprocess
from django.http import StreamingHttpResponse

# Set environment variables for CUDA debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Ensures synchronous error reporting

# Retrieve API keys from environment variables
DEFAULT_WANDB_API_KEY = config("WANDB_API_KEY", default="")
DEFAULT_HF_API_KEY = config("HF_API_KEY", default="")

print(f"CUDA available: {torch.cuda.is_available()}")

def stream_training_output(request):
    # Start subprocess to run model training and capture terminal output
    def event_stream():
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # Ensure real-time output

        process = subprocess.Popen(
            [sys.executable, "manage.py", "runserver"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        for line in process.stdout:
            yield f"data: {line.strip()}\n\n"  # SSE format
            sys.stdout.flush()

        for line in process.stderr:
            yield f"data: [ERROR] {line.strip()}\n\n"
            sys.stdout.flush()

        process.stdout.close()
        process.stderr.close()

    return StreamingHttpResponse(event_stream(), content_type="text/event-stream")

def get_model_size(model_name: str) -> int:
    """
    Attempts to get model parameter count. If not directly available from config,
    fallback to a manual mapping for known models.
    """
    manual_model_sizes = {
        "meta-llama/Llama-3-3B": 3_000_000_000,
        "meta-llama/Llama-3.2-3B-Instruct": 3_000_000_000,
        "meta-llama/Llama-2-7b-hf": 7_000_000_000,
        "meta-llama/Llama-3-8B": 8_000_000_000,
        "meta-llama/Llama-2-13b-hf": 13_000_000_000,
        "google/gemma-2-2b-it":2_000_000_000,
        # Add more known models here
    }

    model_name_clean = model_name.lower()
    for key in manual_model_sizes:
        if key.lower() in model_name_clean:
            return manual_model_sizes[key]

    try:
        model_info = hf_hub_download(model_name, repo_type='model', filename='config.json',token=DEFAULT_HF_API_KEY)
        model_config = AutoConfig.from_pretrained(model_info)
        if hasattr(model_config, 'num_parameters'):
            return model_config.num_parameters()
    except Exception as e:
        print(f"Warning: Could not determine model size for {model_name}. Reason: {e}")

    return 0  # Default fallback

def get_target_modules(model_name: str):
    name = model_name.lower()
    if "llama" in name or "mistral" in name:
        return ["q_proj", "v_proj"]
    elif "falcon" in name:
        return ["query_key_value", "dense"]
    elif "bloom" in name:
        return ["query_key_value"]
    elif "gpt" in name:
        return ["c_attn"]
    else:
        return ["q_proj", "v_proj"]  # safe fallback

def train_model_view(request):
    if request.method == "POST":

        try:
            # Parse user-configurable parameters from the request
            model_name = request.POST.get("model_name", "gpt2")
            learning_rate = float(request.POST.get("learning_rate", 2e-5))
            num_epochs = int(request.POST.get("num_epochs", 3))
            batch_size = int(request.POST.get("batch_size", 1))
            project_name = request.POST.get("project_name", "your_project_name")
            gradient_checkpointing = request.POST.get("gradient_checkpointing") == "on"
            max_grad_norm = float(request.POST.get("max_grad_norm", 1.0))
            use_lora = request.POST.get("use_lora") == "on"
            use_qlora = request.POST.get("use_qlora") == "on"
            fp16 = request.POST.get("fp16") == "on"
            bf16 = request.POST.get("bf16") == "on"
            weight_decay = float(request.POST.get("weight_decay", 0.01))
            model_repo = request.POST.get("model_repo", "OpenFinAL/your-model-name")
            dataset_name = request.POST.get("dataset_name", "FinGPT/fingpt-fiqa_qa")  # User-specified dataset
            train_test_split_ratio = float(request.POST.get("train_test_split_ratio", 0.1))  # Split ratio
            model_size = get_model_size(model_name)

            # Only apply QLoRA if the model has 1.3B or more parameters
            if use_qlora and model_size < 1_300_000_000:
                return JsonResponse({
                    "status": "error",
                    "message": "QLoRA can only be applied to models with 1.3B parameters or more."
                })

            # Adjust precision: Only one active, or fallback to fp32
            if use_qlora:
                torch_dtype = None
                fp16 = False
                bf16 = False
            elif fp16:
                torch_dtype = torch.float16
                bf16 = False
            elif bf16:
                torch_dtype = torch.bfloat16
                fp16 = False
            else:
                torch_dtype = torch.float32

            # Load model with quantization if QLoRA is enabled
            if use_qlora:
                quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=True,
                    use_auth_token=DEFAULT_HF_API_KEY
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.bfloat16 if bf16 else None,
                    trust_remote_code=True,
                    use_auth_token=DEFAULT_HF_API_KEY
                )

            # Apply LoRA
            if use_lora or use_qlora:
                lora_config = LoraConfig(
                    r=8,  # LoRA rank
                    lora_alpha=32,
                    target_modules=get_target_modules(model_name),
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()

            # Retrieve API keys from the form or fall back to .env values
            wandb_key = request.POST.get("wandb_key") or DEFAULT_WANDB_API_KEY
            hf_key = request.POST.get("hf_key") or DEFAULT_HF_API_KEY

            if not wandb_key or not hf_key:
                return JsonResponse({
                    "status": "error",
                    "message": "Both W&B and Hugging Face API keys are required either in the .env file or via the form."
                })
            # Check for GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                return JsonResponse({
                    "status": "error",
                    "message": "No GPU found. Please ensure a GPU is available and properly configured."
                })

            # Initialize W&B
            wandb.login(key=wandb_key)
            wandb.init(project=project_name)

            # Login to Hugging Face
            login(token=hf_key)

            # Load dataset
            dataset = load_dataset(dataset_name)
            dataset = dataset.rename_column("input", "Question").rename_column("output", "Answer")
            dataset = dataset.remove_columns([col for col in dataset.column_names["train"] if col not in ["Question", "Answer"]])

            # # Identify existing columns (case-insensitive)
            # existing_columns = {col.lower(): col for col in dataset["train"].column_names}

            # # Define potential column mappings (lowercase for comparison)
            # column_mappings = {
            #     "input": "Question",
            #     "output": "Answer"
            # }

            # # Apply renaming only if necessary
            # for original_col, new_col in column_mappings.items():
            #     if original_col in existing_columns and new_col.lower() not in existing_columns:
            #         dataset = dataset.rename_column(existing_columns[original_col], new_col)

            # # Ensure we retain only "Question" and "Answer" columns, regardless of case
            # required_columns = set(existing_columns.get(col.lower(), col) for col in ["Question", "Answer"] if col.lower() in existing_columns)
            # dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in required_columns])

            # Split dataset based on user-provided ratio
            train_test_split = dataset["train"].train_test_split(test_size=train_test_split_ratio)
            train_dataset = train_test_split["train"]
            eval_dataset = train_test_split["test"]

            # Save train dataset to HF hub
            train_dataset.push_to_hub(f"{model_repo}-train-dataset", token=hf_key)

            # Save test dataset to HF hub
            eval_dataset.push_to_hub(f"{model_repo}-test-dataset", token=hf_key)

            # Load model and tokenizer dynamically with Meta and OpenELM support
            if "llama" in model_name.lower() or "meta" in model_name.lower() or "openelm" in model_name.lower():
                # If model is Llama, Meta, or OpenELM, use a special configuration
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False, trust_remote_code=True,use_auth_token=hf_key)
                tokenizer.add_bos_token = True  
                # dtype = torch.bfloat16 if bf16 else None
                # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)

            else:
                # Default to Hugging Face Auto classes for other models
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=hf_key)
                # dtype = torch.bfloat16 if bf16 else None
                # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)

            # Set padding token
            tokenizer.pad_token = tokenizer.eos_token
            model.resize_token_embeddings(len(tokenizer))
            model.to(device)
            
            # Tokenization function
            def tokenize_function(examples):
                inputs = tokenizer(
                    [f"{q} {a}" for q, a in zip(examples["Question"], examples["Answer"])],
                    padding="max_length",
                    truncation=True,
                    max_length=128  # Adjust max_length as needed
                )
                inputs["labels"] = inputs["input_ids"]
                return inputs

            # Split dataset and preprocess
            train_test_split = dataset["train"].train_test_split(test_size=train_test_split_ratio)
            train_dataset = train_test_split['train'].map(tokenize_function, batched=True)
            eval_dataset = train_test_split['test'].map(tokenize_function, batched=True)

            print("Sample tokenized data:", train_dataset[0])

            # Training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join("results", model_name),  # Results directory
                evaluation_strategy="epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                weight_decay=weight_decay,
                logging_dir=os.path.join("logs"),
                load_best_model_at_end=True,
                save_strategy="epoch",
                report_to="wandb",
                gradient_checkpointing=gradient_checkpointing,
                max_grad_norm=max_grad_norm,
                fp16=fp16,
                bf16=bf16,
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,  # Add tokenizer for tokenized output
            )

            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"✅ {name} requires grad")
                else:
                    print(f"⛔ {name} is frozen")

            # Train the model
            trainer.train()

            # Save model to Hugging Face directly
            model.push_to_hub(model_repo, use_auth_token=hf_key)
            tokenizer.push_to_hub(model_repo, use_auth_token=hf_key)

            # Cleanup
            del train_dataset, eval_dataset
            gc.collect()

            return JsonResponse({"status": "success", "message": f"Training completed successfully for {model_name}!"})

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})
    
    return render(request, "model_training.html")
