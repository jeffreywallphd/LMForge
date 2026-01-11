from django.shortcuts import render
from django.http import JsonResponse
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset
from decouple import config
import torch
import wandb
from huggingface_hub import login
import gc


DEFAULT_WANDB_API_KEY = config("WANDB_API_KEY", default="")
DEFAULT_HF_API_KEY = config("HF_API_KEY", default="")


def train_encoder_view(request):

    if request.method == "POST":
        try:
            learning_rate = float(request.POST.get("learning_rate", 2e-5))
            num_epochs = int(request.POST.get("num_epochs", 3))
            batch_size = int(request.POST.get("batch_size", 8))
            dataset_file = request.POST.get("dataset_name", "finance_qa.json")
            project_name = request.POST.get("project_name", "encoder_training_project")
            model_repo = request.POST.get("model_repo", "OpenFinAL/your-encoder-model")

            # Prefer explicitly provided keys from the form; fall back to configured defaults.
            # Treat empty strings as missing (strip whitespace) because decouple may return
            # empty values if .env contains keys with empty values.
            wandb_key_post = (request.POST.get("wandb_key") or "").strip()
            hf_key_post = (request.POST.get("hf_key") or "").strip()

            wandb_key = wandb_key_post if wandb_key_post else (DEFAULT_WANDB_API_KEY or "").strip()
            hf_key = hf_key_post if hf_key_post else (DEFAULT_HF_API_KEY or "").strip()

            # Debug log (non-sensitive): show whether keys were provided by form or defaults
            source_wandb = "form" if bool(wandb_key_post) else ("default" if bool(DEFAULT_WANDB_API_KEY and DEFAULT_WANDB_API_KEY.strip()) else "missing")
            source_hf = "form" if bool(hf_key_post) else ("default" if bool(DEFAULT_HF_API_KEY and DEFAULT_HF_API_KEY.strip()) else "missing")
            print(f"[encoder_training] wandb_key source={source_wandb}, hf_key source={source_hf}")

            if not (wandb_key and hf_key):
                return JsonResponse({"status": "error", "message": "Encoder - HF and W&B API keys required."})

            wandb.login(key=wandb_key)
            wandb.init(project=project_name)
            login(token=hf_key)

            # Check GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                return JsonResponse({"status": "error", "message": "No GPU found. Please configure CUDA."})

            print("Loading dataset...", flush=True)
            dataset = load_dataset("json", data_files=dataset_file)["train"]

            print("Loading tokenizer/model...", flush=True)
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)

            # ------------------ PREPROCESSING ------------------
            def preprocess_function(examples):
                input_ids = []
                attention_masks = []
                start_positions = []
                end_positions = []

                for example_list in examples["data"]:
                    example = example_list[0]

                    for paragraph in example["paragraphs"]:
                        context = paragraph["context"]

                        for qa in paragraph["qas"]:
                            question = qa["question"]
                            ans = qa["answers"][0]
                            start_char = ans["answer_start"]
                            end_char = start_char + len(ans["text"])

                            encoded = tokenizer(
                                question,
                                context,
                                max_length=384,
                                truncation="only_second",
                                stride=128,
                                return_overflowing_tokens=True,
                                return_offsets_mapping=True,
                                padding="max_length"
                            )

                            offset_mappings = encoded.pop("offset_mapping")

                            for i, offsets in enumerate(offset_mappings):

                                sequence_ids = encoded.sequence_ids(i)

                                # find context token span
                                context_token_ids = [idx for idx, sid in enumerate(sequence_ids) if sid == 1]

                                if not context_token_ids:
                                    continue

                                c_start = context_token_ids[0]
                                c_end = context_token_ids[-1]

                                start_pos = 0
                                end_pos = 0

                                # check answer fits inside span
                                if offsets[c_start][0] <= start_char and offsets[c_end][1] >= end_char:

                                    # locate start token
                                    for idx in range(c_start, c_end + 1):
                                        if offsets[idx][0] <= start_char <= offsets[idx][1]:
                                            start_pos = idx
                                            break

                                    # locate end token
                                    for idx in range(c_start, c_end + 1):
                                        if offsets[idx][0] <= end_char <= offsets[idx][1]:
                                            end_pos = idx
                                            break

                                # collect
                                input_ids.append(encoded["input_ids"][i])
                                attention_masks.append(encoded["attention_mask"][i])
                                start_positions.append(start_pos)
                                end_positions.append(end_pos)

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_masks,
                    "start_positions": start_positions,
                    "end_positions": end_positions,
                }

            print("Tokenizing dataset...", flush=True)
            tokenized_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=["data"]
            )

            print("Training...", flush=True)
            args = TrainingArguments(
                output_dir="./bert_output",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                num_train_epochs=num_epochs,
                weight_decay=0.01,
                save_strategy="epoch",
                logging_steps=25,
                report_to="wandb"
            )

            trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset)
            trainer.train()

            print("Uploading model...", flush=True)
            model.push_to_hub(model_repo)
            tokenizer.push_to_hub(model_repo)

            wandb.finish()
            torch.cuda.empty_cache()
            gc.collect()

            return JsonResponse({"status": "success", "message": "Training complete!"})

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})

    return render(request, "encoder_training.html")
