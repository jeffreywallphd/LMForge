import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return the number of GPUs
print(torch.cuda.get_device_name(0))  # Should show the GPU model


import transformers
import torch
import os
import json
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import time
from datetime import timedelta, datetime
import pandas as pd
from dotenv import load_dotenv
import shutil 


import bitsandbytes as bnb

import evaluate
from sentence_transformers import SentenceTransformer, util
import numpy as np




print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    # Create a dummy 8-bit layer on the GPU
    layer = bnb.nn.Linear8bitLt(32, 32, has_fp16_weights=False, threshold=6.0).cuda()
    
    # Create dummy input
    inp = torch.randn(1, 32).cuda().half() # half precision for Blackwell
    
    # Run a forward pass
    out = layer(inp)
    print("SUCCESS: BitsAndBytes 8-bit kernel executed correctly on this GPU.")
    
except Exception as e:
    print("\nFAILURE: BitsAndBytes could not run on this GPU.")
    print(f"Error details: {e}")

# Load environment variables
load_dotenv(dotenv_path="../../.env") # path is relative to this script, adjust as needed

run_id = "LMForge_RUN08_DGX_SPARK_Llama-3-3-70B-Instruct"  # <- Change this manually for each experiment
batch_size = 10  # <- Change this manually for each experiment

#from transformers.utils import LossKwargs


import logging
logging.basicConfig(filename='generation.log', level=logging.INFO)
logging.info(f"Run ID: {run_id}")


# setting huggingface token
login(token=os.getenv("#"))

# os.environ["HF_HOME"] = "D:/huggingface_cache" 
# os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/huggingface_cache"

# print("HF_HOME:", os.getenv("HF_HOME"))
# print("TRANSFORMERS_CACHE:", os.getenv("TRANSFORMERS_CACHE"))
# print("HUGGINGFACE_HUB_CACHE:", os.getenv("HUGGINGFACE_HUB_CACHE"))

# logging.info(f"HF_HOME: {os.getenv('HF_HOME')}")
# logging.info(f"TRANSFORMERS_CACHE: {os.getenv('TRANSFORMERS_CACHE')}")
# logging.info(f"HUGGINGFACE_HUB_CACHE: {os.getenv('HUGGINGFACE_HUB_CACHE')}")

# transformers.utils.hub.TRANSFORMERS_CACHE = "D:/huggingface_cache"



model_name = "meta-llama/Llama-3.3-70B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True   
)



tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config, 
    device_map="auto",              
    trust_remote_code=True
)


print("Model loaded successfully!")
print(f"Memory footprint: {model.get_memory_footprint() / 1e9} GB")


chunk_sizes = [128, 256, 512, 1024]
questions_num = 2
max_token_list = [128,256,512,1024,2048]


results_df = pd.DataFrame(columns=[
    "chunk_size", "questions_num", "qa_count_mismatch", "total_questions", "token_Size",
    "total_chunks", "success_count", "fail_count",
    "elapsed_time"
])


def power_analysis(chunk_size, max_tokens, qa_results,substring_date,elapsed_time):
    """
    Perform power analysis based on the provided parameters for the current run.
    """
    
    # https://huggingface.co/spaces/evaluate-metric/bertscore
    # https://huggingface.co/tasks/sentence-similarity
    # 1 Metric: ROUGE
    rouge = evaluate.load("rouge")

    originals = []
    generations = []

    for doc in qa_results.values():
        for item in doc:
            chunk = item.get("chunk")
            qa_pairs = item.get("qa_pairs", [])
            if not chunk or not isinstance(qa_pairs, list):
                continue  # Skip if chunk is missing or qa_pairs is not a list
            for pair in qa_pairs:
                answer = pair.get("answer") if isinstance(pair, dict) else None
                if answer:  # Only add if answer exists and is not None/empty
                    originals.append(str(chunk))
                    generations.append(str(answer))


    scores = rouge.compute(predictions=generations, references=originals)
    print(f"ROUGE Scores: {scores}")
    logging.info(f"ROUGE Scores: {scores} for chunk_size {chunk_size}, max_tokens {max_tokens}, questions_num {questions_num}")

    # 2 Metric: BERTScore
    bertscore = evaluate.load("bertscore")
    bert_scores = bertscore.compute(predictions=generations, references=originals, model_type="bert-base-uncased", lang="en")
    P = bert_scores["precision"]
    R = bert_scores["recall"]
    F1 = bert_scores["f1"] 

    print(f"BERTScore: {bert_scores}")
    logging.info(f"BERTScore: {bert_scores} for chunk_size {chunk_size}, max_tokens {max_tokens}, questions_num {questions_num}")

    # 3 Metric: STS (Semantic Textual Similarity)
    sts_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    original_embeddings = sts_model.encode(originals, convert_to_tensor=True) 
    generated_embeddings = sts_model.encode(generations, convert_to_tensor=True)
    sts_scores = util.pytorch_cos_sim(original_embeddings, generated_embeddings).diagonal().cpu().tolist()

    print(f"STS Scores: {sts_scores}")
    logging.info(f"STS Scores: {sts_scores} for chunk_size {chunk_size}, max_tokens {max_tokens}, questions_num {questions_num}")

    # save the scores to a CSV file
    scores_df = pd.DataFrame({
        "chunk_size": [chunk_size],
        "max_tokens": [max_tokens],
        "questions_num": [questions_num],
        "rouge1": [scores["rouge1"]],
        "rouge2": [scores["rouge2"]],
        "rougeL": [scores["rougeL"]],
        "rougeLsum": [scores["rougeLsum"]],
        "bert_score_P": [np.mean(P)],
        "bert_score_R": [np.mean(R)],
        "bert_score_F1": [np.mean(F1)],
        "sts_score": [np.mean(sts_scores)],
        "substring_date": [substring_date],
        "elapsed_time": [elapsed_time],
    })
    
    print("Scores saved to scores.csv")   
    logging.info(f"Scores saved to scores.csv for chunk_size {chunk_size}, max_tokens {max_tokens}, questions_num {questions_num}")
    return scores_df



def build_prompt(chunk, questions_num):
    return f"""
Generate {questions_num} question-answer pairs based on the following text segment. 
Return the result in valid JSON format as a list of objects.

Text Segment:

{chunk}

Response Format:
[
    {{"question": "generated question", "answer": "generated Answer"}},
]

Question answers should be at least 250 words long.

Do NOT include any explanation or preamble before or after the JSON output.
Return ONLY valid JSON output.

Answer:
    """



def load_data(chunk_size):
    path = f"../Spring25/Generate_Paragraphs/Results/extracted_chunks_{chunk_size}_overlap.json"
    if not os.path.exists(path):
        print(f"Missing input file: {path}, skipping.")
        logging.info(f"Missing input file: {path}, skipping.")
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
    



def update_detailed_tracker(detailed_tracker_path,chunk_size, max_tokens, key, value):
    if os.path.exists(detailed_tracker_path):
        detailed_tracker_df = pd.read_csv(detailed_tracker_path)
    else:
        detailed_tracker_df = pd.DataFrame(columns=["chunk_size", "questions_num", "qa_count_mismatch", "total_questions", 
            "max_tokens", "total_chunks", "success_count", "fail_count", "repeat_count", "duplicate_count", "elapsed_time"])
    # Check if the row already exists
    row_match = (
        (detailed_tracker_df["chunk_size"] == chunk_size) &
        (detailed_tracker_df["max_tokens"] == max_tokens) 
    )
    if not detailed_tracker_df.loc[row_match].empty:
        # Update the existing row
        detailed_tracker_df.loc[row_match, key] = value
    else:
        # Add a new row
        new_row = {
            "chunk_size": chunk_size,
            "max_tokens": max_tokens,
            "questions_num": 0,
            "qa_count_mismatch": 0,
            "total_questions": 0,
            "total_chunks": 0,
            "success_count": 0,
            "fail_count": 0,
            "repeat_count": 0,
            "duplicate_count": 0,
            "elapsed_time": 0
        }
        new_row[key] = value
        # Append the new row to the DataFrame
        detailed_tracker_df = pd.concat([detailed_tracker_df, pd.DataFrame([new_row])], ignore_index=True)
        
    # Save the updated DataFrame to CSV
    detailed_tracker_df.to_csv(detailed_tracker_path, index=False)
    print(f"Updated detailed tracker: {detailed_tracker_path}")
    logging.info(f"Updated detailed tracker: {detailed_tracker_path}")


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)




# check if base directory exists, if not create it
if not os.path.exists(run_id):
    os.makedirs(run_id)
# Constants
check_point_path = f"{run_id}/qa_run_tracker.csv"
output_base = f"{run_id}/Generated_Results/Llama-3.3-70B-Instruct-spark"
detailed_tracker_path = f"{run_id}/qa_run_tracker_detailed.csv"

# Initialize tracker CSV if not present
if not os.path.exists(check_point_path):
    tracker_df = pd.DataFrame(columns=["chunk_size", "max_tokens", "questions_num", "completed"])
    for chunk_size in chunk_sizes:
        for max_tokens in max_token_list:
            tracker_df.loc[len(tracker_df)] = [chunk_size, max_tokens, questions_num, False]
    tracker_df.to_csv(check_point_path, index=False)
else:
    tracker_df = pd.read_csv(check_point_path)
    

# Results summary
results_df = pd.DataFrame(columns=[
    "chunk_size", "questions_num", "qa_count_mismatch", "total_questions",
    "max_tokens", "total_chunks", "success_count", "fail_count", "repeat_count",
    "duplicate_count", "elapsed_time"
])

# Check if detailed tracker exists, if not create it
if not os.path.exists(detailed_tracker_path):
    detailed_tracker_df = pd.DataFrame(columns=["chunk_size", "questions_num", "qa_count_mismatch", "total_questions", 
        "max_tokens", "total_chunks", "success_count", "fail_count", "repeat_count", "duplicate_count", "elapsed_time"])
    detailed_tracker_df.to_csv(detailed_tracker_path, index=False)
else:
    detailed_tracker_df = pd.read_csv(detailed_tracker_path)

for chunk_size in chunk_sizes:
    chunk_data = load_data(chunk_size)

    for max_tokens in max_token_list:
        row_match = (
            
            (tracker_df["chunk_size"] == chunk_size) &
            (tracker_df["max_tokens"] == max_tokens) &
            (tracker_df["questions_num"] == questions_num)
        )

        if tracker_df.loc[row_match, "completed"].any():
            print(f"Skipping chunk_size={chunk_size}, max_tokens={max_tokens} (already completed)")
            logging.info(f"Skipping chunk_size={chunk_size}, max_tokens={max_tokens} (already completed)")
            continue
        print(f"Processing chunk_size={chunk_size}, max_tokens={max_tokens}")
        logging.info(f"Processing chunk_size={chunk_size}, max_tokens={max_tokens}")

        output_file_path = f"{output_base}/generation_log_{chunk_size}_Token_{max_tokens}_Q{questions_num}.json"

        # Load existing results if file exists
        if os.path.exists(output_file_path):
            try:
                with open(output_file_path, "r", encoding="utf-8") as f:
                    qa_results = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Output file is corrupted. Starting fresh.")
                logging.info("Warning: Output file is corrupted. Starting fresh.")
                qa_results = {}
        else:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            qa_results = {}

        # Trackers
        total_chunks = 0
        success_count = 0
        fail_count = 0
        total_questions = 0
        qa_count_mismatch = 0
        repeat_count = 0
        duplicate_count = 0
        chunk_counter = 0

        start_time = time.time()

        for doc_name, chunks in chunk_data.items():
            if doc_name in qa_results and qa_results[doc_name]:
                print(f"Skipping {doc_name} (already processed)")
                logging.info(f"Skipping {doc_name} (already processed)")
                continue
            print(f"Processing {doc_name}...")
            logging.info(f"Processing {doc_name}...")
            # Initialize the document in the results dictionary
            qa_results[doc_name] = []

            for chunk in chunks[:900]:  # Adjust slice as needed
                total_chunks += 1
                update_detailed_tracker(detailed_tracker_path,chunk_size, max_tokens, "total_chunks", total_chunks)
                chunk_counter += 1
                prompt = build_prompt(chunk, questions_num)
                try:
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        output_tokens = model.generate(**inputs, max_new_tokens=max_tokens,pad_token_id=tokenizer.eos_token_id)
                    generated_tokens = output_tokens[0][len(inputs["input_ids"][0]):]
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    print(generated_text)
                    qa_pairs = json.loads(generated_text)

                    if isinstance(qa_pairs, list):
                        qa_results[doc_name].append({
                            "chunk": chunk,
                            "qa_pairs": qa_pairs
                        })
                        success_count += 1
                        update_detailed_tracker(detailed_tracker_path,chunk_size, max_tokens, "success_count", success_count)
                        total_questions += len(qa_pairs)
                        update_detailed_tracker(detailed_tracker_path,chunk_size, max_tokens, "total_questions", total_questions)

                        if len(qa_pairs) != questions_num:
                            qa_count_mismatch += 1
                            update_detailed_tracker(detailed_tracker_path,chunk_size, max_tokens, "qa_count_mismatch", qa_count_mismatch)
                            logging.info(f"Warning: Expected {questions_num} questions, got {len(qa_pairs)}")
                        
                        #  question and answer are the same
                        for pair in qa_pairs:
                            if pair["question"] == pair["answer"]:
                                repeat_count += 1
                                update_detailed_tracker(detailed_tracker_path,chunk_size, max_tokens, "repeat_count", repeat_count)
                                logging.info(f"Warning: Question and answer are the same in {doc_name}")
                            # check for duplicates in the same chunk
                            if any(pair["question"] == p["question"] for p in qa_pairs if p != pair):
                                duplicate_count += 1
                                update_detailed_tracker(detailed_tracker_path,chunk_size, max_tokens, "duplicate_count", duplicate_count)
                                logging.info(f"Warning: Duplicate question in {doc_name}")
                    else:
                        fail_count += 1
                        update_detailed_tracker(detailed_tracker_path,chunk_size, max_tokens, "fail_count", fail_count)
                        logging.info(f"Warning: Invalid JSON format in {doc_name}: {generated_text}")

                except Exception as e:
                    print(f"Error processing chunk from {doc_name}: {e}")
                    logging.error(f"Error processing chunk from {doc_name}: {e}")
                    fail_count += 1
                    update_detailed_tracker(detailed_tracker_path,chunk_size, max_tokens, "fail_count", fail_count)
                    continue

                # Save every batch_size chunks
                if chunk_counter % batch_size == 0:
                    with open(output_file_path, "w", encoding="utf-8") as out_file:
                        json.dump(qa_results, out_file, indent=4, ensure_ascii=False)
                    print(f"Checkpoint saved at chunk {chunk_counter}")
                    logging.info(f"Checkpoint saved at chunk {chunk_counter}")

                    # update the detailed tracker
                    

            # Save after each document
            with open(output_file_path, "w", encoding="utf-8") as out_file:
                json.dump(qa_results, out_file, indent=4, ensure_ascii=False)
            print(f"Saved full doc: {doc_name}")
            logging.info(f"Saved full doc: {doc_name}")

        # Final save
        end_time = time.time()
        elapsed_time = timedelta(seconds=end_time - start_time)

        results_df.loc[len(results_df)] = [
            chunk_size, questions_num, qa_count_mismatch, total_questions,
            max_tokens, total_chunks, success_count, fail_count, repeat_count,
            duplicate_count, str(elapsed_time)
        ]

        print(f"Completed {chunk_size}, {max_tokens} | Time: {elapsed_time}")
        logging.info(f"Completed {chunk_size}, {max_tokens} | Time: {elapsed_time}")

        # Update tracker
        tracker_df.loc[row_match, "completed"] = True
        tracker_df.to_csv(check_point_path, index=False)

        scores_df = power_analysis(chunk_size, max_tokens, qa_results,run_id,elapsed_time)
        if os.path.exists(f"{output_base}/scores.csv"):
            scores_df.to_csv(f"{output_base}/scores.csv", mode='a', header=False, index=False)
        else:
            os.makedirs(f"{output_base}/scores", exist_ok=True)
            scores_df.to_csv(f"{output_base}/scores.csv", index=False)
        # save the results to a CSV file
        if os.path.exists(f"{output_base}/qa_generation_results.csv"):
            results_df.to_csv(f"{output_base}/qa_generation_results.csv", mode='a', header=False, index=False)
        else:
            os.makedirs(f"{output_base}/qa_generation_results", exist_ok=True)
            results_df.to_csv(f"{output_base}/qa_generation_results.csv", index=False)
        print(f"Results saved to {output_base}/qa_generation_results.csv")
        logging.info(f"Results saved to {output_base}/qa_generation_results.csv")


# Save summary CSV
csv_output_path = f"{output_base}/qa_generation_results_summary.csv"
results_df.to_csv(csv_output_path, index=False)
print(f"\nSummary saved to {csv_output_path}")
logging.info(f"\nSummary saved to {csv_output_path}")
