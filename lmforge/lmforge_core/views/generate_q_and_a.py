from django.shortcuts import render, redirect
from ..forms.forms import DocumentForm, DocumentProcessingForm
from ..models.scraped_data import ScrapedData, ScrapedDataMeta
from ..utils.qdrant_utils import upsert_qa_items, search_similar
from PyPDF2 import PdfReader
from django.shortcuts import get_object_or_404
from django.contrib import messages
import json
from django.http import JsonResponse, HttpResponse
from django.urls import reverse
from django.http import HttpResponseRedirect
import openai
from datetime import datetime
from openai import OpenAI, OpenAIError
import tiktoken
from decouple import config
from django.db.models import F, Func, Value
from django.core.paginator import Paginator
import pandas as pd
import io
import csv
from huggingface_hub import HfApi
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging
from huggingface_hub import login
import transformers
import re
from huggingface_hub.utils import LocalTokenNotFoundError

# Set up logging
# Set log file name (in current directory)
log_file = "application.log"

# Setup logging to file + console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEFAULT_HF_API_KEY = config("HUGGINGFACE_TOKEN", default="")
# setting huggingface token

def get_model_and_tokenizer():
    if not DEFAULT_HF_API_KEY:
        logging.warning("No Hugging Face token found. Skipping login and model loading.")
        return None, None, device

    try:
        login(token=DEFAULT_HF_API_KEY)

        os.environ["HF_HOME"] = "D:/huggingface_cache" 
        os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
        os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/huggingface_cache"
        logging.info("Hugging Face environment variables set.")

        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        model.to(device)
        return model, tokenizer, device

    except LocalTokenNotFoundError:
        logging.error("Hugging Face token missing. Please configure HUGGINGFACE_TOKEN in .env.")
        return None, None, device
    except Exception as e:
        logging.error(f"Unexpected error while loading model: {e}")
        return None, None, device

# Tokenizer function to count tokens in a text
def count_tokens(text):
    model, tokenizer, _ = get_model_and_tokenizer()
    if not tokenizer:
        logging.warning("No tokenizer available. Returning approximate token count.")
        return len(text.split())  # fallback: rough word count
    return len(tokenizer.encode(text))


# Function to split text into segments while respecting token limits
def split_text(text, max_tokens=1000):
    paragraphs = text.split("\n\n")  # Split by paragraph
    chunks = []
    current_chunk = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = count_tokens(paragraph)
        if current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_tokens = para_tokens
        else:
            current_chunk.append(paragraph)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def llama_chat(prompt: str, max_tokens: int = 25): 
    model, tokenizer, _ = get_model_and_tokenizer()
    if not model or not tokenizer:
        logging.warning("No Hugging Face model loaded. Returning empty response.")
        return "" 
    
    try:
        tokenizer, model = get_model_and_tokenizer()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_tokens = output_tokens[0][len(inputs["input_ids"][0]):]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        logging.info(f"Generated text: {generated_text}")

        # Try extracting JSON manually in extract_qa — don't parse here
        return generated_text

    except Exception as e:
        logging.error(f"Error in llama_chat: {e}")
        return ""

def build_prompt(chunk, questions_num, instruction_prompt=""):
    instruction_part = f''', "instruction": "{instruction_prompt.strip()}"''' if instruction_prompt else ""
    return f"""You are a system that generates question-answer pairs in valid JSON only.

Instructions:
- Generate exactly {questions_num} question-answer pairs.
- Each answer must be 200 words or less
- Output only a valid JSON array of objects.
- No extra text, no comments, no markdown.

Input Text:
\"\"\"{chunk}\"\"\"

Output format:
[
  {{"question": "What is ...?", "answer": "The answer is ..."{instruction_part}}}
]

Respond with only valid JSON.
"""


def extract_qa(text, chunk_limit, questions_num=1, instruction_prompt=""):
    text_chunks = split_text(text, max_tokens=256)  # Adjust as needed
    results = []
    total_chunks = len(text_chunks)

    for i, chunk in enumerate(text_chunks[:chunk_limit]):
        logging.info(f"Processing chunk {i+1}/{total_chunks}")

        strict_prompt = build_prompt(chunk, questions_num, instruction_prompt)
        model_output = llama_chat(strict_prompt, max_tokens=256)

        # Try to extract JSON list
        match = re.search(r'\[\s*{.*?}\s*\]', model_output, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            # Fallback: try cleaning up common LLM quirks
            cleaned_output = model_output.strip()
            cleaned_output = cleaned_output.split("```")[0]  # Remove markdown fences
            cleaned_output = cleaned_output.replace('\n', ' ').replace('\r', '')
            start = cleaned_output.find('[')
            end = cleaned_output.rfind(']')
            if start != -1 and end != -1 and start < end:
                json_str = cleaned_output[start:end+1]
            else:
                logging.error(f"Could not find valid JSON in model output:\n{model_output}")
                continue

        # Try to parse
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                results.extend(data)
            else:
                logging.warning("Parsed JSON is not a list.")
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error: {e}\nProblem JSON:\n{json_str}")


    return json.dumps(results, indent=4)


def generate_q_and_a(request):
    documents_list = ScrapedDataMeta.objects.all().order_by('-created_at')  # Order by latest entries

    # Apply pagination (10 documents per page)
    paginator = Paginator(documents_list, 10)
    page_number = request.GET.get('page')
    documents = paginator.get_page(page_number)

    form = DocumentForm()

    return render(request, "generate_q_and_a.html", {"form": form, "documents":documents})


def document_detail(request):
    selected_document_ids = request.POST.getlist('selected_documents')
    # print(selected_document_ids)

    if not selected_document_ids:
        request.session["redirect_message"] = "You must select at least one document to proceed."
        redirect_url = reverse('dataset-workflow')
        return redirect(redirect_url)

    documents = ScrapedData.objects.filter(id__in=selected_document_ids)
    # print(documents)

    combined_text = "\n\n".join([doc.content for doc in documents])
    text_chunks = split_text(combined_text, max_tokens=1000)  # Adjust chunk size
    total_chunks = len(text_chunks)

    # after generated_json_text = extract_qa(...)
    generated_json_data = json.loads(generated_json_text)

    # optional: enrich each QA item with source metadata
    for idx, item in enumerate(generated_json_data):
        # add any metadata you want searchable later
        item.setdefault('chunk_index', idx)
        item.setdefault('source_documents', selected_document_ids)
        item.setdefault('created_at', datetime.utcnow().isoformat())

    # Upsert into Qdrant (collection name can be customized)
    try:
        upsert_result = upsert_qa_items(generated_json_data, collection_name="qa_chunks")
        logging.info(f"Upsert result: {upsert_result}")
    except Exception as e:
        logging.error(f"Failed to upsert to Qdrant: {e}")

    if request.method == "POST":
        if selected_document_ids:
            request.session['selected_document_ids'] = selected_document_ids


        form = DocumentProcessingForm(request.POST)
        if form.is_valid():
            test_type = form.cleaned_data['test_type']
            num_questions = form.cleaned_data['num_questions']
            num_paragraphs = form.cleaned_data['num_paragraphs']
            instruction_prompt = form.cleaned_data["instruction_prompt"]

            if test_type == 'mockup':
                json_file_path = 'media/JSON/New_Prompt_Simple_QA.json'
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as file:
                        generated_json_text = file.read()

                    generated_json_data = json.loads(generated_json_text)
                    request.session[f'generated_json_combined'] = generated_json_text
                except (FileNotFoundError, json.JSONDecodeError):
                    logging.error(f"Error reading mock-up JSON file: {json_file_path}")
                    generated_json_data = {"error": "Mock-up JSON file not found or invalid"}
            
            else:
                try:
                    generated_json_text = extract_qa(text=combined_text, chunk_limit = num_paragraphs, questions_num=num_questions, instruction_prompt=instruction_prompt)
                    generated_json_data = json.loads(generated_json_text)
                    request.session['generated_json_combined'] = generated_json_text

                except Exception as e:
                    logging.error(f"Error generating Q&A: {e}")
                    

    else:
        form = DocumentProcessingForm()

    return render(request, 'document_detail.html', {
        'documents': documents,
        'form': form,
        'json_data': generated_json_data, 
        'selected_document_ids': selected_document_ids, 
        'total_chunks': total_chunks,
    })


def download_json(request):
    """Serve the generated JSON data as a downloadable file."""
    session_key = f'generated_json_combined'
    generated_json_text = request.session.get(session_key, None)

    # print(generated_json_text)
    # print(type(generated_json_text))

    if generated_json_text is None:
        return JsonResponse({"error": "No generated JSON available for this document"}, status=404)

    response = HttpResponse(
        generated_json_text,
        content_type="application/json"
    )
    response['Content-Disposition'] = f'attachment; filename="document.json"'
    return response


def download_csv(request):
    """Serve the generated JSON data as a downloadable CSV file."""
    session_key = 'generated_json_combined'
    generated_json_text = request.session.get(session_key, None)

    if generated_json_text is None:
        return JsonResponse({"error": "No generated JSON available for this document"}, status=404)

    try:
        json_data = json.loads(generated_json_text)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)

    # Create a response object with CSV content
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="document.csv"'

    # Assuming JSON data is a list of dictionaries
    if isinstance(json_data, list) and json_data:
        fieldnames = json_data[0].keys()  # Get column headers from the first item
        writer = csv.DictWriter(response, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_data)
        print(response)
    else:
        return JsonResponse({"error": "Invalid JSON structure for CSV conversion"}, status=400)

    return response

def upload_parquet_to_huggingface(request):
    """Convert JSON string to Parquet and upload to Hugging Face Hub."""
    session_key = 'generated_json_combined'
    generated_json_text = request.session.get(session_key, None)

    if generated_json_text is None:
        return JsonResponse({"error": "No generated JSON available for this document"}, status=404)
    
    file_name = request.GET.get("file_name", "").strip().replace(" ", "_")
    repo_name = request.GET.get("repo_name", "").strip().replace(" ", "_")

    if not file_name:
        return JsonResponse({"error": "No file name provided"}, status=400)
    
    if not repo_name:
        return JsonResponse({"error": "No repository name provided"}, status=400)

    try:
        json_data = json.loads(generated_json_text)

        if not isinstance(json_data, list) or not all(isinstance(item, dict) for item in json_data):
            return JsonResponse({"error": "Invalid JSON format: Expected a list of dictionaries"}, status=400)

        df = pd.DataFrame(json_data)

        repo_id = f"{repo_name}/{file_name}"  

        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(repo_id)

        return JsonResponse({"success": f"Uploaded successfully as {file_name}, under database {repo_id}"}, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    

def get_huggingface_datasets(request):
    """Fetch the user's Hugging Face datasets and return as JSON."""
    try:
        api = HfApi(token=DEFAULT_HF_API_KEY)
        user_info = api.whoami()
        organizations = user_info.get("orgs", [])

        if organizations:
            dataset_list = [org["name"] for org in organizations]
        else:
            dataset_list = [user_info["name"]]

        #datasets = list(api.list_datasets(author=org_name))
        #dataset_list = [dataset.id for dataset in datasets]

        return JsonResponse({"datasets": dataset_list})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def search_qa(request):
    if request.method not in ["GET", "POST"]:
        return JsonResponse({"error": "Invalid request method."}, status=405)

    # Extract query string (from GET or POST)
    q = request.GET.get("q") if request.method == "GET" else request.POST.get("q")
    q = (q or "").strip()
    if not q:
        return JsonResponse({"error": "Missing query parameter 'q'."}, status=400)

    # Extract optional top_k value
    top_k = request.GET.get("k") if request.method == "GET" else request.POST.get("k")
    top_k = int(top_k) if top_k and top_k.isdigit() else 5

    try:
        results = search_similar(q, top_k=top_k, collection_name="qa_chunks")
        return JsonResponse({"results": results})
    except Exception as e:
        logging.error(f"Search error: {e}")
        return JsonResponse({"error": str(e)}, status=500)