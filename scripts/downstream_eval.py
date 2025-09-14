#!/usr/bin/env python3
import os, time, json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Aliases for convenience
ALIAS = {
    "mbert": "bert-base-multilingual-cased",  # encoder only (not ideal for MMLU)
    "xlmr": "xlm-roberta-base",  # encoder only (same issue)
    "mt5": "google/mt5-base",
    "mgpt": "ai-forever/mGPT",
    "bloomz": "bigscience/bloomz-560m",
    "xglm": "facebook/xglm-564M",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}

# Language codes for Multilingual MMLU
SUPPORTED_LANGUAGES = [
    "ar", "bn", "ca", "da", "de", "es", "eu", "fr", "gu", "hi", 
    "hr", "hu", "hy", "id", "it", "kn", "ml", "mr", "ne", "nl", 
    "pt", "ro", "ru", "sk", "sr", "sv", "ta", "te", "uk", "vi", "zh"
]

def parse_models_arg(models_arg: str):
    """Parse comma-separated model names/aliases"""
    ids = []
    for tok in models_arg.split(","):
        tok = tok.strip()
        ids.append(ALIAS.get(tok.lower(), tok))
    return ids

def format_example(example, language="en"):
    """Format Multilingual MMLU example into a multiple-choice prompt"""
    question = example['Question']
    choices = [example['A'], example['B'], example['C'], example['D']]
    
    # Create a formatted prompt
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        choice_letter = chr(65 + i)  # A, B, C, D
        prompt += f"{choice_letter}. {choice}\n"
    prompt += "Answer: "
    
    # Get the correct answer index (convert letter to index)
    correct_answer_letter = example['Answer']
    correct_answer_idx = ord(correct_answer_letter) - ord('A')  # Convert A->0, B->1, etc.
    
    return prompt, correct_answer_idx

def get_choice_logprobs(model, tokenizer, prompt, choices, device, max_length=512):
    """Calculate log probabilities for each choice"""
    choice_logprobs = []
    
    # First check if prompt alone is too long
    prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    if prompt_inputs.input_ids.shape[1] >= max_length - 2:  # Leave room for answer token
        # Truncate the prompt from the beginning, keeping the "Answer: " part
        prompt_tokens = prompt_inputs.input_ids[0].tolist()
        # Find "Answer: " or truncate intelligently
        answer_pos = prompt.rfind("Answer: ")
        if answer_pos != -1:
            # Keep the question + choices + "Answer: " but truncate the question if needed
            prefix = prompt[:answer_pos]
            suffix = prompt[answer_pos:]
            # Tokenize parts
            suffix_tokens = tokenizer(suffix, add_special_tokens=False).input_ids
            max_prefix_length = max_length - len(suffix_tokens) - 10  # Leave some buffer
            
            if max_prefix_length > 50:  # Ensure we have reasonable space for question
                prefix_tokens = tokenizer(prefix, add_special_tokens=False).input_ids
                if len(prefix_tokens) > max_prefix_length:
                    # Truncate from the beginning of the question
                    truncated_prefix_tokens = prefix_tokens[-max_prefix_length:]
                    prompt = tokenizer.decode(truncated_prefix_tokens + suffix_tokens, skip_special_tokens=True)
                    # Clean up any incomplete words at the start
                    prompt = "..." + prompt[prompt.find(" "):] if " " in prompt else prompt
    
    # Re-tokenize the (potentially truncated) prompt
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length-2).to(device)
    prompt_length = prompt_inputs.input_ids.shape[1]
    
    for i, choice in enumerate(choices):
        choice_letter = chr(65 + i)  # A, B, C, D
        full_text = prompt + choice_letter
        
        # Tokenize full text with truncation
        full_inputs = tokenizer(
            full_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            add_special_tokens=True
        ).to(device)
        
        # Skip if the sequence is still too long (shouldn't happen with truncation)
        if full_inputs.input_ids.shape[1] > max_length:
            logger.warning(f"Sequence still too long after truncation: {full_inputs.input_ids.shape[1]}")
            choice_logprobs.append(float('-inf'))
            continue
        
        with torch.no_grad():
            outputs = model(**full_inputs)
            logits = outputs.logits
            
            # Get the logits for the choice token
            choice_token_ids = tokenizer.encode(choice_letter, add_special_tokens=False)
            if len(choice_token_ids) > 0:
                choice_token_id = choice_token_ids[0]
                # Make sure we don't go out of bounds
                logit_pos = min(prompt_length - 1, logits.shape[1] - 1)
                choice_logit = logits[0, logit_pos, choice_token_id].item()
                choice_logprobs.append(choice_logit)
            else:
                # Fallback if tokenization fails
                choice_logprobs.append(float('-inf'))
    
    return choice_logprobs

def evaluate_model_on_language(model_id, language="en", subset=None, max_samples=None):
    """Evaluate model on specific language"""
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Get model's maximum sequence length
        max_length = getattr(model.config, 'max_position_embeddings', 512)
        max_length = min(max_length, getattr(model.config, 'n_positions', max_length))
        max_length = min(max_length, 512)  # Conservative limit
        logger.info(f"Using max sequence length: {max_length}")
        
        device = next(model.parameters()).device
        model.eval()
        
        # Load dataset for specific language
        dataset_name = f"openai/MMMLU"
        dataset = load_dataset(dataset_name, language)["test"]
        
        # Limit samples if specified
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        correct, total = 0, 0
        skipped = 0
        
        logger.info(f"Evaluating {model_id} on {language} with {len(dataset)} samples")
        
        for i, example in enumerate(dataset):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(dataset)} samples (skipped: {skipped})")
            
            try:
                prompt, correct_answer_idx = format_example(example, language)
                choices = [example['A'], example['B'], example['C'], example['D']]
                
                # Check if the prompt is reasonable length (rough check)
                if len(prompt) > max_length * 4:  # Rough chars to tokens ratio
                    logger.warning(f"Skipping very long sample {i} (length: {len(prompt)})")
                    skipped += 1
                    continue
                
                # Get log probabilities for each choice - FIXED FUNCTION CALL
                choice_logprobs = get_choice_logprobs(model, tokenizer, prompt, choices, device, max_length)
                
                # Skip if all logprobs are invalid
                if all(logprob == float('-inf') for logprob in choice_logprobs):
                    logger.warning(f"Skipping sample {i} - all choices invalid")
                    skipped += 1
                    continue
                
                # Predict the choice with highest log probability
                predicted_idx = np.argmax(choice_logprobs)
                
                if predicted_idx == correct_answer_idx:
                    correct += 1
                total += 1
                    
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                skipped += 1
                continue
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Completed evaluation: {correct}/{total} correct ({accuracy:.4f}), skipped: {skipped}")
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return accuracy, total
        
    except Exception as e:
        logger.error(f"Error evaluating {model_id} on {language}: {e}")
        return 0.0, 0

def evaluate_model(model_id, languages=["en"], max_samples=None):
    """Evaluate model across multiple languages"""
    results = {}
    
    for lang in languages:
        logger.info(f"\n=== Evaluating {model_id} on {lang} ===")
        t0 = time.time()
        
        try:
            accuracy, total_samples = evaluate_model_on_language(
                model_id, language=lang, max_samples=max_samples
            )
            elapsed = time.time() - t0
            
            results[lang] = {
                "accuracy": accuracy,
                "total_samples": total_samples,
                "elapsed_time": elapsed
            }
            
            logger.info(f"Language {lang}: {accuracy:.4f} accuracy ({total_samples} samples, {elapsed:.1f}s)")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {lang}: {e}")
            results[lang] = {
                "accuracy": 0.0,
                "total_samples": 0,
                "elapsed_time": 0,
                "error": str(e)
            }
    
    return results

def main():
    # Configuration
    models = "mgpt"  # Smaller models for testing
    output_dir = "results"
    languages = ['AR_XY', 'BN_BD', 'DE_DE', 'ES_LA', 'FR_FR', 'HI_IN', 'ID_ID', 'IT_IT', 'JA_JP', 'KO_KR', 'PT_BR', 'SW_KE', 'YO_NG', 'ZH_CN']
    max_samples = None
    
    model_ids = parse_models_arg(models)
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate each model
    all_results = {}
    
    for model_id in model_ids:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating model: {model_id}")
        logger.info(f"{'='*50}")
        
        model_results = evaluate_model(model_id, languages, max_samples)
        all_results[model_id] = model_results
        
        # Save individual model results
        model_dir = os.path.join(output_dir, model_id.replace("/", "__"))
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, "results.json"), "w") as f:
            json.dump({
                "model": model_id,
                "languages": model_results,
                "avg_accuracy": np.mean([r["accuracy"] for r in model_results.values() if "error" not in r])
            }, f, indent=2)
        
        # Print summary
        valid_results = {k: v for k, v in model_results.items() if "error" not in v}
        if valid_results:
            avg_acc = np.mean([r["accuracy"] for r in valid_results.values()])
            logger.info(f"Average accuracy for {model_id}: {avg_acc:.4f}")
    
    # Save overall results
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}/")

if __name__ == "__main__":
    main()
