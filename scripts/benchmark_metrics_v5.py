import json
import re
import torch
import logging
import gc
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Any
import numpy as np
from scipy.spatial.distance import cosine
from collections import Counter
import csv
import html
import os
import argparse

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CUDA Device Logging ---
cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
if cuda_visible_devices:
    logging.info(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible_devices}")
    num_gpus = len(cuda_visible_devices.split(','))
    logging.info(f"Number of GPUs available based on CUDA_VISIBLE_DEVICES: {num_gpus}")
else:
    logging.info("CUDA_VISIBLE_DEVICES is not set. PyTorch will use its default behavior.")
    if torch.cuda.is_available():
        logging.info(f"PyTorch sees {torch.cuda.device_count()} CUDA devices.")
    else:
        logging.info("PyTorch does not see any CUDA devices, will use CPU.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_DTYPE = torch.bfloat16

# LLM Model Paths
GEMMA_MODEL_PATH = "/data/nas/mingzhe/pretrained_models/pretrained/google/gemma-2-2b-it"
QWEN_MODEL_PATH = "/data/nas/mingzhe/pretrained_models/pretrained/Qwen/QwQ-32B"

# --- File paths ---
# These will be set by main() based on args and file iteration
# INPUT_FILE = Path("/data/nas/mingzhe/code/MovieLLM/data/ground_truth/gt_100.json")
# RESULTS_SUBDIR = "0511_v4_full"
# OUTPUT_DIR = Path(__file__).parent / "results" / RESULTS_SUBDIR
# OUTPUT_CSV_FILE = OUTPUT_DIR / "benchmark_results_v4.csv" # Default, will be overridden
# LOG_FILE = OUTPUT_DIR / "benchmark_v4.log" # Default, will be overridden

# --- Global variables and model containers ---
gemma_tokenizer = None
gemma_model = None
qwen_tokenizer = None
qwen_model = None

# Regex patterns for parsing XML/HTML-like script segments
SCENE_TAG_PATTERN = re.compile(r"<scene>(.*?)</scene>", re.DOTALL | re.IGNORECASE)
SCENE_HEADING_XML_PATTERN = re.compile(r"<stage_direction>(.*?)</stage_direction>", re.DOTALL | re.IGNORECASE)
SCENE_DESCRIPTION_XML_PATTERN = re.compile(r"<scene_description>(.*?)</scene_description>", re.DOTALL | re.IGNORECASE)
CHARACTER_XML_PATTERN = re.compile(r"<character>(.*?)</character>", re.DOTALL | re.IGNORECASE)
DIALOGUE_XML_PATTERN = re.compile(r"<dialogue>(.*?)</dialogue>", re.DOTALL | re.IGNORECASE)
PARENTHETICAL_XML_PATTERN = re.compile(r"<parenthetical>(.*?)</parenthetical>", re.DOTALL | re.IGNORECASE)
ACTION_XML_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE) 
TAG_CLEANER = re.compile(r'<.*?>')
HTML_TAG_CLEANER = re.compile(r'<[^>]+>')

# --- Model loading and management ---
def load_model_and_tokenizer(model_type: str):
    """Loads the specified LLM model and tokenizer ('gemma' or 'qwen')."""
    global gemma_tokenizer, gemma_model, qwen_tokenizer, qwen_model

    if model_type == "gemma":
        if gemma_tokenizer is None or gemma_model is None:
            logging.info(f"Loading Gemma model: {GEMMA_MODEL_PATH}...")
            try:
                gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_PATH)
                if gemma_tokenizer.pad_token is None:
                    gemma_tokenizer.pad_token = gemma_tokenizer.eos_token
                gemma_model = AutoModelForCausalLM.from_pretrained(
                    GEMMA_MODEL_PATH,
                    torch_dtype=LLM_DTYPE,
                    device_map="auto"
                )
                gemma_model.eval()
                logging.info(f"Gemma model loaded successfully. Device map: {gemma_model.device_map if hasattr(gemma_model, 'device_map') else gemma_model.hf_device_map}")
            except Exception as e:
                logging.error(f"Error loading Gemma model: {e}", exc_info=True)
                raise
        return gemma_tokenizer, gemma_model
    elif model_type == "qwen":
        if qwen_tokenizer is None or qwen_model is None:
            logging.info(f"Loading Qwen model: {QWEN_MODEL_PATH}...")
            try:
                qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
                if qwen_tokenizer.pad_token_id is None: 
                    qwen_tokenizer.pad_token_id = qwen_tokenizer.eos_token_id
                qwen_model = AutoModelForCausalLM.from_pretrained(
                    QWEN_MODEL_PATH,
                    torch_dtype=LLM_DTYPE, 
                    device_map="auto",
                    trust_remote_code=True
                )
                qwen_model.eval()
                logging.info(f"Qwen model loaded successfully. Device map: {qwen_model.device_map if hasattr(qwen_model, 'device_map') else qwen_model.hf_device_map}")
            except Exception as e:
                logging.error(f"Error loading Qwen model: {e}", exc_info=True)
                raise
        return qwen_tokenizer, qwen_model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def cleanup_models(specific_model: str = "all"):
    """Clears specified model ('gemma', 'qwen') or all, and GPU cache."""
    global gemma_tokenizer, gemma_model, qwen_tokenizer, qwen_model
    
    if specific_model == "gemma" or specific_model == "all":
        if gemma_model is not None: logging.info("Clearing Gemma model and tokenizer.")
        gemma_tokenizer = None
        gemma_model = None
    if specific_model == "qwen" or specific_model == "all":
        if qwen_model is not None: logging.info("Clearing Qwen model and tokenizer.")
        qwen_tokenizer = None
        qwen_model = None
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    logging.debug(f"GPU cache flushed after clearing {specific_model} model(s).")

# --- Helper functions (Text Processing & LLM Interaction) ---

def strip_all_tags(text: str) -> str:
    """Strips all XML/HTML-like tags from text."""
    if not isinstance(text, str): return ""
    return HTML_TAG_CLEANER.sub('', text).strip()

def get_embeddings_gemma(texts: List[str], batch_size: int = 16, max_length=512) -> np.ndarray:
    if not texts: return np.array([])
    valid_texts_with_indices = [(idx, strip_all_tags(str(t))) for idx, t in enumerate(texts) if t and isinstance(t, str) and strip_all_tags(str(t))]
    
    if not valid_texts_with_indices:
        try:
            _, model_ref = load_model_and_tokenizer("gemma")
            hidden_size = model_ref.config.hidden_size
        except Exception:
            hidden_size = 2048 # Fallback Gemma-2B hidden size
        return np.zeros((len(texts), hidden_size), dtype=np.float32)

    original_indices = [item[0] for item in valid_texts_with_indices]
    actual_texts_to_embed = [item[1] for item in valid_texts_with_indices]

    tokenizer, model = load_model_and_tokenizer("gemma")
    
    all_embeddings_batched = []
    model.eval() 
    with torch.no_grad():
        for i in range(0, len(actual_texts_to_embed), batch_size):
            batch_texts_for_embedding = actual_texts_to_embed[i:i+batch_size]
            try:
                inputs = tokenizer(batch_texts_for_embedding, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                mean_pooled_embeddings = (sum_embeddings / sum_mask).cpu().numpy().astype(np.float32)
                all_embeddings_batched.append(mean_pooled_embeddings)
            except Exception as e:
                logging.error(f"Error getting Gemma embeddings for batch: {e}", exc_info=True)
                all_embeddings_batched.append(np.zeros((len(batch_texts_for_embedding), model.config.hidden_size), dtype=np.float32))
            finally:
                if DEVICE == "cuda": torch.cuda.empty_cache()
    
    if not all_embeddings_batched:
         return np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)

    concatenated_embeddings = np.concatenate(all_embeddings_batched, axis=0)
    final_embeddings_array = np.zeros((len(texts), concatenated_embeddings.shape[1]), dtype=np.float32)
    for i_emb, original_idx in enumerate(original_indices):
        if i_emb < concatenated_embeddings.shape[0]:
            final_embeddings_array[original_idx] = concatenated_embeddings[i_emb]
        else:
            logging.error("Mismatch in embedding count and original text indices.")
            
    return final_embeddings_array

# Keep run_llm_extraction from v2 base, rename to run_gemma_extraction to be specific
def run_gemma_extraction(prompt_template: str, items: List[str], max_new_tokens=50, input_max_length=2048) -> List[str]:
    if not items: return []
    tokenizer, model = load_model_and_tokenizer("gemma")
    results = []
    model.eval()
    for item_idx, item_content in enumerate(items):
        if not item_content or not isinstance(item_content, str):
            results.append("")
            continue
        
        max_item_content_length = input_max_length - len(prompt_template.format(text="")) - 100 
        truncated_item_content = item_content[:max_item_content_length]

        prompt = prompt_template.format(text=truncated_item_content)
        messages = [{"role": "user", "content": prompt}]
        try:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt", max_length=input_max_length, truncation=True).to(model.device)
            
            if inputs['input_ids'].shape[1] == input_max_length:
                 logging.debug(f"Gemma input (item {item_idx}) for extraction was truncated to {input_max_length} tokens.")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=False, 
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            input_length = inputs['input_ids'].shape[1]
            response_ids = outputs[0][input_length:] if outputs.shape[1] > input_length else outputs[0]
            response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            results.append(response)
            logging.debug(f"Gemma extraction (item {item_idx}) response: '{response[:100]}...'")
        except Exception as e:
            logging.error(f"Error during Gemma extraction (item {item_idx}): {e}", exc_info=True)
            results.append("")
        finally:
            if DEVICE == "cuda": torch.cuda.empty_cache()
    return results

def run_qwen_generation(prompt: str, max_input_tokens=80000, max_new_tokens=4000) -> str:
    tokenizer, model = load_model_and_tokenizer("qwen")
    messages = [{"role": "user", "content": prompt}]
    try:
        text_input_for_model = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text_input_for_model], return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)

        if model_inputs['input_ids'].shape[1] == max_input_tokens:
             logging.debug(f"Qwen input for generation was truncated to {max_input_tokens} tokens.")
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=max_new_tokens, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        input_length = model_inputs.input_ids.shape[1]
        output_ids_tensor = generated_ids[0] if generated_ids.ndim == 2 else generated_ids
        output_ids = output_ids_tensor[input_length:] if output_ids_tensor.shape[0] > input_length else output_ids_tensor
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        logging.debug(f"Qwen generation response (first 100 chars): '{response[:100]}...'")
        return response
    except Exception as e:
        logging.error(f"Error during Qwen generation: {e}", exc_info=True)
        return ""
    finally:
        if DEVICE == "cuda": torch.cuda.empty_cache()

# --- Parsing (Regex for XML/HTML-like from v2 base - KEEP THIS) ---
def parse_script_segment(script_text: str) -> Dict[str, Any]:
    parsed_data = {"scenes": [], "dialogues_ordered": [], "dialogues_by_character": {}, "actions": []}
    if not script_text or not isinstance(script_text, str):
        logging.warning("parse_script_segment received empty or invalid script_text.")
        return parsed_data

    script_text_cleaned_outer = re.sub(r'^```xml\\s*<script>', '', script_text, flags=re.IGNORECASE | re.DOTALL).strip()
    script_text_cleaned_outer = re.sub(r'</script>\\s*```.*?$', '', script_text_cleaned_outer, flags=re.IGNORECASE | re.DOTALL).strip()
    script_text_cleaned_outer = html.unescape(script_text_cleaned_outer)

    scenes_xml_content = SCENE_TAG_PATTERN.findall(script_text_cleaned_outer)

    if not scenes_xml_content:
        logging.debug("No <scene> tags found. Attempting to parse content outside <scene> tags or entire segment as actions.")
        if script_text_cleaned_outer.strip():
            fallback_action_text = TAG_CLEANER.sub('', script_text_cleaned_outer).strip()
            if fallback_action_text:
                parsed_data["actions"].append(fallback_action_text)
                parsed_data["scenes"].append(fallback_action_text) 
                logging.debug(f"No <scene> tags, used entire segment as fallback action/scene: {fallback_action_text[:100]}...")
        else:
            logging.warning("Script segment is empty or unparsable after initial cleaning and no <scene> tags found.")
        return parsed_data

    temp_actions_list = []
    for scene_idx, single_scene_content_raw in enumerate(scenes_xml_content):
        scene_text_for_pr1_parts = []
        initial_stage_directions = SCENE_HEADING_XML_PATTERN.findall(single_scene_content_raw)
        for sd in initial_stage_directions:
            cleaned_sd = TAG_CLEANER.sub('', sd).strip()
            if cleaned_sd:
                scene_text_for_pr1_parts.append(cleaned_sd)
                temp_actions_list.append(cleaned_sd)

        initial_scene_descriptions = SCENE_DESCRIPTION_XML_PATTERN.findall(single_scene_content_raw)
        for desc in initial_scene_descriptions:
            cleaned_desc = TAG_CLEANER.sub('', desc).strip()
            if cleaned_desc:
                scene_text_for_pr1_parts.append(cleaned_desc)
        
        full_scene_text_for_pr1 = "\\n".join(filter(None, scene_text_for_pr1_parts)).strip()
        if full_scene_text_for_pr1:
            parsed_data["scenes"].append(full_scene_text_for_pr1)

        inner_element_regex = re.compile(
            r"<(?P<tag_name>character|dialogue|parenthetical|action|scene_description|stage_direction)>(.*?)</(?P=tag_name)>", 
            re.DOTALL | re.IGNORECASE
        )
        current_character_in_scene = None
        for match in inner_element_regex.finditer(single_scene_content_raw):
            tag_name = match.group("tag_name").lower()
            content_within_tags = match.group(2)
            cleaned_content = TAG_CLEANER.sub('', content_within_tags).strip()
            if not cleaned_content: continue

            if tag_name == "character":
                current_character_in_scene = cleaned_content.upper()
            elif tag_name == "dialogue":
                if current_character_in_scene:
                    parsed_data["dialogues_ordered"].append({
                        "character": current_character_in_scene,
                        "dialogue": cleaned_content 
                    })
                    parsed_data["dialogues_by_character"].setdefault(current_character_in_scene, []).append(cleaned_content)
                else:
                    logging.debug(f"Dialogue '{cleaned_content}' found without a preceding character in scene {scene_idx}.")
            elif tag_name == "parenthetical":
                if current_character_in_scene and parsed_data["dialogues_ordered"] and \
                   parsed_data["dialogues_ordered"][-1]["character"] == current_character_in_scene:
                    parsed_data["dialogues_ordered"][-1]["dialogue"] += f" ({cleaned_content})"
                    if current_character_in_scene in parsed_data["dialogues_by_character"] and \
                       parsed_data["dialogues_by_character"][current_character_in_scene]:
                        parsed_data["dialogues_by_character"][current_character_in_scene][-1] += f" ({cleaned_content})"
                else:
                    temp_actions_list.append(f"({cleaned_content})")
            elif tag_name in ["action", "scene_description", "stage_direction"]:
                temp_actions_list.append(cleaned_content)

    final_actions_list = []
    seen_actions_set = set()
    for act_text in temp_actions_list:
        if act_text not in seen_actions_set:
            final_actions_list.append(act_text)
            seen_actions_set.add(act_text)
    parsed_data["actions"] = final_actions_list
    
    if not parsed_data["scenes"] and not parsed_data["dialogues_ordered"] and not parsed_data["actions"]:
        logging.warning("Parsing resulted in empty data structures for script_segment after processing scenes.")
    else:
        logging.info(f"Regex Parsed: {len(parsed_data['scenes'])} scenes, {len(parsed_data['dialogues_ordered'])} dialogues, {len(parsed_data['actions'])} actions.")
    return parsed_data

def parse_script_segment_base(text: str) -> Dict[str, Any]:
    """针对base纯文本剧本格式的结构化解析"""
    parsed_data = {"scenes": [], "dialogues_ordered": [], "dialogues_by_character": {}, "actions": []}
    if not text or not isinstance(text, str):
        return parsed_data

    # 按空行或场景切换词分段
    scene_split_pattern = re.compile(r'\n\s*\n|(?:INT\.|EXT\.|FADE IN:|FADE OUT:)', re.IGNORECASE)
    scenes = [s.strip() for s in scene_split_pattern.split(text) if s.strip()]
    parsed_data["scenes"] = scenes

    # 逐段处理
    dialogue_pattern = re.compile(r'^([A-Z][A-Z0-9_ \-]+)[\:\. ]+(.+)$', re.MULTILINE)
    for scene in scenes:
        for match in dialogue_pattern.finditer(scene):
            character = match.group(1).strip()
            dialogue = match.group(2).strip()
            parsed_data["dialogues_ordered"].append({"character": character, "dialogue": dialogue})
            parsed_data["dialogues_by_character"].setdefault(character, []).append(dialogue)
        # 动作：去掉所有台词行，剩下的合并为动作
        scene_no_dialogue = dialogue_pattern.sub('', scene).strip()
        if scene_no_dialogue:
            parsed_data["actions"].append(scene_no_dialogue)
    return parsed_data

# --- Helper for Cosine Similarity (from v4 source, slightly more robust) ---
def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray): return 0.0
    if vec1.shape != vec2.shape: return 0.0 # Ensure shapes match
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    similarity = 1.0 - cosine(vec1, vec2) 
    if np.isnan(similarity):
        return 1.0 if np.allclose(vec1, vec2, atol=1e-6) else 0.0
    return float(max(0.0, min(1.0, similarity)))

# --- Metric calculation functions ---

# DC-1: Adjacent Turn Topic Similarity (from v2 base, uses get_embeddings_gemma)
def calculate_dc1_adjacent_similarity(dialogues_ordered: List[Dict[str, str]]) -> float:
    if not isinstance(dialogues_ordered, list) or len(dialogues_ordered) < 2: return 0.0
    dialogue_texts = [str(d.get("dialogue", "")) for d in dialogues_ordered if isinstance(d, dict) and d.get("dialogue")]
    if len(dialogue_texts) < 2: return 0.0
    try:
        embeddings = get_embeddings_gemma(dialogue_texts)
        if embeddings is None or embeddings.shape[0] < 2: return 0.0
        
        similarities = []
        for i in range(embeddings.shape[0] - 1):
            similarities.append(calculate_cosine_similarity(embeddings[i], embeddings[i+1]))
        
        mean_similarity = np.mean(similarities) if similarities else 0.0
        return float(mean_similarity) if not np.isnan(mean_similarity) else 0.0
    except Exception as e:
        logging.error(f"Error in DC1 (Adjacent Similarity): {e}", exc_info=True); return 0.0

# DC-2: Question-Answer Pair Relevance (from v4 source, uses Gemma)
def calculate_dc2_qa_relevance(dialogues_ordered: List[Dict[str, str]]) -> float:
    if not isinstance(dialogues_ordered, list) or len(dialogues_ordered) < 2: return 0.0
    dialogue_structs = [{"text": str(d.get("dialogue", "")), "is_q": False} 
                        for d in dialogues_ordered if isinstance(d, dict) and d.get("dialogue")]
    if len(dialogue_structs) < 2: return 0.0

    try:
        q_prompt = "Is the following dialogue line a question? Answer with only 'yes' or 'no'. Dialogue: {text}"
        q_texts_to_check = [s["text"][:500] for s in dialogue_structs]
        q_responses = run_gemma_extraction(q_prompt, q_texts_to_check, max_new_tokens=5)
        for i, resp in enumerate(q_responses):
            if resp.strip().lower() == "yes":
                dialogue_structs[i]["is_q"] = True

        qa_pairs = []
        for i in range(len(dialogue_structs) - 1):
            if dialogue_structs[i]["is_q"] and not dialogue_structs[i+1]["is_q"]:
                qa_pairs.append((dialogue_structs[i]["text"], dialogue_structs[i+1]["text"]))

        if not qa_pairs: return 0.0

        question_texts = [pair[0] for pair in qa_pairs]
        answer_texts = [pair[1] for pair in qa_pairs]

        q_embeddings = get_embeddings_gemma(question_texts)
        a_embeddings = get_embeddings_gemma(answer_texts)

        if q_embeddings.shape[0] != a_embeddings.shape[0] or q_embeddings.shape[0] == 0:
            logging.warning("DC2 QA: Mismatch in Q/A embedding shapes or no embeddings.")
            return 0.0

        similarities = [calculate_cosine_similarity(q_embeddings[i], a_embeddings[i]) for i in range(q_embeddings.shape[0])]
        mean_similarity = np.mean(similarities) if similarities else 0.0
        return float(mean_similarity) if not np.isnan(mean_similarity) else 0.0
    except Exception as e:
        logging.error(f"Error in DC2 (QA Relevance): {e}", exc_info=True); return 0.0

# DC-3: Dialogue Topic Concentration (from v2 base, uses get_embeddings_gemma)
def calculate_dc3_topic_concentration(dialogues_ordered: List[Dict[str, str]]) -> float:
    if not isinstance(dialogues_ordered, list) or len(dialogues_ordered) < 2: return 0.0
    dialogue_texts = [str(d.get("dialogue", "")) for d in dialogues_ordered if isinstance(d, dict) and d.get("dialogue")]
    if len(dialogue_texts) < 2: return 0.0
    try:
        embeddings = get_embeddings_gemma(dialogue_texts)
        if embeddings is None: return 0.0
        valid_embeddings = embeddings[np.any(embeddings != 0, axis=1)]
        if valid_embeddings.shape[0] < 2: return 0.0
        
        centroid = np.mean(valid_embeddings, axis=0)
        if np.all(centroid == 0): return 0.0
        
        distances = []
        for emb in valid_embeddings:
            dist = cosine(emb, centroid)
            if np.isnan(dist):
                dist = 0.0 if np.allclose(emb, centroid, atol=1e-6) else 1.0
            distances.append(dist)

        avg_distance = np.mean(distances) if distances else 1.0
        score = float(max(0.0, 1.0 - avg_distance))
        return score if not np.isnan(score) else 0.0
    except Exception as e:
        logging.error(f"Error in DC3 (Topic Concentration): {e}", exc_info=True); return 0.0

# CC-1: Character Emotional Stability / Arc (from v2 base `calculate_cc1_emotional_arc`, uses Gemma)
def calculate_cc1_emotional_stability(dialogues_by_character: Dict[str, List[str]]) -> float:
    if not dialogues_by_character: return 0.0
    overall_scores = []
    emotion_map = {"positive": 1, "neutral": 0, "negative": -1}
    emotion_choices_str = "positive, negative, or neutral"
    prompt_template = (f"Classify the emotion of the following dialogue by character {{char}} "
                       f"as {emotion_choices_str}. Output only the single emotion word. Dialogue: {{text}}")
    
    for char, dialogues in dialogues_by_character.items():
        if not isinstance(dialogues, list): continue
        dialogues_to_process_text = [str(d)[:500] for d in dialogues if d and isinstance(d,str)][:20]
        if len(dialogues_to_process_text) < 2 : continue

        try:
            char_specific_prompt = prompt_template.replace("{char}", str(char))
            extracted_emotion_word_responses = run_gemma_extraction(char_specific_prompt, dialogues_to_process_text, max_new_tokens=10)
            
            numeric_emotions = []
            for resp_idx, raw_emo_line in enumerate(extracted_emotion_word_responses):
                cleaned_emo_word = ""
                processed_line = raw_emo_line.lower().strip()
                if "positive" in processed_line: cleaned_emo_word = "positive"
                elif "negative" in processed_line: cleaned_emo_word = "negative"
                elif "neutral" in processed_line: cleaned_emo_word = "neutral"
                
                if cleaned_emo_word in emotion_map:
                    numeric_emotions.append(emotion_map[cleaned_emo_word])
                elif processed_line: 
                    logging.debug(f"CC1: LLM for char '{char}' returned unmapped emotion: '{raw_emo_line}' for dialogue: '{dialogues_to_process_text[resp_idx][:50]}...'")

            if len(numeric_emotions) < 2: 
                logging.debug(f"CC1: Not enough valid emotions ({len(numeric_emotions)}) for char {char} from {len(dialogues_to_process_text)} dialogues.")
                continue

            changes = np.abs(np.diff(numeric_emotions))
            avg_change = np.mean(changes) 
            max_possible_change = 2.0 
            normalized_avg_change = avg_change / max_possible_change if max_possible_change > 0 else 0.0
            stability_score = 1.0 - normalized_avg_change
            overall_scores.append(max(0.0, min(1.0, stability_score)))
        except Exception as e:
            logging.error(f"Error calculating CC1 for char {char}: {e}", exc_info=True)
            continue
            
    final_score = np.mean(overall_scores) if overall_scores else 0.0
    return float(final_score) if not np.isnan(final_score) else 0.0

# CC-2: Character Linguistic Style Consistency (from v2 base, uses get_embeddings_gemma)
def calculate_cc2_linguistic_consistency(dialogues_by_character: Dict[str, List[str]]) -> float:
    if not dialogues_by_character: return 0.0
    char_consistency_scores = []
    min_dialogues_for_char = 3

    for char, dialogues in dialogues_by_character.items():
        if not isinstance(dialogues, list) or len(dialogues) < min_dialogues_for_char: continue
        valid_dialogues = [str(d) for d in dialogues if d and isinstance(d,str)]
        if len(valid_dialogues) < min_dialogues_for_char: continue

        try:
            embeddings = get_embeddings_gemma(valid_dialogues)
            if embeddings is None: continue
            valid_embeddings = embeddings[np.any(embeddings != 0, axis=1)]
            if valid_embeddings.shape[0] < 2: continue
            
            centroid = np.mean(valid_embeddings, axis=0)
            if np.all(centroid == 0): continue

            distances = []
            for emb in valid_embeddings:
                dist = cosine(emb, centroid)
                if np.isnan(dist): dist = 0.0 if np.allclose(emb, centroid, atol=1e-6) else 1.0
                distances.append(dist)
            
            avg_distance = np.mean(distances) if distances else 1.0
            consistency_score = max(0.0, min(1.0, 1.0 - avg_distance))
            char_consistency_scores.append(consistency_score)
        except Exception as e:
            logging.error(f"Error in CC2 for char {char}: {e}", exc_info=True)
            continue
            
    final_score = np.mean(char_consistency_scores) if char_consistency_scores else 0.0
    return float(final_score) if not np.isnan(final_score) else 0.0

# CC-3: Action-Intention Alignment (from v4 source, adapted for regex parser, uses Gemma)
def calculate_cc3_action_intention_alignment(dialogues_ordered: List[Dict[str, str]], actions: List[str]) -> float:
    if not dialogues_ordered or not actions: return 0.0
    
    intention_dialogues_info = []
    intention_prompt = ("Does this dialogue line by character '{char}' clearly express an intention, plan, "
                        "or strong desire that might lead to a physical action? "
                        "Answer with only 'yes' or 'no'. Dialogue: {text}")

    dialogues_to_scan_for_intent = dialogues_ordered[:30]
    texts_for_intent_check = []
    original_indices_for_intent = []

    for i, dialogue_entry in enumerate(dialogues_to_scan_for_intent):
        char = dialogue_entry.get("character", "Unknown")
        text = str(dialogue_entry.get("dialogue", ""))[:500]
        if not text: continue
        texts_for_intent_check.append(text)
        original_indices_for_intent.append(i)
    
    if not texts_for_intent_check: return 0.0
    
    gemma_intent_responses = run_gemma_extraction(
        intention_prompt.replace("{char}", "CHARACTER"),
        texts_for_intent_check, 
        max_new_tokens=5
    )

    texts_of_identified_intentions = []
    for i, response in enumerate(gemma_intent_responses):
        if response.strip().lower() == "yes":
            original_dialogue_index = original_indices_for_intent[i]
            texts_of_identified_intentions.append(dialogues_to_scan_for_intent[original_dialogue_index]["dialogue"])

    if not texts_of_identified_intentions or not actions:
        return 0.0

    intent_embeddings = get_embeddings_gemma(texts_of_identified_intentions)
    action_embeddings = get_embeddings_gemma(actions)

    if intent_embeddings.shape[0] == 0 or action_embeddings.shape[0] == 0: return 0.0
    
    alignment_scores = []
    for i_emb in intent_embeddings:
        if np.all(i_emb == 0): continue
        max_sim_for_this_intent = 0.0
        for a_emb in action_embeddings:
            if np.all(a_emb == 0): continue
            sim = calculate_cosine_similarity(i_emb, a_emb)
            if sim > max_sim_for_this_intent:
                max_sim_for_this_intent = sim
        if max_sim_for_this_intent > 0:
             alignment_scores.append(max_sim_for_this_intent)
    
    if not alignment_scores: return 0.0
    return float(np.mean(alignment_scores))

# PR-1: Scene Similarity (from v2 base, uses get_embeddings_gemma)
def calculate_pr1_scene_similarity(scenes: List[str]) -> float:
    if not isinstance(scenes, list) or len(scenes) < 2: return 0.0
    valid_scenes = [str(s) for s in scenes if s and isinstance(s, str)]
    if len(valid_scenes) < 2: return 0.0
    try:
        embeddings = get_embeddings_gemma(valid_scenes)
        if embeddings is None or embeddings.shape[0] < 2: return 0.0
        
        similarities = []
        for i in range(embeddings.shape[0] - 1):
            similarities.append(calculate_cosine_similarity(embeddings[i], embeddings[i+1]))
        
        mean_similarity = np.mean(similarities) if similarities else 0.0
        return float(mean_similarity) if not np.isnan(mean_similarity) else 0.0
    except Exception as e:
        logging.error(f"Error in PR1 (Scene Similarity): {e}", exc_info=True); return 0.0

# PR-2: Event Coherence (uses Gemma to extract event chains and calculate embedding similarity)
def calculate_pr2_event_coherence(actions: List[str], scenes: List[str]) -> float:
    content_source_text = ""
    source_type = "none"
    if isinstance(scenes, list) and scenes:
        source_type = "scenes"
        content_source_text = "\n---\n".join([str(s)[:2000] for s in scenes if s and isinstance(s, str)])
    elif isinstance(actions, list) and actions:
        source_type = "actions"
        content_source_text = "\n---\n".join([str(a)[:2000] for a in actions if a and isinstance(a, str)])
    if not content_source_text:
        logging.debug(f"PR2: No content from {source_type} for event extraction.")
        return 0.0
    
    max_full_text_len = 25000
    if len(content_source_text) > max_full_text_len:
        logging.warning(f"PR2: Truncating combined text for event extraction from {len(content_source_text)} to {max_full_text_len} chars.")
        content_source_text = content_source_text[:max_full_text_len]

    event_prompt_template = ("List key plot events in this script segment in chronological order. "
                             "Output ONLY a numbered list of short, concise event descriptions (max 15 words each), one event per line.\n\n"
                             "Script Segment:\n{text}\n\nNumbered Events:")
    try:
        logging.debug(f"PR2: Gemma event extraction from {source_type}...")
        raw_llm_output_list = run_gemma_extraction(event_prompt_template, [content_source_text], max_new_tokens=500)
        if not raw_llm_output_list or not raw_llm_output_list[0]: 
            logging.warning(f"PR2: No event output from Gemma for {source_type} input.")
            return 0.0
        raw_llm_output = raw_llm_output_list[0]
        events = []
        for line in raw_llm_output.split('\n'):
            event_text = re.sub(r"^\s*[\d\.\-\*\•]+\s*", "", line.strip()).strip()
            if event_text and 1 < len(event_text.split()) < 25: events.append(event_text)
        
        if len(events) < 2: 
            logging.debug(f"PR2: Not enough valid events ({len(events)}) parsed from {source_type}. Raw: '{raw_llm_output}'")
            return 0.0
        event_embeddings = get_embeddings_gemma(events)
        if event_embeddings is None or event_embeddings.shape[0] < 2: return 0.0
        similarities = []
        for i in range(event_embeddings.shape[0] - 1):
            vec1, vec2 = event_embeddings[i], event_embeddings[i+1]
            if np.all(vec1 == 0) or np.all(vec2 == 0): similarity = 0.0
            else:
                similarity = 1.0 - cosine(vec1, vec2)
                if np.isnan(similarity): similarity = 1.0 if np.allclose(vec1, vec2, atol=1e-6) else 0.0
            similarities.append(similarity)
        mean_similarity = np.mean(similarities) if similarities else 0.0
        return float(mean_similarity) if not np.isnan(mean_similarity) else 0.0
    except Exception as e:
        logging.error(f"Error in PR2 ({source_type}): {e}", exc_info=True); return 0.0

# --- Helper for Finding Script Files ---
def find_all_json_files(input_root: Path) -> List[Tuple[Path, Path]]:
    """
    Iterates through all .json and .jsonl files in the specified input_root directory.
    Returns a list of tuples: (absolute_file_path, relative_file_path_from_input_root)
    """
    all_files = []
    if not input_root.is_dir():
        logging.error(f"Provided input root is not a directory or does not exist: {input_root}")
        return all_files
        
    file_patterns = ["*.json", "*.jsonl"]
    for pattern in file_patterns:
        for file_path in input_root.rglob(pattern):
            if file_path.is_file():
                # Avoid adding duplicates if a file somehow matches multiple patterns (unlikely for .json/.jsonl)
                # However, rglob can return duplicates if symlinks create loops, though not relevant here.
                # A simple check is to see if we already have this absolute path.
                if not any(existing_abs_path == file_path for existing_abs_path, _ in all_files):
                    relative_path = file_path.relative_to(input_root)
                    all_files.append((file_path, relative_path))
            
    if not all_files:
        logging.warning(f"No .json or .jsonl files found in the input root: {input_root}")
    else:
        logging.info(f"Found {len(all_files)} .json/.jsonl files to process in {input_root}")
    return all_files

# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="MovieLLM Benchmark Metrics Calculator v4")
    parser.add_argument("--input_roots", type=str, required=True,
                        help="The root directory containing .json script files to process.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="The subdirectory name under 'scripts/results/' where outputs will be saved.")
    # Add other arguments like model paths, device, etc., if needed here or keep them as global constants if they rarely change.
    
    args = parser.parse_args()

    input_root_paths = [Path(root) for root in args.input_roots.split(",")]
    base_results_path = Path(__file__).parent / "results" / args.results_dir

    # --- Pre-load models ---
    gemma_preloaded_successfully = False
    qwen_preloaded_successfully = False
    try:
        logging.info("Attempting to pre-load Gemma model...")
        load_model_and_tokenizer("gemma")
        logging.info("Gemma pre-loaded.")
        gemma_preloaded_successfully = True
    except Exception as e:
        logging.warning(f"Could not pre-load Gemma model: {e}. Will attempt to load on demand within metric calculations if needed (or skip).")

    try:
        logging.info("Attempting to pre-load Qwen model...")
        load_model_and_tokenizer("qwen")
        logging.info("Qwen pre-loaded.")
        qwen_preloaded_successfully = True
    except Exception as e:
        logging.warning(f"Could not pre-load Qwen model: {e}. PR2/PR3 calculations might fail or be skipped if Qwen is required.")

    for input_root_path in input_root_paths:
        # Setup a general log for the main script operations, distinct from per-file logs
        main_log_path = base_results_path / f"main_script_log_{input_root_path.name}.log"
        main_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger for main script messages (e.g., file discovery, model pre-loading)
        # This will be temporarily overridden by process_single_file for its specific log
        original_handlers = list(logging.getLogger().handlers)
        main_file_handler = logging.FileHandler(main_log_path, mode='a', encoding='utf-8') # Append if script is run multiple times for different roots
        main_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s'))
        
        # Add stream handler for console output as well for main script events
        main_stream_handler = logging.StreamHandler()
        main_stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        logging.getLogger().handlers = [main_file_handler, main_stream_handler]
        logging.getLogger().setLevel(logging.INFO) # Ensure logging level is set for the root logger

        logging.info(f"Starting benchmark processing for input root: {input_root_path.resolve()}")
        logging.info(f"Results will be stored under: {base_results_path.resolve()}")

        json_files_to_process = find_all_json_files(input_root_path)

        if not json_files_to_process:
            logging.error(f"No .json files found in input root '{input_root_path}'. Exiting.")
            return
            
        total_files = len(json_files_to_process)
        logging.info(f"Found {total_files} JSON files to process in {input_root_path.name}.")

        for idx, (abs_file_path, rel_file_path) in enumerate(json_files_to_process):
            logging.info(f"--- Preparing to process file {idx + 1}/{total_files}: {abs_file_path.name} ---")
            
            # Construct output paths for this specific file
            # Output structure: <base_results_path>/<input_root_name>/<relative_file_path_parent_dirs>/<filename_without_ext>.(csv|log)
            # e.g. results/0512_run/gt_100/subdir/script.csv
            
            # Use input_root_path.name to create a subdirectory for that root, then the relative path
            file_specific_output_dir = base_results_path / input_root_path.name / rel_file_path.parent
            file_specific_output_dir.mkdir(parents=True, exist_ok=True)
            
            output_csv = file_specific_output_dir / rel_file_path.with_suffix(".csv").name
            output_log = file_specific_output_dir / rel_file_path.with_suffix(".log").name

            # The main logger has already logged the start of processing this file.
            # process_single_file will set up its own logger.
            
            try:
                process_single_file(
                    input_file_path=abs_file_path,
                    output_csv_path=output_csv,
                    output_log_path=output_log,
                    gemma_available_globally=gemma_preloaded_successfully,
                    qwen_available_globally=qwen_preloaded_successfully
                )
            except Exception as e:
                # Log this error to the main log, as the file-specific log might not have been set up or might have failed
                logging.error(f"Critical error during process_single_file call for {abs_file_path.name}: {e}", exc_info=True)
            finally:
                # Restore main log handlers for the next file or script completion messages
                logging.getLogger().handlers = [main_file_handler, main_stream_handler]
                logging.info(f"--- Finished processing file {idx + 1}/{total_files}: {abs_file_path.name} ---")


        # --- Main script level model cleanup ---
        if gemma_preloaded_successfully or qwen_preloaded_successfully:
            logging.info(f"All files in {input_root_path.name} processed. Cleaning up pre-loaded models.")
            cleanup_models("all") # "all" or specify ["gemma", "qwen"]
        else:
            logging.info(f"No models were successfully pre-loaded at the start for {input_root_path.name}, no global cleanup needed from main.")
        
        logging.info(f"Benchmarking script finished for input root: {input_root_path.name}.")
        
        # Close main log handlers
        main_file_handler.close()
        main_stream_handler.close()
        logging.getLogger().handlers = original_handlers # Restore to whatever was original before script started

# process_single_file accepts paths and global model load status
def process_single_file(
    input_file_path: Path, 
    output_csv_path: Path, 
    output_log_path: Path,
    gemma_available_globally: bool, 
    qwen_available_globally: bool
):
    # --- Per-file Logging Setup ---
    # Remove any existing handlers from the root logger and set up file-specific logging
    # This ensures that logging for this specific file goes ONLY to its dedicated log file.
    file_logger = logging.getLogger() # Get the root logger
    
    # Backup and clear existing handlers
    # We don't want main script logs (like "processing file x of y") in the individual file logs.
    original_handlers_process_file = list(file_logger.handlers)
    for handler in file_logger.handlers[:]: # Iterate over a copy
        file_logger.removeHandler(handler)
        handler.close() # Important to close handlers before removing

    file_log_handler = logging.FileHandler(output_log_path, mode='w', encoding='utf-8') # 'w' to overwrite for each file
    file_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s'))
    
    # Optional: Add a stream handler if you want console output *during* single file processing
    # console_handler_process_file = logging.StreamHandler()
    # console_handler_process_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # file_logger.addHandler(console_handler_process_file)
    
    file_logger.addHandler(file_log_handler)
    file_logger.setLevel(logging.INFO) # Or any level appropriate for file processing

    logging.info(f"--- Starting processing for file: {input_file_path.resolve()} ---")
    logging.info(f"Output CSV will be saved to: {output_csv_path.resolve()}")
    logging.info(f"Log for this file will be saved to: {output_log_path.resolve()}")
    
    # Ensure output directory for CSV/Log exists (it should have been created by main, but double check)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory '{output_csv_path.parent.resolve()}' ensured.")

    results_data = []
    line_count = 0 
    processed_successfully_count = 0
    json_key_error_count = 0
    regex_parsing_error_count = 0 

    if not input_file_path.is_file():
        logging.error(f"Input file not found: {input_file_path.resolve()} (This should not happen if discovered by main)")
        for handler in file_logger.handlers[:]:
            file_logger.removeHandler(handler)
            handler.close()
        return

    logging.info(f"Processing file: {input_file_path.resolve()}")
    
    # ========== Automatically determine file format, flexibly supporting two modes ==========
    all_entries = []
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        first_char = infile.read(1)
        infile.seek(0)
        if first_char == '[':
            # json array mode (like gemini)
            data = json.load(infile)
            for idx, item in enumerate(data):
                movie_name = item.get('movie_name', f'Unknown_L{idx+1}')
                imdb_id = item.get('imdb_id', 'N/A')
                resp = item.get('response', '')
                # Remove ```xml and <script> etc. wrappers
                if isinstance(resp, str):
                    text = resp.strip()
                    if text.startswith('```xml'):
                        text = text[len('```xml'):].strip()
                    if text.startswith('<script>'):
                        text = text[len('<script>'):].strip()
                    if text.endswith('```'):
                        text = text[:-3].strip()
                    if text.endswith('</script>'):
                        text = text[:-len('</script>')].strip()
                    script_segment = text
                else:
                    script_segment = ''
                all_entries.append({
                    'movie_name': movie_name,
                    'imdb_id': imdb_id,
                    'script_segment': script_segment
                })
        else:
            # jsonl mode (like gt_100)
            for idx, line in enumerate(infile):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    movie_name = item.get('movie_name', f'Unknown_L{idx+1}')
                    imdb_id = item.get('imdb_id', 'N/A')
                    script_segment = item.get('script_segment', '')
                    all_entries.append({
                        'movie_name': movie_name,
                        'imdb_id': imdb_id,
                        'script_segment': script_segment
                    })
                except Exception as e:
                    logging.error(f"Error parsing JSONL line {idx+1}: {e}")
                    all_entries.append({
                        'movie_name': f'Error_Line_{idx+1}',
                        'imdb_id': 'N/A',
                        'script_segment': ''
                    })
    # ========== End ==========

    line_count = len(all_entries)
    for line_idx, entry in enumerate(all_entries):
        current_line_metrics = {
            "movie_name": entry.get("movie_name", f"Error_Entry_{line_idx+1}"),
            "imdb_id": entry.get("imdb_id", "N/A"),
            "line_number": line_idx+1,
            "regex_parsing_successful": 0,
            "dc1_adjacent_similarity": 0.0, "dc2_qa_relevance": 0.0, "dc3_topic_concentration": 0.0,
            "cc1_emotional_stability": 0.0, "cc2_linguistic_consistency": 0.0, "cc3_action_intention_alignment": 0.0,
            "pr1_scene_similarity": 0.0, "pr2_event_coherence": 0.0
        }
        original_script_segment_text = entry.get("script_segment", "")
        try:
            if not original_script_segment_text or not isinstance(original_script_segment_text, str):
                logging.warning(f"Entry {line_idx+1}: Empty/invalid 'script_segment' for '{current_line_metrics['movie_name']}'.")
                json_key_error_count += 1
                results_data.append(current_line_metrics)
                continue
            logging.info(f"Entry {line_idx+1}: Analyzing segment for '{current_line_metrics['movie_name']}' ({current_line_metrics['imdb_id']})")
            parsed_script_content = parse_script_segment(original_script_segment_text)
            # If all fields are empty, fallback to base parsing
            if not parsed_script_content["scenes"] and not parsed_script_content["dialogues_ordered"] and not parsed_script_content["dialogues_by_character"]:
                parsed_script_content = parse_script_segment_base(original_script_segment_text)
                logging.info(f"Entry {line_idx+1}: Fallback to base parsing for '{current_line_metrics['movie_name']}'.")
            if not parsed_script_content or not any(parsed_script_content.values()):
                logging.warning(f"Entry {line_idx+1}: Regex parsing yielded insufficient data for '{current_line_metrics['movie_name']}'.")
                regex_parsing_error_count += 1
                results_data.append(current_line_metrics)
                continue
            else:
                current_line_metrics["regex_parsing_successful"] = 1
            dialogues_ordered = parsed_script_content.get("dialogues_ordered", [])
            dialogues_by_char = parsed_script_content.get("dialogues_by_character", {})
            actions_list = parsed_script_content.get("actions", [])
            scenes_list = parsed_script_content.get("scenes", [])
            current_line_metrics["dc1_adjacent_similarity"] = calculate_dc1_adjacent_similarity(dialogues_ordered)
            current_line_metrics["dc2_qa_relevance"] = calculate_dc2_qa_relevance(dialogues_ordered)
            current_line_metrics["dc3_topic_concentration"] = calculate_dc3_topic_concentration(dialogues_ordered)
            current_line_metrics["cc1_emotional_stability"] = calculate_cc1_emotional_stability(dialogues_by_char)
            current_line_metrics["cc2_linguistic_consistency"] = calculate_cc2_linguistic_consistency(dialogues_by_char)
            current_line_metrics["cc3_action_intention_alignment"] = calculate_cc3_action_intention_alignment(dialogues_ordered, actions_list)
            current_line_metrics["pr1_scene_similarity"] = calculate_pr1_scene_similarity(scenes_list)
            if qwen_available_globally:
                try:
                    current_line_metrics["pr2_event_coherence"] = calculate_pr2_event_coherence(actions_list, scenes_list)
                except Exception as qwen_metric_err:
                    logging.error(f"Entry {line_idx+1}: Error during Qwen-based PR2 calculation: {qwen_metric_err}", exc_info=True)
            else:
                logging.warning(f"Entry {line_idx+1}: Skipping PR2 calculation for '{current_line_metrics['movie_name']}' as Qwen model was not successfully pre-loaded.")
            results_data.append(current_line_metrics)
            processed_successfully_count += 1
            logging.info(f"  Entry {line_idx+1} Metrics for '{current_line_metrics['movie_name']}': "
                         f"DC1={current_line_metrics['dc1_adjacent_similarity']:.3f}, DC2={current_line_metrics['dc2_qa_relevance']:.3f}, DC3={current_line_metrics['dc3_topic_concentration']:.3f}, "
                         f"CC1={current_line_metrics['cc1_emotional_stability']:.3f}, CC2={current_line_metrics['cc2_linguistic_consistency']:.3f}, CC3={current_line_metrics['cc3_action_intention_alignment']:.3f}, "
                         f"PR1={current_line_metrics['pr1_scene_similarity']:.3f}, PR2={current_line_metrics['pr2_event_coherence']:.3f}")
        except json.JSONDecodeError as e_json_load: # Handles json.loads for JSONL lines
            logging.error(f"Entry {line_idx+1} (JSONL line): JSONDecodeError: {e_json_load}. Raw line: '{original_script_segment_text[:100]}...'")
            json_key_error_count += 1; results_data.append(current_line_metrics)
        except Exception as e_general_proc: # Handles other errors during an entry's processing
            logging.error(f"Entry {line_idx+1}: Unexpected error processing entry for '{current_line_metrics.get('movie_name', 'UNKNOWN_ENTRY')}': {e_general_proc}", exc_info=True)
            results_data.append(current_line_metrics) # Record with defaults/partial data
        finally:
            if DEVICE == "cuda": torch.cuda.empty_cache()
            gc.collect()
        # End of for loop over all_entries
        logging.info(f"--- Processing Complete for file: {input_file_path.name} ---")
        # line_count here will be the number of the last entry processed or attempted
        logging.info(f"Total entries yielded by parser for processing: {len(all_entries if all_entries else [])}")
        logging.info(f"Entries for which metric calculation was fully or partially attempted: {line_count}")
        logging.info(f"Entries processed successfully for all metrics: {processed_successfully_count}")
        logging.info(f"JSON parsing errors or missing/invalid 'script_segment' key: {json_key_error_count}")
        logging.info(f"Regex parsing failures (no useful data extracted from script_segment): {regex_parsing_error_count}")

        # --- Crucial: Restore original logging handlers that process_single_file started with ---
        # This means restoring the main script's log handlers.
        for handler in file_logger.handlers[:]: 
            file_logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers_process_file: 
             if handler not in file_logger.handlers: 
                file_logger.addHandler(handler)

    if results_data:
        csv_header = [
            "movie_name", "imdb_id", "line_number", "regex_parsing_successful",
            "dc1_adjacent_similarity", "dc2_qa_relevance", "dc3_topic_concentration",
            "cc1_emotional_stability", "cc2_linguistic_consistency", "cc3_action_intention_alignment",
            "pr1_scene_similarity", "pr2_event_coherence"
        ]
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header, restval='0.0', extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results_data)
            logging.info(f"Benchmark results for '{input_file_path.name}' saved to: {output_csv_path.resolve()}")
        except IOError as e:
            logging.error(f"Error writing results to CSV file '{output_csv_path.resolve()}': {e}")
        except Exception as e:
            logging.error(f"Unexpected error writing CSV for '{input_file_path.name}': {e}", exc_info=True)
    else:
        logging.warning(f"No results data was generated to save to CSV for file '{input_file_path.name}'.")

if __name__ == "__main__":
    # Basic logging setup before parsing args, in case of early errors in main()
    # This will be reconfigured by main() once args are parsed.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()