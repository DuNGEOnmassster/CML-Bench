from functools import partial
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import logging
import re
import time
import os
import json

# For safety reason, we do not upload the utils/llm_utils.py file to the repo
from utils.llm_utils import call_gpt_api, call_gemini_api, call_siliconflow_api, call_nvidia_api, call_intern_api

prompt_start_llm = r"""
You are an expert AI scriptwriter. Your task is to generate a detailed and professional movie script segment based on the provided Movie Title and Movie Summary. The script should be formatted in an XML-like structure, mirroring professional screenplay standards.

**Input:**
Movie Title: {movie_name}
Movie Summary: {summary}

"""

prompt_end_llm = r"""
Please generate the script segment based on the Movie Title and Summary provided above, adhering strictly to this XML-like format and content guidelines. Ensure the output is a single block of text starting with `<script>` and ending with `</script>`.
"""

prompt_rag_instructions_content = r"""
In your generated movie script segment, strictly adhere to the following structured format and guidelines to facilitate automated parsing and evaluation:

**Overall Structure:**
- The entire script segment must be enclosed within `<script> ... </script>` tags.
- Divide the script into multiple `<scene> ... </scene>` blocks.

**Scene Elements (within each `<scene>`):**
- **Stage Direction:** Each scene must start with a `<stage_direction>` tag, specifying the location, environment, and time (e.g., `<stage_direction>INT. LOCATION - DAY</stage_direction>`).
- **Scene Description:** Detailed visual and narrative descriptions should be encapsulated within `<scene_description>` tags, highlighting:
    - Visual setting and atmosphere.
    - Character actions, movements, and key non-verbal expressions.
    - Significant audio or visual cues.
    - The chronological progression of events within the scene.
- **Character Dialogue:**
    - Clearly introduce each speaking character with a `<character>` tag (character names should be uppercase).
    - Follow immediately with their speech enclosed in `<dialogue>` tags.
    - Use optional `<parenthetical>` tags for brief, specific acting instructions or delivery notes directly related to dialogue (e.g., `<parenthetical>(whispering)</parenthetical>`).
- **Action (optional):** Significant non-dialogue actions important for plot advancement or character development may be included within `<action>` tags.

**Content and Style Guidelines:**
- Ensure your script logically develops key events and character interactions from the provided Movie Summary.
- Dialogue should be natural, engaging, and meaningful for character and plot progression.
- Scene descriptions must be vivid, concise, visually descriptive, and serve to enhance visualization.
- Maintain consistent character voice, motivations, and behaviors.
- Provide clear and coherent transitions among descriptions, dialogues, and actions, creating a seamless script segment that feels integral to a larger professional screenplay.
- Adhere strictly to professional screenplay formatting and tone.

**Example Snippet of Expected Format:**
```xml
<script>
  <scene>
    <stage_direction>INT. COFFEE SHOP - DAY</stage_direction>
    <scene_description>The coffee shop is bustling. ANNA (30s), dressed in a sharp business suit, sips her latte, looking impatient. MARK (30s), disheveled and out of breath, rushes in.</scene_description>
    <character>MARK</character>
    <dialogue>Sorry I'm late! The traffic was insane.</dialogue>
    <character>ANNA</character>
    <parenthetical>(glancing at her watch)</parenthetical>
    <dialogue>Insane or you overslept?</dialogue>
    <scene_description>Mark pulls out a chair and slumps into it, running a hand through his messy hair. He looks exhausted.</scene_description>
    <character>MARK</character>
    <dialogue>Okay, a bit of both. But mostly insane traffic.</dialogue>
  </scene>
  <scene>
    <stage_direction>EXT. PARK - LATER</stage_direction>
    <scene_description>Sunlight dapples through the trees. Anna and Mark walk along a paved path, a little more relaxed now.</scene_description>
    <character>ANNA</character>
    <dialogue>So, about the Henderson account... We need a new strategy.</dialogue>
  </scene>
</script>
```
"""

prompt_in_paper_base = prompt_start_llm + prompt_end_llm
prompt_in_paper_instruction = prompt_start_llm + prompt_rag_instructions_content + prompt_end_llm


def get_llm_response(row, model_name='gpt-4o', prompt_version='0506', task_name='gt_100'):

    movie_name = row.get('movie_name', '')
    imdb_id = row.get('imdb_id', '')
    summary = row.get('summary', '')

    if prompt_version == 'base':
        prompt_template = prompt_in_paper_base
    elif prompt_version == 'instruction':
        prompt_template = prompt_in_paper_instruction
    else:
        raise ValueError(f"Invalid prompt version: {prompt_version}")
    # Check if temp folder exists
    response = None
    temp_dir = f"data/temp/{task_name}/{prompt_version}/{model_name.replace('/', '_')}"
    if os.path.exists(temp_dir):
        # Check if temp file exists
        if os.path.exists(f"{temp_dir}/{imdb_id}.txt"):
            with open(f"{temp_dir}/{imdb_id}.txt", "r", encoding="utf-8") as f:
                response = f.read().strip()
                if response is not None and response != '':
                    return {
                        'movie_name': movie_name,
                        'imdb_id': imdb_id,
                        'response': response
                    }
                else:
                    print(f"Response is empty for {imdb_id}")
    else:
        os.makedirs(temp_dir)

    prompt = prompt_template.format_map({
        'movie_name': movie_name,
        'summary': summary
    })


    print(f"{'-'*50}\nPROMPT - {prompt_version}:\n{prompt}\n{'-'*50}")
    max_retry = 10
    while True:
        try:
            # SiliconFlow
            if model_name in [
                "Pro/deepseek-ai/DeepSeek-V3",
                "Pro/deepseek-ai/DeepSeek-R1",
                "Qwen/QwQ-32B",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "Qwen/Qwen2.5-72B-Instruct",
                "Qwen/Qwen2.5-32B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "internlm/internlm2_5-20b-chat"]:
                response = call_siliconflow_api(prompt, model_name=model_name)
            # Google
            elif 'gemini' in model_name.lower():
                response = call_gemini_api(prompt, model_name=model_name)
            # OpenAI
            elif 'gpt' in model_name.lower() or 'o1' in model_name.lower() or 'o3' in model_name.lower() or 'o4' in model_name.lower():
                response, _ = call_gpt_api(prompt, model_name=model_name, max_length=16384, temperature=0.0)
            # NVIDIA
            elif model_name in [
                "meta/llama-4-maverick-17b-128e-instruct",
                "meta/llama-4-scout-17b-16e-instruct",
                "microsoft/phi-4-mini-instruct"]:
                response = call_nvidia_api(prompt, model_name=model_name)
            # ShanghaiAILab
            elif model_name in ["internlm3-8b-instruct"]:
                response = call_intern_api(prompt, model_name=model_name)
            else:
                raise ValueError(f"Invalid model name: {model_name}")

            assert response is not None and response != '', f"Response is empty for {imdb_id}"

            break
        except Exception as e:
            logging.error(e)
            response = f"ERROR:{e}"
            time.sleep(60)
            max_retry -= 1
            if max_retry <= 0:
                print(f"Failed to get response from {model_name} after {max_retry} retries")
                break

        print(f"{'-'*50}\nRESPONSE - {prompt_version}:\n{response}\n{'-'*50}")

        # save response to file
        if not response.startswith('ERROR'):
            with open(f"{temp_dir}/{imdb_id}.txt", "w") as f:
                f.write(response)

    return {
        'movie_name': movie_name,
        'imdb_id': imdb_id,
        'response': response
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name, e.g. 'microsoft/phi-4-multimodal-instruct' or 'meta/llama-3.2-11b-vision-instruct'")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes")
    parser.add_argument("--prompt_version", type=str, default='0430', help="Prompt version")
    parser.add_argument("--task_name", type=str, default='gt_100', help="Task name")
    args = parser.parse_args()

    model_name = args.model_name
    num_processes = args.num_processes
    prompt_version = args.prompt_version
    task_name = args.task_name

    # read jsonl file
    test_data = pd.read_json(f"../data/ground_truth/{task_name}.json", lines=True).to_dict('records')

    print(f"Load data: Rows = {len(test_data)}, Model = {model_name}, Prompt = {prompt_version}")

    # parallel processing
    process_row_with_prompt_version = \
        partial(get_llm_response, prompt_version=prompt_version, model_name=model_name, task_name=task_name)
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.map(process_row_with_prompt_version, test_data),
            total=len(test_data),
            desc="Processing"
        ))

    # save to JSON file
    output_dir = f"data/result/{task_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f"{output_dir}/{model_name.replace('/', '_')}_{prompt_version}.json"
    json.dump(results, open(output_file, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    # results can be written to file or processed directly here
    print(f"Done for Model = {model_name}, Prompt = {prompt_version}, processed {len(results)} rows.")

if __name__ == '__main__':
    main()
