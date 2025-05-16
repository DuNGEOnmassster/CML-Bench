

models=(
    # GEMINI_API:
    "gemini-2.0-flash-001"
    "gemini-2.0-flash-lite-001"
    # "gemini-2.0-flash-thinking-exp-01-21"
    # # "gemini-2.0-pro-exp-02-05" # Very slow
    # # "gemini-2.5-pro-preview-03-25"   No API

    # # OPENAI_API:
    # 'gpt-4o-2024-11-20'
    # 'gpt-4o-mini-2024-07-18'
    'o3-mini'
    'o4-mini-2025-04-16'


    # # # NVIDIA_API:
    "meta/llama-4-maverick-17b-128e-instruct"
    "meta/llama-4-scout-17b-16e-instruct"
    # "microsoft/phi-4-mini-instruct"

    # # SILICONFLOW_API:
    # "Pro/deepseek-ai/DeepSeek-V3"
    # "Pro/deepseek-ai/DeepSeek-R1"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    # "internlm/internlm2_5-20b-chat"
    # # Qwen
    # "Qwen/QwQ-32B"
    # "Qwen/Qwen2.5-72B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"

    # # INTERN_API:
    "internlm3-8b-instruct"
    )


# - **Long-context**
#     - Mamba-2.8B
#     - FILM-7B
#     - LongWriter-llama3.1-8B


task_name='gt_100'
num_processes=16


# for prompt_version in 'rag'
# for prompt_version in 'rag_instruction'
for prompt_version in 'base2'
do
    for model_name in ${models[@]}
    do
        echo "Processing $model_name" "$prompt_version"
        python inference/request_LLM.py \
            --model_name $model_name \
            --num_processes $num_processes \
            --prompt_version $prompt_version \
            --task_name $task_name \
            > logs/${model_name//\//_}_${prompt_version}.log # 2>&1
    done
done


# After completing all work, git commit and push
git add .
git commit -m "update"
git push
