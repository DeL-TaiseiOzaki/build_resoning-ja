import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
import json
from tqdm import tqdm

# データセットのロード
dataset = load_dataset("フィルター後のデータ格納リポジトリ")

#トークナイザーのロード（必要に応じて）
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

# vllm LLMのロード
llm = LLM(model="Qwen/Qwen2.5-32B-Instruct", gpu_memory_utilization=0.95, max_model_len=16384)

# 翻訳用のプロンプト作成関数
def create_translation_prompt(text):
    return f"英語原文：{text}\n\n 上記の英文を日本語に翻訳してください．\n翻訳した文章以外出力には含め内でください．"

# セルフリファイン機構を使用した翻訳関数
def self_refine_translation(input,text):
    # 初回翻訳
    prompt = create_translation_prompt(text)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    initial_translation = outputs[0].outputs[0].text.strip()
    
    # セルフリファインプロンプト
    refinement_prompt = (
        f"英語原文：{initial_translation}\n\n上記の日本語訳をより自然で読みやすい表現に修正してください．\nなお上記の文章は以下の指示に対する推論，または出力であることを考慮してください．\n\n指示：{input}\n\n出力には翻訳した文章以外含めないでください．"
    )
    
    outputs = llm.generate([refinement_prompt], sampling_params=sampling_params)
    refined_translation = outputs[0].outputs[0].text.strip()
    return refined_translation

# 翻訳結果を保存するリスト
translated_data = []

# データの処理（約2万件）
for i, data_point in enumerate(tqdm(dataset['train'], desc="Translating data")):
    print(f"現在の進捗：{i}ファイル目を翻訳中です．")
    translated_point = {}

    # 'instruction' の翻訳
    english_instruction = data_point.get('instruction', "")
    english_reasoning = data_point.get('reasoning', "")
    english_output = data_point.get('output', "")

    if not english_instruction:
        translated_instruction = ""
    else:
        prompt = create_translation_prompt(english_instruction)
        sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
        outputs = llm.generate([prompt], sampling_params=sampling_params)
        translated_instruction = outputs[0].outputs[0].text.strip()
    translated_point['instruction'] = translated_instruction

    torch.cuda.empty_cache()  # GPUメモリをクリア

    # 'reasoning' の翻訳（セルフリファイン機構を使用）
    if not english_reasoning:
        translated_reasoning = ""
    else:
        translated_reasoning = self_refine_translation(translated_instruction,english_reasoning)
    translated_point['reasoning'] = translated_reasoning

    torch.cuda.empty_cache()  # GPUメモリをクリア

    # 'output' の翻訳（セルフリファイン機構を使用）
    if not english_output:
        translated_output = ""
    else:
        translated_output = self_refine_translation(translated_instruction,english_output)
    translated_point['output'] = translated_output

    # 翻訳結果をリストに追加
    translated_data.append(translated_point)

    torch.cuda.empty_cache()  # GPUメモリをクリア

# 結果をJSONファイルに保存
with open('translated_data.json', 'w', encoding='utf-8') as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=4)

print("翻訳が完了し、'translated_data.json' に保存されました。")
