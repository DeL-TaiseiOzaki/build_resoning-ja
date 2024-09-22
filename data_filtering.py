from datasets import load_dataset
import random
import json
import re
from langdetect import detect, LangDetectException
import spacy

# spaCyの英語モデルの読み込み
nlp = spacy.load("en_core_web_sm")

# データセットのロード
dataset = load_dataset("SkunkworksAI/reasoning-0.01")['train']

# URLパターンを定義 (http, https, wwwなどを含む)
url_pattern = re.compile(r'http[s]?://|www\.')
# 数字だけの文を検出するパターン
number_only_pattern = re.compile(r'^\d+$')
# 特殊文字をカウントするパターン
special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')

# 構文解析による構造チェック
def passes_syntactic_analysis(instruction):
    doc = nlp(instruction)
    
    # 文の中に少なくとも1つの動詞と1つの名詞が存在することを確認
    has_verb = any(token.pos_ == "VERB" for token in doc)
    has_noun = any(token.pos_ == "NOUN" for token in doc)
    
    return has_verb and has_noun

# フィルタ関数：各フィルタ条件を満たさないデータを除外
def is_valid_data(data_point):
    instruction = data_point['instruction']
    output = data_point['reasoning']

    # BEGININPUT を含むかチェック
    if "BEGININPUT" in instruction:
        return False
    
    # URLを含むかチェック
    if url_pattern.search(instruction):
        return False
    
    # 数字だけの文を含むかチェック
    if number_only_pattern.match(output):
        return False
    
    # 文の長さをチェック（短すぎる・長すぎる文を除外）
    if len(instruction) < 10 or len(instruction) > 500:
        return False
    
    # 特殊文字の割合をチェック（10%以上の特殊文字を持つ文を除外）
    total_chars = len(instruction)
    special_chars = len(special_char_pattern.findall(instruction))
    if special_chars / total_chars > 0.1:
        return False
    
    # 英語かどうかをチェック
    try:
        lang = detect(instruction)
        if lang != 'en':
            return False
    except LangDetectException:
        # 言語検出に失敗した場合も除外
        return False
    
    # 句読点の割合チェック（極端に多すぎる・少なすぎる場合を除外）
    punctuation_count = sum([1 for char in instruction if char in ".,!?;:"])
    if punctuation_count / total_chars < 0.005 or punctuation_count / total_chars > 0.2:
        return False
    
    # 構文解析をパスするか確認（動詞と名詞が少なくとも1つずつあるか）
    if not passes_syntactic_analysis(instruction):
        return False
    
    return True

# フィルタ処理と重複除去
filtered_data = [data_point for data_point in dataset if is_valid_data(data_point)]
unique_instructions = list({data['instruction']: data for data in filtered_data}.values())

# フィルタ後のデータの一部を表示
for i in unique_instructions[:5]:
    print(i['instruction'])

# フィルタ後のデータ数を表示
print(f"フィルタ後のデータ数: {len(unique_instructions)}")

# データが1万件未満の場合、全てを使用
if len(unique_instructions) < 5000:
    selected_data = unique_instructions
    print("フィルタ後のデータが1万件未満のため、全てのデータを使用します。")
else:
    # データをシャッフルして1万件を抽出
    random.seed(42)  # 再現性のためのシード設定
    random.shuffle(unique_instructions)
    selected_data = unique_instructions[:10000]
    print("フィルタ後のデータから1万件を抽出しました。")

# 抽出したデータをJSONファイルに保存
with open('filtered_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(selected_data, f, ensure_ascii=False, indent=4)

print("抽出したデータを 'filtered_dataset.json' に保存しました。")
