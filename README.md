# build_resoning-ja

## 概要

このリポジトリは、`SkunkworksAI/reasoning-0.01` データセットから高品質な日本語翻訳データを作成するための2つのPythonスクリプトを提供します。

1. **データフィルタリングスクリプト** (`data_filtering.py`)
   - データセットをフィルタリングし、機械学習モデルのトレーニングに適した高品質なデータを抽出します。
2. **データ翻訳スクリプト** (`data_translation.py`)
   - フィルタリングされたデータを英語から日本語に翻訳します。セルフリファイン機構を用いて、翻訳の自然さと読みやすさを向上させます。

---

## 前提条件

### ハードウェア要件

- **GPU環境**：大容量のモデルを使用するため、GPUメモリが**32GB以上**の環境を推奨します。

### ソフトウェア要件

- **Python 3.8** 以上
- 以下のPythonパッケージ（`requirements.txt` 参照）：
  - `datasets`
  - `transformers`
  - `vllm`
  - `spacy`
  - `langdetect`
  - その他必要なパッケージ

---

## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

### 2. 仮想環境の作成（推奨）

```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合は 'venv\Scripts\activate'
```

### 3. 必要なパッケージのインストール

```bash
pip install -r requirements.txt
```

### 4. spaCyの英語モデルのダウンロード

```bash
python -m spacy download en_core_web_sm
```

---

## 使用方法

### 1. データフィルタリング

`data_filtering.py` を実行して、データセットをフィルタリングします。

```bash
python data_filtering.py
```

**スクリプトの機能：**

- `SkunkworksAI/reasoning-0.01` データセットの読み込み
- データのフィルタリング（詳細は後述）
- 重複した `instruction` の削除
- フィルタリング後のデータを `filtered_dataset.json` に保存

### 2. データ翻訳

`data_translation.py` を実行して、フィルタリングされたデータを翻訳します。

```bash
python data_translation.py
```

**スクリプトの機能：**

- `filtered_dataset.json` の読み込み
- 大規模言語モデル `Qwen/Qwen2.5-32B-Instruct` のロード
- 翻訳とセルフリファインの実行
- 翻訳結果を `translated_data.json` に保存

---

## 詳細な説明

### データフィルタリングスクリプト (`data_filtering.py`)

**目的：**

- データセットから品質の低いデータや不適切なデータを除外し、高品質なデータセットを作成します。

**フィルタリング条件：**

1. **`instruction` に "BEGININPUT" が含まれていないこと**
2. **`instruction` にURLが含まれていないこと**
   - `http://`、`https://`、`www.` などの文字列を検出
3. **`output` が数字のみの文でないこと**
4. **`instruction` の長さが10文字以上500文字以下であること**
5. **特殊文字の割合が10%未満であること**
   - 特殊文字：英数字と空白以外の文字
6. **英語の文章であること**
   - `langdetect` を使用して言語を判定
7. **句読点の割合が0.5%以上20%以下であること**
8. **構文解析をパスすること**
   - `spaCy` を使用して、少なくとも1つの名詞と1つの動詞が含まれているか確認

**実行後の結果：**

- フィルタリングされたデータが `filtered_dataset.json` に保存されます。

### データ翻訳スクリプト (`data_translation.py`)

**目的：**

- フィルタリングされた英語のデータを日本語に翻訳し、セルフリファイン機構で翻訳の品質を向上させます。

**主な処理：**

1. **モデルとトークナイザーのロード**

   - `transformers` と `vllm` を使用して、`Qwen/Qwen2.5-32B-Instruct` モデルをロードします。

2. **翻訳プロンプトの作成**

   - 英語のテキストを日本語に翻訳するためのプロンプトを生成します。

3. **セルフリファイン機構の実装**

   - 初回翻訳後、翻訳結果をより自然で読みやすい表現に修正します。

4. **データの翻訳**

   - 各データポイントの `instruction`、`reasoning`、`output` を翻訳します。

5. **結果の保存**

   - 翻訳結果を `translated_data.json` に保存します。

**注意点：**

- **GPUメモリのクリア**

  - 大容量のモデルを使用しているため、`torch.cuda.empty_cache()` を使用して適宜GPUメモリを解放します。

- **進捗の表示**

  - `tqdm` を使用して、翻訳の進捗状況を表示します。

---

## 出力ファイル

- **`filtered_dataset.json`**

  - データフィルタリングスクリプトの出力ファイル。フィルタリングされたデータが保存されています。

- **`translated_data.json`**

  - データ翻訳スクリプトの出力ファイル。日本語に翻訳されたデータが保存されています。

---

## トラブルシューティング

- **メモリエラーが発生する場合**

  - `data_translation.py` 内の `gpu_memory_utilization` の値を調整して、GPUメモリの使用率を下げてください。

- **依存関係のエラーが発生する場合**

  - `requirements.txt` を再度確認し、必要なパッケージが正しくインストールされているか確認してください。

- **モデルのダウンロードに失敗する場合**

  - ネットワーク環境を確認し、モデルのダウンロードが可能な状態であるか確認してください。

---

## ライセンス

このプロジェクトはMITライセンスの下で提供されています。詳細は `LICENSE` ファイルをご覧ください。

---

## 謝辞

- **データセット提供者**

  - `SkunkworksAI/reasoning-0.01` データセットの提供に感謝します。

- **オープンソースコミュニティ**

  - `vllm`、`transformers`、`spaCy`、`langdetect` などのオープンソースライブラリの開発者に感謝します。

- **モデル開発者**

  - `Qwen/Qwen2.5-32B-Instruct` モデルの開発に感謝します。

---

## 更新履歴

- **YYYY/MM/DD**

  - 初回リリース

---

## お問い合わせ

ご質問や問題がある場合は、以下の連絡先までご連絡ください。

- Email: [your.email@example.com](mailto:your.email@example.com)
- GitHub Issues: [https://github.com/yourusername/yourrepository/issues](https://github.com/yourusername/yourrepository/issues)

---

## 参考文献

- [SkunkworksAI/reasoning-0.01 データセット](https://huggingface.co/datasets/SkunkworksAI/reasoning-0.01)
- [vLLM: A High-Throughput and Memory-Efficient Inference and Serving Engine for LLMs](https://github.com/vllm-project/vllm)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [spaCy Documentation](https://spacy.io/usage)
- [langdetect Documentation](https://pypi.org/project/langdetect/)
- [Qwen-7B and Qwen-14B Models](https://huggingface.co/Qwen)

---

## 注意事項

- **モデルのライセンスと利用規約**

  - `Qwen/Qwen2.5-32B-Instruct` モデルを使用する際は、必ずモデルのライセンスや利用規約を確認してください。

- **データの取り扱い**

  - データセットや翻訳結果を公開・共有する際は、著作権やプライバシーに関する法律や規約を遵守してください。

- **商用利用について**

  - 商用目的での利用を検討されている場合は、各ライブラリやモデルのライセンスを確認し、必要に応じて適切な手続きを行ってください。

---

## 開発者向け情報

### コードの構造

- **`data_filtering.py`**

  - データセットのロード、フィルタリング条件の定義、フィルタリング処理、結果の保存。

- **`data_translation.py`**

  - モデルとトークナイザーのロード、翻訳関数の定義、セルフリファイン機構の実装、翻訳処理、結果の保存。

### カスタマイズ方法

- **フィルタリング条件の調整**

  - `data_filtering.py` 内の `is_valid_data` 関数でフィルタリング条件を変更できます。

- **モデルの変更**

  - `data_translation.py` の `model` パラメータを変更することで、別のモデルを使用できます。ただし、モデルの互換性とライセンスに注意してください。

---

## よくある質問

**Q1: 翻訳に時間がかかります。高速化する方法はありますか？**

A1: モデルのサイズを小さいものに変更する、バッチ処理を最適化するなどの方法があります。ただし、モデルサイズを小さくすると翻訳品質が低下する可能性があります。

**Q2: CPU環境でも実行できますか？**

A2: 大容量のモデルを使用しているため、GPU環境が必須です。CPU環境では実行が非常に困難か、実行できない可能性があります。

**Q3: データセットを公開しても問題ありませんか？**

A3: データセットのライセンスや利用規約を確認し、公開が許可されているか確認してください。

---

## 貢献方法

このプロジェクトへの貢献は歓迎します。バグの報告や機能の提案、プルリクエストなど、お気軽にご参加ください。

---

以上が、このリポジトリのREADMEとなります。スクリプトの使用方法や注意点を詳しく説明していますので、ご活用ください。
