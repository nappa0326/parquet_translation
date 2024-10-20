import os
import torch
import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import re

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Set environment variable for debugging

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルとトークナイザーの準備
model_name = "facebook/mbart-large-50-one-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX")

# モデルをGPUに移動
model.to(device)

# ターゲット言語を日本語に設定
tokenizer.src_lang = "en_XX"


# 単一の文を翻訳
def translate_single_sentence(sentence, target_lang="ja_XX"):
    # キャッシュに存在する場合は再利用
    if sentence in translation_cache:
        cached_text = translation_cache[sentence]
        print(f"Cache hit: {cached_text}")
        return cached_text

    # テキストをトークナイズ
    model_inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # 入力をGPUに移動
    model_inputs = model_inputs.to(device)

    # 翻訳を生成
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        num_beams=3,
        early_stopping=True
    )

    # 翻訳結果をデコード
    translated_sentence = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    # キャッシュに保存
    translation_cache[sentence] = translated_sentence
    print(f"Translated: {translated_sentence}")

    # キャッシュを定期的に保存
    save_translation_cache()
    return translated_sentence


# テキストを翻訳
def translate_text(text, target_lang="ja_XX"):
    # テキストのトークン数を確認
    model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=False)
    token_length = model_inputs.input_ids.shape[1]

    # テキストが512トークン以内の場合はそのまま翻訳
    if token_length <= 512:
        translated_text = translate_single_sentence(text, target_lang)
        return translated_text

    # テキストを区切り文字で分割（.,!?）
    sentences = re.split(r'([.,!?])', text)

    translated_sentences = []
    current_segment = ""

    for i in range(0, len(sentences), 2):  # 文と区切り文字をセットで処理
        # 文章が空の場合をスキップ
        if not sentences[i].strip():
            continue

        sentence = sentences[i].strip()  # 余分な空白を削除

        # 現在のセグメントに追加していく
        current_segment += sentence

        # 次の区切り文字があれば、それもセグメントに追加
        if i + 1 < len(sentences):
            current_segment += sentences[i + 1].strip()

        # セグメントのトークン数を確認
        model_inputs = tokenizer(current_segment, return_tensors="pt", padding=True, truncation=False)
        token_length = model_inputs.input_ids.shape[1]

        # 512トークンを超えた場合は前のセグメントまで翻訳
        if token_length > 512:
            translated_sentence = translate_single_sentence(current_segment.strip(), target_lang)
            translated_sentences.append(translated_sentence)
            current_segment = ""  # セグメントをリセット

    # 最後に残ったセグメントがあれば翻訳
    if current_segment:
        translated_sentence = translate_single_sentence(current_segment.strip(), target_lang)
        translated_sentences.append(translated_sentence)

    # 翻訳結果を結合して最終テキストにする
    translated_text = ''.join(translated_sentences).strip()
    return translated_text


# chat列の翻訳
def translate_chat(chat):
    # chat列を複写
    chat_org = chat

    # 元の文字列にUSER,ASSISTANTがなくなるまでループ
    while 'USER:' in chat or 'ASSISTANT:' in chat:
        # chat列の先頭の空白,改行,タブ等を削除
        chat = chat.strip()

        # 先頭がUSERの場合
        if chat.startswith('USER:'):
            # 先頭のUSER:を削除
            chat = chat.replace('USER:', '', 1)

            # 先頭から次のASSISTANT:の直前までを切り出す
            user_part = chat.split('ASSISTANT:')[0].strip()
            chat = chat.replace(user_part, '', 1)

            # user_partが空でない場合
            if user_part:
                # ユーザーの発言を翻訳して元のchat列を置換
                user_translated = translate_text(user_part)
                chat_org = chat_org.replace(user_part, user_translated)
        # 先頭がASSISTANTの場合
        elif chat.startswith('ASSISTANT:'):
            # 先頭のASSISTANT:を削除
            chat = chat.replace('ASSISTANT:', '', 1)

            # 先頭から<|endoftext|>の直前まで切り出す
            assistant_part = chat.split('<|endoftext|>')[0].strip()

            # <|endoftext|>まで含めて削除
            chat = chat.replace(assistant_part, '', 1)
            chat = chat.strip()
            chat = chat.replace('<|endoftext|>', '', 1)

            # <functioncall>の直前までを取り出す(なければ末尾まで)
            assistant_part = assistant_part.split('<functioncall>')[0].strip()

            # assistant_partが空でない場合
            if assistant_part:
                # ASSISTANTの発言を翻訳して元のchat列を置換
                assistant_translated = translate_text(assistant_part)
                chat_org = chat_org.replace(assistant_part, assistant_translated)
        # 先頭がFUNCTION RESPONSE:の場合
        elif chat.startswith('FUNCTION RESPONSE:'):
            # 先頭のFUNCTION RESPONSE:を削除
            chat = chat.replace('FUNCTION RESPONSE:', '', 1)

            # 先頭から次のASSISTANT:の直前までを切り出す
            func_part = chat.split('ASSISTANT:')[0].strip()
            chat = chat.replace(func_part, '', 1)

    return chat_org


# system列の翻訳
def translate_system(system):
    system_part = system.split('SYSTEM:')[1].split(' -')[0].strip()
    system_translated = translate_text(system_part)
    system = system.replace(system_part, system_translated)
    return system


# conversations列の翻訳
def translate_conversations(conversations):
    for entry in conversations:
        if entry['from'] == 'human':
            entry['value'] = translate_text(entry['value'])
        elif entry['from'] == 'gpt' and '<functioncall>' not in entry['value']:
            entry['value'] = translate_text(entry['value'].split('<|endoftext|>')[0]) + '<|endoftext|>'
    return conversations


# 分割処理関数
def process_chunk(df_chunk, start_index):
    # このチャンクがすでに処理済みの場合はスキップ
    if os.path.exists(f'translated_chunk_{start_index}.parquet'):
        return

    # chat列の翻訳
    df_chunk.loc[:, 'chat'] = df_chunk['chat'].apply(translate_chat)

    # system列の翻訳
    df_chunk.loc[:, 'system'] = df_chunk['system'].apply(translate_system)

    # conversations列の翻訳
    df_chunk.loc[:, 'conversations'] = df_chunk['conversations'].apply(translate_conversations)

    # 翻訳結果を保存
    output_file = f'translated_chunk_{start_index}.parquet'
    df_chunk.to_parquet(output_file)
    print(f"Saved: {output_file}")


# キャッシュの定期的な保存関数
def save_translation_cache():
    global cached_text_num

    # 保存されているキャッシュされた翻訳結果数が100行以上増えた場合に保存
    if len(translation_cache) - cached_text_num < 100:
        return

    # キャッシュをデータフレームに変換して保存
    cache_df = pd.DataFrame(list(translation_cache.items()), columns=['text', 'translated_text'])
    cache_df.to_csv(cache_file, index=False)

    # 保存されているキャッシュされた翻訳結果数を更新
    cached_text_num = len(translation_cache)


# メイン処理
def process_data_in_chunks(df, chunk_size):
    total_rows = len(df)
    for start_index in range(0, total_rows, chunk_size):
        end_index = min(start_index + chunk_size, total_rows)
        df_chunk = df.iloc[start_index:end_index]
        process_chunk(df_chunk, start_index)


# Parquetファイル
parquet_file = 'train-00000-of-00002-6f3344faa23e9b0a.parquet'

# Parquetファイルの読み込み
df = pd.read_parquet(parquet_file)
print(f"Total rows: {len(df)}")

# チャンクサイズを1000行に設定
chunk_size = 1000

# 最初の25行に制限しチャンクサイズを10行に設定(テスト用)
#df = df.head(25)
#chunk_size = 10

# 翻訳結果のキャッシュファイル
cache_file = 'translation_cache.csv'

# 翻訳結果のキャッシュファイルの読み込み
if os.path.exists(cache_file):
    translation_cache = pd.read_csv(cache_file).set_index('text')['translated_text'].to_dict()
else:
    translation_cache = {}

# キャッシュされている翻訳結果数を保存
cached_text_num = len(translation_cache)

# データを指定行数ずつ処理
process_data_in_chunks(df, chunk_size)

# 最終的にキャッシュを保存
save_translation_cache()

# 全ての翻訳結果を結合
translated_files = [f'translated_chunk_{i}.parquet' for i in range(0, len(df), 100)]
translated_df = pd.concat([pd.read_parquet(file) for file in translated_files], ignore_index=True)

# 結果を保存(元のファイル名に'_translated'を追加)
output_file = parquet_file.replace('.parquet', '_translated.parquet')
translated_df.to_parquet(output_file)