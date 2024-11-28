import os
import torch
import pandas as pd
from PIL._imaging import display
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

    # 固有名詞をプレースホルダーに置き換える（例: 映画タイトルや曲名）
    placeholders = {}
    # ここで固有名詞を特定し、プレースホルダーに置き換える処理を追加
    # 例: sentence = sentence.replace("MovieTitle", "<MOVIE_TITLE>")
    # placeholders["<MOVIE_TITLE>"] = "MovieTitle"

    # テキストをトークナイズ
    model_inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # 入力をGPUに移動
    model_inputs = model_inputs.to(device)

    # 翻訳を生成
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        num_beams=3,  # ビーム数を増やす
        early_stopping=True,
        do_sample=True,  # temperatureを使用するためTrue
        temperature=0.1  # 温度を調整
    )

    # 翻訳結果をデコード
    translated_sentence = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    # プレースホルダーを元の固有名詞に戻す
    for placeholder, original in placeholders.items():
        translated_sentence = translated_sentence.replace(placeholder, original)

    # キャッシュに保存
    translation_cache[sentence] = translated_sentence
    print(f"Translated: {translated_sentence}")

    # キャッシュを定期的に保存
    save_translation_cache()
    return translated_sentence


# Noneの要素を前後の要素と結合する
def combine_none_elements(lst):
    # 結果を格納するリスト
    result = []
    i = 0

    while i < len(lst):
        if lst[i] is None:
            # 前後の要素が存在するか確認して結合
            if i > 0 and i < len(lst) - 1:
                combined = lst[i - 1] + '.' + lst[i + 1]
                result.pop()  # 前の要素を削除
                result.append(combined)  # 結合した要素を追加
                i += 2  # 後ろの要素も処理済みなので2つ進める
            else:
                i += 1  # 範囲外の場合は次へ進む
        else:
            # Noneでない場合はそのままリストに追加
            result.append(lst[i])
            i += 1

    return result


# テキストを区切り文字で分割
def split_text_with_separators(text):
    # 正規表現パターンを定義し、キャプチャグループを使用
    pattern = r'([.,:;!?](?=\s|$)|\n)'
    # テキストを分割し、セパレータも含める
    segments = re.split(pattern, text)
    # 空のセグメントを除去
    segments = [segment for segment in segments if segment.strip() or segment in {'.', ',', ':', ';', '!', '?', '\n'}]
    return segments


# テキストを翻訳
def translate_text(text, target_lang="ja_XX"):
    ## テキストのトークン数を確認
    #model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=False)
    #token_length = model_inputs.input_ids.shape[1]

    ## テキストが512トークン以内の場合はそのまま翻訳
    #if token_length <= 512:
    #    translated_text = translate_single_sentence(text, target_lang)
    #    return translated_text

    # テキストを区切り文字で分割（.,:;!?[改行]もしくは[空白]もしくは行末）
    sentences = split_text_with_separators(text)

    # Noneの要素を検索して、Noneの前の要素と後の要素を'.'で結合し1つの要素にして結合した要素は削除する。
    # 例: ['Hello', None, 'world', None, 'Goodbye', None] -> ['Hello.world.Goodbye.']
    sentences = combine_none_elements(sentences)

    translated_sentences = []
    current_segment = ""

    # 索引を初期化
    i = 0

    # 文と区切り文字をセットで処理
    while True:
        # 最後の文章を処理した場合は終了
        if i >= len(sentences):
            break

        # 文章を取得
        sentence = sentences[i]

        # 索引を進める
        i += 1

        # 文章が改行でない場合
        # もしくは現在のセグメントが空かつ文章が改行の場合
        if sentence != '\n' or (not current_segment and sentence == '\n'):
            # 余分な空白を削除
            sentence = sentence.strip()

        # 空白のみの場合は次の要素へ
        if not sentence:
            continue

        # 現在のセグメントに追加
        current_segment += sentence

        # 次の文が存在する場合
        # かつ区切り文字の場合は区切り文字を追加
        if i < len(sentences) and sentences[i] in ['.', ',', ':', ';', '!', '?']:
            # 区切り文字を追加
            current_segment += sentences[i].strip()

            # 索引を進める
            i += 1

        # 現在のセグメントが半角数字と半角記号だけで構成されている場合はスキップ
        if re.match(r'^[\d\W]+$', current_segment):
            continue

        # セグメントのトークン数を確認
        model_inputs = tokenizer(current_segment, return_tensors="pt", padding=True, truncation=False)
        token_length = model_inputs.input_ids.shape[1]

        # 512トークンを超えた場合
        # もしくは現在のセグメントが3文字以上かつ末尾が区切り文字(.!?[改行])の場合
        # もしくは現在のセグメントが4文字以上かつ末尾2文字が':'もしくは';'と改行の場合も翻訳
        if (token_length > 512 or
                (3 <= len(current_segment) and current_segment[-1] in ['.', '!', '?', '\n']) or
                (4 <= len(current_segment) and current_segment[-2] in [':', ';'] and current_segment[-1] == '\n')):
            # 改行追加フラグを初期化
            add_newline = False

            # 現在のセグメントの末尾が改行の場合
            if current_segment[-1] == '\n':
                # 改行を削除
                current_segment = current_segment[:-1]

                # 改行追加フラグをTrueに設定
                add_newline = True
            # 次の文が改行の場合
            elif i < len(sentences) and sentences[i] == '\n':
                # 索引を進める
                i += 1

                # 改行追加フラグをTrueに設定
                add_newline = True

            # 現在のセグメントを翻訳
            translated_sentence = translate_single_sentence(current_segment.strip(), target_lang)

            # 改行追加フラグがTrueの場合は改行を追加
            if add_newline:
                translated_sentence += '\n'

            # 翻訳結果を追加
            translated_sentences.append(translated_sentence)

            # セグメントをリセット
            current_segment = ""

    # 最後に残ったセグメントがあれば処理
    if current_segment:
        # 翻訳後の文章を初期化
        translated_sentence = current_segment

        # 最後に残ったセグメントが3文字以上かつ半角数字と半角記号だけでない場合は翻訳
        if 3 <= len(current_segment) and not re.match(r'^[\d\W]+$', current_segment):
            translated_sentence = translate_single_sentence(current_segment.strip(), target_lang)
        # 翻訳されなかった場合
        else:
            print(f"Skipped: {current_segment}")

        # 翻訳結果を追加
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
        elif entry['from'] == 'system':
            # ' -'が含まれる場合は' -'までを翻訳
            # ' -'が含まれない場合は全て翻訳
            if ' -' in entry['value']:
                system_part = entry['value'].split(' -')[0].strip()
                system_translated = translate_text(system_part)
                entry['value'] = entry['value'].replace(system_part, system_translated)
            else:
                entry['value'] = translate_text(entry['value'])

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

    # デバッグ用に先頭10件を表示
    #for index, row in df_chunk.head(10).iterrows():
    #    print(f"{index}: chat: {row['chat']}, system: {row['system']}, conversations: {row['conversations']}")

    # 翻訳結果を保存
    output_file = f'translated_chunk_{start_index}.parquet'
    df_chunk.to_parquet(output_file)
    print(f"Saved: {output_file}")


# 翻訳キャッシュの定期的な保存関数
def save_translation_cache(force=False):
    global cached_text_num, translation_cache, cache_file_name, cache_save_stride

    # 保存されているキャッシュされた翻訳結果数が一定数以上増えた場合に保存
    # 強制的に保存する場合もある
    if len(translation_cache) - cached_text_num < cache_save_stride and not force:
        return

    # 新しいエントリを抽出
    new_entries = list(translation_cache.items())[cached_text_num:]

    # 新しいエントリがある場合のみ追加
    if new_entries:
        new_entries_df = pd.DataFrame(new_entries, columns=['text', 'translated_text'])
        # ファイルに追記
        new_entries_df.to_csv(cache_file_name, mode='a', header=not pd.io.common.file_exists(cache_file_name), index=False)

        # 保存されているキャッシュされた翻訳結果数を更新
        cached_text_num = len(translation_cache)
        print(f"Saved translation cache: {cache_file_name}")


# メイン処理
def process_data_in_chunks(df_base, chunk_size):
    # 全行数を取得
    total_rows = len(df_base)

    # チャンクサイズごとにデータを処理
    for start_index in range(0, total_rows, chunk_size):
        # チャンクを取得して処理
        end_index = min(start_index + chunk_size, total_rows)
        df_chunk = df_base.iloc[start_index:end_index]
        process_chunk(df_chunk, start_index)

        # 翻訳キャッシュを強制保存
        save_translation_cache(force=True)


# Parquetファイル
parquet_file_name = 'train-00001-of-00002-41f063cddf49c933.parquet'

# Parquetファイルの読み込み
df = pd.read_parquet(parquet_file_name)
print(f"Total rows: {len(df)}")

# チャンクサイズを1000行に設定
chunk_size_default = 1000

# 最初の100行に制限しチャンクサイズを25行に設定(テスト用)
#df = df.head(100)
#chunk_size_default = 25

# 翻訳結果のキャッシュファイル
cache_file_name = 'translation_cache.csv'

# 翻訳結果のキャッシュファイルを出力するストライドの行数を設定
# 10行ごとにキャッシュを保存
cache_save_stride = 10

# 翻訳結果のキャッシュファイルの読み込み
if os.path.exists(cache_file_name):
    translation_cache = pd.read_csv(cache_file_name).set_index('text')['translated_text'].to_dict()
else:
    translation_cache = {}

# キャッシュされている翻訳結果数を保存
cached_text_num = len(translation_cache)

# データを指定行数ずつ処理
process_data_in_chunks(df, chunk_size_default)

# 最終的にキャッシュを強制保存
save_translation_cache(force=True)

# 全ての翻訳結果を結合
translated_files = [f'translated_chunk_{i}.parquet' for i in range(0, len(df), chunk_size_default)]
translated_df = pd.concat([pd.read_parquet(file) for file in translated_files], ignore_index=True)

# 結果を保存(元のファイル名に'_translated'を追加)
output_file_name = parquet_file_name.replace('.parquet', '_translated.parquet')
translated_df.to_parquet(output_file_name)