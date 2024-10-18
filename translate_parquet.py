import torch
import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parquetファイルの読み込み
df = pd.read_parquet('train-00000-of-00002-6f3344faa23e9b0a.parquet')

# デバッグ用にdfの最初の10行だけ残して後は削除
#df = df.head(10)

# デバッグ用にCSVファイルとして出力
#df.to_csv('dataset_debug.csv', index=False)

# モデルとトークナイザーの準備
model_name = "facebook/mbart-large-50-one-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX")

# モデルをGPUに移動
model.to(device)

# ターゲット言語を日本語に設定
tokenizer.src_lang = "en_XX"


def translate_text(text, target_lang="ja_XX"):
    # テキストをトークナイズ
    model_inputs = tokenizer(text, return_tensors="pt")

    # 入力をGPUに移動
    model_inputs = model_inputs.to(device)

    # 翻訳を生成
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        num_beams=3,  # ビームサーチの幅を設定
        early_stopping=True
    )
    # 翻訳結果をデコード
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(f"Translated text: {translated_text}")
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


df['chat'] = df['chat'].apply(translate_chat)


# system列の翻訳
def translate_system(system):
    system_part = system.split('SYSTEM:')[1].split(' -')[0].strip()
    system_translated = translate_text(system_part)
    system = system.replace(system_part, system_translated)
    return system


df['system'] = df['system'].apply(translate_system)


# conversations列の翻訳
def translate_conversations(conversations):
    for entry in conversations:
        if entry['from'] == 'human':
            entry['value'] = translate_text(entry['value'])
        elif entry['from'] == 'gpt' and '<functioncall>' not in entry['value']:
            entry['value'] = translate_text(entry['value'].split('<|endoftext|>')[0]) + '<|endoftext|>'
    return conversations


df['conversations'] = df['conversations'].apply(translate_conversations)

# デバッグ用にCSVファイルとして出力
#df.to_csv('translated_dataset_debug.csv', index=False)

# 翻訳結果をParquetファイルに保存
df.to_parquet('translated_dataset.parquet')