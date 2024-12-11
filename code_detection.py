import re
from pygments.lexers import guess_lexer, ClassNotFound
import csv

def score_line(text):
    score = 0

    programming_keywords = {
        # 共通キーワード
        "print", "def", "class", "import", "return", "if", "elif", "else",
        "for", "while", "try", "except", "with", "lambda", "echo", "function",
        "namespace", "using", "console", "let", "const", "public", "private",
        "protected", "extends", "interface", "<html>", "<body>", "<div>",

        # C++
        "std", "cin", "cout", "vector", "map", "new", "delete", "template",
        "typename", "virtual", "override", "nullptr",

        # C#
        "using", "System", "var", "new", "async", "await", "get", "set",
        "delegate", "event", "yield",

        # Java
        "package", "import", "class", "public", "static", "void", "main",
        "try", "catch", "finally", "throw", "throws", "new", "extends",
        "implements",

        # JavaScript
        "document", "window", "let", "const", "var", "function", "=>", "export",
        "import", "from", "require", "module.exports",

        # TypeScript
        "type", "interface", "implements", "extends", "readonly", "enum",

        # PHP
        "<?php", "echo", "function", "class", "public", "private", "protected",
        "include", "require", "use", "namespace",

        # SQL
        "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "JOIN",
        "INNER", "LEFT", "RIGHT", "FULL", "CREATE", "TABLE", "DROP", "ALTER",
        "GRANT", "REVOKE", "USE",

        # HTML
        "<html>", "<head>", "<body>", "<title>", "<div>", "<span>", "<h1>",
        "<h2>", "<p>", "<a>", "<img>", "<ul>", "<ol>", "<li>", "<table>",
        "<tr>", "<td>", "<th>", "<form>", "<input>", "<button>", "<script>", "<style>",

        # Python
        "def", "class", "import", "from", "return", "if", "elif", "else",
        "for", "while", "try", "except", "finally", "with", "lambda",
        "yield", "async", "await", "as",
    }

    # 1. キーワードマッチ
    if any(keyword in text for keyword in programming_keywords):
        score += 2

    # 2. 記号の密度
    code_symbols = "{}[]();=+-*/<>"
    symbol_count = sum(text.count(ch) for ch in code_symbols)
    if len(text) > 0 and symbol_count / len(text) > 0.05:  # 閾値を下げる
        score += 2  # スコアを増加

    # 3. 構文解析
    try:
        lexer = guess_lexer(text)
        if lexer.name in ["Python", "PHP", "JavaScript", "HTML", "C++", "C#", "Java", "Mojo"]:
            score += 3
    except ClassNotFound:
        pass

    # 4. HTMLタグ検出
    html_pattern = r"<\/?[a-zA-Z][^>]*>"
    if re.search(html_pattern, text):
        score += 2

    # 5. 数式らしさを追加
    formula_pattern = r"^[a-zA-Z\s]*=[^=]+$"
    if re.match(formula_pattern, text):
        score += 2

    # プログラミング言語ではあまり使用されず自然言語で頻繁に使われる単語リスト
    # |で区切った文字列を作成
    common_words = "|".join([
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "is", "are", "was", "were",
    ])

    # common_wordsからタイトルケースの単語リストを作成
    common_title_words = "|".join([word.title() for word in common_words.split("|")])

    # 6.文の長さや構造(自然言語で頻繁に使われる単語が含まれている場合スコアを減点)
    if re.search(r'\b(' + common_words + r')\b', text):
        score -= 1

    # 7.タイトルケースの単語が含まれている場合スコアを大きく減点
    if re.search(r'\b(' + common_title_words + r')\b', text):
        score -= 2

    # 8.特定の数学的記号や変数名のスコア減点
    math_like_symbols = "=()"
    if all(ch in math_like_symbols or ch.isalnum() for ch in text.replace(" ", "")):
        # 文中に自然言語らしい単語があれば減点
        if re.search(r'\b(' + common_words + r')\b', text.lower()):
            score -= 1

    return score

def is_code_snippet(text, threshold=3):
    """
    スコアを計算し、閾値を超えた場合にコード/数式と判定
    """
    return score_line(text) >= threshold

## テストケース
#test_cases = [
#    "print('Hello, world!')",  # Pythonコード
#    "ただの日本語テキスト",       # テキスト
#    "x = 42",                  # シンプルなコード
#    "E = mc^2",                # 数式
#    "if (x == 10) { y = x + 1; }",  # C系コード
#    "こんにちは、世界！",          # テキスト
#    "<html><body>Hello!</body></html>",  # HTMLコード
#    "function test() { return true; }"  # JavaScriptコード
#]
#
#for text in test_cases:
#    print(f"'{text}' -> {is_code_snippet(text)}")
#
## テストデータ(英語の文章とプログラムソースコード)
#test_data = [
#    "This is a sample text.",
#    "print('Hello, world!')",
#    "def add(a, b):\n    return a + b",
#    "import numpy as np",
#    "for i in range(10):\n    print(i)",
#    "x = 10",
#    "y = x + 20",
#]
#
## テストデータの分類結果を表示
#for data in test_data:
#    print(f"{data} -> {is_code_snippet(data)}")


def main():
    file_path = 'translation_cache.csv'
    output_file_name = 'translation_cache.csv'

    try:
        # csvファイルの内容を読み込む
        with open(file_path, mode='r', encoding='utf-8') as file:
            # ヘッダを利用して辞書形式でcsvを読み込む
            reader = csv.DictReader(file)

            # ヘッダー情報を保存
            fieldnames = reader.fieldnames

            # 新しいcsvファイルを作成
            with open(output_file_name, mode='w', encoding='utf-8', newline='') as output_file:
                # ヘッダーを書き込む
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()

                # csvの内容を1行ずつコード・計算式など翻訳が必要ない文字列かどうか判定
                for line_number, row in enumerate(reader, start=1):
                    # textとtranslated_textが同じ場合はスキップ
                    if row["text"] == row["translated_text"]:
                        # 新しいcsvファイルに書き込む
                        writer.writerow(row)
                        continue

                    # textの内容がコード・計算式など翻訳が必要ない文字列の場合
                    if is_code_snippet(row["text"]):
                        print(f"行番号 {line_number}: {row}")

                        # textをtranslated_textにコピー
                        row["translated_text"] = row["text"]

                        # 新しいcsvファイルに書き込む
                        writer.writerow(row)
                    # textの内容がコード・計算式など翻訳が必要ない文字列でない場合
                    else:
                        # 新しいcsvファイルに書き込む
                        writer.writerow(row)

    except FileNotFoundError:
        print(f"{file_path}が見つかりませんでした。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
