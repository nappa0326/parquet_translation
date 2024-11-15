import pandas as pd
import csv

def find_error_lines(file_path):
    error_lines = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # ヘッダーを読み飛ばす
        expected_fields = len(header)

        for line_number, row in enumerate(reader, start=2):  # ヘッダー行を考慮して2から開始
            if len(row) != expected_fields:
                error_lines.append((line_number, row))

    return error_lines


def main():
    file_path = 'translation_cache.csv'
    error_lines = find_error_lines(file_path)

    if error_lines:
        print("エラーが発生した行:")
        for line_number, row in error_lines:
            print(f"行番号 {line_number}: {row}")
    else:
        print("エラーは見つかりませんでした。")


if __name__ == "__main__":
    main()