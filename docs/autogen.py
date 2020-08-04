# augtogen.py
# APIのドキュメントをdoc_stringから生成する。
#
# Qumicoをgit cloneしたディレクトリで実行してください。
# 実行例
# python3 docs/autogen.py

import importlib
import inspect
import os
import re

module_name = 'qumico.Qumico'
output_dir = 'docs/API'
output_file = 'api.md'

# markdownファイルの先頭に来る文字列
md_header = """
# Qumico API
---

            """

# markdownファイルの最後に来る文字列
md_footer = """
"""


def make_markdown(output_file, qumico_doc, func_docs):
    with open(output_file, "w",  encoding='utf-8') as fp:
        fp.write(md_header + '\n')

        # Qumicoクラスの情報出力
        lines = qumico_doc.split('\n')
        md_text = ""
        for l in lines:
            md_text += re.sub('^\\s+', '', l)
            md_text += '\n'

        fp.write(md_text + '\n')

        # methodの出力
        for func_name, func_doc in func_docs:
            # docstringが存在しないmethodは、APIとして出力されない。
            if func_doc == "":
                continue
            fp.write('### ' + func_name + '\n\n')

            lines = func_doc.split('\n')
            md_text = ""
            for l in lines:
                md_text += re.sub('^\\s+', '', l)
                md_text += '\r\n'

            fp.write(md_text + '\n')

        fp.write(md_footer)


def main():
    module = importlib.import_module(module_name)

    # Qumicoの情報取得
    qumico_doc = inspect.getdoc(module.Qumico)

    # 関数の情報取得
    func_docs = []
    functions = inspect.getmembers(module.Qumico, inspect.isfunction)

    for func_name, func_object in functions:
        func_doc = inspect.getdoc(func_object) or ''
        func_docs.append((func_name, func_doc),)

    # 結果を書き込み。
    target_path = os.path.join(output_dir, output_file)
    make_markdown(target_path, qumico_doc, func_docs)
    print("generated  ", target_path)


if __name__ == "__main__":
    main()
