# pre_process_tool.py

## get_data_path_list
指定したディレクトリのファイルリストを作成する。

### 引数
- data_root_path: ファイルリストを作成する最上位ディレクトリを指定する。
- depth: =1: data_root_path直下のファイルリストを作成する。
=2: data_root_pathの2階層下のファイルリストを作成する。

### 戻り値
作成したファイルリストをリスト形式で返す。
