import os

def get_data_path_list(data_root_path, depth=1):
    """
        指定したディレクトリのファイルリストを作成する。
        ### 引数
        - data_root_path: ファイルリストを作成する最上位ディレクトリを指定する。
        - depth: =1: data_root_path直下のファイルリストを作成する。
                 =2: data_root_pathの2階層下のファイルリストを作成する。
        ### 戻り値
        作成したファイルリストをリスト形式で返す。
    """
    data_path_list = []
    if depth == 2:
        for index, folder in enumerate(os.listdir(data_root_path)):
            for filename in os.listdir(data_root_path + "/" + folder):
                data_path_list.append(data_root_path + "/" + folder + "/" + filename)
    elif depth == 1:
        for filename in os.listdir(data_root_path):
            data_path_list.append(data_root_path + "/" + filename)
    else:
        pass
    data_path_list = sorted(data_path_list)
    return data_path_list




