import numpy as np


def sigmoid(x):
    """
        ndarray形式のデータに対してシグモイド関数を提供する。
        #### 引数
        - x： numpy形式のtensor
        #### 戻り値
        シグモイド関数の計算結果
    """
    return 1 / (1 + np.exp(-x))

def softmax(x,axis=0):
    """
        ndarray形式のデータに対してソフトマックス関数を提供する。
        #### 引数
        - x： numpy形式のtensor
        - axis: ソフトマックスを計算する軸方向を指定する。xが2次元の場合に有効(default=0)
        #### 戻り値
        ソフトマックス関数の計算結果
    """
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=axis)
        y = np.exp(x) / np.sum(np.exp(x), axis=axis)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

def onehot_encoding(x, classes):
    """
        ndarray形式のデータに対して、one-hotエンコーディングを行うメソッドを提供する。
        #### 引数
        - x： 戻り値として出力する行を指定する。
        - classes: one-hotエンコーディングするクラスを指定する。
        #### 戻り値
        指定した行のone-hotエンコーディング結果を返す。
    """
    output = np.identity(classes)[x]
    return output


def onehot_decoding(x):
    """
        one-hotエンコードされているデータに対して、行ごとに最大値のリストを返す関数を提供する。
        #### 引数
        - x: one-hotエンコードされているデータ(array_like)
        #### 戻り値
        行ごとに最大引数値のリストを返す。
    """
    x = np.asarray(x)
    y = np.argmax(x, axis=-1)
    return y
