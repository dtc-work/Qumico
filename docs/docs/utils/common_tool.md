# common_tool.py

## onehot_decoding
one-hotエンコードされているデータに対して、行ごとに最大値のリストを返す関数を提供する。

#### 引数
- x: one-hotエンコードされているデータ(array_like)

#### 戻り値
行ごとに最大値のリストを返す。
<br>
<br>

## onehot_encoding
ndarray形式のデータに対して、one-hotエンコーディングを行うメソッドを提供する。

#### 引数
- x： 戻り値として出力する行を指定する。
- classes: one-hotエンコーディングする階調を指定する。

#### 戻り値
指定した行のone-hotエンコーディング結果を返す。
<br>
<br>

## sigmoid
ndarray形式のデータに対してシグモイド関数を提供する。

#### 引数
- x： numpy形式のtensor

#### 戻り値
シグモイド関数の計算結果
<br>
<br>

## softmax
ndarray形式のデータに対してソフトマックス関数を提供する。

#### 引数
- x： numpy形式のtensor
- axis: ソフトマックスを計算する軸方向を指定する。xが2次元の場合に有効(default=0)

#### 戻り値
ソフトマックス関数の計算結果


