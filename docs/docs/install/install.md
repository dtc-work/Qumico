# Install 方法

## 事前準備

Qumicoは、Python3.6以上で動作します。
また、Qumicoを使うためには下記のpython3ライブラリが必要で、事前に用意してください。

```sh
$ pip3 install [インストールするpython3ライブラリ]
```

インストールするpython3ライブラリ項目
```
 onnx==1.5.0
 tf2onnx==1.5.1
 tensorflow>=1.13, <2.0
 Keras
 numpy
 pillow
 opencv-python
```

## git clone

作業用ディレクトリに移動し、git cloneを実行してください。
```sh
git clone https://github.com/PasonaTech-Inc/Qumico
```

## インストール

Qumicoディレクトリへ移動し、以下のようにsetup.pyを実行してください。

```sh
cd Qumico
sudo pip3 install -e .　
```

##  アンインストール

```sh
python3 setup.py install --record files.txt
cat files.txt | xargs rm -rf
```

## サンプルの動作確認
Qumicoディレクトリから以下のコマンドを実行してください。

```sh
cd samples/mlp/tensorflow
python3 mlp_train.py 
```
プログラムが起動し、以下の表示がされれば正しくインストールされています。

```sh
onnx/tensorflow_mlp.onnxを作成しました.
```