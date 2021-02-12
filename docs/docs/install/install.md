# Install 方法

## 事前準備

Qumicoは、Python3.8以上で動作します。

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
python setup.py install --record files.txt
cat files.txt | xargs rm -rf
```

## サンプルの動作確認
Qumicoディレクトリから以下のコマンドを実行してください。

```sh
cd samples/mlp/tensorflow
python mlp_train.py 
```
プログラムが起動し、以下の表示がされれば正しくインストールされています。

```sh
onnx/tensorflow_mlp.onnxを作成しました.
```