# Convolution
Convolution層を使ったMNISTのサンプルを通して、Qumicoがどのように動作をするかを確認できます。

## 学習とモデルの保存
Qumicoをgit cloneしたディレクトリの samples/conv/tensorflow へ移動してください。
まずは、MNISTのデータをダウンロードし、学習と学習結果の保存を行います。

```sh
python3 conv_train.py 
```

プログラム実行後、このように表示されれば正常に終了しています。
```
onnx/tensorflow_conv.onnxを作成しました.
```

modelディレクトリにsample.ckptファイル、onnxディレクトリにtensorflow_conv.onnxのファイルが生成されていれば、学習結果の保存は成功です。

## Pythonを使った推論
Qumicoを使用してC言語に変換する前に、学習したモデルが正しく推論できるかを確認します。次のコマンドを実行してください。
```sh
python3 conv_infer.py
```
MNISTの推論結果として、このように表示されれば成功です。後でC言語を使った推論結果と比較します。
```
Predict Index  [[7 2 1 0 4 1 4 9 5 9]]
```

## Cソースへの変換
上で生成したonnxファイルを使用し、ニューラルネットワークをCソースに変換します。

```sh
python3 gen_c.py 
```

このように表示されば、正常に終了しています。
```
Cソースを生成しました。出力先: out_c
```

out_cディレクトリに、includeとlibディレクトリ、qumico.cとqumico.soが出力されていれば、Cソースへの変換は成功です。

## C言語での実行
上で生成した共有ライブラリqumico.soを使って推論を実行してみます。
```sh
 python3 conv_infer_c.py 
```

このように表示され、Pythonを使った推論と同じ結果になれば、C言語で正しく推論ができています。
```
Predict Index [7, 2, 1, 0, 4, 1, 4, 9, 5, 9]
```
