# AutoML_DogCat(TFLie形式)
GCP AutoML Vision(分類)で学習したモデルをCソースに変換し、RaspberryPiで動作させることができます。  
このサンプルでは、犬・猫を分類するモデルを含んでいます。

以下の環境でデモの動作を確認しています。
- Raspberry Pi 3 Model B
- Raspbian、Buster GNU/Linux 10.1
- python3-opencv 3.2.0+dfsg-6
- libopencv-core3.2 3.2.0+dfsg-6

## フォルダ構成
ルートフォルダ: samples/automl_dogcat/tensorflow/

- images:推論サンプル画像
- model: TFLite形式のモデル格納先
- symbol: デモ向けの識別結果表示用の画像(犬、猫を格納済み)
- asm-rpi.py: Cソースから生成されるアセンブラコードを出力するスクリプト(※RaspberryPi向け)
- automl_common.py: 本サンプルの共通設定
- automl_infer.py: TensorFlowを利用した推論スクリプト
- automl_tflite_to_onnx.py: TFLite形式をONNX形式に変換するスクリプト
- build-rpi.py: RaspberryPi上でのビルドスクリプト
- demo.py: デモ用スクリプト
- gen_c.py: ONNX形式モデルからCソースを生成するスクリプト(※PC向け)
- gen_c_rpi.py: ONNX形式モデルからCソースを生成するスクリプト(※RaspberryPi向け)


## モデルの格納
GCP AutoML Vision(分類)で学習した後、エッジデバイス向けにGCPから取得します。  
このサンプルでは、犬・猫分類のモデルを以下に格納済みです。  

`samples/automl_dogcat/tensorflow/`  

- dict.txt
- metadata.json
- model.tflite  

## Pythonを使った推論
Qumicoをgit cloneしたディレクトリ「samples/automl_dogcat/tensorflow」 へ移動してください。  
Qumicoを使用してC言語に変換する前に、学習したモデルが正しく推論できるかを確認します。次のコマンドを実行してください。  

```sh
python3 automl_infer.py
```

推論結果として、最終行に以下が表示されれば成功です。  
後でC言語を使った推論結果と比較します。  

```
predictions dog [[ 19 237]]
```
※デフォルトでは images/dog1.jpeg を推論します。


## ONNX形式への変換
C生成の前にTFLite形式をONNX形式に変換します。  


```sh
python3 automl_tflite_to_onnx.py 
```

プログラム実行後、このように表示されれば正常に終了しています。  

```
onnx/model.onnxを作成しました。
```

onnxディレクトリにmodel.json, model.onnxが生成されていれば、onnx変換は成功です。

## Cソースへの変換
上で生成したonnxファイルを使用し、ニューラルネットワークをCソースに変換します。

```sh
python3 gen_c.py 
```

このように表示されば、正常に終了しています。  

```
Cソースを生成しました。出力先: out_c
```

out_cディレクトリに、includeディレクトリとlibディレクトリ、qumico.cとqumico.soが出力されていれば、Cソースへの変換は成功です。

## C言語での実行
上で生成した共有ライブラリqumico.soを使って推論を実行してみます。  

```sh
 python3 automl_infer_c.py 
```

最後の2行が次のように表示され、Pythonを使った推論と同じ分類結果になれば、C言語で正しく推論ができています。  

```
prediction dog [[ 14 242]]
finish
```

## RapsberryPi上での実行
「ONNX形式への変換」を実施した後、RaspberryPi向けCソースを生成します。  
デフォルトでは、OpenMP=有効、SIMD(Neon) =有効となります。  

```sh
python3 gen_c_rpi.py 
```

その後、以下のフォルダをRaspberryPi上の任意のフォルダへ移植する。  

`samples/automl_dogcat/tensorflow/`

以下操作をRaspberryPi上で実施します。

1. ビルド  
RaspberryPiへの移植先フォルダに移動し、build-rpi.pyを実行する。  
out_cフォルダに共有ライブラリとして、qumico.soが生成される。  

2. デモ  
demo.pyを実行する。  
カメラが有効となり、動画の撮影対象に猫・犬があれば、その識別結果をシンボル画像表示する。  


## 別モデルの利用手順
このサンプル構成を利用し、別モデルを利用する場合の手順をまとめます。

1.GCP AutoML Vision(分類)で学習したモデルを格納する  
　格納先: samples/automl_dogcat/tensorflow/model  
　GCPから取得されるファイルすべてを格納する。  
　・dict.txt  
　・metadata.json  
　・model.tflite  

2.automl_tflite_to_onnx.pyを実行する  
　以下にONNXファイルが生成される。  
　samples/automl_dogcat/tensorflow/onnx/model.onnx  

　補足：また中間ファイルとして、以下のJSONが生成されます。  
　samples/automl_dogcat/tensorflow/model/model.json

3.gen_c_rpi.pyを実行する  
　以下にCソースが生成される。  
　samples/automl_dogcat/tensorflow/out_c  

4.ソースをRaspberryPiへ移植する  
　以下のフォルダごとRaspberryPiへ移植する。  
　samples/automl_dogcat/tensorflow/

5.ビルド  
　RaspberryPiへの移植先フォルダに移動し、build-rpi.pyを実行する。  
　out_cフォルダに共有ライブラリとして、qumico.soが生成される。  

6.デモ  
　demo.pyの「識別結果に応じた表示用シンボル画像の表示設定」(80行目あたり)にある表示用シンボル画像を差し替える。  
　識別結果に応じて、表示するシンボル画像を切り替える修正をする。  

　シンボル画像格納先： samples/automl_dogcat/tensorflow/symbol  

　サンプルでは識別用の画像として、次のsymbolフォルダの画像を利用している。  
　サンプル例：dogの場合、sysmbol/dog.png  
　　　　　　　catの場合、sysmbol/cat.png  

　その後、demo.pyを実行する。  
　