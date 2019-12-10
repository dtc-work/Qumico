# Mobilenet(TFLie形式)
TFLite形式のMobilenetモデルをC言語に変換し、RaspberryPiで動作させることができます。

※ このサンプルのモデルは、以下を利用しています。  
モデル: [Mobilenet_V1_0.25_128_quant](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz)  
URL: [TensorFlow Hosted Models](https://www.tensorflow.org/lite/guide/hosted_models)

## モデル
量子化済みMobilenet(入力サイズ=128x128)を以下に格納しています。  


samples/mobilenet/tensorflow/model/mobilenet_v1_0.25_128_quant.tflite

## Pythonを使った推論

Qumicoを使用してC言語に変換する前に、TensorFlowにてモデルが正常に推論できるかを確認します。  
サンプル画像(samples/mobilenet/tensorflow/images/tiger.jpeg)を推論するために次のコマンドを実行してください。  

```sh
python3 mobilenet_infer.py
```

推論結果として、最終行に以下が表示されれば成功です。  
後でC言語を使った推論結果と比較します。  

`tiger`

## TFLiteモデルからONNXモデルへの変換
Qumicoをgit cloneしたディレクトリの samples/mobilenet/tensorflow へ移動してください。  

```sh
cd Qumico/samples/mobilenet/tensorflow
```

次のコマンドを実行してください。

```sh
python3 mobilenet_tflite_to_onnx.py
```

プログラム実行後、次のように表示されてれば正常に終了しています。  

```
onnx/mobilenet_v1_0.25_128_quant.onnxを作成しました。
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
 python3 mobilenet_infer_c.py 
```

このように表示され、Pythonを使った推論と同じ結果になれば、C言語で正しく推論ができています。  
`tiger`

