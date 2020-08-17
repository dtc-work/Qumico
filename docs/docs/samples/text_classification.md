# text_classification
---
twitterに投稿されたニュースをフェイクニュースかどうかを識別するモデルを使い、RaspberryPiで動作させることができます。
このサンプルには、テキストを識別する学習済みモデルを含んでいます。

以下の環境でデモの動作を確認しています。
- Raspberry Pi 3 Model B
- Raspbian、Buster GNU/Linux 10.1
- python3-opencv 3.2.0+dfsg-6
- libopencv-core3.2 3.2.0+dfsg-6


## フォルダ構成
ルートフォルダ: samples/text_classification/keras

- data:学習用データ
- images: デモで使用する画像ファイル
- model: hdf形式のモデル格納先
- onnx: onnxファイル出力先
- out_c: Cソースの生成先
- build-rpi.py: RaspberryPi上でのビルドスクリプト
- conv_to_onnx.py: ONNXへの変換スクリプト
- demo.py: デモ用スクリプト
- gen_c.py: ONNX形式モデルからCソースを生成するスクリプト(※PC向け)
- infer.py: 推論用スクリプト
- infer_c.py: C言語を使用した推論用スクリプト
- model.py: モデルの定義ファイル
- train.py: 学習用スクリプト
- utils.py: ユーティリティ関数の定義ファイル



## 学習済みモデル
学習済みのモデルを以下に格納しています。

`samples/text_classification/keras/model`  

- label_encoder.pickle
- tokenizer.json
- TweetDisaster.hdf5
- TweetDisaster.json
- TweetDisaster.yaml

## NLTKライブラリのダウンロード

ターミナルより、pythonインタプリタを起動し、以下のコマンドを入力してください。
pythonインタプリタはterminalよりpython、もしくはpython3で起動します。
pythonインタプリタが起動すると >>> のプロンプトが表示されます。

```python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
>>> exit()
```
pythonインタプリタからnltk.download()を実行する事で、必要なデータがダウンロードされます。

## Pythonを使った推論

Qumicoをgit cloneしたディレクトリから「samples/text_classification/keras」 へ移動してください。  
Qumicoを使用してC言語に変換する前に、学習したモデルが正しく推論できるかを確認します。次のコマンドを実行してください。  

推論をするために、以下のサイトから、train.csvとtest.csvをダウンロードし、samples/text_classification/keras/data に保存してください。

https://www.kaggle.com/c/nlp-getting-started

```sh
python3 infer.py
```

最後に推論結果として、このような表示がでれば正しく動いています。
後でC言語を使った推論結果と比較します。  

```text
prediction: Real (0.644)  : @thehill this is 1 example of y the Conservatives annihilated Burton v Wiimington Prkng Auth while Liberals stood by &amp;have done nothing
prediction: Real (0.620)  : Aug 3 1915ÛÓKAISERJAEGERS WIPED OUT.; Francis Joseph's Crack Regiment Annihilated on Carso Plateau.http://t.co/D1sPSwl66H
prediction: Fake (0.257)  : They should all die! All of them! Everything annihilated!
```


## ONNX形式への変換

C生成の前にHDF形式のモデルをONNX形式に変換します。  


```sh
python3 conv_to_onnx.py
```

プログラム実行後、このように表示されれば正常に終了しています。  

```
ONNXファイルを生成しました。出力先: onnx/TweetDisaster.onnx
```

onnxディレクトリにTweetDisaster.onnxが生成されていれば、onnx変換は成功です。

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
 python3 infer_c.py 
```

推論結果が、先に実行したPythonと同じあればC言語への変換は正しく行われています。

```text
prediction: Real (0.644)  : @thehill this is 1 example of y the Conservatives annihilated Burton v Wiimington Prkng Auth while Liberals stood by &amp;have done nothing
prediction: Real (0.620)  : Aug 3 1915ÛÓKAISERJAEGERS WIPED OUT.; Francis Joseph's Crack Regiment Annihilated on Carso Plateau.http://t.co/D1sPSwl66H
prediction: Fake (0.257)  : They should all die! All of them! Everything annihilated!
```


## RapsberryPi上での実行

PC上の、以下のフォルダをRaspberryPi上の任意のフォルダへコピーしてください。

`samples/text_classification/keras/`

以下の操作をRaspberryPi上で実施します。

1 ビルド  
RaspberryPiへのコピー先フォルダに移動し、build-rpi.pyを実行します。

```
python build-rpi.py
``` 

このように表示されれば成功です。

```
qumico.so を生成しました。
```

2 デモ  
demo.pyを実行する。  

```
python demo.py
``` 

demo.pyを実行したターミナルがキーボードからの入力待ちになります。
好きな英文を入力してください。
ニュースとしての信頼性を表示し、FakeかRealの結果を画像で表示します。
デモを終了するには、.のみ入力しEnterキーを押してください。

なかなかRealと判定される英文は難しいですが、下の文はRealと判定されます。
動作確認としてお使いください。

```
@thehill this is 1 example of y the Conservatives annihilated Burton v Wiimington Prkng Auth while Liberals stood by &amp;have done nothing
```
