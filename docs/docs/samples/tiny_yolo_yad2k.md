# Tiny YOLOv2(YAD2K)
Tiny YOLOv2を使用して物体識別を行います。
このモデルでは、あらかじめdarknetのTiny YOLOv2で学習済みのデータを用いて、
物体識別の推論のみを行います。
また、このサンプルでは、RaspberryPi3による推論実行を行います。

## KerasモデルからONNXモデルへの変換
Qumicoをgit cloneしたディレクトリの samples/tiny_yolo_v2_yad2k/keras へ移動してください。
```sh
cd Qumico/samples/tiny_yolo_v2_yad2k/keras
```

次のコマンドを実行してください。

```sh
python3 tiny_yolo_v2_yad2k_conv_onnx.py
```
プログラム実行後、次のように表示されてれば正常に終了しています。
```
onnx/tiny_yolo_v2_yad2k.onnxを作成しました。
```

## Cソースへの変換(1)
上で生成したonnxファイルを使用し、ニューラルネットワークをCソースに変換します。次のコマンドを実行してください。

gen_c.pyを次のように実行してください。
```sh
python3 gen_c.py 
```
このように表示されば、正常に終了しています。
```
[1, 416, 416, 3]
[1, 13, 13, 125]
Cソースを生成しました。出力先: out_c
```

out_cディレクトリディレクトリに、次のファイルが作成できていればCソースへの変換は成功です。
- includeディレクトリ
- initializersディレクトリ
- libディレクトリ
- qumico.c
- qumico.so



## 最適化の実行
推論モデルの実行効率を上げるため、モデル最適化の実行を行います。

次のコマンドを実行してください。

```sh
python3 tiny_yolo_v2_yad2k_optimize_onnx.py
```
プログラム実行後、特に表示するメッセージはありません。


## Pythonを使った推論
Qumicoを使用してC言語に変換する前に、学習したモデルが正しく推論できるかを確認します。次のコマンドを実行してください。
```sh
python3 tiny_yolo_v2_yad2k_infer.py
```

このように表示されば、正常に終了しています。
```
finish
```

## Cソースへの変換(2)
最適化したニューラルネットワークモデルのonnxファイルを使用し、Cソースに変換します。次のコマンドを実行してください。

gen_c_optimize.pyを次のように実行してください。
```sh
python3 gen_c_optimize.py 
```
このように表示されば、正常に終了しています。
```
[1, 416, 416, 3]
[1, 13, 13, 125]
Cソースを生成しました。出力先: out_c_optimize
```
out_c_optimizeディレクトリに、次のファイルが作成できていればCソースへの変換は成功です。
- includeディレクトリ
- initializersディレクトリ
- libディレクトリ
- qumico.c
- qumico.so


---------------------------------------------------------------------------------------------------------------------------

## RaspberryPi3上での実行準備

RaspberryPi3 では、Raspbian stretch(2019-04-11)で確認を行っています。

Python3のライブラリとして
- numpy (バージョン1.10.0以上)
- opencv-python
を使用します。

下記のコマンドを実行してライブラリをインストールしてください。
```sh
sudo pip3 install opencv-python
```

OpenCV等のパッケージを使用します。
下記のコマンドを実行してパッケージをインストールしてください。
```sh
sudo apt-get install libcblas3
sudo apt-get install libatlas3-base
sudo apt-get install libjasper1
sudo apt-get install libqt4-test
sudo apt-get install libqtgui4
sudo apt-get install python3-pil
```
一部のパッケージは既にインストール済みで更新されない場合があります。


## RaspberryPi3へのコピー
Qumico/samples/tiny_yolo_v2_yad2k/ディレクトリ以下をRaspberryPi3にコピーしてください。

## 実行するコードの選択

上記で、ONNXモデルを最適化しなかったものと最適化したものの2種類のCソースコードへの変換を行いました。  
そのCソースの中から、どちらをRaspberryPi3上で実行するかを選択してください。  
下記の実行例は、ONNXモデルを最適化したものを主として説明を行い、最適化しなかったものについて補足する形式で説明を行います。

## RaspberryPi3上でのビルド
RaspberryPi3上で、コピーしたディレクトリ中のtiny_yolo_v2_yad2k/kerasに移動してください。  
```sh
cd keras
```

次にビルドを実行してください。
```sh
python3 build.py
```

最適化していないONNXモデルを使用する場合  
build.py中の次の箇所にある、"out_c_optimize"を、"out_c"に書き換えてください。
だいたい14行目程度にあります。その後にビルドを実行してください。
```python
    c_path = path.join(path.dirname(path.abspath("__file__")), "out_c_optimize", "qumico.c")
```


## 推論の実行(1)

画像を1回だけ推論する場合と、連続して推論する場合があります。
1回だけ推論する場合は、主に静止画ファイルの推論に、連続して推論する場合は、主にPiCameraからの入力を連続して推論する場合に使用します。

1回だけ推論する場合、下記のコマンドを実行してください。(images/person.jpgを推論する)
```sh
python3 tiny_yolo_v2_yad2k_infer_c.py
```
pythonで推論した場合と同じ結果になれば、正しく推論されています。

(例)
```
run:start 2019-06-17 19:11:46.450442
run:end 2019-06-17 19:11:55.944661
post:end 2019-06-17 19:11:56.076693
sheep 0.8137884 (428, 145) (590, 336)
person 0.6564155 (178, 108) (282, 371)
cow 0.4457574 (66, 267) (188, 356)
draw 2019-06-17 19:11:56.476736
elapsed_time:10.164890766143799[sec]
finish
```

最適化していないONNXモデルを使用する場合  
tiny_yolo_v2_yad2k_infer_c.py中の次の箇所にある、"out_c_optimize"を、"out_c"に書き換えてください。
だいたい46行目程度にあります。その後に推論を実行してください。
```python
    so_lib_path= path.join(path.dirname(__file__), "out_c_optimize", "qumico.so")
```


## 推論の実行(2)

連続して推論する場合、下記のコマンドを実行してください。
```sh
python3 app.py
```

最適化していないONNXモデルを使用する場合  
app.py中の次の箇所にある、"out_c_optimize"を、"out_c"に書き換えてください。
だいたい65行目程度にあります。その後に推論を実行してください。
```python
    dll_path= path.join(path.dirname(path.abspath("__file__")), "out_c_optimize", "qumico.c")
```

