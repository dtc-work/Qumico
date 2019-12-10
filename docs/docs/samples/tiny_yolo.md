# Tiny YOLOv2
Tiny YOLOv2を使用して物体識別を行います。


## はじめに
Qumicoをgit cloneしたディレクトリの samples/tiny_yolo_v2/tensorflow へ移動してください。
```
cd Qumcio/samples/tiny_yolo_v2/tensorflow
```

!!!warning
TensorFlowのONNXの出力部分にバグがあるたため、Tiny YOLOv2のサンプルを動かすためにはGPU環境が必要です。


## 学習
VOC2007のデータを利用して学習を行います。
学習を試行する時間の無い方のために、学習済みのデータを用意しています。
学習済みのデータを使用する方は、**学習済みデータの使い方**まで進んでください。

tiny_yolo2_train.py の root_path, data_list_path, label_list_path の設定で学習データを格納したフォルダを設定します。　　

(例)
```python
    # 学習データルートパス(絶対パス)を設定する
    # root_path = ""
    # 画像ファイルフォルダ
    data_list_path = root_path + "JPEGImages"
    # タグデータフォルダ（サンプルでは xml に対応しています、
    # 他のデータで学習させたい場合は、utilsフォルダ中のannotation_dataset_toolフ
ァイルに読み込むロジックを追加してください。
    label_list_path = root_path + "Annotations"
```
root_pathの前のコメント(#)を削除してください。  
学習ルートパスとして ./data を指定する場合は、root_path = "data" を定義します。  
root_path = "data"の場合に、data_list_path として ./data/JPEGImages/を使う場合、  
label_list_path として ./data/Annotations/ を使う場合には、
data_list_pathとlabel_list_pathの変更はありません。

次のコマンドを実行して、学習を行います。
```sh
python3 tiny_yolo2_train.py
```

プログラム実行後、このように表示されれば正常に終了しています。
```
onnx/tensorflow_mlp.onnxを作成しました。
```

ファイルが生成されていれば、学習結果の保存は成功です。  
**Pythonを使った推論**まで進んでください。

## 学習済みデータの使い方
model/ ディレクトリに、学習済みデータを用意していますので、
次のようにリネームしてください。
```sh
mv model/tiny_yolo2_backup.ckpt      model/tiny_yolo2.ckpt
mv model/tiny_yolo2_backup.ckpt.meta model/tiny_yolo2.ckpt.meta
mv model/tiny_yolo2_backup.pb        model/tiny_yolo2.pb
```

## Pythonを使った推論
Qumicoを使用してC言語に変換する前に、学習したモデルが正しく推論できるかを確認します。  
トレーニングデータを使った学習を行った方は、tiny_yolo2_train.py の root_pathに設定したパスを、tiny_yolo2_infter.py の root_path にも設定してください。  
学習済みデータを使った方は、tiny_yolo2_train.py の root_path に "data" を指定してください。  


その後、次のコマンドを実行してください。
```sh
python3 tiny_yolo2_infer.py
```
Tiny YOLOv2の推論結果として、次のように表示されれば成功です。  

![still_life](sample_tiny_yolo_v2_infer.png)


## Cソースへの変換
上で生成したonnxファイルを使用し、ニューラルネットワークをCソースに変換します。

```sh
python3 gen_c.py 
```
このように表示されば、正常に終了しています。
```
[1, 3, 416, 416]
[1, 125, 13, 13]
Cソースを生成しました。出力先: out_c
```
out_cディレクトリに、includeとlibディレクトリ、qumico.cとqumico.soが出力されていれば、Cソースへの変換は成功です。

## C言語での推論実行
上で生成した共有ライブラリqumico.soを使って推論を実行します。
```sh
python3 tiny_yolo2_infer_c.py 
```
Pythonを使った推論と同じ結果が表示されれば、C言語で正しく推論ができています。


