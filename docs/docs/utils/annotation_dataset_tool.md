# annotation_dataset_tool.py

## AnnotationDatasetTool

## image_generator

画像の生成を行う。
#### 引数
- full_path: 入力する画像ファイルのパス
- target_h: 出力画像サイズ(縦)
- target_w: 出力画像サイズ(横)
- rescale: リサイズ
- is_opencv: Trueの場合OpenCVから読み込み、Falseの場合画像ファイルから読み込みを行う(default=True)
- format: 出力画像tensorのディメンジョン順(default=NHWC: バッチ数/縦/横/チャネル)
- histogram: 現在未使用(default=True)
#### 戻り値
画像データ、リサイズ比率(縦)、リサイズ比率(横)  のリスト
<br>
<br>

## index_reset

バッチ実行リストの初期化を行う。
#### 引数
なし
#### 戻り値
なし
<br>
<br>

## next_batch

ランダム順にバッチ学習を行う。
DatasetToolクラスのインスタンシエーション時にrepeat=True設定をしていれば重複あり、repeat=Falseを設定していれば重複なしで、バッチ実行データを選択する。

#### 引数
- batch_size: バッチサイズ
#### 戻り値
バッチ実行回数
<br>
<br>

## next_batch_once

あらかじめ決めた順にバッチ学習を行う。
#### 引数
- batch_size: バッチサイズ
#### 戻り値
バッチ実行回数


