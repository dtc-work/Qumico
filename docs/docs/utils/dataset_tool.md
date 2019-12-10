
# DatasetTool.py

## DatasetTool
バッチ実行を管理するクラス。

### 使用例

```python
    # DatasetToolのインスタンス化を行います。
    q = DatasetTool()
    # ｋ回のバッチ実行を行います。
    q.next_batch(self, k)
```

## index_reset
バッチ実行リストの初期化を行う。

### 引数
なし

### 戻り値
なし
<br>
<br>

## next_batch
ランダム順にバッチ学習を行う。

DatasetToolクラスのインスタンシエーション時にrepeat=True設定をしていれば重複あり、repeat=Falseを設定していれば重複なしで、バッチ実行データを選択する。

### 引数
- batch_size: バッチサイズ

### 戻り値
バッチ実行回数
<br>
<br>

## next_batch_once
あらかじめ決めた順にバッチ学習を行う。

### 引数
- batch_size: バッチサイズ

### 戻り値
バッチ実行回数

