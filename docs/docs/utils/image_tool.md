
# image_tool.py

## resize_image_array_set
リスト中の画像をリサイズする。

### 引数
- image_sets: リサイズする画像リスト
- w_in: リサイズする画像の横幅
- h_in: リサイズする画像の縦幅
- input_mode: 入力画像のチャネル順を指定する。(default='RGB')
- resize: リサイズ有無(default=False)
- w_resize: リサイズ後の横幅
- h_resize: リサイズ後の縦幅
- channel_out: リサイズ後のチャネル数(default=1)

### 戻り値
リサイズした画像リスト
 
