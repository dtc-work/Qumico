
![Qumico](/docs/logo.jpg)


## 概要
`Qumico`(クミコ)は、パソナテックが独自開発したフレームワークです。<br>
IoT、ロボット、AI家電等、エッジ機器でディープラーニングを動作させることが可能です。<br>
特別なハードウェアがなくてもお手持ちのハードウェアでディープラーニングを動作させることができます。<br>


## 特徴
1. ディープラーニングの学習には、TensorFlow、Keras等の業界標準のフレームワークを使用可能
2. 学習結果を早い段階で組込機器へ実装できるため、実機でのスピーディーな性能評価が可能
3. SoCやカスタムCPUといった業界標準フレームワークが対応していないボードでも、エッジAIを動作させることが可能
4. 業界標準のONNXフォーマットから組込用Cソースを生成、組込機器でディープラーニング動作
5. C言語をベースに、様々な最適化を実施可能


## インストール方法
インストール方法は[Qumicoのインストール文書](/docs/docs/install/install.md)を参考して、順番通りインストールします。


## 内容

- [qumico](/qumico)：Qumicoアプリケーション

- [sample](/samples)：Qumico用サンプル

- [docs](/docs)：Qumico技術文書

- [tests](/tests)：Qumicoテスト

- [tools](/tools)：Qumico周辺ツール


## レファレンス
- GitHubサイト
    - [ONNX/GitHubサイト](https://github.com/onnx/)
    - [Tenserflow/GitHubサイト](https://github.com/tensorflow)
    - [Keras/GitHubサイト](https://github.com/keras-team/keras)
- 公式サイト
    - [ONNX](http://onnx.ai/)
    - [Tenserflow](https://www.tensorflow.org/)
    - [Keras](https://keras.io/)


## ドキュメントを作成

mkdocsを使ってドキュメントを作る方法です。

まずはmkdocsをインストールしましょう。

```
pip install mkdocs
pip install mkdocs-material
pip install pygments
```

次にgit cloneしたディレクトリの下のdocsへ移動し、mkdocsをつかってドキュメントを作成してください。

```
cd docs
mkdocs build
```

siteディレクトリにindex.htmlが作成されるので、そこからブラウザでドキュメントを見てください。
<br>

---

##### [Pasona Tech, Inc. @2019](https://pasona.tech/)
