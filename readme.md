
![Qumico](/docs/logo.jpg)


## NEW!
Google Colaboratoryを利用して、各サンプルを簡単に試せるチュートリアルを準備しました。

- [mlp](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/blob/master/samples/mlp/tensorflow/mlp_colab.ipynb)：MultiLayer Perceptron
- [conv](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/blob/master/samples/conv/tensorflow/conv_colab.ipynb)：Convolution
- [vgg](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/blob/master/samples/vgg16/keras/vgg16_colab.ipynb)：VGG16
- [tiny_yolo_v2](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/tree/master/samples/tiny_yolo_v2/tensorflow/tiny_yolo2_colab.ipynb)：tiny_yolo_v2
- [tiny_yolo_v2_yad2k](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/tree//samples/tiny_yolo_v2_yad2k/keras/tiny_yolo_v2_yad2k_colab.ipynb)：tiny_yolo_v2(推論のみ)
- [mobilenet](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/tree/master/samples/mobilenet/tensorflow/mobilenet_colab.ipynb)：Mobilenet V1
- [automl_dogcat](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/tree/master/samples/automl_dogcat/tensorflow/DogCat_colab.ipynb)：AutoMLを利用した犬猫の識別 



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
6. Google Cloud AutoMLに対応。TensorFlow LiteフォーマットからのC言語生成をサポート


## インストール方法
インストール方法は[Qumicoのインストール文書](/docs/docs/install/install.md)を参考して、順番通りインストールします。


## 内容

- [qumico](/qumico)：Qumicoアプリケーション

- [sample](/samples)：Qumico用サンプル

- [docs](/docs)：Qumico技術文書

- [tests](/tests)：Qumicoテスト

- [tools](/tools)：Qumico周辺ツール


## サンプル
- [mlp](/docs/docs/samples/mlp.md)：MultiLayer Perceptronサンプル - [colab](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/blob/master/samples/mlp/tensorflow/mlp_colab.ipynb)：colabサンプル
- [conv](/docs/docs/samples/conv.md)：Convolutionサンプル - [colab](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/blob/master/samples/conv/tensorflow/conv_colab.ipynb)：colabサンプル
- [vgg](/docs/docs/samples/vgg16.md)：VGG16サンプル - [colab](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/blob/master/samples/vgg16/keras/vgg16_colab.ipynb)：colabサンプル
- [tiny_yolo_v2](/docs/docs/samples/tiny_yolo.md)：tiny_yolo_v2サンプル(要CUDA, モデル学習あり) - [colab](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/tree/master/samples/tiny_yolo_v2/tensorflow/tiny_yolo2_colab.ipynb)：colabサンプル
- [tiny_yolo_v2_yad2k](/docs/docs/samples/tiny_yolo_yad2k.md)：tiny_yolo_v2_yad2k(推論のみ, RapsberryPi向け) - [colab](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/tree//samples/tiny_yolo_v2_yad2k/keras/tiny_yolo_v2_yad2k_colab.ipynb)：colabサンプル
- [mobilenet](/docs/docs/samples/mobilenet.md)：Mobilenet V1(量子化済み, TFLite形式) - [colab](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/tree/master/samples/mobilenet/tensorflow/mobilenet_colab.ipynb)：colabサンプル
- [automl_dogcat](/docs/docs/samples/automl_dogcat.md)：AutoMLを利用した識別 - [colab](https://colab.research.google.com/github.com/PasonaTech-Inc/Qumico/tree/master/samples/automl_dogcat/tensorflow/DogCat_colab.ipynb)：colabサンプル


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
