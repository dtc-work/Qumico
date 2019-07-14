import numpy as np
from PIL import Image

def resize_image_array_set(image_sets, w_in, h_in, input_mode='RGB', resize=False, w_resize=200, h_resize=200,
                           channel_out=1):
    """
    リスト中の画像をリサイズする。
    #### 引数
    - image_sets: リサイズする画像リスト
    - w_in: リサイズする画像の横幅
    - h_in: リサイズする画像の縦幅
    - input_mode: 入力画像のチャネル順を指定する。(default='RGB')
    - resize: リサイズ有無(default=False)
    - w_resize: リサイズ後の横幅
    - h_resize: リサイズ後の縦幅
    - channel_out: リサイズ後のチャネル数(default=1)
    #### 戻り値
    リサイズした画像リスト
    """
    new_shape = (image_sets.shape[0], h_resize, w_resize, channel_out)
    output = np.empty(new_shape)
    for index, image in enumerate(image_sets):
        resize_image = resize_image_array(image, w_in, h_in, input_mode, resize, w_resize, h_resize, channel_out)
        output[index] = resize_image
    return output


def resize_image_array(image_array, w_in, h_in, input_mode='RGB', resize=False, w_resize=200, h_resize=200,
                       channel_out=1):
    image = image_encode(image_array, W=w_in, H=h_in, MODE=input_mode)
    if resize:
        img_resize = image.resize((w_resize, h_resize), Image.LANCZOS)
        # img_resize.show()
        image = img_resize

    output_array = image_decode(image, input_mode='RGB', channel_out=channel_out)

    return output_array


def image_encode(image_array, W=32, H=32, MODE='RGB'):
    image_reshape = image_array
    if MODE is 'RGB':
        if np.ndim(image_array) == 1:
            image_reshape = np.reshape(image_array, newshape=[3, W, H]).transpose([1, 2, 0])

    image = Image.fromarray(image_reshape, mode=MODE)
    return image


def image_decode(image, input_mode='RGB', channel_out=1):
    image_array = np.asarray(image)
    # if input_mode is 'RGB':
    #     image_array = image_array.transpose([2, 0, 1])

    image_out = image_array
    return image_out
