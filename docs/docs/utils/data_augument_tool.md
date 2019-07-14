# data_augument_tool.py

## RandomHorizontalFlip
Randomly horizontally flips the Image with the probability *p* Parameters

- p: float

The probability with which the image is flipped Returns

- numpy.ndaaray

Flipped image in the numpy format of shape `HxWxC`

- numpy.ndarray

Tranformed bounding box co-ordinates of the format `n x 4` where n is number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
<br>
<br>

## HorizontalFlip
Randomly horizontally flips the Image with the probability *p* Parameters

- p: float

The probability with which the image is flipped Returns

- numpy.ndaaray

Flipped image in the numpy format of shape `HxWxC`

- numpy.ndarray

Tranformed bounding box co-ordinates of the format `n x 4` where n is number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
<br>
<br>

## RandomScale
Randomly scales an image    

Bounding boxes which have an area of less than 25% in the remaining in the transformed image is dropped. The resolution is maintained, and the remaining area if any is filled by black color.

### Parameters
- scale: float or tuple(float)

if **float**, the image is scaled by a factor drawn randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**, the `scale` is drawn randomly from values specified by the tuple

### Returns
- numpy.ndaaray

Scaled image in the numpy format of shape `HxWxC`

- numpy.ndarray

Tranformed bounding box co-ordinates of the format `n x 4` where n is number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
<br>
<br>

## Scale
Scales the image    

Bounding boxes which have an area of less than 25% in the remaining in the transformed image is dropped. The resolution is maintained, and the remaining area if any is filled by black color.

### Parameters
- scale_x: float

The factor by which the image is scaled horizontally

- scale_y: float

The factor by which the image is scaled vertically

### Returns
- numpy.ndaaray

Scaled image in the numpy format of shape `HxWxC`

- numpy.ndarray

Tranformed bounding box co-ordinates of the format `n x 4` where n is number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
<br>
<br>

## RandomTranslate
Randomly Translates the image    

Bounding boxes which have an area of less than 25% in the remaining in the  transformed image is dropped. The resolution is maintained, and the remaining area if any is filled by black color.

### Parameters
- translate: float or tuple(float)

if **float**, the image is translated by a factor drawn randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**, `translate` is drawn randomly from values specified by the tuple

### Returns
- numpy.ndaaray

Translated image in the numpy format of shape `HxWxC`

- numpy.ndarray

Tranformed bounding box co-ordinates of the format `n x 4` where n is number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
<br>
<br>

## Translate
Randomly Translates the image    

Bounding boxes which have an area of less than 25% in the remaining in the  transformed image is dropped. The resolution is maintained, and the remaining area if any is filled by black color.

### Parameters
- translate: float or tuple(float)

if **float**, the image is translated by a factor drawn randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**, `translate` is drawn randomly from values specified by the tuple

### Returns
- numpy.ndaaray

Translated image in the numpy format of shape `HxWxC`

- numpy.ndarray

Tranformed bounding box co-ordinates of the format `n x 4` where n is number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
<br>
<br>

## RandomRotate
Randomly rotates an image    

Bounding boxes which have an area of less than 25% in the remaining in the  transformed image is dropped. The resolution is maintained, and the remaining area if any is filled by black color.

### Parameters
- angle: float or tuple(float)

if **float**, the image is rotated by a factor drawn randomly from a range (-`angle`, `angle`). If **tuple**, the `angle` is drawn randomly from values specified by the tuple

### Returns
- numpy.ndaaray

Rotated image in the numpy format of shape `HxWxC`

- numpy.ndarray

Tranformed bounding box co-ordinates of the format `n x 4` where n is number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
<br>
<br>

## Rotate
Rotates an image    

Bounding boxes which have an area of less than 25% in the remaining in the transformed image is dropped. The resolution is maintained, and the remaining area if any is filled by black color.

### Parameters
= angle: float

The angle by which the image is to be rotated 

### Returns
- numpy.ndaaray

Rotated image in the numpy format of shape `HxWxC`

- numpy.ndarray

Tranformed bounding box co-ordinates of the format `n x 4` where n is number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

### __call__
- Args:

img (PIL Image): Image to be flipped.

- Returns:

PIL Image: Randomly flipped image.
<br>
<br>

## RandomShear
Randomly shears an image in horizontal direction   

Bounding boxes which have an area of less than 25% in the remaining in the transformed image is dropped. The resolution is maintained, and the remaining area if any is filled by black color.

### Parameters
- shear_factor: float or tuple(float)

if **float**, the image is sheared horizontally by a factor drawn randomly from a range (-`shear_factor`, `shear_factor`). If **tuple**, the `shear_factor` is drawn randomly from values specified by the tuple

### Returns
- numpy.ndaaray

Sheared image in the numpy format of shape `HxWxC`

- numpy.ndarray

Tranformed bounding box co-ordinates of the format `n x 4` where n is number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
<br>
<br>

## Shear
Shears an image in horizontal direction   

Bounding boxes which have an area of less than 25% in the remaining in the transformed image is dropped. The resolution is maintained, and the remaining area if any is filled by black color.

### Parameters
- shear_factor: float

Factor by which the image is sheared in the x-direction

### Returns
- numpy.ndaaray

Sheared image in the numpy format of shape `HxWxC`

- numpy.ndarray

Tranformed bounding box co-ordinates of the format `n x 4` where n is number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
<br>
<br>

## Resize
Resize the image in accordance to `image_letter_box` function in darknet 

The aspect ratio is maintained. The longer side is resized to the input size of the network, while the remaining space on the shorter side is filled with black color. **This should be the last transform**

### Parameters
-inp_dim : tuple(int)

tuple containing the size to which the image will be resized.

### Returns
- numpy.ndaaray

Sheared image in the numpy format of shape `HxWxC`

- numpy.ndarray

Resized bounding box co-ordinates of the format `n x 4` where n is number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
<br>
<br>

## RandomHSV
HSV Transform to vary hue saturation and brightness
- Hue has a range of 0-179
- Saturation and Brightness have a range of 0-255. 
- Chose the amount you want to change thhe above quantities accordingly. 

### Parameters
- hue : None or int or tuple (int)

If None, the hue of the image is left unchanged. If int, a random int is uniformly sampled from (-hue, hue) and added to the hue of the image. If tuple, the int is sampled from the range specified by the tuple.   

- saturation : None or int or tuple(int)

If None, the saturation of the image is left unchanged. If int, a random int is uniformly sampled from (-saturation, saturation) and added to the hue of the image. If tuple, the int is sampled from the range  specified by the tuple.

- brightness : None or int or tuple(int)

If None, the brightness of the image is left unchanged. If int, a random int is uniformly sampled from (-brightness, brightness) and added to the hue of the image. If tuple, the int is sampled from the range  specified by the tuple.

### Returns
- numpy.ndaaray

Transformed image in the numpy format of shape `HxWxC`

- numpy.ndarray

Resized bounding box co-ordinates of the format `n x 4` where n is number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
<br>
<br>

## Sequence
Initialise Sequence object. Apply a Sequence of transformations to the images/boxes.

### Parameters
- augemnetations : list 

List containing Transformation Objects in Sequence they are to be applied

- probs : int or list 

If **int**, the probability with which each of the transformation will be applied. If **list**, the length must be equal to *augmentations*. Each element of this list is the probability with which each corresponding transformation is applied

### Returns
- Sequence
- Sequence Object 
