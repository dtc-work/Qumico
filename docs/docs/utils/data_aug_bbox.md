# data_aug_bbox.py

## clip_box
Clip the bounding boxes to the borders of an image

### Parameters

- bbox: numpy.ndarray

Numpy array containing bounding boxes of shape `N X 4` where N is the number of bounding boxes and the bounding boxes are represented in the format `x1 y1 x2 y2`

- clip_box: numpy.ndarray

An array of shape (4,) specifying the diagonal co-ordinates of the image The coordinates are represented in the format `x1 y1 x2 y2`

- alpha: float

If the fraction of a bounding box left in the image after being clipped is less than `alpha` the bounding box is dropped. 

### Returns
- numpy.ndarray

Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the number of bounding boxes left are being clipped and the bounding boxes are represented in the format `x1 y1 x2 y2` 
<br>
<br>

## draw_rect
Draw the rectangle on the image

### Parameters
- im : numpy.ndarray

numpy image 

- cords: numpy.ndarray

Numpy array containing bounding boxes of shape `N X 4` where N is the number of bounding boxes and the bounding boxes are represented in the format `x1 y1 x2 y2`

### Returns
- numpy.ndarray

numpy image with bounding boxes drawn on it
<br>
<br>

## get_corners
Get corners of bounding boxes

### Parameters

- bboxes: numpy.ndarray

Numpy array containing bounding boxes of shape `N X 4` where N is the number of bounding boxes and the bounding boxes are represented in the format `x1 y1 x2 y2`

### returns
- numpy.ndarray

Numpy array of shape `N x 8` containing N bounding boxes each described by their corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
<br>

## get_enclosing_box
Get an enclosing box for ratated corners of a bounding box

### Parameters
- corners : numpy.ndarray

Numpy array of shape `N x 8` containing N bounding boxes each described by their corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  

### Returns 
- numpy.ndarray

Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the number of bounding boxes and the bounding boxes are represented in the format `x1 y1 x2 y2`
<br>
<br>
  
## letterbox_image
resize image with unchanged aspect ratio using padding

### Parameters
-img : numpy.ndarray

Image 

- inp_dim: tuple(int)

shape of the reszied image

### Returns
- numpy.ndarray:

Resized image
<br>
<br>
  
## rotate_box
Rotate the bounding box.


### Parameters

- corners : numpy.ndarray

Numpy array of shape `N x 8` containing N bounding boxes each described by their corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

- angle : float

angle by which the image is to be rotated

- cx : int

x coordinate of the center of image (about which the box will be rotated)

- cy : int

y coordinate of the center of image (about which the box will be rotated)

- h : int 

height of the image

- w : int 

width of the image

### Returns
- numpy.ndarray

Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
<br>  
<br>
  
## rotate_im
- Rotate the image.

Rotate the image such that the rotated image is enclosed inside the tightest rectangle. The area not occupied by the pixels of the original image is colored black. 

### Parameters

- image : numpy.ndarray

numpy image

- angle : float

angle by which the image is to be rotated

### Returns
- numpy.ndarray

Rotated Image
 
