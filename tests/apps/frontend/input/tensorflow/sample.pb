
@
inputPlaceholder*
dtype0*
shape:’’’’’’’’’
?
validPlaceholder*
dtype0*
shape:’’’’’’’’’

J
reshape/shapeConst*%
valueB"’’’’         *
dtype0
?
reshapeReshapeinputreshape/shape*
T0*
Tshape0
Y
conv1/truncated_normal/shapeConst*%
valueB"             *
dtype0
H
conv1/truncated_normal/meanConst*
valueB
 *    *
dtype0
J
conv1/truncated_normal/stddevConst*
valueB
 *ĶĢĢ=*
dtype0

&conv1/truncated_normal/TruncatedNormalTruncatedNormalconv1/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 
q
conv1/truncated_normal/mulMul&conv1/truncated_normal/TruncatedNormalconv1/truncated_normal/stddev*
T0
_
conv1/truncated_normalAddconv1/truncated_normal/mulconv1/truncated_normal/mean*
T0
c
W_conv1
VariableV2*
dtype0*
	container *
shape: *
shared_name 

W_conv1/AssignAssignW_conv1conv1/truncated_normal*
validate_shape(*
use_locking(*
T0*
_class
loc:@W_conv1
F
W_conv1/readIdentityW_conv1*
T0*
_class
loc:@W_conv1
<
conv1/ConstConst*
valueB *ĶĢĢ=*
dtype0
W
b_conv1
VariableV2*
shape: *
shared_name *
dtype0*
	container 
|
b_conv1/AssignAssignb_conv1conv1/Const*
use_locking(*
T0*
_class
loc:@b_conv1*
validate_shape(
F
b_conv1/readIdentityb_conv1*
T0*
_class
loc:@b_conv1
¤
conv1/Conv2DConv2DreshapeW_conv1/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
T
conv1/BiasAddBiasAddconv1/Conv2Db_conv1/read*
T0*
data_formatNHWC
%
conv1Reluconv1/BiasAdd*
T0
t
p_pool1MaxPoolconv1*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
Y
conv2/truncated_normal/shapeConst*%
valueB"          @   *
dtype0
H
conv2/truncated_normal/meanConst*
valueB
 *    *
dtype0
J
conv2/truncated_normal/stddevConst*
dtype0*
valueB
 *ĶĢĢ=

&conv2/truncated_normal/TruncatedNormalTruncatedNormalconv2/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 
q
conv2/truncated_normal/mulMul&conv2/truncated_normal/TruncatedNormalconv2/truncated_normal/stddev*
T0
_
conv2/truncated_normalAddconv2/truncated_normal/mulconv2/truncated_normal/mean*
T0
c
W_conv2
VariableV2*
shared_name *
dtype0*
	container *
shape: @

W_conv2/AssignAssignW_conv2conv2/truncated_normal*
validate_shape(*
use_locking(*
T0*
_class
loc:@W_conv2
F
W_conv2/readIdentityW_conv2*
T0*
_class
loc:@W_conv2
<
conv2/ConstConst*
dtype0*
valueB@*ĶĢĢ=
W
b_conv2
VariableV2*
dtype0*
	container *
shape:@*
shared_name 
|
b_conv2/AssignAssignb_conv2conv2/Const*
T0*
_class
loc:@b_conv2*
validate_shape(*
use_locking(
F
b_conv2/readIdentityb_conv2*
T0*
_class
loc:@b_conv2
¤
conv2/Conv2DConv2Dp_pool1W_conv2/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
T
conv2/BiasAddBiasAddconv2/Conv2Db_conv2/read*
T0*
data_formatNHWC
%
conv2Reluconv2/BiasAdd*
T0
t
p_pool2MaxPoolconv2*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*
T0
B
flatten/shapeConst*
valueB"’’’’@  *
dtype0
A
flattenReshapep_pool2flatten/shape*
T0*
Tshape0
O
fc1/truncated_normal/shapeConst*
valueB"@     *
dtype0
F
fc1/truncated_normal/meanConst*
valueB
 *    *
dtype0
H
fc1/truncated_normal/stddevConst*
dtype0*
valueB
 *ĶĢĢ=

$fc1/truncated_normal/TruncatedNormalTruncatedNormalfc1/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 
k
fc1/truncated_normal/mulMul$fc1/truncated_normal/TruncatedNormalfc1/truncated_normal/stddev*
T0
Y
fc1/truncated_normalAddfc1/truncated_normal/mulfc1/truncated_normal/mean*
T0
[
W_fc1
VariableV2*
shape:
Ą*
shared_name *
dtype0*
	container 

W_fc1/AssignAssignW_fc1fc1/truncated_normal*
validate_shape(*
use_locking(*
T0*
_class

loc:@W_fc1
@

W_fc1/readIdentityW_fc1*
T0*
_class

loc:@W_fc1
;
	fc1/ConstConst*
dtype0*
valueB*ĶĢĢ=
V
b_fc1
VariableV2*
shared_name *
dtype0*
	container *
shape:
t
b_fc1/AssignAssignb_fc1	fc1/Const*
use_locking(*
T0*
_class

loc:@b_fc1*
validate_shape(
@

b_fc1/readIdentityb_fc1*
T0*
_class

loc:@b_fc1
X

fc1/MatMulMatMulflatten
W_fc1/read*
T0*
transpose_a( *
transpose_b( 
/
fc1/AddAdd
fc1/MatMul
b_fc1/read*
T0

fc1Relufc1/Add*
T0
:
dropout1/rateConst*
valueB
 *   ?*
dtype0
5
dropout1/ShapeShapefc1*
T0*
out_type0
;
dropout1/sub/xConst*
valueB
 *  ?*
dtype0
;
dropout1/subSubdropout1/sub/xdropout1/rate*
T0
H
dropout1/random_uniform/minConst*
valueB
 *    *
dtype0
H
dropout1/random_uniform/maxConst*
dtype0*
valueB
 *  ?
u
%dropout1/random_uniform/RandomUniformRandomUniformdropout1/Shape*
T0*
dtype0*
seed2 *

seed 
e
dropout1/random_uniform/subSubdropout1/random_uniform/maxdropout1/random_uniform/min*
T0
o
dropout1/random_uniform/mulMul%dropout1/random_uniform/RandomUniformdropout1/random_uniform/sub*
T0
a
dropout1/random_uniformAdddropout1/random_uniform/muldropout1/random_uniform/min*
T0
C
dropout1/addAdddropout1/subdropout1/random_uniform*
T0
.
dropout1/FloorFloordropout1/add*
T0
7
dropout1/truedivRealDivfc1dropout1/sub*
T0
>
dropout1/mulMuldropout1/truedivdropout1/Floor*
T0
O
fc2/truncated_normal/shapeConst*
valueB"   
   *
dtype0
F
fc2/truncated_normal/meanConst*
valueB
 *    *
dtype0
H
fc2/truncated_normal/stddevConst*
valueB
 *ĶĢĢ=*
dtype0

$fc2/truncated_normal/TruncatedNormalTruncatedNormalfc2/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 
k
fc2/truncated_normal/mulMul$fc2/truncated_normal/TruncatedNormalfc2/truncated_normal/stddev*
T0
Y
fc2/truncated_normalAddfc2/truncated_normal/mulfc2/truncated_normal/mean*
T0
Z
W_fc2
VariableV2*
shared_name *
dtype0*
	container *
shape:	


W_fc2/AssignAssignW_fc2fc2/truncated_normal*
T0*
_class

loc:@W_fc2*
validate_shape(*
use_locking(
@

W_fc2/readIdentityW_fc2*
T0*
_class

loc:@W_fc2
:
	fc2/ConstConst*
valueB
*ĶĢĢ=*
dtype0
U
b_fc2
VariableV2*
shape:
*
shared_name *
dtype0*
	container 
t
b_fc2/AssignAssignb_fc2	fc2/Const*
T0*
_class

loc:@b_fc2*
validate_shape(*
use_locking(
@

b_fc2/readIdentityb_fc2*
T0*
_class

loc:@b_fc2
]

fc2/MatMulMatMuldropout1/mul
W_fc2/read*
transpose_a( *
transpose_b( *
T0
/
fc2/AddAdd
fc2/MatMul
b_fc2/read*
T0

fc2Relufc2/Add*
T0

outputSoftmaxfc2*
T0
!
	train/LogLogoutput*
T0
+
	train/mulMulvalid	train/Log*
T0
@
train/ConstConst*
valueB"       *
dtype0
N
	train/SumSum	train/multrain/Const*

Tidx0*
	keep_dims( *
T0
$
	train/NegNeg	train/Sum*
T0
>
train/gradients/ShapeConst*
valueB *
dtype0
F
train/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0
i
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0
H
"train/gradients/train/Neg_grad/NegNegtrain/gradients/Fill*
T0
a
,train/gradients/train/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0

&train/gradients/train/Sum_grad/ReshapeReshape"train/gradients/train/Neg_grad/Neg,train/gradients/train/Sum_grad/Reshape/shape*
T0*
Tshape0
Q
$train/gradients/train/Sum_grad/ShapeShape	train/mul*
T0*
out_type0

#train/gradients/train/Sum_grad/TileTile&train/gradients/train/Sum_grad/Reshape$train/gradients/train/Sum_grad/Shape*
T0*

Tmultiples0
M
$train/gradients/train/mul_grad/ShapeShapevalid*
T0*
out_type0
S
&train/gradients/train/mul_grad/Shape_1Shape	train/Log*
T0*
out_type0
¤
4train/gradients/train/mul_grad/BroadcastGradientArgsBroadcastGradientArgs$train/gradients/train/mul_grad/Shape&train/gradients/train/mul_grad/Shape_1*
T0
b
"train/gradients/train/mul_grad/MulMul#train/gradients/train/Sum_grad/Tile	train/Log*
T0
©
"train/gradients/train/mul_grad/SumSum"train/gradients/train/mul_grad/Mul4train/gradients/train/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 

&train/gradients/train/mul_grad/ReshapeReshape"train/gradients/train/mul_grad/Sum$train/gradients/train/mul_grad/Shape*
T0*
Tshape0
`
$train/gradients/train/mul_grad/Mul_1Mulvalid#train/gradients/train/Sum_grad/Tile*
T0
Æ
$train/gradients/train/mul_grad/Sum_1Sum$train/gradients/train/mul_grad/Mul_16train/gradients/train/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 

(train/gradients/train/mul_grad/Reshape_1Reshape$train/gradients/train/mul_grad/Sum_1&train/gradients/train/mul_grad/Shape_1*
T0*
Tshape0

/train/gradients/train/mul_grad/tuple/group_depsNoOp'^train/gradients/train/mul_grad/Reshape)^train/gradients/train/mul_grad/Reshape_1
į
7train/gradients/train/mul_grad/tuple/control_dependencyIdentity&train/gradients/train/mul_grad/Reshape0^train/gradients/train/mul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@train/gradients/train/mul_grad/Reshape
ē
9train/gradients/train/mul_grad/tuple/control_dependency_1Identity(train/gradients/train/mul_grad/Reshape_10^train/gradients/train/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/train/mul_grad/Reshape_1

)train/gradients/train/Log_grad/Reciprocal
Reciprocaloutput:^train/gradients/train/mul_grad/tuple/control_dependency_1*
T0

"train/gradients/train/Log_grad/mulMul9train/gradients/train/mul_grad/tuple/control_dependency_1)train/gradients/train/Log_grad/Reciprocal*
T0
[
train/gradients/output_grad/mulMul"train/gradients/train/Log_grad/muloutput*
T0
d
1train/gradients/output_grad/Sum/reduction_indicesConst*
valueB :
’’’’’’’’’*
dtype0
 
train/gradients/output_grad/SumSumtrain/gradients/output_grad/mul1train/gradients/output_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims(
t
train/gradients/output_grad/subSub"train/gradients/train/Log_grad/multrain/gradients/output_grad/Sum*
T0
Z
!train/gradients/output_grad/mul_1Multrain/gradients/output_grad/suboutput*
T0
^
!train/gradients/fc2_grad/ReluGradReluGrad!train/gradients/output_grad/mul_1fc2*
T0
P
"train/gradients/fc2/Add_grad/ShapeShape
fc2/MatMul*
T0*
out_type0
R
$train/gradients/fc2/Add_grad/Shape_1Const*
dtype0*
valueB:


2train/gradients/fc2/Add_grad/BroadcastGradientArgsBroadcastGradientArgs"train/gradients/fc2/Add_grad/Shape$train/gradients/fc2/Add_grad/Shape_1*
T0
¤
 train/gradients/fc2/Add_grad/SumSum!train/gradients/fc2_grad/ReluGrad2train/gradients/fc2/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

$train/gradients/fc2/Add_grad/ReshapeReshape train/gradients/fc2/Add_grad/Sum"train/gradients/fc2/Add_grad/Shape*
T0*
Tshape0
Ø
"train/gradients/fc2/Add_grad/Sum_1Sum!train/gradients/fc2_grad/ReluGrad4train/gradients/fc2/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

&train/gradients/fc2/Add_grad/Reshape_1Reshape"train/gradients/fc2/Add_grad/Sum_1$train/gradients/fc2/Add_grad/Shape_1*
T0*
Tshape0

-train/gradients/fc2/Add_grad/tuple/group_depsNoOp%^train/gradients/fc2/Add_grad/Reshape'^train/gradients/fc2/Add_grad/Reshape_1
Ł
5train/gradients/fc2/Add_grad/tuple/control_dependencyIdentity$train/gradients/fc2/Add_grad/Reshape.^train/gradients/fc2/Add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@train/gradients/fc2/Add_grad/Reshape
ß
7train/gradients/fc2/Add_grad/tuple/control_dependency_1Identity&train/gradients/fc2/Add_grad/Reshape_1.^train/gradients/fc2/Add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@train/gradients/fc2/Add_grad/Reshape_1
¢
&train/gradients/fc2/MatMul_grad/MatMulMatMul5train/gradients/fc2/Add_grad/tuple/control_dependency
W_fc2/read*
T0*
transpose_a( *
transpose_b(
¦
(train/gradients/fc2/MatMul_grad/MatMul_1MatMuldropout1/mul5train/gradients/fc2/Add_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 

0train/gradients/fc2/MatMul_grad/tuple/group_depsNoOp'^train/gradients/fc2/MatMul_grad/MatMul)^train/gradients/fc2/MatMul_grad/MatMul_1
ć
8train/gradients/fc2/MatMul_grad/tuple/control_dependencyIdentity&train/gradients/fc2/MatMul_grad/MatMul1^train/gradients/fc2/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@train/gradients/fc2/MatMul_grad/MatMul
é
:train/gradients/fc2/MatMul_grad/tuple/control_dependency_1Identity(train/gradients/fc2/MatMul_grad/MatMul_11^train/gradients/fc2/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/fc2/MatMul_grad/MatMul_1
[
'train/gradients/dropout1/mul_grad/ShapeShapedropout1/truediv*
T0*
out_type0
[
)train/gradients/dropout1/mul_grad/Shape_1Shapedropout1/Floor*
T0*
out_type0
­
7train/gradients/dropout1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs'train/gradients/dropout1/mul_grad/Shape)train/gradients/dropout1/mul_grad/Shape_1*
T0

%train/gradients/dropout1/mul_grad/MulMul8train/gradients/fc2/MatMul_grad/tuple/control_dependencydropout1/Floor*
T0
²
%train/gradients/dropout1/mul_grad/SumSum%train/gradients/dropout1/mul_grad/Mul7train/gradients/dropout1/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 

)train/gradients/dropout1/mul_grad/ReshapeReshape%train/gradients/dropout1/mul_grad/Sum'train/gradients/dropout1/mul_grad/Shape*
T0*
Tshape0

'train/gradients/dropout1/mul_grad/Mul_1Muldropout1/truediv8train/gradients/fc2/MatMul_grad/tuple/control_dependency*
T0
ø
'train/gradients/dropout1/mul_grad/Sum_1Sum'train/gradients/dropout1/mul_grad/Mul_19train/gradients/dropout1/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
”
+train/gradients/dropout1/mul_grad/Reshape_1Reshape'train/gradients/dropout1/mul_grad/Sum_1)train/gradients/dropout1/mul_grad/Shape_1*
T0*
Tshape0

2train/gradients/dropout1/mul_grad/tuple/group_depsNoOp*^train/gradients/dropout1/mul_grad/Reshape,^train/gradients/dropout1/mul_grad/Reshape_1
ķ
:train/gradients/dropout1/mul_grad/tuple/control_dependencyIdentity)train/gradients/dropout1/mul_grad/Reshape3^train/gradients/dropout1/mul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/dropout1/mul_grad/Reshape
ó
<train/gradients/dropout1/mul_grad/tuple/control_dependency_1Identity+train/gradients/dropout1/mul_grad/Reshape_13^train/gradients/dropout1/mul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/dropout1/mul_grad/Reshape_1
R
+train/gradients/dropout1/truediv_grad/ShapeShapefc1*
T0*
out_type0
V
-train/gradients/dropout1/truediv_grad/Shape_1Const*
valueB *
dtype0
¹
;train/gradients/dropout1/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs+train/gradients/dropout1/truediv_grad/Shape-train/gradients/dropout1/truediv_grad/Shape_1*
T0

-train/gradients/dropout1/truediv_grad/RealDivRealDiv:train/gradients/dropout1/mul_grad/tuple/control_dependencydropout1/sub*
T0
Ā
)train/gradients/dropout1/truediv_grad/SumSum-train/gradients/dropout1/truediv_grad/RealDiv;train/gradients/dropout1/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
-train/gradients/dropout1/truediv_grad/ReshapeReshape)train/gradients/dropout1/truediv_grad/Sum+train/gradients/dropout1/truediv_grad/Shape*
T0*
Tshape0
>
)train/gradients/dropout1/truediv_grad/NegNegfc1*
T0
|
/train/gradients/dropout1/truediv_grad/RealDiv_1RealDiv)train/gradients/dropout1/truediv_grad/Negdropout1/sub*
T0

/train/gradients/dropout1/truediv_grad/RealDiv_2RealDiv/train/gradients/dropout1/truediv_grad/RealDiv_1dropout1/sub*
T0
¦
)train/gradients/dropout1/truediv_grad/mulMul:train/gradients/dropout1/mul_grad/tuple/control_dependency/train/gradients/dropout1/truediv_grad/RealDiv_2*
T0
Ā
+train/gradients/dropout1/truediv_grad/Sum_1Sum)train/gradients/dropout1/truediv_grad/mul=train/gradients/dropout1/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
­
/train/gradients/dropout1/truediv_grad/Reshape_1Reshape+train/gradients/dropout1/truediv_grad/Sum_1-train/gradients/dropout1/truediv_grad/Shape_1*
T0*
Tshape0
 
6train/gradients/dropout1/truediv_grad/tuple/group_depsNoOp.^train/gradients/dropout1/truediv_grad/Reshape0^train/gradients/dropout1/truediv_grad/Reshape_1
ż
>train/gradients/dropout1/truediv_grad/tuple/control_dependencyIdentity-train/gradients/dropout1/truediv_grad/Reshape7^train/gradients/dropout1/truediv_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/dropout1/truediv_grad/Reshape

@train/gradients/dropout1/truediv_grad/tuple/control_dependency_1Identity/train/gradients/dropout1/truediv_grad/Reshape_17^train/gradients/dropout1/truediv_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/dropout1/truediv_grad/Reshape_1
{
!train/gradients/fc1_grad/ReluGradReluGrad>train/gradients/dropout1/truediv_grad/tuple/control_dependencyfc1*
T0
P
"train/gradients/fc1/Add_grad/ShapeShape
fc1/MatMul*
T0*
out_type0
S
$train/gradients/fc1/Add_grad/Shape_1Const*
valueB:*
dtype0

2train/gradients/fc1/Add_grad/BroadcastGradientArgsBroadcastGradientArgs"train/gradients/fc1/Add_grad/Shape$train/gradients/fc1/Add_grad/Shape_1*
T0
¤
 train/gradients/fc1/Add_grad/SumSum!train/gradients/fc1_grad/ReluGrad2train/gradients/fc1/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

$train/gradients/fc1/Add_grad/ReshapeReshape train/gradients/fc1/Add_grad/Sum"train/gradients/fc1/Add_grad/Shape*
T0*
Tshape0
Ø
"train/gradients/fc1/Add_grad/Sum_1Sum!train/gradients/fc1_grad/ReluGrad4train/gradients/fc1/Add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 

&train/gradients/fc1/Add_grad/Reshape_1Reshape"train/gradients/fc1/Add_grad/Sum_1$train/gradients/fc1/Add_grad/Shape_1*
T0*
Tshape0

-train/gradients/fc1/Add_grad/tuple/group_depsNoOp%^train/gradients/fc1/Add_grad/Reshape'^train/gradients/fc1/Add_grad/Reshape_1
Ł
5train/gradients/fc1/Add_grad/tuple/control_dependencyIdentity$train/gradients/fc1/Add_grad/Reshape.^train/gradients/fc1/Add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@train/gradients/fc1/Add_grad/Reshape
ß
7train/gradients/fc1/Add_grad/tuple/control_dependency_1Identity&train/gradients/fc1/Add_grad/Reshape_1.^train/gradients/fc1/Add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@train/gradients/fc1/Add_grad/Reshape_1
¢
&train/gradients/fc1/MatMul_grad/MatMulMatMul5train/gradients/fc1/Add_grad/tuple/control_dependency
W_fc1/read*
transpose_b(*
T0*
transpose_a( 
”
(train/gradients/fc1/MatMul_grad/MatMul_1MatMulflatten5train/gradients/fc1/Add_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 

0train/gradients/fc1/MatMul_grad/tuple/group_depsNoOp'^train/gradients/fc1/MatMul_grad/MatMul)^train/gradients/fc1/MatMul_grad/MatMul_1
ć
8train/gradients/fc1/MatMul_grad/tuple/control_dependencyIdentity&train/gradients/fc1/MatMul_grad/MatMul1^train/gradients/fc1/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@train/gradients/fc1/MatMul_grad/MatMul
é
:train/gradients/fc1/MatMul_grad/tuple/control_dependency_1Identity(train/gradients/fc1/MatMul_grad/MatMul_11^train/gradients/fc1/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/fc1/MatMul_grad/MatMul_1
M
"train/gradients/flatten_grad/ShapeShapep_pool2*
T0*
out_type0
¤
$train/gradients/flatten_grad/ReshapeReshape8train/gradients/fc1/MatMul_grad/tuple/control_dependency"train/gradients/flatten_grad/Shape*
T0*
Tshape0
Č
(train/gradients/p_pool2_grad/MaxPoolGradMaxPoolGradconv2p_pool2$train/gradients/flatten_grad/Reshape*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

i
#train/gradients/conv2_grad/ReluGradReluGrad(train/gradients/p_pool2_grad/MaxPoolGradconv2*
T0

.train/gradients/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad#train/gradients/conv2_grad/ReluGrad*
T0*
data_formatNHWC

3train/gradients/conv2/BiasAdd_grad/tuple/group_depsNoOp/^train/gradients/conv2/BiasAdd_grad/BiasAddGrad$^train/gradients/conv2_grad/ReluGrad
ć
;train/gradients/conv2/BiasAdd_grad/tuple/control_dependencyIdentity#train/gradients/conv2_grad/ReluGrad4^train/gradients/conv2/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@train/gradients/conv2_grad/ReluGrad
ū
=train/gradients/conv2/BiasAdd_grad/tuple/control_dependency_1Identity.train/gradients/conv2/BiasAdd_grad/BiasAddGrad4^train/gradients/conv2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/conv2/BiasAdd_grad/BiasAddGrad
k
(train/gradients/conv2/Conv2D_grad/ShapeNShapeNp_pool1W_conv2/read*
T0*
out_type0*
N
ø
5train/gradients/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(train/gradients/conv2/Conv2D_grad/ShapeNW_conv2/read;train/gradients/conv2/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
·
6train/gradients/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterp_pool1*train/gradients/conv2/Conv2D_grad/ShapeN:1;train/gradients/conv2/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations

«
2train/gradients/conv2/Conv2D_grad/tuple/group_depsNoOp7^train/gradients/conv2/Conv2D_grad/Conv2DBackpropFilter6^train/gradients/conv2/Conv2D_grad/Conv2DBackpropInput

:train/gradients/conv2/Conv2D_grad/tuple/control_dependencyIdentity5train/gradients/conv2/Conv2D_grad/Conv2DBackpropInput3^train/gradients/conv2/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/conv2/Conv2D_grad/Conv2DBackpropInput

<train/gradients/conv2/Conv2D_grad/tuple/control_dependency_1Identity6train/gradients/conv2/Conv2D_grad/Conv2DBackpropFilter3^train/gradients/conv2/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@train/gradients/conv2/Conv2D_grad/Conv2DBackpropFilter
Ž
(train/gradients/p_pool1_grad/MaxPoolGradMaxPoolGradconv1p_pool1:train/gradients/conv2/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*
T0
i
#train/gradients/conv1_grad/ReluGradReluGrad(train/gradients/p_pool1_grad/MaxPoolGradconv1*
T0

.train/gradients/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad#train/gradients/conv1_grad/ReluGrad*
T0*
data_formatNHWC

3train/gradients/conv1/BiasAdd_grad/tuple/group_depsNoOp/^train/gradients/conv1/BiasAdd_grad/BiasAddGrad$^train/gradients/conv1_grad/ReluGrad
ć
;train/gradients/conv1/BiasAdd_grad/tuple/control_dependencyIdentity#train/gradients/conv1_grad/ReluGrad4^train/gradients/conv1/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@train/gradients/conv1_grad/ReluGrad
ū
=train/gradients/conv1/BiasAdd_grad/tuple/control_dependency_1Identity.train/gradients/conv1/BiasAdd_grad/BiasAddGrad4^train/gradients/conv1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/conv1/BiasAdd_grad/BiasAddGrad
k
(train/gradients/conv1/Conv2D_grad/ShapeNShapeNreshapeW_conv1/read*
T0*
out_type0*
N
ø
5train/gradients/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(train/gradients/conv1/Conv2D_grad/ShapeNW_conv1/read;train/gradients/conv1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
·
6train/gradients/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterreshape*train/gradients/conv1/Conv2D_grad/ShapeN:1;train/gradients/conv1/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations

«
2train/gradients/conv1/Conv2D_grad/tuple/group_depsNoOp7^train/gradients/conv1/Conv2D_grad/Conv2DBackpropFilter6^train/gradients/conv1/Conv2D_grad/Conv2DBackpropInput

:train/gradients/conv1/Conv2D_grad/tuple/control_dependencyIdentity5train/gradients/conv1/Conv2D_grad/Conv2DBackpropInput3^train/gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/conv1/Conv2D_grad/Conv2DBackpropInput

<train/gradients/conv1/Conv2D_grad/tuple/control_dependency_1Identity6train/gradients/conv1/Conv2D_grad/Conv2DBackpropFilter3^train/gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@train/gradients/conv1/Conv2D_grad/Conv2DBackpropFilter
h
train/beta1_power/initial_valueConst*
dtype0*
_class
loc:@W_conv1*
valueB
 *fff?
y
train/beta1_power
VariableV2*
dtype0*
	container *
shape: *
shared_name *
_class
loc:@W_conv1
¤
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
validate_shape(*
use_locking(*
T0*
_class
loc:@W_conv1
Z
train/beta1_power/readIdentitytrain/beta1_power*
T0*
_class
loc:@W_conv1
h
train/beta2_power/initial_valueConst*
dtype0*
_class
loc:@W_conv1*
valueB
 *w¾?
y
train/beta2_power
VariableV2*
dtype0*
	container *
shape: *
shared_name *
_class
loc:@W_conv1
¤
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
validate_shape(*
use_locking(*
T0*
_class
loc:@W_conv1
Z
train/beta2_power/readIdentitytrain/beta2_power*
T0*
_class
loc:@W_conv1
w
W_conv1/Adam/Initializer/zerosConst*%
valueB *    *
_class
loc:@W_conv1*
dtype0

W_conv1/Adam
VariableV2*
shared_name *
_class
loc:@W_conv1*
dtype0*
	container *
shape: 

W_conv1/Adam/AssignAssignW_conv1/AdamW_conv1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@W_conv1*
validate_shape(
P
W_conv1/Adam/readIdentityW_conv1/Adam*
T0*
_class
loc:@W_conv1
y
 W_conv1/Adam_1/Initializer/zerosConst*%
valueB *    *
_class
loc:@W_conv1*
dtype0

W_conv1/Adam_1
VariableV2*
shared_name *
_class
loc:@W_conv1*
dtype0*
	container *
shape: 

W_conv1/Adam_1/AssignAssignW_conv1/Adam_1 W_conv1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@W_conv1*
validate_shape(
T
W_conv1/Adam_1/readIdentityW_conv1/Adam_1*
T0*
_class
loc:@W_conv1
k
b_conv1/Adam/Initializer/zerosConst*
dtype0*
valueB *    *
_class
loc:@b_conv1
x
b_conv1/Adam
VariableV2*
_class
loc:@b_conv1*
dtype0*
	container *
shape: *
shared_name 

b_conv1/Adam/AssignAssignb_conv1/Adamb_conv1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@b_conv1*
validate_shape(
P
b_conv1/Adam/readIdentityb_conv1/Adam*
T0*
_class
loc:@b_conv1
m
 b_conv1/Adam_1/Initializer/zerosConst*
valueB *    *
_class
loc:@b_conv1*
dtype0
z
b_conv1/Adam_1
VariableV2*
_class
loc:@b_conv1*
dtype0*
	container *
shape: *
shared_name 

b_conv1/Adam_1/AssignAssignb_conv1/Adam_1 b_conv1/Adam_1/Initializer/zeros*
T0*
_class
loc:@b_conv1*
validate_shape(*
use_locking(
T
b_conv1/Adam_1/readIdentityb_conv1/Adam_1*
T0*
_class
loc:@b_conv1

.W_conv2/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   *
_class
loc:@W_conv2*
dtype0
m
$W_conv2/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@W_conv2*
dtype0
³
W_conv2/Adam/Initializer/zerosFill.W_conv2/Adam/Initializer/zeros/shape_as_tensor$W_conv2/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@W_conv2

W_conv2/Adam
VariableV2*
shared_name *
_class
loc:@W_conv2*
dtype0*
	container *
shape: @

W_conv2/Adam/AssignAssignW_conv2/AdamW_conv2/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@W_conv2
P
W_conv2/Adam/readIdentityW_conv2/Adam*
T0*
_class
loc:@W_conv2

0W_conv2/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   *
_class
loc:@W_conv2*
dtype0
o
&W_conv2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@W_conv2*
dtype0
¹
 W_conv2/Adam_1/Initializer/zerosFill0W_conv2/Adam_1/Initializer/zeros/shape_as_tensor&W_conv2/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@W_conv2

W_conv2/Adam_1
VariableV2*
dtype0*
	container *
shape: @*
shared_name *
_class
loc:@W_conv2

W_conv2/Adam_1/AssignAssignW_conv2/Adam_1 W_conv2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@W_conv2*
validate_shape(
T
W_conv2/Adam_1/readIdentityW_conv2/Adam_1*
T0*
_class
loc:@W_conv2
k
b_conv2/Adam/Initializer/zerosConst*
valueB@*    *
_class
loc:@b_conv2*
dtype0
x
b_conv2/Adam
VariableV2*
_class
loc:@b_conv2*
dtype0*
	container *
shape:@*
shared_name 

b_conv2/Adam/AssignAssignb_conv2/Adamb_conv2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@b_conv2*
validate_shape(
P
b_conv2/Adam/readIdentityb_conv2/Adam*
T0*
_class
loc:@b_conv2
m
 b_conv2/Adam_1/Initializer/zerosConst*
valueB@*    *
_class
loc:@b_conv2*
dtype0
z
b_conv2/Adam_1
VariableV2*
shared_name *
_class
loc:@b_conv2*
dtype0*
	container *
shape:@

b_conv2/Adam_1/AssignAssignb_conv2/Adam_1 b_conv2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@b_conv2*
validate_shape(
T
b_conv2/Adam_1/readIdentityb_conv2/Adam_1*
T0*
_class
loc:@b_conv2
{
,W_fc1/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"@     *
_class

loc:@W_fc1
i
"W_fc1/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class

loc:@W_fc1*
dtype0
«
W_fc1/Adam/Initializer/zerosFill,W_fc1/Adam/Initializer/zeros/shape_as_tensor"W_fc1/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class

loc:@W_fc1
z

W_fc1/Adam
VariableV2*
shape:
Ą*
shared_name *
_class

loc:@W_fc1*
dtype0*
	container 

W_fc1/Adam/AssignAssign
W_fc1/AdamW_fc1/Adam/Initializer/zeros*
use_locking(*
T0*
_class

loc:@W_fc1*
validate_shape(
J
W_fc1/Adam/readIdentity
W_fc1/Adam*
T0*
_class

loc:@W_fc1
}
.W_fc1/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"@     *
_class

loc:@W_fc1*
dtype0
k
$W_fc1/Adam_1/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *
_class

loc:@W_fc1
±
W_fc1/Adam_1/Initializer/zerosFill.W_fc1/Adam_1/Initializer/zeros/shape_as_tensor$W_fc1/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class

loc:@W_fc1
|
W_fc1/Adam_1
VariableV2*
dtype0*
	container *
shape:
Ą*
shared_name *
_class

loc:@W_fc1

W_fc1/Adam_1/AssignAssignW_fc1/Adam_1W_fc1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class

loc:@W_fc1*
validate_shape(
N
W_fc1/Adam_1/readIdentityW_fc1/Adam_1*
T0*
_class

loc:@W_fc1
u
,b_fc1/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class

loc:@b_fc1*
dtype0
i
"b_fc1/Adam/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *
_class

loc:@b_fc1
«
b_fc1/Adam/Initializer/zerosFill,b_fc1/Adam/Initializer/zeros/shape_as_tensor"b_fc1/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class

loc:@b_fc1
u

b_fc1/Adam
VariableV2*
dtype0*
	container *
shape:*
shared_name *
_class

loc:@b_fc1

b_fc1/Adam/AssignAssign
b_fc1/Adamb_fc1/Adam/Initializer/zeros*
use_locking(*
T0*
_class

loc:@b_fc1*
validate_shape(
J
b_fc1/Adam/readIdentity
b_fc1/Adam*
T0*
_class

loc:@b_fc1
w
.b_fc1/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class

loc:@b_fc1*
dtype0
k
$b_fc1/Adam_1/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *
_class

loc:@b_fc1
±
b_fc1/Adam_1/Initializer/zerosFill.b_fc1/Adam_1/Initializer/zeros/shape_as_tensor$b_fc1/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class

loc:@b_fc1
w
b_fc1/Adam_1
VariableV2*
dtype0*
	container *
shape:*
shared_name *
_class

loc:@b_fc1

b_fc1/Adam_1/AssignAssignb_fc1/Adam_1b_fc1/Adam_1/Initializer/zeros*
T0*
_class

loc:@b_fc1*
validate_shape(*
use_locking(
N
b_fc1/Adam_1/readIdentityb_fc1/Adam_1*
T0*
_class

loc:@b_fc1
{
,W_fc2/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"   
   *
_class

loc:@W_fc2*
dtype0
i
"W_fc2/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class

loc:@W_fc2*
dtype0
«
W_fc2/Adam/Initializer/zerosFill,W_fc2/Adam/Initializer/zeros/shape_as_tensor"W_fc2/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class

loc:@W_fc2
y

W_fc2/Adam
VariableV2*
shared_name *
_class

loc:@W_fc2*
dtype0*
	container *
shape:	


W_fc2/Adam/AssignAssign
W_fc2/AdamW_fc2/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class

loc:@W_fc2
J
W_fc2/Adam/readIdentity
W_fc2/Adam*
T0*
_class

loc:@W_fc2
}
.W_fc2/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   
   *
_class

loc:@W_fc2*
dtype0
k
$W_fc2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class

loc:@W_fc2*
dtype0
±
W_fc2/Adam_1/Initializer/zerosFill.W_fc2/Adam_1/Initializer/zeros/shape_as_tensor$W_fc2/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class

loc:@W_fc2
{
W_fc2/Adam_1
VariableV2*
dtype0*
	container *
shape:	
*
shared_name *
_class

loc:@W_fc2

W_fc2/Adam_1/AssignAssignW_fc2/Adam_1W_fc2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class

loc:@W_fc2*
validate_shape(
N
W_fc2/Adam_1/readIdentityW_fc2/Adam_1*
T0*
_class

loc:@W_fc2
g
b_fc2/Adam/Initializer/zerosConst*
dtype0*
valueB
*    *
_class

loc:@b_fc2
t

b_fc2/Adam
VariableV2*
shape:
*
shared_name *
_class

loc:@b_fc2*
dtype0*
	container 

b_fc2/Adam/AssignAssign
b_fc2/Adamb_fc2/Adam/Initializer/zeros*
T0*
_class

loc:@b_fc2*
validate_shape(*
use_locking(
J
b_fc2/Adam/readIdentity
b_fc2/Adam*
T0*
_class

loc:@b_fc2
i
b_fc2/Adam_1/Initializer/zerosConst*
valueB
*    *
_class

loc:@b_fc2*
dtype0
v
b_fc2/Adam_1
VariableV2*
dtype0*
	container *
shape:
*
shared_name *
_class

loc:@b_fc2

b_fc2/Adam_1/AssignAssignb_fc2/Adam_1b_fc2/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class

loc:@b_fc2
N
b_fc2/Adam_1/readIdentityb_fc2/Adam_1*
T0*
_class

loc:@b_fc2
E
train/Adam/learning_rateConst*
valueB
 *·Ń8*
dtype0
=
train/Adam/beta1Const*
valueB
 *fff?*
dtype0
=
train/Adam/beta2Const*
valueB
 *w¾?*
dtype0
?
train/Adam/epsilonConst*
dtype0*
valueB
 *wĢ+2
ć
#train/Adam/update_W_conv1/ApplyAdam	ApplyAdamW_conv1W_conv1/AdamW_conv1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon<train/gradients/conv1/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_class
loc:@W_conv1
ä
#train/Adam/update_b_conv1/ApplyAdam	ApplyAdamb_conv1b_conv1/Adamb_conv1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@b_conv1*
use_nesterov( *
use_locking( 
ć
#train/Adam/update_W_conv2/ApplyAdam	ApplyAdamW_conv2W_conv2/AdamW_conv2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon<train/gradients/conv2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@W_conv2*
use_nesterov( 
ä
#train/Adam/update_b_conv2/ApplyAdam	ApplyAdamb_conv2b_conv2/Adamb_conv2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/conv2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@b_conv2*
use_nesterov( 
×
!train/Adam/update_W_fc1/ApplyAdam	ApplyAdamW_fc1
W_fc1/AdamW_fc1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/fc1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class

loc:@W_fc1*
use_nesterov( 
Ō
!train/Adam/update_b_fc1/ApplyAdam	ApplyAdamb_fc1
b_fc1/Adamb_fc1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon7train/gradients/fc1/Add_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_class

loc:@b_fc1
×
!train/Adam/update_W_fc2/ApplyAdam	ApplyAdamW_fc2
W_fc2/AdamW_fc2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/fc2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class

loc:@W_fc2*
use_nesterov( 
Ō
!train/Adam/update_b_fc2/ApplyAdam	ApplyAdamb_fc2
b_fc2/Adamb_fc2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon7train/gradients/fc2/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class

loc:@b_fc2*
use_nesterov( 

train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1$^train/Adam/update_W_conv1/ApplyAdam$^train/Adam/update_W_conv2/ApplyAdam"^train/Adam/update_W_fc1/ApplyAdam"^train/Adam/update_W_fc2/ApplyAdam$^train/Adam/update_b_conv1/ApplyAdam$^train/Adam/update_b_conv2/ApplyAdam"^train/Adam/update_b_fc1/ApplyAdam"^train/Adam/update_b_fc2/ApplyAdam*
T0*
_class
loc:@W_conv1

train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
validate_shape(*
use_locking( *
T0*
_class
loc:@W_conv1

train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2$^train/Adam/update_W_conv1/ApplyAdam$^train/Adam/update_W_conv2/ApplyAdam"^train/Adam/update_W_fc1/ApplyAdam"^train/Adam/update_W_fc2/ApplyAdam$^train/Adam/update_b_conv1/ApplyAdam$^train/Adam/update_b_conv2/ApplyAdam"^train/Adam/update_b_fc1/ApplyAdam"^train/Adam/update_b_fc2/ApplyAdam*
T0*
_class
loc:@W_conv1

train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
T0*
_class
loc:@W_conv1*
validate_shape(*
use_locking( 
ä

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1$^train/Adam/update_W_conv1/ApplyAdam$^train/Adam/update_W_conv2/ApplyAdam"^train/Adam/update_W_fc1/ApplyAdam"^train/Adam/update_W_fc2/ApplyAdam$^train/Adam/update_b_conv1/ApplyAdam$^train/Adam/update_b_conv2/ApplyAdam"^train/Adam/update_b_fc1/ApplyAdam"^train/Adam/update_b_fc2/ApplyAdam
B
predict/ArgMax/dimensionConst*
value	B :*
dtype0
b
predict/ArgMaxArgMaxoutputpredict/ArgMax/dimension*
T0*
output_type0	*

Tidx0
D
predict/ArgMax_1/dimensionConst*
dtype0*
value	B :
e
predict/ArgMax_1ArgMaxvalidpredict/ArgMax_1/dimension*

Tidx0*
T0*
output_type0	
A
predict/EqualEqualpredict/ArgMaxpredict/ArgMax_1*
T0	
K
predict/CastCastpredict/Equal*

SrcT0
*
Truncate( *

DstT0
;
predict/ConstConst*
dtype0*
valueB: 
W
predict/MeanMeanpredict/Castpredict/Const*

Tidx0*
	keep_dims( *
T0
A
save/filename/inputConst*
valueB Bmodel*
dtype0
V
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: 
M

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0

save/save/tensor_namesConst*Ū
valueŃBĪBW_conv1BW_conv1/AdamBW_conv1/Adam_1BW_conv2BW_conv2/AdamBW_conv2/Adam_1BW_fc1B
W_fc1/AdamBW_fc1/Adam_1BW_fc2B
W_fc2/AdamBW_fc2/Adam_1Bb_conv1Bb_conv1/AdamBb_conv1/Adam_1Bb_conv2Bb_conv2/AdamBb_conv2/Adam_1Bb_fc1B
b_fc1/AdamBb_fc1/Adam_1Bb_fc2B
b_fc2/AdamBb_fc2/Adam_1Btrain/beta1_powerBtrain/beta2_power*
dtype0
z
save/save/shapes_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ć
	save/save
SaveSlices
save/Constsave/save/tensor_namessave/save/shapes_and_slicesW_conv1W_conv1/AdamW_conv1/Adam_1W_conv2W_conv2/AdamW_conv2/Adam_1W_fc1
W_fc1/AdamW_fc1/Adam_1W_fc2
W_fc2/AdamW_fc2/Adam_1b_conv1b_conv1/Adamb_conv1/Adam_1b_conv2b_conv2/Adamb_conv2/Adam_1b_fc1
b_fc1/Adamb_fc1/Adam_1b_fc2
b_fc2/Adamb_fc2/Adam_1train/beta1_powertrain/beta2_power*#
T
2
c
save/control_dependencyIdentity
save/Const
^save/save*
T0*
_class
loc:@save/Const

save/RestoreV2/tensor_namesConst"/device:CPU:0*Ū
valueŃBĪBW_conv1BW_conv1/AdamBW_conv1/Adam_1BW_conv2BW_conv2/AdamBW_conv2/Adam_1BW_fc1B
W_fc1/AdamBW_fc1/Adam_1BW_fc2B
W_fc2/AdamBW_fc2/Adam_1Bb_conv1Bb_conv1/AdamBb_conv1/Adam_1Bb_conv2Bb_conv2/AdamBb_conv2/Adam_1Bb_fc1B
b_fc1/AdamBb_fc1/Adam_1Bb_fc2B
b_fc2/AdamBb_fc2/Adam_1Btrain/beta1_powerBtrain/beta2_power*
dtype0

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*(
dtypes
2
|
save/AssignAssignW_conv1save/RestoreV2*
use_locking(*
T0*
_class
loc:@W_conv1*
validate_shape(

save/Assign_1AssignW_conv1/Adamsave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@W_conv1*
validate_shape(

save/Assign_2AssignW_conv1/Adam_1save/RestoreV2:2*
T0*
_class
loc:@W_conv1*
validate_shape(*
use_locking(

save/Assign_3AssignW_conv2save/RestoreV2:3*
validate_shape(*
use_locking(*
T0*
_class
loc:@W_conv2

save/Assign_4AssignW_conv2/Adamsave/RestoreV2:4*
validate_shape(*
use_locking(*
T0*
_class
loc:@W_conv2

save/Assign_5AssignW_conv2/Adam_1save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@W_conv2*
validate_shape(
|
save/Assign_6AssignW_fc1save/RestoreV2:6*
use_locking(*
T0*
_class

loc:@W_fc1*
validate_shape(

save/Assign_7Assign
W_fc1/Adamsave/RestoreV2:7*
use_locking(*
T0*
_class

loc:@W_fc1*
validate_shape(

save/Assign_8AssignW_fc1/Adam_1save/RestoreV2:8*
T0*
_class

loc:@W_fc1*
validate_shape(*
use_locking(
|
save/Assign_9AssignW_fc2save/RestoreV2:9*
use_locking(*
T0*
_class

loc:@W_fc2*
validate_shape(

save/Assign_10Assign
W_fc2/Adamsave/RestoreV2:10*
use_locking(*
T0*
_class

loc:@W_fc2*
validate_shape(

save/Assign_11AssignW_fc2/Adam_1save/RestoreV2:11*
T0*
_class

loc:@W_fc2*
validate_shape(*
use_locking(

save/Assign_12Assignb_conv1save/RestoreV2:12*
validate_shape(*
use_locking(*
T0*
_class
loc:@b_conv1

save/Assign_13Assignb_conv1/Adamsave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@b_conv1*
validate_shape(

save/Assign_14Assignb_conv1/Adam_1save/RestoreV2:14*
T0*
_class
loc:@b_conv1*
validate_shape(*
use_locking(

save/Assign_15Assignb_conv2save/RestoreV2:15*
T0*
_class
loc:@b_conv2*
validate_shape(*
use_locking(

save/Assign_16Assignb_conv2/Adamsave/RestoreV2:16*
use_locking(*
T0*
_class
loc:@b_conv2*
validate_shape(

save/Assign_17Assignb_conv2/Adam_1save/RestoreV2:17*
T0*
_class
loc:@b_conv2*
validate_shape(*
use_locking(
~
save/Assign_18Assignb_fc1save/RestoreV2:18*
T0*
_class

loc:@b_fc1*
validate_shape(*
use_locking(

save/Assign_19Assign
b_fc1/Adamsave/RestoreV2:19*
use_locking(*
T0*
_class

loc:@b_fc1*
validate_shape(

save/Assign_20Assignb_fc1/Adam_1save/RestoreV2:20*
use_locking(*
T0*
_class

loc:@b_fc1*
validate_shape(
~
save/Assign_21Assignb_fc2save/RestoreV2:21*
use_locking(*
T0*
_class

loc:@b_fc2*
validate_shape(

save/Assign_22Assign
b_fc2/Adamsave/RestoreV2:22*
validate_shape(*
use_locking(*
T0*
_class

loc:@b_fc2

save/Assign_23Assignb_fc2/Adam_1save/RestoreV2:23*
validate_shape(*
use_locking(*
T0*
_class

loc:@b_fc2

save/Assign_24Assigntrain/beta1_powersave/RestoreV2:24*
T0*
_class
loc:@W_conv1*
validate_shape(*
use_locking(

save/Assign_25Assigntrain/beta2_powersave/RestoreV2:25*
use_locking(*
T0*
_class
loc:@W_conv1*
validate_shape(
Ę
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
¢
initNoOp^W_conv1/Adam/Assign^W_conv1/Adam_1/Assign^W_conv1/Assign^W_conv2/Adam/Assign^W_conv2/Adam_1/Assign^W_conv2/Assign^W_fc1/Adam/Assign^W_fc1/Adam_1/Assign^W_fc1/Assign^W_fc2/Adam/Assign^W_fc2/Adam_1/Assign^W_fc2/Assign^b_conv1/Adam/Assign^b_conv1/Adam_1/Assign^b_conv1/Assign^b_conv2/Adam/Assign^b_conv2/Adam_1/Assign^b_conv2/Assign^b_fc1/Adam/Assign^b_fc1/Adam_1/Assign^b_fc1/Assign^b_fc2/Adam/Assign^b_fc2/Adam_1/Assign^b_fc2/Assign^train/beta1_power/Assign^train/beta2_power/Assign"