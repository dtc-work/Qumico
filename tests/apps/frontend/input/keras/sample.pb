
K
input_1Placeholder*
dtype0*&
shape:ĸĸĸĸĸĸĸĸĸāā
^
!block1_conv1/random_uniform/shapeConst*%
valueB"         @   *
dtype0
L
block1_conv1/random_uniform/minConst*
dtype0*
valueB
 *8JĖ―
L
block1_conv1/random_uniform/maxConst*
valueB
 *8JĖ=*
dtype0

)block1_conv1/random_uniform/RandomUniformRandomUniform!block1_conv1/random_uniform/shape*
seedąĸå)*
T0*
dtype0*
seed2Æķ
q
block1_conv1/random_uniform/subSubblock1_conv1/random_uniform/maxblock1_conv1/random_uniform/min*
T0
{
block1_conv1/random_uniform/mulMul)block1_conv1/random_uniform/RandomUniformblock1_conv1/random_uniform/sub*
T0
m
block1_conv1/random_uniformAddblock1_conv1/random_uniform/mulblock1_conv1/random_uniform/min*
T0
o
block1_conv1/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape:@
°
block1_conv1/kernel/AssignAssignblock1_conv1/kernelblock1_conv1/random_uniform*
use_locking(*
T0*&
_class
loc:@block1_conv1/kernel*
validate_shape(
j
block1_conv1/kernel/readIdentityblock1_conv1/kernel*
T0*&
_class
loc:@block1_conv1/kernel
C
block1_conv1/ConstConst*
dtype0*
valueB@*    
a
block1_conv1/bias
VariableV2*
dtype0*
	container *
shape:@*
shared_name 
Ą
block1_conv1/bias/AssignAssignblock1_conv1/biasblock1_conv1/Const*
T0*$
_class
loc:@block1_conv1/bias*
validate_shape(*
use_locking(
d
block1_conv1/bias/readIdentityblock1_conv1/bias*
T0*$
_class
loc:@block1_conv1/bias
[
&block1_conv1/convolution/dilation_rateConst*
valueB"      *
dtype0
ž
block1_conv1/convolutionConv2Dinput_1block1_conv1/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
q
block1_conv1/BiasAddBiasAddblock1_conv1/convolutionblock1_conv1/bias/read*
T0*
data_formatNHWC
8
block1_conv1/ReluRelublock1_conv1/BiasAdd*
T0
^
!block1_conv2/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0
L
block1_conv2/random_uniform/minConst*
valueB
 *:Í―*
dtype0
L
block1_conv2/random_uniform/maxConst*
valueB
 *:Í=*
dtype0

)block1_conv2/random_uniform/RandomUniformRandomUniform!block1_conv2/random_uniform/shape*
seedąĸå)*
T0*
dtype0*
seed2ē
q
block1_conv2/random_uniform/subSubblock1_conv2/random_uniform/maxblock1_conv2/random_uniform/min*
T0
{
block1_conv2/random_uniform/mulMul)block1_conv2/random_uniform/RandomUniformblock1_conv2/random_uniform/sub*
T0
m
block1_conv2/random_uniformAddblock1_conv2/random_uniform/mulblock1_conv2/random_uniform/min*
T0
o
block1_conv2/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape:@@
°
block1_conv2/kernel/AssignAssignblock1_conv2/kernelblock1_conv2/random_uniform*
use_locking(*
T0*&
_class
loc:@block1_conv2/kernel*
validate_shape(
j
block1_conv2/kernel/readIdentityblock1_conv2/kernel*
T0*&
_class
loc:@block1_conv2/kernel
C
block1_conv2/ConstConst*
valueB@*    *
dtype0
a
block1_conv2/bias
VariableV2*
shape:@*
shared_name *
dtype0*
	container 
Ą
block1_conv2/bias/AssignAssignblock1_conv2/biasblock1_conv2/Const*
use_locking(*
T0*$
_class
loc:@block1_conv2/bias*
validate_shape(
d
block1_conv2/bias/readIdentityblock1_conv2/bias*
T0*$
_class
loc:@block1_conv2/bias
[
&block1_conv2/convolution/dilation_rateConst*
valueB"      *
dtype0
Æ
block1_conv2/convolutionConv2Dblock1_conv1/Relublock1_conv2/kernel/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
q
block1_conv2/BiasAddBiasAddblock1_conv2/convolutionblock1_conv2/bias/read*
data_formatNHWC*
T0
8
block1_conv2/ReluRelublock1_conv2/BiasAdd*
T0

block1_pool/MaxPoolMaxPoolblock1_conv2/Relu*
ksize
*
paddingVALID*
T0*
data_formatNHWC*
strides

^
!block2_conv1/random_uniform/shapeConst*%
valueB"      @      *
dtype0
L
block2_conv1/random_uniform/minConst*
valueB
 *ï[q―*
dtype0
L
block2_conv1/random_uniform/maxConst*
valueB
 *ï[q=*
dtype0

)block2_conv1/random_uniform/RandomUniformRandomUniform!block2_conv1/random_uniform/shape*
dtype0*
seed2Îå*
seedąĸå)*
T0
q
block2_conv1/random_uniform/subSubblock2_conv1/random_uniform/maxblock2_conv1/random_uniform/min*
T0
{
block2_conv1/random_uniform/mulMul)block2_conv1/random_uniform/RandomUniformblock2_conv1/random_uniform/sub*
T0
m
block2_conv1/random_uniformAddblock2_conv1/random_uniform/mulblock2_conv1/random_uniform/min*
T0
p
block2_conv1/kernel
VariableV2*
dtype0*
	container *
shape:@*
shared_name 
°
block2_conv1/kernel/AssignAssignblock2_conv1/kernelblock2_conv1/random_uniform*
T0*&
_class
loc:@block2_conv1/kernel*
validate_shape(*
use_locking(
j
block2_conv1/kernel/readIdentityblock2_conv1/kernel*
T0*&
_class
loc:@block2_conv1/kernel
D
block2_conv1/ConstConst*
valueB*    *
dtype0
b
block2_conv1/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name 
Ą
block2_conv1/bias/AssignAssignblock2_conv1/biasblock2_conv1/Const*
T0*$
_class
loc:@block2_conv1/bias*
validate_shape(*
use_locking(
d
block2_conv1/bias/readIdentityblock2_conv1/bias*
T0*$
_class
loc:@block2_conv1/bias
[
&block2_conv1/convolution/dilation_rateConst*
valueB"      *
dtype0
Č
block2_conv1/convolutionConv2Dblock1_pool/MaxPoolblock2_conv1/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations

q
block2_conv1/BiasAddBiasAddblock2_conv1/convolutionblock2_conv1/bias/read*
data_formatNHWC*
T0
8
block2_conv1/ReluRelublock2_conv1/BiasAdd*
T0
^
!block2_conv2/random_uniform/shapeConst*%
valueB"            *
dtype0
L
block2_conv2/random_uniform/minConst*
valueB
 *ėQ―*
dtype0
L
block2_conv2/random_uniform/maxConst*
dtype0*
valueB
 *ėQ=

)block2_conv2/random_uniform/RandomUniformRandomUniform!block2_conv2/random_uniform/shape*
seedąĸå)*
T0*
dtype0*
seed2ŌĒŲ
q
block2_conv2/random_uniform/subSubblock2_conv2/random_uniform/maxblock2_conv2/random_uniform/min*
T0
{
block2_conv2/random_uniform/mulMul)block2_conv2/random_uniform/RandomUniformblock2_conv2/random_uniform/sub*
T0
m
block2_conv2/random_uniformAddblock2_conv2/random_uniform/mulblock2_conv2/random_uniform/min*
T0
q
block2_conv2/kernel
VariableV2*
dtype0*
	container *
shape:*
shared_name 
°
block2_conv2/kernel/AssignAssignblock2_conv2/kernelblock2_conv2/random_uniform*
use_locking(*
T0*&
_class
loc:@block2_conv2/kernel*
validate_shape(
j
block2_conv2/kernel/readIdentityblock2_conv2/kernel*
T0*&
_class
loc:@block2_conv2/kernel
D
block2_conv2/ConstConst*
valueB*    *
dtype0
b
block2_conv2/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container 
Ą
block2_conv2/bias/AssignAssignblock2_conv2/biasblock2_conv2/Const*
use_locking(*
T0*$
_class
loc:@block2_conv2/bias*
validate_shape(
d
block2_conv2/bias/readIdentityblock2_conv2/bias*
T0*$
_class
loc:@block2_conv2/bias
[
&block2_conv2/convolution/dilation_rateConst*
valueB"      *
dtype0
Æ
block2_conv2/convolutionConv2Dblock2_conv1/Relublock2_conv2/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
q
block2_conv2/BiasAddBiasAddblock2_conv2/convolutionblock2_conv2/bias/read*
T0*
data_formatNHWC
8
block2_conv2/ReluRelublock2_conv2/BiasAdd*
T0

block2_pool/MaxPoolMaxPoolblock2_conv2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
^
!block3_conv1/random_uniform/shapeConst*%
valueB"            *
dtype0
L
block3_conv1/random_uniform/minConst*
valueB
 *ŦŠ*―*
dtype0
L
block3_conv1/random_uniform/maxConst*
valueB
 *ŦŠ*=*
dtype0

)block3_conv1/random_uniform/RandomUniformRandomUniform!block3_conv1/random_uniform/shape*
dtype0*
seed2ÉÃ*
seedąĸå)*
T0
q
block3_conv1/random_uniform/subSubblock3_conv1/random_uniform/maxblock3_conv1/random_uniform/min*
T0
{
block3_conv1/random_uniform/mulMul)block3_conv1/random_uniform/RandomUniformblock3_conv1/random_uniform/sub*
T0
m
block3_conv1/random_uniformAddblock3_conv1/random_uniform/mulblock3_conv1/random_uniform/min*
T0
q
block3_conv1/kernel
VariableV2*
dtype0*
	container *
shape:*
shared_name 
°
block3_conv1/kernel/AssignAssignblock3_conv1/kernelblock3_conv1/random_uniform*
use_locking(*
T0*&
_class
loc:@block3_conv1/kernel*
validate_shape(
j
block3_conv1/kernel/readIdentityblock3_conv1/kernel*
T0*&
_class
loc:@block3_conv1/kernel
D
block3_conv1/ConstConst*
dtype0*
valueB*    
b
block3_conv1/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name 
Ą
block3_conv1/bias/AssignAssignblock3_conv1/biasblock3_conv1/Const*
use_locking(*
T0*$
_class
loc:@block3_conv1/bias*
validate_shape(
d
block3_conv1/bias/readIdentityblock3_conv1/bias*
T0*$
_class
loc:@block3_conv1/bias
[
&block3_conv1/convolution/dilation_rateConst*
valueB"      *
dtype0
Č
block3_conv1/convolutionConv2Dblock2_pool/MaxPoolblock3_conv1/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
q
block3_conv1/BiasAddBiasAddblock3_conv1/convolutionblock3_conv1/bias/read*
T0*
data_formatNHWC
8
block3_conv1/ReluRelublock3_conv1/BiasAdd*
T0
^
!block3_conv2/random_uniform/shapeConst*%
valueB"            *
dtype0
L
block3_conv2/random_uniform/minConst*
valueB
 *:Í―*
dtype0
L
block3_conv2/random_uniform/maxConst*
dtype0*
valueB
 *:Í=

)block3_conv2/random_uniform/RandomUniformRandomUniform!block3_conv2/random_uniform/shape*
T0*
dtype0*
seed2ė*
seedąĸå)
q
block3_conv2/random_uniform/subSubblock3_conv2/random_uniform/maxblock3_conv2/random_uniform/min*
T0
{
block3_conv2/random_uniform/mulMul)block3_conv2/random_uniform/RandomUniformblock3_conv2/random_uniform/sub*
T0
m
block3_conv2/random_uniformAddblock3_conv2/random_uniform/mulblock3_conv2/random_uniform/min*
T0
q
block3_conv2/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape:
°
block3_conv2/kernel/AssignAssignblock3_conv2/kernelblock3_conv2/random_uniform*
use_locking(*
T0*&
_class
loc:@block3_conv2/kernel*
validate_shape(
j
block3_conv2/kernel/readIdentityblock3_conv2/kernel*
T0*&
_class
loc:@block3_conv2/kernel
D
block3_conv2/ConstConst*
valueB*    *
dtype0
b
block3_conv2/bias
VariableV2*
shared_name *
dtype0*
	container *
shape:
Ą
block3_conv2/bias/AssignAssignblock3_conv2/biasblock3_conv2/Const*
use_locking(*
T0*$
_class
loc:@block3_conv2/bias*
validate_shape(
d
block3_conv2/bias/readIdentityblock3_conv2/bias*
T0*$
_class
loc:@block3_conv2/bias
[
&block3_conv2/convolution/dilation_rateConst*
valueB"      *
dtype0
Æ
block3_conv2/convolutionConv2Dblock3_conv1/Relublock3_conv2/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
q
block3_conv2/BiasAddBiasAddblock3_conv2/convolutionblock3_conv2/bias/read*
data_formatNHWC*
T0
8
block3_conv2/ReluRelublock3_conv2/BiasAdd*
T0
^
!block3_conv3/random_uniform/shapeConst*%
valueB"            *
dtype0
L
block3_conv3/random_uniform/minConst*
valueB
 *:Í―*
dtype0
L
block3_conv3/random_uniform/maxConst*
valueB
 *:Í=*
dtype0

)block3_conv3/random_uniform/RandomUniformRandomUniform!block3_conv3/random_uniform/shape*
dtype0*
seed2*
seedąĸå)*
T0
q
block3_conv3/random_uniform/subSubblock3_conv3/random_uniform/maxblock3_conv3/random_uniform/min*
T0
{
block3_conv3/random_uniform/mulMul)block3_conv3/random_uniform/RandomUniformblock3_conv3/random_uniform/sub*
T0
m
block3_conv3/random_uniformAddblock3_conv3/random_uniform/mulblock3_conv3/random_uniform/min*
T0
q
block3_conv3/kernel
VariableV2*
dtype0*
	container *
shape:*
shared_name 
°
block3_conv3/kernel/AssignAssignblock3_conv3/kernelblock3_conv3/random_uniform*
validate_shape(*
use_locking(*
T0*&
_class
loc:@block3_conv3/kernel
j
block3_conv3/kernel/readIdentityblock3_conv3/kernel*
T0*&
_class
loc:@block3_conv3/kernel
D
block3_conv3/ConstConst*
valueB*    *
dtype0
b
block3_conv3/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name 
Ą
block3_conv3/bias/AssignAssignblock3_conv3/biasblock3_conv3/Const*
T0*$
_class
loc:@block3_conv3/bias*
validate_shape(*
use_locking(
d
block3_conv3/bias/readIdentityblock3_conv3/bias*
T0*$
_class
loc:@block3_conv3/bias
[
&block3_conv3/convolution/dilation_rateConst*
dtype0*
valueB"      
Æ
block3_conv3/convolutionConv2Dblock3_conv2/Relublock3_conv3/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
q
block3_conv3/BiasAddBiasAddblock3_conv3/convolutionblock3_conv3/bias/read*
T0*
data_formatNHWC
8
block3_conv3/ReluRelublock3_conv3/BiasAdd*
T0

block3_pool/MaxPoolMaxPoolblock3_conv3/Relu*
ksize
*
paddingVALID*
T0*
data_formatNHWC*
strides

^
!block4_conv1/random_uniform/shapeConst*%
valueB"            *
dtype0
L
block4_conv1/random_uniform/minConst*
valueB
 *ï[ņž*
dtype0
L
block4_conv1/random_uniform/maxConst*
valueB
 *ï[ņ<*
dtype0

)block4_conv1/random_uniform/RandomUniformRandomUniform!block4_conv1/random_uniform/shape*
T0*
dtype0*
seed2ĖÁ*
seedąĸå)
q
block4_conv1/random_uniform/subSubblock4_conv1/random_uniform/maxblock4_conv1/random_uniform/min*
T0
{
block4_conv1/random_uniform/mulMul)block4_conv1/random_uniform/RandomUniformblock4_conv1/random_uniform/sub*
T0
m
block4_conv1/random_uniformAddblock4_conv1/random_uniform/mulblock4_conv1/random_uniform/min*
T0
q
block4_conv1/kernel
VariableV2*
shape:*
shared_name *
dtype0*
	container 
°
block4_conv1/kernel/AssignAssignblock4_conv1/kernelblock4_conv1/random_uniform*
use_locking(*
T0*&
_class
loc:@block4_conv1/kernel*
validate_shape(
j
block4_conv1/kernel/readIdentityblock4_conv1/kernel*
T0*&
_class
loc:@block4_conv1/kernel
D
block4_conv1/ConstConst*
valueB*    *
dtype0
b
block4_conv1/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container 
Ą
block4_conv1/bias/AssignAssignblock4_conv1/biasblock4_conv1/Const*
validate_shape(*
use_locking(*
T0*$
_class
loc:@block4_conv1/bias
d
block4_conv1/bias/readIdentityblock4_conv1/bias*
T0*$
_class
loc:@block4_conv1/bias
[
&block4_conv1/convolution/dilation_rateConst*
dtype0*
valueB"      
Č
block4_conv1/convolutionConv2Dblock3_pool/MaxPoolblock4_conv1/kernel/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
q
block4_conv1/BiasAddBiasAddblock4_conv1/convolutionblock4_conv1/bias/read*
T0*
data_formatNHWC
8
block4_conv1/ReluRelublock4_conv1/BiasAdd*
T0
^
!block4_conv2/random_uniform/shapeConst*%
valueB"            *
dtype0
L
block4_conv2/random_uniform/minConst*
dtype0*
valueB
 *ėŅž
L
block4_conv2/random_uniform/maxConst*
dtype0*
valueB
 *ėŅ<

)block4_conv2/random_uniform/RandomUniformRandomUniform!block4_conv2/random_uniform/shape*
T0*
dtype0*
seed2åŊ*
seedąĸå)
q
block4_conv2/random_uniform/subSubblock4_conv2/random_uniform/maxblock4_conv2/random_uniform/min*
T0
{
block4_conv2/random_uniform/mulMul)block4_conv2/random_uniform/RandomUniformblock4_conv2/random_uniform/sub*
T0
m
block4_conv2/random_uniformAddblock4_conv2/random_uniform/mulblock4_conv2/random_uniform/min*
T0
q
block4_conv2/kernel
VariableV2*
shape:*
shared_name *
dtype0*
	container 
°
block4_conv2/kernel/AssignAssignblock4_conv2/kernelblock4_conv2/random_uniform*
validate_shape(*
use_locking(*
T0*&
_class
loc:@block4_conv2/kernel
j
block4_conv2/kernel/readIdentityblock4_conv2/kernel*
T0*&
_class
loc:@block4_conv2/kernel
D
block4_conv2/ConstConst*
valueB*    *
dtype0
b
block4_conv2/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container 
Ą
block4_conv2/bias/AssignAssignblock4_conv2/biasblock4_conv2/Const*
use_locking(*
T0*$
_class
loc:@block4_conv2/bias*
validate_shape(
d
block4_conv2/bias/readIdentityblock4_conv2/bias*
T0*$
_class
loc:@block4_conv2/bias
[
&block4_conv2/convolution/dilation_rateConst*
valueB"      *
dtype0
Æ
block4_conv2/convolutionConv2Dblock4_conv1/Relublock4_conv2/kernel/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
q
block4_conv2/BiasAddBiasAddblock4_conv2/convolutionblock4_conv2/bias/read*
T0*
data_formatNHWC
8
block4_conv2/ReluRelublock4_conv2/BiasAdd*
T0
^
!block4_conv3/random_uniform/shapeConst*%
valueB"            *
dtype0
L
block4_conv3/random_uniform/minConst*
valueB
 *ėŅž*
dtype0
L
block4_conv3/random_uniform/maxConst*
valueB
 *ėŅ<*
dtype0

)block4_conv3/random_uniform/RandomUniformRandomUniform!block4_conv3/random_uniform/shape*
T0*
dtype0*
seed2ŧy*
seedąĸå)
q
block4_conv3/random_uniform/subSubblock4_conv3/random_uniform/maxblock4_conv3/random_uniform/min*
T0
{
block4_conv3/random_uniform/mulMul)block4_conv3/random_uniform/RandomUniformblock4_conv3/random_uniform/sub*
T0
m
block4_conv3/random_uniformAddblock4_conv3/random_uniform/mulblock4_conv3/random_uniform/min*
T0
q
block4_conv3/kernel
VariableV2*
shape:*
shared_name *
dtype0*
	container 
°
block4_conv3/kernel/AssignAssignblock4_conv3/kernelblock4_conv3/random_uniform*
validate_shape(*
use_locking(*
T0*&
_class
loc:@block4_conv3/kernel
j
block4_conv3/kernel/readIdentityblock4_conv3/kernel*
T0*&
_class
loc:@block4_conv3/kernel
D
block4_conv3/ConstConst*
valueB*    *
dtype0
b
block4_conv3/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name 
Ą
block4_conv3/bias/AssignAssignblock4_conv3/biasblock4_conv3/Const*
use_locking(*
T0*$
_class
loc:@block4_conv3/bias*
validate_shape(
d
block4_conv3/bias/readIdentityblock4_conv3/bias*
T0*$
_class
loc:@block4_conv3/bias
[
&block4_conv3/convolution/dilation_rateConst*
valueB"      *
dtype0
Æ
block4_conv3/convolutionConv2Dblock4_conv2/Relublock4_conv3/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
q
block4_conv3/BiasAddBiasAddblock4_conv3/convolutionblock4_conv3/bias/read*
T0*
data_formatNHWC
8
block4_conv3/ReluRelublock4_conv3/BiasAdd*
T0

block4_pool/MaxPoolMaxPoolblock4_conv3/Relu*
ksize
*
paddingVALID*
T0*
data_formatNHWC*
strides

^
!block5_conv1/random_uniform/shapeConst*%
valueB"            *
dtype0
L
block5_conv1/random_uniform/minConst*
valueB
 *ėŅž*
dtype0
L
block5_conv1/random_uniform/maxConst*
dtype0*
valueB
 *ėŅ<

)block5_conv1/random_uniform/RandomUniformRandomUniform!block5_conv1/random_uniform/shape*
seedąĸå)*
T0*
dtype0*
seed2ļÅÖ
q
block5_conv1/random_uniform/subSubblock5_conv1/random_uniform/maxblock5_conv1/random_uniform/min*
T0
{
block5_conv1/random_uniform/mulMul)block5_conv1/random_uniform/RandomUniformblock5_conv1/random_uniform/sub*
T0
m
block5_conv1/random_uniformAddblock5_conv1/random_uniform/mulblock5_conv1/random_uniform/min*
T0
q
block5_conv1/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape:
°
block5_conv1/kernel/AssignAssignblock5_conv1/kernelblock5_conv1/random_uniform*
use_locking(*
T0*&
_class
loc:@block5_conv1/kernel*
validate_shape(
j
block5_conv1/kernel/readIdentityblock5_conv1/kernel*
T0*&
_class
loc:@block5_conv1/kernel
D
block5_conv1/ConstConst*
valueB*    *
dtype0
b
block5_conv1/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name 
Ą
block5_conv1/bias/AssignAssignblock5_conv1/biasblock5_conv1/Const*
T0*$
_class
loc:@block5_conv1/bias*
validate_shape(*
use_locking(
d
block5_conv1/bias/readIdentityblock5_conv1/bias*
T0*$
_class
loc:@block5_conv1/bias
[
&block5_conv1/convolution/dilation_rateConst*
valueB"      *
dtype0
Č
block5_conv1/convolutionConv2Dblock4_pool/MaxPoolblock5_conv1/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
q
block5_conv1/BiasAddBiasAddblock5_conv1/convolutionblock5_conv1/bias/read*
T0*
data_formatNHWC
8
block5_conv1/ReluRelublock5_conv1/BiasAdd*
T0
^
!block5_conv2/random_uniform/shapeConst*
dtype0*%
valueB"            
L
block5_conv2/random_uniform/minConst*
dtype0*
valueB
 *ėŅž
L
block5_conv2/random_uniform/maxConst*
dtype0*
valueB
 *ėŅ<

)block5_conv2/random_uniform/RandomUniformRandomUniform!block5_conv2/random_uniform/shape*
dtype0*
seed2ęįį*
seedąĸå)*
T0
q
block5_conv2/random_uniform/subSubblock5_conv2/random_uniform/maxblock5_conv2/random_uniform/min*
T0
{
block5_conv2/random_uniform/mulMul)block5_conv2/random_uniform/RandomUniformblock5_conv2/random_uniform/sub*
T0
m
block5_conv2/random_uniformAddblock5_conv2/random_uniform/mulblock5_conv2/random_uniform/min*
T0
q
block5_conv2/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape:
°
block5_conv2/kernel/AssignAssignblock5_conv2/kernelblock5_conv2/random_uniform*
use_locking(*
T0*&
_class
loc:@block5_conv2/kernel*
validate_shape(
j
block5_conv2/kernel/readIdentityblock5_conv2/kernel*
T0*&
_class
loc:@block5_conv2/kernel
D
block5_conv2/ConstConst*
valueB*    *
dtype0
b
block5_conv2/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name 
Ą
block5_conv2/bias/AssignAssignblock5_conv2/biasblock5_conv2/Const*
validate_shape(*
use_locking(*
T0*$
_class
loc:@block5_conv2/bias
d
block5_conv2/bias/readIdentityblock5_conv2/bias*
T0*$
_class
loc:@block5_conv2/bias
[
&block5_conv2/convolution/dilation_rateConst*
dtype0*
valueB"      
Æ
block5_conv2/convolutionConv2Dblock5_conv1/Relublock5_conv2/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
q
block5_conv2/BiasAddBiasAddblock5_conv2/convolutionblock5_conv2/bias/read*
T0*
data_formatNHWC
8
block5_conv2/ReluRelublock5_conv2/BiasAdd*
T0
^
!block5_conv3/random_uniform/shapeConst*%
valueB"            *
dtype0
L
block5_conv3/random_uniform/minConst*
valueB
 *ėŅž*
dtype0
L
block5_conv3/random_uniform/maxConst*
valueB
 *ėŅ<*
dtype0

)block5_conv3/random_uniform/RandomUniformRandomUniform!block5_conv3/random_uniform/shape*
dtype0*
seed2ĀÆ*
seedąĸå)*
T0
q
block5_conv3/random_uniform/subSubblock5_conv3/random_uniform/maxblock5_conv3/random_uniform/min*
T0
{
block5_conv3/random_uniform/mulMul)block5_conv3/random_uniform/RandomUniformblock5_conv3/random_uniform/sub*
T0
m
block5_conv3/random_uniformAddblock5_conv3/random_uniform/mulblock5_conv3/random_uniform/min*
T0
q
block5_conv3/kernel
VariableV2*
dtype0*
	container *
shape:*
shared_name 
°
block5_conv3/kernel/AssignAssignblock5_conv3/kernelblock5_conv3/random_uniform*
use_locking(*
T0*&
_class
loc:@block5_conv3/kernel*
validate_shape(
j
block5_conv3/kernel/readIdentityblock5_conv3/kernel*
T0*&
_class
loc:@block5_conv3/kernel
D
block5_conv3/ConstConst*
valueB*    *
dtype0
b
block5_conv3/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name 
Ą
block5_conv3/bias/AssignAssignblock5_conv3/biasblock5_conv3/Const*
use_locking(*
T0*$
_class
loc:@block5_conv3/bias*
validate_shape(
d
block5_conv3/bias/readIdentityblock5_conv3/bias*
T0*$
_class
loc:@block5_conv3/bias
[
&block5_conv3/convolution/dilation_rateConst*
dtype0*
valueB"      
Æ
block5_conv3/convolutionConv2Dblock5_conv2/Relublock5_conv3/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
q
block5_conv3/BiasAddBiasAddblock5_conv3/convolutionblock5_conv3/bias/read*
T0*
data_formatNHWC
8
block5_conv3/ReluRelublock5_conv3/BiasAdd*
T0

block5_pool/MaxPoolMaxPoolblock5_conv3/Relu*
ksize
*
paddingVALID*
T0*
data_formatNHWC*
strides

f
1global_average_pooling2d_1/Mean/reduction_indicesConst*
valueB"      *
dtype0

global_average_pooling2d_1/MeanMeanblock5_pool/MaxPool1global_average_pooling2d_1/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0
Q
dense_1/random_uniform/shapeConst*
valueB"      *
dtype0
G
dense_1/random_uniform/minConst*
valueB
 *  ―*
dtype0
G
dense_1/random_uniform/maxConst*
valueB
 *  =*
dtype0

$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
T0*
dtype0*
seed2S*
seedąĸå)
b
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0
l
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0
^
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0
d
dense_1/kernel
VariableV2*
shape:
*
shared_name *
dtype0*
	container 

dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(
[
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel
?
dense_1/ConstConst*
valueB*    *
dtype0
]
dense_1/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name 

dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
validate_shape(*
use_locking(*
T0*
_class
loc:@dense_1/bias
U
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias
}
dense_1/MatMulMatMulglobal_average_pooling2d_1/Meandense_1/kernel/read*
transpose_a( *
transpose_b( *
T0
]
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC
.
dense_1/ReluReludense_1/BiasAdd*
T0
Q
dense_2/random_uniform/shapeConst*
valueB"      *
dtype0
G
dense_2/random_uniform/minConst*
valueB
 *Ņb―*
dtype0
G
dense_2/random_uniform/maxConst*
valueB
 *Ņb=*
dtype0

$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
dtype0*
seed2öēģ*
seedąĸå)*
T0
b
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0
l
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0
^
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0
c
dense_2/kernel
VariableV2*
dtype0*
	container *
shape:	*
shared_name 

dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
use_locking(
[
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel
>
dense_2/ConstConst*
valueB*    *
dtype0
\
dense_2/bias
VariableV2*
shared_name *
dtype0*
	container *
shape:

dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
validate_shape(*
use_locking(*
T0*
_class
loc:@dense_2/bias
U
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias
j
dense_2/MatMulMatMuldense_1/Reludense_2/kernel/read*
transpose_b( *
T0*
transpose_a( 
]
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC
4
dense_2/SoftmaxSoftmaxdense_2/BiasAdd*
T0
D
PlaceholderPlaceholder*
dtype0*
shape:@

AssignAssignblock1_conv1/kernelPlaceholder*
validate_shape(*
use_locking( *
T0*&
_class
loc:@block1_conv1/kernel
:
Placeholder_1Placeholder*
dtype0*
shape:@

Assign_1Assignblock1_conv1/biasPlaceholder_1*
use_locking( *
T0*$
_class
loc:@block1_conv1/bias*
validate_shape(
F
Placeholder_2Placeholder*
dtype0*
shape:@@

Assign_2Assignblock1_conv2/kernelPlaceholder_2*
T0*&
_class
loc:@block1_conv2/kernel*
validate_shape(*
use_locking( 
:
Placeholder_3Placeholder*
dtype0*
shape:@

Assign_3Assignblock1_conv2/biasPlaceholder_3*
validate_shape(*
use_locking( *
T0*$
_class
loc:@block1_conv2/bias
G
Placeholder_4Placeholder*
shape:@*
dtype0

Assign_4Assignblock2_conv1/kernelPlaceholder_4*
T0*&
_class
loc:@block2_conv1/kernel*
validate_shape(*
use_locking( 
;
Placeholder_5Placeholder*
dtype0*
shape:

Assign_5Assignblock2_conv1/biasPlaceholder_5*
use_locking( *
T0*$
_class
loc:@block2_conv1/bias*
validate_shape(
H
Placeholder_6Placeholder*
dtype0*
shape:

Assign_6Assignblock2_conv2/kernelPlaceholder_6*
use_locking( *
T0*&
_class
loc:@block2_conv2/kernel*
validate_shape(
;
Placeholder_7Placeholder*
dtype0*
shape:

Assign_7Assignblock2_conv2/biasPlaceholder_7*
use_locking( *
T0*$
_class
loc:@block2_conv2/bias*
validate_shape(
H
Placeholder_8Placeholder*
dtype0*
shape:

Assign_8Assignblock3_conv1/kernelPlaceholder_8*
validate_shape(*
use_locking( *
T0*&
_class
loc:@block3_conv1/kernel
;
Placeholder_9Placeholder*
dtype0*
shape:

Assign_9Assignblock3_conv1/biasPlaceholder_9*
use_locking( *
T0*$
_class
loc:@block3_conv1/bias*
validate_shape(
I
Placeholder_10Placeholder*
dtype0*
shape:

	Assign_10Assignblock3_conv2/kernelPlaceholder_10*
use_locking( *
T0*&
_class
loc:@block3_conv2/kernel*
validate_shape(
<
Placeholder_11Placeholder*
dtype0*
shape:

	Assign_11Assignblock3_conv2/biasPlaceholder_11*
validate_shape(*
use_locking( *
T0*$
_class
loc:@block3_conv2/bias
I
Placeholder_12Placeholder*
shape:*
dtype0

	Assign_12Assignblock3_conv3/kernelPlaceholder_12*
T0*&
_class
loc:@block3_conv3/kernel*
validate_shape(*
use_locking( 
<
Placeholder_13Placeholder*
dtype0*
shape:

	Assign_13Assignblock3_conv3/biasPlaceholder_13*
validate_shape(*
use_locking( *
T0*$
_class
loc:@block3_conv3/bias
I
Placeholder_14Placeholder*
shape:*
dtype0

	Assign_14Assignblock4_conv1/kernelPlaceholder_14*
T0*&
_class
loc:@block4_conv1/kernel*
validate_shape(*
use_locking( 
<
Placeholder_15Placeholder*
dtype0*
shape:

	Assign_15Assignblock4_conv1/biasPlaceholder_15*
use_locking( *
T0*$
_class
loc:@block4_conv1/bias*
validate_shape(
I
Placeholder_16Placeholder*
dtype0*
shape:

	Assign_16Assignblock4_conv2/kernelPlaceholder_16*
use_locking( *
T0*&
_class
loc:@block4_conv2/kernel*
validate_shape(
<
Placeholder_17Placeholder*
shape:*
dtype0

	Assign_17Assignblock4_conv2/biasPlaceholder_17*
use_locking( *
T0*$
_class
loc:@block4_conv2/bias*
validate_shape(
I
Placeholder_18Placeholder*
dtype0*
shape:

	Assign_18Assignblock4_conv3/kernelPlaceholder_18*
use_locking( *
T0*&
_class
loc:@block4_conv3/kernel*
validate_shape(
<
Placeholder_19Placeholder*
dtype0*
shape:

	Assign_19Assignblock4_conv3/biasPlaceholder_19*
validate_shape(*
use_locking( *
T0*$
_class
loc:@block4_conv3/bias
I
Placeholder_20Placeholder*
dtype0*
shape:

	Assign_20Assignblock5_conv1/kernelPlaceholder_20*
use_locking( *
T0*&
_class
loc:@block5_conv1/kernel*
validate_shape(
<
Placeholder_21Placeholder*
dtype0*
shape:

	Assign_21Assignblock5_conv1/biasPlaceholder_21*
validate_shape(*
use_locking( *
T0*$
_class
loc:@block5_conv1/bias
I
Placeholder_22Placeholder*
shape:*
dtype0

	Assign_22Assignblock5_conv2/kernelPlaceholder_22*
use_locking( *
T0*&
_class
loc:@block5_conv2/kernel*
validate_shape(
<
Placeholder_23Placeholder*
dtype0*
shape:

	Assign_23Assignblock5_conv2/biasPlaceholder_23*
use_locking( *
T0*$
_class
loc:@block5_conv2/bias*
validate_shape(
I
Placeholder_24Placeholder*
dtype0*
shape:

	Assign_24Assignblock5_conv3/kernelPlaceholder_24*
use_locking( *
T0*&
_class
loc:@block5_conv3/kernel*
validate_shape(
<
Placeholder_25Placeholder*
dtype0*
shape:

	Assign_25Assignblock5_conv3/biasPlaceholder_25*
use_locking( *
T0*$
_class
loc:@block5_conv3/bias*
validate_shape(
A
Placeholder_26Placeholder*
dtype0*
shape:


	Assign_26Assigndense_1/kernelPlaceholder_26*
use_locking( *
T0*!
_class
loc:@dense_1/kernel*
validate_shape(
<
Placeholder_27Placeholder*
dtype0*
shape:

	Assign_27Assigndense_1/biasPlaceholder_27*
use_locking( *
T0*
_class
loc:@dense_1/bias*
validate_shape(
@
Placeholder_28Placeholder*
dtype0*
shape:	

	Assign_28Assigndense_2/kernelPlaceholder_28*
use_locking( *
T0*!
_class
loc:@dense_2/kernel*
validate_shape(
;
Placeholder_29Placeholder*
shape:*
dtype0

	Assign_29Assigndense_2/biasPlaceholder_29*
use_locking( *
T0*
_class
loc:@dense_2/bias*
validate_shape(
x
IsVariableInitializedIsVariableInitializedblock1_conv1/kernel*
dtype0*&
_class
loc:@block1_conv1/kernel
v
IsVariableInitialized_1IsVariableInitializedblock1_conv1/bias*$
_class
loc:@block1_conv1/bias*
dtype0
z
IsVariableInitialized_2IsVariableInitializedblock1_conv2/kernel*&
_class
loc:@block1_conv2/kernel*
dtype0
v
IsVariableInitialized_3IsVariableInitializedblock1_conv2/bias*
dtype0*$
_class
loc:@block1_conv2/bias
z
IsVariableInitialized_4IsVariableInitializedblock2_conv1/kernel*&
_class
loc:@block2_conv1/kernel*
dtype0
v
IsVariableInitialized_5IsVariableInitializedblock2_conv1/bias*$
_class
loc:@block2_conv1/bias*
dtype0
z
IsVariableInitialized_6IsVariableInitializedblock2_conv2/kernel*
dtype0*&
_class
loc:@block2_conv2/kernel
v
IsVariableInitialized_7IsVariableInitializedblock2_conv2/bias*$
_class
loc:@block2_conv2/bias*
dtype0
z
IsVariableInitialized_8IsVariableInitializedblock3_conv1/kernel*&
_class
loc:@block3_conv1/kernel*
dtype0
v
IsVariableInitialized_9IsVariableInitializedblock3_conv1/bias*
dtype0*$
_class
loc:@block3_conv1/bias
{
IsVariableInitialized_10IsVariableInitializedblock3_conv2/kernel*
dtype0*&
_class
loc:@block3_conv2/kernel
w
IsVariableInitialized_11IsVariableInitializedblock3_conv2/bias*$
_class
loc:@block3_conv2/bias*
dtype0
{
IsVariableInitialized_12IsVariableInitializedblock3_conv3/kernel*&
_class
loc:@block3_conv3/kernel*
dtype0
w
IsVariableInitialized_13IsVariableInitializedblock3_conv3/bias*$
_class
loc:@block3_conv3/bias*
dtype0
{
IsVariableInitialized_14IsVariableInitializedblock4_conv1/kernel*&
_class
loc:@block4_conv1/kernel*
dtype0
w
IsVariableInitialized_15IsVariableInitializedblock4_conv1/bias*
dtype0*$
_class
loc:@block4_conv1/bias
{
IsVariableInitialized_16IsVariableInitializedblock4_conv2/kernel*&
_class
loc:@block4_conv2/kernel*
dtype0
w
IsVariableInitialized_17IsVariableInitializedblock4_conv2/bias*
dtype0*$
_class
loc:@block4_conv2/bias
{
IsVariableInitialized_18IsVariableInitializedblock4_conv3/kernel*
dtype0*&
_class
loc:@block4_conv3/kernel
w
IsVariableInitialized_19IsVariableInitializedblock4_conv3/bias*$
_class
loc:@block4_conv3/bias*
dtype0
{
IsVariableInitialized_20IsVariableInitializedblock5_conv1/kernel*&
_class
loc:@block5_conv1/kernel*
dtype0
w
IsVariableInitialized_21IsVariableInitializedblock5_conv1/bias*$
_class
loc:@block5_conv1/bias*
dtype0
{
IsVariableInitialized_22IsVariableInitializedblock5_conv2/kernel*&
_class
loc:@block5_conv2/kernel*
dtype0
w
IsVariableInitialized_23IsVariableInitializedblock5_conv2/bias*$
_class
loc:@block5_conv2/bias*
dtype0
{
IsVariableInitialized_24IsVariableInitializedblock5_conv3/kernel*&
_class
loc:@block5_conv3/kernel*
dtype0
w
IsVariableInitialized_25IsVariableInitializedblock5_conv3/bias*$
_class
loc:@block5_conv3/bias*
dtype0
q
IsVariableInitialized_26IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0
m
IsVariableInitialized_27IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0
q
IsVariableInitialized_28IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0
m
IsVariableInitialized_29IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0
Ā
initNoOp^block1_conv1/bias/Assign^block1_conv1/kernel/Assign^block1_conv2/bias/Assign^block1_conv2/kernel/Assign^block2_conv1/bias/Assign^block2_conv1/kernel/Assign^block2_conv2/bias/Assign^block2_conv2/kernel/Assign^block3_conv1/bias/Assign^block3_conv1/kernel/Assign^block3_conv2/bias/Assign^block3_conv2/kernel/Assign^block3_conv3/bias/Assign^block3_conv3/kernel/Assign^block4_conv1/bias/Assign^block4_conv1/kernel/Assign^block4_conv2/bias/Assign^block4_conv2/kernel/Assign^block4_conv3/bias/Assign^block4_conv3/kernel/Assign^block5_conv1/bias/Assign^block5_conv1/kernel/Assign^block5_conv2/bias/Assign^block5_conv2/kernel/Assign^block5_conv3/bias/Assign^block5_conv3/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign
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
dtype0*
shape: 

save/save/tensor_namesConst*Ų
valueÏBĖBblock1_conv1/biasBblock1_conv1/kernelBblock1_conv2/biasBblock1_conv2/kernelBblock2_conv1/biasBblock2_conv1/kernelBblock2_conv2/biasBblock2_conv2/kernelBblock3_conv1/biasBblock3_conv1/kernelBblock3_conv2/biasBblock3_conv2/kernelBblock3_conv3/biasBblock3_conv3/kernelBblock4_conv1/biasBblock4_conv1/kernelBblock4_conv2/biasBblock4_conv2/kernelBblock4_conv3/biasBblock4_conv3/kernelBblock5_conv1/biasBblock5_conv1/kernelBblock5_conv2/biasBblock5_conv2/kernelBblock5_conv3/biasBblock5_conv3/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernel*
dtype0

save/save/shapes_and_slicesConst*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Å
	save/save
SaveSlices
save/Constsave/save/tensor_namessave/save/shapes_and_slicesblock1_conv1/biasblock1_conv1/kernelblock1_conv2/biasblock1_conv2/kernelblock2_conv1/biasblock2_conv1/kernelblock2_conv2/biasblock2_conv2/kernelblock3_conv1/biasblock3_conv1/kernelblock3_conv2/biasblock3_conv2/kernelblock3_conv3/biasblock3_conv3/kernelblock4_conv1/biasblock4_conv1/kernelblock4_conv2/biasblock4_conv2/kernelblock4_conv3/biasblock4_conv3/kernelblock5_conv1/biasblock5_conv1/kernelblock5_conv2/biasblock5_conv2/kernelblock5_conv3/biasblock5_conv3/kerneldense_1/biasdense_1/kerneldense_2/biasdense_2/kernel*'
T"
 2
c
save/control_dependencyIdentity
save/Const
^save/save*
T0*
_class
loc:@save/Const

save/RestoreV2/tensor_namesConst"/device:CPU:0*Ų
valueÏBĖBblock1_conv1/biasBblock1_conv1/kernelBblock1_conv2/biasBblock1_conv2/kernelBblock2_conv1/biasBblock2_conv1/kernelBblock2_conv2/biasBblock2_conv2/kernelBblock3_conv1/biasBblock3_conv1/kernelBblock3_conv2/biasBblock3_conv2/kernelBblock3_conv3/biasBblock3_conv3/kernelBblock4_conv1/biasBblock4_conv1/kernelBblock4_conv2/biasBblock4_conv2/kernelBblock4_conv3/biasBblock4_conv3/kernelBblock5_conv1/biasBblock5_conv1/kernelBblock5_conv2/biasBblock5_conv2/kernelBblock5_conv3/biasBblock5_conv3/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernel*
dtype0

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ē
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*,
dtypes"
 2

save/AssignAssignblock1_conv1/biassave/RestoreV2*
T0*$
_class
loc:@block1_conv1/bias*
validate_shape(*
use_locking(

save/Assign_1Assignblock1_conv1/kernelsave/RestoreV2:1*
use_locking(*
T0*&
_class
loc:@block1_conv1/kernel*
validate_shape(

save/Assign_2Assignblock1_conv2/biassave/RestoreV2:2*
validate_shape(*
use_locking(*
T0*$
_class
loc:@block1_conv2/bias

save/Assign_3Assignblock1_conv2/kernelsave/RestoreV2:3*
use_locking(*
T0*&
_class
loc:@block1_conv2/kernel*
validate_shape(

save/Assign_4Assignblock2_conv1/biassave/RestoreV2:4*
T0*$
_class
loc:@block2_conv1/bias*
validate_shape(*
use_locking(

save/Assign_5Assignblock2_conv1/kernelsave/RestoreV2:5*
use_locking(*
T0*&
_class
loc:@block2_conv1/kernel*
validate_shape(

save/Assign_6Assignblock2_conv2/biassave/RestoreV2:6*
use_locking(*
T0*$
_class
loc:@block2_conv2/bias*
validate_shape(

save/Assign_7Assignblock2_conv2/kernelsave/RestoreV2:7*
validate_shape(*
use_locking(*
T0*&
_class
loc:@block2_conv2/kernel

save/Assign_8Assignblock3_conv1/biassave/RestoreV2:8*
validate_shape(*
use_locking(*
T0*$
_class
loc:@block3_conv1/bias

save/Assign_9Assignblock3_conv1/kernelsave/RestoreV2:9*
use_locking(*
T0*&
_class
loc:@block3_conv1/kernel*
validate_shape(

save/Assign_10Assignblock3_conv2/biassave/RestoreV2:10*
use_locking(*
T0*$
_class
loc:@block3_conv2/bias*
validate_shape(

save/Assign_11Assignblock3_conv2/kernelsave/RestoreV2:11*
use_locking(*
T0*&
_class
loc:@block3_conv2/kernel*
validate_shape(

save/Assign_12Assignblock3_conv3/biassave/RestoreV2:12*
T0*$
_class
loc:@block3_conv3/bias*
validate_shape(*
use_locking(

save/Assign_13Assignblock3_conv3/kernelsave/RestoreV2:13*
use_locking(*
T0*&
_class
loc:@block3_conv3/kernel*
validate_shape(

save/Assign_14Assignblock4_conv1/biassave/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@block4_conv1/bias*
validate_shape(

save/Assign_15Assignblock4_conv1/kernelsave/RestoreV2:15*
validate_shape(*
use_locking(*
T0*&
_class
loc:@block4_conv1/kernel

save/Assign_16Assignblock4_conv2/biassave/RestoreV2:16*
use_locking(*
T0*$
_class
loc:@block4_conv2/bias*
validate_shape(

save/Assign_17Assignblock4_conv2/kernelsave/RestoreV2:17*
use_locking(*
T0*&
_class
loc:@block4_conv2/kernel*
validate_shape(

save/Assign_18Assignblock4_conv3/biassave/RestoreV2:18*
use_locking(*
T0*$
_class
loc:@block4_conv3/bias*
validate_shape(

save/Assign_19Assignblock4_conv3/kernelsave/RestoreV2:19*
use_locking(*
T0*&
_class
loc:@block4_conv3/kernel*
validate_shape(

save/Assign_20Assignblock5_conv1/biassave/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@block5_conv1/bias*
validate_shape(

save/Assign_21Assignblock5_conv1/kernelsave/RestoreV2:21*
T0*&
_class
loc:@block5_conv1/kernel*
validate_shape(*
use_locking(

save/Assign_22Assignblock5_conv2/biassave/RestoreV2:22*
T0*$
_class
loc:@block5_conv2/bias*
validate_shape(*
use_locking(

save/Assign_23Assignblock5_conv2/kernelsave/RestoreV2:23*
use_locking(*
T0*&
_class
loc:@block5_conv2/kernel*
validate_shape(

save/Assign_24Assignblock5_conv3/biassave/RestoreV2:24*
validate_shape(*
use_locking(*
T0*$
_class
loc:@block5_conv3/bias

save/Assign_25Assignblock5_conv3/kernelsave/RestoreV2:25*
use_locking(*
T0*&
_class
loc:@block5_conv3/kernel*
validate_shape(

save/Assign_26Assigndense_1/biassave/RestoreV2:26*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(

save/Assign_27Assigndense_1/kernelsave/RestoreV2:27*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
use_locking(

save/Assign_28Assigndense_2/biassave/RestoreV2:28*
validate_shape(*
use_locking(*
T0*
_class
loc:@dense_2/bias

save/Assign_29Assigndense_2/kernelsave/RestoreV2:29*
validate_shape(*
use_locking(*
T0*!
_class
loc:@dense_2/kernel

save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9"