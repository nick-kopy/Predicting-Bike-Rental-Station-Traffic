Ⱦ
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02unknown8??
z
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_34/kernel
s
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes

:d*
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
?
gru_16/gru_cell_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_namegru_16/gru_cell_20/kernel
?
-gru_16/gru_cell_20/kernel/Read/ReadVariableOpReadVariableOpgru_16/gru_cell_20/kernel*
_output_shapes
:	?*
dtype0
?
#gru_16/gru_cell_20/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*4
shared_name%#gru_16/gru_cell_20/recurrent_kernel
?
7gru_16/gru_cell_20/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_16/gru_cell_20/recurrent_kernel*
_output_shapes
:	d?*
dtype0
?
gru_16/gru_cell_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_namegru_16/gru_cell_20/bias
?
+gru_16/gru_cell_20/bias/Read/ReadVariableOpReadVariableOpgru_16/gru_cell_20/bias*
_output_shapes
:	?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
RMSprop/dense_34/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*,
shared_nameRMSprop/dense_34/kernel/rms
?
/RMSprop/dense_34/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_34/kernel/rms*
_output_shapes

:d*
dtype0
?
RMSprop/dense_34/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/dense_34/bias/rms
?
-RMSprop/dense_34/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_34/bias/rms*
_output_shapes
:*
dtype0
?
%RMSprop/gru_16/gru_cell_20/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*6
shared_name'%RMSprop/gru_16/gru_cell_20/kernel/rms
?
9RMSprop/gru_16/gru_cell_20/kernel/rms/Read/ReadVariableOpReadVariableOp%RMSprop/gru_16/gru_cell_20/kernel/rms*
_output_shapes
:	?*
dtype0
?
/RMSprop/gru_16/gru_cell_20/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*@
shared_name1/RMSprop/gru_16/gru_cell_20/recurrent_kernel/rms
?
CRMSprop/gru_16/gru_cell_20/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp/RMSprop/gru_16/gru_cell_20/recurrent_kernel/rms*
_output_shapes
:	d?*
dtype0
?
#RMSprop/gru_16/gru_cell_20/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#RMSprop/gru_16/gru_cell_20/bias/rms
?
7RMSprop/gru_16/gru_cell_20/bias/rms/Read/ReadVariableOpReadVariableOp#RMSprop/gru_16/gru_cell_20/bias/rms*
_output_shapes
:	?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
?

cell
_inbound_nodes

state_spec
_outbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
{
_inbound_nodes
_outbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
|
_inbound_nodes

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
v
iter
	 decay
!learning_rate
"momentum
#rho	rmsJ	rmsK	$rmsL	%rmsM	&rmsN
#
$0
%1
&2
3
4
#
$0
%1
&2
3
4
 
?
'layer_metrics
(metrics
	variables
trainable_variables
)non_trainable_variables
regularization_losses
*layer_regularization_losses

+layers
 
~

$kernel
%recurrent_kernel
&bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
 
 
 

$0
%1
&2

$0
%1
&2
 
?
0layer_metrics
1metrics
	variables
trainable_variables
2non_trainable_variables
regularization_losses
3layer_regularization_losses

4layers

5states
 
 
 
 
 
?
6layer_metrics
7metrics
	variables
trainable_variables
8non_trainable_variables
regularization_losses
9layer_regularization_losses

:layers
 
[Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_34/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
;layer_metrics
<metrics
	variables
trainable_variables
=non_trainable_variables
regularization_losses
>layer_regularization_losses

?layers
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEgru_16/gru_cell_20/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#gru_16/gru_cell_20/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEgru_16/gru_cell_20/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

@0
 
 

0
1
2

$0
%1
&2

$0
%1
&2
 
?
Alayer_metrics
Bmetrics
,	variables
-trainable_variables
Cnon_trainable_variables
.regularization_losses
Dlayer_regularization_losses

Elayers
 
 
 
 


0
 
 
 
 
 
 
 
 
 
 
 
4
	Ftotal
	Gcount
H	variables
I	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

H	variables
??
VARIABLE_VALUERMSprop/dense_34/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/dense_34/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE%RMSprop/gru_16/gru_cell_20/kernel/rmsDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/RMSprop/gru_16/gru_cell_20/recurrent_kernel/rmsDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE#RMSprop/gru_16/gru_cell_20/bias/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_gru_16_inputPlaceholder*+
_output_shapes
:?????????x*
dtype0* 
shape:?????????x
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_16_inputgru_16/gru_cell_20/biasgru_16/gru_cell_20/kernel#gru_16/gru_cell_20/recurrent_kerneldense_34/kerneldense_34/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_514516
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp-gru_16/gru_cell_20/kernel/Read/ReadVariableOp7gru_16/gru_cell_20/recurrent_kernel/Read/ReadVariableOp+gru_16/gru_cell_20/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/RMSprop/dense_34/kernel/rms/Read/ReadVariableOp-RMSprop/dense_34/bias/rms/Read/ReadVariableOp9RMSprop/gru_16/gru_cell_20/kernel/rms/Read/ReadVariableOpCRMSprop/gru_16/gru_cell_20/recurrent_kernel/rms/Read/ReadVariableOp7RMSprop/gru_16/gru_cell_20/bias/rms/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_516167
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_34/kerneldense_34/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhogru_16/gru_cell_20/kernel#gru_16/gru_cell_20/recurrent_kernelgru_16/gru_cell_20/biastotalcountRMSprop/dense_34/kernel/rmsRMSprop/dense_34/bias/rms%RMSprop/gru_16/gru_cell_20/kernel/rms/RMSprop/gru_16/gru_cell_20/recurrent_kernel/rms#RMSprop/gru_16/gru_cell_20/bias/rms*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_516228??
?v
?
I__inference_sequential_34_layer_call_and_return_conditional_losses_514690
gru_16_input.
*gru_16_gru_cell_20_readvariableop_resource5
1gru_16_gru_cell_20_matmul_readvariableop_resource7
3gru_16_gru_cell_20_matmul_1_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource
identity??gru_16/whileX
gru_16/ShapeShapegru_16_input*
T0*
_output_shapes
:2
gru_16/Shape?
gru_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_16/strided_slice/stack?
gru_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_16/strided_slice/stack_1?
gru_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_16/strided_slice/stack_2?
gru_16/strided_sliceStridedSlicegru_16/Shape:output:0#gru_16/strided_slice/stack:output:0%gru_16/strided_slice/stack_1:output:0%gru_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_16/strided_slicej
gru_16/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
gru_16/zeros/mul/y?
gru_16/zeros/mulMulgru_16/strided_slice:output:0gru_16/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_16/zeros/mulm
gru_16/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_16/zeros/Less/y?
gru_16/zeros/LessLessgru_16/zeros/mul:z:0gru_16/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_16/zeros/Lessp
gru_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
gru_16/zeros/packed/1?
gru_16/zeros/packedPackgru_16/strided_slice:output:0gru_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_16/zeros/packedm
gru_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_16/zeros/Const?
gru_16/zerosFillgru_16/zeros/packed:output:0gru_16/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/zeros?
gru_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_16/transpose/perm?
gru_16/transpose	Transposegru_16_inputgru_16/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2
gru_16/transposed
gru_16/Shape_1Shapegru_16/transpose:y:0*
T0*
_output_shapes
:2
gru_16/Shape_1?
gru_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_16/strided_slice_1/stack?
gru_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_1/stack_1?
gru_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_1/stack_2?
gru_16/strided_slice_1StridedSlicegru_16/Shape_1:output:0%gru_16/strided_slice_1/stack:output:0'gru_16/strided_slice_1/stack_1:output:0'gru_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_16/strided_slice_1?
"gru_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_16/TensorArrayV2/element_shape?
gru_16/TensorArrayV2TensorListReserve+gru_16/TensorArrayV2/element_shape:output:0gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_16/TensorArrayV2?
<gru_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_16/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_16/transpose:y:0Egru_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_16/TensorArrayUnstack/TensorListFromTensor?
gru_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_16/strided_slice_2/stack?
gru_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_2/stack_1?
gru_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_2/stack_2?
gru_16/strided_slice_2StridedSlicegru_16/transpose:y:0%gru_16/strided_slice_2/stack:output:0'gru_16/strided_slice_2/stack_1:output:0'gru_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_16/strided_slice_2?
!gru_16/gru_cell_20/ReadVariableOpReadVariableOp*gru_16_gru_cell_20_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_16/gru_cell_20/ReadVariableOp?
gru_16/gru_cell_20/unstackUnpack)gru_16/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_16/gru_cell_20/unstack?
(gru_16/gru_cell_20/MatMul/ReadVariableOpReadVariableOp1gru_16_gru_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(gru_16/gru_cell_20/MatMul/ReadVariableOp?
gru_16/gru_cell_20/MatMulMatMulgru_16/strided_slice_2:output:00gru_16/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/MatMul?
gru_16/gru_cell_20/BiasAddBiasAdd#gru_16/gru_cell_20/MatMul:product:0#gru_16/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/BiasAddv
gru_16/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_16/gru_cell_20/Const?
"gru_16/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_16/gru_cell_20/split/split_dim?
gru_16/gru_cell_20/splitSplit+gru_16/gru_cell_20/split/split_dim:output:0#gru_16/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_16/gru_cell_20/split?
*gru_16/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp3gru_16_gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02,
*gru_16/gru_cell_20/MatMul_1/ReadVariableOp?
gru_16/gru_cell_20/MatMul_1MatMulgru_16/zeros:output:02gru_16/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/MatMul_1?
gru_16/gru_cell_20/BiasAdd_1BiasAdd%gru_16/gru_cell_20/MatMul_1:product:0#gru_16/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/BiasAdd_1?
gru_16/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_16/gru_cell_20/Const_1?
$gru_16/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_16/gru_cell_20/split_1/split_dim?
gru_16/gru_cell_20/split_1SplitV%gru_16/gru_cell_20/BiasAdd_1:output:0#gru_16/gru_cell_20/Const_1:output:0-gru_16/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_16/gru_cell_20/split_1?
gru_16/gru_cell_20/addAddV2!gru_16/gru_cell_20/split:output:0#gru_16/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add?
gru_16/gru_cell_20/SigmoidSigmoidgru_16/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/Sigmoid?
gru_16/gru_cell_20/add_1AddV2!gru_16/gru_cell_20/split:output:1#gru_16/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add_1?
gru_16/gru_cell_20/Sigmoid_1Sigmoidgru_16/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/Sigmoid_1?
gru_16/gru_cell_20/mulMul gru_16/gru_cell_20/Sigmoid_1:y:0#gru_16/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/mul?
gru_16/gru_cell_20/add_2AddV2!gru_16/gru_cell_20/split:output:2gru_16/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add_2?
gru_16/gru_cell_20/TanhTanhgru_16/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/Tanh?
gru_16/gru_cell_20/mul_1Mulgru_16/gru_cell_20/Sigmoid:y:0gru_16/zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/mul_1y
gru_16/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_16/gru_cell_20/sub/x?
gru_16/gru_cell_20/subSub!gru_16/gru_cell_20/sub/x:output:0gru_16/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/sub?
gru_16/gru_cell_20/mul_2Mulgru_16/gru_cell_20/sub:z:0gru_16/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/mul_2?
gru_16/gru_cell_20/add_3AddV2gru_16/gru_cell_20/mul_1:z:0gru_16/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add_3?
$gru_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2&
$gru_16/TensorArrayV2_1/element_shape?
gru_16/TensorArrayV2_1TensorListReserve-gru_16/TensorArrayV2_1/element_shape:output:0gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_16/TensorArrayV2_1\
gru_16/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_16/time?
gru_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_16/while/maximum_iterationsx
gru_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_16/while/loop_counter?
gru_16/whileWhile"gru_16/while/loop_counter:output:0(gru_16/while/maximum_iterations:output:0gru_16/time:output:0gru_16/TensorArrayV2_1:handle:0gru_16/zeros:output:0gru_16/strided_slice_1:output:0>gru_16/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_16_gru_cell_20_readvariableop_resource1gru_16_gru_cell_20_matmul_readvariableop_resource3gru_16_gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*$
bodyR
gru_16_while_body_514585*$
condR
gru_16_while_cond_514584*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
gru_16/while?
7gru_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7gru_16/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_16/TensorArrayV2Stack/TensorListStackTensorListStackgru_16/while:output:3@gru_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype02+
)gru_16/TensorArrayV2Stack/TensorListStack?
gru_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_16/strided_slice_3/stack?
gru_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_16/strided_slice_3/stack_1?
gru_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_3/stack_2?
gru_16/strided_slice_3StridedSlice2gru_16/TensorArrayV2Stack/TensorListStack:tensor:0%gru_16/strided_slice_3/stack:output:0'gru_16/strided_slice_3/stack_1:output:0'gru_16/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
gru_16/strided_slice_3?
gru_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_16/transpose_1/perm?
gru_16/transpose_1	Transpose2gru_16/TensorArrayV2Stack/TensorListStack:tensor:0 gru_16/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2
gru_16/transpose_1t
gru_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_16/runtimey
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_17/dropout/Const?
dropout_17/dropout/MulMulgru_16/strided_slice_3:output:0!dropout_17/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout_17/dropout/Mul?
dropout_17/dropout/ShapeShapegru_16/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape?
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype021
/dropout_17/dropout/random_uniform/RandomUniform?
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_17/dropout/GreaterEqual/y?
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2!
dropout_17/dropout/GreaterEqual?
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout_17/dropout/Cast?
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout_17/dropout/Mul_1?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldropout_17/dropout/Mul_1:z:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_34/BiasAdds
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_34/Relu~
IdentityIdentitydense_34/Relu:activations:0^gru_16/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::2
gru_16/whilegru_16/while:Y U
+
_output_shapes
:?????????x
&
_user_specified_namegru_16_input
?!
?
while_body_513809
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_20_513831_0
while_gru_cell_20_513833_0
while_gru_cell_20_513835_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_20_513831
while_gru_cell_20_513833
while_gru_cell_20_513835??)while/gru_cell_20/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/gru_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_20_513831_0while_gru_cell_20_513833_0while_gru_cell_20_513835_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_5135102+
)while/gru_cell_20/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_20/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/gru_cell_20/StatefulPartitionedCall:output:1*^while/gru_cell_20/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2
while/Identity_4"6
while_gru_cell_20_513831while_gru_cell_20_513831_0"6
while_gru_cell_20_513833while_gru_cell_20_513833_0"6
while_gru_cell_20_513835while_gru_cell_20_513835_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::2V
)while/gru_cell_20/StatefulPartitionedCall)while/gru_cell_20/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
G
+__inference_dropout_17_layer_call_fn_515965

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_5143682
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?W
?
B__inference_gru_16_layer_call_and_return_conditional_losses_515916

inputs'
#gru_cell_20_readvariableop_resource.
*gru_cell_20_matmul_readvariableop_resource0
,gru_cell_20_matmul_1_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:x?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_20/ReadVariableOpReadVariableOp#gru_cell_20_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_20/ReadVariableOp?
gru_cell_20/unstackUnpack"gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_20/unstack?
!gru_cell_20/MatMul/ReadVariableOpReadVariableOp*gru_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_20/MatMul/ReadVariableOp?
gru_cell_20/MatMulMatMulstrided_slice_2:output:0)gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/MatMul?
gru_cell_20/BiasAddBiasAddgru_cell_20/MatMul:product:0gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/BiasAddh
gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_20/Const?
gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_20/split/split_dim?
gru_cell_20/splitSplit$gru_cell_20/split/split_dim:output:0gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_20/split?
#gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02%
#gru_cell_20/MatMul_1/ReadVariableOp?
gru_cell_20/MatMul_1MatMulzeros:output:0+gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/MatMul_1?
gru_cell_20/BiasAdd_1BiasAddgru_cell_20/MatMul_1:product:0gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_20/BiasAdd_1
gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_cell_20/Const_1?
gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_20/split_1/split_dim?
gru_cell_20/split_1SplitVgru_cell_20/BiasAdd_1:output:0gru_cell_20/Const_1:output:0&gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_20/split_1?
gru_cell_20/addAddV2gru_cell_20/split:output:0gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add|
gru_cell_20/SigmoidSigmoidgru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Sigmoid?
gru_cell_20/add_1AddV2gru_cell_20/split:output:1gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_1?
gru_cell_20/Sigmoid_1Sigmoidgru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Sigmoid_1?
gru_cell_20/mulMulgru_cell_20/Sigmoid_1:y:0gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul?
gru_cell_20/add_2AddV2gru_cell_20/split:output:2gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_2u
gru_cell_20/TanhTanhgru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Tanh?
gru_cell_20/mul_1Mulgru_cell_20/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul_1k
gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_20/sub/x?
gru_cell_20/subSubgru_cell_20/sub/x:output:0gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/sub?
gru_cell_20/mul_2Mulgru_cell_20/sub:z:0gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul_2?
gru_cell_20/add_3AddV2gru_cell_20/mul_1:z:0gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_20_readvariableop_resource*gru_cell_20_matmul_readvariableop_resource,gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_515826*
condR
while_cond_515825*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????x:::2
whilewhile:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_514368

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?l
?
I__inference_sequential_34_layer_call_and_return_conditional_losses_515228

inputs.
*gru_16_gru_cell_20_readvariableop_resource5
1gru_16_gru_cell_20_matmul_readvariableop_resource7
3gru_16_gru_cell_20_matmul_1_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource
identity??gru_16/whileR
gru_16/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_16/Shape?
gru_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_16/strided_slice/stack?
gru_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_16/strided_slice/stack_1?
gru_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_16/strided_slice/stack_2?
gru_16/strided_sliceStridedSlicegru_16/Shape:output:0#gru_16/strided_slice/stack:output:0%gru_16/strided_slice/stack_1:output:0%gru_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_16/strided_slicej
gru_16/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
gru_16/zeros/mul/y?
gru_16/zeros/mulMulgru_16/strided_slice:output:0gru_16/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_16/zeros/mulm
gru_16/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_16/zeros/Less/y?
gru_16/zeros/LessLessgru_16/zeros/mul:z:0gru_16/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_16/zeros/Lessp
gru_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
gru_16/zeros/packed/1?
gru_16/zeros/packedPackgru_16/strided_slice:output:0gru_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_16/zeros/packedm
gru_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_16/zeros/Const?
gru_16/zerosFillgru_16/zeros/packed:output:0gru_16/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/zeros?
gru_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_16/transpose/perm?
gru_16/transpose	Transposeinputsgru_16/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2
gru_16/transposed
gru_16/Shape_1Shapegru_16/transpose:y:0*
T0*
_output_shapes
:2
gru_16/Shape_1?
gru_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_16/strided_slice_1/stack?
gru_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_1/stack_1?
gru_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_1/stack_2?
gru_16/strided_slice_1StridedSlicegru_16/Shape_1:output:0%gru_16/strided_slice_1/stack:output:0'gru_16/strided_slice_1/stack_1:output:0'gru_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_16/strided_slice_1?
"gru_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_16/TensorArrayV2/element_shape?
gru_16/TensorArrayV2TensorListReserve+gru_16/TensorArrayV2/element_shape:output:0gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_16/TensorArrayV2?
<gru_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_16/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_16/transpose:y:0Egru_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_16/TensorArrayUnstack/TensorListFromTensor?
gru_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_16/strided_slice_2/stack?
gru_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_2/stack_1?
gru_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_2/stack_2?
gru_16/strided_slice_2StridedSlicegru_16/transpose:y:0%gru_16/strided_slice_2/stack:output:0'gru_16/strided_slice_2/stack_1:output:0'gru_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_16/strided_slice_2?
!gru_16/gru_cell_20/ReadVariableOpReadVariableOp*gru_16_gru_cell_20_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_16/gru_cell_20/ReadVariableOp?
gru_16/gru_cell_20/unstackUnpack)gru_16/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_16/gru_cell_20/unstack?
(gru_16/gru_cell_20/MatMul/ReadVariableOpReadVariableOp1gru_16_gru_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(gru_16/gru_cell_20/MatMul/ReadVariableOp?
gru_16/gru_cell_20/MatMulMatMulgru_16/strided_slice_2:output:00gru_16/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/MatMul?
gru_16/gru_cell_20/BiasAddBiasAdd#gru_16/gru_cell_20/MatMul:product:0#gru_16/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/BiasAddv
gru_16/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_16/gru_cell_20/Const?
"gru_16/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_16/gru_cell_20/split/split_dim?
gru_16/gru_cell_20/splitSplit+gru_16/gru_cell_20/split/split_dim:output:0#gru_16/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_16/gru_cell_20/split?
*gru_16/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp3gru_16_gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02,
*gru_16/gru_cell_20/MatMul_1/ReadVariableOp?
gru_16/gru_cell_20/MatMul_1MatMulgru_16/zeros:output:02gru_16/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/MatMul_1?
gru_16/gru_cell_20/BiasAdd_1BiasAdd%gru_16/gru_cell_20/MatMul_1:product:0#gru_16/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/BiasAdd_1?
gru_16/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_16/gru_cell_20/Const_1?
$gru_16/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_16/gru_cell_20/split_1/split_dim?
gru_16/gru_cell_20/split_1SplitV%gru_16/gru_cell_20/BiasAdd_1:output:0#gru_16/gru_cell_20/Const_1:output:0-gru_16/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_16/gru_cell_20/split_1?
gru_16/gru_cell_20/addAddV2!gru_16/gru_cell_20/split:output:0#gru_16/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add?
gru_16/gru_cell_20/SigmoidSigmoidgru_16/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/Sigmoid?
gru_16/gru_cell_20/add_1AddV2!gru_16/gru_cell_20/split:output:1#gru_16/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add_1?
gru_16/gru_cell_20/Sigmoid_1Sigmoidgru_16/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/Sigmoid_1?
gru_16/gru_cell_20/mulMul gru_16/gru_cell_20/Sigmoid_1:y:0#gru_16/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/mul?
gru_16/gru_cell_20/add_2AddV2!gru_16/gru_cell_20/split:output:2gru_16/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add_2?
gru_16/gru_cell_20/TanhTanhgru_16/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/Tanh?
gru_16/gru_cell_20/mul_1Mulgru_16/gru_cell_20/Sigmoid:y:0gru_16/zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/mul_1y
gru_16/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_16/gru_cell_20/sub/x?
gru_16/gru_cell_20/subSub!gru_16/gru_cell_20/sub/x:output:0gru_16/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/sub?
gru_16/gru_cell_20/mul_2Mulgru_16/gru_cell_20/sub:z:0gru_16/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/mul_2?
gru_16/gru_cell_20/add_3AddV2gru_16/gru_cell_20/mul_1:z:0gru_16/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add_3?
$gru_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2&
$gru_16/TensorArrayV2_1/element_shape?
gru_16/TensorArrayV2_1TensorListReserve-gru_16/TensorArrayV2_1/element_shape:output:0gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_16/TensorArrayV2_1\
gru_16/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_16/time?
gru_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_16/while/maximum_iterationsx
gru_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_16/while/loop_counter?
gru_16/whileWhile"gru_16/while/loop_counter:output:0(gru_16/while/maximum_iterations:output:0gru_16/time:output:0gru_16/TensorArrayV2_1:handle:0gru_16/zeros:output:0gru_16/strided_slice_1:output:0>gru_16/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_16_gru_cell_20_readvariableop_resource1gru_16_gru_cell_20_matmul_readvariableop_resource3gru_16_gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*$
bodyR
gru_16_while_body_515130*$
condR
gru_16_while_cond_515129*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
gru_16/while?
7gru_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7gru_16/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_16/TensorArrayV2Stack/TensorListStackTensorListStackgru_16/while:output:3@gru_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype02+
)gru_16/TensorArrayV2Stack/TensorListStack?
gru_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_16/strided_slice_3/stack?
gru_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_16/strided_slice_3/stack_1?
gru_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_3/stack_2?
gru_16/strided_slice_3StridedSlice2gru_16/TensorArrayV2Stack/TensorListStack:tensor:0%gru_16/strided_slice_3/stack:output:0'gru_16/strided_slice_3/stack_1:output:0'gru_16/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
gru_16/strided_slice_3?
gru_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_16/transpose_1/perm?
gru_16/transpose_1	Transpose2gru_16/TensorArrayV2Stack/TensorListStack:tensor:0 gru_16/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2
gru_16/transpose_1t
gru_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_16/runtime?
dropout_17/IdentityIdentitygru_16/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d2
dropout_17/Identity?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldropout_17/Identity:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_34/BiasAdds
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_34/Relu~
IdentityIdentitydense_34/Relu:activations:0^gru_16/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::2
gru_16/whilegru_16/while:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?	
?
gru_16_while_cond_514955*
&gru_16_while_gru_16_while_loop_counter0
,gru_16_while_gru_16_while_maximum_iterations
gru_16_while_placeholder
gru_16_while_placeholder_1
gru_16_while_placeholder_2,
(gru_16_while_less_gru_16_strided_slice_1B
>gru_16_while_gru_16_while_cond_514955___redundant_placeholder0B
>gru_16_while_gru_16_while_cond_514955___redundant_placeholder1B
>gru_16_while_gru_16_while_cond_514955___redundant_placeholder2B
>gru_16_while_gru_16_while_cond_514955___redundant_placeholder3
gru_16_while_identity
?
gru_16/while/LessLessgru_16_while_placeholder(gru_16_while_less_gru_16_strided_slice_1*
T0*
_output_shapes
: 2
gru_16/while/Lessr
gru_16/while/IdentityIdentitygru_16/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_16/while/Identity"7
gru_16_while_identitygru_16/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
.__inference_sequential_34_layer_call_fn_514872
gru_16_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_5144462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????x
&
_user_specified_namegru_16_input
?W
?
B__inference_gru_16_layer_call_and_return_conditional_losses_514321

inputs'
#gru_cell_20_readvariableop_resource.
*gru_cell_20_matmul_readvariableop_resource0
,gru_cell_20_matmul_1_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:x?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_20/ReadVariableOpReadVariableOp#gru_cell_20_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_20/ReadVariableOp?
gru_cell_20/unstackUnpack"gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_20/unstack?
!gru_cell_20/MatMul/ReadVariableOpReadVariableOp*gru_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_20/MatMul/ReadVariableOp?
gru_cell_20/MatMulMatMulstrided_slice_2:output:0)gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/MatMul?
gru_cell_20/BiasAddBiasAddgru_cell_20/MatMul:product:0gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/BiasAddh
gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_20/Const?
gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_20/split/split_dim?
gru_cell_20/splitSplit$gru_cell_20/split/split_dim:output:0gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_20/split?
#gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02%
#gru_cell_20/MatMul_1/ReadVariableOp?
gru_cell_20/MatMul_1MatMulzeros:output:0+gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/MatMul_1?
gru_cell_20/BiasAdd_1BiasAddgru_cell_20/MatMul_1:product:0gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_20/BiasAdd_1
gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_cell_20/Const_1?
gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_20/split_1/split_dim?
gru_cell_20/split_1SplitVgru_cell_20/BiasAdd_1:output:0gru_cell_20/Const_1:output:0&gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_20/split_1?
gru_cell_20/addAddV2gru_cell_20/split:output:0gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add|
gru_cell_20/SigmoidSigmoidgru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Sigmoid?
gru_cell_20/add_1AddV2gru_cell_20/split:output:1gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_1?
gru_cell_20/Sigmoid_1Sigmoidgru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Sigmoid_1?
gru_cell_20/mulMulgru_cell_20/Sigmoid_1:y:0gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul?
gru_cell_20/add_2AddV2gru_cell_20/split:output:2gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_2u
gru_cell_20/TanhTanhgru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Tanh?
gru_cell_20/mul_1Mulgru_cell_20/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul_1k
gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_20/sub/x?
gru_cell_20/subSubgru_cell_20/sub/x:output:0gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/sub?
gru_cell_20/mul_2Mulgru_cell_20/sub:z:0gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul_2?
gru_cell_20/add_3AddV2gru_cell_20/mul_1:z:0gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_20_readvariableop_resource*gru_cell_20_matmul_readvariableop_resource,gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_514231*
condR
while_cond_514230*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????x:::2
whilewhile:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?!
?
while_body_513927
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_20_513949_0
while_gru_cell_20_513951_0
while_gru_cell_20_513953_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_20_513949
while_gru_cell_20_513951
while_gru_cell_20_513953??)while/gru_cell_20/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/gru_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_20_513949_0while_gru_cell_20_513951_0while_gru_cell_20_513953_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_5135502+
)while/gru_cell_20/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_20/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/gru_cell_20/StatefulPartitionedCall:output:1*^while/gru_cell_20/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2
while/Identity_4"6
while_gru_cell_20_513949while_gru_cell_20_513949_0"6
while_gru_cell_20_513951while_gru_cell_20_513951_0"6
while_gru_cell_20_513953while_gru_cell_20_513953_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::2V
)while/gru_cell_20/StatefulPartitionedCall)while/gru_cell_20/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_gru_16_layer_call_fn_515598
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_16_layer_call_and_return_conditional_losses_5139912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
'__inference_gru_16_layer_call_fn_515927

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_16_layer_call_and_return_conditional_losses_5141622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????x:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
while_cond_513926
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_513926___redundant_placeholder04
0while_while_cond_513926___redundant_placeholder14
0while_while_cond_513926___redundant_placeholder24
0while_while_cond_513926___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?^
?

&sequential_34_gru_16_while_body_513340F
Bsequential_34_gru_16_while_sequential_34_gru_16_while_loop_counterL
Hsequential_34_gru_16_while_sequential_34_gru_16_while_maximum_iterations*
&sequential_34_gru_16_while_placeholder,
(sequential_34_gru_16_while_placeholder_1,
(sequential_34_gru_16_while_placeholder_2E
Asequential_34_gru_16_while_sequential_34_gru_16_strided_slice_1_0?
}sequential_34_gru_16_while_tensorarrayv2read_tensorlistgetitem_sequential_34_gru_16_tensorarrayunstack_tensorlistfromtensor_0D
@sequential_34_gru_16_while_gru_cell_20_readvariableop_resource_0K
Gsequential_34_gru_16_while_gru_cell_20_matmul_readvariableop_resource_0M
Isequential_34_gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0'
#sequential_34_gru_16_while_identity)
%sequential_34_gru_16_while_identity_1)
%sequential_34_gru_16_while_identity_2)
%sequential_34_gru_16_while_identity_3)
%sequential_34_gru_16_while_identity_4C
?sequential_34_gru_16_while_sequential_34_gru_16_strided_slice_1
{sequential_34_gru_16_while_tensorarrayv2read_tensorlistgetitem_sequential_34_gru_16_tensorarrayunstack_tensorlistfromtensorB
>sequential_34_gru_16_while_gru_cell_20_readvariableop_resourceI
Esequential_34_gru_16_while_gru_cell_20_matmul_readvariableop_resourceK
Gsequential_34_gru_16_while_gru_cell_20_matmul_1_readvariableop_resource??
Lsequential_34/gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2N
Lsequential_34/gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_34/gru_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_34_gru_16_while_tensorarrayv2read_tensorlistgetitem_sequential_34_gru_16_tensorarrayunstack_tensorlistfromtensor_0&sequential_34_gru_16_while_placeholderUsequential_34/gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02@
>sequential_34/gru_16/while/TensorArrayV2Read/TensorListGetItem?
5sequential_34/gru_16/while/gru_cell_20/ReadVariableOpReadVariableOp@sequential_34_gru_16_while_gru_cell_20_readvariableop_resource_0*
_output_shapes
:	?*
dtype027
5sequential_34/gru_16/while/gru_cell_20/ReadVariableOp?
.sequential_34/gru_16/while/gru_cell_20/unstackUnpack=sequential_34/gru_16/while/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num20
.sequential_34/gru_16/while/gru_cell_20/unstack?
<sequential_34/gru_16/while/gru_cell_20/MatMul/ReadVariableOpReadVariableOpGsequential_34_gru_16_while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02>
<sequential_34/gru_16/while/gru_cell_20/MatMul/ReadVariableOp?
-sequential_34/gru_16/while/gru_cell_20/MatMulMatMulEsequential_34/gru_16/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential_34/gru_16/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-sequential_34/gru_16/while/gru_cell_20/MatMul?
.sequential_34/gru_16/while/gru_cell_20/BiasAddBiasAdd7sequential_34/gru_16/while/gru_cell_20/MatMul:product:07sequential_34/gru_16/while/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????20
.sequential_34/gru_16/while/gru_cell_20/BiasAdd?
,sequential_34/gru_16/while/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_34/gru_16/while/gru_cell_20/Const?
6sequential_34/gru_16/while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6sequential_34/gru_16/while/gru_cell_20/split/split_dim?
,sequential_34/gru_16/while/gru_cell_20/splitSplit?sequential_34/gru_16/while/gru_cell_20/split/split_dim:output:07sequential_34/gru_16/while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2.
,sequential_34/gru_16/while/gru_cell_20/split?
>sequential_34/gru_16/while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOpIsequential_34_gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02@
>sequential_34/gru_16/while/gru_cell_20/MatMul_1/ReadVariableOp?
/sequential_34/gru_16/while/gru_cell_20/MatMul_1MatMul(sequential_34_gru_16_while_placeholder_2Fsequential_34/gru_16/while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_34/gru_16/while/gru_cell_20/MatMul_1?
0sequential_34/gru_16/while/gru_cell_20/BiasAdd_1BiasAdd9sequential_34/gru_16/while/gru_cell_20/MatMul_1:product:07sequential_34/gru_16/while/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????22
0sequential_34/gru_16/while/gru_cell_20/BiasAdd_1?
.sequential_34/gru_16/while/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????20
.sequential_34/gru_16/while/gru_cell_20/Const_1?
8sequential_34/gru_16/while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2:
8sequential_34/gru_16/while/gru_cell_20/split_1/split_dim?
.sequential_34/gru_16/while/gru_cell_20/split_1SplitV9sequential_34/gru_16/while/gru_cell_20/BiasAdd_1:output:07sequential_34/gru_16/while/gru_cell_20/Const_1:output:0Asequential_34/gru_16/while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split20
.sequential_34/gru_16/while/gru_cell_20/split_1?
*sequential_34/gru_16/while/gru_cell_20/addAddV25sequential_34/gru_16/while/gru_cell_20/split:output:07sequential_34/gru_16/while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2,
*sequential_34/gru_16/while/gru_cell_20/add?
.sequential_34/gru_16/while/gru_cell_20/SigmoidSigmoid.sequential_34/gru_16/while/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d20
.sequential_34/gru_16/while/gru_cell_20/Sigmoid?
,sequential_34/gru_16/while/gru_cell_20/add_1AddV25sequential_34/gru_16/while/gru_cell_20/split:output:17sequential_34/gru_16/while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2.
,sequential_34/gru_16/while/gru_cell_20/add_1?
0sequential_34/gru_16/while/gru_cell_20/Sigmoid_1Sigmoid0sequential_34/gru_16/while/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d22
0sequential_34/gru_16/while/gru_cell_20/Sigmoid_1?
*sequential_34/gru_16/while/gru_cell_20/mulMul4sequential_34/gru_16/while/gru_cell_20/Sigmoid_1:y:07sequential_34/gru_16/while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2,
*sequential_34/gru_16/while/gru_cell_20/mul?
,sequential_34/gru_16/while/gru_cell_20/add_2AddV25sequential_34/gru_16/while/gru_cell_20/split:output:2.sequential_34/gru_16/while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2.
,sequential_34/gru_16/while/gru_cell_20/add_2?
+sequential_34/gru_16/while/gru_cell_20/TanhTanh0sequential_34/gru_16/while/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2-
+sequential_34/gru_16/while/gru_cell_20/Tanh?
,sequential_34/gru_16/while/gru_cell_20/mul_1Mul2sequential_34/gru_16/while/gru_cell_20/Sigmoid:y:0(sequential_34_gru_16_while_placeholder_2*
T0*'
_output_shapes
:?????????d2.
,sequential_34/gru_16/while/gru_cell_20/mul_1?
,sequential_34/gru_16/while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,sequential_34/gru_16/while/gru_cell_20/sub/x?
*sequential_34/gru_16/while/gru_cell_20/subSub5sequential_34/gru_16/while/gru_cell_20/sub/x:output:02sequential_34/gru_16/while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2,
*sequential_34/gru_16/while/gru_cell_20/sub?
,sequential_34/gru_16/while/gru_cell_20/mul_2Mul.sequential_34/gru_16/while/gru_cell_20/sub:z:0/sequential_34/gru_16/while/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2.
,sequential_34/gru_16/while/gru_cell_20/mul_2?
,sequential_34/gru_16/while/gru_cell_20/add_3AddV20sequential_34/gru_16/while/gru_cell_20/mul_1:z:00sequential_34/gru_16/while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2.
,sequential_34/gru_16/while/gru_cell_20/add_3?
?sequential_34/gru_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_34_gru_16_while_placeholder_1&sequential_34_gru_16_while_placeholder0sequential_34/gru_16/while/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype02A
?sequential_34/gru_16/while/TensorArrayV2Write/TensorListSetItem?
 sequential_34/gru_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_34/gru_16/while/add/y?
sequential_34/gru_16/while/addAddV2&sequential_34_gru_16_while_placeholder)sequential_34/gru_16/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_34/gru_16/while/add?
"sequential_34/gru_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_34/gru_16/while/add_1/y?
 sequential_34/gru_16/while/add_1AddV2Bsequential_34_gru_16_while_sequential_34_gru_16_while_loop_counter+sequential_34/gru_16/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_34/gru_16/while/add_1?
#sequential_34/gru_16/while/IdentityIdentity$sequential_34/gru_16/while/add_1:z:0*
T0*
_output_shapes
: 2%
#sequential_34/gru_16/while/Identity?
%sequential_34/gru_16/while/Identity_1IdentityHsequential_34_gru_16_while_sequential_34_gru_16_while_maximum_iterations*
T0*
_output_shapes
: 2'
%sequential_34/gru_16/while/Identity_1?
%sequential_34/gru_16/while/Identity_2Identity"sequential_34/gru_16/while/add:z:0*
T0*
_output_shapes
: 2'
%sequential_34/gru_16/while/Identity_2?
%sequential_34/gru_16/while/Identity_3IdentityOsequential_34/gru_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2'
%sequential_34/gru_16/while/Identity_3?
%sequential_34/gru_16/while/Identity_4Identity0sequential_34/gru_16/while/gru_cell_20/add_3:z:0*
T0*'
_output_shapes
:?????????d2'
%sequential_34/gru_16/while/Identity_4"?
Gsequential_34_gru_16_while_gru_cell_20_matmul_1_readvariableop_resourceIsequential_34_gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0"?
Esequential_34_gru_16_while_gru_cell_20_matmul_readvariableop_resourceGsequential_34_gru_16_while_gru_cell_20_matmul_readvariableop_resource_0"?
>sequential_34_gru_16_while_gru_cell_20_readvariableop_resource@sequential_34_gru_16_while_gru_cell_20_readvariableop_resource_0"S
#sequential_34_gru_16_while_identity,sequential_34/gru_16/while/Identity:output:0"W
%sequential_34_gru_16_while_identity_1.sequential_34/gru_16/while/Identity_1:output:0"W
%sequential_34_gru_16_while_identity_2.sequential_34/gru_16/while/Identity_2:output:0"W
%sequential_34_gru_16_while_identity_3.sequential_34/gru_16/while/Identity_3:output:0"W
%sequential_34_gru_16_while_identity_4.sequential_34/gru_16/while/Identity_4:output:0"?
?sequential_34_gru_16_while_sequential_34_gru_16_strided_slice_1Asequential_34_gru_16_while_sequential_34_gru_16_strided_slice_1_0"?
{sequential_34_gru_16_while_tensorarrayv2read_tensorlistgetitem_sequential_34_gru_16_tensorarrayunstack_tensorlistfromtensor}sequential_34_gru_16_while_tensorarrayv2read_tensorlistgetitem_sequential_34_gru_16_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_515666
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_515666___redundant_placeholder04
0while_while_cond_515666___redundant_placeholder14
0while_while_cond_515666___redundant_placeholder24
0while_while_cond_515666___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?@
?
while_body_515327
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_20_readvariableop_resource_06
2while_gru_cell_20_matmul_readvariableop_resource_08
4while_gru_cell_20_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_20_readvariableop_resource4
0while_gru_cell_20_matmul_readvariableop_resource6
2while_gru_cell_20_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_20/ReadVariableOpReadVariableOp+while_gru_cell_20_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_20/ReadVariableOp?
while/gru_cell_20/unstackUnpack(while/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_20/unstack?
'while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_20/MatMul/ReadVariableOp?
while/gru_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/MatMul?
while/gru_cell_20/BiasAddBiasAdd"while/gru_cell_20/MatMul:product:0"while/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/BiasAddt
while/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_20/Const?
!while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_20/split/split_dim?
while/gru_cell_20/splitSplit*while/gru_cell_20/split/split_dim:output:0"while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_20/split?
)while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02+
)while/gru_cell_20/MatMul_1/ReadVariableOp?
while/gru_cell_20/MatMul_1MatMulwhile_placeholder_21while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/MatMul_1?
while/gru_cell_20/BiasAdd_1BiasAdd$while/gru_cell_20/MatMul_1:product:0"while/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/BiasAdd_1?
while/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
while/gru_cell_20/Const_1?
#while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_20/split_1/split_dim?
while/gru_cell_20/split_1SplitV$while/gru_cell_20/BiasAdd_1:output:0"while/gru_cell_20/Const_1:output:0,while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_20/split_1?
while/gru_cell_20/addAddV2 while/gru_cell_20/split:output:0"while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add?
while/gru_cell_20/SigmoidSigmoidwhile/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Sigmoid?
while/gru_cell_20/add_1AddV2 while/gru_cell_20/split:output:1"while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_1?
while/gru_cell_20/Sigmoid_1Sigmoidwhile/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Sigmoid_1?
while/gru_cell_20/mulMulwhile/gru_cell_20/Sigmoid_1:y:0"while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul?
while/gru_cell_20/add_2AddV2 while/gru_cell_20/split:output:2while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_2?
while/gru_cell_20/TanhTanhwhile/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Tanh?
while/gru_cell_20/mul_1Mulwhile/gru_cell_20/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul_1w
while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_20/sub/x?
while/gru_cell_20/subSub while/gru_cell_20/sub/x:output:0while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/sub?
while/gru_cell_20/mul_2Mulwhile/gru_cell_20/sub:z:0while/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul_2?
while/gru_cell_20/add_3AddV2while/gru_cell_20/mul_1:z:0while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_20/add_3:z:0*
T0*'
_output_shapes
:?????????d2
while/Identity_4"j
2while_gru_cell_20_matmul_1_readvariableop_resource4while_gru_cell_20_matmul_1_readvariableop_resource_0"f
0while_gru_cell_20_matmul_readvariableop_resource2while_gru_cell_20_matmul_readvariableop_resource_0"X
)while_gru_cell_20_readvariableop_resource+while_gru_cell_20_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
&sequential_34_gru_16_while_cond_513339F
Bsequential_34_gru_16_while_sequential_34_gru_16_while_loop_counterL
Hsequential_34_gru_16_while_sequential_34_gru_16_while_maximum_iterations*
&sequential_34_gru_16_while_placeholder,
(sequential_34_gru_16_while_placeholder_1,
(sequential_34_gru_16_while_placeholder_2H
Dsequential_34_gru_16_while_less_sequential_34_gru_16_strided_slice_1^
Zsequential_34_gru_16_while_sequential_34_gru_16_while_cond_513339___redundant_placeholder0^
Zsequential_34_gru_16_while_sequential_34_gru_16_while_cond_513339___redundant_placeholder1^
Zsequential_34_gru_16_while_sequential_34_gru_16_while_cond_513339___redundant_placeholder2^
Zsequential_34_gru_16_while_sequential_34_gru_16_while_cond_513339___redundant_placeholder3'
#sequential_34_gru_16_while_identity
?
sequential_34/gru_16/while/LessLess&sequential_34_gru_16_while_placeholderDsequential_34_gru_16_while_less_sequential_34_gru_16_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_34/gru_16/while/Less?
#sequential_34/gru_16/while/IdentityIdentity#sequential_34/gru_16/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_34/gru_16/while/Identity"S
#sequential_34_gru_16_while_identity,sequential_34/gru_16/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
??
?
!__inference__wrapped_model_513438
gru_16_input<
8sequential_34_gru_16_gru_cell_20_readvariableop_resourceC
?sequential_34_gru_16_gru_cell_20_matmul_readvariableop_resourceE
Asequential_34_gru_16_gru_cell_20_matmul_1_readvariableop_resource9
5sequential_34_dense_34_matmul_readvariableop_resource:
6sequential_34_dense_34_biasadd_readvariableop_resource
identity??sequential_34/gru_16/whilet
sequential_34/gru_16/ShapeShapegru_16_input*
T0*
_output_shapes
:2
sequential_34/gru_16/Shape?
(sequential_34/gru_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_34/gru_16/strided_slice/stack?
*sequential_34/gru_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_34/gru_16/strided_slice/stack_1?
*sequential_34/gru_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_34/gru_16/strided_slice/stack_2?
"sequential_34/gru_16/strided_sliceStridedSlice#sequential_34/gru_16/Shape:output:01sequential_34/gru_16/strided_slice/stack:output:03sequential_34/gru_16/strided_slice/stack_1:output:03sequential_34/gru_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_34/gru_16/strided_slice?
 sequential_34/gru_16/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2"
 sequential_34/gru_16/zeros/mul/y?
sequential_34/gru_16/zeros/mulMul+sequential_34/gru_16/strided_slice:output:0)sequential_34/gru_16/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_34/gru_16/zeros/mul?
!sequential_34/gru_16/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_34/gru_16/zeros/Less/y?
sequential_34/gru_16/zeros/LessLess"sequential_34/gru_16/zeros/mul:z:0*sequential_34/gru_16/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_34/gru_16/zeros/Less?
#sequential_34/gru_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2%
#sequential_34/gru_16/zeros/packed/1?
!sequential_34/gru_16/zeros/packedPack+sequential_34/gru_16/strided_slice:output:0,sequential_34/gru_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_34/gru_16/zeros/packed?
 sequential_34/gru_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_34/gru_16/zeros/Const?
sequential_34/gru_16/zerosFill*sequential_34/gru_16/zeros/packed:output:0)sequential_34/gru_16/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
sequential_34/gru_16/zeros?
#sequential_34/gru_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_34/gru_16/transpose/perm?
sequential_34/gru_16/transpose	Transposegru_16_input,sequential_34/gru_16/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2 
sequential_34/gru_16/transpose?
sequential_34/gru_16/Shape_1Shape"sequential_34/gru_16/transpose:y:0*
T0*
_output_shapes
:2
sequential_34/gru_16/Shape_1?
*sequential_34/gru_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_34/gru_16/strided_slice_1/stack?
,sequential_34/gru_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_34/gru_16/strided_slice_1/stack_1?
,sequential_34/gru_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_34/gru_16/strided_slice_1/stack_2?
$sequential_34/gru_16/strided_slice_1StridedSlice%sequential_34/gru_16/Shape_1:output:03sequential_34/gru_16/strided_slice_1/stack:output:05sequential_34/gru_16/strided_slice_1/stack_1:output:05sequential_34/gru_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_34/gru_16/strided_slice_1?
0sequential_34/gru_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_34/gru_16/TensorArrayV2/element_shape?
"sequential_34/gru_16/TensorArrayV2TensorListReserve9sequential_34/gru_16/TensorArrayV2/element_shape:output:0-sequential_34/gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_34/gru_16/TensorArrayV2?
Jsequential_34/gru_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jsequential_34/gru_16/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_34/gru_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_34/gru_16/transpose:y:0Ssequential_34/gru_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_34/gru_16/TensorArrayUnstack/TensorListFromTensor?
*sequential_34/gru_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_34/gru_16/strided_slice_2/stack?
,sequential_34/gru_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_34/gru_16/strided_slice_2/stack_1?
,sequential_34/gru_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_34/gru_16/strided_slice_2/stack_2?
$sequential_34/gru_16/strided_slice_2StridedSlice"sequential_34/gru_16/transpose:y:03sequential_34/gru_16/strided_slice_2/stack:output:05sequential_34/gru_16/strided_slice_2/stack_1:output:05sequential_34/gru_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2&
$sequential_34/gru_16/strided_slice_2?
/sequential_34/gru_16/gru_cell_20/ReadVariableOpReadVariableOp8sequential_34_gru_16_gru_cell_20_readvariableop_resource*
_output_shapes
:	?*
dtype021
/sequential_34/gru_16/gru_cell_20/ReadVariableOp?
(sequential_34/gru_16/gru_cell_20/unstackUnpack7sequential_34/gru_16/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2*
(sequential_34/gru_16/gru_cell_20/unstack?
6sequential_34/gru_16/gru_cell_20/MatMul/ReadVariableOpReadVariableOp?sequential_34_gru_16_gru_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype028
6sequential_34/gru_16/gru_cell_20/MatMul/ReadVariableOp?
'sequential_34/gru_16/gru_cell_20/MatMulMatMul-sequential_34/gru_16/strided_slice_2:output:0>sequential_34/gru_16/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_34/gru_16/gru_cell_20/MatMul?
(sequential_34/gru_16/gru_cell_20/BiasAddBiasAdd1sequential_34/gru_16/gru_cell_20/MatMul:product:01sequential_34/gru_16/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2*
(sequential_34/gru_16/gru_cell_20/BiasAdd?
&sequential_34/gru_16/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_34/gru_16/gru_cell_20/Const?
0sequential_34/gru_16/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_34/gru_16/gru_cell_20/split/split_dim?
&sequential_34/gru_16/gru_cell_20/splitSplit9sequential_34/gru_16/gru_cell_20/split/split_dim:output:01sequential_34/gru_16/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2(
&sequential_34/gru_16/gru_cell_20/split?
8sequential_34/gru_16/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOpAsequential_34_gru_16_gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02:
8sequential_34/gru_16/gru_cell_20/MatMul_1/ReadVariableOp?
)sequential_34/gru_16/gru_cell_20/MatMul_1MatMul#sequential_34/gru_16/zeros:output:0@sequential_34/gru_16/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_34/gru_16/gru_cell_20/MatMul_1?
*sequential_34/gru_16/gru_cell_20/BiasAdd_1BiasAdd3sequential_34/gru_16/gru_cell_20/MatMul_1:product:01sequential_34/gru_16/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2,
*sequential_34/gru_16/gru_cell_20/BiasAdd_1?
(sequential_34/gru_16/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2*
(sequential_34/gru_16/gru_cell_20/Const_1?
2sequential_34/gru_16/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2sequential_34/gru_16/gru_cell_20/split_1/split_dim?
(sequential_34/gru_16/gru_cell_20/split_1SplitV3sequential_34/gru_16/gru_cell_20/BiasAdd_1:output:01sequential_34/gru_16/gru_cell_20/Const_1:output:0;sequential_34/gru_16/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2*
(sequential_34/gru_16/gru_cell_20/split_1?
$sequential_34/gru_16/gru_cell_20/addAddV2/sequential_34/gru_16/gru_cell_20/split:output:01sequential_34/gru_16/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2&
$sequential_34/gru_16/gru_cell_20/add?
(sequential_34/gru_16/gru_cell_20/SigmoidSigmoid(sequential_34/gru_16/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2*
(sequential_34/gru_16/gru_cell_20/Sigmoid?
&sequential_34/gru_16/gru_cell_20/add_1AddV2/sequential_34/gru_16/gru_cell_20/split:output:11sequential_34/gru_16/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2(
&sequential_34/gru_16/gru_cell_20/add_1?
*sequential_34/gru_16/gru_cell_20/Sigmoid_1Sigmoid*sequential_34/gru_16/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2,
*sequential_34/gru_16/gru_cell_20/Sigmoid_1?
$sequential_34/gru_16/gru_cell_20/mulMul.sequential_34/gru_16/gru_cell_20/Sigmoid_1:y:01sequential_34/gru_16/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2&
$sequential_34/gru_16/gru_cell_20/mul?
&sequential_34/gru_16/gru_cell_20/add_2AddV2/sequential_34/gru_16/gru_cell_20/split:output:2(sequential_34/gru_16/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2(
&sequential_34/gru_16/gru_cell_20/add_2?
%sequential_34/gru_16/gru_cell_20/TanhTanh*sequential_34/gru_16/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2'
%sequential_34/gru_16/gru_cell_20/Tanh?
&sequential_34/gru_16/gru_cell_20/mul_1Mul,sequential_34/gru_16/gru_cell_20/Sigmoid:y:0#sequential_34/gru_16/zeros:output:0*
T0*'
_output_shapes
:?????????d2(
&sequential_34/gru_16/gru_cell_20/mul_1?
&sequential_34/gru_16/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&sequential_34/gru_16/gru_cell_20/sub/x?
$sequential_34/gru_16/gru_cell_20/subSub/sequential_34/gru_16/gru_cell_20/sub/x:output:0,sequential_34/gru_16/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2&
$sequential_34/gru_16/gru_cell_20/sub?
&sequential_34/gru_16/gru_cell_20/mul_2Mul(sequential_34/gru_16/gru_cell_20/sub:z:0)sequential_34/gru_16/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2(
&sequential_34/gru_16/gru_cell_20/mul_2?
&sequential_34/gru_16/gru_cell_20/add_3AddV2*sequential_34/gru_16/gru_cell_20/mul_1:z:0*sequential_34/gru_16/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2(
&sequential_34/gru_16/gru_cell_20/add_3?
2sequential_34/gru_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   24
2sequential_34/gru_16/TensorArrayV2_1/element_shape?
$sequential_34/gru_16/TensorArrayV2_1TensorListReserve;sequential_34/gru_16/TensorArrayV2_1/element_shape:output:0-sequential_34/gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_34/gru_16/TensorArrayV2_1x
sequential_34/gru_16/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_34/gru_16/time?
-sequential_34/gru_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_34/gru_16/while/maximum_iterations?
'sequential_34/gru_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_34/gru_16/while/loop_counter?
sequential_34/gru_16/whileWhile0sequential_34/gru_16/while/loop_counter:output:06sequential_34/gru_16/while/maximum_iterations:output:0"sequential_34/gru_16/time:output:0-sequential_34/gru_16/TensorArrayV2_1:handle:0#sequential_34/gru_16/zeros:output:0-sequential_34/gru_16/strided_slice_1:output:0Lsequential_34/gru_16/TensorArrayUnstack/TensorListFromTensor:output_handle:08sequential_34_gru_16_gru_cell_20_readvariableop_resource?sequential_34_gru_16_gru_cell_20_matmul_readvariableop_resourceAsequential_34_gru_16_gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*2
body*R(
&sequential_34_gru_16_while_body_513340*2
cond*R(
&sequential_34_gru_16_while_cond_513339*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
sequential_34/gru_16/while?
Esequential_34/gru_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2G
Esequential_34/gru_16/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_34/gru_16/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_34/gru_16/while:output:3Nsequential_34/gru_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype029
7sequential_34/gru_16/TensorArrayV2Stack/TensorListStack?
*sequential_34/gru_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_34/gru_16/strided_slice_3/stack?
,sequential_34/gru_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_34/gru_16/strided_slice_3/stack_1?
,sequential_34/gru_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_34/gru_16/strided_slice_3/stack_2?
$sequential_34/gru_16/strided_slice_3StridedSlice@sequential_34/gru_16/TensorArrayV2Stack/TensorListStack:tensor:03sequential_34/gru_16/strided_slice_3/stack:output:05sequential_34/gru_16/strided_slice_3/stack_1:output:05sequential_34/gru_16/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2&
$sequential_34/gru_16/strided_slice_3?
%sequential_34/gru_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_34/gru_16/transpose_1/perm?
 sequential_34/gru_16/transpose_1	Transpose@sequential_34/gru_16/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_34/gru_16/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2"
 sequential_34/gru_16/transpose_1?
sequential_34/gru_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_34/gru_16/runtime?
!sequential_34/dropout_17/IdentityIdentity-sequential_34/gru_16/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d2#
!sequential_34/dropout_17/Identity?
,sequential_34/dense_34/MatMul/ReadVariableOpReadVariableOp5sequential_34_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_34/dense_34/MatMul/ReadVariableOp?
sequential_34/dense_34/MatMulMatMul*sequential_34/dropout_17/Identity:output:04sequential_34/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_34/dense_34/MatMul?
-sequential_34/dense_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_34_dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_34/dense_34/BiasAdd/ReadVariableOp?
sequential_34/dense_34/BiasAddBiasAdd'sequential_34/dense_34/MatMul:product:05sequential_34/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_34/dense_34/BiasAdd?
sequential_34/dense_34/ReluRelu'sequential_34/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_34/dense_34/Relu?
IdentityIdentity)sequential_34/dense_34/Relu:activations:0^sequential_34/gru_16/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::28
sequential_34/gru_16/whilesequential_34/gru_16/while:Y U
+
_output_shapes
:?????????x
&
_user_specified_namegru_16_input
?
?
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_516065

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1?y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????d2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????d2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????d2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????d2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????d2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????d2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????d2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:?????????d2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????d2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????d2
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:?????????d2

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:?????????d2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????d::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0
?@
?
while_body_515667
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_20_readvariableop_resource_06
2while_gru_cell_20_matmul_readvariableop_resource_08
4while_gru_cell_20_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_20_readvariableop_resource4
0while_gru_cell_20_matmul_readvariableop_resource6
2while_gru_cell_20_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_20/ReadVariableOpReadVariableOp+while_gru_cell_20_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_20/ReadVariableOp?
while/gru_cell_20/unstackUnpack(while/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_20/unstack?
'while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_20/MatMul/ReadVariableOp?
while/gru_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/MatMul?
while/gru_cell_20/BiasAddBiasAdd"while/gru_cell_20/MatMul:product:0"while/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/BiasAddt
while/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_20/Const?
!while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_20/split/split_dim?
while/gru_cell_20/splitSplit*while/gru_cell_20/split/split_dim:output:0"while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_20/split?
)while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02+
)while/gru_cell_20/MatMul_1/ReadVariableOp?
while/gru_cell_20/MatMul_1MatMulwhile_placeholder_21while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/MatMul_1?
while/gru_cell_20/BiasAdd_1BiasAdd$while/gru_cell_20/MatMul_1:product:0"while/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/BiasAdd_1?
while/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
while/gru_cell_20/Const_1?
#while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_20/split_1/split_dim?
while/gru_cell_20/split_1SplitV$while/gru_cell_20/BiasAdd_1:output:0"while/gru_cell_20/Const_1:output:0,while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_20/split_1?
while/gru_cell_20/addAddV2 while/gru_cell_20/split:output:0"while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add?
while/gru_cell_20/SigmoidSigmoidwhile/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Sigmoid?
while/gru_cell_20/add_1AddV2 while/gru_cell_20/split:output:1"while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_1?
while/gru_cell_20/Sigmoid_1Sigmoidwhile/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Sigmoid_1?
while/gru_cell_20/mulMulwhile/gru_cell_20/Sigmoid_1:y:0"while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul?
while/gru_cell_20/add_2AddV2 while/gru_cell_20/split:output:2while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_2?
while/gru_cell_20/TanhTanhwhile/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Tanh?
while/gru_cell_20/mul_1Mulwhile/gru_cell_20/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul_1w
while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_20/sub/x?
while/gru_cell_20/subSub while/gru_cell_20/sub/x:output:0while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/sub?
while/gru_cell_20/mul_2Mulwhile/gru_cell_20/sub:z:0while/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul_2?
while/gru_cell_20/add_3AddV2while/gru_cell_20/mul_1:z:0while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_20/add_3:z:0*
T0*'
_output_shapes
:?????????d2
while/Identity_4"j
2while_gru_cell_20_matmul_1_readvariableop_resource4while_gru_cell_20_matmul_1_readvariableop_resource_0"f
0while_gru_cell_20_matmul_readvariableop_resource2while_gru_cell_20_matmul_readvariableop_resource_0"X
)while_gru_cell_20_readvariableop_resource+while_gru_cell_20_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
e
F__inference_dropout_17_layer_call_and_return_conditional_losses_514363

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?J
?
gru_16_while_body_515130*
&gru_16_while_gru_16_while_loop_counter0
,gru_16_while_gru_16_while_maximum_iterations
gru_16_while_placeholder
gru_16_while_placeholder_1
gru_16_while_placeholder_2)
%gru_16_while_gru_16_strided_slice_1_0e
agru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_06
2gru_16_while_gru_cell_20_readvariableop_resource_0=
9gru_16_while_gru_cell_20_matmul_readvariableop_resource_0?
;gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0
gru_16_while_identity
gru_16_while_identity_1
gru_16_while_identity_2
gru_16_while_identity_3
gru_16_while_identity_4'
#gru_16_while_gru_16_strided_slice_1c
_gru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor4
0gru_16_while_gru_cell_20_readvariableop_resource;
7gru_16_while_gru_cell_20_matmul_readvariableop_resource=
9gru_16_while_gru_cell_20_matmul_1_readvariableop_resource??
>gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0gru_16_while_placeholderGgru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0gru_16/while/TensorArrayV2Read/TensorListGetItem?
'gru_16/while/gru_cell_20/ReadVariableOpReadVariableOp2gru_16_while_gru_cell_20_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_16/while/gru_cell_20/ReadVariableOp?
 gru_16/while/gru_cell_20/unstackUnpack/gru_16/while/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_16/while/gru_cell_20/unstack?
.gru_16/while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp9gru_16_while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.gru_16/while/gru_cell_20/MatMul/ReadVariableOp?
gru_16/while/gru_cell_20/MatMulMatMul7gru_16/while/TensorArrayV2Read/TensorListGetItem:item:06gru_16/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_16/while/gru_cell_20/MatMul?
 gru_16/while/gru_cell_20/BiasAddBiasAdd)gru_16/while/gru_cell_20/MatMul:product:0)gru_16/while/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_16/while/gru_cell_20/BiasAdd?
gru_16/while/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
gru_16/while/gru_cell_20/Const?
(gru_16/while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_16/while/gru_cell_20/split/split_dim?
gru_16/while/gru_cell_20/splitSplit1gru_16/while/gru_cell_20/split/split_dim:output:0)gru_16/while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2 
gru_16/while/gru_cell_20/split?
0gru_16/while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp;gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype022
0gru_16/while/gru_cell_20/MatMul_1/ReadVariableOp?
!gru_16/while/gru_cell_20/MatMul_1MatMulgru_16_while_placeholder_28gru_16/while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_16/while/gru_cell_20/MatMul_1?
"gru_16/while/gru_cell_20/BiasAdd_1BiasAdd+gru_16/while/gru_cell_20/MatMul_1:product:0)gru_16/while/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_16/while/gru_cell_20/BiasAdd_1?
 gru_16/while/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2"
 gru_16/while/gru_cell_20/Const_1?
*gru_16/while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_16/while/gru_cell_20/split_1/split_dim?
 gru_16/while/gru_cell_20/split_1SplitV+gru_16/while/gru_cell_20/BiasAdd_1:output:0)gru_16/while/gru_cell_20/Const_1:output:03gru_16/while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2"
 gru_16/while/gru_cell_20/split_1?
gru_16/while/gru_cell_20/addAddV2'gru_16/while/gru_cell_20/split:output:0)gru_16/while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/add?
 gru_16/while/gru_cell_20/SigmoidSigmoid gru_16/while/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2"
 gru_16/while/gru_cell_20/Sigmoid?
gru_16/while/gru_cell_20/add_1AddV2'gru_16/while/gru_cell_20/split:output:1)gru_16/while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/add_1?
"gru_16/while/gru_cell_20/Sigmoid_1Sigmoid"gru_16/while/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2$
"gru_16/while/gru_cell_20/Sigmoid_1?
gru_16/while/gru_cell_20/mulMul&gru_16/while/gru_cell_20/Sigmoid_1:y:0)gru_16/while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/mul?
gru_16/while/gru_cell_20/add_2AddV2'gru_16/while/gru_cell_20/split:output:2 gru_16/while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/add_2?
gru_16/while/gru_cell_20/TanhTanh"gru_16/while/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/Tanh?
gru_16/while/gru_cell_20/mul_1Mul$gru_16/while/gru_cell_20/Sigmoid:y:0gru_16_while_placeholder_2*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/mul_1?
gru_16/while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_16/while/gru_cell_20/sub/x?
gru_16/while/gru_cell_20/subSub'gru_16/while/gru_cell_20/sub/x:output:0$gru_16/while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/sub?
gru_16/while/gru_cell_20/mul_2Mul gru_16/while/gru_cell_20/sub:z:0!gru_16/while/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/mul_2?
gru_16/while/gru_cell_20/add_3AddV2"gru_16/while/gru_cell_20/mul_1:z:0"gru_16/while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/add_3?
1gru_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_16_while_placeholder_1gru_16_while_placeholder"gru_16/while/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_16/while/TensorArrayV2Write/TensorListSetItemj
gru_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_16/while/add/y?
gru_16/while/addAddV2gru_16_while_placeholdergru_16/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_16/while/addn
gru_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_16/while/add_1/y?
gru_16/while/add_1AddV2&gru_16_while_gru_16_while_loop_countergru_16/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_16/while/add_1s
gru_16/while/IdentityIdentitygru_16/while/add_1:z:0*
T0*
_output_shapes
: 2
gru_16/while/Identity?
gru_16/while/Identity_1Identity,gru_16_while_gru_16_while_maximum_iterations*
T0*
_output_shapes
: 2
gru_16/while/Identity_1u
gru_16/while/Identity_2Identitygru_16/while/add:z:0*
T0*
_output_shapes
: 2
gru_16/while/Identity_2?
gru_16/while/Identity_3IdentityAgru_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru_16/while/Identity_3?
gru_16/while/Identity_4Identity"gru_16/while/gru_cell_20/add_3:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/Identity_4"L
#gru_16_while_gru_16_strided_slice_1%gru_16_while_gru_16_strided_slice_1_0"x
9gru_16_while_gru_cell_20_matmul_1_readvariableop_resource;gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0"t
7gru_16_while_gru_cell_20_matmul_readvariableop_resource9gru_16_while_gru_cell_20_matmul_readvariableop_resource_0"f
0gru_16_while_gru_cell_20_readvariableop_resource2gru_16_while_gru_cell_20_readvariableop_resource_0"7
gru_16_while_identitygru_16/while/Identity:output:0";
gru_16_while_identity_1 gru_16/while/Identity_1:output:0";
gru_16_while_identity_2 gru_16/while/Identity_2:output:0";
gru_16_while_identity_3 gru_16/while/Identity_3:output:0";
gru_16_while_identity_4 gru_16/while/Identity_4:output:0"?
_gru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensoragru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?	
?
gru_16_while_cond_514584*
&gru_16_while_gru_16_while_loop_counter0
,gru_16_while_gru_16_while_maximum_iterations
gru_16_while_placeholder
gru_16_while_placeholder_1
gru_16_while_placeholder_2,
(gru_16_while_less_gru_16_strided_slice_1B
>gru_16_while_gru_16_while_cond_514584___redundant_placeholder0B
>gru_16_while_gru_16_while_cond_514584___redundant_placeholder1B
>gru_16_while_gru_16_while_cond_514584___redundant_placeholder2B
>gru_16_while_gru_16_while_cond_514584___redundant_placeholder3
gru_16_while_identity
?
gru_16/while/LessLessgru_16_while_placeholder(gru_16_while_less_gru_16_strided_slice_1*
T0*
_output_shapes
: 2
gru_16/while/Lessr
gru_16/while/IdentityIdentitygru_16/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_16/while/Identity"7
gru_16_while_identitygru_16/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?J
?
gru_16_while_body_514585*
&gru_16_while_gru_16_while_loop_counter0
,gru_16_while_gru_16_while_maximum_iterations
gru_16_while_placeholder
gru_16_while_placeholder_1
gru_16_while_placeholder_2)
%gru_16_while_gru_16_strided_slice_1_0e
agru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_06
2gru_16_while_gru_cell_20_readvariableop_resource_0=
9gru_16_while_gru_cell_20_matmul_readvariableop_resource_0?
;gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0
gru_16_while_identity
gru_16_while_identity_1
gru_16_while_identity_2
gru_16_while_identity_3
gru_16_while_identity_4'
#gru_16_while_gru_16_strided_slice_1c
_gru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor4
0gru_16_while_gru_cell_20_readvariableop_resource;
7gru_16_while_gru_cell_20_matmul_readvariableop_resource=
9gru_16_while_gru_cell_20_matmul_1_readvariableop_resource??
>gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0gru_16_while_placeholderGgru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0gru_16/while/TensorArrayV2Read/TensorListGetItem?
'gru_16/while/gru_cell_20/ReadVariableOpReadVariableOp2gru_16_while_gru_cell_20_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_16/while/gru_cell_20/ReadVariableOp?
 gru_16/while/gru_cell_20/unstackUnpack/gru_16/while/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_16/while/gru_cell_20/unstack?
.gru_16/while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp9gru_16_while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.gru_16/while/gru_cell_20/MatMul/ReadVariableOp?
gru_16/while/gru_cell_20/MatMulMatMul7gru_16/while/TensorArrayV2Read/TensorListGetItem:item:06gru_16/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_16/while/gru_cell_20/MatMul?
 gru_16/while/gru_cell_20/BiasAddBiasAdd)gru_16/while/gru_cell_20/MatMul:product:0)gru_16/while/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_16/while/gru_cell_20/BiasAdd?
gru_16/while/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
gru_16/while/gru_cell_20/Const?
(gru_16/while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_16/while/gru_cell_20/split/split_dim?
gru_16/while/gru_cell_20/splitSplit1gru_16/while/gru_cell_20/split/split_dim:output:0)gru_16/while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2 
gru_16/while/gru_cell_20/split?
0gru_16/while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp;gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype022
0gru_16/while/gru_cell_20/MatMul_1/ReadVariableOp?
!gru_16/while/gru_cell_20/MatMul_1MatMulgru_16_while_placeholder_28gru_16/while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_16/while/gru_cell_20/MatMul_1?
"gru_16/while/gru_cell_20/BiasAdd_1BiasAdd+gru_16/while/gru_cell_20/MatMul_1:product:0)gru_16/while/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_16/while/gru_cell_20/BiasAdd_1?
 gru_16/while/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2"
 gru_16/while/gru_cell_20/Const_1?
*gru_16/while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_16/while/gru_cell_20/split_1/split_dim?
 gru_16/while/gru_cell_20/split_1SplitV+gru_16/while/gru_cell_20/BiasAdd_1:output:0)gru_16/while/gru_cell_20/Const_1:output:03gru_16/while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2"
 gru_16/while/gru_cell_20/split_1?
gru_16/while/gru_cell_20/addAddV2'gru_16/while/gru_cell_20/split:output:0)gru_16/while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/add?
 gru_16/while/gru_cell_20/SigmoidSigmoid gru_16/while/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2"
 gru_16/while/gru_cell_20/Sigmoid?
gru_16/while/gru_cell_20/add_1AddV2'gru_16/while/gru_cell_20/split:output:1)gru_16/while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/add_1?
"gru_16/while/gru_cell_20/Sigmoid_1Sigmoid"gru_16/while/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2$
"gru_16/while/gru_cell_20/Sigmoid_1?
gru_16/while/gru_cell_20/mulMul&gru_16/while/gru_cell_20/Sigmoid_1:y:0)gru_16/while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/mul?
gru_16/while/gru_cell_20/add_2AddV2'gru_16/while/gru_cell_20/split:output:2 gru_16/while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/add_2?
gru_16/while/gru_cell_20/TanhTanh"gru_16/while/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/Tanh?
gru_16/while/gru_cell_20/mul_1Mul$gru_16/while/gru_cell_20/Sigmoid:y:0gru_16_while_placeholder_2*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/mul_1?
gru_16/while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_16/while/gru_cell_20/sub/x?
gru_16/while/gru_cell_20/subSub'gru_16/while/gru_cell_20/sub/x:output:0$gru_16/while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/sub?
gru_16/while/gru_cell_20/mul_2Mul gru_16/while/gru_cell_20/sub:z:0!gru_16/while/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/mul_2?
gru_16/while/gru_cell_20/add_3AddV2"gru_16/while/gru_cell_20/mul_1:z:0"gru_16/while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/add_3?
1gru_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_16_while_placeholder_1gru_16_while_placeholder"gru_16/while/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_16/while/TensorArrayV2Write/TensorListSetItemj
gru_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_16/while/add/y?
gru_16/while/addAddV2gru_16_while_placeholdergru_16/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_16/while/addn
gru_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_16/while/add_1/y?
gru_16/while/add_1AddV2&gru_16_while_gru_16_while_loop_countergru_16/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_16/while/add_1s
gru_16/while/IdentityIdentitygru_16/while/add_1:z:0*
T0*
_output_shapes
: 2
gru_16/while/Identity?
gru_16/while/Identity_1Identity,gru_16_while_gru_16_while_maximum_iterations*
T0*
_output_shapes
: 2
gru_16/while/Identity_1u
gru_16/while/Identity_2Identitygru_16/while/add:z:0*
T0*
_output_shapes
: 2
gru_16/while/Identity_2?
gru_16/while/Identity_3IdentityAgru_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru_16/while/Identity_3?
gru_16/while/Identity_4Identity"gru_16/while/gru_cell_20/add_3:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/Identity_4"L
#gru_16_while_gru_16_strided_slice_1%gru_16_while_gru_16_strided_slice_1_0"x
9gru_16_while_gru_cell_20_matmul_1_readvariableop_resource;gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0"t
7gru_16_while_gru_cell_20_matmul_readvariableop_resource9gru_16_while_gru_cell_20_matmul_readvariableop_resource_0"f
0gru_16_while_gru_cell_20_readvariableop_resource2gru_16_while_gru_cell_20_readvariableop_resource_0"7
gru_16_while_identitygru_16/while/Identity:output:0";
gru_16_while_identity_1 gru_16/while/Identity_1:output:0";
gru_16_while_identity_2 gru_16/while/Identity_2:output:0";
gru_16_while_identity_3 gru_16/while/Identity_3:output:0";
gru_16_while_identity_4 gru_16/while/Identity_4:output:0"?
_gru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensoragru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?<
?
B__inference_gru_16_layer_call_and_return_conditional_losses_513991

inputs
gru_cell_20_513915
gru_cell_20_513917
gru_cell_20_513919
identity??#gru_cell_20/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#gru_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_20_513915gru_cell_20_513917gru_cell_20_513919*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_5135502%
#gru_cell_20/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_20_513915gru_cell_20_513917gru_cell_20_513919*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_513927*
condR
while_cond_513926*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^gru_cell_20/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2J
#gru_cell_20/StatefulPartitionedCall#gru_cell_20/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_515825
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_515825___redundant_placeholder04
0while_while_cond_515825___redundant_placeholder14
0while_while_cond_515825___redundant_placeholder24
0while_while_cond_515825___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?@
?
while_body_515826
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_20_readvariableop_resource_06
2while_gru_cell_20_matmul_readvariableop_resource_08
4while_gru_cell_20_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_20_readvariableop_resource4
0while_gru_cell_20_matmul_readvariableop_resource6
2while_gru_cell_20_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_20/ReadVariableOpReadVariableOp+while_gru_cell_20_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_20/ReadVariableOp?
while/gru_cell_20/unstackUnpack(while/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_20/unstack?
'while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_20/MatMul/ReadVariableOp?
while/gru_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/MatMul?
while/gru_cell_20/BiasAddBiasAdd"while/gru_cell_20/MatMul:product:0"while/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/BiasAddt
while/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_20/Const?
!while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_20/split/split_dim?
while/gru_cell_20/splitSplit*while/gru_cell_20/split/split_dim:output:0"while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_20/split?
)while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02+
)while/gru_cell_20/MatMul_1/ReadVariableOp?
while/gru_cell_20/MatMul_1MatMulwhile_placeholder_21while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/MatMul_1?
while/gru_cell_20/BiasAdd_1BiasAdd$while/gru_cell_20/MatMul_1:product:0"while/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/BiasAdd_1?
while/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
while/gru_cell_20/Const_1?
#while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_20/split_1/split_dim?
while/gru_cell_20/split_1SplitV$while/gru_cell_20/BiasAdd_1:output:0"while/gru_cell_20/Const_1:output:0,while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_20/split_1?
while/gru_cell_20/addAddV2 while/gru_cell_20/split:output:0"while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add?
while/gru_cell_20/SigmoidSigmoidwhile/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Sigmoid?
while/gru_cell_20/add_1AddV2 while/gru_cell_20/split:output:1"while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_1?
while/gru_cell_20/Sigmoid_1Sigmoidwhile/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Sigmoid_1?
while/gru_cell_20/mulMulwhile/gru_cell_20/Sigmoid_1:y:0"while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul?
while/gru_cell_20/add_2AddV2 while/gru_cell_20/split:output:2while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_2?
while/gru_cell_20/TanhTanhwhile/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Tanh?
while/gru_cell_20/mul_1Mulwhile/gru_cell_20/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul_1w
while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_20/sub/x?
while/gru_cell_20/subSub while/gru_cell_20/sub/x:output:0while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/sub?
while/gru_cell_20/mul_2Mulwhile/gru_cell_20/sub:z:0while/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul_2?
while/gru_cell_20/add_3AddV2while/gru_cell_20/mul_1:z:0while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_20/add_3:z:0*
T0*'
_output_shapes
:?????????d2
while/Identity_4"j
2while_gru_cell_20_matmul_1_readvariableop_resource4while_gru_cell_20_matmul_1_readvariableop_resource_0"f
0while_gru_cell_20_matmul_readvariableop_resource2while_gru_cell_20_matmul_readvariableop_resource_0"X
)while_gru_cell_20_readvariableop_resource+while_gru_cell_20_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?X
?
B__inference_gru_16_layer_call_and_return_conditional_losses_515417
inputs_0'
#gru_cell_20_readvariableop_resource.
*gru_cell_20_matmul_readvariableop_resource0
,gru_cell_20_matmul_1_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_20/ReadVariableOpReadVariableOp#gru_cell_20_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_20/ReadVariableOp?
gru_cell_20/unstackUnpack"gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_20/unstack?
!gru_cell_20/MatMul/ReadVariableOpReadVariableOp*gru_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_20/MatMul/ReadVariableOp?
gru_cell_20/MatMulMatMulstrided_slice_2:output:0)gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/MatMul?
gru_cell_20/BiasAddBiasAddgru_cell_20/MatMul:product:0gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/BiasAddh
gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_20/Const?
gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_20/split/split_dim?
gru_cell_20/splitSplit$gru_cell_20/split/split_dim:output:0gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_20/split?
#gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02%
#gru_cell_20/MatMul_1/ReadVariableOp?
gru_cell_20/MatMul_1MatMulzeros:output:0+gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/MatMul_1?
gru_cell_20/BiasAdd_1BiasAddgru_cell_20/MatMul_1:product:0gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_20/BiasAdd_1
gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_cell_20/Const_1?
gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_20/split_1/split_dim?
gru_cell_20/split_1SplitVgru_cell_20/BiasAdd_1:output:0gru_cell_20/Const_1:output:0&gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_20/split_1?
gru_cell_20/addAddV2gru_cell_20/split:output:0gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add|
gru_cell_20/SigmoidSigmoidgru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Sigmoid?
gru_cell_20/add_1AddV2gru_cell_20/split:output:1gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_1?
gru_cell_20/Sigmoid_1Sigmoidgru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Sigmoid_1?
gru_cell_20/mulMulgru_cell_20/Sigmoid_1:y:0gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul?
gru_cell_20/add_2AddV2gru_cell_20/split:output:2gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_2u
gru_cell_20/TanhTanhgru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Tanh?
gru_cell_20/mul_1Mulgru_cell_20/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul_1k
gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_20/sub/x?
gru_cell_20/subSubgru_cell_20/sub/x:output:0gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/sub?
gru_cell_20/mul_2Mulgru_cell_20/sub:z:0gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul_2?
gru_cell_20/add_3AddV2gru_cell_20/mul_1:z:0gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_20_readvariableop_resource*gru_cell_20_matmul_readvariableop_resource,gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_515327*
condR
while_cond_515326*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_513510

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1?y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????d2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????d2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????d2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????d2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????d2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????d2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????d2
Tanh\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:?????????d2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????d2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????d2
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:?????????d2

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:?????????d2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????d::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates
?v
?
I__inference_sequential_34_layer_call_and_return_conditional_losses_515061

inputs.
*gru_16_gru_cell_20_readvariableop_resource5
1gru_16_gru_cell_20_matmul_readvariableop_resource7
3gru_16_gru_cell_20_matmul_1_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource
identity??gru_16/whileR
gru_16/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_16/Shape?
gru_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_16/strided_slice/stack?
gru_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_16/strided_slice/stack_1?
gru_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_16/strided_slice/stack_2?
gru_16/strided_sliceStridedSlicegru_16/Shape:output:0#gru_16/strided_slice/stack:output:0%gru_16/strided_slice/stack_1:output:0%gru_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_16/strided_slicej
gru_16/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
gru_16/zeros/mul/y?
gru_16/zeros/mulMulgru_16/strided_slice:output:0gru_16/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_16/zeros/mulm
gru_16/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_16/zeros/Less/y?
gru_16/zeros/LessLessgru_16/zeros/mul:z:0gru_16/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_16/zeros/Lessp
gru_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
gru_16/zeros/packed/1?
gru_16/zeros/packedPackgru_16/strided_slice:output:0gru_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_16/zeros/packedm
gru_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_16/zeros/Const?
gru_16/zerosFillgru_16/zeros/packed:output:0gru_16/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/zeros?
gru_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_16/transpose/perm?
gru_16/transpose	Transposeinputsgru_16/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2
gru_16/transposed
gru_16/Shape_1Shapegru_16/transpose:y:0*
T0*
_output_shapes
:2
gru_16/Shape_1?
gru_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_16/strided_slice_1/stack?
gru_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_1/stack_1?
gru_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_1/stack_2?
gru_16/strided_slice_1StridedSlicegru_16/Shape_1:output:0%gru_16/strided_slice_1/stack:output:0'gru_16/strided_slice_1/stack_1:output:0'gru_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_16/strided_slice_1?
"gru_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_16/TensorArrayV2/element_shape?
gru_16/TensorArrayV2TensorListReserve+gru_16/TensorArrayV2/element_shape:output:0gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_16/TensorArrayV2?
<gru_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_16/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_16/transpose:y:0Egru_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_16/TensorArrayUnstack/TensorListFromTensor?
gru_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_16/strided_slice_2/stack?
gru_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_2/stack_1?
gru_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_2/stack_2?
gru_16/strided_slice_2StridedSlicegru_16/transpose:y:0%gru_16/strided_slice_2/stack:output:0'gru_16/strided_slice_2/stack_1:output:0'gru_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_16/strided_slice_2?
!gru_16/gru_cell_20/ReadVariableOpReadVariableOp*gru_16_gru_cell_20_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_16/gru_cell_20/ReadVariableOp?
gru_16/gru_cell_20/unstackUnpack)gru_16/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_16/gru_cell_20/unstack?
(gru_16/gru_cell_20/MatMul/ReadVariableOpReadVariableOp1gru_16_gru_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(gru_16/gru_cell_20/MatMul/ReadVariableOp?
gru_16/gru_cell_20/MatMulMatMulgru_16/strided_slice_2:output:00gru_16/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/MatMul?
gru_16/gru_cell_20/BiasAddBiasAdd#gru_16/gru_cell_20/MatMul:product:0#gru_16/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/BiasAddv
gru_16/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_16/gru_cell_20/Const?
"gru_16/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_16/gru_cell_20/split/split_dim?
gru_16/gru_cell_20/splitSplit+gru_16/gru_cell_20/split/split_dim:output:0#gru_16/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_16/gru_cell_20/split?
*gru_16/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp3gru_16_gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02,
*gru_16/gru_cell_20/MatMul_1/ReadVariableOp?
gru_16/gru_cell_20/MatMul_1MatMulgru_16/zeros:output:02gru_16/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/MatMul_1?
gru_16/gru_cell_20/BiasAdd_1BiasAdd%gru_16/gru_cell_20/MatMul_1:product:0#gru_16/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/BiasAdd_1?
gru_16/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_16/gru_cell_20/Const_1?
$gru_16/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_16/gru_cell_20/split_1/split_dim?
gru_16/gru_cell_20/split_1SplitV%gru_16/gru_cell_20/BiasAdd_1:output:0#gru_16/gru_cell_20/Const_1:output:0-gru_16/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_16/gru_cell_20/split_1?
gru_16/gru_cell_20/addAddV2!gru_16/gru_cell_20/split:output:0#gru_16/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add?
gru_16/gru_cell_20/SigmoidSigmoidgru_16/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/Sigmoid?
gru_16/gru_cell_20/add_1AddV2!gru_16/gru_cell_20/split:output:1#gru_16/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add_1?
gru_16/gru_cell_20/Sigmoid_1Sigmoidgru_16/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/Sigmoid_1?
gru_16/gru_cell_20/mulMul gru_16/gru_cell_20/Sigmoid_1:y:0#gru_16/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/mul?
gru_16/gru_cell_20/add_2AddV2!gru_16/gru_cell_20/split:output:2gru_16/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add_2?
gru_16/gru_cell_20/TanhTanhgru_16/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/Tanh?
gru_16/gru_cell_20/mul_1Mulgru_16/gru_cell_20/Sigmoid:y:0gru_16/zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/mul_1y
gru_16/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_16/gru_cell_20/sub/x?
gru_16/gru_cell_20/subSub!gru_16/gru_cell_20/sub/x:output:0gru_16/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/sub?
gru_16/gru_cell_20/mul_2Mulgru_16/gru_cell_20/sub:z:0gru_16/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/mul_2?
gru_16/gru_cell_20/add_3AddV2gru_16/gru_cell_20/mul_1:z:0gru_16/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add_3?
$gru_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2&
$gru_16/TensorArrayV2_1/element_shape?
gru_16/TensorArrayV2_1TensorListReserve-gru_16/TensorArrayV2_1/element_shape:output:0gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_16/TensorArrayV2_1\
gru_16/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_16/time?
gru_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_16/while/maximum_iterationsx
gru_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_16/while/loop_counter?
gru_16/whileWhile"gru_16/while/loop_counter:output:0(gru_16/while/maximum_iterations:output:0gru_16/time:output:0gru_16/TensorArrayV2_1:handle:0gru_16/zeros:output:0gru_16/strided_slice_1:output:0>gru_16/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_16_gru_cell_20_readvariableop_resource1gru_16_gru_cell_20_matmul_readvariableop_resource3gru_16_gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*$
bodyR
gru_16_while_body_514956*$
condR
gru_16_while_cond_514955*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
gru_16/while?
7gru_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7gru_16/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_16/TensorArrayV2Stack/TensorListStackTensorListStackgru_16/while:output:3@gru_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype02+
)gru_16/TensorArrayV2Stack/TensorListStack?
gru_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_16/strided_slice_3/stack?
gru_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_16/strided_slice_3/stack_1?
gru_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_3/stack_2?
gru_16/strided_slice_3StridedSlice2gru_16/TensorArrayV2Stack/TensorListStack:tensor:0%gru_16/strided_slice_3/stack:output:0'gru_16/strided_slice_3/stack_1:output:0'gru_16/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
gru_16/strided_slice_3?
gru_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_16/transpose_1/perm?
gru_16/transpose_1	Transpose2gru_16/TensorArrayV2Stack/TensorListStack:tensor:0 gru_16/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2
gru_16/transpose_1t
gru_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_16/runtimey
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_17/dropout/Const?
dropout_17/dropout/MulMulgru_16/strided_slice_3:output:0!dropout_17/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout_17/dropout/Mul?
dropout_17/dropout/ShapeShapegru_16/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape?
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype021
/dropout_17/dropout/random_uniform/RandomUniform?
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_17/dropout/GreaterEqual/y?
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2!
dropout_17/dropout/GreaterEqual?
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout_17/dropout/Cast?
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout_17/dropout/Mul_1?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldropout_17/dropout/Mul_1:z:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_34/BiasAdds
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_34/Relu~
IdentityIdentitydense_34/Relu:activations:0^gru_16/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::2
gru_16/whilegru_16/while:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
e
F__inference_dropout_17_layer_call_and_return_conditional_losses_515950

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
while_cond_513808
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_513808___redundant_placeholder04
0while_while_cond_513808___redundant_placeholder14
0while_while_cond_513808___redundant_placeholder24
0while_while_cond_513808___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?X
?
B__inference_gru_16_layer_call_and_return_conditional_losses_515576
inputs_0'
#gru_cell_20_readvariableop_resource.
*gru_cell_20_matmul_readvariableop_resource0
,gru_cell_20_matmul_1_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_20/ReadVariableOpReadVariableOp#gru_cell_20_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_20/ReadVariableOp?
gru_cell_20/unstackUnpack"gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_20/unstack?
!gru_cell_20/MatMul/ReadVariableOpReadVariableOp*gru_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_20/MatMul/ReadVariableOp?
gru_cell_20/MatMulMatMulstrided_slice_2:output:0)gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/MatMul?
gru_cell_20/BiasAddBiasAddgru_cell_20/MatMul:product:0gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/BiasAddh
gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_20/Const?
gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_20/split/split_dim?
gru_cell_20/splitSplit$gru_cell_20/split/split_dim:output:0gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_20/split?
#gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02%
#gru_cell_20/MatMul_1/ReadVariableOp?
gru_cell_20/MatMul_1MatMulzeros:output:0+gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/MatMul_1?
gru_cell_20/BiasAdd_1BiasAddgru_cell_20/MatMul_1:product:0gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_20/BiasAdd_1
gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_cell_20/Const_1?
gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_20/split_1/split_dim?
gru_cell_20/split_1SplitVgru_cell_20/BiasAdd_1:output:0gru_cell_20/Const_1:output:0&gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_20/split_1?
gru_cell_20/addAddV2gru_cell_20/split:output:0gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add|
gru_cell_20/SigmoidSigmoidgru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Sigmoid?
gru_cell_20/add_1AddV2gru_cell_20/split:output:1gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_1?
gru_cell_20/Sigmoid_1Sigmoidgru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Sigmoid_1?
gru_cell_20/mulMulgru_cell_20/Sigmoid_1:y:0gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul?
gru_cell_20/add_2AddV2gru_cell_20/split:output:2gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_2u
gru_cell_20/TanhTanhgru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Tanh?
gru_cell_20/mul_1Mulgru_cell_20/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul_1k
gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_20/sub/x?
gru_cell_20/subSubgru_cell_20/sub/x:output:0gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/sub?
gru_cell_20/mul_2Mulgru_cell_20/sub:z:0gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul_2?
gru_cell_20/add_3AddV2gru_cell_20/mul_1:z:0gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_20_readvariableop_resource*gru_cell_20_matmul_readvariableop_resource,gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_515486*
condR
while_cond_515485*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?	
?
,__inference_gru_cell_20_layer_call_fn_516093

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_5135502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????d:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0
?
?
$__inference_signature_wrapper_514516
gru_16_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_5134382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????x
&
_user_specified_namegru_16_input
?@
?
while_body_514072
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_20_readvariableop_resource_06
2while_gru_cell_20_matmul_readvariableop_resource_08
4while_gru_cell_20_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_20_readvariableop_resource4
0while_gru_cell_20_matmul_readvariableop_resource6
2while_gru_cell_20_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_20/ReadVariableOpReadVariableOp+while_gru_cell_20_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_20/ReadVariableOp?
while/gru_cell_20/unstackUnpack(while/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_20/unstack?
'while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_20/MatMul/ReadVariableOp?
while/gru_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/MatMul?
while/gru_cell_20/BiasAddBiasAdd"while/gru_cell_20/MatMul:product:0"while/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/BiasAddt
while/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_20/Const?
!while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_20/split/split_dim?
while/gru_cell_20/splitSplit*while/gru_cell_20/split/split_dim:output:0"while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_20/split?
)while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02+
)while/gru_cell_20/MatMul_1/ReadVariableOp?
while/gru_cell_20/MatMul_1MatMulwhile_placeholder_21while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/MatMul_1?
while/gru_cell_20/BiasAdd_1BiasAdd$while/gru_cell_20/MatMul_1:product:0"while/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/BiasAdd_1?
while/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
while/gru_cell_20/Const_1?
#while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_20/split_1/split_dim?
while/gru_cell_20/split_1SplitV$while/gru_cell_20/BiasAdd_1:output:0"while/gru_cell_20/Const_1:output:0,while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_20/split_1?
while/gru_cell_20/addAddV2 while/gru_cell_20/split:output:0"while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add?
while/gru_cell_20/SigmoidSigmoidwhile/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Sigmoid?
while/gru_cell_20/add_1AddV2 while/gru_cell_20/split:output:1"while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_1?
while/gru_cell_20/Sigmoid_1Sigmoidwhile/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Sigmoid_1?
while/gru_cell_20/mulMulwhile/gru_cell_20/Sigmoid_1:y:0"while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul?
while/gru_cell_20/add_2AddV2 while/gru_cell_20/split:output:2while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_2?
while/gru_cell_20/TanhTanhwhile/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Tanh?
while/gru_cell_20/mul_1Mulwhile/gru_cell_20/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul_1w
while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_20/sub/x?
while/gru_cell_20/subSub while/gru_cell_20/sub/x:output:0while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/sub?
while/gru_cell_20/mul_2Mulwhile/gru_cell_20/sub:z:0while/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul_2?
while/gru_cell_20/add_3AddV2while/gru_cell_20/mul_1:z:0while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_20/add_3:z:0*
T0*'
_output_shapes
:?????????d2
while/Identity_4"j
2while_gru_cell_20_matmul_1_readvariableop_resource4while_gru_cell_20_matmul_1_readvariableop_resource_0"f
0while_gru_cell_20_matmul_readvariableop_resource2while_gru_cell_20_matmul_readvariableop_resource_0"X
)while_gru_cell_20_readvariableop_resource+while_gru_cell_20_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
I__inference_sequential_34_layer_call_and_return_conditional_losses_514478

inputs
gru_16_514464
gru_16_514466
gru_16_514468
dense_34_514472
dense_34_514474
identity?? dense_34/StatefulPartitionedCall?gru_16/StatefulPartitionedCall?
gru_16/StatefulPartitionedCallStatefulPartitionedCallinputsgru_16_514464gru_16_514466gru_16_514468*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_16_layer_call_and_return_conditional_losses_5143212 
gru_16/StatefulPartitionedCall?
dropout_17/PartitionedCallPartitionedCall'gru_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_5143682
dropout_17/PartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_34_514472dense_34_514474*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_5143922"
 dense_34/StatefulPartitionedCall?
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall^gru_16/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2@
gru_16/StatefulPartitionedCallgru_16/StatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?	
?
gru_16_while_cond_514758*
&gru_16_while_gru_16_while_loop_counter0
,gru_16_while_gru_16_while_maximum_iterations
gru_16_while_placeholder
gru_16_while_placeholder_1
gru_16_while_placeholder_2,
(gru_16_while_less_gru_16_strided_slice_1B
>gru_16_while_gru_16_while_cond_514758___redundant_placeholder0B
>gru_16_while_gru_16_while_cond_514758___redundant_placeholder1B
>gru_16_while_gru_16_while_cond_514758___redundant_placeholder2B
>gru_16_while_gru_16_while_cond_514758___redundant_placeholder3
gru_16_while_identity
?
gru_16/while/LessLessgru_16_while_placeholder(gru_16_while_less_gru_16_strided_slice_1*
T0*
_output_shapes
: 2
gru_16/while/Lessr
gru_16/while/IdentityIdentitygru_16/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_16/while/Identity"7
gru_16_while_identitygru_16/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_gru_16_layer_call_fn_515938

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_16_layer_call_and_return_conditional_losses_5143212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????x:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
while_cond_515326
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_515326___redundant_placeholder04
0while_while_cond_515326___redundant_placeholder14
0while_while_cond_515326___redundant_placeholder24
0while_while_cond_515326___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
d
+__inference_dropout_17_layer_call_fn_515960

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_5143632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?J
?	
"__inference__traced_restore_516228
file_prefix$
 assignvariableop_dense_34_kernel$
 assignvariableop_1_dense_34_bias#
assignvariableop_2_rmsprop_iter$
 assignvariableop_3_rmsprop_decay,
(assignvariableop_4_rmsprop_learning_rate'
#assignvariableop_5_rmsprop_momentum"
assignvariableop_6_rmsprop_rho0
,assignvariableop_7_gru_16_gru_cell_20_kernel:
6assignvariableop_8_gru_16_gru_cell_20_recurrent_kernel.
*assignvariableop_9_gru_16_gru_cell_20_bias
assignvariableop_10_total
assignvariableop_11_count3
/assignvariableop_12_rmsprop_dense_34_kernel_rms1
-assignvariableop_13_rmsprop_dense_34_bias_rms=
9assignvariableop_14_rmsprop_gru_16_gru_cell_20_kernel_rmsG
Cassignvariableop_15_rmsprop_gru_16_gru_cell_20_recurrent_kernel_rms;
7assignvariableop_16_rmsprop_gru_16_gru_cell_20_bias_rms
identity_18??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_34_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_34_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_rmsprop_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_rmsprop_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp(assignvariableop_4_rmsprop_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_rmsprop_momentumIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_rhoIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp,assignvariableop_7_gru_16_gru_cell_20_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp6assignvariableop_8_gru_16_gru_cell_20_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_gru_16_gru_cell_20_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_rmsprop_dense_34_kernel_rmsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp-assignvariableop_13_rmsprop_dense_34_bias_rmsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp9assignvariableop_14_rmsprop_gru_16_gru_cell_20_kernel_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpCassignvariableop_15_rmsprop_gru_16_gru_cell_20_recurrent_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp7assignvariableop_16_rmsprop_gru_16_gru_cell_20_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_169
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_17?
Identity_18IdentityIdentity_17:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_18"#
identity_18Identity_18:output:0*Y
_input_shapesH
F: :::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
while_cond_515485
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_515485___redundant_placeholder04
0while_while_cond_515485___redundant_placeholder14
0while_while_cond_515485___redundant_placeholder24
0while_while_cond_515485___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_515955

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
D__inference_dense_34_layer_call_and_return_conditional_losses_515976

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d:::O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
D__inference_dense_34_layer_call_and_return_conditional_losses_514392

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d:::O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
while_cond_514230
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_514230___redundant_placeholder04
0while_while_cond_514230___redundant_placeholder14
0while_while_cond_514230___redundant_placeholder24
0while_while_cond_514230___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?W
?
B__inference_gru_16_layer_call_and_return_conditional_losses_514162

inputs'
#gru_cell_20_readvariableop_resource.
*gru_cell_20_matmul_readvariableop_resource0
,gru_cell_20_matmul_1_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:x?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_20/ReadVariableOpReadVariableOp#gru_cell_20_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_20/ReadVariableOp?
gru_cell_20/unstackUnpack"gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_20/unstack?
!gru_cell_20/MatMul/ReadVariableOpReadVariableOp*gru_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_20/MatMul/ReadVariableOp?
gru_cell_20/MatMulMatMulstrided_slice_2:output:0)gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/MatMul?
gru_cell_20/BiasAddBiasAddgru_cell_20/MatMul:product:0gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/BiasAddh
gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_20/Const?
gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_20/split/split_dim?
gru_cell_20/splitSplit$gru_cell_20/split/split_dim:output:0gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_20/split?
#gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02%
#gru_cell_20/MatMul_1/ReadVariableOp?
gru_cell_20/MatMul_1MatMulzeros:output:0+gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/MatMul_1?
gru_cell_20/BiasAdd_1BiasAddgru_cell_20/MatMul_1:product:0gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_20/BiasAdd_1
gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_cell_20/Const_1?
gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_20/split_1/split_dim?
gru_cell_20/split_1SplitVgru_cell_20/BiasAdd_1:output:0gru_cell_20/Const_1:output:0&gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_20/split_1?
gru_cell_20/addAddV2gru_cell_20/split:output:0gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add|
gru_cell_20/SigmoidSigmoidgru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Sigmoid?
gru_cell_20/add_1AddV2gru_cell_20/split:output:1gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_1?
gru_cell_20/Sigmoid_1Sigmoidgru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Sigmoid_1?
gru_cell_20/mulMulgru_cell_20/Sigmoid_1:y:0gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul?
gru_cell_20/add_2AddV2gru_cell_20/split:output:2gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_2u
gru_cell_20/TanhTanhgru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Tanh?
gru_cell_20/mul_1Mulgru_cell_20/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul_1k
gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_20/sub/x?
gru_cell_20/subSubgru_cell_20/sub/x:output:0gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/sub?
gru_cell_20/mul_2Mulgru_cell_20/sub:z:0gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul_2?
gru_cell_20/add_3AddV2gru_cell_20/mul_1:z:0gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_20_readvariableop_resource*gru_cell_20_matmul_readvariableop_resource,gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_514072*
condR
while_cond_514071*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????x:::2
whilewhile:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
'__inference_gru_16_layer_call_fn_515587
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_16_layer_call_and_return_conditional_losses_5138732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?<
?
B__inference_gru_16_layer_call_and_return_conditional_losses_513873

inputs
gru_cell_20_513797
gru_cell_20_513799
gru_cell_20_513801
identity??#gru_cell_20/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#gru_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_20_513797gru_cell_20_513799gru_cell_20_513801*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_5135102%
#gru_cell_20/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_20_513797gru_cell_20_513799gru_cell_20_513801*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_513809*
condR
while_cond_513808*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^gru_cell_20/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2J
#gru_cell_20/StatefulPartitionedCall#gru_cell_20/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?W
?
B__inference_gru_16_layer_call_and_return_conditional_losses_515757

inputs'
#gru_cell_20_readvariableop_resource.
*gru_cell_20_matmul_readvariableop_resource0
,gru_cell_20_matmul_1_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:x?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_20/ReadVariableOpReadVariableOp#gru_cell_20_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_20/ReadVariableOp?
gru_cell_20/unstackUnpack"gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_20/unstack?
!gru_cell_20/MatMul/ReadVariableOpReadVariableOp*gru_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_20/MatMul/ReadVariableOp?
gru_cell_20/MatMulMatMulstrided_slice_2:output:0)gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/MatMul?
gru_cell_20/BiasAddBiasAddgru_cell_20/MatMul:product:0gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/BiasAddh
gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_20/Const?
gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_20/split/split_dim?
gru_cell_20/splitSplit$gru_cell_20/split/split_dim:output:0gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_20/split?
#gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02%
#gru_cell_20/MatMul_1/ReadVariableOp?
gru_cell_20/MatMul_1MatMulzeros:output:0+gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_20/MatMul_1?
gru_cell_20/BiasAdd_1BiasAddgru_cell_20/MatMul_1:product:0gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_20/BiasAdd_1
gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_cell_20/Const_1?
gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_20/split_1/split_dim?
gru_cell_20/split_1SplitVgru_cell_20/BiasAdd_1:output:0gru_cell_20/Const_1:output:0&gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_20/split_1?
gru_cell_20/addAddV2gru_cell_20/split:output:0gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add|
gru_cell_20/SigmoidSigmoidgru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Sigmoid?
gru_cell_20/add_1AddV2gru_cell_20/split:output:1gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_1?
gru_cell_20/Sigmoid_1Sigmoidgru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Sigmoid_1?
gru_cell_20/mulMulgru_cell_20/Sigmoid_1:y:0gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul?
gru_cell_20/add_2AddV2gru_cell_20/split:output:2gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_2u
gru_cell_20/TanhTanhgru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/Tanh?
gru_cell_20/mul_1Mulgru_cell_20/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul_1k
gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_20/sub/x?
gru_cell_20/subSubgru_cell_20/sub/x:output:0gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/sub?
gru_cell_20/mul_2Mulgru_cell_20/sub:z:0gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/mul_2?
gru_cell_20/add_3AddV2gru_cell_20/mul_1:z:0gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_20/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_20_readvariableop_resource*gru_cell_20_matmul_readvariableop_resource,gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_515667*
condR
while_cond_515666*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????x:::2
whilewhile:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
I__inference_sequential_34_layer_call_and_return_conditional_losses_514446

inputs
gru_16_514432
gru_16_514434
gru_16_514436
dense_34_514440
dense_34_514442
identity?? dense_34/StatefulPartitionedCall?"dropout_17/StatefulPartitionedCall?gru_16/StatefulPartitionedCall?
gru_16/StatefulPartitionedCallStatefulPartitionedCallinputsgru_16_514432gru_16_514434gru_16_514436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_16_layer_call_and_return_conditional_losses_5141622 
gru_16/StatefulPartitionedCall?
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall'gru_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_5143632$
"dropout_17/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_34_514440dense_34_514442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_5143922"
 dense_34/StatefulPartitionedCall?
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall^gru_16/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2@
gru_16/StatefulPartitionedCallgru_16/StatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
.__inference_sequential_34_layer_call_fn_515258

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_5144782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
.__inference_sequential_34_layer_call_fn_515243

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_5144462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?.
?
__inference__traced_save_516167
file_prefix.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop8
4savev2_gru_16_gru_cell_20_kernel_read_readvariableopB
>savev2_gru_16_gru_cell_20_recurrent_kernel_read_readvariableop6
2savev2_gru_16_gru_cell_20_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_rmsprop_dense_34_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_34_bias_rms_read_readvariableopD
@savev2_rmsprop_gru_16_gru_cell_20_kernel_rms_read_readvariableopN
Jsavev2_rmsprop_gru_16_gru_cell_20_recurrent_kernel_rms_read_readvariableopB
>savev2_rmsprop_gru_16_gru_cell_20_bias_rms_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_88707f3af86b4ea18eaf12371dceb013/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop4savev2_gru_16_gru_cell_20_kernel_read_readvariableop>savev2_gru_16_gru_cell_20_recurrent_kernel_read_readvariableop2savev2_gru_16_gru_cell_20_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_rmsprop_dense_34_kernel_rms_read_readvariableop4savev2_rmsprop_dense_34_bias_rms_read_readvariableop@savev2_rmsprop_gru_16_gru_cell_20_kernel_rms_read_readvariableopJsavev2_rmsprop_gru_16_gru_cell_20_recurrent_kernel_rms_read_readvariableop>savev2_rmsprop_gru_16_gru_cell_20_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 * 
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapesv
t: :d:: : : : : :	?:	d?:	?: : :d::	?:	d?:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:%	!

_output_shapes
:	d?:%
!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	?:%!

_output_shapes
:	d?:%!

_output_shapes
:	?:

_output_shapes
: 
?
?
.__inference_sequential_34_layer_call_fn_514887
gru_16_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_5144782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????x
&
_user_specified_namegru_16_input
?J
?
gru_16_while_body_514956*
&gru_16_while_gru_16_while_loop_counter0
,gru_16_while_gru_16_while_maximum_iterations
gru_16_while_placeholder
gru_16_while_placeholder_1
gru_16_while_placeholder_2)
%gru_16_while_gru_16_strided_slice_1_0e
agru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_06
2gru_16_while_gru_cell_20_readvariableop_resource_0=
9gru_16_while_gru_cell_20_matmul_readvariableop_resource_0?
;gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0
gru_16_while_identity
gru_16_while_identity_1
gru_16_while_identity_2
gru_16_while_identity_3
gru_16_while_identity_4'
#gru_16_while_gru_16_strided_slice_1c
_gru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor4
0gru_16_while_gru_cell_20_readvariableop_resource;
7gru_16_while_gru_cell_20_matmul_readvariableop_resource=
9gru_16_while_gru_cell_20_matmul_1_readvariableop_resource??
>gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0gru_16_while_placeholderGgru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0gru_16/while/TensorArrayV2Read/TensorListGetItem?
'gru_16/while/gru_cell_20/ReadVariableOpReadVariableOp2gru_16_while_gru_cell_20_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_16/while/gru_cell_20/ReadVariableOp?
 gru_16/while/gru_cell_20/unstackUnpack/gru_16/while/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_16/while/gru_cell_20/unstack?
.gru_16/while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp9gru_16_while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.gru_16/while/gru_cell_20/MatMul/ReadVariableOp?
gru_16/while/gru_cell_20/MatMulMatMul7gru_16/while/TensorArrayV2Read/TensorListGetItem:item:06gru_16/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_16/while/gru_cell_20/MatMul?
 gru_16/while/gru_cell_20/BiasAddBiasAdd)gru_16/while/gru_cell_20/MatMul:product:0)gru_16/while/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_16/while/gru_cell_20/BiasAdd?
gru_16/while/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
gru_16/while/gru_cell_20/Const?
(gru_16/while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_16/while/gru_cell_20/split/split_dim?
gru_16/while/gru_cell_20/splitSplit1gru_16/while/gru_cell_20/split/split_dim:output:0)gru_16/while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2 
gru_16/while/gru_cell_20/split?
0gru_16/while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp;gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype022
0gru_16/while/gru_cell_20/MatMul_1/ReadVariableOp?
!gru_16/while/gru_cell_20/MatMul_1MatMulgru_16_while_placeholder_28gru_16/while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_16/while/gru_cell_20/MatMul_1?
"gru_16/while/gru_cell_20/BiasAdd_1BiasAdd+gru_16/while/gru_cell_20/MatMul_1:product:0)gru_16/while/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_16/while/gru_cell_20/BiasAdd_1?
 gru_16/while/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2"
 gru_16/while/gru_cell_20/Const_1?
*gru_16/while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_16/while/gru_cell_20/split_1/split_dim?
 gru_16/while/gru_cell_20/split_1SplitV+gru_16/while/gru_cell_20/BiasAdd_1:output:0)gru_16/while/gru_cell_20/Const_1:output:03gru_16/while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2"
 gru_16/while/gru_cell_20/split_1?
gru_16/while/gru_cell_20/addAddV2'gru_16/while/gru_cell_20/split:output:0)gru_16/while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/add?
 gru_16/while/gru_cell_20/SigmoidSigmoid gru_16/while/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2"
 gru_16/while/gru_cell_20/Sigmoid?
gru_16/while/gru_cell_20/add_1AddV2'gru_16/while/gru_cell_20/split:output:1)gru_16/while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/add_1?
"gru_16/while/gru_cell_20/Sigmoid_1Sigmoid"gru_16/while/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2$
"gru_16/while/gru_cell_20/Sigmoid_1?
gru_16/while/gru_cell_20/mulMul&gru_16/while/gru_cell_20/Sigmoid_1:y:0)gru_16/while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/mul?
gru_16/while/gru_cell_20/add_2AddV2'gru_16/while/gru_cell_20/split:output:2 gru_16/while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/add_2?
gru_16/while/gru_cell_20/TanhTanh"gru_16/while/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/Tanh?
gru_16/while/gru_cell_20/mul_1Mul$gru_16/while/gru_cell_20/Sigmoid:y:0gru_16_while_placeholder_2*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/mul_1?
gru_16/while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_16/while/gru_cell_20/sub/x?
gru_16/while/gru_cell_20/subSub'gru_16/while/gru_cell_20/sub/x:output:0$gru_16/while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/sub?
gru_16/while/gru_cell_20/mul_2Mul gru_16/while/gru_cell_20/sub:z:0!gru_16/while/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/mul_2?
gru_16/while/gru_cell_20/add_3AddV2"gru_16/while/gru_cell_20/mul_1:z:0"gru_16/while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/add_3?
1gru_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_16_while_placeholder_1gru_16_while_placeholder"gru_16/while/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_16/while/TensorArrayV2Write/TensorListSetItemj
gru_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_16/while/add/y?
gru_16/while/addAddV2gru_16_while_placeholdergru_16/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_16/while/addn
gru_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_16/while/add_1/y?
gru_16/while/add_1AddV2&gru_16_while_gru_16_while_loop_countergru_16/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_16/while/add_1s
gru_16/while/IdentityIdentitygru_16/while/add_1:z:0*
T0*
_output_shapes
: 2
gru_16/while/Identity?
gru_16/while/Identity_1Identity,gru_16_while_gru_16_while_maximum_iterations*
T0*
_output_shapes
: 2
gru_16/while/Identity_1u
gru_16/while/Identity_2Identitygru_16/while/add:z:0*
T0*
_output_shapes
: 2
gru_16/while/Identity_2?
gru_16/while/Identity_3IdentityAgru_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru_16/while/Identity_3?
gru_16/while/Identity_4Identity"gru_16/while/gru_cell_20/add_3:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/Identity_4"L
#gru_16_while_gru_16_strided_slice_1%gru_16_while_gru_16_strided_slice_1_0"x
9gru_16_while_gru_cell_20_matmul_1_readvariableop_resource;gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0"t
7gru_16_while_gru_cell_20_matmul_readvariableop_resource9gru_16_while_gru_cell_20_matmul_readvariableop_resource_0"f
0gru_16_while_gru_cell_20_readvariableop_resource2gru_16_while_gru_cell_20_readvariableop_resource_0"7
gru_16_while_identitygru_16/while/Identity:output:0";
gru_16_while_identity_1 gru_16/while/Identity_1:output:0";
gru_16_while_identity_2 gru_16/while/Identity_2:output:0";
gru_16_while_identity_3 gru_16/while/Identity_3:output:0";
gru_16_while_identity_4 gru_16/while/Identity_4:output:0"?
_gru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensoragru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?@
?
while_body_515486
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_20_readvariableop_resource_06
2while_gru_cell_20_matmul_readvariableop_resource_08
4while_gru_cell_20_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_20_readvariableop_resource4
0while_gru_cell_20_matmul_readvariableop_resource6
2while_gru_cell_20_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_20/ReadVariableOpReadVariableOp+while_gru_cell_20_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_20/ReadVariableOp?
while/gru_cell_20/unstackUnpack(while/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_20/unstack?
'while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_20/MatMul/ReadVariableOp?
while/gru_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/MatMul?
while/gru_cell_20/BiasAddBiasAdd"while/gru_cell_20/MatMul:product:0"while/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/BiasAddt
while/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_20/Const?
!while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_20/split/split_dim?
while/gru_cell_20/splitSplit*while/gru_cell_20/split/split_dim:output:0"while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_20/split?
)while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02+
)while/gru_cell_20/MatMul_1/ReadVariableOp?
while/gru_cell_20/MatMul_1MatMulwhile_placeholder_21while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/MatMul_1?
while/gru_cell_20/BiasAdd_1BiasAdd$while/gru_cell_20/MatMul_1:product:0"while/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/BiasAdd_1?
while/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
while/gru_cell_20/Const_1?
#while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_20/split_1/split_dim?
while/gru_cell_20/split_1SplitV$while/gru_cell_20/BiasAdd_1:output:0"while/gru_cell_20/Const_1:output:0,while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_20/split_1?
while/gru_cell_20/addAddV2 while/gru_cell_20/split:output:0"while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add?
while/gru_cell_20/SigmoidSigmoidwhile/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Sigmoid?
while/gru_cell_20/add_1AddV2 while/gru_cell_20/split:output:1"while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_1?
while/gru_cell_20/Sigmoid_1Sigmoidwhile/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Sigmoid_1?
while/gru_cell_20/mulMulwhile/gru_cell_20/Sigmoid_1:y:0"while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul?
while/gru_cell_20/add_2AddV2 while/gru_cell_20/split:output:2while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_2?
while/gru_cell_20/TanhTanhwhile/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Tanh?
while/gru_cell_20/mul_1Mulwhile/gru_cell_20/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul_1w
while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_20/sub/x?
while/gru_cell_20/subSub while/gru_cell_20/sub/x:output:0while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/sub?
while/gru_cell_20/mul_2Mulwhile/gru_cell_20/sub:z:0while/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul_2?
while/gru_cell_20/add_3AddV2while/gru_cell_20/mul_1:z:0while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_20/add_3:z:0*
T0*'
_output_shapes
:?????????d2
while/Identity_4"j
2while_gru_cell_20_matmul_1_readvariableop_resource4while_gru_cell_20_matmul_1_readvariableop_resource_0"f
0while_gru_cell_20_matmul_readvariableop_resource2while_gru_cell_20_matmul_readvariableop_resource_0"X
)while_gru_cell_20_readvariableop_resource+while_gru_cell_20_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?l
?
I__inference_sequential_34_layer_call_and_return_conditional_losses_514857
gru_16_input.
*gru_16_gru_cell_20_readvariableop_resource5
1gru_16_gru_cell_20_matmul_readvariableop_resource7
3gru_16_gru_cell_20_matmul_1_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource
identity??gru_16/whileX
gru_16/ShapeShapegru_16_input*
T0*
_output_shapes
:2
gru_16/Shape?
gru_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_16/strided_slice/stack?
gru_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_16/strided_slice/stack_1?
gru_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_16/strided_slice/stack_2?
gru_16/strided_sliceStridedSlicegru_16/Shape:output:0#gru_16/strided_slice/stack:output:0%gru_16/strided_slice/stack_1:output:0%gru_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_16/strided_slicej
gru_16/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
gru_16/zeros/mul/y?
gru_16/zeros/mulMulgru_16/strided_slice:output:0gru_16/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_16/zeros/mulm
gru_16/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_16/zeros/Less/y?
gru_16/zeros/LessLessgru_16/zeros/mul:z:0gru_16/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_16/zeros/Lessp
gru_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
gru_16/zeros/packed/1?
gru_16/zeros/packedPackgru_16/strided_slice:output:0gru_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_16/zeros/packedm
gru_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_16/zeros/Const?
gru_16/zerosFillgru_16/zeros/packed:output:0gru_16/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/zeros?
gru_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_16/transpose/perm?
gru_16/transpose	Transposegru_16_inputgru_16/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2
gru_16/transposed
gru_16/Shape_1Shapegru_16/transpose:y:0*
T0*
_output_shapes
:2
gru_16/Shape_1?
gru_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_16/strided_slice_1/stack?
gru_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_1/stack_1?
gru_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_1/stack_2?
gru_16/strided_slice_1StridedSlicegru_16/Shape_1:output:0%gru_16/strided_slice_1/stack:output:0'gru_16/strided_slice_1/stack_1:output:0'gru_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_16/strided_slice_1?
"gru_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_16/TensorArrayV2/element_shape?
gru_16/TensorArrayV2TensorListReserve+gru_16/TensorArrayV2/element_shape:output:0gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_16/TensorArrayV2?
<gru_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_16/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_16/transpose:y:0Egru_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_16/TensorArrayUnstack/TensorListFromTensor?
gru_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_16/strided_slice_2/stack?
gru_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_2/stack_1?
gru_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_2/stack_2?
gru_16/strided_slice_2StridedSlicegru_16/transpose:y:0%gru_16/strided_slice_2/stack:output:0'gru_16/strided_slice_2/stack_1:output:0'gru_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_16/strided_slice_2?
!gru_16/gru_cell_20/ReadVariableOpReadVariableOp*gru_16_gru_cell_20_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_16/gru_cell_20/ReadVariableOp?
gru_16/gru_cell_20/unstackUnpack)gru_16/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_16/gru_cell_20/unstack?
(gru_16/gru_cell_20/MatMul/ReadVariableOpReadVariableOp1gru_16_gru_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(gru_16/gru_cell_20/MatMul/ReadVariableOp?
gru_16/gru_cell_20/MatMulMatMulgru_16/strided_slice_2:output:00gru_16/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/MatMul?
gru_16/gru_cell_20/BiasAddBiasAdd#gru_16/gru_cell_20/MatMul:product:0#gru_16/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/BiasAddv
gru_16/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_16/gru_cell_20/Const?
"gru_16/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_16/gru_cell_20/split/split_dim?
gru_16/gru_cell_20/splitSplit+gru_16/gru_cell_20/split/split_dim:output:0#gru_16/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_16/gru_cell_20/split?
*gru_16/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp3gru_16_gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02,
*gru_16/gru_cell_20/MatMul_1/ReadVariableOp?
gru_16/gru_cell_20/MatMul_1MatMulgru_16/zeros:output:02gru_16/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/MatMul_1?
gru_16/gru_cell_20/BiasAdd_1BiasAdd%gru_16/gru_cell_20/MatMul_1:product:0#gru_16/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_16/gru_cell_20/BiasAdd_1?
gru_16/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_16/gru_cell_20/Const_1?
$gru_16/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_16/gru_cell_20/split_1/split_dim?
gru_16/gru_cell_20/split_1SplitV%gru_16/gru_cell_20/BiasAdd_1:output:0#gru_16/gru_cell_20/Const_1:output:0-gru_16/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_16/gru_cell_20/split_1?
gru_16/gru_cell_20/addAddV2!gru_16/gru_cell_20/split:output:0#gru_16/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add?
gru_16/gru_cell_20/SigmoidSigmoidgru_16/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/Sigmoid?
gru_16/gru_cell_20/add_1AddV2!gru_16/gru_cell_20/split:output:1#gru_16/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add_1?
gru_16/gru_cell_20/Sigmoid_1Sigmoidgru_16/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/Sigmoid_1?
gru_16/gru_cell_20/mulMul gru_16/gru_cell_20/Sigmoid_1:y:0#gru_16/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/mul?
gru_16/gru_cell_20/add_2AddV2!gru_16/gru_cell_20/split:output:2gru_16/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add_2?
gru_16/gru_cell_20/TanhTanhgru_16/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/Tanh?
gru_16/gru_cell_20/mul_1Mulgru_16/gru_cell_20/Sigmoid:y:0gru_16/zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/mul_1y
gru_16/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_16/gru_cell_20/sub/x?
gru_16/gru_cell_20/subSub!gru_16/gru_cell_20/sub/x:output:0gru_16/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/sub?
gru_16/gru_cell_20/mul_2Mulgru_16/gru_cell_20/sub:z:0gru_16/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/mul_2?
gru_16/gru_cell_20/add_3AddV2gru_16/gru_cell_20/mul_1:z:0gru_16/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/gru_cell_20/add_3?
$gru_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2&
$gru_16/TensorArrayV2_1/element_shape?
gru_16/TensorArrayV2_1TensorListReserve-gru_16/TensorArrayV2_1/element_shape:output:0gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_16/TensorArrayV2_1\
gru_16/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_16/time?
gru_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_16/while/maximum_iterationsx
gru_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_16/while/loop_counter?
gru_16/whileWhile"gru_16/while/loop_counter:output:0(gru_16/while/maximum_iterations:output:0gru_16/time:output:0gru_16/TensorArrayV2_1:handle:0gru_16/zeros:output:0gru_16/strided_slice_1:output:0>gru_16/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_16_gru_cell_20_readvariableop_resource1gru_16_gru_cell_20_matmul_readvariableop_resource3gru_16_gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*$
bodyR
gru_16_while_body_514759*$
condR
gru_16_while_cond_514758*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
gru_16/while?
7gru_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7gru_16/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_16/TensorArrayV2Stack/TensorListStackTensorListStackgru_16/while:output:3@gru_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype02+
)gru_16/TensorArrayV2Stack/TensorListStack?
gru_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_16/strided_slice_3/stack?
gru_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_16/strided_slice_3/stack_1?
gru_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_16/strided_slice_3/stack_2?
gru_16/strided_slice_3StridedSlice2gru_16/TensorArrayV2Stack/TensorListStack:tensor:0%gru_16/strided_slice_3/stack:output:0'gru_16/strided_slice_3/stack_1:output:0'gru_16/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
gru_16/strided_slice_3?
gru_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_16/transpose_1/perm?
gru_16/transpose_1	Transpose2gru_16/TensorArrayV2Stack/TensorListStack:tensor:0 gru_16/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2
gru_16/transpose_1t
gru_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_16/runtime?
dropout_17/IdentityIdentitygru_16/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d2
dropout_17/Identity?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldropout_17/Identity:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_34/BiasAdds
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_34/Relu~
IdentityIdentitydense_34/Relu:activations:0^gru_16/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::2
gru_16/whilegru_16/while:Y U
+
_output_shapes
:?????????x
&
_user_specified_namegru_16_input
?J
?
gru_16_while_body_514759*
&gru_16_while_gru_16_while_loop_counter0
,gru_16_while_gru_16_while_maximum_iterations
gru_16_while_placeholder
gru_16_while_placeholder_1
gru_16_while_placeholder_2)
%gru_16_while_gru_16_strided_slice_1_0e
agru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_06
2gru_16_while_gru_cell_20_readvariableop_resource_0=
9gru_16_while_gru_cell_20_matmul_readvariableop_resource_0?
;gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0
gru_16_while_identity
gru_16_while_identity_1
gru_16_while_identity_2
gru_16_while_identity_3
gru_16_while_identity_4'
#gru_16_while_gru_16_strided_slice_1c
_gru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor4
0gru_16_while_gru_cell_20_readvariableop_resource;
7gru_16_while_gru_cell_20_matmul_readvariableop_resource=
9gru_16_while_gru_cell_20_matmul_1_readvariableop_resource??
>gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0gru_16_while_placeholderGgru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0gru_16/while/TensorArrayV2Read/TensorListGetItem?
'gru_16/while/gru_cell_20/ReadVariableOpReadVariableOp2gru_16_while_gru_cell_20_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_16/while/gru_cell_20/ReadVariableOp?
 gru_16/while/gru_cell_20/unstackUnpack/gru_16/while/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_16/while/gru_cell_20/unstack?
.gru_16/while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp9gru_16_while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.gru_16/while/gru_cell_20/MatMul/ReadVariableOp?
gru_16/while/gru_cell_20/MatMulMatMul7gru_16/while/TensorArrayV2Read/TensorListGetItem:item:06gru_16/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_16/while/gru_cell_20/MatMul?
 gru_16/while/gru_cell_20/BiasAddBiasAdd)gru_16/while/gru_cell_20/MatMul:product:0)gru_16/while/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_16/while/gru_cell_20/BiasAdd?
gru_16/while/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
gru_16/while/gru_cell_20/Const?
(gru_16/while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_16/while/gru_cell_20/split/split_dim?
gru_16/while/gru_cell_20/splitSplit1gru_16/while/gru_cell_20/split/split_dim:output:0)gru_16/while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2 
gru_16/while/gru_cell_20/split?
0gru_16/while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp;gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype022
0gru_16/while/gru_cell_20/MatMul_1/ReadVariableOp?
!gru_16/while/gru_cell_20/MatMul_1MatMulgru_16_while_placeholder_28gru_16/while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_16/while/gru_cell_20/MatMul_1?
"gru_16/while/gru_cell_20/BiasAdd_1BiasAdd+gru_16/while/gru_cell_20/MatMul_1:product:0)gru_16/while/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_16/while/gru_cell_20/BiasAdd_1?
 gru_16/while/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2"
 gru_16/while/gru_cell_20/Const_1?
*gru_16/while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_16/while/gru_cell_20/split_1/split_dim?
 gru_16/while/gru_cell_20/split_1SplitV+gru_16/while/gru_cell_20/BiasAdd_1:output:0)gru_16/while/gru_cell_20/Const_1:output:03gru_16/while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2"
 gru_16/while/gru_cell_20/split_1?
gru_16/while/gru_cell_20/addAddV2'gru_16/while/gru_cell_20/split:output:0)gru_16/while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/add?
 gru_16/while/gru_cell_20/SigmoidSigmoid gru_16/while/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2"
 gru_16/while/gru_cell_20/Sigmoid?
gru_16/while/gru_cell_20/add_1AddV2'gru_16/while/gru_cell_20/split:output:1)gru_16/while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/add_1?
"gru_16/while/gru_cell_20/Sigmoid_1Sigmoid"gru_16/while/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2$
"gru_16/while/gru_cell_20/Sigmoid_1?
gru_16/while/gru_cell_20/mulMul&gru_16/while/gru_cell_20/Sigmoid_1:y:0)gru_16/while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/mul?
gru_16/while/gru_cell_20/add_2AddV2'gru_16/while/gru_cell_20/split:output:2 gru_16/while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/add_2?
gru_16/while/gru_cell_20/TanhTanh"gru_16/while/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/Tanh?
gru_16/while/gru_cell_20/mul_1Mul$gru_16/while/gru_cell_20/Sigmoid:y:0gru_16_while_placeholder_2*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/mul_1?
gru_16/while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_16/while/gru_cell_20/sub/x?
gru_16/while/gru_cell_20/subSub'gru_16/while/gru_cell_20/sub/x:output:0$gru_16/while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/gru_cell_20/sub?
gru_16/while/gru_cell_20/mul_2Mul gru_16/while/gru_cell_20/sub:z:0!gru_16/while/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/mul_2?
gru_16/while/gru_cell_20/add_3AddV2"gru_16/while/gru_cell_20/mul_1:z:0"gru_16/while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2 
gru_16/while/gru_cell_20/add_3?
1gru_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_16_while_placeholder_1gru_16_while_placeholder"gru_16/while/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_16/while/TensorArrayV2Write/TensorListSetItemj
gru_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_16/while/add/y?
gru_16/while/addAddV2gru_16_while_placeholdergru_16/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_16/while/addn
gru_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_16/while/add_1/y?
gru_16/while/add_1AddV2&gru_16_while_gru_16_while_loop_countergru_16/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_16/while/add_1s
gru_16/while/IdentityIdentitygru_16/while/add_1:z:0*
T0*
_output_shapes
: 2
gru_16/while/Identity?
gru_16/while/Identity_1Identity,gru_16_while_gru_16_while_maximum_iterations*
T0*
_output_shapes
: 2
gru_16/while/Identity_1u
gru_16/while/Identity_2Identitygru_16/while/add:z:0*
T0*
_output_shapes
: 2
gru_16/while/Identity_2?
gru_16/while/Identity_3IdentityAgru_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru_16/while/Identity_3?
gru_16/while/Identity_4Identity"gru_16/while/gru_cell_20/add_3:z:0*
T0*'
_output_shapes
:?????????d2
gru_16/while/Identity_4"L
#gru_16_while_gru_16_strided_slice_1%gru_16_while_gru_16_strided_slice_1_0"x
9gru_16_while_gru_cell_20_matmul_1_readvariableop_resource;gru_16_while_gru_cell_20_matmul_1_readvariableop_resource_0"t
7gru_16_while_gru_cell_20_matmul_readvariableop_resource9gru_16_while_gru_cell_20_matmul_readvariableop_resource_0"f
0gru_16_while_gru_cell_20_readvariableop_resource2gru_16_while_gru_cell_20_readvariableop_resource_0"7
gru_16_while_identitygru_16/while/Identity:output:0";
gru_16_while_identity_1 gru_16/while/Identity_1:output:0";
gru_16_while_identity_2 gru_16/while/Identity_2:output:0";
gru_16_while_identity_3 gru_16/while/Identity_3:output:0";
gru_16_while_identity_4 gru_16/while/Identity_4:output:0"?
_gru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensoragru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?@
?
while_body_514231
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_20_readvariableop_resource_06
2while_gru_cell_20_matmul_readvariableop_resource_08
4while_gru_cell_20_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_20_readvariableop_resource4
0while_gru_cell_20_matmul_readvariableop_resource6
2while_gru_cell_20_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_20/ReadVariableOpReadVariableOp+while_gru_cell_20_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_20/ReadVariableOp?
while/gru_cell_20/unstackUnpack(while/gru_cell_20/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_20/unstack?
'while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_20/MatMul/ReadVariableOp?
while/gru_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/MatMul?
while/gru_cell_20/BiasAddBiasAdd"while/gru_cell_20/MatMul:product:0"while/gru_cell_20/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/BiasAddt
while/gru_cell_20/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_20/Const?
!while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_20/split/split_dim?
while/gru_cell_20/splitSplit*while/gru_cell_20/split/split_dim:output:0"while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_20/split?
)while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02+
)while/gru_cell_20/MatMul_1/ReadVariableOp?
while/gru_cell_20/MatMul_1MatMulwhile_placeholder_21while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/MatMul_1?
while/gru_cell_20/BiasAdd_1BiasAdd$while/gru_cell_20/MatMul_1:product:0"while/gru_cell_20/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_20/BiasAdd_1?
while/gru_cell_20/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
while/gru_cell_20/Const_1?
#while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_20/split_1/split_dim?
while/gru_cell_20/split_1SplitV$while/gru_cell_20/BiasAdd_1:output:0"while/gru_cell_20/Const_1:output:0,while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_20/split_1?
while/gru_cell_20/addAddV2 while/gru_cell_20/split:output:0"while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add?
while/gru_cell_20/SigmoidSigmoidwhile/gru_cell_20/add:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Sigmoid?
while/gru_cell_20/add_1AddV2 while/gru_cell_20/split:output:1"while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_1?
while/gru_cell_20/Sigmoid_1Sigmoidwhile/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Sigmoid_1?
while/gru_cell_20/mulMulwhile/gru_cell_20/Sigmoid_1:y:0"while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul?
while/gru_cell_20/add_2AddV2 while/gru_cell_20/split:output:2while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_2?
while/gru_cell_20/TanhTanhwhile/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/Tanh?
while/gru_cell_20/mul_1Mulwhile/gru_cell_20/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul_1w
while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_20/sub/x?
while/gru_cell_20/subSub while/gru_cell_20/sub/x:output:0while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/sub?
while/gru_cell_20/mul_2Mulwhile/gru_cell_20/sub:z:0while/gru_cell_20/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/mul_2?
while/gru_cell_20/add_3AddV2while/gru_cell_20/mul_1:z:0while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_20/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_20/add_3:z:0*
T0*'
_output_shapes
:?????????d2
while/Identity_4"j
2while_gru_cell_20_matmul_1_readvariableop_resource4while_gru_cell_20_matmul_1_readvariableop_resource_0"f
0while_gru_cell_20_matmul_readvariableop_resource2while_gru_cell_20_matmul_readvariableop_resource_0"X
)while_gru_cell_20_readvariableop_resource+while_gru_cell_20_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?	
?
gru_16_while_cond_515129*
&gru_16_while_gru_16_while_loop_counter0
,gru_16_while_gru_16_while_maximum_iterations
gru_16_while_placeholder
gru_16_while_placeholder_1
gru_16_while_placeholder_2,
(gru_16_while_less_gru_16_strided_slice_1B
>gru_16_while_gru_16_while_cond_515129___redundant_placeholder0B
>gru_16_while_gru_16_while_cond_515129___redundant_placeholder1B
>gru_16_while_gru_16_while_cond_515129___redundant_placeholder2B
>gru_16_while_gru_16_while_cond_515129___redundant_placeholder3
gru_16_while_identity
?
gru_16/while/LessLessgru_16_while_placeholder(gru_16_while_less_gru_16_strided_slice_1*
T0*
_output_shapes
: 2
gru_16/while/Lessr
gru_16/while/IdentityIdentitygru_16/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_16/while/Identity"7
gru_16_while_identitygru_16/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
~
)__inference_dense_34_layer_call_fn_515985

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_5143922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_516025

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1?y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????d2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????d2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????d2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????d2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????d2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????d2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????d2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:?????????d2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????d2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????d2
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:?????????d2

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:?????????d2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????d::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0
?
?
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_513550

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1?y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????d2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????d2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????d2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????d2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????d2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????d2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????d2
Tanh\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:?????????d2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????d2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????d2
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:?????????d2

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:?????????d2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????d::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates
?	
?
,__inference_gru_cell_20_layer_call_fn_516079

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_5135102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????d:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0
?
?
while_cond_514071
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_514071___redundant_placeholder04
0while_while_cond_514071___redundant_placeholder14
0while_while_cond_514071___redundant_placeholder24
0while_while_cond_514071___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
gru_16_input9
serving_default_gru_16_input:0?????????x<
dense_340
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?#
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
O__call__
*P&call_and_return_all_conditional_losses
Q_default_save_signature"? 
_tf_keras_sequential? {"class_name": "Sequential", "name": "sequential_34", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_34", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_16_input"}}, {"class_name": "GRU", "config": {"name": "gru_16", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 120, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_34", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_16_input"}}, {"class_name": "GRU", "config": {"name": "gru_16", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?

cell
_inbound_nodes

state_spec
_outbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"?

_tf_keras_rnn_layer?	{"class_name": "GRU", "name": "gru_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_16", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [16, 120, 1]}}
?
_inbound_nodes
_outbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
_inbound_nodes

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [16, 100]}}
?
iter
	 decay
!learning_rate
"momentum
#rho	rmsJ	rmsK	$rmsL	%rmsM	&rmsN"
	optimizer
C
$0
%1
&2
3
4"
trackable_list_wrapper
C
$0
%1
&2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
'layer_metrics
(metrics
	variables
trainable_variables
)non_trainable_variables
regularization_losses
*layer_regularization_losses

+layers
O__call__
Q_default_save_signature
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
,
Xserving_default"
signature_map
?

$kernel
%recurrent_kernel
&bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GRUCell", "name": "gru_cell_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_20", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0layer_metrics
1metrics
	variables
trainable_variables
2non_trainable_variables
regularization_losses
3layer_regularization_losses

4layers

5states
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
6layer_metrics
7metrics
	variables
trainable_variables
8non_trainable_variables
regularization_losses
9layer_regularization_losses

:layers
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:d2dense_34/kernel
:2dense_34/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;layer_metrics
<metrics
	variables
trainable_variables
=non_trainable_variables
regularization_losses
>layer_regularization_losses

?layers
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
,:*	?2gru_16/gru_cell_20/kernel
6:4	d?2#gru_16/gru_cell_20/recurrent_kernel
*:(	?2gru_16/gru_cell_20/bias
 "
trackable_dict_wrapper
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Alayer_metrics
Bmetrics
,	variables
-trainable_variables
Cnon_trainable_variables
.regularization_losses
Dlayer_regularization_losses

Elayers
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Ftotal
	Gcount
H	variables
I	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
F0
G1"
trackable_list_wrapper
-
H	variables"
_generic_user_object
+:)d2RMSprop/dense_34/kernel/rms
%:#2RMSprop/dense_34/bias/rms
6:4	?2%RMSprop/gru_16/gru_cell_20/kernel/rms
@:>	d?2/RMSprop/gru_16/gru_cell_20/recurrent_kernel/rms
4:2	?2#RMSprop/gru_16/gru_cell_20/bias/rms
?2?
.__inference_sequential_34_layer_call_fn_514872
.__inference_sequential_34_layer_call_fn_515243
.__inference_sequential_34_layer_call_fn_514887
.__inference_sequential_34_layer_call_fn_515258?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_34_layer_call_and_return_conditional_losses_515228
I__inference_sequential_34_layer_call_and_return_conditional_losses_514857
I__inference_sequential_34_layer_call_and_return_conditional_losses_515061
I__inference_sequential_34_layer_call_and_return_conditional_losses_514690?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_513438?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
gru_16_input?????????x
?2?
'__inference_gru_16_layer_call_fn_515598
'__inference_gru_16_layer_call_fn_515938
'__inference_gru_16_layer_call_fn_515927
'__inference_gru_16_layer_call_fn_515587?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_gru_16_layer_call_and_return_conditional_losses_515757
B__inference_gru_16_layer_call_and_return_conditional_losses_515576
B__inference_gru_16_layer_call_and_return_conditional_losses_515916
B__inference_gru_16_layer_call_and_return_conditional_losses_515417?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_17_layer_call_fn_515965
+__inference_dropout_17_layer_call_fn_515960?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_17_layer_call_and_return_conditional_losses_515950
F__inference_dropout_17_layer_call_and_return_conditional_losses_515955?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_34_layer_call_fn_515985?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_34_layer_call_and_return_conditional_losses_515976?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
8B6
$__inference_signature_wrapper_514516gru_16_input
?2?
,__inference_gru_cell_20_layer_call_fn_516093
,__inference_gru_cell_20_layer_call_fn_516079?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_516025
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_516065?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
!__inference__wrapped_model_513438w&$%9?6
/?,
*?'
gru_16_input?????????x
? "3?0
.
dense_34"?
dense_34??????????
D__inference_dense_34_layer_call_and_return_conditional_losses_515976\/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? |
)__inference_dense_34_layer_call_fn_515985O/?,
%?"
 ?
inputs?????????d
? "???????????
F__inference_dropout_17_layer_call_and_return_conditional_losses_515950\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
F__inference_dropout_17_layer_call_and_return_conditional_losses_515955\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? ~
+__inference_dropout_17_layer_call_fn_515960O3?0
)?&
 ?
inputs?????????d
p
? "??????????d~
+__inference_dropout_17_layer_call_fn_515965O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
B__inference_gru_16_layer_call_and_return_conditional_losses_515417}&$%O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????d
? ?
B__inference_gru_16_layer_call_and_return_conditional_losses_515576}&$%O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????d
? ?
B__inference_gru_16_layer_call_and_return_conditional_losses_515757m&$%??<
5?2
$?!
inputs?????????x

 
p

 
? "%?"
?
0?????????d
? ?
B__inference_gru_16_layer_call_and_return_conditional_losses_515916m&$%??<
5?2
$?!
inputs?????????x

 
p 

 
? "%?"
?
0?????????d
? ?
'__inference_gru_16_layer_call_fn_515587p&$%O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "??????????d?
'__inference_gru_16_layer_call_fn_515598p&$%O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "??????????d?
'__inference_gru_16_layer_call_fn_515927`&$%??<
5?2
$?!
inputs?????????x

 
p

 
? "??????????d?
'__inference_gru_16_layer_call_fn_515938`&$%??<
5?2
$?!
inputs?????????x

 
p 

 
? "??????????d?
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_516025?&$%\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????d
p
? "R?O
H?E
?
0/0?????????d
$?!
?
0/1/0?????????d
? ?
G__inference_gru_cell_20_layer_call_and_return_conditional_losses_516065?&$%\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????d
p 
? "R?O
H?E
?
0/0?????????d
$?!
?
0/1/0?????????d
? ?
,__inference_gru_cell_20_layer_call_fn_516079?&$%\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????d
p
? "D?A
?
0?????????d
"?
?
1/0?????????d?
,__inference_gru_cell_20_layer_call_fn_516093?&$%\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????d
p 
? "D?A
?
0?????????d
"?
?
1/0?????????d?
I__inference_sequential_34_layer_call_and_return_conditional_losses_514690q&$%A?>
7?4
*?'
gru_16_input?????????x
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_34_layer_call_and_return_conditional_losses_514857q&$%A?>
7?4
*?'
gru_16_input?????????x
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_34_layer_call_and_return_conditional_losses_515061k&$%;?8
1?.
$?!
inputs?????????x
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_34_layer_call_and_return_conditional_losses_515228k&$%;?8
1?.
$?!
inputs?????????x
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_34_layer_call_fn_514872d&$%A?>
7?4
*?'
gru_16_input?????????x
p

 
? "???????????
.__inference_sequential_34_layer_call_fn_514887d&$%A?>
7?4
*?'
gru_16_input?????????x
p 

 
? "???????????
.__inference_sequential_34_layer_call_fn_515243^&$%;?8
1?.
$?!
inputs?????????x
p

 
? "???????????
.__inference_sequential_34_layer_call_fn_515258^&$%;?8
1?.
$?!
inputs?????????x
p 

 
? "???????????
$__inference_signature_wrapper_514516?&$%I?F
? 
??<
:
gru_16_input*?'
gru_16_input?????????x"3?0
.
dense_34"?
dense_34?????????