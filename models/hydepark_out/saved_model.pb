ŕ
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
 ?"serve*2.3.02unknown8??
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:d*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
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
gru_4/gru_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_namegru_4/gru_cell_4/kernel
?
+gru_4/gru_cell_4/kernel/Read/ReadVariableOpReadVariableOpgru_4/gru_cell_4/kernel*
_output_shapes
:	?*
dtype0
?
!gru_4/gru_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*2
shared_name#!gru_4/gru_cell_4/recurrent_kernel
?
5gru_4/gru_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_4/gru_cell_4/recurrent_kernel*
_output_shapes
:	d?*
dtype0
?
gru_4/gru_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_namegru_4/gru_cell_4/bias
?
)gru_4/gru_cell_4/bias/Read/ReadVariableOpReadVariableOpgru_4/gru_cell_4/bias*
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
RMSprop/dense_4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameRMSprop/dense_4/kernel/rms
?
.RMSprop/dense_4/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_4/kernel/rms*
_output_shapes

:d*
dtype0
?
RMSprop/dense_4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_4/bias/rms
?
,RMSprop/dense_4/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_4/bias/rms*
_output_shapes
:*
dtype0
?
#RMSprop/gru_4/gru_cell_4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#RMSprop/gru_4/gru_cell_4/kernel/rms
?
7RMSprop/gru_4/gru_cell_4/kernel/rms/Read/ReadVariableOpReadVariableOp#RMSprop/gru_4/gru_cell_4/kernel/rms*
_output_shapes
:	?*
dtype0
?
-RMSprop/gru_4/gru_cell_4/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*>
shared_name/-RMSprop/gru_4/gru_cell_4/recurrent_kernel/rms
?
ARMSprop/gru_4/gru_cell_4/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp-RMSprop/gru_4/gru_cell_4/recurrent_kernel/rms*
_output_shapes
:	d?*
dtype0
?
!RMSprop/gru_4/gru_cell_4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*2
shared_name#!RMSprop/gru_4/gru_cell_4/bias/rms
?
5RMSprop/gru_4/gru_cell_4/bias/rms/Read/ReadVariableOpReadVariableOp!RMSprop/gru_4/gru_cell_4/bias/rms*
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
trainable_variables
	variables
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
trainable_variables
regularization_losses
	variables
	keras_api
{
_inbound_nodes
_outbound_nodes
trainable_variables
	variables
regularization_losses
	keras_api
|
_inbound_nodes

kernel
bias
trainable_variables
	variables
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
(non_trainable_variables
trainable_variables
)layer_regularization_losses

*layers
+metrics
	variables
regularization_losses
 
~

$kernel
%recurrent_kernel
&bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
 
 
 

$0
%1
&2
 

$0
%1
&2
?
0layer_metrics
1non_trainable_variables
trainable_variables
2layer_regularization_losses

3layers
regularization_losses
4metrics
	variables

5states
 
 
 
 
 
?
6layer_metrics
7non_trainable_variables
trainable_variables
8layer_regularization_losses

9layers
:metrics
	variables
regularization_losses
 
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
;layer_metrics
<non_trainable_variables
trainable_variables
=layer_regularization_losses

>layers
?metrics
	variables
regularization_losses
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
][
VARIABLE_VALUEgru_4/gru_cell_4/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!gru_4/gru_cell_4/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEgru_4/gru_cell_4/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2

@0
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
Bnon_trainable_variables
,trainable_variables
Clayer_regularization_losses

Dlayers
Emetrics
-	variables
.regularization_losses
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
VARIABLE_VALUERMSprop/dense_4/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUERMSprop/dense_4/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#RMSprop/gru_4/gru_cell_4/kernel/rmsNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-RMSprop/gru_4/gru_cell_4/recurrent_kernel/rmsNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!RMSprop/gru_4/gru_cell_4/bias/rmsNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_gru_4_inputPlaceholder*+
_output_shapes
:?????????x*
dtype0* 
shape:?????????x
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_4_inputgru_4/gru_cell_4/biasgru_4/gru_cell_4/kernel!gru_4/gru_cell_4/recurrent_kerneldense_4/kerneldense_4/bias*
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
$__inference_signature_wrapper_340450
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp+gru_4/gru_cell_4/kernel/Read/ReadVariableOp5gru_4/gru_cell_4/recurrent_kernel/Read/ReadVariableOp)gru_4/gru_cell_4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.RMSprop/dense_4/kernel/rms/Read/ReadVariableOp,RMSprop/dense_4/bias/rms/Read/ReadVariableOp7RMSprop/gru_4/gru_cell_4/kernel/rms/Read/ReadVariableOpARMSprop/gru_4/gru_cell_4/recurrent_kernel/rms/Read/ReadVariableOp5RMSprop/gru_4/gru_cell_4/bias/rms/Read/ReadVariableOpConst*
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
__inference__traced_save_342101
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhogru_4/gru_cell_4/kernel!gru_4/gru_cell_4/recurrent_kernelgru_4/gru_cell_4/biastotalcountRMSprop/dense_4/kernel/rmsRMSprop/dense_4/bias/rms#RMSprop/gru_4/gru_cell_4/kernel/rms-RMSprop/gru_4/gru_cell_4/recurrent_kernel/rms!RMSprop/gru_4/gru_cell_4/bias/rms*
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
"__inference__traced_restore_342162ؐ
?
?
-__inference_sequential_4_layer_call_fn_341177

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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_3403802
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
?
?
gru_4_while_cond_340889(
$gru_4_while_gru_4_while_loop_counter.
*gru_4_while_gru_4_while_maximum_iterations
gru_4_while_placeholder
gru_4_while_placeholder_1
gru_4_while_placeholder_2*
&gru_4_while_less_gru_4_strided_slice_1@
<gru_4_while_gru_4_while_cond_340889___redundant_placeholder0@
<gru_4_while_gru_4_while_cond_340889___redundant_placeholder1@
<gru_4_while_gru_4_while_cond_340889___redundant_placeholder2@
<gru_4_while_gru_4_while_cond_340889___redundant_placeholder3
gru_4_while_identity
?
gru_4/while/LessLessgru_4_while_placeholder&gru_4_while_less_gru_4_strided_slice_1*
T0*
_output_shapes
: 2
gru_4/while/Lesso
gru_4/while/IdentityIdentitygru_4/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_4/while/Identity"5
gru_4_while_identitygru_4/while/Identity:output:0*@
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
?	
?
+__inference_gru_cell_4_layer_call_fn_342013

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
GPU 2J 8? *O
fJRH
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_3394442
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
while_cond_339860
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_339860___redundant_placeholder04
0while_while_cond_339860___redundant_placeholder14
0while_while_cond_339860___redundant_placeholder24
0while_while_cond_339860___redundant_placeholder3
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
?
?
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_341999

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
?
?
while_cond_341600
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_341600___redundant_placeholder04
0while_while_cond_341600___redundant_placeholder14
0while_while_cond_341600___redundant_placeholder24
0while_while_cond_341600___redundant_placeholder3
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
?
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_341884

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
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
 *???>2
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
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_341889

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
?j
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_341162

inputs,
(gru_4_gru_cell_4_readvariableop_resource3
/gru_4_gru_cell_4_matmul_readvariableop_resource5
1gru_4_gru_cell_4_matmul_1_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??gru_4/whileP
gru_4/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_4/Shape?
gru_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice/stack?
gru_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice/stack_1?
gru_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice/stack_2?
gru_4/strided_sliceStridedSlicegru_4/Shape:output:0"gru_4/strided_slice/stack:output:0$gru_4/strided_slice/stack_1:output:0$gru_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_4/strided_sliceh
gru_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
gru_4/zeros/mul/y?
gru_4/zeros/mulMulgru_4/strided_slice:output:0gru_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_4/zeros/mulk
gru_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_4/zeros/Less/y
gru_4/zeros/LessLessgru_4/zeros/mul:z:0gru_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_4/zeros/Lessn
gru_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
gru_4/zeros/packed/1?
gru_4/zeros/packedPackgru_4/strided_slice:output:0gru_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_4/zeros/packedk
gru_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_4/zeros/Const?
gru_4/zerosFillgru_4/zeros/packed:output:0gru_4/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/zeros?
gru_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_4/transpose/perm?
gru_4/transpose	Transposeinputsgru_4/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2
gru_4/transposea
gru_4/Shape_1Shapegru_4/transpose:y:0*
T0*
_output_shapes
:2
gru_4/Shape_1?
gru_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice_1/stack?
gru_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_1/stack_1?
gru_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_1/stack_2?
gru_4/strided_slice_1StridedSlicegru_4/Shape_1:output:0$gru_4/strided_slice_1/stack:output:0&gru_4/strided_slice_1/stack_1:output:0&gru_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_4/strided_slice_1?
!gru_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru_4/TensorArrayV2/element_shape?
gru_4/TensorArrayV2TensorListReserve*gru_4/TensorArrayV2/element_shape:output:0gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_4/TensorArrayV2?
;gru_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2=
;gru_4/TensorArrayUnstack/TensorListFromTensor/element_shape?
-gru_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_4/transpose:y:0Dgru_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_4/TensorArrayUnstack/TensorListFromTensor?
gru_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice_2/stack?
gru_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_2/stack_1?
gru_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_2/stack_2?
gru_4/strided_slice_2StridedSlicegru_4/transpose:y:0$gru_4/strided_slice_2/stack:output:0&gru_4/strided_slice_2/stack_1:output:0&gru_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_4/strided_slice_2?
gru_4/gru_cell_4/ReadVariableOpReadVariableOp(gru_4_gru_cell_4_readvariableop_resource*
_output_shapes
:	?*
dtype02!
gru_4/gru_cell_4/ReadVariableOp?
gru_4/gru_cell_4/unstackUnpack'gru_4/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_4/gru_cell_4/unstack?
&gru_4/gru_cell_4/MatMul/ReadVariableOpReadVariableOp/gru_4_gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&gru_4/gru_cell_4/MatMul/ReadVariableOp?
gru_4/gru_cell_4/MatMulMatMulgru_4/strided_slice_2:output:0.gru_4/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/MatMul?
gru_4/gru_cell_4/BiasAddBiasAdd!gru_4/gru_cell_4/MatMul:product:0!gru_4/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/BiasAddr
gru_4/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/gru_cell_4/Const?
 gru_4/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 gru_4/gru_cell_4/split/split_dim?
gru_4/gru_cell_4/splitSplit)gru_4/gru_cell_4/split/split_dim:output:0!gru_4/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_4/gru_cell_4/split?
(gru_4/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp1gru_4_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02*
(gru_4/gru_cell_4/MatMul_1/ReadVariableOp?
gru_4/gru_cell_4/MatMul_1MatMulgru_4/zeros:output:00gru_4/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/MatMul_1?
gru_4/gru_cell_4/BiasAdd_1BiasAdd#gru_4/gru_cell_4/MatMul_1:product:0!gru_4/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/BiasAdd_1?
gru_4/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_4/gru_cell_4/Const_1?
"gru_4/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_4/gru_cell_4/split_1/split_dim?
gru_4/gru_cell_4/split_1SplitV#gru_4/gru_cell_4/BiasAdd_1:output:0!gru_4/gru_cell_4/Const_1:output:0+gru_4/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_4/gru_cell_4/split_1?
gru_4/gru_cell_4/addAddV2gru_4/gru_cell_4/split:output:0!gru_4/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add?
gru_4/gru_cell_4/SigmoidSigmoidgru_4/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/Sigmoid?
gru_4/gru_cell_4/add_1AddV2gru_4/gru_cell_4/split:output:1!gru_4/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add_1?
gru_4/gru_cell_4/Sigmoid_1Sigmoidgru_4/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/Sigmoid_1?
gru_4/gru_cell_4/mulMulgru_4/gru_cell_4/Sigmoid_1:y:0!gru_4/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/mul?
gru_4/gru_cell_4/add_2AddV2gru_4/gru_cell_4/split:output:2gru_4/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add_2?
gru_4/gru_cell_4/TanhTanhgru_4/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/Tanh?
gru_4/gru_cell_4/mul_1Mulgru_4/gru_cell_4/Sigmoid:y:0gru_4/zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/mul_1u
gru_4/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_4/gru_cell_4/sub/x?
gru_4/gru_cell_4/subSubgru_4/gru_cell_4/sub/x:output:0gru_4/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/sub?
gru_4/gru_cell_4/mul_2Mulgru_4/gru_cell_4/sub:z:0gru_4/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/mul_2?
gru_4/gru_cell_4/add_3AddV2gru_4/gru_cell_4/mul_1:z:0gru_4/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add_3?
#gru_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2%
#gru_4/TensorArrayV2_1/element_shape?
gru_4/TensorArrayV2_1TensorListReserve,gru_4/TensorArrayV2_1/element_shape:output:0gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_4/TensorArrayV2_1Z

gru_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru_4/time?
gru_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
gru_4/while/maximum_iterationsv
gru_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_4/while/loop_counter?
gru_4/whileWhile!gru_4/while/loop_counter:output:0'gru_4/while/maximum_iterations:output:0gru_4/time:output:0gru_4/TensorArrayV2_1:handle:0gru_4/zeros:output:0gru_4/strided_slice_1:output:0=gru_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_4_gru_cell_4_readvariableop_resource/gru_4_gru_cell_4_matmul_readvariableop_resource1gru_4_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*#
bodyR
gru_4_while_body_341064*#
condR
gru_4_while_cond_341063*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
gru_4/while?
6gru_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   28
6gru_4/TensorArrayV2Stack/TensorListStack/element_shape?
(gru_4/TensorArrayV2Stack/TensorListStackTensorListStackgru_4/while:output:3?gru_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype02*
(gru_4/TensorArrayV2Stack/TensorListStack?
gru_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_4/strided_slice_3/stack?
gru_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice_3/stack_1?
gru_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_3/stack_2?
gru_4/strided_slice_3StridedSlice1gru_4/TensorArrayV2Stack/TensorListStack:tensor:0$gru_4/strided_slice_3/stack:output:0&gru_4/strided_slice_3/stack_1:output:0&gru_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
gru_4/strided_slice_3?
gru_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_4/transpose_1/perm?
gru_4/transpose_1	Transpose1gru_4/TensorArrayV2Stack/TensorListStack:tensor:0gru_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2
gru_4/transpose_1r
gru_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_4/runtime?
dropout_4/IdentityIdentitygru_4/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d2
dropout_4/Identity?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldropout_4/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Relu|
IdentityIdentitydense_4/Relu:activations:0^gru_4/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::2
gru_4/whilegru_4/while:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
F
*__inference_dropout_4_layer_call_fn_341899

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
GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_3403022
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
?
?
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_339444

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
?
?
-__inference_sequential_4_layer_call_fn_341192

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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_3404122
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
?
__inference__traced_save_342101
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop6
2savev2_gru_4_gru_cell_4_kernel_read_readvariableop@
<savev2_gru_4_gru_cell_4_recurrent_kernel_read_readvariableop4
0savev2_gru_4_gru_cell_4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_rmsprop_dense_4_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_4_bias_rms_read_readvariableopB
>savev2_rmsprop_gru_4_gru_cell_4_kernel_rms_read_readvariableopL
Hsavev2_rmsprop_gru_4_gru_cell_4_recurrent_kernel_rms_read_readvariableop@
<savev2_rmsprop_gru_4_gru_cell_4_bias_rms_read_readvariableop
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
value3B1 B+_temp_e14d165855b14bdea0fa59acee73b560/part2	
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
ShardedFilename?	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop2savev2_gru_4_gru_cell_4_kernel_read_readvariableop<savev2_gru_4_gru_cell_4_recurrent_kernel_read_readvariableop0savev2_gru_4_gru_cell_4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_rmsprop_dense_4_kernel_rms_read_readvariableop3savev2_rmsprop_dense_4_bias_rms_read_readvariableop>savev2_rmsprop_gru_4_gru_cell_4_kernel_rms_read_readvariableopHsavev2_rmsprop_gru_4_gru_cell_4_recurrent_kernel_rms_read_readvariableop<savev2_rmsprop_gru_4_gru_cell_4_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
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
?
while_cond_341759
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_341759___redundant_placeholder04
0while_while_cond_341759___redundant_placeholder14
0while_while_cond_341759___redundant_placeholder24
0while_while_cond_341759___redundant_placeholder3
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
?
?
gru_4_while_cond_340692(
$gru_4_while_gru_4_while_loop_counter.
*gru_4_while_gru_4_while_maximum_iterations
gru_4_while_placeholder
gru_4_while_placeholder_1
gru_4_while_placeholder_2*
&gru_4_while_less_gru_4_strided_slice_1@
<gru_4_while_gru_4_while_cond_340692___redundant_placeholder0@
<gru_4_while_gru_4_while_cond_340692___redundant_placeholder1@
<gru_4_while_gru_4_while_cond_340692___redundant_placeholder2@
<gru_4_while_gru_4_while_cond_340692___redundant_placeholder3
gru_4_while_identity
?
gru_4/while/LessLessgru_4_while_placeholder&gru_4_while_less_gru_4_strided_slice_1*
T0*
_output_shapes
: 2
gru_4/while/Lesso
gru_4/while/IdentityIdentitygru_4/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_4/while/Identity"5
gru_4_while_identitygru_4/while/Identity:output:0*@
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
?
?
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_339484

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
?
?
gru_4_while_cond_341063(
$gru_4_while_gru_4_while_loop_counter.
*gru_4_while_gru_4_while_maximum_iterations
gru_4_while_placeholder
gru_4_while_placeholder_1
gru_4_while_placeholder_2*
&gru_4_while_less_gru_4_strided_slice_1@
<gru_4_while_gru_4_while_cond_341063___redundant_placeholder0@
<gru_4_while_gru_4_while_cond_341063___redundant_placeholder1@
<gru_4_while_gru_4_while_cond_341063___redundant_placeholder2@
<gru_4_while_gru_4_while_cond_341063___redundant_placeholder3
gru_4_while_identity
?
gru_4/while/LessLessgru_4_while_placeholder&gru_4_while_less_gru_4_strided_slice_1*
T0*
_output_shapes
: 2
gru_4/while/Lesso
gru_4/while/IdentityIdentitygru_4/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_4/while/Identity"5
gru_4_while_identitygru_4/while/Identity:output:0*@
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
-__inference_sequential_4_layer_call_fn_340821
gru_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_3404122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????x
%
_user_specified_namegru_4_input
?K
?	
"__inference__traced_restore_342162
file_prefix#
assignvariableop_dense_4_kernel#
assignvariableop_1_dense_4_bias#
assignvariableop_2_rmsprop_iter$
 assignvariableop_3_rmsprop_decay,
(assignvariableop_4_rmsprop_learning_rate'
#assignvariableop_5_rmsprop_momentum"
assignvariableop_6_rmsprop_rho.
*assignvariableop_7_gru_4_gru_cell_4_kernel8
4assignvariableop_8_gru_4_gru_cell_4_recurrent_kernel,
(assignvariableop_9_gru_4_gru_cell_4_bias
assignvariableop_10_total
assignvariableop_11_count2
.assignvariableop_12_rmsprop_dense_4_kernel_rms0
,assignvariableop_13_rmsprop_dense_4_bias_rms;
7assignvariableop_14_rmsprop_gru_4_gru_cell_4_kernel_rmsE
Aassignvariableop_15_rmsprop_gru_4_gru_cell_4_recurrent_kernel_rms9
5assignvariableop_16_rmsprop_gru_4_gru_cell_4_bias_rms
identity_18??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp*assignvariableop_7_gru_4_gru_cell_4_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp4assignvariableop_8_gru_4_gru_cell_4_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp(assignvariableop_9_gru_4_gru_cell_4_biasIdentity_9:output:0"/device:CPU:0*
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
AssignVariableOp_12AssignVariableOp.assignvariableop_12_rmsprop_dense_4_kernel_rmsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp,assignvariableop_13_rmsprop_dense_4_bias_rmsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_rmsprop_gru_4_gru_cell_4_kernel_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpAassignvariableop_15_rmsprop_gru_4_gru_cell_4_recurrent_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp5assignvariableop_16_rmsprop_gru_4_gru_cell_4_bias_rmsIdentity_16:output:0"/device:CPU:0*
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
?!
?
while_body_339861
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_4_339883_0
while_gru_cell_4_339885_0
while_gru_cell_4_339887_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_4_339883
while_gru_cell_4_339885
while_gru_cell_4_339887??(while/gru_cell_4/StatefulPartitionedCall?
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
(while/gru_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_4_339883_0while_gru_cell_4_339885_0while_gru_cell_4_339887_0*
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
GPU 2J 8? *O
fJRH
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_3394842*
(while/gru_cell_4/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_4/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity1while/gru_cell_4/StatefulPartitionedCall:output:1)^while/gru_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2
while/Identity_4"4
while_gru_cell_4_339883while_gru_cell_4_339883_0"4
while_gru_cell_4_339885while_gru_cell_4_339885_0"4
while_gru_cell_4_339887while_gru_cell_4_339887_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::2T
(while/gru_cell_4/StatefulPartitionedCall(while/gru_cell_4/StatefulPartitionedCall: 
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
while_cond_341260
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_341260___redundant_placeholder04
0while_while_cond_341260___redundant_placeholder14
0while_while_cond_341260___redundant_placeholder24
0while_while_cond_341260___redundant_placeholder3
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
?!
?
while_body_339743
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_4_339765_0
while_gru_cell_4_339767_0
while_gru_cell_4_339769_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_4_339765
while_gru_cell_4_339767
while_gru_cell_4_339769??(while/gru_cell_4/StatefulPartitionedCall?
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
(while/gru_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_4_339765_0while_gru_cell_4_339767_0while_gru_cell_4_339769_0*
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
GPU 2J 8? *O
fJRH
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_3394442*
(while/gru_cell_4/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_4/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity1while/gru_cell_4/StatefulPartitionedCall:output:1)^while/gru_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2
while/Identity_4"4
while_gru_cell_4_339765while_gru_cell_4_339765_0"4
while_gru_cell_4_339767while_gru_cell_4_339767_0"4
while_gru_cell_4_339769while_gru_cell_4_339769_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????d: : :::2T
(while/gru_cell_4/StatefulPartitionedCall(while/gru_cell_4/StatefulPartitionedCall: 
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_340412

inputs
gru_4_340398
gru_4_340400
gru_4_340402
dense_4_340406
dense_4_340408
identity??dense_4/StatefulPartitionedCall?gru_4/StatefulPartitionedCall?
gru_4/StatefulPartitionedCallStatefulPartitionedCallinputsgru_4_340398gru_4_340400gru_4_340402*
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
GPU 2J 8? *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_3402552
gru_4/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall&gru_4/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_3403022
dropout_4/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_4_340406dense_4_340408*
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
GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3403262!
dense_4/StatefulPartitionedCall?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall^gru_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2>
gru_4/StatefulPartitionedCallgru_4/StatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?s
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_340624
gru_4_input,
(gru_4_gru_cell_4_readvariableop_resource3
/gru_4_gru_cell_4_matmul_readvariableop_resource5
1gru_4_gru_cell_4_matmul_1_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??gru_4/whileU
gru_4/ShapeShapegru_4_input*
T0*
_output_shapes
:2
gru_4/Shape?
gru_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice/stack?
gru_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice/stack_1?
gru_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice/stack_2?
gru_4/strided_sliceStridedSlicegru_4/Shape:output:0"gru_4/strided_slice/stack:output:0$gru_4/strided_slice/stack_1:output:0$gru_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_4/strided_sliceh
gru_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
gru_4/zeros/mul/y?
gru_4/zeros/mulMulgru_4/strided_slice:output:0gru_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_4/zeros/mulk
gru_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_4/zeros/Less/y
gru_4/zeros/LessLessgru_4/zeros/mul:z:0gru_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_4/zeros/Lessn
gru_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
gru_4/zeros/packed/1?
gru_4/zeros/packedPackgru_4/strided_slice:output:0gru_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_4/zeros/packedk
gru_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_4/zeros/Const?
gru_4/zerosFillgru_4/zeros/packed:output:0gru_4/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/zeros?
gru_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_4/transpose/perm?
gru_4/transpose	Transposegru_4_inputgru_4/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2
gru_4/transposea
gru_4/Shape_1Shapegru_4/transpose:y:0*
T0*
_output_shapes
:2
gru_4/Shape_1?
gru_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice_1/stack?
gru_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_1/stack_1?
gru_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_1/stack_2?
gru_4/strided_slice_1StridedSlicegru_4/Shape_1:output:0$gru_4/strided_slice_1/stack:output:0&gru_4/strided_slice_1/stack_1:output:0&gru_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_4/strided_slice_1?
!gru_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru_4/TensorArrayV2/element_shape?
gru_4/TensorArrayV2TensorListReserve*gru_4/TensorArrayV2/element_shape:output:0gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_4/TensorArrayV2?
;gru_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2=
;gru_4/TensorArrayUnstack/TensorListFromTensor/element_shape?
-gru_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_4/transpose:y:0Dgru_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_4/TensorArrayUnstack/TensorListFromTensor?
gru_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice_2/stack?
gru_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_2/stack_1?
gru_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_2/stack_2?
gru_4/strided_slice_2StridedSlicegru_4/transpose:y:0$gru_4/strided_slice_2/stack:output:0&gru_4/strided_slice_2/stack_1:output:0&gru_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_4/strided_slice_2?
gru_4/gru_cell_4/ReadVariableOpReadVariableOp(gru_4_gru_cell_4_readvariableop_resource*
_output_shapes
:	?*
dtype02!
gru_4/gru_cell_4/ReadVariableOp?
gru_4/gru_cell_4/unstackUnpack'gru_4/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_4/gru_cell_4/unstack?
&gru_4/gru_cell_4/MatMul/ReadVariableOpReadVariableOp/gru_4_gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&gru_4/gru_cell_4/MatMul/ReadVariableOp?
gru_4/gru_cell_4/MatMulMatMulgru_4/strided_slice_2:output:0.gru_4/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/MatMul?
gru_4/gru_cell_4/BiasAddBiasAdd!gru_4/gru_cell_4/MatMul:product:0!gru_4/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/BiasAddr
gru_4/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/gru_cell_4/Const?
 gru_4/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 gru_4/gru_cell_4/split/split_dim?
gru_4/gru_cell_4/splitSplit)gru_4/gru_cell_4/split/split_dim:output:0!gru_4/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_4/gru_cell_4/split?
(gru_4/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp1gru_4_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02*
(gru_4/gru_cell_4/MatMul_1/ReadVariableOp?
gru_4/gru_cell_4/MatMul_1MatMulgru_4/zeros:output:00gru_4/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/MatMul_1?
gru_4/gru_cell_4/BiasAdd_1BiasAdd#gru_4/gru_cell_4/MatMul_1:product:0!gru_4/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/BiasAdd_1?
gru_4/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_4/gru_cell_4/Const_1?
"gru_4/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_4/gru_cell_4/split_1/split_dim?
gru_4/gru_cell_4/split_1SplitV#gru_4/gru_cell_4/BiasAdd_1:output:0!gru_4/gru_cell_4/Const_1:output:0+gru_4/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_4/gru_cell_4/split_1?
gru_4/gru_cell_4/addAddV2gru_4/gru_cell_4/split:output:0!gru_4/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add?
gru_4/gru_cell_4/SigmoidSigmoidgru_4/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/Sigmoid?
gru_4/gru_cell_4/add_1AddV2gru_4/gru_cell_4/split:output:1!gru_4/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add_1?
gru_4/gru_cell_4/Sigmoid_1Sigmoidgru_4/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/Sigmoid_1?
gru_4/gru_cell_4/mulMulgru_4/gru_cell_4/Sigmoid_1:y:0!gru_4/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/mul?
gru_4/gru_cell_4/add_2AddV2gru_4/gru_cell_4/split:output:2gru_4/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add_2?
gru_4/gru_cell_4/TanhTanhgru_4/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/Tanh?
gru_4/gru_cell_4/mul_1Mulgru_4/gru_cell_4/Sigmoid:y:0gru_4/zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/mul_1u
gru_4/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_4/gru_cell_4/sub/x?
gru_4/gru_cell_4/subSubgru_4/gru_cell_4/sub/x:output:0gru_4/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/sub?
gru_4/gru_cell_4/mul_2Mulgru_4/gru_cell_4/sub:z:0gru_4/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/mul_2?
gru_4/gru_cell_4/add_3AddV2gru_4/gru_cell_4/mul_1:z:0gru_4/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add_3?
#gru_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2%
#gru_4/TensorArrayV2_1/element_shape?
gru_4/TensorArrayV2_1TensorListReserve,gru_4/TensorArrayV2_1/element_shape:output:0gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_4/TensorArrayV2_1Z

gru_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru_4/time?
gru_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
gru_4/while/maximum_iterationsv
gru_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_4/while/loop_counter?
gru_4/whileWhile!gru_4/while/loop_counter:output:0'gru_4/while/maximum_iterations:output:0gru_4/time:output:0gru_4/TensorArrayV2_1:handle:0gru_4/zeros:output:0gru_4/strided_slice_1:output:0=gru_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_4_gru_cell_4_readvariableop_resource/gru_4_gru_cell_4_matmul_readvariableop_resource1gru_4_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*#
bodyR
gru_4_while_body_340519*#
condR
gru_4_while_cond_340518*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
gru_4/while?
6gru_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   28
6gru_4/TensorArrayV2Stack/TensorListStack/element_shape?
(gru_4/TensorArrayV2Stack/TensorListStackTensorListStackgru_4/while:output:3?gru_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype02*
(gru_4/TensorArrayV2Stack/TensorListStack?
gru_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_4/strided_slice_3/stack?
gru_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice_3/stack_1?
gru_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_3/stack_2?
gru_4/strided_slice_3StridedSlice1gru_4/TensorArrayV2Stack/TensorListStack:tensor:0$gru_4/strided_slice_3/stack:output:0&gru_4/strided_slice_3/stack_1:output:0&gru_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
gru_4/strided_slice_3?
gru_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_4/transpose_1/perm?
gru_4/transpose_1	Transpose1gru_4/TensorArrayV2Stack/TensorListStack:tensor:0gru_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2
gru_4/transpose_1r
gru_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_4/runtimew
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_4/dropout/Const?
dropout_4/dropout/MulMulgru_4/strided_slice_3:output:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout_4/dropout/Mul?
dropout_4/dropout/ShapeShapegru_4/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout_4/dropout/Mul_1?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Relu|
IdentityIdentitydense_4/Relu:activations:0^gru_4/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::2
gru_4/whilegru_4/while:X T
+
_output_shapes
:?????????x
%
_user_specified_namegru_4_input
?
?
while_cond_340164
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_340164___redundant_placeholder04
0while_while_cond_340164___redundant_placeholder14
0while_while_cond_340164___redundant_placeholder24
0while_while_cond_340164___redundant_placeholder3
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
?H
?
gru_4_while_body_341064(
$gru_4_while_gru_4_while_loop_counter.
*gru_4_while_gru_4_while_maximum_iterations
gru_4_while_placeholder
gru_4_while_placeholder_1
gru_4_while_placeholder_2'
#gru_4_while_gru_4_strided_slice_1_0c
_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_04
0gru_4_while_gru_cell_4_readvariableop_resource_0;
7gru_4_while_gru_cell_4_matmul_readvariableop_resource_0=
9gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0
gru_4_while_identity
gru_4_while_identity_1
gru_4_while_identity_2
gru_4_while_identity_3
gru_4_while_identity_4%
!gru_4_while_gru_4_strided_slice_1a
]gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor2
.gru_4_while_gru_cell_4_readvariableop_resource9
5gru_4_while_gru_cell_4_matmul_readvariableop_resource;
7gru_4_while_gru_cell_4_matmul_1_readvariableop_resource??
=gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape?
/gru_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0gru_4_while_placeholderFgru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype021
/gru_4/while/TensorArrayV2Read/TensorListGetItem?
%gru_4/while/gru_cell_4/ReadVariableOpReadVariableOp0gru_4_while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	?*
dtype02'
%gru_4/while/gru_cell_4/ReadVariableOp?
gru_4/while/gru_cell_4/unstackUnpack-gru_4/while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2 
gru_4/while/gru_cell_4/unstack?
,gru_4/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp7gru_4_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02.
,gru_4/while/gru_cell_4/MatMul/ReadVariableOp?
gru_4/while/gru_cell_4/MatMulMatMul6gru_4/while/TensorArrayV2Read/TensorListGetItem:item:04gru_4/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_4/while/gru_cell_4/MatMul?
gru_4/while/gru_cell_4/BiasAddBiasAdd'gru_4/while/gru_cell_4/MatMul:product:0'gru_4/while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2 
gru_4/while/gru_cell_4/BiasAdd~
gru_4/while/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/while/gru_cell_4/Const?
&gru_4/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&gru_4/while/gru_cell_4/split/split_dim?
gru_4/while/gru_cell_4/splitSplit/gru_4/while/gru_cell_4/split/split_dim:output:0'gru_4/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_4/while/gru_cell_4/split?
.gru_4/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp9gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype020
.gru_4/while/gru_cell_4/MatMul_1/ReadVariableOp?
gru_4/while/gru_cell_4/MatMul_1MatMulgru_4_while_placeholder_26gru_4/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_4/while/gru_cell_4/MatMul_1?
 gru_4/while/gru_cell_4/BiasAdd_1BiasAdd)gru_4/while/gru_cell_4/MatMul_1:product:0'gru_4/while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2"
 gru_4/while/gru_cell_4/BiasAdd_1?
gru_4/while/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2 
gru_4/while/gru_cell_4/Const_1?
(gru_4/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_4/while/gru_cell_4/split_1/split_dim?
gru_4/while/gru_cell_4/split_1SplitV)gru_4/while/gru_cell_4/BiasAdd_1:output:0'gru_4/while/gru_cell_4/Const_1:output:01gru_4/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2 
gru_4/while/gru_cell_4/split_1?
gru_4/while/gru_cell_4/addAddV2%gru_4/while/gru_cell_4/split:output:0'gru_4/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add?
gru_4/while/gru_cell_4/SigmoidSigmoidgru_4/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2 
gru_4/while/gru_cell_4/Sigmoid?
gru_4/while/gru_cell_4/add_1AddV2%gru_4/while/gru_cell_4/split:output:1'gru_4/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add_1?
 gru_4/while/gru_cell_4/Sigmoid_1Sigmoid gru_4/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2"
 gru_4/while/gru_cell_4/Sigmoid_1?
gru_4/while/gru_cell_4/mulMul$gru_4/while/gru_cell_4/Sigmoid_1:y:0'gru_4/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/mul?
gru_4/while/gru_cell_4/add_2AddV2%gru_4/while/gru_cell_4/split:output:2gru_4/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add_2?
gru_4/while/gru_cell_4/TanhTanh gru_4/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/Tanh?
gru_4/while/gru_cell_4/mul_1Mul"gru_4/while/gru_cell_4/Sigmoid:y:0gru_4_while_placeholder_2*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/mul_1?
gru_4/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_4/while/gru_cell_4/sub/x?
gru_4/while/gru_cell_4/subSub%gru_4/while/gru_cell_4/sub/x:output:0"gru_4/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/sub?
gru_4/while/gru_cell_4/mul_2Mulgru_4/while/gru_cell_4/sub:z:0gru_4/while/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/mul_2?
gru_4/while/gru_cell_4/add_3AddV2 gru_4/while/gru_cell_4/mul_1:z:0 gru_4/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add_3?
0gru_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_4_while_placeholder_1gru_4_while_placeholder gru_4/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype022
0gru_4/while/TensorArrayV2Write/TensorListSetItemh
gru_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/while/add/y?
gru_4/while/addAddV2gru_4_while_placeholdergru_4/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_4/while/addl
gru_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/while/add_1/y?
gru_4/while/add_1AddV2$gru_4_while_gru_4_while_loop_countergru_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_4/while/add_1p
gru_4/while/IdentityIdentitygru_4/while/add_1:z:0*
T0*
_output_shapes
: 2
gru_4/while/Identity?
gru_4/while/Identity_1Identity*gru_4_while_gru_4_while_maximum_iterations*
T0*
_output_shapes
: 2
gru_4/while/Identity_1r
gru_4/while/Identity_2Identitygru_4/while/add:z:0*
T0*
_output_shapes
: 2
gru_4/while/Identity_2?
gru_4/while/Identity_3Identity@gru_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru_4/while/Identity_3?
gru_4/while/Identity_4Identity gru_4/while/gru_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/Identity_4"H
!gru_4_while_gru_4_strided_slice_1#gru_4_while_gru_4_strided_slice_1_0"t
7gru_4_while_gru_cell_4_matmul_1_readvariableop_resource9gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0"p
5gru_4_while_gru_cell_4_matmul_readvariableop_resource7gru_4_while_gru_cell_4_matmul_readvariableop_resource_0"b
.gru_4_while_gru_cell_4_readvariableop_resource0gru_4_while_gru_cell_4_readvariableop_resource_0"5
gru_4_while_identitygru_4/while/Identity:output:0"9
gru_4_while_identity_1gru_4/while/Identity_1:output:0"9
gru_4_while_identity_2gru_4/while/Identity_2:output:0"9
gru_4_while_identity_3gru_4/while/Identity_3:output:0"9
gru_4_while_identity_4gru_4/while/Identity_4:output:0"?
]gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0*>
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
A__inference_gru_4_layer_call_and_return_conditional_losses_339925

inputs
gru_cell_4_339849
gru_cell_4_339851
gru_cell_4_339853
identity??"gru_cell_4/StatefulPartitionedCall?whileD
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
"gru_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_4_339849gru_cell_4_339851gru_cell_4_339853*
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
GPU 2J 8? *O
fJRH
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_3394842$
"gru_cell_4/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_4_339849gru_cell_4_339851gru_cell_4_339853*
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
while_body_339861*
condR
while_cond_339860*8
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
IdentityIdentitystrided_slice_3:output:0#^gru_cell_4/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2H
"gru_cell_4/StatefulPartitionedCall"gru_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?Z
?

$sequential_4_gru_4_while_body_339274B
>sequential_4_gru_4_while_sequential_4_gru_4_while_loop_counterH
Dsequential_4_gru_4_while_sequential_4_gru_4_while_maximum_iterations(
$sequential_4_gru_4_while_placeholder*
&sequential_4_gru_4_while_placeholder_1*
&sequential_4_gru_4_while_placeholder_2A
=sequential_4_gru_4_while_sequential_4_gru_4_strided_slice_1_0}
ysequential_4_gru_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_4_tensorarrayunstack_tensorlistfromtensor_0A
=sequential_4_gru_4_while_gru_cell_4_readvariableop_resource_0H
Dsequential_4_gru_4_while_gru_cell_4_matmul_readvariableop_resource_0J
Fsequential_4_gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0%
!sequential_4_gru_4_while_identity'
#sequential_4_gru_4_while_identity_1'
#sequential_4_gru_4_while_identity_2'
#sequential_4_gru_4_while_identity_3'
#sequential_4_gru_4_while_identity_4?
;sequential_4_gru_4_while_sequential_4_gru_4_strided_slice_1{
wsequential_4_gru_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_4_tensorarrayunstack_tensorlistfromtensor?
;sequential_4_gru_4_while_gru_cell_4_readvariableop_resourceF
Bsequential_4_gru_4_while_gru_cell_4_matmul_readvariableop_resourceH
Dsequential_4_gru_4_while_gru_cell_4_matmul_1_readvariableop_resource??
Jsequential_4/gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jsequential_4/gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape?
<sequential_4/gru_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_4_gru_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_4_tensorarrayunstack_tensorlistfromtensor_0$sequential_4_gru_4_while_placeholderSsequential_4/gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02>
<sequential_4/gru_4/while/TensorArrayV2Read/TensorListGetItem?
2sequential_4/gru_4/while/gru_cell_4/ReadVariableOpReadVariableOp=sequential_4_gru_4_while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	?*
dtype024
2sequential_4/gru_4/while/gru_cell_4/ReadVariableOp?
+sequential_4/gru_4/while/gru_cell_4/unstackUnpack:sequential_4/gru_4/while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2-
+sequential_4/gru_4/while/gru_cell_4/unstack?
9sequential_4/gru_4/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOpDsequential_4_gru_4_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02;
9sequential_4/gru_4/while/gru_cell_4/MatMul/ReadVariableOp?
*sequential_4/gru_4/while/gru_cell_4/MatMulMatMulCsequential_4/gru_4/while/TensorArrayV2Read/TensorListGetItem:item:0Asequential_4/gru_4/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_4/gru_4/while/gru_cell_4/MatMul?
+sequential_4/gru_4/while/gru_cell_4/BiasAddBiasAdd4sequential_4/gru_4/while/gru_cell_4/MatMul:product:04sequential_4/gru_4/while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2-
+sequential_4/gru_4/while/gru_cell_4/BiasAdd?
)sequential_4/gru_4/while/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_4/gru_4/while/gru_cell_4/Const?
3sequential_4/gru_4/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3sequential_4/gru_4/while/gru_cell_4/split/split_dim?
)sequential_4/gru_4/while/gru_cell_4/splitSplit<sequential_4/gru_4/while/gru_cell_4/split/split_dim:output:04sequential_4/gru_4/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2+
)sequential_4/gru_4/while/gru_cell_4/split?
;sequential_4/gru_4/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpFsequential_4_gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02=
;sequential_4/gru_4/while/gru_cell_4/MatMul_1/ReadVariableOp?
,sequential_4/gru_4/while/gru_cell_4/MatMul_1MatMul&sequential_4_gru_4_while_placeholder_2Csequential_4/gru_4/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,sequential_4/gru_4/while/gru_cell_4/MatMul_1?
-sequential_4/gru_4/while/gru_cell_4/BiasAdd_1BiasAdd6sequential_4/gru_4/while/gru_cell_4/MatMul_1:product:04sequential_4/gru_4/while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2/
-sequential_4/gru_4/while/gru_cell_4/BiasAdd_1?
+sequential_4/gru_4/while/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2-
+sequential_4/gru_4/while/gru_cell_4/Const_1?
5sequential_4/gru_4/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5sequential_4/gru_4/while/gru_cell_4/split_1/split_dim?
+sequential_4/gru_4/while/gru_cell_4/split_1SplitV6sequential_4/gru_4/while/gru_cell_4/BiasAdd_1:output:04sequential_4/gru_4/while/gru_cell_4/Const_1:output:0>sequential_4/gru_4/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2-
+sequential_4/gru_4/while/gru_cell_4/split_1?
'sequential_4/gru_4/while/gru_cell_4/addAddV22sequential_4/gru_4/while/gru_cell_4/split:output:04sequential_4/gru_4/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2)
'sequential_4/gru_4/while/gru_cell_4/add?
+sequential_4/gru_4/while/gru_cell_4/SigmoidSigmoid+sequential_4/gru_4/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2-
+sequential_4/gru_4/while/gru_cell_4/Sigmoid?
)sequential_4/gru_4/while/gru_cell_4/add_1AddV22sequential_4/gru_4/while/gru_cell_4/split:output:14sequential_4/gru_4/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2+
)sequential_4/gru_4/while/gru_cell_4/add_1?
-sequential_4/gru_4/while/gru_cell_4/Sigmoid_1Sigmoid-sequential_4/gru_4/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2/
-sequential_4/gru_4/while/gru_cell_4/Sigmoid_1?
'sequential_4/gru_4/while/gru_cell_4/mulMul1sequential_4/gru_4/while/gru_cell_4/Sigmoid_1:y:04sequential_4/gru_4/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2)
'sequential_4/gru_4/while/gru_cell_4/mul?
)sequential_4/gru_4/while/gru_cell_4/add_2AddV22sequential_4/gru_4/while/gru_cell_4/split:output:2+sequential_4/gru_4/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2+
)sequential_4/gru_4/while/gru_cell_4/add_2?
(sequential_4/gru_4/while/gru_cell_4/TanhTanh-sequential_4/gru_4/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2*
(sequential_4/gru_4/while/gru_cell_4/Tanh?
)sequential_4/gru_4/while/gru_cell_4/mul_1Mul/sequential_4/gru_4/while/gru_cell_4/Sigmoid:y:0&sequential_4_gru_4_while_placeholder_2*
T0*'
_output_shapes
:?????????d2+
)sequential_4/gru_4/while/gru_cell_4/mul_1?
)sequential_4/gru_4/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)sequential_4/gru_4/while/gru_cell_4/sub/x?
'sequential_4/gru_4/while/gru_cell_4/subSub2sequential_4/gru_4/while/gru_cell_4/sub/x:output:0/sequential_4/gru_4/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2)
'sequential_4/gru_4/while/gru_cell_4/sub?
)sequential_4/gru_4/while/gru_cell_4/mul_2Mul+sequential_4/gru_4/while/gru_cell_4/sub:z:0,sequential_4/gru_4/while/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2+
)sequential_4/gru_4/while/gru_cell_4/mul_2?
)sequential_4/gru_4/while/gru_cell_4/add_3AddV2-sequential_4/gru_4/while/gru_cell_4/mul_1:z:0-sequential_4/gru_4/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2+
)sequential_4/gru_4/while/gru_cell_4/add_3?
=sequential_4/gru_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_4_gru_4_while_placeholder_1$sequential_4_gru_4_while_placeholder-sequential_4/gru_4/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype02?
=sequential_4/gru_4/while/TensorArrayV2Write/TensorListSetItem?
sequential_4/gru_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_4/gru_4/while/add/y?
sequential_4/gru_4/while/addAddV2$sequential_4_gru_4_while_placeholder'sequential_4/gru_4/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_4/gru_4/while/add?
 sequential_4/gru_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_4/gru_4/while/add_1/y?
sequential_4/gru_4/while/add_1AddV2>sequential_4_gru_4_while_sequential_4_gru_4_while_loop_counter)sequential_4/gru_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_4/gru_4/while/add_1?
!sequential_4/gru_4/while/IdentityIdentity"sequential_4/gru_4/while/add_1:z:0*
T0*
_output_shapes
: 2#
!sequential_4/gru_4/while/Identity?
#sequential_4/gru_4/while/Identity_1IdentityDsequential_4_gru_4_while_sequential_4_gru_4_while_maximum_iterations*
T0*
_output_shapes
: 2%
#sequential_4/gru_4/while/Identity_1?
#sequential_4/gru_4/while/Identity_2Identity sequential_4/gru_4/while/add:z:0*
T0*
_output_shapes
: 2%
#sequential_4/gru_4/while/Identity_2?
#sequential_4/gru_4/while/Identity_3IdentityMsequential_4/gru_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2%
#sequential_4/gru_4/while/Identity_3?
#sequential_4/gru_4/while/Identity_4Identity-sequential_4/gru_4/while/gru_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????d2%
#sequential_4/gru_4/while/Identity_4"?
Dsequential_4_gru_4_while_gru_cell_4_matmul_1_readvariableop_resourceFsequential_4_gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0"?
Bsequential_4_gru_4_while_gru_cell_4_matmul_readvariableop_resourceDsequential_4_gru_4_while_gru_cell_4_matmul_readvariableop_resource_0"|
;sequential_4_gru_4_while_gru_cell_4_readvariableop_resource=sequential_4_gru_4_while_gru_cell_4_readvariableop_resource_0"O
!sequential_4_gru_4_while_identity*sequential_4/gru_4/while/Identity:output:0"S
#sequential_4_gru_4_while_identity_1,sequential_4/gru_4/while/Identity_1:output:0"S
#sequential_4_gru_4_while_identity_2,sequential_4/gru_4/while/Identity_2:output:0"S
#sequential_4_gru_4_while_identity_3,sequential_4/gru_4/while/Identity_3:output:0"S
#sequential_4_gru_4_while_identity_4,sequential_4/gru_4/while/Identity_4:output:0"|
;sequential_4_gru_4_while_sequential_4_gru_4_strided_slice_1=sequential_4_gru_4_while_sequential_4_gru_4_strided_slice_1_0"?
wsequential_4_gru_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_4_tensorarrayunstack_tensorlistfromtensorysequential_4_gru_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_4_tensorarrayunstack_tensorlistfromtensor_0*>
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
while_cond_341419
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_341419___redundant_placeholder04
0while_while_cond_341419___redundant_placeholder14
0while_while_cond_341419___redundant_placeholder24
0while_while_cond_341419___redundant_placeholder3
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
?
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_340380

inputs
gru_4_340366
gru_4_340368
gru_4_340370
dense_4_340374
dense_4_340376
identity??dense_4/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?gru_4/StatefulPartitionedCall?
gru_4/StatefulPartitionedCallStatefulPartitionedCallinputsgru_4_340366gru_4_340368gru_4_340370*
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
GPU 2J 8? *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_3400962
gru_4/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall&gru_4/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_3402972#
!dropout_4/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_4_340374dense_4_340376*
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
GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3403262!
dense_4/StatefulPartitionedCall?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall^gru_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2>
gru_4/StatefulPartitionedCallgru_4/StatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
$sequential_4_gru_4_while_cond_339273B
>sequential_4_gru_4_while_sequential_4_gru_4_while_loop_counterH
Dsequential_4_gru_4_while_sequential_4_gru_4_while_maximum_iterations(
$sequential_4_gru_4_while_placeholder*
&sequential_4_gru_4_while_placeholder_1*
&sequential_4_gru_4_while_placeholder_2D
@sequential_4_gru_4_while_less_sequential_4_gru_4_strided_slice_1Z
Vsequential_4_gru_4_while_sequential_4_gru_4_while_cond_339273___redundant_placeholder0Z
Vsequential_4_gru_4_while_sequential_4_gru_4_while_cond_339273___redundant_placeholder1Z
Vsequential_4_gru_4_while_sequential_4_gru_4_while_cond_339273___redundant_placeholder2Z
Vsequential_4_gru_4_while_sequential_4_gru_4_while_cond_339273___redundant_placeholder3%
!sequential_4_gru_4_while_identity
?
sequential_4/gru_4/while/LessLess$sequential_4_gru_4_while_placeholder@sequential_4_gru_4_while_less_sequential_4_gru_4_strided_slice_1*
T0*
_output_shapes
: 2
sequential_4/gru_4/while/Less?
!sequential_4/gru_4/while/IdentityIdentity!sequential_4/gru_4/while/Less:z:0*
T0
*
_output_shapes
: 2#
!sequential_4/gru_4/while/Identity"O
!sequential_4_gru_4_while_identity*sequential_4/gru_4/while/Identity:output:0*@
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
??
?
while_body_341261
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_4_readvariableop_resource_05
1while_gru_cell_4_matmul_readvariableop_resource_07
3while_gru_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_4_readvariableop_resource3
/while_gru_cell_4_matmul_readvariableop_resource5
1while_gru_cell_4_matmul_1_readvariableop_resource??
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
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_4/ReadVariableOp?
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_4/unstack?
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02(
&while/gru_cell_4/MatMul/ReadVariableOp?
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/MatMul?
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/BiasAddr
while/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_4/Const?
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell_4/split/split_dim?
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_4/split?
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02*
(while/gru_cell_4/MatMul_1/ReadVariableOp?
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/MatMul_1?
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/BiasAdd_1?
while/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
while/gru_cell_4/Const_1?
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"while/gru_cell_4/split_1/split_dim?
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0!while/gru_cell_4/Const_1:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_4/split_1?
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add?
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Sigmoid?
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_1?
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Sigmoid_1?
while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul?
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_2?
while/gru_cell_4/TanhTanhwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Tanh?
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul_1u
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_4/sub/x?
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/sub?
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0while/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul_2?
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
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
while/Identity_3~
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????d2
while/Identity_4"h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
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
?
?
while_cond_340005
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_340005___redundant_placeholder04
0while_while_cond_340005___redundant_placeholder14
0while_while_cond_340005___redundant_placeholder24
0while_while_cond_340005___redundant_placeholder3
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
A__inference_gru_4_layer_call_and_return_conditional_losses_340255

inputs&
"gru_cell_4_readvariableop_resource-
)gru_cell_4_matmul_readvariableop_resource/
+gru_cell_4_matmul_1_readvariableop_resource
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
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_4/ReadVariableOp?
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_4/unstack?
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 gru_cell_4/MatMul/ReadVariableOp?
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/MatMul?
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/BiasAddf
gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_4/Const?
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_4/split/split_dim?
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_4/split?
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02$
"gru_cell_4/MatMul_1/ReadVariableOp?
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/MatMul_1?
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_4/BiasAdd_1}
gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_cell_4/Const_1?
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_4/split_1/split_dim?
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const_1:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_4/split_1?
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/addy
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Sigmoid?
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_1
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Sigmoid_1?
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul?
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_2r
gru_cell_4/TanhTanhgru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Tanh?
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul_1i
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_4/sub/x?
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/sub?
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul_2?
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
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
while_body_340165*
condR
while_cond_340164*8
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
&__inference_gru_4_layer_call_fn_341872
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
GPU 2J 8? *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_3399252
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
?H
?
gru_4_while_body_340890(
$gru_4_while_gru_4_while_loop_counter.
*gru_4_while_gru_4_while_maximum_iterations
gru_4_while_placeholder
gru_4_while_placeholder_1
gru_4_while_placeholder_2'
#gru_4_while_gru_4_strided_slice_1_0c
_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_04
0gru_4_while_gru_cell_4_readvariableop_resource_0;
7gru_4_while_gru_cell_4_matmul_readvariableop_resource_0=
9gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0
gru_4_while_identity
gru_4_while_identity_1
gru_4_while_identity_2
gru_4_while_identity_3
gru_4_while_identity_4%
!gru_4_while_gru_4_strided_slice_1a
]gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor2
.gru_4_while_gru_cell_4_readvariableop_resource9
5gru_4_while_gru_cell_4_matmul_readvariableop_resource;
7gru_4_while_gru_cell_4_matmul_1_readvariableop_resource??
=gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape?
/gru_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0gru_4_while_placeholderFgru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype021
/gru_4/while/TensorArrayV2Read/TensorListGetItem?
%gru_4/while/gru_cell_4/ReadVariableOpReadVariableOp0gru_4_while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	?*
dtype02'
%gru_4/while/gru_cell_4/ReadVariableOp?
gru_4/while/gru_cell_4/unstackUnpack-gru_4/while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2 
gru_4/while/gru_cell_4/unstack?
,gru_4/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp7gru_4_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02.
,gru_4/while/gru_cell_4/MatMul/ReadVariableOp?
gru_4/while/gru_cell_4/MatMulMatMul6gru_4/while/TensorArrayV2Read/TensorListGetItem:item:04gru_4/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_4/while/gru_cell_4/MatMul?
gru_4/while/gru_cell_4/BiasAddBiasAdd'gru_4/while/gru_cell_4/MatMul:product:0'gru_4/while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2 
gru_4/while/gru_cell_4/BiasAdd~
gru_4/while/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/while/gru_cell_4/Const?
&gru_4/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&gru_4/while/gru_cell_4/split/split_dim?
gru_4/while/gru_cell_4/splitSplit/gru_4/while/gru_cell_4/split/split_dim:output:0'gru_4/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_4/while/gru_cell_4/split?
.gru_4/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp9gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype020
.gru_4/while/gru_cell_4/MatMul_1/ReadVariableOp?
gru_4/while/gru_cell_4/MatMul_1MatMulgru_4_while_placeholder_26gru_4/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_4/while/gru_cell_4/MatMul_1?
 gru_4/while/gru_cell_4/BiasAdd_1BiasAdd)gru_4/while/gru_cell_4/MatMul_1:product:0'gru_4/while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2"
 gru_4/while/gru_cell_4/BiasAdd_1?
gru_4/while/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2 
gru_4/while/gru_cell_4/Const_1?
(gru_4/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_4/while/gru_cell_4/split_1/split_dim?
gru_4/while/gru_cell_4/split_1SplitV)gru_4/while/gru_cell_4/BiasAdd_1:output:0'gru_4/while/gru_cell_4/Const_1:output:01gru_4/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2 
gru_4/while/gru_cell_4/split_1?
gru_4/while/gru_cell_4/addAddV2%gru_4/while/gru_cell_4/split:output:0'gru_4/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add?
gru_4/while/gru_cell_4/SigmoidSigmoidgru_4/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2 
gru_4/while/gru_cell_4/Sigmoid?
gru_4/while/gru_cell_4/add_1AddV2%gru_4/while/gru_cell_4/split:output:1'gru_4/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add_1?
 gru_4/while/gru_cell_4/Sigmoid_1Sigmoid gru_4/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2"
 gru_4/while/gru_cell_4/Sigmoid_1?
gru_4/while/gru_cell_4/mulMul$gru_4/while/gru_cell_4/Sigmoid_1:y:0'gru_4/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/mul?
gru_4/while/gru_cell_4/add_2AddV2%gru_4/while/gru_cell_4/split:output:2gru_4/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add_2?
gru_4/while/gru_cell_4/TanhTanh gru_4/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/Tanh?
gru_4/while/gru_cell_4/mul_1Mul"gru_4/while/gru_cell_4/Sigmoid:y:0gru_4_while_placeholder_2*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/mul_1?
gru_4/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_4/while/gru_cell_4/sub/x?
gru_4/while/gru_cell_4/subSub%gru_4/while/gru_cell_4/sub/x:output:0"gru_4/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/sub?
gru_4/while/gru_cell_4/mul_2Mulgru_4/while/gru_cell_4/sub:z:0gru_4/while/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/mul_2?
gru_4/while/gru_cell_4/add_3AddV2 gru_4/while/gru_cell_4/mul_1:z:0 gru_4/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add_3?
0gru_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_4_while_placeholder_1gru_4_while_placeholder gru_4/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype022
0gru_4/while/TensorArrayV2Write/TensorListSetItemh
gru_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/while/add/y?
gru_4/while/addAddV2gru_4_while_placeholdergru_4/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_4/while/addl
gru_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/while/add_1/y?
gru_4/while/add_1AddV2$gru_4_while_gru_4_while_loop_countergru_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_4/while/add_1p
gru_4/while/IdentityIdentitygru_4/while/add_1:z:0*
T0*
_output_shapes
: 2
gru_4/while/Identity?
gru_4/while/Identity_1Identity*gru_4_while_gru_4_while_maximum_iterations*
T0*
_output_shapes
: 2
gru_4/while/Identity_1r
gru_4/while/Identity_2Identitygru_4/while/add:z:0*
T0*
_output_shapes
: 2
gru_4/while/Identity_2?
gru_4/while/Identity_3Identity@gru_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru_4/while/Identity_3?
gru_4/while/Identity_4Identity gru_4/while/gru_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/Identity_4"H
!gru_4_while_gru_4_strided_slice_1#gru_4_while_gru_4_strided_slice_1_0"t
7gru_4_while_gru_cell_4_matmul_1_readvariableop_resource9gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0"p
5gru_4_while_gru_cell_4_matmul_readvariableop_resource7gru_4_while_gru_cell_4_matmul_readvariableop_resource_0"b
.gru_4_while_gru_cell_4_readvariableop_resource0gru_4_while_gru_cell_4_readvariableop_resource_0"5
gru_4_while_identitygru_4/while/Identity:output:0"9
gru_4_while_identity_1gru_4/while/Identity_1:output:0"9
gru_4_while_identity_2gru_4/while/Identity_2:output:0"9
gru_4_while_identity_3gru_4/while/Identity_3:output:0"9
gru_4_while_identity_4gru_4/while/Identity_4:output:0"?
]gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0*>
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
̓
?
!__inference__wrapped_model_339372
gru_4_input9
5sequential_4_gru_4_gru_cell_4_readvariableop_resource@
<sequential_4_gru_4_gru_cell_4_matmul_readvariableop_resourceB
>sequential_4_gru_4_gru_cell_4_matmul_1_readvariableop_resource7
3sequential_4_dense_4_matmul_readvariableop_resource8
4sequential_4_dense_4_biasadd_readvariableop_resource
identity??sequential_4/gru_4/whileo
sequential_4/gru_4/ShapeShapegru_4_input*
T0*
_output_shapes
:2
sequential_4/gru_4/Shape?
&sequential_4/gru_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_4/gru_4/strided_slice/stack?
(sequential_4/gru_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_4/gru_4/strided_slice/stack_1?
(sequential_4/gru_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_4/gru_4/strided_slice/stack_2?
 sequential_4/gru_4/strided_sliceStridedSlice!sequential_4/gru_4/Shape:output:0/sequential_4/gru_4/strided_slice/stack:output:01sequential_4/gru_4/strided_slice/stack_1:output:01sequential_4/gru_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential_4/gru_4/strided_slice?
sequential_4/gru_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2 
sequential_4/gru_4/zeros/mul/y?
sequential_4/gru_4/zeros/mulMul)sequential_4/gru_4/strided_slice:output:0'sequential_4/gru_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_4/gru_4/zeros/mul?
sequential_4/gru_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
sequential_4/gru_4/zeros/Less/y?
sequential_4/gru_4/zeros/LessLess sequential_4/gru_4/zeros/mul:z:0(sequential_4/gru_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential_4/gru_4/zeros/Less?
!sequential_4/gru_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2#
!sequential_4/gru_4/zeros/packed/1?
sequential_4/gru_4/zeros/packedPack)sequential_4/gru_4/strided_slice:output:0*sequential_4/gru_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
sequential_4/gru_4/zeros/packed?
sequential_4/gru_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
sequential_4/gru_4/zeros/Const?
sequential_4/gru_4/zerosFill(sequential_4/gru_4/zeros/packed:output:0'sequential_4/gru_4/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
sequential_4/gru_4/zeros?
!sequential_4/gru_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!sequential_4/gru_4/transpose/perm?
sequential_4/gru_4/transpose	Transposegru_4_input*sequential_4/gru_4/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2
sequential_4/gru_4/transpose?
sequential_4/gru_4/Shape_1Shape sequential_4/gru_4/transpose:y:0*
T0*
_output_shapes
:2
sequential_4/gru_4/Shape_1?
(sequential_4/gru_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_4/gru_4/strided_slice_1/stack?
*sequential_4/gru_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/gru_4/strided_slice_1/stack_1?
*sequential_4/gru_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/gru_4/strided_slice_1/stack_2?
"sequential_4/gru_4/strided_slice_1StridedSlice#sequential_4/gru_4/Shape_1:output:01sequential_4/gru_4/strided_slice_1/stack:output:03sequential_4/gru_4/strided_slice_1/stack_1:output:03sequential_4/gru_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_4/gru_4/strided_slice_1?
.sequential_4/gru_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.sequential_4/gru_4/TensorArrayV2/element_shape?
 sequential_4/gru_4/TensorArrayV2TensorListReserve7sequential_4/gru_4/TensorArrayV2/element_shape:output:0+sequential_4/gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 sequential_4/gru_4/TensorArrayV2?
Hsequential_4/gru_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2J
Hsequential_4/gru_4/TensorArrayUnstack/TensorListFromTensor/element_shape?
:sequential_4/gru_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_4/gru_4/transpose:y:0Qsequential_4/gru_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:sequential_4/gru_4/TensorArrayUnstack/TensorListFromTensor?
(sequential_4/gru_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_4/gru_4/strided_slice_2/stack?
*sequential_4/gru_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/gru_4/strided_slice_2/stack_1?
*sequential_4/gru_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/gru_4/strided_slice_2/stack_2?
"sequential_4/gru_4/strided_slice_2StridedSlice sequential_4/gru_4/transpose:y:01sequential_4/gru_4/strided_slice_2/stack:output:03sequential_4/gru_4/strided_slice_2/stack_1:output:03sequential_4/gru_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2$
"sequential_4/gru_4/strided_slice_2?
,sequential_4/gru_4/gru_cell_4/ReadVariableOpReadVariableOp5sequential_4_gru_4_gru_cell_4_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,sequential_4/gru_4/gru_cell_4/ReadVariableOp?
%sequential_4/gru_4/gru_cell_4/unstackUnpack4sequential_4/gru_4/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2'
%sequential_4/gru_4/gru_cell_4/unstack?
3sequential_4/gru_4/gru_cell_4/MatMul/ReadVariableOpReadVariableOp<sequential_4_gru_4_gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype025
3sequential_4/gru_4/gru_cell_4/MatMul/ReadVariableOp?
$sequential_4/gru_4/gru_cell_4/MatMulMatMul+sequential_4/gru_4/strided_slice_2:output:0;sequential_4/gru_4/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$sequential_4/gru_4/gru_cell_4/MatMul?
%sequential_4/gru_4/gru_cell_4/BiasAddBiasAdd.sequential_4/gru_4/gru_cell_4/MatMul:product:0.sequential_4/gru_4/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2'
%sequential_4/gru_4/gru_cell_4/BiasAdd?
#sequential_4/gru_4/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_4/gru_4/gru_cell_4/Const?
-sequential_4/gru_4/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_4/gru_4/gru_cell_4/split/split_dim?
#sequential_4/gru_4/gru_cell_4/splitSplit6sequential_4/gru_4/gru_cell_4/split/split_dim:output:0.sequential_4/gru_4/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2%
#sequential_4/gru_4/gru_cell_4/split?
5sequential_4/gru_4/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp>sequential_4_gru_4_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype027
5sequential_4/gru_4/gru_cell_4/MatMul_1/ReadVariableOp?
&sequential_4/gru_4/gru_cell_4/MatMul_1MatMul!sequential_4/gru_4/zeros:output:0=sequential_4/gru_4/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential_4/gru_4/gru_cell_4/MatMul_1?
'sequential_4/gru_4/gru_cell_4/BiasAdd_1BiasAdd0sequential_4/gru_4/gru_cell_4/MatMul_1:product:0.sequential_4/gru_4/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2)
'sequential_4/gru_4/gru_cell_4/BiasAdd_1?
%sequential_4/gru_4/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2'
%sequential_4/gru_4/gru_cell_4/Const_1?
/sequential_4/gru_4/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_4/gru_4/gru_cell_4/split_1/split_dim?
%sequential_4/gru_4/gru_cell_4/split_1SplitV0sequential_4/gru_4/gru_cell_4/BiasAdd_1:output:0.sequential_4/gru_4/gru_cell_4/Const_1:output:08sequential_4/gru_4/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2'
%sequential_4/gru_4/gru_cell_4/split_1?
!sequential_4/gru_4/gru_cell_4/addAddV2,sequential_4/gru_4/gru_cell_4/split:output:0.sequential_4/gru_4/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2#
!sequential_4/gru_4/gru_cell_4/add?
%sequential_4/gru_4/gru_cell_4/SigmoidSigmoid%sequential_4/gru_4/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2'
%sequential_4/gru_4/gru_cell_4/Sigmoid?
#sequential_4/gru_4/gru_cell_4/add_1AddV2,sequential_4/gru_4/gru_cell_4/split:output:1.sequential_4/gru_4/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2%
#sequential_4/gru_4/gru_cell_4/add_1?
'sequential_4/gru_4/gru_cell_4/Sigmoid_1Sigmoid'sequential_4/gru_4/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2)
'sequential_4/gru_4/gru_cell_4/Sigmoid_1?
!sequential_4/gru_4/gru_cell_4/mulMul+sequential_4/gru_4/gru_cell_4/Sigmoid_1:y:0.sequential_4/gru_4/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2#
!sequential_4/gru_4/gru_cell_4/mul?
#sequential_4/gru_4/gru_cell_4/add_2AddV2,sequential_4/gru_4/gru_cell_4/split:output:2%sequential_4/gru_4/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2%
#sequential_4/gru_4/gru_cell_4/add_2?
"sequential_4/gru_4/gru_cell_4/TanhTanh'sequential_4/gru_4/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2$
"sequential_4/gru_4/gru_cell_4/Tanh?
#sequential_4/gru_4/gru_cell_4/mul_1Mul)sequential_4/gru_4/gru_cell_4/Sigmoid:y:0!sequential_4/gru_4/zeros:output:0*
T0*'
_output_shapes
:?????????d2%
#sequential_4/gru_4/gru_cell_4/mul_1?
#sequential_4/gru_4/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#sequential_4/gru_4/gru_cell_4/sub/x?
!sequential_4/gru_4/gru_cell_4/subSub,sequential_4/gru_4/gru_cell_4/sub/x:output:0)sequential_4/gru_4/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2#
!sequential_4/gru_4/gru_cell_4/sub?
#sequential_4/gru_4/gru_cell_4/mul_2Mul%sequential_4/gru_4/gru_cell_4/sub:z:0&sequential_4/gru_4/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2%
#sequential_4/gru_4/gru_cell_4/mul_2?
#sequential_4/gru_4/gru_cell_4/add_3AddV2'sequential_4/gru_4/gru_cell_4/mul_1:z:0'sequential_4/gru_4/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2%
#sequential_4/gru_4/gru_cell_4/add_3?
0sequential_4/gru_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0sequential_4/gru_4/TensorArrayV2_1/element_shape?
"sequential_4/gru_4/TensorArrayV2_1TensorListReserve9sequential_4/gru_4/TensorArrayV2_1/element_shape:output:0+sequential_4/gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_4/gru_4/TensorArrayV2_1t
sequential_4/gru_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_4/gru_4/time?
+sequential_4/gru_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+sequential_4/gru_4/while/maximum_iterations?
%sequential_4/gru_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_4/gru_4/while/loop_counter?
sequential_4/gru_4/whileWhile.sequential_4/gru_4/while/loop_counter:output:04sequential_4/gru_4/while/maximum_iterations:output:0 sequential_4/gru_4/time:output:0+sequential_4/gru_4/TensorArrayV2_1:handle:0!sequential_4/gru_4/zeros:output:0+sequential_4/gru_4/strided_slice_1:output:0Jsequential_4/gru_4/TensorArrayUnstack/TensorListFromTensor:output_handle:05sequential_4_gru_4_gru_cell_4_readvariableop_resource<sequential_4_gru_4_gru_cell_4_matmul_readvariableop_resource>sequential_4_gru_4_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*0
body(R&
$sequential_4_gru_4_while_body_339274*0
cond(R&
$sequential_4_gru_4_while_cond_339273*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
sequential_4/gru_4/while?
Csequential_4/gru_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2E
Csequential_4/gru_4/TensorArrayV2Stack/TensorListStack/element_shape?
5sequential_4/gru_4/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_4/gru_4/while:output:3Lsequential_4/gru_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype027
5sequential_4/gru_4/TensorArrayV2Stack/TensorListStack?
(sequential_4/gru_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(sequential_4/gru_4/strided_slice_3/stack?
*sequential_4/gru_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/gru_4/strided_slice_3/stack_1?
*sequential_4/gru_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/gru_4/strided_slice_3/stack_2?
"sequential_4/gru_4/strided_slice_3StridedSlice>sequential_4/gru_4/TensorArrayV2Stack/TensorListStack:tensor:01sequential_4/gru_4/strided_slice_3/stack:output:03sequential_4/gru_4/strided_slice_3/stack_1:output:03sequential_4/gru_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2$
"sequential_4/gru_4/strided_slice_3?
#sequential_4/gru_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_4/gru_4/transpose_1/perm?
sequential_4/gru_4/transpose_1	Transpose>sequential_4/gru_4/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_4/gru_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2 
sequential_4/gru_4/transpose_1?
sequential_4/gru_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_4/gru_4/runtime?
sequential_4/dropout_4/IdentityIdentity+sequential_4/gru_4/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d2!
sequential_4/dropout_4/Identity?
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02,
*sequential_4/dense_4/MatMul/ReadVariableOp?
sequential_4/dense_4/MatMulMatMul(sequential_4/dropout_4/Identity:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/MatMul?
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOp?
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/BiasAdd?
sequential_4/dense_4/ReluRelu%sequential_4/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/Relu?
IdentityIdentity'sequential_4/dense_4/Relu:activations:0^sequential_4/gru_4/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::24
sequential_4/gru_4/whilesequential_4/gru_4/while:X T
+
_output_shapes
:?????????x
%
_user_specified_namegru_4_input
??
?
while_body_340006
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_4_readvariableop_resource_05
1while_gru_cell_4_matmul_readvariableop_resource_07
3while_gru_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_4_readvariableop_resource3
/while_gru_cell_4_matmul_readvariableop_resource5
1while_gru_cell_4_matmul_1_readvariableop_resource??
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
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_4/ReadVariableOp?
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_4/unstack?
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02(
&while/gru_cell_4/MatMul/ReadVariableOp?
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/MatMul?
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/BiasAddr
while/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_4/Const?
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell_4/split/split_dim?
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_4/split?
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02*
(while/gru_cell_4/MatMul_1/ReadVariableOp?
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/MatMul_1?
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/BiasAdd_1?
while/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
while/gru_cell_4/Const_1?
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"while/gru_cell_4/split_1/split_dim?
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0!while/gru_cell_4/Const_1:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_4/split_1?
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add?
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Sigmoid?
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_1?
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Sigmoid_1?
while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul?
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_2?
while/gru_cell_4/TanhTanhwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Tanh?
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul_1u
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_4/sub/x?
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/sub?
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0while/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul_2?
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
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
while/Identity_3~
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????d2
while/Identity_4"h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
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
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_340297

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
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
 *???>2
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
?
&__inference_gru_4_layer_call_fn_341861
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
GPU 2J 8? *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_3398072
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
?
?
gru_4_while_cond_340518(
$gru_4_while_gru_4_while_loop_counter.
*gru_4_while_gru_4_while_maximum_iterations
gru_4_while_placeholder
gru_4_while_placeholder_1
gru_4_while_placeholder_2*
&gru_4_while_less_gru_4_strided_slice_1@
<gru_4_while_gru_4_while_cond_340518___redundant_placeholder0@
<gru_4_while_gru_4_while_cond_340518___redundant_placeholder1@
<gru_4_while_gru_4_while_cond_340518___redundant_placeholder2@
<gru_4_while_gru_4_while_cond_340518___redundant_placeholder3
gru_4_while_identity
?
gru_4/while/LessLessgru_4_while_placeholder&gru_4_while_less_gru_4_strided_slice_1*
T0*
_output_shapes
: 2
gru_4/while/Lesso
gru_4/while/IdentityIdentitygru_4/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_4/while/Identity"5
gru_4_while_identitygru_4/while/Identity:output:0*@
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
?
?
C__inference_dense_4_layer_call_and_return_conditional_losses_341910

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
?
?
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_341959

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
?
?
$__inference_signature_wrapper_340450
gru_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
!__inference__wrapped_model_3393722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????x
%
_user_specified_namegru_4_input
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_340302

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
?
}
(__inference_dense_4_layer_call_fn_341919

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
GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3403262
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
+__inference_gru_cell_4_layer_call_fn_342027

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
GPU 2J 8? *O
fJRH
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_3394842
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
?<
?
A__inference_gru_4_layer_call_and_return_conditional_losses_339807

inputs
gru_cell_4_339731
gru_cell_4_339733
gru_cell_4_339735
identity??"gru_cell_4/StatefulPartitionedCall?whileD
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
"gru_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_4_339731gru_cell_4_339733gru_cell_4_339735*
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
GPU 2J 8? *O
fJRH
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_3394442$
"gru_cell_4/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_4_339731gru_cell_4_339733gru_cell_4_339735*
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
while_body_339743*
condR
while_cond_339742*8
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
IdentityIdentitystrided_slice_3:output:0#^gru_cell_4/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2H
"gru_cell_4/StatefulPartitionedCall"gru_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?
while_body_341601
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_4_readvariableop_resource_05
1while_gru_cell_4_matmul_readvariableop_resource_07
3while_gru_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_4_readvariableop_resource3
/while_gru_cell_4_matmul_readvariableop_resource5
1while_gru_cell_4_matmul_1_readvariableop_resource??
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
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_4/ReadVariableOp?
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_4/unstack?
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02(
&while/gru_cell_4/MatMul/ReadVariableOp?
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/MatMul?
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/BiasAddr
while/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_4/Const?
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell_4/split/split_dim?
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_4/split?
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02*
(while/gru_cell_4/MatMul_1/ReadVariableOp?
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/MatMul_1?
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/BiasAdd_1?
while/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
while/gru_cell_4/Const_1?
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"while/gru_cell_4/split_1/split_dim?
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0!while/gru_cell_4/Const_1:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_4/split_1?
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add?
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Sigmoid?
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_1?
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Sigmoid_1?
while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul?
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_2?
while/gru_cell_4/TanhTanhwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Tanh?
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul_1u
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_4/sub/x?
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/sub?
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0while/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul_2?
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
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
while/Identity_3~
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????d2
while/Identity_4"h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
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
?s
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_340995

inputs,
(gru_4_gru_cell_4_readvariableop_resource3
/gru_4_gru_cell_4_matmul_readvariableop_resource5
1gru_4_gru_cell_4_matmul_1_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??gru_4/whileP
gru_4/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_4/Shape?
gru_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice/stack?
gru_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice/stack_1?
gru_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice/stack_2?
gru_4/strided_sliceStridedSlicegru_4/Shape:output:0"gru_4/strided_slice/stack:output:0$gru_4/strided_slice/stack_1:output:0$gru_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_4/strided_sliceh
gru_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
gru_4/zeros/mul/y?
gru_4/zeros/mulMulgru_4/strided_slice:output:0gru_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_4/zeros/mulk
gru_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_4/zeros/Less/y
gru_4/zeros/LessLessgru_4/zeros/mul:z:0gru_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_4/zeros/Lessn
gru_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
gru_4/zeros/packed/1?
gru_4/zeros/packedPackgru_4/strided_slice:output:0gru_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_4/zeros/packedk
gru_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_4/zeros/Const?
gru_4/zerosFillgru_4/zeros/packed:output:0gru_4/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/zeros?
gru_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_4/transpose/perm?
gru_4/transpose	Transposeinputsgru_4/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2
gru_4/transposea
gru_4/Shape_1Shapegru_4/transpose:y:0*
T0*
_output_shapes
:2
gru_4/Shape_1?
gru_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice_1/stack?
gru_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_1/stack_1?
gru_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_1/stack_2?
gru_4/strided_slice_1StridedSlicegru_4/Shape_1:output:0$gru_4/strided_slice_1/stack:output:0&gru_4/strided_slice_1/stack_1:output:0&gru_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_4/strided_slice_1?
!gru_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru_4/TensorArrayV2/element_shape?
gru_4/TensorArrayV2TensorListReserve*gru_4/TensorArrayV2/element_shape:output:0gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_4/TensorArrayV2?
;gru_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2=
;gru_4/TensorArrayUnstack/TensorListFromTensor/element_shape?
-gru_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_4/transpose:y:0Dgru_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_4/TensorArrayUnstack/TensorListFromTensor?
gru_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice_2/stack?
gru_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_2/stack_1?
gru_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_2/stack_2?
gru_4/strided_slice_2StridedSlicegru_4/transpose:y:0$gru_4/strided_slice_2/stack:output:0&gru_4/strided_slice_2/stack_1:output:0&gru_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_4/strided_slice_2?
gru_4/gru_cell_4/ReadVariableOpReadVariableOp(gru_4_gru_cell_4_readvariableop_resource*
_output_shapes
:	?*
dtype02!
gru_4/gru_cell_4/ReadVariableOp?
gru_4/gru_cell_4/unstackUnpack'gru_4/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_4/gru_cell_4/unstack?
&gru_4/gru_cell_4/MatMul/ReadVariableOpReadVariableOp/gru_4_gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&gru_4/gru_cell_4/MatMul/ReadVariableOp?
gru_4/gru_cell_4/MatMulMatMulgru_4/strided_slice_2:output:0.gru_4/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/MatMul?
gru_4/gru_cell_4/BiasAddBiasAdd!gru_4/gru_cell_4/MatMul:product:0!gru_4/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/BiasAddr
gru_4/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/gru_cell_4/Const?
 gru_4/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 gru_4/gru_cell_4/split/split_dim?
gru_4/gru_cell_4/splitSplit)gru_4/gru_cell_4/split/split_dim:output:0!gru_4/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_4/gru_cell_4/split?
(gru_4/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp1gru_4_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02*
(gru_4/gru_cell_4/MatMul_1/ReadVariableOp?
gru_4/gru_cell_4/MatMul_1MatMulgru_4/zeros:output:00gru_4/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/MatMul_1?
gru_4/gru_cell_4/BiasAdd_1BiasAdd#gru_4/gru_cell_4/MatMul_1:product:0!gru_4/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/BiasAdd_1?
gru_4/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_4/gru_cell_4/Const_1?
"gru_4/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_4/gru_cell_4/split_1/split_dim?
gru_4/gru_cell_4/split_1SplitV#gru_4/gru_cell_4/BiasAdd_1:output:0!gru_4/gru_cell_4/Const_1:output:0+gru_4/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_4/gru_cell_4/split_1?
gru_4/gru_cell_4/addAddV2gru_4/gru_cell_4/split:output:0!gru_4/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add?
gru_4/gru_cell_4/SigmoidSigmoidgru_4/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/Sigmoid?
gru_4/gru_cell_4/add_1AddV2gru_4/gru_cell_4/split:output:1!gru_4/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add_1?
gru_4/gru_cell_4/Sigmoid_1Sigmoidgru_4/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/Sigmoid_1?
gru_4/gru_cell_4/mulMulgru_4/gru_cell_4/Sigmoid_1:y:0!gru_4/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/mul?
gru_4/gru_cell_4/add_2AddV2gru_4/gru_cell_4/split:output:2gru_4/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add_2?
gru_4/gru_cell_4/TanhTanhgru_4/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/Tanh?
gru_4/gru_cell_4/mul_1Mulgru_4/gru_cell_4/Sigmoid:y:0gru_4/zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/mul_1u
gru_4/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_4/gru_cell_4/sub/x?
gru_4/gru_cell_4/subSubgru_4/gru_cell_4/sub/x:output:0gru_4/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/sub?
gru_4/gru_cell_4/mul_2Mulgru_4/gru_cell_4/sub:z:0gru_4/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/mul_2?
gru_4/gru_cell_4/add_3AddV2gru_4/gru_cell_4/mul_1:z:0gru_4/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add_3?
#gru_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2%
#gru_4/TensorArrayV2_1/element_shape?
gru_4/TensorArrayV2_1TensorListReserve,gru_4/TensorArrayV2_1/element_shape:output:0gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_4/TensorArrayV2_1Z

gru_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru_4/time?
gru_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
gru_4/while/maximum_iterationsv
gru_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_4/while/loop_counter?
gru_4/whileWhile!gru_4/while/loop_counter:output:0'gru_4/while/maximum_iterations:output:0gru_4/time:output:0gru_4/TensorArrayV2_1:handle:0gru_4/zeros:output:0gru_4/strided_slice_1:output:0=gru_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_4_gru_cell_4_readvariableop_resource/gru_4_gru_cell_4_matmul_readvariableop_resource1gru_4_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*#
bodyR
gru_4_while_body_340890*#
condR
gru_4_while_cond_340889*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
gru_4/while?
6gru_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   28
6gru_4/TensorArrayV2Stack/TensorListStack/element_shape?
(gru_4/TensorArrayV2Stack/TensorListStackTensorListStackgru_4/while:output:3?gru_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype02*
(gru_4/TensorArrayV2Stack/TensorListStack?
gru_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_4/strided_slice_3/stack?
gru_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice_3/stack_1?
gru_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_3/stack_2?
gru_4/strided_slice_3StridedSlice1gru_4/TensorArrayV2Stack/TensorListStack:tensor:0$gru_4/strided_slice_3/stack:output:0&gru_4/strided_slice_3/stack_1:output:0&gru_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
gru_4/strided_slice_3?
gru_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_4/transpose_1/perm?
gru_4/transpose_1	Transpose1gru_4/TensorArrayV2Stack/TensorListStack:tensor:0gru_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2
gru_4/transpose_1r
gru_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_4/runtimew
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_4/dropout/Const?
dropout_4/dropout/MulMulgru_4/strided_slice_3:output:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout_4/dropout/Mul?
dropout_4/dropout/ShapeShapegru_4/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout_4/dropout/Mul_1?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Relu|
IdentityIdentitydense_4/Relu:activations:0^gru_4/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::2
gru_4/whilegru_4/while:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
??
?
while_body_341760
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_4_readvariableop_resource_05
1while_gru_cell_4_matmul_readvariableop_resource_07
3while_gru_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_4_readvariableop_resource3
/while_gru_cell_4_matmul_readvariableop_resource5
1while_gru_cell_4_matmul_1_readvariableop_resource??
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
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_4/ReadVariableOp?
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_4/unstack?
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02(
&while/gru_cell_4/MatMul/ReadVariableOp?
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/MatMul?
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/BiasAddr
while/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_4/Const?
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell_4/split/split_dim?
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_4/split?
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02*
(while/gru_cell_4/MatMul_1/ReadVariableOp?
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/MatMul_1?
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/BiasAdd_1?
while/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
while/gru_cell_4/Const_1?
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"while/gru_cell_4/split_1/split_dim?
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0!while/gru_cell_4/Const_1:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_4/split_1?
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add?
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Sigmoid?
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_1?
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Sigmoid_1?
while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul?
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_2?
while/gru_cell_4/TanhTanhwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Tanh?
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul_1u
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_4/sub/x?
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/sub?
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0while/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul_2?
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
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
while/Identity_3~
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????d2
while/Identity_4"h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
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
?H
?
gru_4_while_body_340693(
$gru_4_while_gru_4_while_loop_counter.
*gru_4_while_gru_4_while_maximum_iterations
gru_4_while_placeholder
gru_4_while_placeholder_1
gru_4_while_placeholder_2'
#gru_4_while_gru_4_strided_slice_1_0c
_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_04
0gru_4_while_gru_cell_4_readvariableop_resource_0;
7gru_4_while_gru_cell_4_matmul_readvariableop_resource_0=
9gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0
gru_4_while_identity
gru_4_while_identity_1
gru_4_while_identity_2
gru_4_while_identity_3
gru_4_while_identity_4%
!gru_4_while_gru_4_strided_slice_1a
]gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor2
.gru_4_while_gru_cell_4_readvariableop_resource9
5gru_4_while_gru_cell_4_matmul_readvariableop_resource;
7gru_4_while_gru_cell_4_matmul_1_readvariableop_resource??
=gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape?
/gru_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0gru_4_while_placeholderFgru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype021
/gru_4/while/TensorArrayV2Read/TensorListGetItem?
%gru_4/while/gru_cell_4/ReadVariableOpReadVariableOp0gru_4_while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	?*
dtype02'
%gru_4/while/gru_cell_4/ReadVariableOp?
gru_4/while/gru_cell_4/unstackUnpack-gru_4/while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2 
gru_4/while/gru_cell_4/unstack?
,gru_4/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp7gru_4_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02.
,gru_4/while/gru_cell_4/MatMul/ReadVariableOp?
gru_4/while/gru_cell_4/MatMulMatMul6gru_4/while/TensorArrayV2Read/TensorListGetItem:item:04gru_4/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_4/while/gru_cell_4/MatMul?
gru_4/while/gru_cell_4/BiasAddBiasAdd'gru_4/while/gru_cell_4/MatMul:product:0'gru_4/while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2 
gru_4/while/gru_cell_4/BiasAdd~
gru_4/while/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/while/gru_cell_4/Const?
&gru_4/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&gru_4/while/gru_cell_4/split/split_dim?
gru_4/while/gru_cell_4/splitSplit/gru_4/while/gru_cell_4/split/split_dim:output:0'gru_4/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_4/while/gru_cell_4/split?
.gru_4/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp9gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype020
.gru_4/while/gru_cell_4/MatMul_1/ReadVariableOp?
gru_4/while/gru_cell_4/MatMul_1MatMulgru_4_while_placeholder_26gru_4/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_4/while/gru_cell_4/MatMul_1?
 gru_4/while/gru_cell_4/BiasAdd_1BiasAdd)gru_4/while/gru_cell_4/MatMul_1:product:0'gru_4/while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2"
 gru_4/while/gru_cell_4/BiasAdd_1?
gru_4/while/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2 
gru_4/while/gru_cell_4/Const_1?
(gru_4/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_4/while/gru_cell_4/split_1/split_dim?
gru_4/while/gru_cell_4/split_1SplitV)gru_4/while/gru_cell_4/BiasAdd_1:output:0'gru_4/while/gru_cell_4/Const_1:output:01gru_4/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2 
gru_4/while/gru_cell_4/split_1?
gru_4/while/gru_cell_4/addAddV2%gru_4/while/gru_cell_4/split:output:0'gru_4/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add?
gru_4/while/gru_cell_4/SigmoidSigmoidgru_4/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2 
gru_4/while/gru_cell_4/Sigmoid?
gru_4/while/gru_cell_4/add_1AddV2%gru_4/while/gru_cell_4/split:output:1'gru_4/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add_1?
 gru_4/while/gru_cell_4/Sigmoid_1Sigmoid gru_4/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2"
 gru_4/while/gru_cell_4/Sigmoid_1?
gru_4/while/gru_cell_4/mulMul$gru_4/while/gru_cell_4/Sigmoid_1:y:0'gru_4/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/mul?
gru_4/while/gru_cell_4/add_2AddV2%gru_4/while/gru_cell_4/split:output:2gru_4/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add_2?
gru_4/while/gru_cell_4/TanhTanh gru_4/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/Tanh?
gru_4/while/gru_cell_4/mul_1Mul"gru_4/while/gru_cell_4/Sigmoid:y:0gru_4_while_placeholder_2*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/mul_1?
gru_4/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_4/while/gru_cell_4/sub/x?
gru_4/while/gru_cell_4/subSub%gru_4/while/gru_cell_4/sub/x:output:0"gru_4/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/sub?
gru_4/while/gru_cell_4/mul_2Mulgru_4/while/gru_cell_4/sub:z:0gru_4/while/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/mul_2?
gru_4/while/gru_cell_4/add_3AddV2 gru_4/while/gru_cell_4/mul_1:z:0 gru_4/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add_3?
0gru_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_4_while_placeholder_1gru_4_while_placeholder gru_4/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype022
0gru_4/while/TensorArrayV2Write/TensorListSetItemh
gru_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/while/add/y?
gru_4/while/addAddV2gru_4_while_placeholdergru_4/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_4/while/addl
gru_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/while/add_1/y?
gru_4/while/add_1AddV2$gru_4_while_gru_4_while_loop_countergru_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_4/while/add_1p
gru_4/while/IdentityIdentitygru_4/while/add_1:z:0*
T0*
_output_shapes
: 2
gru_4/while/Identity?
gru_4/while/Identity_1Identity*gru_4_while_gru_4_while_maximum_iterations*
T0*
_output_shapes
: 2
gru_4/while/Identity_1r
gru_4/while/Identity_2Identitygru_4/while/add:z:0*
T0*
_output_shapes
: 2
gru_4/while/Identity_2?
gru_4/while/Identity_3Identity@gru_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru_4/while/Identity_3?
gru_4/while/Identity_4Identity gru_4/while/gru_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/Identity_4"H
!gru_4_while_gru_4_strided_slice_1#gru_4_while_gru_4_strided_slice_1_0"t
7gru_4_while_gru_cell_4_matmul_1_readvariableop_resource9gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0"p
5gru_4_while_gru_cell_4_matmul_readvariableop_resource7gru_4_while_gru_cell_4_matmul_readvariableop_resource_0"b
.gru_4_while_gru_cell_4_readvariableop_resource0gru_4_while_gru_cell_4_readvariableop_resource_0"5
gru_4_while_identitygru_4/while/Identity:output:0"9
gru_4_while_identity_1gru_4/while/Identity_1:output:0"9
gru_4_while_identity_2gru_4/while/Identity_2:output:0"9
gru_4_while_identity_3gru_4/while/Identity_3:output:0"9
gru_4_while_identity_4gru_4/while/Identity_4:output:0"?
]gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0*>
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
?W
?
A__inference_gru_4_layer_call_and_return_conditional_losses_341351

inputs&
"gru_cell_4_readvariableop_resource-
)gru_cell_4_matmul_readvariableop_resource/
+gru_cell_4_matmul_1_readvariableop_resource
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
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_4/ReadVariableOp?
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_4/unstack?
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 gru_cell_4/MatMul/ReadVariableOp?
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/MatMul?
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/BiasAddf
gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_4/Const?
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_4/split/split_dim?
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_4/split?
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02$
"gru_cell_4/MatMul_1/ReadVariableOp?
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/MatMul_1?
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_4/BiasAdd_1}
gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_cell_4/Const_1?
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_4/split_1/split_dim?
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const_1:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_4/split_1?
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/addy
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Sigmoid?
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_1
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Sigmoid_1?
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul?
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_2r
gru_cell_4/TanhTanhgru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Tanh?
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul_1i
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_4/sub/x?
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/sub?
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul_2?
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
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
while_body_341261*
condR
while_cond_341260*8
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
??
?
while_body_341420
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_4_readvariableop_resource_05
1while_gru_cell_4_matmul_readvariableop_resource_07
3while_gru_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_4_readvariableop_resource3
/while_gru_cell_4_matmul_readvariableop_resource5
1while_gru_cell_4_matmul_1_readvariableop_resource??
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
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_4/ReadVariableOp?
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_4/unstack?
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02(
&while/gru_cell_4/MatMul/ReadVariableOp?
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/MatMul?
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/BiasAddr
while/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_4/Const?
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell_4/split/split_dim?
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_4/split?
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02*
(while/gru_cell_4/MatMul_1/ReadVariableOp?
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/MatMul_1?
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/BiasAdd_1?
while/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
while/gru_cell_4/Const_1?
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"while/gru_cell_4/split_1/split_dim?
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0!while/gru_cell_4/Const_1:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_4/split_1?
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add?
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Sigmoid?
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_1?
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Sigmoid_1?
while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul?
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_2?
while/gru_cell_4/TanhTanhwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Tanh?
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul_1u
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_4/sub/x?
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/sub?
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0while/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul_2?
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
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
while/Identity_3~
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????d2
while/Identity_4"h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
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
?W
?
A__inference_gru_4_layer_call_and_return_conditional_losses_341510

inputs&
"gru_cell_4_readvariableop_resource-
)gru_cell_4_matmul_readvariableop_resource/
+gru_cell_4_matmul_1_readvariableop_resource
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
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_4/ReadVariableOp?
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_4/unstack?
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 gru_cell_4/MatMul/ReadVariableOp?
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/MatMul?
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/BiasAddf
gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_4/Const?
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_4/split/split_dim?
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_4/split?
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02$
"gru_cell_4/MatMul_1/ReadVariableOp?
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/MatMul_1?
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_4/BiasAdd_1}
gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_cell_4/Const_1?
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_4/split_1/split_dim?
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const_1:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_4/split_1?
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/addy
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Sigmoid?
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_1
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Sigmoid_1?
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul?
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_2r
gru_cell_4/TanhTanhgru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Tanh?
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul_1i
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_4/sub/x?
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/sub?
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul_2?
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
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
while_body_341420*
condR
while_cond_341419*8
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
??
?
while_body_340165
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_4_readvariableop_resource_05
1while_gru_cell_4_matmul_readvariableop_resource_07
3while_gru_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_4_readvariableop_resource3
/while_gru_cell_4_matmul_readvariableop_resource5
1while_gru_cell_4_matmul_1_readvariableop_resource??
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
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_4/ReadVariableOp?
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_4/unstack?
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02(
&while/gru_cell_4/MatMul/ReadVariableOp?
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/MatMul?
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/BiasAddr
while/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_4/Const?
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell_4/split/split_dim?
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_4/split?
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02*
(while/gru_cell_4/MatMul_1/ReadVariableOp?
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/MatMul_1?
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_4/BiasAdd_1?
while/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
while/gru_cell_4/Const_1?
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"while/gru_cell_4/split_1/split_dim?
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0!while/gru_cell_4/Const_1:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
while/gru_cell_4/split_1?
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add?
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Sigmoid?
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_1?
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Sigmoid_1?
while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul?
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_2?
while/gru_cell_4/TanhTanhwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/Tanh?
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul_1u
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_4/sub/x?
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/sub?
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0while/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/mul_2?
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
while/gru_cell_4/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
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
while/Identity_3~
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????d2
while/Identity_4"h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
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
?W
?
A__inference_gru_4_layer_call_and_return_conditional_losses_341850
inputs_0&
"gru_cell_4_readvariableop_resource-
)gru_cell_4_matmul_readvariableop_resource/
+gru_cell_4_matmul_1_readvariableop_resource
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
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_4/ReadVariableOp?
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_4/unstack?
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 gru_cell_4/MatMul/ReadVariableOp?
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/MatMul?
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/BiasAddf
gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_4/Const?
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_4/split/split_dim?
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_4/split?
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02$
"gru_cell_4/MatMul_1/ReadVariableOp?
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/MatMul_1?
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_4/BiasAdd_1}
gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_cell_4/Const_1?
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_4/split_1/split_dim?
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const_1:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_4/split_1?
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/addy
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Sigmoid?
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_1
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Sigmoid_1?
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul?
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_2r
gru_cell_4/TanhTanhgru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Tanh?
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul_1i
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_4/sub/x?
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/sub?
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul_2?
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
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
while_body_341760*
condR
while_cond_341759*8
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
?
?
-__inference_sequential_4_layer_call_fn_340806
gru_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_3403802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????x
%
_user_specified_namegru_4_input
?W
?
A__inference_gru_4_layer_call_and_return_conditional_losses_341691
inputs_0&
"gru_cell_4_readvariableop_resource-
)gru_cell_4_matmul_readvariableop_resource/
+gru_cell_4_matmul_1_readvariableop_resource
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
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_4/ReadVariableOp?
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_4/unstack?
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 gru_cell_4/MatMul/ReadVariableOp?
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/MatMul?
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/BiasAddf
gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_4/Const?
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_4/split/split_dim?
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_4/split?
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02$
"gru_cell_4/MatMul_1/ReadVariableOp?
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/MatMul_1?
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_4/BiasAdd_1}
gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_cell_4/Const_1?
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_4/split_1/split_dim?
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const_1:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_4/split_1?
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/addy
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Sigmoid?
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_1
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Sigmoid_1?
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul?
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_2r
gru_cell_4/TanhTanhgru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Tanh?
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul_1i
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_4/sub/x?
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/sub?
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul_2?
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
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
while_body_341601*
condR
while_cond_341600*8
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
?
c
*__inference_dropout_4_layer_call_fn_341894

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
GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_3402972
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
?
?
while_cond_339742
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_339742___redundant_placeholder04
0while_while_cond_339742___redundant_placeholder14
0while_while_cond_339742___redundant_placeholder24
0while_while_cond_339742___redundant_placeholder3
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
?
?
C__inference_dense_4_layer_call_and_return_conditional_losses_340326

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
?
?
&__inference_gru_4_layer_call_fn_341521

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
GPU 2J 8? *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_3400962
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
?
?
&__inference_gru_4_layer_call_fn_341532

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
GPU 2J 8? *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_3402552
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
?W
?
A__inference_gru_4_layer_call_and_return_conditional_losses_340096

inputs&
"gru_cell_4_readvariableop_resource-
)gru_cell_4_matmul_readvariableop_resource/
+gru_cell_4_matmul_1_readvariableop_resource
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
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_4/ReadVariableOp?
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_4/unstack?
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 gru_cell_4/MatMul/ReadVariableOp?
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/MatMul?
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/BiasAddf
gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_4/Const?
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_4/split/split_dim?
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_4/split?
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02$
"gru_cell_4/MatMul_1/ReadVariableOp?
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_4/MatMul_1?
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_4/BiasAdd_1}
gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_cell_4/Const_1?
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_4/split_1/split_dim?
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const_1:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_cell_4/split_1?
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/addy
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Sigmoid?
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_1
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Sigmoid_1?
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul?
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_2r
gru_cell_4/TanhTanhgru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/Tanh?
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul_1i
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_4/sub/x?
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/sub?
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/mul_2?
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_cell_4/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
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
while_body_340006*
condR
while_cond_340005*8
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
?H
?
gru_4_while_body_340519(
$gru_4_while_gru_4_while_loop_counter.
*gru_4_while_gru_4_while_maximum_iterations
gru_4_while_placeholder
gru_4_while_placeholder_1
gru_4_while_placeholder_2'
#gru_4_while_gru_4_strided_slice_1_0c
_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_04
0gru_4_while_gru_cell_4_readvariableop_resource_0;
7gru_4_while_gru_cell_4_matmul_readvariableop_resource_0=
9gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0
gru_4_while_identity
gru_4_while_identity_1
gru_4_while_identity_2
gru_4_while_identity_3
gru_4_while_identity_4%
!gru_4_while_gru_4_strided_slice_1a
]gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor2
.gru_4_while_gru_cell_4_readvariableop_resource9
5gru_4_while_gru_cell_4_matmul_readvariableop_resource;
7gru_4_while_gru_cell_4_matmul_1_readvariableop_resource??
=gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape?
/gru_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0gru_4_while_placeholderFgru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype021
/gru_4/while/TensorArrayV2Read/TensorListGetItem?
%gru_4/while/gru_cell_4/ReadVariableOpReadVariableOp0gru_4_while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	?*
dtype02'
%gru_4/while/gru_cell_4/ReadVariableOp?
gru_4/while/gru_cell_4/unstackUnpack-gru_4/while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2 
gru_4/while/gru_cell_4/unstack?
,gru_4/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp7gru_4_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02.
,gru_4/while/gru_cell_4/MatMul/ReadVariableOp?
gru_4/while/gru_cell_4/MatMulMatMul6gru_4/while/TensorArrayV2Read/TensorListGetItem:item:04gru_4/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_4/while/gru_cell_4/MatMul?
gru_4/while/gru_cell_4/BiasAddBiasAdd'gru_4/while/gru_cell_4/MatMul:product:0'gru_4/while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2 
gru_4/while/gru_cell_4/BiasAdd~
gru_4/while/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/while/gru_cell_4/Const?
&gru_4/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&gru_4/while/gru_cell_4/split/split_dim?
gru_4/while/gru_cell_4/splitSplit/gru_4/while/gru_cell_4/split/split_dim:output:0'gru_4/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_4/while/gru_cell_4/split?
.gru_4/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp9gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype020
.gru_4/while/gru_cell_4/MatMul_1/ReadVariableOp?
gru_4/while/gru_cell_4/MatMul_1MatMulgru_4_while_placeholder_26gru_4/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_4/while/gru_cell_4/MatMul_1?
 gru_4/while/gru_cell_4/BiasAdd_1BiasAdd)gru_4/while/gru_cell_4/MatMul_1:product:0'gru_4/while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2"
 gru_4/while/gru_cell_4/BiasAdd_1?
gru_4/while/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2 
gru_4/while/gru_cell_4/Const_1?
(gru_4/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_4/while/gru_cell_4/split_1/split_dim?
gru_4/while/gru_cell_4/split_1SplitV)gru_4/while/gru_cell_4/BiasAdd_1:output:0'gru_4/while/gru_cell_4/Const_1:output:01gru_4/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2 
gru_4/while/gru_cell_4/split_1?
gru_4/while/gru_cell_4/addAddV2%gru_4/while/gru_cell_4/split:output:0'gru_4/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add?
gru_4/while/gru_cell_4/SigmoidSigmoidgru_4/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2 
gru_4/while/gru_cell_4/Sigmoid?
gru_4/while/gru_cell_4/add_1AddV2%gru_4/while/gru_cell_4/split:output:1'gru_4/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add_1?
 gru_4/while/gru_cell_4/Sigmoid_1Sigmoid gru_4/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2"
 gru_4/while/gru_cell_4/Sigmoid_1?
gru_4/while/gru_cell_4/mulMul$gru_4/while/gru_cell_4/Sigmoid_1:y:0'gru_4/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/mul?
gru_4/while/gru_cell_4/add_2AddV2%gru_4/while/gru_cell_4/split:output:2gru_4/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add_2?
gru_4/while/gru_cell_4/TanhTanh gru_4/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/Tanh?
gru_4/while/gru_cell_4/mul_1Mul"gru_4/while/gru_cell_4/Sigmoid:y:0gru_4_while_placeholder_2*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/mul_1?
gru_4/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_4/while/gru_cell_4/sub/x?
gru_4/while/gru_cell_4/subSub%gru_4/while/gru_cell_4/sub/x:output:0"gru_4/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/sub?
gru_4/while/gru_cell_4/mul_2Mulgru_4/while/gru_cell_4/sub:z:0gru_4/while/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/mul_2?
gru_4/while/gru_cell_4/add_3AddV2 gru_4/while/gru_cell_4/mul_1:z:0 gru_4/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/gru_cell_4/add_3?
0gru_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_4_while_placeholder_1gru_4_while_placeholder gru_4/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype022
0gru_4/while/TensorArrayV2Write/TensorListSetItemh
gru_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/while/add/y?
gru_4/while/addAddV2gru_4_while_placeholdergru_4/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_4/while/addl
gru_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/while/add_1/y?
gru_4/while/add_1AddV2$gru_4_while_gru_4_while_loop_countergru_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_4/while/add_1p
gru_4/while/IdentityIdentitygru_4/while/add_1:z:0*
T0*
_output_shapes
: 2
gru_4/while/Identity?
gru_4/while/Identity_1Identity*gru_4_while_gru_4_while_maximum_iterations*
T0*
_output_shapes
: 2
gru_4/while/Identity_1r
gru_4/while/Identity_2Identitygru_4/while/add:z:0*
T0*
_output_shapes
: 2
gru_4/while/Identity_2?
gru_4/while/Identity_3Identity@gru_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru_4/while/Identity_3?
gru_4/while/Identity_4Identity gru_4/while/gru_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/while/Identity_4"H
!gru_4_while_gru_4_strided_slice_1#gru_4_while_gru_4_strided_slice_1_0"t
7gru_4_while_gru_cell_4_matmul_1_readvariableop_resource9gru_4_while_gru_cell_4_matmul_1_readvariableop_resource_0"p
5gru_4_while_gru_cell_4_matmul_readvariableop_resource7gru_4_while_gru_cell_4_matmul_readvariableop_resource_0"b
.gru_4_while_gru_cell_4_readvariableop_resource0gru_4_while_gru_cell_4_readvariableop_resource_0"5
gru_4_while_identitygru_4/while/Identity:output:0"9
gru_4_while_identity_1gru_4/while/Identity_1:output:0"9
gru_4_while_identity_2gru_4/while/Identity_2:output:0"9
gru_4_while_identity_3gru_4/while/Identity_3:output:0"9
gru_4_while_identity_4gru_4/while/Identity_4:output:0"?
]gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0*>
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
?j
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_340791
gru_4_input,
(gru_4_gru_cell_4_readvariableop_resource3
/gru_4_gru_cell_4_matmul_readvariableop_resource5
1gru_4_gru_cell_4_matmul_1_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??gru_4/whileU
gru_4/ShapeShapegru_4_input*
T0*
_output_shapes
:2
gru_4/Shape?
gru_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice/stack?
gru_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice/stack_1?
gru_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice/stack_2?
gru_4/strided_sliceStridedSlicegru_4/Shape:output:0"gru_4/strided_slice/stack:output:0$gru_4/strided_slice/stack_1:output:0$gru_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_4/strided_sliceh
gru_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
gru_4/zeros/mul/y?
gru_4/zeros/mulMulgru_4/strided_slice:output:0gru_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_4/zeros/mulk
gru_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_4/zeros/Less/y
gru_4/zeros/LessLessgru_4/zeros/mul:z:0gru_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_4/zeros/Lessn
gru_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
gru_4/zeros/packed/1?
gru_4/zeros/packedPackgru_4/strided_slice:output:0gru_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_4/zeros/packedk
gru_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_4/zeros/Const?
gru_4/zerosFillgru_4/zeros/packed:output:0gru_4/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/zeros?
gru_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_4/transpose/perm?
gru_4/transpose	Transposegru_4_inputgru_4/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2
gru_4/transposea
gru_4/Shape_1Shapegru_4/transpose:y:0*
T0*
_output_shapes
:2
gru_4/Shape_1?
gru_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice_1/stack?
gru_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_1/stack_1?
gru_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_1/stack_2?
gru_4/strided_slice_1StridedSlicegru_4/Shape_1:output:0$gru_4/strided_slice_1/stack:output:0&gru_4/strided_slice_1/stack_1:output:0&gru_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_4/strided_slice_1?
!gru_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru_4/TensorArrayV2/element_shape?
gru_4/TensorArrayV2TensorListReserve*gru_4/TensorArrayV2/element_shape:output:0gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_4/TensorArrayV2?
;gru_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2=
;gru_4/TensorArrayUnstack/TensorListFromTensor/element_shape?
-gru_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_4/transpose:y:0Dgru_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_4/TensorArrayUnstack/TensorListFromTensor?
gru_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice_2/stack?
gru_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_2/stack_1?
gru_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_2/stack_2?
gru_4/strided_slice_2StridedSlicegru_4/transpose:y:0$gru_4/strided_slice_2/stack:output:0&gru_4/strided_slice_2/stack_1:output:0&gru_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_4/strided_slice_2?
gru_4/gru_cell_4/ReadVariableOpReadVariableOp(gru_4_gru_cell_4_readvariableop_resource*
_output_shapes
:	?*
dtype02!
gru_4/gru_cell_4/ReadVariableOp?
gru_4/gru_cell_4/unstackUnpack'gru_4/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_4/gru_cell_4/unstack?
&gru_4/gru_cell_4/MatMul/ReadVariableOpReadVariableOp/gru_4_gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&gru_4/gru_cell_4/MatMul/ReadVariableOp?
gru_4/gru_cell_4/MatMulMatMulgru_4/strided_slice_2:output:0.gru_4/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/MatMul?
gru_4/gru_cell_4/BiasAddBiasAdd!gru_4/gru_cell_4/MatMul:product:0!gru_4/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/BiasAddr
gru_4/gru_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_4/gru_cell_4/Const?
 gru_4/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 gru_4/gru_cell_4/split/split_dim?
gru_4/gru_cell_4/splitSplit)gru_4/gru_cell_4/split/split_dim:output:0!gru_4/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_4/gru_cell_4/split?
(gru_4/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp1gru_4_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02*
(gru_4/gru_cell_4/MatMul_1/ReadVariableOp?
gru_4/gru_cell_4/MatMul_1MatMulgru_4/zeros:output:00gru_4/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/MatMul_1?
gru_4/gru_cell_4/BiasAdd_1BiasAdd#gru_4/gru_cell_4/MatMul_1:product:0!gru_4/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_4/gru_cell_4/BiasAdd_1?
gru_4/gru_cell_4/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ????2
gru_4/gru_cell_4/Const_1?
"gru_4/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_4/gru_cell_4/split_1/split_dim?
gru_4/gru_cell_4/split_1SplitV#gru_4/gru_cell_4/BiasAdd_1:output:0!gru_4/gru_cell_4/Const_1:output:0+gru_4/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????d:?????????d:?????????d*
	num_split2
gru_4/gru_cell_4/split_1?
gru_4/gru_cell_4/addAddV2gru_4/gru_cell_4/split:output:0!gru_4/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add?
gru_4/gru_cell_4/SigmoidSigmoidgru_4/gru_cell_4/add:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/Sigmoid?
gru_4/gru_cell_4/add_1AddV2gru_4/gru_cell_4/split:output:1!gru_4/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add_1?
gru_4/gru_cell_4/Sigmoid_1Sigmoidgru_4/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/Sigmoid_1?
gru_4/gru_cell_4/mulMulgru_4/gru_cell_4/Sigmoid_1:y:0!gru_4/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/mul?
gru_4/gru_cell_4/add_2AddV2gru_4/gru_cell_4/split:output:2gru_4/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add_2?
gru_4/gru_cell_4/TanhTanhgru_4/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/Tanh?
gru_4/gru_cell_4/mul_1Mulgru_4/gru_cell_4/Sigmoid:y:0gru_4/zeros:output:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/mul_1u
gru_4/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_4/gru_cell_4/sub/x?
gru_4/gru_cell_4/subSubgru_4/gru_cell_4/sub/x:output:0gru_4/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/sub?
gru_4/gru_cell_4/mul_2Mulgru_4/gru_cell_4/sub:z:0gru_4/gru_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/mul_2?
gru_4/gru_cell_4/add_3AddV2gru_4/gru_cell_4/mul_1:z:0gru_4/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
gru_4/gru_cell_4/add_3?
#gru_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2%
#gru_4/TensorArrayV2_1/element_shape?
gru_4/TensorArrayV2_1TensorListReserve,gru_4/TensorArrayV2_1/element_shape:output:0gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_4/TensorArrayV2_1Z

gru_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru_4/time?
gru_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
gru_4/while/maximum_iterationsv
gru_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_4/while/loop_counter?
gru_4/whileWhile!gru_4/while/loop_counter:output:0'gru_4/while/maximum_iterations:output:0gru_4/time:output:0gru_4/TensorArrayV2_1:handle:0gru_4/zeros:output:0gru_4/strided_slice_1:output:0=gru_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_4_gru_cell_4_readvariableop_resource/gru_4_gru_cell_4_matmul_readvariableop_resource1gru_4_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*#
bodyR
gru_4_while_body_340693*#
condR
gru_4_while_cond_340692*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations 2
gru_4/while?
6gru_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   28
6gru_4/TensorArrayV2Stack/TensorListStack/element_shape?
(gru_4/TensorArrayV2Stack/TensorListStackTensorListStackgru_4/while:output:3?gru_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????d*
element_dtype02*
(gru_4/TensorArrayV2Stack/TensorListStack?
gru_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_4/strided_slice_3/stack?
gru_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_4/strided_slice_3/stack_1?
gru_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_4/strided_slice_3/stack_2?
gru_4/strided_slice_3StridedSlice1gru_4/TensorArrayV2Stack/TensorListStack:tensor:0$gru_4/strided_slice_3/stack:output:0&gru_4/strided_slice_3/stack_1:output:0&gru_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
gru_4/strided_slice_3?
gru_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_4/transpose_1/perm?
gru_4/transpose_1	Transpose1gru_4/TensorArrayV2Stack/TensorListStack:tensor:0gru_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????xd2
gru_4/transpose_1r
gru_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_4/runtime?
dropout_4/IdentityIdentitygru_4/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d2
dropout_4/Identity?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldropout_4/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Relu|
IdentityIdentitydense_4/Relu:activations:0^gru_4/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????x:::::2
gru_4/whilegru_4/while:X T
+
_output_shapes
:?????????x
%
_user_specified_namegru_4_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
gru_4_input8
serving_default_gru_4_input:0?????????x;
dense_40
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ȩ
?#
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
O__call__
P_default_save_signature
*Q&call_and_return_all_conditional_losses"? 
_tf_keras_sequential? {"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_4_input"}}, {"class_name": "GRU", "config": {"name": "gru_4", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 120, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_4_input"}}, {"class_name": "GRU", "config": {"name": "gru_4", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?

cell
_inbound_nodes

state_spec
_outbound_nodes
trainable_variables
regularization_losses
	variables
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"?

_tf_keras_rnn_layer?	{"class_name": "GRU", "name": "gru_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_4", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [16, 120, 1]}}
?
_inbound_nodes
_outbound_nodes
trainable_variables
	variables
regularization_losses
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
?
_inbound_nodes

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [16, 100]}}
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
(non_trainable_variables
trainable_variables
)layer_regularization_losses

*layers
+metrics
	variables
regularization_losses
O__call__
P_default_save_signature
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
,
Xserving_default"
signature_map
?

$kernel
%recurrent_kernel
&bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GRUCell", "name": "gru_cell_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_4", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
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
 "
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
?
0layer_metrics
1non_trainable_variables
trainable_variables
2layer_regularization_losses

3layers
regularization_losses
4metrics
	variables

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
7non_trainable_variables
trainable_variables
8layer_regularization_losses

9layers
:metrics
	variables
regularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :d2dense_4/kernel
:2dense_4/bias
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
<non_trainable_variables
trainable_variables
=layer_regularization_losses

>layers
?metrics
	variables
regularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
*:(	?2gru_4/gru_cell_4/kernel
4:2	d?2!gru_4/gru_cell_4/recurrent_kernel
(:&	?2gru_4/gru_cell_4/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
@0"
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
Bnon_trainable_variables
,trainable_variables
Clayer_regularization_losses

Dlayers
Emetrics
-	variables
.regularization_losses
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
'

0"
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
*:(d2RMSprop/dense_4/kernel/rms
$:"2RMSprop/dense_4/bias/rms
4:2	?2#RMSprop/gru_4/gru_cell_4/kernel/rms
>:<	d?2-RMSprop/gru_4/gru_cell_4/recurrent_kernel/rms
2:0	?2!RMSprop/gru_4/gru_cell_4/bias/rms
?2?
-__inference_sequential_4_layer_call_fn_340806
-__inference_sequential_4_layer_call_fn_341177
-__inference_sequential_4_layer_call_fn_340821
-__inference_sequential_4_layer_call_fn_341192?
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
!__inference__wrapped_model_339372?
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
annotations? *.?+
)?&
gru_4_input?????????x
?2?
H__inference_sequential_4_layer_call_and_return_conditional_losses_340995
H__inference_sequential_4_layer_call_and_return_conditional_losses_340791
H__inference_sequential_4_layer_call_and_return_conditional_losses_340624
H__inference_sequential_4_layer_call_and_return_conditional_losses_341162?
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
?2?
&__inference_gru_4_layer_call_fn_341521
&__inference_gru_4_layer_call_fn_341861
&__inference_gru_4_layer_call_fn_341872
&__inference_gru_4_layer_call_fn_341532?
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
A__inference_gru_4_layer_call_and_return_conditional_losses_341691
A__inference_gru_4_layer_call_and_return_conditional_losses_341351
A__inference_gru_4_layer_call_and_return_conditional_losses_341850
A__inference_gru_4_layer_call_and_return_conditional_losses_341510?
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
*__inference_dropout_4_layer_call_fn_341899
*__inference_dropout_4_layer_call_fn_341894?
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
E__inference_dropout_4_layer_call_and_return_conditional_losses_341889
E__inference_dropout_4_layer_call_and_return_conditional_losses_341884?
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
(__inference_dense_4_layer_call_fn_341919?
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
C__inference_dense_4_layer_call_and_return_conditional_losses_341910?
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
7B5
$__inference_signature_wrapper_340450gru_4_input
?2?
+__inference_gru_cell_4_layer_call_fn_342013
+__inference_gru_cell_4_layer_call_fn_342027?
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
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_341999
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_341959?
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
!__inference__wrapped_model_339372t&$%8?5
.?+
)?&
gru_4_input?????????x
? "1?.
,
dense_4!?
dense_4??????????
C__inference_dense_4_layer_call_and_return_conditional_losses_341910\/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? {
(__inference_dense_4_layer_call_fn_341919O/?,
%?"
 ?
inputs?????????d
? "???????????
E__inference_dropout_4_layer_call_and_return_conditional_losses_341884\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
E__inference_dropout_4_layer_call_and_return_conditional_losses_341889\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? }
*__inference_dropout_4_layer_call_fn_341894O3?0
)?&
 ?
inputs?????????d
p
? "??????????d}
*__inference_dropout_4_layer_call_fn_341899O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
A__inference_gru_4_layer_call_and_return_conditional_losses_341351m&$%??<
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
A__inference_gru_4_layer_call_and_return_conditional_losses_341510m&$%??<
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
A__inference_gru_4_layer_call_and_return_conditional_losses_341691}&$%O?L
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
A__inference_gru_4_layer_call_and_return_conditional_losses_341850}&$%O?L
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
&__inference_gru_4_layer_call_fn_341521`&$%??<
5?2
$?!
inputs?????????x

 
p

 
? "??????????d?
&__inference_gru_4_layer_call_fn_341532`&$%??<
5?2
$?!
inputs?????????x

 
p 

 
? "??????????d?
&__inference_gru_4_layer_call_fn_341861p&$%O?L
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
&__inference_gru_4_layer_call_fn_341872p&$%O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "??????????d?
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_341959?&$%\?Y
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
F__inference_gru_cell_4_layer_call_and_return_conditional_losses_341999?&$%\?Y
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
+__inference_gru_cell_4_layer_call_fn_342013?&$%\?Y
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
+__inference_gru_cell_4_layer_call_fn_342027?&$%\?Y
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_340624p&$%@?=
6?3
)?&
gru_4_input?????????x
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_340791p&$%@?=
6?3
)?&
gru_4_input?????????x
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_340995k&$%;?8
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_341162k&$%;?8
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
-__inference_sequential_4_layer_call_fn_340806c&$%@?=
6?3
)?&
gru_4_input?????????x
p

 
? "???????????
-__inference_sequential_4_layer_call_fn_340821c&$%@?=
6?3
)?&
gru_4_input?????????x
p 

 
? "???????????
-__inference_sequential_4_layer_call_fn_341177^&$%;?8
1?.
$?!
inputs?????????x
p

 
? "???????????
-__inference_sequential_4_layer_call_fn_341192^&$%;?8
1?.
$?!
inputs?????????x
p 

 
? "???????????
$__inference_signature_wrapper_340450?&$%G?D
? 
=?:
8
gru_4_input)?&
gru_4_input?????????x"1?.
,
dense_4!?
dense_4?????????