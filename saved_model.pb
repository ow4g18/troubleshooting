ЄЕ<
Ё*ч)
A
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
]
Complex	
real"T	
imag"T
out"Tout"
Ttype0:
2"
Touttype0:
2
P

ComplexAbs
x"T	
y"Tout"
Ttype0:
2"
Touttype0:
2
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Cos
x"T
y"T"
Ttype:

2
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
:
FloorMod
x"T
y"T
z"T"
Ttype:
	2	
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

	MirrorPad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	"&
modestring:
REFLECT	SYMMETRIC
=
Mul
x"T
y"T
z"T"
Ttype:
2	
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
w
PadV2

input"T
paddings"	Tpaddings
constant_values"T
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
s
RFFT
input"Treal

fft_length
output"Tcomplex"
Trealtype0:
2"
Tcomplextype0:
2
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
S
Real

input"T
output"Tout"
Ttype0:
2"
Touttype0:
2
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
О
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.0-dev202008252v1.12.1-40136-g5b3cd9ce718Хс5

res_net/fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@*#
shared_nameres_net/fc1/kernel
{
&res_net/fc1/kernel/Read/ReadVariableOpReadVariableOpres_net/fc1/kernel* 
_output_shapes
:
@*
dtype0
y
res_net/fc1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameres_net/fc1/bias
r
$res_net/fc1/bias/Read/ReadVariableOpReadVariableOpres_net/fc1/bias*
_output_shapes	
:*
dtype0

res_net/fc2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameres_net/fc2/kernel
{
&res_net/fc2/kernel/Read/ReadVariableOpReadVariableOpres_net/fc2/kernel* 
_output_shapes
:
*
dtype0
y
res_net/fc2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameres_net/fc2/bias
r
$res_net/fc2/bias/Read/ReadVariableOpReadVariableOpres_net/fc2/bias*
_output_shapes	
:*
dtype0

res_net/fc3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameres_net/fc3/kernel
z
&res_net/fc3/kernel/Read/ReadVariableOpReadVariableOpres_net/fc3/kernel*
_output_shapes
:	*
dtype0
x
res_net/fc3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameres_net/fc3/bias
q
$res_net/fc3/bias/Read/ReadVariableOpReadVariableOpres_net/fc3/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
Ђ
!res_net/resnet_block/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!res_net/resnet_block/conv1/kernel

5res_net/resnet_block/conv1/kernel/Read/ReadVariableOpReadVariableOp!res_net/resnet_block/conv1/kernel*"
_output_shapes
: *
dtype0

res_net/resnet_block/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!res_net/resnet_block/conv1/bias

3res_net/resnet_block/conv1/bias/Read/ReadVariableOpReadVariableOpres_net/resnet_block/conv1/bias*
_output_shapes
: *
dtype0
Ђ
!res_net/resnet_block/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *2
shared_name#!res_net/resnet_block/conv2/kernel

5res_net/resnet_block/conv2/kernel/Read/ReadVariableOpReadVariableOp!res_net/resnet_block/conv2/kernel*"
_output_shapes
:  *
dtype0

res_net/resnet_block/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!res_net/resnet_block/conv2/bias

3res_net/resnet_block/conv2/bias/Read/ReadVariableOpReadVariableOpres_net/resnet_block/conv2/bias*
_output_shapes
: *
dtype0
Ђ
!res_net/resnet_block/conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *2
shared_name#!res_net/resnet_block/conv3/kernel

5res_net/resnet_block/conv3/kernel/Read/ReadVariableOpReadVariableOp!res_net/resnet_block/conv3/kernel*"
_output_shapes
:  *
dtype0

res_net/resnet_block/conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!res_net/resnet_block/conv3/bias

3res_net/resnet_block/conv3/bias/Read/ReadVariableOpReadVariableOpres_net/resnet_block/conv3/bias*
_output_shapes
: *
dtype0
Ј
$res_net/resnet_block/shortcut/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$res_net/resnet_block/shortcut/kernel
Ё
8res_net/resnet_block/shortcut/kernel/Read/ReadVariableOpReadVariableOp$res_net/resnet_block/shortcut/kernel*"
_output_shapes
: *
dtype0

"res_net/resnet_block/shortcut/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"res_net/resnet_block/shortcut/bias

6res_net/resnet_block/shortcut/bias/Read/ReadVariableOpReadVariableOp"res_net/resnet_block/shortcut/bias*
_output_shapes
: *
dtype0
І
#res_net/resnet_block_1/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*4
shared_name%#res_net/resnet_block_1/conv1/kernel

7res_net/resnet_block_1/conv1/kernel/Read/ReadVariableOpReadVariableOp#res_net/resnet_block_1/conv1/kernel*"
_output_shapes
: @*
dtype0

!res_net/resnet_block_1/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!res_net/resnet_block_1/conv1/bias

5res_net/resnet_block_1/conv1/bias/Read/ReadVariableOpReadVariableOp!res_net/resnet_block_1/conv1/bias*
_output_shapes
:@*
dtype0
І
#res_net/resnet_block_1/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#res_net/resnet_block_1/conv2/kernel

7res_net/resnet_block_1/conv2/kernel/Read/ReadVariableOpReadVariableOp#res_net/resnet_block_1/conv2/kernel*"
_output_shapes
:@@*
dtype0

!res_net/resnet_block_1/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!res_net/resnet_block_1/conv2/bias

5res_net/resnet_block_1/conv2/bias/Read/ReadVariableOpReadVariableOp!res_net/resnet_block_1/conv2/bias*
_output_shapes
:@*
dtype0
І
#res_net/resnet_block_1/conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#res_net/resnet_block_1/conv3/kernel

7res_net/resnet_block_1/conv3/kernel/Read/ReadVariableOpReadVariableOp#res_net/resnet_block_1/conv3/kernel*"
_output_shapes
:@@*
dtype0

!res_net/resnet_block_1/conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!res_net/resnet_block_1/conv3/bias

5res_net/resnet_block_1/conv3/bias/Read/ReadVariableOpReadVariableOp!res_net/resnet_block_1/conv3/bias*
_output_shapes
:@*
dtype0
Ќ
&res_net/resnet_block_1/shortcut/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*7
shared_name(&res_net/resnet_block_1/shortcut/kernel
Ѕ
:res_net/resnet_block_1/shortcut/kernel/Read/ReadVariableOpReadVariableOp&res_net/resnet_block_1/shortcut/kernel*"
_output_shapes
: @*
dtype0
 
$res_net/resnet_block_1/shortcut/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$res_net/resnet_block_1/shortcut/bias

8res_net/resnet_block_1/shortcut/bias/Read/ReadVariableOpReadVariableOp$res_net/resnet_block_1/shortcut/bias*
_output_shapes
:@*
dtype0
Ї
#res_net/resnet_block_2/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#res_net/resnet_block_2/conv1/kernel
 
7res_net/resnet_block_2/conv1/kernel/Read/ReadVariableOpReadVariableOp#res_net/resnet_block_2/conv1/kernel*#
_output_shapes
:@*
dtype0

!res_net/resnet_block_2/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!res_net/resnet_block_2/conv1/bias

5res_net/resnet_block_2/conv1/bias/Read/ReadVariableOpReadVariableOp!res_net/resnet_block_2/conv1/bias*
_output_shapes	
:*
dtype0
Ј
#res_net/resnet_block_2/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#res_net/resnet_block_2/conv2/kernel
Ё
7res_net/resnet_block_2/conv2/kernel/Read/ReadVariableOpReadVariableOp#res_net/resnet_block_2/conv2/kernel*$
_output_shapes
:*
dtype0

!res_net/resnet_block_2/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!res_net/resnet_block_2/conv2/bias

5res_net/resnet_block_2/conv2/bias/Read/ReadVariableOpReadVariableOp!res_net/resnet_block_2/conv2/bias*
_output_shapes	
:*
dtype0
Ј
#res_net/resnet_block_2/conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#res_net/resnet_block_2/conv3/kernel
Ё
7res_net/resnet_block_2/conv3/kernel/Read/ReadVariableOpReadVariableOp#res_net/resnet_block_2/conv3/kernel*$
_output_shapes
:*
dtype0

!res_net/resnet_block_2/conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!res_net/resnet_block_2/conv3/bias

5res_net/resnet_block_2/conv3/bias/Read/ReadVariableOpReadVariableOp!res_net/resnet_block_2/conv3/bias*
_output_shapes	
:*
dtype0
­
&res_net/resnet_block_2/shortcut/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&res_net/resnet_block_2/shortcut/kernel
І
:res_net/resnet_block_2/shortcut/kernel/Read/ReadVariableOpReadVariableOp&res_net/resnet_block_2/shortcut/kernel*#
_output_shapes
:@*
dtype0
Ё
$res_net/resnet_block_2/shortcut/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$res_net/resnet_block_2/shortcut/bias

8res_net/resnet_block_2/shortcut/bias/Read/ReadVariableOpReadVariableOp$res_net/resnet_block_2/shortcut/bias*
_output_shapes	
:*
dtype0
Ј
#res_net/resnet_block_3/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#res_net/resnet_block_3/conv1/kernel
Ё
7res_net/resnet_block_3/conv1/kernel/Read/ReadVariableOpReadVariableOp#res_net/resnet_block_3/conv1/kernel*$
_output_shapes
:*
dtype0

!res_net/resnet_block_3/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!res_net/resnet_block_3/conv1/bias

5res_net/resnet_block_3/conv1/bias/Read/ReadVariableOpReadVariableOp!res_net/resnet_block_3/conv1/bias*
_output_shapes	
:*
dtype0
Ј
#res_net/resnet_block_3/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#res_net/resnet_block_3/conv2/kernel
Ё
7res_net/resnet_block_3/conv2/kernel/Read/ReadVariableOpReadVariableOp#res_net/resnet_block_3/conv2/kernel*$
_output_shapes
:*
dtype0

!res_net/resnet_block_3/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!res_net/resnet_block_3/conv2/bias

5res_net/resnet_block_3/conv2/bias/Read/ReadVariableOpReadVariableOp!res_net/resnet_block_3/conv2/bias*
_output_shapes	
:*
dtype0
Ј
#res_net/resnet_block_3/conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#res_net/resnet_block_3/conv3/kernel
Ё
7res_net/resnet_block_3/conv3/kernel/Read/ReadVariableOpReadVariableOp#res_net/resnet_block_3/conv3/kernel*$
_output_shapes
:*
dtype0

!res_net/resnet_block_3/conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!res_net/resnet_block_3/conv3/bias

5res_net/resnet_block_3/conv3/bias/Read/ReadVariableOpReadVariableOp!res_net/resnet_block_3/conv3/bias*
_output_shapes	
:*
dtype0
Ў
&res_net/resnet_block_3/shortcut/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&res_net/resnet_block_3/shortcut/kernel
Ї
:res_net/resnet_block_3/shortcut/kernel/Read/ReadVariableOpReadVariableOp&res_net/resnet_block_3/shortcut/kernel*$
_output_shapes
:*
dtype0
Ё
$res_net/resnet_block_3/shortcut/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$res_net/resnet_block_3/shortcut/bias

8res_net/resnet_block_3/shortcut/bias/Read/ReadVariableOpReadVariableOp$res_net/resnet_block_3/shortcut/bias*
_output_shapes	
:*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
n
accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator
g
accumulator/Read/ReadVariableOpReadVariableOpaccumulator*
_output_shapes
:*
dtype0
r
accumulator_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_1
k
!accumulator_1/Read/ReadVariableOpReadVariableOpaccumulator_1*
_output_shapes
:*
dtype0
r
accumulator_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_2
k
!accumulator_2/Read/ReadVariableOpReadVariableOpaccumulator_2*
_output_shapes
:*
dtype0
r
accumulator_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_3
k
!accumulator_3/Read/ReadVariableOpReadVariableOpaccumulator_3*
_output_shapes
:*
dtype0
№(
ConstConst*
_output_shapes
:	*
dtype0*Б(
valueЇ(BЄ(	"(                    Цїд=                ыP>                Q8>                KШ>                Ьѕ>                BЈ?                Р%?                7#:?                nнM?                Nљ`?                |s?                Jz?РvЏ<            њi?3З=            :X?З>            HG?рЪa>            ёw7?>            vЬ'?gА>            ?ієЮ>            9	?Уь>            ѕ#і>ю?            !Йй>o#?             єН> !?            8ЭЂ>d.?            ё=>с;?            &\>7рH?            g)>цU?            -xя=њb?            .П=Hn?            ­ЦЗ<ЫAz?                џy?"Р<            жyn?S1=            Ъ*c?БЉц=            ~X?	О>            )M?о[K>            ЁrB?{5v>            ы7?Ь(>            W-?QлЄ>            Pe#?a5Й>            c?є9Э>            .?Єыр>            dй?8Mє>            иј>А?            0жх>ш?            Vг>ПT?            С>6q?            _)Џ>Pk(?            лw>D1?            5>fќ9?            ўЊu>AB?            яСS>K?            "P2>їkS?            R>{Ћ[?            с=ЮЮc?            JЁ=Џжk?            иТC=вУs?            6#<ц{?                dЏ|?ё&T<            zu?`/=            Њm?Гв=            if?К\Я=            $К^?p>            QvW?Л&">            bHP?yо>>            и/I?Ђ@[>            6,B?)Ow>            =;?ђ>            Ъa4?l<>            -?гЫЄ>            xх&?5В>            C ?тxП>            тГ?;Ь>             6?Пй>            хЩ?6lц>            Рn?"ѓ>            u$ ?Зџ>            .еѓ>i?            Љч>+??            ПMл> Y?            Ь8Я>c?            <BУ>т^?            QiЗ>WK$?            Е­Ћ>&)*?            К >Ѓј/?            Э>К5?            m$>Ъm;?            @А{>№A?            лLe>ЩЌF?            sO>Ѓ8L?            [!9>ЉЗQ?            W#>*W?            .П>4\?            Ў№=.ъa?            4>Ц=:8g?            +=zl?            ~ъd=XБq?            2=имv?            Y<4§{?                _э~?KP;            Птy?ЈУ<            Ћтt?Jе1=            	эo?З=            Љk?ЖђЇ=            X f?B§Ю=            ђHa?oИѕ=            V{\?Њ>            SЗW?Ж"!>            ЦќR?ц4>            KN?ХбF>            ЃI?юqY>            E?ўэk>            en@?lF~>            
с;?ы=>            \\7?IG>            )р2?Ў?>            \l.?G'Ѓ>            и *?PўЋ>            l%?(ХД>            B!?з{Н>            ю?Ц"Ц>            §Ђ?КЮ>            	_?эAз>            Б"?Кп>            бэ?_$ш>            OР?b№>            ?ЫЫј>            2іў>ч ?            PЦі>и?            \Єю>в­?            Cц>оЗ?            Со>Л?            Хж>З?            цЄЮ>­?            6ЦЦ>х?            oєО>Щ ?            o/З>Hh$?            wЏ>D(?            ЫЇ>},?            X+ >Tъ/?            Ф>Д3?            '>эw7?            _>б5;?            H$>мэ>?            u> B?            rЭf>ЄLF?            А1X>ѓI?            =ЌI>ёM?            В<;>д0Q?            у,>=ЧT?            >=XX?            џo> ф[?            -V>uj_?            Ђш=Ныb?            ЅРЬ=ыgf?            Б=џоi?            vw=Qm?            Йt=4Оp?            ==`&t?            !d=Оw?            іЂ<Pшz?            4эо;&B~?                Зh~?                -{?                LЬw?                ыt?                Bq?                Ѕn?                Щj?                ъg?                bd?                d5a?                ^?                ЛчZ?                (ЧW?                ЃЊT?                -Q?                Ћ}N?                .mK?                `H?                пWE?                SB?                њQ??                КT<?                1[9?                Xe6?                .s3?                Ё0?                Њ-?                IВ*?                kЮ'?                ю$?                "?                Є7?                a?                Х?                [П?                Iѓ?                v*?                йd?                Ђ?                Jу?                F'?                Vn?                И ?                ќ>                +Ќі>                еRё>                sџы>                Вц>                ujс>                И(м>                Кьж>                zЖб>                щЬ>                [Ч>                ­5Т>                пН>                ћЗ>                ВцВ>                Aз­>                'ЭЈ>                dШЃ>                жШ>                }Ю>                iй>                hщ>                ў>                Н>                ђ7>                ,Иx>                u
o>                Xfe>                <Ь[>                ;R>                jДH>                ј6?>                ИТ5>                яW,>                {і">                8>                сN>                п>                ћ=                A/щ=                >иж=                XФ=                Т_В=                y= =                ,=                Yx=                ЦzT=                О0=                8#=                яRг<                kЁ<                o_<                    

NoOpNoOp
ѕ
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*­
valueЂB B

	n_filters
	n_kernels
n_fc
mel
	delta

block1

block2

block3

	block4

flatten
fc1
fc2
fc3
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
 
 
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
Е
	n_kernels
	conv1
	relu1
	conv2
	relu2
	 conv3
!shortcut
"	out_block
#regularization_losses
$	variables
%trainable_variables
&	keras_api
Е
	n_kernels
	'conv1
	(relu1
	)conv2
	*relu2
	+conv3
,shortcut
-	out_block
.regularization_losses
/	variables
0trainable_variables
1	keras_api
Е
	n_kernels
	2conv1
	3relu1
	4conv2
	5relu2
	6conv3
7shortcut
8	out_block
9regularization_losses
:	variables
;trainable_variables
<	keras_api
Е
	n_kernels
	=conv1
	>relu1
	?conv2
	@relu2
	Aconv3
Bshortcut
C	out_block
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
R
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
h

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
h

Rkernel
Sbias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
h

Xkernel
Ybias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
6
^iter
	_decay
`learning_rate
amomentum
 
Ј
b0
c1
d2
e3
f4
g5
h6
i7
j8
k9
l10
m11
n12
o13
p14
q15
r16
s17
t18
u19
v20
w21
x22
y23
z24
{25
|26
}27
~28
29
30
31
L32
M33
R34
S35
X36
Y37
Ј
b0
c1
d2
e3
f4
g5
h6
i7
j8
k9
l10
m11
n12
o13
p14
q15
r16
s17
t18
u19
v20
w21
x22
y23
z24
{25
|26
}27
~28
29
30
31
L32
M33
R34
S35
X36
Y37
В
regularization_losses
layers
layer_metrics
non_trainable_variables
	variables
metrics
trainable_variables
 layer_regularization_losses
 
 
 
 
В
layer_metrics
layers
regularization_losses
non_trainable_variables
	variables
metrics
trainable_variables
 layer_regularization_losses
 
 
 
В
layer_metrics
layers
regularization_losses
non_trainable_variables
	variables
metrics
trainable_variables
 layer_regularization_losses
l

bkernel
cbias
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
l

dkernel
ebias
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
 	keras_api
l

fkernel
gbias
Ёregularization_losses
Ђ	variables
Ѓtrainable_variables
Є	keras_api
l

hkernel
ibias
Ѕregularization_losses
І	variables
Їtrainable_variables
Ј	keras_api
V
Љregularization_losses
Њ	variables
Ћtrainable_variables
Ќ	keras_api
 
8
b0
c1
d2
e3
f4
g5
h6
i7
8
b0
c1
d2
e3
f4
g5
h6
i7
В
#regularization_losses
­layers
Ўlayer_metrics
Џnon_trainable_variables
$	variables
Аmetrics
%trainable_variables
 Бlayer_regularization_losses
l

jkernel
kbias
Вregularization_losses
Г	variables
Дtrainable_variables
Е	keras_api
V
Жregularization_losses
З	variables
Иtrainable_variables
Й	keras_api
l

lkernel
mbias
Кregularization_losses
Л	variables
Мtrainable_variables
Н	keras_api
V
Оregularization_losses
П	variables
Рtrainable_variables
С	keras_api
l

nkernel
obias
Тregularization_losses
У	variables
Фtrainable_variables
Х	keras_api
l

pkernel
qbias
Цregularization_losses
Ч	variables
Шtrainable_variables
Щ	keras_api
V
Ъregularization_losses
Ы	variables
Ьtrainable_variables
Э	keras_api
 
8
j0
k1
l2
m3
n4
o5
p6
q7
8
j0
k1
l2
m3
n4
o5
p6
q7
В
.regularization_losses
Юlayers
Яlayer_metrics
аnon_trainable_variables
/	variables
бmetrics
0trainable_variables
 вlayer_regularization_losses
l

rkernel
sbias
гregularization_losses
д	variables
еtrainable_variables
ж	keras_api
V
зregularization_losses
и	variables
йtrainable_variables
к	keras_api
l

tkernel
ubias
лregularization_losses
м	variables
нtrainable_variables
о	keras_api
V
пregularization_losses
р	variables
сtrainable_variables
т	keras_api
l

vkernel
wbias
уregularization_losses
ф	variables
хtrainable_variables
ц	keras_api
l

xkernel
ybias
чregularization_losses
ш	variables
щtrainable_variables
ъ	keras_api
V
ыregularization_losses
ь	variables
эtrainable_variables
ю	keras_api
 
8
r0
s1
t2
u3
v4
w5
x6
y7
8
r0
s1
t2
u3
v4
w5
x6
y7
В
9regularization_losses
яlayers
№layer_metrics
ёnon_trainable_variables
:	variables
ђmetrics
;trainable_variables
 ѓlayer_regularization_losses
l

zkernel
{bias
єregularization_losses
ѕ	variables
іtrainable_variables
ї	keras_api
V
јregularization_losses
љ	variables
њtrainable_variables
ћ	keras_api
l

|kernel
}bias
ќregularization_losses
§	variables
ўtrainable_variables
џ	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
l

~kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
 
:
z0
{1
|2
}3
~4
5
6
7
:
z0
{1
|2
}3
~4
5
6
7
В
Dregularization_losses
layers
layer_metrics
non_trainable_variables
E	variables
metrics
Ftrainable_variables
 layer_regularization_losses
 
 
 
В
layer_metrics
layers
Hregularization_losses
non_trainable_variables
I	variables
metrics
Jtrainable_variables
 layer_regularization_losses
MK
VARIABLE_VALUEres_net/fc1/kernel%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEres_net/fc1/bias#fc1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
В
layer_metrics
layers
Nregularization_losses
non_trainable_variables
O	variables
metrics
Ptrainable_variables
 layer_regularization_losses
MK
VARIABLE_VALUEres_net/fc2/kernel%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEres_net/fc2/bias#fc2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

R0
S1
В
layer_metrics
 layers
Tregularization_losses
Ёnon_trainable_variables
U	variables
Ђmetrics
Vtrainable_variables
 Ѓlayer_regularization_losses
MK
VARIABLE_VALUEres_net/fc3/kernel%fc3/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEres_net/fc3/bias#fc3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

X0
Y1
В
Єlayer_metrics
Ѕlayers
Zregularization_losses
Іnon_trainable_variables
[	variables
Їmetrics
\trainable_variables
 Јlayer_regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!res_net/resnet_block/conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEres_net/resnet_block/conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!res_net/resnet_block/conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEres_net/resnet_block/conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!res_net/resnet_block/conv3/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEres_net/resnet_block/conv3/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$res_net/resnet_block/shortcut/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"res_net/resnet_block/shortcut/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#res_net/resnet_block_1/conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!res_net/resnet_block_1/conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#res_net/resnet_block_1/conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!res_net/resnet_block_1/conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#res_net/resnet_block_1/conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!res_net/resnet_block_1/conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&res_net/resnet_block_1/shortcut/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$res_net/resnet_block_1/shortcut/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#res_net/resnet_block_2/conv1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!res_net/resnet_block_2/conv1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#res_net/resnet_block_2/conv2/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!res_net/resnet_block_2/conv2/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#res_net/resnet_block_2/conv3/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!res_net/resnet_block_2/conv3/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&res_net/resnet_block_2/shortcut/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$res_net/resnet_block_2/shortcut/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#res_net/resnet_block_3/conv1/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!res_net/resnet_block_3/conv1/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#res_net/resnet_block_3/conv2/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!res_net/resnet_block_3/conv2/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#res_net/resnet_block_3/conv3/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!res_net/resnet_block_3/conv3/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&res_net/resnet_block_3/shortcut/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$res_net/resnet_block_3/shortcut/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
F
0
1
2
3
4
	5

6
7
8
9
 
 
@
Љ0
Њ1
Ћ2
Ќ3
­4
Ў5
Џ6
А7
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

b0
c1

b0
c1
Е
Бlayer_metrics
Вlayers
regularization_losses
Гnon_trainable_variables
	variables
Дmetrics
trainable_variables
 Еlayer_regularization_losses
 
 
 
Е
Жlayer_metrics
Зlayers
regularization_losses
Иnon_trainable_variables
	variables
Йmetrics
trainable_variables
 Кlayer_regularization_losses
 

d0
e1

d0
e1
Е
Лlayer_metrics
Мlayers
regularization_losses
Нnon_trainable_variables
	variables
Оmetrics
trainable_variables
 Пlayer_regularization_losses
 
 
 
Е
Рlayer_metrics
Сlayers
regularization_losses
Тnon_trainable_variables
	variables
Уmetrics
trainable_variables
 Фlayer_regularization_losses
 

f0
g1

f0
g1
Е
Хlayer_metrics
Цlayers
Ёregularization_losses
Чnon_trainable_variables
Ђ	variables
Шmetrics
Ѓtrainable_variables
 Щlayer_regularization_losses
 

h0
i1

h0
i1
Е
Ъlayer_metrics
Ыlayers
Ѕregularization_losses
Ьnon_trainable_variables
І	variables
Эmetrics
Їtrainable_variables
 Юlayer_regularization_losses
 
 
 
Е
Яlayer_metrics
аlayers
Љregularization_losses
бnon_trainable_variables
Њ	variables
вmetrics
Ћtrainable_variables
 гlayer_regularization_losses
1
0
1
2
3
 4
!5
"6
 
 
 
 
 

j0
k1

j0
k1
Е
дlayer_metrics
еlayers
Вregularization_losses
жnon_trainable_variables
Г	variables
зmetrics
Дtrainable_variables
 иlayer_regularization_losses
 
 
 
Е
йlayer_metrics
кlayers
Жregularization_losses
лnon_trainable_variables
З	variables
мmetrics
Иtrainable_variables
 нlayer_regularization_losses
 

l0
m1

l0
m1
Е
оlayer_metrics
пlayers
Кregularization_losses
рnon_trainable_variables
Л	variables
сmetrics
Мtrainable_variables
 тlayer_regularization_losses
 
 
 
Е
уlayer_metrics
фlayers
Оregularization_losses
хnon_trainable_variables
П	variables
цmetrics
Рtrainable_variables
 чlayer_regularization_losses
 

n0
o1

n0
o1
Е
шlayer_metrics
щlayers
Тregularization_losses
ъnon_trainable_variables
У	variables
ыmetrics
Фtrainable_variables
 ьlayer_regularization_losses
 

p0
q1

p0
q1
Е
эlayer_metrics
юlayers
Цregularization_losses
яnon_trainable_variables
Ч	variables
№metrics
Шtrainable_variables
 ёlayer_regularization_losses
 
 
 
Е
ђlayer_metrics
ѓlayers
Ъregularization_losses
єnon_trainable_variables
Ы	variables
ѕmetrics
Ьtrainable_variables
 іlayer_regularization_losses
1
'0
(1
)2
*3
+4
,5
-6
 
 
 
 
 

r0
s1

r0
s1
Е
їlayer_metrics
јlayers
гregularization_losses
љnon_trainable_variables
д	variables
њmetrics
еtrainable_variables
 ћlayer_regularization_losses
 
 
 
Е
ќlayer_metrics
§layers
зregularization_losses
ўnon_trainable_variables
и	variables
џmetrics
йtrainable_variables
 layer_regularization_losses
 

t0
u1

t0
u1
Е
layer_metrics
layers
лregularization_losses
non_trainable_variables
м	variables
metrics
нtrainable_variables
 layer_regularization_losses
 
 
 
Е
layer_metrics
layers
пregularization_losses
non_trainable_variables
р	variables
metrics
сtrainable_variables
 layer_regularization_losses
 

v0
w1

v0
w1
Е
layer_metrics
layers
уregularization_losses
non_trainable_variables
ф	variables
metrics
хtrainable_variables
 layer_regularization_losses
 

x0
y1

x0
y1
Е
layer_metrics
layers
чregularization_losses
non_trainable_variables
ш	variables
metrics
щtrainable_variables
 layer_regularization_losses
 
 
 
Е
layer_metrics
layers
ыregularization_losses
non_trainable_variables
ь	variables
metrics
эtrainable_variables
 layer_regularization_losses
1
20
31
42
53
64
75
86
 
 
 
 
 

z0
{1

z0
{1
Е
layer_metrics
layers
єregularization_losses
non_trainable_variables
ѕ	variables
metrics
іtrainable_variables
 layer_regularization_losses
 
 
 
Е
layer_metrics
 layers
јregularization_losses
Ёnon_trainable_variables
љ	variables
Ђmetrics
њtrainable_variables
 Ѓlayer_regularization_losses
 

|0
}1

|0
}1
Е
Єlayer_metrics
Ѕlayers
ќregularization_losses
Іnon_trainable_variables
§	variables
Їmetrics
ўtrainable_variables
 Јlayer_regularization_losses
 
 
 
Е
Љlayer_metrics
Њlayers
regularization_losses
Ћnon_trainable_variables
	variables
Ќmetrics
trainable_variables
 ­layer_regularization_losses
 

~0
1

~0
1
Е
Ўlayer_metrics
Џlayers
regularization_losses
Аnon_trainable_variables
	variables
Бmetrics
trainable_variables
 Вlayer_regularization_losses
 

0
1

0
1
Е
Гlayer_metrics
Дlayers
regularization_losses
Еnon_trainable_variables
	variables
Жmetrics
trainable_variables
 Зlayer_regularization_losses
 
 
 
Е
Иlayer_metrics
Йlayers
regularization_losses
Кnon_trainable_variables
	variables
Лmetrics
trainable_variables
 Мlayer_regularization_losses
1
=0
>1
?2
@3
A4
B5
C6
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
8

Нtotal

Оcount
П	variables
Р	keras_api
I

Сtotal

Тcount
У
_fn_kwargs
Ф	variables
Х	keras_api
\
Ц
thresholds
Чtrue_positives
Шfalse_positives
Щ	variables
Ъ	keras_api
\
Ы
thresholds
Ьtrue_positives
Эfalse_negatives
Ю	variables
Я	keras_api
C
а
thresholds
бaccumulator
в	variables
г	keras_api
C
д
thresholds
еaccumulator
ж	variables
з	keras_api
C
и
thresholds
йaccumulator
к	variables
л	keras_api
C
м
thresholds
нaccumulator
о	variables
п	keras_api
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
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Н0
О1

П	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

С0
Т1

Ф	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE

Ч0
Ш1

Щ	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

Ь0
Э1

Ю	variables
 
[Y
VARIABLE_VALUEaccumulator:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUE

б0

в	variables
 
][
VARIABLE_VALUEaccumulator_1:keras_api/metrics/5/accumulator/.ATTRIBUTES/VARIABLE_VALUE

е0

ж	variables
 
][
VARIABLE_VALUEaccumulator_2:keras_api/metrics/6/accumulator/.ATTRIBUTES/VARIABLE_VALUE

й0

к	variables
 
][
VARIABLE_VALUEaccumulator_3:keras_api/metrics/7/accumulator/.ATTRIBUTES/VARIABLE_VALUE

н0

о	variables

serving_default_input_1Placeholder*,
_output_shapes
:џџџџџџџџџ*
dtype0*!
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Const!res_net/resnet_block/conv1/kernelres_net/resnet_block/conv1/bias!res_net/resnet_block/conv2/kernelres_net/resnet_block/conv2/bias!res_net/resnet_block/conv3/kernelres_net/resnet_block/conv3/bias$res_net/resnet_block/shortcut/kernel"res_net/resnet_block/shortcut/bias#res_net/resnet_block_1/conv1/kernel!res_net/resnet_block_1/conv1/bias#res_net/resnet_block_1/conv2/kernel!res_net/resnet_block_1/conv2/bias#res_net/resnet_block_1/conv3/kernel!res_net/resnet_block_1/conv3/bias&res_net/resnet_block_1/shortcut/kernel$res_net/resnet_block_1/shortcut/bias#res_net/resnet_block_2/conv1/kernel!res_net/resnet_block_2/conv1/bias#res_net/resnet_block_2/conv2/kernel!res_net/resnet_block_2/conv2/bias#res_net/resnet_block_2/conv3/kernel!res_net/resnet_block_2/conv3/bias&res_net/resnet_block_2/shortcut/kernel$res_net/resnet_block_2/shortcut/bias#res_net/resnet_block_3/conv1/kernel!res_net/resnet_block_3/conv1/bias#res_net/resnet_block_3/conv2/kernel!res_net/resnet_block_3/conv2/bias#res_net/resnet_block_3/conv3/kernel!res_net/resnet_block_3/conv3/bias&res_net/resnet_block_3/shortcut/kernel$res_net/resnet_block_3/shortcut/biasres_net/fc1/kernelres_net/fc1/biasres_net/fc2/kernelres_net/fc2/biasres_net/fc3/kernelres_net/fc3/bias*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_213674
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
й
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&res_net/fc1/kernel/Read/ReadVariableOp$res_net/fc1/bias/Read/ReadVariableOp&res_net/fc2/kernel/Read/ReadVariableOp$res_net/fc2/bias/Read/ReadVariableOp&res_net/fc3/kernel/Read/ReadVariableOp$res_net/fc3/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOp5res_net/resnet_block/conv1/kernel/Read/ReadVariableOp3res_net/resnet_block/conv1/bias/Read/ReadVariableOp5res_net/resnet_block/conv2/kernel/Read/ReadVariableOp3res_net/resnet_block/conv2/bias/Read/ReadVariableOp5res_net/resnet_block/conv3/kernel/Read/ReadVariableOp3res_net/resnet_block/conv3/bias/Read/ReadVariableOp8res_net/resnet_block/shortcut/kernel/Read/ReadVariableOp6res_net/resnet_block/shortcut/bias/Read/ReadVariableOp7res_net/resnet_block_1/conv1/kernel/Read/ReadVariableOp5res_net/resnet_block_1/conv1/bias/Read/ReadVariableOp7res_net/resnet_block_1/conv2/kernel/Read/ReadVariableOp5res_net/resnet_block_1/conv2/bias/Read/ReadVariableOp7res_net/resnet_block_1/conv3/kernel/Read/ReadVariableOp5res_net/resnet_block_1/conv3/bias/Read/ReadVariableOp:res_net/resnet_block_1/shortcut/kernel/Read/ReadVariableOp8res_net/resnet_block_1/shortcut/bias/Read/ReadVariableOp7res_net/resnet_block_2/conv1/kernel/Read/ReadVariableOp5res_net/resnet_block_2/conv1/bias/Read/ReadVariableOp7res_net/resnet_block_2/conv2/kernel/Read/ReadVariableOp5res_net/resnet_block_2/conv2/bias/Read/ReadVariableOp7res_net/resnet_block_2/conv3/kernel/Read/ReadVariableOp5res_net/resnet_block_2/conv3/bias/Read/ReadVariableOp:res_net/resnet_block_2/shortcut/kernel/Read/ReadVariableOp8res_net/resnet_block_2/shortcut/bias/Read/ReadVariableOp7res_net/resnet_block_3/conv1/kernel/Read/ReadVariableOp5res_net/resnet_block_3/conv1/bias/Read/ReadVariableOp7res_net/resnet_block_3/conv2/kernel/Read/ReadVariableOp5res_net/resnet_block_3/conv2/bias/Read/ReadVariableOp7res_net/resnet_block_3/conv3/kernel/Read/ReadVariableOp5res_net/resnet_block_3/conv3/bias/Read/ReadVariableOp:res_net/resnet_block_3/shortcut/kernel/Read/ReadVariableOp8res_net/resnet_block_3/shortcut/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOpaccumulator/Read/ReadVariableOp!accumulator_1/Read/ReadVariableOp!accumulator_2/Read/ReadVariableOp!accumulator_3/Read/ReadVariableOpConst_1*C
Tin<
:28	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_217980

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameres_net/fc1/kernelres_net/fc1/biasres_net/fc2/kernelres_net/fc2/biasres_net/fc3/kernelres_net/fc3/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentum!res_net/resnet_block/conv1/kernelres_net/resnet_block/conv1/bias!res_net/resnet_block/conv2/kernelres_net/resnet_block/conv2/bias!res_net/resnet_block/conv3/kernelres_net/resnet_block/conv3/bias$res_net/resnet_block/shortcut/kernel"res_net/resnet_block/shortcut/bias#res_net/resnet_block_1/conv1/kernel!res_net/resnet_block_1/conv1/bias#res_net/resnet_block_1/conv2/kernel!res_net/resnet_block_1/conv2/bias#res_net/resnet_block_1/conv3/kernel!res_net/resnet_block_1/conv3/bias&res_net/resnet_block_1/shortcut/kernel$res_net/resnet_block_1/shortcut/bias#res_net/resnet_block_2/conv1/kernel!res_net/resnet_block_2/conv1/bias#res_net/resnet_block_2/conv2/kernel!res_net/resnet_block_2/conv2/bias#res_net/resnet_block_2/conv3/kernel!res_net/resnet_block_2/conv3/bias&res_net/resnet_block_2/shortcut/kernel$res_net/resnet_block_2/shortcut/bias#res_net/resnet_block_3/conv1/kernel!res_net/resnet_block_3/conv1/bias#res_net/resnet_block_3/conv2/kernel!res_net/resnet_block_3/conv2/bias#res_net/resnet_block_3/conv3/kernel!res_net/resnet_block_3/conv3/bias&res_net/resnet_block_3/shortcut/kernel$res_net/resnet_block_3/shortcut/biastotalcounttotal_1count_1true_positivesfalse_positivestrue_positives_1false_negativesaccumulatoraccumulator_1accumulator_2accumulator_3*B
Tin;
927*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_218152ё2
ь
~
)__inference_shortcut_layer_call_fn_217532

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_shortcut_layer_call_and_return_conditional_losses_2108052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@ ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Ж
Ж
A__inference_conv2_layer_call_and_return_conditional_losses_217339

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@ :::S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Й
Й
D__inference_shortcut_layer_call_and_return_conditional_losses_210805

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@ :::S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Б

C__inference_res_net_layer_call_and_return_conditional_losses_214596
input_1 
log_mel_spectrogram_matmul_bB
>resnet_block_conv1_conv1d_expanddims_1_readvariableop_resource6
2resnet_block_conv1_biasadd_readvariableop_resourceB
>resnet_block_conv2_conv1d_expanddims_1_readvariableop_resource6
2resnet_block_conv2_biasadd_readvariableop_resourceB
>resnet_block_conv3_conv1d_expanddims_1_readvariableop_resource6
2resnet_block_conv3_biasadd_readvariableop_resourceE
Aresnet_block_shortcut_conv1d_expanddims_1_readvariableop_resource9
5resnet_block_shortcut_biasadd_readvariableop_resourceD
@resnet_block_1_conv1_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_1_conv1_biasadd_readvariableop_resourceD
@resnet_block_1_conv2_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_1_conv2_biasadd_readvariableop_resourceD
@resnet_block_1_conv3_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_1_conv3_biasadd_readvariableop_resourceG
Cresnet_block_1_shortcut_conv1d_expanddims_1_readvariableop_resource;
7resnet_block_1_shortcut_biasadd_readvariableop_resourceD
@resnet_block_2_conv1_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_2_conv1_biasadd_readvariableop_resourceD
@resnet_block_2_conv2_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_2_conv2_biasadd_readvariableop_resourceD
@resnet_block_2_conv3_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_2_conv3_biasadd_readvariableop_resourceG
Cresnet_block_2_shortcut_conv1d_expanddims_1_readvariableop_resource;
7resnet_block_2_shortcut_biasadd_readvariableop_resourceD
@resnet_block_3_conv1_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_3_conv1_biasadd_readvariableop_resourceD
@resnet_block_3_conv2_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_3_conv2_biasadd_readvariableop_resourceD
@resnet_block_3_conv3_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_3_conv3_biasadd_readvariableop_resourceG
Cresnet_block_3_shortcut_conv1d_expanddims_1_readvariableop_resource;
7resnet_block_3_shortcut_biasadd_readvariableop_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource&
"fc2_matmul_readvariableop_resource'
#fc2_biasadd_readvariableop_resource&
"fc3_matmul_readvariableop_resource'
#fc3_biasadd_readvariableop_resource
identityp
SqueezeSqueezeinput_1*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2	
Squeeze
%log_mel_spectrogram/stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2'
%log_mel_spectrogram/stft/frame_length
#log_mel_spectrogram/stft/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2%
#log_mel_spectrogram/stft/frame_step
log_mel_spectrogram/stft/ConstConst*
_output_shapes
: *
dtype0*
value
B :2 
log_mel_spectrogram/stft/Const
#log_mel_spectrogram/stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#log_mel_spectrogram/stft/frame/axis
$log_mel_spectrogram/stft/frame/ShapeShapeSqueeze:output:0*
T0*
_output_shapes
:2&
$log_mel_spectrogram/stft/frame/Shape
#log_mel_spectrogram/stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#log_mel_spectrogram/stft/frame/Rank
*log_mel_spectrogram/stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*log_mel_spectrogram/stft/frame/range/start
*log_mel_spectrogram/stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*log_mel_spectrogram/stft/frame/range/delta
$log_mel_spectrogram/stft/frame/rangeRange3log_mel_spectrogram/stft/frame/range/start:output:0,log_mel_spectrogram/stft/frame/Rank:output:03log_mel_spectrogram/stft/frame/range/delta:output:0*
_output_shapes
:2&
$log_mel_spectrogram/stft/frame/rangeЛ
2log_mel_spectrogram/stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ24
2log_mel_spectrogram/stft/frame/strided_slice/stackЖ
4log_mel_spectrogram/stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4log_mel_spectrogram/stft/frame/strided_slice/stack_1Ж
4log_mel_spectrogram/stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4log_mel_spectrogram/stft/frame/strided_slice/stack_2
,log_mel_spectrogram/stft/frame/strided_sliceStridedSlice-log_mel_spectrogram/stft/frame/range:output:0;log_mel_spectrogram/stft/frame/strided_slice/stack:output:0=log_mel_spectrogram/stft/frame/strided_slice/stack_1:output:0=log_mel_spectrogram/stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,log_mel_spectrogram/stft/frame/strided_slice
$log_mel_spectrogram/stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$log_mel_spectrogram/stft/frame/sub/yЭ
"log_mel_spectrogram/stft/frame/subSub,log_mel_spectrogram/stft/frame/Rank:output:0-log_mel_spectrogram/stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/subг
$log_mel_spectrogram/stft/frame/sub_1Sub&log_mel_spectrogram/stft/frame/sub:z:05log_mel_spectrogram/stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/sub_1
'log_mel_spectrogram/stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'log_mel_spectrogram/stft/frame/packed/1
%log_mel_spectrogram/stft/frame/packedPack5log_mel_spectrogram/stft/frame/strided_slice:output:00log_mel_spectrogram/stft/frame/packed/1:output:0(log_mel_spectrogram/stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2'
%log_mel_spectrogram/stft/frame/packedЂ
.log_mel_spectrogram/stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.log_mel_spectrogram/stft/frame/split/split_dimК
$log_mel_spectrogram/stft/frame/splitSplitV-log_mel_spectrogram/stft/frame/Shape:output:0.log_mel_spectrogram/stft/frame/packed:output:07log_mel_spectrogram/stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2&
$log_mel_spectrogram/stft/frame/split
,log_mel_spectrogram/stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2.
,log_mel_spectrogram/stft/frame/Reshape/shapeЃ
.log_mel_spectrogram/stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 20
.log_mel_spectrogram/stft/frame/Reshape/shape_1ф
&log_mel_spectrogram/stft/frame/ReshapeReshape-log_mel_spectrogram/stft/frame/split:output:17log_mel_spectrogram/stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2(
&log_mel_spectrogram/stft/frame/Reshape
#log_mel_spectrogram/stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2%
#log_mel_spectrogram/stft/frame/Size
%log_mel_spectrogram/stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%log_mel_spectrogram/stft/frame/Size_1
$log_mel_spectrogram/stft/frame/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$log_mel_spectrogram/stft/frame/ConstЁ
"log_mel_spectrogram/stft/frame/NegNeg/log_mel_spectrogram/stft/frame/Reshape:output:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/Negе
'log_mel_spectrogram/stft/frame/floordivFloorDiv&log_mel_spectrogram/stft/frame/Neg:y:0,log_mel_spectrogram/stft/frame_step:output:0*
T0*
_output_shapes
: 2)
'log_mel_spectrogram/stft/frame/floordivЁ
$log_mel_spectrogram/stft/frame/Neg_1Neg+log_mel_spectrogram/stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/Neg_1
&log_mel_spectrogram/stft/frame/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_spectrogram/stft/frame/sub_2/yЯ
$log_mel_spectrogram/stft/frame/sub_2Sub(log_mel_spectrogram/stft/frame/Neg_1:y:0/log_mel_spectrogram/stft/frame/sub_2/y:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/sub_2Ш
"log_mel_spectrogram/stft/frame/mulMul,log_mel_spectrogram/stft/frame_step:output:0(log_mel_spectrogram/stft/frame/sub_2:z:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/mulЪ
"log_mel_spectrogram/stft/frame/addAddV2.log_mel_spectrogram/stft/frame_length:output:0&log_mel_spectrogram/stft/frame/mul:z:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/addЭ
$log_mel_spectrogram/stft/frame/sub_3Sub&log_mel_spectrogram/stft/frame/add:z:0/log_mel_spectrogram/stft/frame/Reshape:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/sub_3
(log_mel_spectrogram/stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2*
(log_mel_spectrogram/stft/frame/Maximum/xй
&log_mel_spectrogram/stft/frame/MaximumMaximum1log_mel_spectrogram/stft/frame/Maximum/x:output:0(log_mel_spectrogram/stft/frame/sub_3:z:0*
T0*
_output_shapes
: 2(
&log_mel_spectrogram/stft/frame/Maximum
*log_mel_spectrogram/stft/frame/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*log_mel_spectrogram/stft/frame/zeros/mul/yп
(log_mel_spectrogram/stft/frame/zeros/mulMul,log_mel_spectrogram/stft/frame/Size:output:03log_mel_spectrogram/stft/frame/zeros/mul/y:output:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/frame/zeros/mul
+log_mel_spectrogram/stft/frame/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2-
+log_mel_spectrogram/stft/frame/zeros/Less/yу
)log_mel_spectrogram/stft/frame/zeros/LessLess,log_mel_spectrogram/stft/frame/zeros/mul:z:04log_mel_spectrogram/stft/frame/zeros/Less/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/zeros/Less 
-log_mel_spectrogram/stft/frame/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-log_mel_spectrogram/stft/frame/zeros/packed/1і
+log_mel_spectrogram/stft/frame/zeros/packedPack,log_mel_spectrogram/stft/frame/Size:output:06log_mel_spectrogram/stft/frame/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2-
+log_mel_spectrogram/stft/frame/zeros/packed
*log_mel_spectrogram/stft/frame/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2,
*log_mel_spectrogram/stft/frame/zeros/Constш
$log_mel_spectrogram/stft/frame/zerosFill4log_mel_spectrogram/stft/frame/zeros/packed:output:03log_mel_spectrogram/stft/frame/zeros/Const:output:0*
T0*
_output_shapes

:2&
$log_mel_spectrogram/stft/frame/zeros
+log_mel_spectrogram/stft/frame/packed_1/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2-
+log_mel_spectrogram/stft/frame/packed_1/0/0ю
)log_mel_spectrogram/stft/frame/packed_1/0Pack4log_mel_spectrogram/stft/frame/packed_1/0/0:output:0*log_mel_spectrogram/stft/frame/Maximum:z:0*
N*
T0*
_output_shapes
:2+
)log_mel_spectrogram/stft/frame/packed_1/0Р
'log_mel_spectrogram/stft/frame/packed_1Pack2log_mel_spectrogram/stft/frame/packed_1/0:output:0*
N*
T0*
_output_shapes

:2)
'log_mel_spectrogram/stft/frame/packed_1
,log_mel_spectrogram/stft/frame/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/frame/zeros_1/mul/yч
*log_mel_spectrogram/stft/frame/zeros_1/mulMul.log_mel_spectrogram/stft/frame/Size_1:output:05log_mel_spectrogram/stft/frame/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2,
*log_mel_spectrogram/stft/frame/zeros_1/mulЁ
-log_mel_spectrogram/stft/frame/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2/
-log_mel_spectrogram/stft/frame/zeros_1/Less/yы
+log_mel_spectrogram/stft/frame/zeros_1/LessLess.log_mel_spectrogram/stft/frame/zeros_1/mul:z:06log_mel_spectrogram/stft/frame/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2-
+log_mel_spectrogram/stft/frame/zeros_1/LessЄ
/log_mel_spectrogram/stft/frame/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :21
/log_mel_spectrogram/stft/frame/zeros_1/packed/1ў
-log_mel_spectrogram/stft/frame/zeros_1/packedPack.log_mel_spectrogram/stft/frame/Size_1:output:08log_mel_spectrogram/stft/frame/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2/
-log_mel_spectrogram/stft/frame/zeros_1/packed
,log_mel_spectrogram/stft/frame/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/zeros_1/Constю
&log_mel_spectrogram/stft/frame/zeros_1Fill6log_mel_spectrogram/stft/frame/zeros_1/packed:output:05log_mel_spectrogram/stft/frame/zeros_1/Const:output:0*
T0*
_output_shapes

: 2(
&log_mel_spectrogram/stft/frame/zeros_1
*log_mel_spectrogram/stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*log_mel_spectrogram/stft/frame/concat/axisг
%log_mel_spectrogram/stft/frame/concatConcatV2-log_mel_spectrogram/stft/frame/zeros:output:00log_mel_spectrogram/stft/frame/packed_1:output:0/log_mel_spectrogram/stft/frame/zeros_1:output:03log_mel_spectrogram/stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes

:2'
%log_mel_spectrogram/stft/frame/concat
$log_mel_spectrogram/stft/frame/PadV2PadV2Squeeze:output:0.log_mel_spectrogram/stft/frame/concat:output:0-log_mel_spectrogram/stft/frame/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2&
$log_mel_spectrogram/stft/frame/PadV2­
&log_mel_spectrogram/stft/frame/Shape_1Shape-log_mel_spectrogram/stft/frame/PadV2:output:0*
T0*
_output_shapes
:2(
&log_mel_spectrogram/stft/frame/Shape_1
&log_mel_spectrogram/stft/frame/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_spectrogram/stft/frame/add_1/yо
$log_mel_spectrogram/stft/frame/add_1AddV25log_mel_spectrogram/stft/frame/strided_slice:output:0/log_mel_spectrogram/stft/frame/add_1/y:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/add_1й
4log_mel_spectrogram/stft/frame/strided_slice_1/stackPack5log_mel_spectrogram/stft/frame/strided_slice:output:0*
N*
T0*
_output_shapes
:26
4log_mel_spectrogram/stft/frame/strided_slice_1/stackа
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_1Pack(log_mel_spectrogram/stft/frame/add_1:z:0*
N*
T0*
_output_shapes
:28
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_1К
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_2Ј
.log_mel_spectrogram/stft/frame/strided_slice_1StridedSlice/log_mel_spectrogram/stft/frame/Shape_1:output:0=log_mel_spectrogram/stft/frame/strided_slice_1/stack:output:0?log_mel_spectrogram/stft/frame/strided_slice_1/stack_1:output:0?log_mel_spectrogram/stft/frame/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.log_mel_spectrogram/stft/frame/strided_slice_1
(log_mel_spectrogram/stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(log_mel_spectrogram/stft/frame/gcd/Const
+log_mel_spectrogram/stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_spectrogram/stft/frame/floordiv_1/yщ
)log_mel_spectrogram/stft/frame/floordiv_1FloorDiv.log_mel_spectrogram/stft/frame_length:output:04log_mel_spectrogram/stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/floordiv_1
+log_mel_spectrogram/stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_spectrogram/stft/frame/floordiv_2/yч
)log_mel_spectrogram/stft/frame/floordiv_2FloorDiv,log_mel_spectrogram/stft/frame_step:output:04log_mel_spectrogram/stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/floordiv_2
+log_mel_spectrogram/stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_spectrogram/stft/frame/floordiv_3/yђ
)log_mel_spectrogram/stft/frame/floordiv_3FloorDiv7log_mel_spectrogram/stft/frame/strided_slice_1:output:04log_mel_spectrogram/stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/floordiv_3
&log_mel_spectrogram/stft/frame/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_spectrogram/stft/frame/mul_1/yд
$log_mel_spectrogram/stft/frame/mul_1Mul-log_mel_spectrogram/stft/frame/floordiv_3:z:0/log_mel_spectrogram/stft/frame/mul_1/y:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/mul_1Ф
0log_mel_spectrogram/stft/frame/concat_1/values_1Pack(log_mel_spectrogram/stft/frame/mul_1:z:0*
N*
T0*
_output_shapes
:22
0log_mel_spectrogram/stft/frame/concat_1/values_1
,log_mel_spectrogram/stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/concat_1/axisм
'log_mel_spectrogram/stft/frame/concat_1ConcatV2-log_mel_spectrogram/stft/frame/split:output:09log_mel_spectrogram/stft/frame/concat_1/values_1:output:0-log_mel_spectrogram/stft/frame/split:output:25log_mel_spectrogram/stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'log_mel_spectrogram/stft/frame/concat_1Њ
2log_mel_spectrogram/stft/frame/concat_2/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :24
2log_mel_spectrogram/stft/frame/concat_2/values_1/1
0log_mel_spectrogram/stft/frame/concat_2/values_1Pack-log_mel_spectrogram/stft/frame/floordiv_3:z:0;log_mel_spectrogram/stft/frame/concat_2/values_1/1:output:0*
N*
T0*
_output_shapes
:22
0log_mel_spectrogram/stft/frame/concat_2/values_1
,log_mel_spectrogram/stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/concat_2/axisм
'log_mel_spectrogram/stft/frame/concat_2ConcatV2-log_mel_spectrogram/stft/frame/split:output:09log_mel_spectrogram/stft/frame/concat_2/values_1:output:0-log_mel_spectrogram/stft/frame/split:output:25log_mel_spectrogram/stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2)
'log_mel_spectrogram/stft/frame/concat_2 
)log_mel_spectrogram/stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2+
)log_mel_spectrogram/stft/frame/zeros_likeЊ
.log_mel_spectrogram/stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:20
.log_mel_spectrogram/stft/frame/ones_like/ShapeЂ
.log_mel_spectrogram/stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :20
.log_mel_spectrogram/stft/frame/ones_like/Constѓ
(log_mel_spectrogram/stft/frame/ones_likeFill7log_mel_spectrogram/stft/frame/ones_like/Shape:output:07log_mel_spectrogram/stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2*
(log_mel_spectrogram/stft/frame/ones_likeњ
+log_mel_spectrogram/stft/frame/StridedSliceStridedSlice-log_mel_spectrogram/stft/frame/PadV2:output:02log_mel_spectrogram/stft/frame/zeros_like:output:00log_mel_spectrogram/stft/frame/concat_1:output:01log_mel_spectrogram/stft/frame/ones_like:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2-
+log_mel_spectrogram/stft/frame/StridedSlice
(log_mel_spectrogram/stft/frame/Reshape_1Reshape4log_mel_spectrogram/stft/frame/StridedSlice:output:00log_mel_spectrogram/stft/frame/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
(log_mel_spectrogram/stft/frame/Reshape_1
,log_mel_spectrogram/stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/range_1/start
,log_mel_spectrogram/stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/frame/range_1/delta
&log_mel_spectrogram/stft/frame/range_1Range5log_mel_spectrogram/stft/frame/range_1/start:output:0(log_mel_spectrogram/stft/frame/Neg_1:y:05log_mel_spectrogram/stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2(
&log_mel_spectrogram/stft/frame/range_1с
$log_mel_spectrogram/stft/frame/mul_2Mul/log_mel_spectrogram/stft/frame/range_1:output:0-log_mel_spectrogram/stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2&
$log_mel_spectrogram/stft/frame/mul_2І
0log_mel_spectrogram/stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0log_mel_spectrogram/stft/frame/Reshape_2/shape/1ћ
.log_mel_spectrogram/stft/frame/Reshape_2/shapePack(log_mel_spectrogram/stft/frame/Neg_1:y:09log_mel_spectrogram/stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:20
.log_mel_spectrogram/stft/frame/Reshape_2/shapeє
(log_mel_spectrogram/stft/frame/Reshape_2Reshape(log_mel_spectrogram/stft/frame/mul_2:z:07log_mel_spectrogram/stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(log_mel_spectrogram/stft/frame/Reshape_2
,log_mel_spectrogram/stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/range_2/start
,log_mel_spectrogram/stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/frame/range_2/delta
&log_mel_spectrogram/stft/frame/range_2Range5log_mel_spectrogram/stft/frame/range_2/start:output:0-log_mel_spectrogram/stft/frame/floordiv_1:z:05log_mel_spectrogram/stft/frame/range_2/delta:output:0*
_output_shapes
: 2(
&log_mel_spectrogram/stft/frame/range_2І
0log_mel_spectrogram/stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :22
0log_mel_spectrogram/stft/frame/Reshape_3/shape/0
.log_mel_spectrogram/stft/frame/Reshape_3/shapePack9log_mel_spectrogram/stft/frame/Reshape_3/shape/0:output:0-log_mel_spectrogram/stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:20
.log_mel_spectrogram/stft/frame/Reshape_3/shapeђ
(log_mel_spectrogram/stft/frame/Reshape_3Reshape/log_mel_spectrogram/stft/frame/range_2:output:07log_mel_spectrogram/stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

: 2*
(log_mel_spectrogram/stft/frame/Reshape_3э
$log_mel_spectrogram/stft/frame/add_2AddV21log_mel_spectrogram/stft/frame/Reshape_2:output:01log_mel_spectrogram/stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$log_mel_spectrogram/stft/frame/add_2и
'log_mel_spectrogram/stft/frame/GatherV2GatherV21log_mel_spectrogram/stft/frame/Reshape_1:output:0(log_mel_spectrogram/stft/frame/add_2:z:05log_mel_spectrogram/stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 2)
'log_mel_spectrogram/stft/frame/GatherV2є
0log_mel_spectrogram/stft/frame/concat_3/values_1Pack(log_mel_spectrogram/stft/frame/Neg_1:y:0.log_mel_spectrogram/stft/frame_length:output:0*
N*
T0*
_output_shapes
:22
0log_mel_spectrogram/stft/frame/concat_3/values_1
,log_mel_spectrogram/stft/frame/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/concat_3/axisм
'log_mel_spectrogram/stft/frame/concat_3ConcatV2-log_mel_spectrogram/stft/frame/split:output:09log_mel_spectrogram/stft/frame/concat_3/values_1:output:0-log_mel_spectrogram/stft/frame/split:output:25log_mel_spectrogram/stft/frame/concat_3/axis:output:0*
N*
T0*
_output_shapes
:2)
'log_mel_spectrogram/stft/frame/concat_3њ
(log_mel_spectrogram/stft/frame/Reshape_4Reshape0log_mel_spectrogram/stft/frame/GatherV2:output:00log_mel_spectrogram/stft/frame/concat_3:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2*
(log_mel_spectrogram/stft/frame/Reshape_4 
-log_mel_spectrogram/stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-log_mel_spectrogram/stft/hann_window/periodicЦ
)log_mel_spectrogram/stft/hann_window/CastCast6log_mel_spectrogram/stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2+
)log_mel_spectrogram/stft/hann_window/CastЄ
/log_mel_spectrogram/stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :21
/log_mel_spectrogram/stft/hann_window/FloorMod/yѕ
-log_mel_spectrogram/stft/hann_window/FloorModFloorMod.log_mel_spectrogram/stft/frame_length:output:08log_mel_spectrogram/stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2/
-log_mel_spectrogram/stft/hann_window/FloorMod
*log_mel_spectrogram/stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*log_mel_spectrogram/stft/hann_window/sub/xф
(log_mel_spectrogram/stft/hann_window/subSub3log_mel_spectrogram/stft/hann_window/sub/x:output:01log_mel_spectrogram/stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/hann_window/subй
(log_mel_spectrogram/stft/hann_window/mulMul-log_mel_spectrogram/stft/hann_window/Cast:y:0,log_mel_spectrogram/stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/hann_window/mulм
(log_mel_spectrogram/stft/hann_window/addAddV2.log_mel_spectrogram/stft/frame_length:output:0,log_mel_spectrogram/stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/hann_window/add
,log_mel_spectrogram/stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/hann_window/sub_1/yх
*log_mel_spectrogram/stft/hann_window/sub_1Sub,log_mel_spectrogram/stft/hann_window/add:z:05log_mel_spectrogram/stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2,
*log_mel_spectrogram/stft/hann_window/sub_1Т
+log_mel_spectrogram/stft/hann_window/Cast_1Cast.log_mel_spectrogram/stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+log_mel_spectrogram/stft/hann_window/Cast_1І
0log_mel_spectrogram/stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0log_mel_spectrogram/stft/hann_window/range/startІ
0log_mel_spectrogram/stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0log_mel_spectrogram/stft/hann_window/range/deltaЄ
*log_mel_spectrogram/stft/hann_window/rangeRange9log_mel_spectrogram/stft/hann_window/range/start:output:0.log_mel_spectrogram/stft/frame_length:output:09log_mel_spectrogram/stft/hann_window/range/delta:output:0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/rangeЬ
+log_mel_spectrogram/stft/hann_window/Cast_2Cast3log_mel_spectrogram/stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2-
+log_mel_spectrogram/stft/hann_window/Cast_2
*log_mel_spectrogram/stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2,
*log_mel_spectrogram/stft/hann_window/Constы
*log_mel_spectrogram/stft/hann_window/mul_1Mul3log_mel_spectrogram/stft/hann_window/Const:output:0/log_mel_spectrogram/stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/mul_1ю
,log_mel_spectrogram/stft/hann_window/truedivRealDiv.log_mel_spectrogram/stft/hann_window/mul_1:z:0/log_mel_spectrogram/stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2.
,log_mel_spectrogram/stft/hann_window/truedivГ
(log_mel_spectrogram/stft/hann_window/CosCos0log_mel_spectrogram/stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2*
(log_mel_spectrogram/stft/hann_window/CosЁ
,log_mel_spectrogram/stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,log_mel_spectrogram/stft/hann_window/mul_2/xъ
*log_mel_spectrogram/stft/hann_window/mul_2Mul5log_mel_spectrogram/stft/hann_window/mul_2/x:output:0,log_mel_spectrogram/stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/mul_2Ё
,log_mel_spectrogram/stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,log_mel_spectrogram/stft/hann_window/sub_2/xь
*log_mel_spectrogram/stft/hann_window/sub_2Sub5log_mel_spectrogram/stft/hann_window/sub_2/x:output:0.log_mel_spectrogram/stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/sub_2н
log_mel_spectrogram/stft/mulMul1log_mel_spectrogram/stft/frame/Reshape_4:output:0.log_mel_spectrogram/stft/hann_window/sub_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/stft/mulЋ
$log_mel_spectrogram/stft/rfft/packedPack'log_mel_spectrogram/stft/Const:output:0*
N*
T0*
_output_shapes
:2&
$log_mel_spectrogram/stft/rfft/packed
(log_mel_spectrogram/stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2*
(log_mel_spectrogram/stft/rfft/fft_lengthЩ
log_mel_spectrogram/stft/rfftRFFT log_mel_spectrogram/stft/mul:z:01log_mel_spectrogram/stft/rfft/fft_length:output:0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/stft/rfft
log_mel_spectrogram/Abs
ComplexAbs&log_mel_spectrogram/stft/rfft:output:0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Abs
log_mel_spectrogram/SquareSquarelog_mel_spectrogram/Abs:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/SquareН
log_mel_spectrogram/MatMulBatchMatMulV2log_mel_spectrogram/Square:y:0log_mel_spectrogram_matmul_b*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/MatMul
log_mel_spectrogram/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
log_mel_spectrogram/ConstЃ
log_mel_spectrogram/MaxMax#log_mel_spectrogram/MatMul:output:0"log_mel_spectrogram/Const:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Max
log_mel_spectrogram/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2
log_mel_spectrogram/Maximum/xШ
log_mel_spectrogram/MaximumMaximum&log_mel_spectrogram/Maximum/x:output:0#log_mel_spectrogram/MatMul:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Maximum
log_mel_spectrogram/LogLoglog_mel_spectrogram/Maximum:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Log
log_mel_spectrogram/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/Const_1
log_mel_spectrogram/Log_1Log$log_mel_spectrogram/Const_1:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Log_1З
log_mel_spectrogram/truedivRealDivlog_mel_spectrogram/Log:y:0log_mel_spectrogram/Log_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/truediv{
log_mel_spectrogram/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/mul/xД
log_mel_spectrogram/mulMul"log_mel_spectrogram/mul/x:output:0log_mel_spectrogram/truediv:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/mul
log_mel_spectrogram/Maximum_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2!
log_mel_spectrogram/Maximum_1/xЖ
log_mel_spectrogram/Maximum_1Maximum(log_mel_spectrogram/Maximum_1/x:output:0 log_mel_spectrogram/Max:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Maximum_1
log_mel_spectrogram/Log_2Log!log_mel_spectrogram/Maximum_1:z:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Log_2
log_mel_spectrogram/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/Const_2
log_mel_spectrogram/Log_3Log$log_mel_spectrogram/Const_2:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Log_3Ј
log_mel_spectrogram/truediv_1RealDivlog_mel_spectrogram/Log_2:y:0log_mel_spectrogram/Log_3:y:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/truediv_1
log_mel_spectrogram/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/mul_1/xЇ
log_mel_spectrogram/mul_1Mul$log_mel_spectrogram/mul_1/x:output:0!log_mel_spectrogram/truediv_1:z:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/mul_1Ћ
log_mel_spectrogram/subSublog_mel_spectrogram/mul:z:0log_mel_spectrogram/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/sub
log_mel_spectrogram/Const_3Const*
_output_shapes
:*
dtype0*!
valueB"          2
log_mel_spectrogram/Const_3Ё
log_mel_spectrogram/Max_1Maxlog_mel_spectrogram/sub:z:0$log_mel_spectrogram/Const_3:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Max_1
log_mel_spectrogram/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B2
log_mel_spectrogram/sub_1/yЈ
log_mel_spectrogram/sub_1Sub"log_mel_spectrogram/Max_1:output:0$log_mel_spectrogram/sub_1/y:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/sub_1Л
log_mel_spectrogram/Maximum_2Maximumlog_mel_spectrogram/sub:z:0log_mel_spectrogram/sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Maximum_2
"log_mel_spectrogram/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"log_mel_spectrogram/ExpandDims/dimи
log_mel_spectrogram/ExpandDims
ExpandDims!log_mel_spectrogram/Maximum_2:z:0+log_mel_spectrogram/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2 
log_mel_spectrogram/ExpandDimsy
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm
	transpose	Transpose'log_mel_spectrogram/ExpandDims:output:0transpose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
	transpose
)mfccs_from_log_mel_spectrograms/dct/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)mfccs_from_log_mel_spectrograms/dct/Const
*mfccs_from_log_mel_spectrograms/dct/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*mfccs_from_log_mel_spectrograms/dct/Cast/xС
(mfccs_from_log_mel_spectrograms/dct/CastCast3mfccs_from_log_mel_spectrograms/dct/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(mfccs_from_log_mel_spectrograms/dct/CastЄ
/mfccs_from_log_mel_spectrograms/dct/range/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/mfccs_from_log_mel_spectrograms/dct/range/startЄ
/mfccs_from_log_mel_spectrograms/dct/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/mfccs_from_log_mel_spectrograms/dct/range/deltaв
.mfccs_from_log_mel_spectrograms/dct/range/CastCast8mfccs_from_log_mel_spectrograms/dct/range/start:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.mfccs_from_log_mel_spectrograms/dct/range/Castж
0mfccs_from_log_mel_spectrograms/dct/range/Cast_1Cast8mfccs_from_log_mel_spectrograms/dct/range/delta:output:0*

DstT0*

SrcT0*
_output_shapes
: 22
0mfccs_from_log_mel_spectrograms/dct/range/Cast_1
)mfccs_from_log_mel_spectrograms/dct/rangeRange2mfccs_from_log_mel_spectrograms/dct/range/Cast:y:0,mfccs_from_log_mel_spectrograms/dct/Cast:y:04mfccs_from_log_mel_spectrograms/dct/range/Cast_1:y:0*

Tidx0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/rangeВ
'mfccs_from_log_mel_spectrograms/dct/NegNeg2mfccs_from_log_mel_spectrograms/dct/range:output:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/Neg
)mfccs_from_log_mel_spectrograms/dct/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *лI@2+
)mfccs_from_log_mel_spectrograms/dct/mul/yп
'mfccs_from_log_mel_spectrograms/dct/mulMul+mfccs_from_log_mel_spectrograms/dct/Neg:y:02mfccs_from_log_mel_spectrograms/dct/mul/y:output:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/mul
+mfccs_from_log_mel_spectrograms/dct/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2-
+mfccs_from_log_mel_spectrograms/dct/mul_1/yх
)mfccs_from_log_mel_spectrograms/dct/mul_1Mul+mfccs_from_log_mel_spectrograms/dct/mul:z:04mfccs_from_log_mel_spectrograms/dct/mul_1/y:output:0*
T0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/mul_1ч
+mfccs_from_log_mel_spectrograms/dct/truedivRealDiv-mfccs_from_log_mel_spectrograms/dct/mul_1:z:0,mfccs_from_log_mel_spectrograms/dct/Cast:y:0*
T0*
_output_shapes
:2-
+mfccs_from_log_mel_spectrograms/dct/truedivц
+mfccs_from_log_mel_spectrograms/dct/ComplexComplex2mfccs_from_log_mel_spectrograms/dct/Const:output:0/mfccs_from_log_mel_spectrograms/dct/truediv:z:0*
_output_shapes
:2-
+mfccs_from_log_mel_spectrograms/dct/ComplexБ
'mfccs_from_log_mel_spectrograms/dct/ExpExp1mfccs_from_log_mel_spectrograms/dct/Complex:out:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/ExpЃ
+mfccs_from_log_mel_spectrograms/dct/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J   @    2-
+mfccs_from_log_mel_spectrograms/dct/mul_2/xх
)mfccs_from_log_mel_spectrograms/dct/mul_2Mul4mfccs_from_log_mel_spectrograms/dct/mul_2/x:output:0+mfccs_from_log_mel_spectrograms/dct/Exp:y:0*
T0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/mul_2Њ
.mfccs_from_log_mel_spectrograms/dct/rfft/ConstConst*
_output_shapes
:*
dtype0*
valueB:
20
.mfccs_from_log_mel_spectrograms/dct/rfft/Constп
5mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                27
5mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddingsь
,mfccs_from_log_mel_spectrograms/dct/rfft/PadPadtranspose:y:0>mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
2.
,mfccs_from_log_mel_spectrograms/dct/rfft/PadД
3mfccs_from_log_mel_spectrograms/dct/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:
25
3mfccs_from_log_mel_spectrograms/dct/rfft/fft_length
(mfccs_from_log_mel_spectrograms/dct/rfftRFFT5mfccs_from_log_mel_spectrograms/dct/rfft/Pad:output:0<mfccs_from_log_mel_spectrograms/dct/rfft/fft_length:output:0*/
_output_shapes
:џџџџџџџџџ@2*
(mfccs_from_log_mel_spectrograms/dct/rfftУ
7mfccs_from_log_mel_spectrograms/dct/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7mfccs_from_log_mel_spectrograms/dct/strided_slice/stackЧ
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1Ч
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2с
1mfccs_from_log_mel_spectrograms/dct/strided_sliceStridedSlice1mfccs_from_log_mel_spectrograms/dct/rfft:output:0@mfccs_from_log_mel_spectrograms/dct/strided_slice/stack:output:0Bmfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1:output:0Bmfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

begin_mask*
ellipsis_mask23
1mfccs_from_log_mel_spectrograms/dct/strided_slice
)mfccs_from_log_mel_spectrograms/dct/mul_3Mul:mfccs_from_log_mel_spectrograms/dct/strided_slice:output:0-mfccs_from_log_mel_spectrograms/dct/mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2+
)mfccs_from_log_mel_spectrograms/dct/mul_3М
(mfccs_from_log_mel_spectrograms/dct/RealReal-mfccs_from_log_mel_spectrograms/dct/mul_3:z:0*/
_output_shapes
:џџџџџџџџџ@2*
(mfccs_from_log_mel_spectrograms/dct/Real
&mfccs_from_log_mel_spectrograms/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2(
&mfccs_from_log_mel_spectrograms/Cast/xЕ
$mfccs_from_log_mel_spectrograms/CastCast/mfccs_from_log_mel_spectrograms/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$mfccs_from_log_mel_spectrograms/Cast
%mfccs_from_log_mel_spectrograms/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%mfccs_from_log_mel_spectrograms/mul/yЬ
#mfccs_from_log_mel_spectrograms/mulMul(mfccs_from_log_mel_spectrograms/Cast:y:0.mfccs_from_log_mel_spectrograms/mul/y:output:0*
T0*
_output_shapes
: 2%
#mfccs_from_log_mel_spectrograms/mulЁ
%mfccs_from_log_mel_spectrograms/RsqrtRsqrt'mfccs_from_log_mel_spectrograms/mul:z:0*
T0*
_output_shapes
: 2'
%mfccs_from_log_mel_spectrograms/Rsqrtэ
%mfccs_from_log_mel_spectrograms/mul_1Mul1mfccs_from_log_mel_spectrograms/dct/Real:output:0)mfccs_from_log_mel_spectrograms/Rsqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2'
%mfccs_from_log_mel_spectrograms/mul_1c
SquareSquaretranspose:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
	keep_dims(2
Sum\
SqrtSqrtSum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Sqrt
delta/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose/permГ
delta/transpose	Transpose)mfccs_from_log_mel_spectrograms/mul_1:z:0delta/transpose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose
delta/ConstConst*
_output_shapes

:*
dtype0*9
value0B."                               2
delta/ConstЉ
delta/MirrorPad	MirrorPaddelta/transpose:y:0delta/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
mode	SYMMETRIC2
delta/MirrorPads
delta/arange/startConst*
_output_shapes
: *
dtype0*
valueB :
ќџџџџџџџџ2
delta/arange/startj
delta/arange/limitConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange/limitj
delta/arange/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange/delta
delta/arangeRangedelta/arange/start:output:0delta/arange/limit:output:0delta/arange/delta:output:0*
_output_shapes
:	2
delta/arangek

delta/CastCastdelta/arange:output:0*

DstT0*

SrcT0*
_output_shapes
:	2

delta/Cast
delta/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
delta/Reshape/shape
delta/ReshapeReshapedelta/Cast:y:0delta/Reshape/shape:output:0*
T0*&
_output_shapes
:	2
delta/ReshapeХ
delta/convolutionConv2Ddelta/MirrorPad:output:0delta/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
delta/convolutiong
delta/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pB2
delta/truediv/y
delta/truedivRealDivdelta/convolution:output:0delta/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/truediv
delta/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose_1/permЁ
delta/transpose_1	Transposedelta/truediv:z:0delta/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose_1
delta/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose_2/permЅ
delta/transpose_2	Transposedelta/transpose_1:y:0delta/transpose_2/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose_2
delta/Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                               2
delta/Const_1Б
delta/MirrorPad_1	MirrorPaddelta/transpose_2:y:0delta/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
mode	SYMMETRIC2
delta/MirrorPad_1w
delta/arange_1/startConst*
_output_shapes
: *
dtype0*
valueB :
ќџџџџџџџџ2
delta/arange_1/startn
delta/arange_1/limitConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange_1/limitn
delta/arange_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange_1/deltaЂ
delta/arange_1Rangedelta/arange_1/start:output:0delta/arange_1/limit:output:0delta/arange_1/delta:output:0*
_output_shapes
:	2
delta/arange_1q
delta/Cast_1Castdelta/arange_1:output:0*

DstT0*

SrcT0*
_output_shapes
:	2
delta/Cast_1
delta/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
delta/Reshape_1/shape
delta/Reshape_1Reshapedelta/Cast_1:y:0delta/Reshape_1/shape:output:0*
T0*&
_output_shapes
:	2
delta/Reshape_1Э
delta/convolution_1Conv2Ddelta/MirrorPad_1:output:0delta/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
delta/convolution_1k
delta/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pB2
delta/truediv_1/yЁ
delta/truediv_1RealDivdelta/convolution_1:output:0delta/truediv_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/truediv_1
delta/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose_3/permЃ
delta/transpose_3	Transposedelta/truediv_1:z:0delta/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisи
concatConcatV2)mfccs_from_log_mel_spectrograms/mul_1:z:0delta/transpose_1:y:0delta/transpose_3:y:0Sqrt:y:0concat/axis:output:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@2
concat
	Squeeze_1Squeezeconcat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@*
squeeze_dims
2
	Squeeze_1
(resnet_block/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(resnet_block/conv1/conv1d/ExpandDims/dimл
$resnet_block/conv1/conv1d/ExpandDims
ExpandDimsSqueeze_1:output:01resnet_block/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2&
$resnet_block/conv1/conv1d/ExpandDimsё
5resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>resnet_block_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOp
*resnet_block/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*resnet_block/conv1/conv1d/ExpandDims_1/dim
&resnet_block/conv1/conv1d/ExpandDims_1
ExpandDims=resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:03resnet_block/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2(
&resnet_block/conv1/conv1d/ExpandDims_1
resnet_block/conv1/conv1dConv2D-resnet_block/conv1/conv1d/ExpandDims:output:0/resnet_block/conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/conv1/conv1dЫ
!resnet_block/conv1/conv1d/SqueezeSqueeze"resnet_block/conv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2#
!resnet_block/conv1/conv1d/SqueezeХ
)resnet_block/conv1/BiasAdd/ReadVariableOpReadVariableOp2resnet_block_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)resnet_block/conv1/BiasAdd/ReadVariableOpи
resnet_block/conv1/BiasAddBiasAdd*resnet_block/conv1/conv1d/Squeeze:output:01resnet_block/conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/conv1/BiasAdd
resnet_block/relu1/ReluRelu#resnet_block/conv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/relu1/Relu
(resnet_block/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(resnet_block/conv2/conv1d/ExpandDims/dimю
$resnet_block/conv2/conv1d/ExpandDims
ExpandDims%resnet_block/relu1/Relu:activations:01resnet_block/conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2&
$resnet_block/conv2/conv1d/ExpandDimsё
5resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>resnet_block_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype027
5resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOp
*resnet_block/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*resnet_block/conv2/conv1d/ExpandDims_1/dim
&resnet_block/conv2/conv1d/ExpandDims_1
ExpandDims=resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:03resnet_block/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2(
&resnet_block/conv2/conv1d/ExpandDims_1
resnet_block/conv2/conv1dConv2D-resnet_block/conv2/conv1d/ExpandDims:output:0/resnet_block/conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/conv2/conv1dЫ
!resnet_block/conv2/conv1d/SqueezeSqueeze"resnet_block/conv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2#
!resnet_block/conv2/conv1d/SqueezeХ
)resnet_block/conv2/BiasAdd/ReadVariableOpReadVariableOp2resnet_block_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)resnet_block/conv2/BiasAdd/ReadVariableOpи
resnet_block/conv2/BiasAddBiasAdd*resnet_block/conv2/conv1d/Squeeze:output:01resnet_block/conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/conv2/BiasAdd
resnet_block/relu2/ReluRelu#resnet_block/conv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/relu2/Relu
(resnet_block/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(resnet_block/conv3/conv1d/ExpandDims/dimю
$resnet_block/conv3/conv1d/ExpandDims
ExpandDims%resnet_block/relu2/Relu:activations:01resnet_block/conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2&
$resnet_block/conv3/conv1d/ExpandDimsё
5resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>resnet_block_conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype027
5resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOp
*resnet_block/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*resnet_block/conv3/conv1d/ExpandDims_1/dim
&resnet_block/conv3/conv1d/ExpandDims_1
ExpandDims=resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:03resnet_block/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2(
&resnet_block/conv3/conv1d/ExpandDims_1
resnet_block/conv3/conv1dConv2D-resnet_block/conv3/conv1d/ExpandDims:output:0/resnet_block/conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/conv3/conv1dЫ
!resnet_block/conv3/conv1d/SqueezeSqueeze"resnet_block/conv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2#
!resnet_block/conv3/conv1d/SqueezeХ
)resnet_block/conv3/BiasAdd/ReadVariableOpReadVariableOp2resnet_block_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)resnet_block/conv3/BiasAdd/ReadVariableOpи
resnet_block/conv3/BiasAddBiasAdd*resnet_block/conv3/conv1d/Squeeze:output:01resnet_block/conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/conv3/BiasAddЅ
+resnet_block/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2-
+resnet_block/shortcut/conv1d/ExpandDims/dimф
'resnet_block/shortcut/conv1d/ExpandDims
ExpandDimsSqueeze_1:output:04resnet_block/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2)
'resnet_block/shortcut/conv1d/ExpandDimsњ
8resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAresnet_block_shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02:
8resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOp 
-resnet_block/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-resnet_block/shortcut/conv1d/ExpandDims_1/dim
)resnet_block/shortcut/conv1d/ExpandDims_1
ExpandDims@resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:06resnet_block/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2+
)resnet_block/shortcut/conv1d/ExpandDims_1
resnet_block/shortcut/conv1dConv2D0resnet_block/shortcut/conv1d/ExpandDims:output:02resnet_block/shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/shortcut/conv1dд
$resnet_block/shortcut/conv1d/SqueezeSqueeze%resnet_block/shortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2&
$resnet_block/shortcut/conv1d/SqueezeЮ
,resnet_block/shortcut/BiasAdd/ReadVariableOpReadVariableOp5resnet_block_shortcut_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,resnet_block/shortcut/BiasAdd/ReadVariableOpф
resnet_block/shortcut/BiasAddBiasAdd-resnet_block/shortcut/conv1d/Squeeze:output:04resnet_block/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/shortcut/BiasAddИ
resnet_block/add/addAddV2#resnet_block/conv3/BiasAdd:output:0&resnet_block/shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/add/add
resnet_block/out_block/ReluReluresnet_block/add/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/out_block/ReluЃ
*resnet_block_1/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_1/conv1/conv1d/ExpandDims/dimј
&resnet_block_1/conv1/conv1d/ExpandDims
ExpandDims)resnet_block/out_block/Relu:activations:03resnet_block_1/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2(
&resnet_block_1/conv1/conv1d/ExpandDimsї
7resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_1_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype029
7resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_1/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_1/conv1/conv1d/ExpandDims_1/dim
(resnet_block_1/conv1/conv1d/ExpandDims_1
ExpandDims?resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_1/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2*
(resnet_block_1/conv1/conv1d/ExpandDims_1
resnet_block_1/conv1/conv1dConv2D/resnet_block_1/conv1/conv1d/ExpandDims:output:01resnet_block_1/conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
resnet_block_1/conv1/conv1dб
#resnet_block_1/conv1/conv1d/SqueezeSqueeze$resnet_block_1/conv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_1/conv1/conv1d/SqueezeЫ
+resnet_block_1/conv1/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+resnet_block_1/conv1/BiasAdd/ReadVariableOpр
resnet_block_1/conv1/BiasAddBiasAdd,resnet_block_1/conv1/conv1d/Squeeze:output:03resnet_block_1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/conv1/BiasAdd
resnet_block_1/relu1/ReluRelu%resnet_block_1/conv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/relu1/ReluЃ
*resnet_block_1/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_1/conv2/conv1d/ExpandDims/dimі
&resnet_block_1/conv2/conv1d/ExpandDims
ExpandDims'resnet_block_1/relu1/Relu:activations:03resnet_block_1/conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2(
&resnet_block_1/conv2/conv1d/ExpandDimsї
7resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_1_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_1/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_1/conv2/conv1d/ExpandDims_1/dim
(resnet_block_1/conv2/conv1d/ExpandDims_1
ExpandDims?resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_1/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2*
(resnet_block_1/conv2/conv1d/ExpandDims_1
resnet_block_1/conv2/conv1dConv2D/resnet_block_1/conv2/conv1d/ExpandDims:output:01resnet_block_1/conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
resnet_block_1/conv2/conv1dб
#resnet_block_1/conv2/conv1d/SqueezeSqueeze$resnet_block_1/conv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_1/conv2/conv1d/SqueezeЫ
+resnet_block_1/conv2/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+resnet_block_1/conv2/BiasAdd/ReadVariableOpр
resnet_block_1/conv2/BiasAddBiasAdd,resnet_block_1/conv2/conv1d/Squeeze:output:03resnet_block_1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/conv2/BiasAdd
resnet_block_1/relu2/ReluRelu%resnet_block_1/conv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/relu2/ReluЃ
*resnet_block_1/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_1/conv3/conv1d/ExpandDims/dimі
&resnet_block_1/conv3/conv1d/ExpandDims
ExpandDims'resnet_block_1/relu2/Relu:activations:03resnet_block_1/conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2(
&resnet_block_1/conv3/conv1d/ExpandDimsї
7resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_1_conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_1/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_1/conv3/conv1d/ExpandDims_1/dim
(resnet_block_1/conv3/conv1d/ExpandDims_1
ExpandDims?resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_1/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2*
(resnet_block_1/conv3/conv1d/ExpandDims_1
resnet_block_1/conv3/conv1dConv2D/resnet_block_1/conv3/conv1d/ExpandDims:output:01resnet_block_1/conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
resnet_block_1/conv3/conv1dб
#resnet_block_1/conv3/conv1d/SqueezeSqueeze$resnet_block_1/conv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_1/conv3/conv1d/SqueezeЫ
+resnet_block_1/conv3/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_1_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+resnet_block_1/conv3/BiasAdd/ReadVariableOpр
resnet_block_1/conv3/BiasAddBiasAdd,resnet_block_1/conv3/conv1d/Squeeze:output:03resnet_block_1/conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/conv3/BiasAddЉ
-resnet_block_1/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-resnet_block_1/shortcut/conv1d/ExpandDims/dim
)resnet_block_1/shortcut/conv1d/ExpandDims
ExpandDims)resnet_block/out_block/Relu:activations:06resnet_block_1/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2+
)resnet_block_1/shortcut/conv1d/ExpandDims
:resnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCresnet_block_1_shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02<
:resnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOpЄ
/resnet_block_1/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/resnet_block_1/shortcut/conv1d/ExpandDims_1/dim
+resnet_block_1/shortcut/conv1d/ExpandDims_1
ExpandDimsBresnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:08resnet_block_1/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2-
+resnet_block_1/shortcut/conv1d/ExpandDims_1
resnet_block_1/shortcut/conv1dConv2D2resnet_block_1/shortcut/conv1d/ExpandDims:output:04resnet_block_1/shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2 
resnet_block_1/shortcut/conv1dк
&resnet_block_1/shortcut/conv1d/SqueezeSqueeze'resnet_block_1/shortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2(
&resnet_block_1/shortcut/conv1d/Squeezeд
.resnet_block_1/shortcut/BiasAdd/ReadVariableOpReadVariableOp7resnet_block_1_shortcut_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.resnet_block_1/shortcut/BiasAdd/ReadVariableOpь
resnet_block_1/shortcut/BiasAddBiasAdd/resnet_block_1/shortcut/conv1d/Squeeze:output:06resnet_block_1/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2!
resnet_block_1/shortcut/BiasAddФ
resnet_block_1/add_1/addAddV2%resnet_block_1/conv3/BiasAdd:output:0(resnet_block_1/shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/add_1/add
resnet_block_1/out_block/ReluReluresnet_block_1/add_1/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/out_block/ReluЃ
*resnet_block_2/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_2/conv1/conv1d/ExpandDims/dimњ
&resnet_block_2/conv1/conv1d/ExpandDims
ExpandDims+resnet_block_1/out_block/Relu:activations:03resnet_block_2/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2(
&resnet_block_2/conv1/conv1d/ExpandDimsј
7resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_2_conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype029
7resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_2/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_2/conv1/conv1d/ExpandDims_1/dim
(resnet_block_2/conv1/conv1d/ExpandDims_1
ExpandDims?resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_2/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2*
(resnet_block_2/conv1/conv1d/ExpandDims_1
resnet_block_2/conv1/conv1dConv2D/resnet_block_2/conv1/conv1d/ExpandDims:output:01resnet_block_2/conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_2/conv1/conv1dв
#resnet_block_2/conv1/conv1d/SqueezeSqueeze$resnet_block_2/conv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_2/conv1/conv1d/SqueezeЬ
+resnet_block_2/conv1/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_2/conv1/BiasAdd/ReadVariableOpс
resnet_block_2/conv1/BiasAddBiasAdd,resnet_block_2/conv1/conv1d/Squeeze:output:03resnet_block_2/conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/conv1/BiasAdd
resnet_block_2/relu1/ReluRelu%resnet_block_2/conv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/relu1/ReluЃ
*resnet_block_2/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_2/conv2/conv1d/ExpandDims/dimї
&resnet_block_2/conv2/conv1d/ExpandDims
ExpandDims'resnet_block_2/relu1/Relu:activations:03resnet_block_2/conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_2/conv2/conv1d/ExpandDimsљ
7resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_2_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_2/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_2/conv2/conv1d/ExpandDims_1/dim
(resnet_block_2/conv2/conv1d/ExpandDims_1
ExpandDims?resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_2/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_2/conv2/conv1d/ExpandDims_1
resnet_block_2/conv2/conv1dConv2D/resnet_block_2/conv2/conv1d/ExpandDims:output:01resnet_block_2/conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_2/conv2/conv1dв
#resnet_block_2/conv2/conv1d/SqueezeSqueeze$resnet_block_2/conv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_2/conv2/conv1d/SqueezeЬ
+resnet_block_2/conv2/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_2/conv2/BiasAdd/ReadVariableOpс
resnet_block_2/conv2/BiasAddBiasAdd,resnet_block_2/conv2/conv1d/Squeeze:output:03resnet_block_2/conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/conv2/BiasAdd
resnet_block_2/relu2/ReluRelu%resnet_block_2/conv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/relu2/ReluЃ
*resnet_block_2/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_2/conv3/conv1d/ExpandDims/dimї
&resnet_block_2/conv3/conv1d/ExpandDims
ExpandDims'resnet_block_2/relu2/Relu:activations:03resnet_block_2/conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_2/conv3/conv1d/ExpandDimsљ
7resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_2_conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_2/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_2/conv3/conv1d/ExpandDims_1/dim
(resnet_block_2/conv3/conv1d/ExpandDims_1
ExpandDims?resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_2/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_2/conv3/conv1d/ExpandDims_1
resnet_block_2/conv3/conv1dConv2D/resnet_block_2/conv3/conv1d/ExpandDims:output:01resnet_block_2/conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_2/conv3/conv1dв
#resnet_block_2/conv3/conv1d/SqueezeSqueeze$resnet_block_2/conv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_2/conv3/conv1d/SqueezeЬ
+resnet_block_2/conv3/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_2_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_2/conv3/BiasAdd/ReadVariableOpс
resnet_block_2/conv3/BiasAddBiasAdd,resnet_block_2/conv3/conv1d/Squeeze:output:03resnet_block_2/conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/conv3/BiasAddЉ
-resnet_block_2/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-resnet_block_2/shortcut/conv1d/ExpandDims/dim
)resnet_block_2/shortcut/conv1d/ExpandDims
ExpandDims+resnet_block_1/out_block/Relu:activations:06resnet_block_2/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2+
)resnet_block_2/shortcut/conv1d/ExpandDims
:resnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCresnet_block_2_shortcut_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02<
:resnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOpЄ
/resnet_block_2/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/resnet_block_2/shortcut/conv1d/ExpandDims_1/dim
+resnet_block_2/shortcut/conv1d/ExpandDims_1
ExpandDimsBresnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:08resnet_block_2/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2-
+resnet_block_2/shortcut/conv1d/ExpandDims_1
resnet_block_2/shortcut/conv1dConv2D2resnet_block_2/shortcut/conv1d/ExpandDims:output:04resnet_block_2/shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2 
resnet_block_2/shortcut/conv1dл
&resnet_block_2/shortcut/conv1d/SqueezeSqueeze'resnet_block_2/shortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2(
&resnet_block_2/shortcut/conv1d/Squeezeе
.resnet_block_2/shortcut/BiasAdd/ReadVariableOpReadVariableOp7resnet_block_2_shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.resnet_block_2/shortcut/BiasAdd/ReadVariableOpэ
resnet_block_2/shortcut/BiasAddBiasAdd/resnet_block_2/shortcut/conv1d/Squeeze:output:06resnet_block_2/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2!
resnet_block_2/shortcut/BiasAddХ
resnet_block_2/add_2/addAddV2%resnet_block_2/conv3/BiasAdd:output:0(resnet_block_2/shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/add_2/add
resnet_block_2/out_block/ReluReluresnet_block_2/add_2/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/out_block/ReluЃ
*resnet_block_3/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_3/conv1/conv1d/ExpandDims/dimћ
&resnet_block_3/conv1/conv1d/ExpandDims
ExpandDims+resnet_block_2/out_block/Relu:activations:03resnet_block_3/conv1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_3/conv1/conv1d/ExpandDimsљ
7resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_3_conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_3/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_3/conv1/conv1d/ExpandDims_1/dim
(resnet_block_3/conv1/conv1d/ExpandDims_1
ExpandDims?resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_3/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_3/conv1/conv1d/ExpandDims_1
resnet_block_3/conv1/conv1dConv2D/resnet_block_3/conv1/conv1d/ExpandDims:output:01resnet_block_3/conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_3/conv1/conv1dв
#resnet_block_3/conv1/conv1d/SqueezeSqueeze$resnet_block_3/conv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_3/conv1/conv1d/SqueezeЬ
+resnet_block_3/conv1/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_3/conv1/BiasAdd/ReadVariableOpс
resnet_block_3/conv1/BiasAddBiasAdd,resnet_block_3/conv1/conv1d/Squeeze:output:03resnet_block_3/conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/conv1/BiasAdd
resnet_block_3/relu1/ReluRelu%resnet_block_3/conv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/relu1/ReluЃ
*resnet_block_3/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_3/conv2/conv1d/ExpandDims/dimї
&resnet_block_3/conv2/conv1d/ExpandDims
ExpandDims'resnet_block_3/relu1/Relu:activations:03resnet_block_3/conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_3/conv2/conv1d/ExpandDimsљ
7resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_3_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_3/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_3/conv2/conv1d/ExpandDims_1/dim
(resnet_block_3/conv2/conv1d/ExpandDims_1
ExpandDims?resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_3/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_3/conv2/conv1d/ExpandDims_1
resnet_block_3/conv2/conv1dConv2D/resnet_block_3/conv2/conv1d/ExpandDims:output:01resnet_block_3/conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_3/conv2/conv1dв
#resnet_block_3/conv2/conv1d/SqueezeSqueeze$resnet_block_3/conv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_3/conv2/conv1d/SqueezeЬ
+resnet_block_3/conv2/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_3/conv2/BiasAdd/ReadVariableOpс
resnet_block_3/conv2/BiasAddBiasAdd,resnet_block_3/conv2/conv1d/Squeeze:output:03resnet_block_3/conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/conv2/BiasAdd
resnet_block_3/relu2/ReluRelu%resnet_block_3/conv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/relu2/ReluЃ
*resnet_block_3/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_3/conv3/conv1d/ExpandDims/dimї
&resnet_block_3/conv3/conv1d/ExpandDims
ExpandDims'resnet_block_3/relu2/Relu:activations:03resnet_block_3/conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_3/conv3/conv1d/ExpandDimsљ
7resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_3_conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_3/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_3/conv3/conv1d/ExpandDims_1/dim
(resnet_block_3/conv3/conv1d/ExpandDims_1
ExpandDims?resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_3/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_3/conv3/conv1d/ExpandDims_1
resnet_block_3/conv3/conv1dConv2D/resnet_block_3/conv3/conv1d/ExpandDims:output:01resnet_block_3/conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_3/conv3/conv1dв
#resnet_block_3/conv3/conv1d/SqueezeSqueeze$resnet_block_3/conv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_3/conv3/conv1d/SqueezeЬ
+resnet_block_3/conv3/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_3/conv3/BiasAdd/ReadVariableOpс
resnet_block_3/conv3/BiasAddBiasAdd,resnet_block_3/conv3/conv1d/Squeeze:output:03resnet_block_3/conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/conv3/BiasAddЉ
-resnet_block_3/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-resnet_block_3/shortcut/conv1d/ExpandDims/dim
)resnet_block_3/shortcut/conv1d/ExpandDims
ExpandDims+resnet_block_2/out_block/Relu:activations:06resnet_block_3/shortcut/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2+
)resnet_block_3/shortcut/conv1d/ExpandDims
:resnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCresnet_block_3_shortcut_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02<
:resnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOpЄ
/resnet_block_3/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/resnet_block_3/shortcut/conv1d/ExpandDims_1/dim
+resnet_block_3/shortcut/conv1d/ExpandDims_1
ExpandDimsBresnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:08resnet_block_3/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2-
+resnet_block_3/shortcut/conv1d/ExpandDims_1
resnet_block_3/shortcut/conv1dConv2D2resnet_block_3/shortcut/conv1d/ExpandDims:output:04resnet_block_3/shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2 
resnet_block_3/shortcut/conv1dл
&resnet_block_3/shortcut/conv1d/SqueezeSqueeze'resnet_block_3/shortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2(
&resnet_block_3/shortcut/conv1d/Squeezeе
.resnet_block_3/shortcut/BiasAdd/ReadVariableOpReadVariableOp7resnet_block_3_shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.resnet_block_3/shortcut/BiasAdd/ReadVariableOpэ
resnet_block_3/shortcut/BiasAddBiasAdd/resnet_block_3/shortcut/conv1d/Squeeze:output:06resnet_block_3/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2!
resnet_block_3/shortcut/BiasAddХ
resnet_block_3/add_3/addAddV2%resnet_block_3/conv3/BiasAdd:output:0(resnet_block_3/shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/add_3/add
resnet_block_3/out_block/ReluReluresnet_block_3/add_3/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/out_block/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
flatten/ConstЅ
flatten/ReshapeReshape+resnet_block_3/out_block/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ@2
flatten/Reshape
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype02
fc1/MatMul/ReadVariableOp

fc1/MatMulMatMulflatten/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc1/MatMul
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc1/BiasAdd/ReadVariableOp
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
fc1/BiasAdde
fc1/ReluRelufc1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc1/Relu
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
fc2/MatMul/ReadVariableOp

fc2/MatMulMatMulfc1/Relu:activations:0!fc2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc2/MatMul
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc2/BiasAdd/ReadVariableOp
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
fc2/BiasAdde
fc2/ReluRelufc2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc2/Relu
fc3/MatMul/ReadVariableOpReadVariableOp"fc3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
fc3/MatMul/ReadVariableOp

fc3/MatMulMatMulfc2/Relu:activations:0!fc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2

fc3/MatMul
fc3/BiasAdd/ReadVariableOpReadVariableOp#fc3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc3/BiasAdd/ReadVariableOp
fc3/BiasAddBiasAddfc3/MatMul:product:0"fc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fc3/BiasAddm
fc3/SigmoidSigmoidfc3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fc3/Sigmoidc
IdentityIdentityfc3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:џџџџџџџџџ:	:::::::::::::::::::::::::::::::::::::::U Q
,
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:%!

_output_shapes
:	
Г

J__inference_resnet_block_3_layer_call_and_return_conditional_losses_211884

inputs
conv1_211859
conv1_211861
conv2_211865
conv2_211867
conv3_211871
conv3_211873
shortcut_211876
shortcut_211878
identityЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallЂconv3/StatefulPartitionedCallЂ shortcut/StatefulPartitionedCall
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_211859conv1_211861*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2115362
conv1/StatefulPartitionedCall№
relu1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu1_layer_call_and_return_conditional_losses_2115712
relu1/PartitionedCallЂ
conv2/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0conv2_211865conv2_211867*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2116142
conv2/StatefulPartitionedCall№
relu2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu2_layer_call_and_return_conditional_losses_2116492
relu2/PartitionedCallЂ
conv3/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0conv3_211871conv3_211873*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2116922
conv3/StatefulPartitionedCall
 shortcut/StatefulPartitionedCallStatefulPartitionedCallinputsshortcut_211876shortcut_211878*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_shortcut_layer_call_and_return_conditional_losses_2117472"
 shortcut/StatefulPartitionedCallЅ
add/addAddV2&conv3/StatefulPartitionedCall:output:0)shortcut/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
add/addс
out_block/PartitionedCallPartitionedCalladd/add:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_out_block_layer_call_and_return_conditional_losses_2117832
out_block/PartitionedCallў
IdentityIdentity"out_block/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall!^shortcut/StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:џџџџџџџџџ@::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2D
 shortcut/StatefulPartitionedCall shortcut/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ж
Ж
A__inference_conv3_layer_call_and_return_conditional_losses_217499

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@@:::S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Б

C__inference_res_net_layer_call_and_return_conditional_losses_214135
input_1 
log_mel_spectrogram_matmul_bB
>resnet_block_conv1_conv1d_expanddims_1_readvariableop_resource6
2resnet_block_conv1_biasadd_readvariableop_resourceB
>resnet_block_conv2_conv1d_expanddims_1_readvariableop_resource6
2resnet_block_conv2_biasadd_readvariableop_resourceB
>resnet_block_conv3_conv1d_expanddims_1_readvariableop_resource6
2resnet_block_conv3_biasadd_readvariableop_resourceE
Aresnet_block_shortcut_conv1d_expanddims_1_readvariableop_resource9
5resnet_block_shortcut_biasadd_readvariableop_resourceD
@resnet_block_1_conv1_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_1_conv1_biasadd_readvariableop_resourceD
@resnet_block_1_conv2_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_1_conv2_biasadd_readvariableop_resourceD
@resnet_block_1_conv3_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_1_conv3_biasadd_readvariableop_resourceG
Cresnet_block_1_shortcut_conv1d_expanddims_1_readvariableop_resource;
7resnet_block_1_shortcut_biasadd_readvariableop_resourceD
@resnet_block_2_conv1_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_2_conv1_biasadd_readvariableop_resourceD
@resnet_block_2_conv2_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_2_conv2_biasadd_readvariableop_resourceD
@resnet_block_2_conv3_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_2_conv3_biasadd_readvariableop_resourceG
Cresnet_block_2_shortcut_conv1d_expanddims_1_readvariableop_resource;
7resnet_block_2_shortcut_biasadd_readvariableop_resourceD
@resnet_block_3_conv1_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_3_conv1_biasadd_readvariableop_resourceD
@resnet_block_3_conv2_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_3_conv2_biasadd_readvariableop_resourceD
@resnet_block_3_conv3_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_3_conv3_biasadd_readvariableop_resourceG
Cresnet_block_3_shortcut_conv1d_expanddims_1_readvariableop_resource;
7resnet_block_3_shortcut_biasadd_readvariableop_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource&
"fc2_matmul_readvariableop_resource'
#fc2_biasadd_readvariableop_resource&
"fc3_matmul_readvariableop_resource'
#fc3_biasadd_readvariableop_resource
identityp
SqueezeSqueezeinput_1*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2	
Squeeze
%log_mel_spectrogram/stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2'
%log_mel_spectrogram/stft/frame_length
#log_mel_spectrogram/stft/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2%
#log_mel_spectrogram/stft/frame_step
log_mel_spectrogram/stft/ConstConst*
_output_shapes
: *
dtype0*
value
B :2 
log_mel_spectrogram/stft/Const
#log_mel_spectrogram/stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#log_mel_spectrogram/stft/frame/axis
$log_mel_spectrogram/stft/frame/ShapeShapeSqueeze:output:0*
T0*
_output_shapes
:2&
$log_mel_spectrogram/stft/frame/Shape
#log_mel_spectrogram/stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#log_mel_spectrogram/stft/frame/Rank
*log_mel_spectrogram/stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*log_mel_spectrogram/stft/frame/range/start
*log_mel_spectrogram/stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*log_mel_spectrogram/stft/frame/range/delta
$log_mel_spectrogram/stft/frame/rangeRange3log_mel_spectrogram/stft/frame/range/start:output:0,log_mel_spectrogram/stft/frame/Rank:output:03log_mel_spectrogram/stft/frame/range/delta:output:0*
_output_shapes
:2&
$log_mel_spectrogram/stft/frame/rangeЛ
2log_mel_spectrogram/stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ24
2log_mel_spectrogram/stft/frame/strided_slice/stackЖ
4log_mel_spectrogram/stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4log_mel_spectrogram/stft/frame/strided_slice/stack_1Ж
4log_mel_spectrogram/stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4log_mel_spectrogram/stft/frame/strided_slice/stack_2
,log_mel_spectrogram/stft/frame/strided_sliceStridedSlice-log_mel_spectrogram/stft/frame/range:output:0;log_mel_spectrogram/stft/frame/strided_slice/stack:output:0=log_mel_spectrogram/stft/frame/strided_slice/stack_1:output:0=log_mel_spectrogram/stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,log_mel_spectrogram/stft/frame/strided_slice
$log_mel_spectrogram/stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$log_mel_spectrogram/stft/frame/sub/yЭ
"log_mel_spectrogram/stft/frame/subSub,log_mel_spectrogram/stft/frame/Rank:output:0-log_mel_spectrogram/stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/subг
$log_mel_spectrogram/stft/frame/sub_1Sub&log_mel_spectrogram/stft/frame/sub:z:05log_mel_spectrogram/stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/sub_1
'log_mel_spectrogram/stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'log_mel_spectrogram/stft/frame/packed/1
%log_mel_spectrogram/stft/frame/packedPack5log_mel_spectrogram/stft/frame/strided_slice:output:00log_mel_spectrogram/stft/frame/packed/1:output:0(log_mel_spectrogram/stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2'
%log_mel_spectrogram/stft/frame/packedЂ
.log_mel_spectrogram/stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.log_mel_spectrogram/stft/frame/split/split_dimК
$log_mel_spectrogram/stft/frame/splitSplitV-log_mel_spectrogram/stft/frame/Shape:output:0.log_mel_spectrogram/stft/frame/packed:output:07log_mel_spectrogram/stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2&
$log_mel_spectrogram/stft/frame/split
,log_mel_spectrogram/stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2.
,log_mel_spectrogram/stft/frame/Reshape/shapeЃ
.log_mel_spectrogram/stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 20
.log_mel_spectrogram/stft/frame/Reshape/shape_1ф
&log_mel_spectrogram/stft/frame/ReshapeReshape-log_mel_spectrogram/stft/frame/split:output:17log_mel_spectrogram/stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2(
&log_mel_spectrogram/stft/frame/Reshape
#log_mel_spectrogram/stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2%
#log_mel_spectrogram/stft/frame/Size
%log_mel_spectrogram/stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%log_mel_spectrogram/stft/frame/Size_1
$log_mel_spectrogram/stft/frame/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$log_mel_spectrogram/stft/frame/ConstЁ
"log_mel_spectrogram/stft/frame/NegNeg/log_mel_spectrogram/stft/frame/Reshape:output:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/Negе
'log_mel_spectrogram/stft/frame/floordivFloorDiv&log_mel_spectrogram/stft/frame/Neg:y:0,log_mel_spectrogram/stft/frame_step:output:0*
T0*
_output_shapes
: 2)
'log_mel_spectrogram/stft/frame/floordivЁ
$log_mel_spectrogram/stft/frame/Neg_1Neg+log_mel_spectrogram/stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/Neg_1
&log_mel_spectrogram/stft/frame/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_spectrogram/stft/frame/sub_2/yЯ
$log_mel_spectrogram/stft/frame/sub_2Sub(log_mel_spectrogram/stft/frame/Neg_1:y:0/log_mel_spectrogram/stft/frame/sub_2/y:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/sub_2Ш
"log_mel_spectrogram/stft/frame/mulMul,log_mel_spectrogram/stft/frame_step:output:0(log_mel_spectrogram/stft/frame/sub_2:z:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/mulЪ
"log_mel_spectrogram/stft/frame/addAddV2.log_mel_spectrogram/stft/frame_length:output:0&log_mel_spectrogram/stft/frame/mul:z:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/addЭ
$log_mel_spectrogram/stft/frame/sub_3Sub&log_mel_spectrogram/stft/frame/add:z:0/log_mel_spectrogram/stft/frame/Reshape:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/sub_3
(log_mel_spectrogram/stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2*
(log_mel_spectrogram/stft/frame/Maximum/xй
&log_mel_spectrogram/stft/frame/MaximumMaximum1log_mel_spectrogram/stft/frame/Maximum/x:output:0(log_mel_spectrogram/stft/frame/sub_3:z:0*
T0*
_output_shapes
: 2(
&log_mel_spectrogram/stft/frame/Maximum
*log_mel_spectrogram/stft/frame/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*log_mel_spectrogram/stft/frame/zeros/mul/yп
(log_mel_spectrogram/stft/frame/zeros/mulMul,log_mel_spectrogram/stft/frame/Size:output:03log_mel_spectrogram/stft/frame/zeros/mul/y:output:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/frame/zeros/mul
+log_mel_spectrogram/stft/frame/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2-
+log_mel_spectrogram/stft/frame/zeros/Less/yу
)log_mel_spectrogram/stft/frame/zeros/LessLess,log_mel_spectrogram/stft/frame/zeros/mul:z:04log_mel_spectrogram/stft/frame/zeros/Less/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/zeros/Less 
-log_mel_spectrogram/stft/frame/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-log_mel_spectrogram/stft/frame/zeros/packed/1і
+log_mel_spectrogram/stft/frame/zeros/packedPack,log_mel_spectrogram/stft/frame/Size:output:06log_mel_spectrogram/stft/frame/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2-
+log_mel_spectrogram/stft/frame/zeros/packed
*log_mel_spectrogram/stft/frame/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2,
*log_mel_spectrogram/stft/frame/zeros/Constш
$log_mel_spectrogram/stft/frame/zerosFill4log_mel_spectrogram/stft/frame/zeros/packed:output:03log_mel_spectrogram/stft/frame/zeros/Const:output:0*
T0*
_output_shapes

:2&
$log_mel_spectrogram/stft/frame/zeros
+log_mel_spectrogram/stft/frame/packed_1/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2-
+log_mel_spectrogram/stft/frame/packed_1/0/0ю
)log_mel_spectrogram/stft/frame/packed_1/0Pack4log_mel_spectrogram/stft/frame/packed_1/0/0:output:0*log_mel_spectrogram/stft/frame/Maximum:z:0*
N*
T0*
_output_shapes
:2+
)log_mel_spectrogram/stft/frame/packed_1/0Р
'log_mel_spectrogram/stft/frame/packed_1Pack2log_mel_spectrogram/stft/frame/packed_1/0:output:0*
N*
T0*
_output_shapes

:2)
'log_mel_spectrogram/stft/frame/packed_1
,log_mel_spectrogram/stft/frame/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/frame/zeros_1/mul/yч
*log_mel_spectrogram/stft/frame/zeros_1/mulMul.log_mel_spectrogram/stft/frame/Size_1:output:05log_mel_spectrogram/stft/frame/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2,
*log_mel_spectrogram/stft/frame/zeros_1/mulЁ
-log_mel_spectrogram/stft/frame/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2/
-log_mel_spectrogram/stft/frame/zeros_1/Less/yы
+log_mel_spectrogram/stft/frame/zeros_1/LessLess.log_mel_spectrogram/stft/frame/zeros_1/mul:z:06log_mel_spectrogram/stft/frame/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2-
+log_mel_spectrogram/stft/frame/zeros_1/LessЄ
/log_mel_spectrogram/stft/frame/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :21
/log_mel_spectrogram/stft/frame/zeros_1/packed/1ў
-log_mel_spectrogram/stft/frame/zeros_1/packedPack.log_mel_spectrogram/stft/frame/Size_1:output:08log_mel_spectrogram/stft/frame/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2/
-log_mel_spectrogram/stft/frame/zeros_1/packed
,log_mel_spectrogram/stft/frame/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/zeros_1/Constю
&log_mel_spectrogram/stft/frame/zeros_1Fill6log_mel_spectrogram/stft/frame/zeros_1/packed:output:05log_mel_spectrogram/stft/frame/zeros_1/Const:output:0*
T0*
_output_shapes

: 2(
&log_mel_spectrogram/stft/frame/zeros_1
*log_mel_spectrogram/stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*log_mel_spectrogram/stft/frame/concat/axisг
%log_mel_spectrogram/stft/frame/concatConcatV2-log_mel_spectrogram/stft/frame/zeros:output:00log_mel_spectrogram/stft/frame/packed_1:output:0/log_mel_spectrogram/stft/frame/zeros_1:output:03log_mel_spectrogram/stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes

:2'
%log_mel_spectrogram/stft/frame/concat
$log_mel_spectrogram/stft/frame/PadV2PadV2Squeeze:output:0.log_mel_spectrogram/stft/frame/concat:output:0-log_mel_spectrogram/stft/frame/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2&
$log_mel_spectrogram/stft/frame/PadV2­
&log_mel_spectrogram/stft/frame/Shape_1Shape-log_mel_spectrogram/stft/frame/PadV2:output:0*
T0*
_output_shapes
:2(
&log_mel_spectrogram/stft/frame/Shape_1
&log_mel_spectrogram/stft/frame/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_spectrogram/stft/frame/add_1/yо
$log_mel_spectrogram/stft/frame/add_1AddV25log_mel_spectrogram/stft/frame/strided_slice:output:0/log_mel_spectrogram/stft/frame/add_1/y:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/add_1й
4log_mel_spectrogram/stft/frame/strided_slice_1/stackPack5log_mel_spectrogram/stft/frame/strided_slice:output:0*
N*
T0*
_output_shapes
:26
4log_mel_spectrogram/stft/frame/strided_slice_1/stackа
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_1Pack(log_mel_spectrogram/stft/frame/add_1:z:0*
N*
T0*
_output_shapes
:28
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_1К
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_2Ј
.log_mel_spectrogram/stft/frame/strided_slice_1StridedSlice/log_mel_spectrogram/stft/frame/Shape_1:output:0=log_mel_spectrogram/stft/frame/strided_slice_1/stack:output:0?log_mel_spectrogram/stft/frame/strided_slice_1/stack_1:output:0?log_mel_spectrogram/stft/frame/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.log_mel_spectrogram/stft/frame/strided_slice_1
(log_mel_spectrogram/stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(log_mel_spectrogram/stft/frame/gcd/Const
+log_mel_spectrogram/stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_spectrogram/stft/frame/floordiv_1/yщ
)log_mel_spectrogram/stft/frame/floordiv_1FloorDiv.log_mel_spectrogram/stft/frame_length:output:04log_mel_spectrogram/stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/floordiv_1
+log_mel_spectrogram/stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_spectrogram/stft/frame/floordiv_2/yч
)log_mel_spectrogram/stft/frame/floordiv_2FloorDiv,log_mel_spectrogram/stft/frame_step:output:04log_mel_spectrogram/stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/floordiv_2
+log_mel_spectrogram/stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_spectrogram/stft/frame/floordiv_3/yђ
)log_mel_spectrogram/stft/frame/floordiv_3FloorDiv7log_mel_spectrogram/stft/frame/strided_slice_1:output:04log_mel_spectrogram/stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/floordiv_3
&log_mel_spectrogram/stft/frame/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_spectrogram/stft/frame/mul_1/yд
$log_mel_spectrogram/stft/frame/mul_1Mul-log_mel_spectrogram/stft/frame/floordiv_3:z:0/log_mel_spectrogram/stft/frame/mul_1/y:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/mul_1Ф
0log_mel_spectrogram/stft/frame/concat_1/values_1Pack(log_mel_spectrogram/stft/frame/mul_1:z:0*
N*
T0*
_output_shapes
:22
0log_mel_spectrogram/stft/frame/concat_1/values_1
,log_mel_spectrogram/stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/concat_1/axisм
'log_mel_spectrogram/stft/frame/concat_1ConcatV2-log_mel_spectrogram/stft/frame/split:output:09log_mel_spectrogram/stft/frame/concat_1/values_1:output:0-log_mel_spectrogram/stft/frame/split:output:25log_mel_spectrogram/stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'log_mel_spectrogram/stft/frame/concat_1Њ
2log_mel_spectrogram/stft/frame/concat_2/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :24
2log_mel_spectrogram/stft/frame/concat_2/values_1/1
0log_mel_spectrogram/stft/frame/concat_2/values_1Pack-log_mel_spectrogram/stft/frame/floordiv_3:z:0;log_mel_spectrogram/stft/frame/concat_2/values_1/1:output:0*
N*
T0*
_output_shapes
:22
0log_mel_spectrogram/stft/frame/concat_2/values_1
,log_mel_spectrogram/stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/concat_2/axisм
'log_mel_spectrogram/stft/frame/concat_2ConcatV2-log_mel_spectrogram/stft/frame/split:output:09log_mel_spectrogram/stft/frame/concat_2/values_1:output:0-log_mel_spectrogram/stft/frame/split:output:25log_mel_spectrogram/stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2)
'log_mel_spectrogram/stft/frame/concat_2 
)log_mel_spectrogram/stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2+
)log_mel_spectrogram/stft/frame/zeros_likeЊ
.log_mel_spectrogram/stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:20
.log_mel_spectrogram/stft/frame/ones_like/ShapeЂ
.log_mel_spectrogram/stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :20
.log_mel_spectrogram/stft/frame/ones_like/Constѓ
(log_mel_spectrogram/stft/frame/ones_likeFill7log_mel_spectrogram/stft/frame/ones_like/Shape:output:07log_mel_spectrogram/stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2*
(log_mel_spectrogram/stft/frame/ones_likeњ
+log_mel_spectrogram/stft/frame/StridedSliceStridedSlice-log_mel_spectrogram/stft/frame/PadV2:output:02log_mel_spectrogram/stft/frame/zeros_like:output:00log_mel_spectrogram/stft/frame/concat_1:output:01log_mel_spectrogram/stft/frame/ones_like:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2-
+log_mel_spectrogram/stft/frame/StridedSlice
(log_mel_spectrogram/stft/frame/Reshape_1Reshape4log_mel_spectrogram/stft/frame/StridedSlice:output:00log_mel_spectrogram/stft/frame/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
(log_mel_spectrogram/stft/frame/Reshape_1
,log_mel_spectrogram/stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/range_1/start
,log_mel_spectrogram/stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/frame/range_1/delta
&log_mel_spectrogram/stft/frame/range_1Range5log_mel_spectrogram/stft/frame/range_1/start:output:0(log_mel_spectrogram/stft/frame/Neg_1:y:05log_mel_spectrogram/stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2(
&log_mel_spectrogram/stft/frame/range_1с
$log_mel_spectrogram/stft/frame/mul_2Mul/log_mel_spectrogram/stft/frame/range_1:output:0-log_mel_spectrogram/stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2&
$log_mel_spectrogram/stft/frame/mul_2І
0log_mel_spectrogram/stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0log_mel_spectrogram/stft/frame/Reshape_2/shape/1ћ
.log_mel_spectrogram/stft/frame/Reshape_2/shapePack(log_mel_spectrogram/stft/frame/Neg_1:y:09log_mel_spectrogram/stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:20
.log_mel_spectrogram/stft/frame/Reshape_2/shapeє
(log_mel_spectrogram/stft/frame/Reshape_2Reshape(log_mel_spectrogram/stft/frame/mul_2:z:07log_mel_spectrogram/stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(log_mel_spectrogram/stft/frame/Reshape_2
,log_mel_spectrogram/stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/range_2/start
,log_mel_spectrogram/stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/frame/range_2/delta
&log_mel_spectrogram/stft/frame/range_2Range5log_mel_spectrogram/stft/frame/range_2/start:output:0-log_mel_spectrogram/stft/frame/floordiv_1:z:05log_mel_spectrogram/stft/frame/range_2/delta:output:0*
_output_shapes
: 2(
&log_mel_spectrogram/stft/frame/range_2І
0log_mel_spectrogram/stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :22
0log_mel_spectrogram/stft/frame/Reshape_3/shape/0
.log_mel_spectrogram/stft/frame/Reshape_3/shapePack9log_mel_spectrogram/stft/frame/Reshape_3/shape/0:output:0-log_mel_spectrogram/stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:20
.log_mel_spectrogram/stft/frame/Reshape_3/shapeђ
(log_mel_spectrogram/stft/frame/Reshape_3Reshape/log_mel_spectrogram/stft/frame/range_2:output:07log_mel_spectrogram/stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

: 2*
(log_mel_spectrogram/stft/frame/Reshape_3э
$log_mel_spectrogram/stft/frame/add_2AddV21log_mel_spectrogram/stft/frame/Reshape_2:output:01log_mel_spectrogram/stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$log_mel_spectrogram/stft/frame/add_2и
'log_mel_spectrogram/stft/frame/GatherV2GatherV21log_mel_spectrogram/stft/frame/Reshape_1:output:0(log_mel_spectrogram/stft/frame/add_2:z:05log_mel_spectrogram/stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 2)
'log_mel_spectrogram/stft/frame/GatherV2є
0log_mel_spectrogram/stft/frame/concat_3/values_1Pack(log_mel_spectrogram/stft/frame/Neg_1:y:0.log_mel_spectrogram/stft/frame_length:output:0*
N*
T0*
_output_shapes
:22
0log_mel_spectrogram/stft/frame/concat_3/values_1
,log_mel_spectrogram/stft/frame/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/concat_3/axisм
'log_mel_spectrogram/stft/frame/concat_3ConcatV2-log_mel_spectrogram/stft/frame/split:output:09log_mel_spectrogram/stft/frame/concat_3/values_1:output:0-log_mel_spectrogram/stft/frame/split:output:25log_mel_spectrogram/stft/frame/concat_3/axis:output:0*
N*
T0*
_output_shapes
:2)
'log_mel_spectrogram/stft/frame/concat_3њ
(log_mel_spectrogram/stft/frame/Reshape_4Reshape0log_mel_spectrogram/stft/frame/GatherV2:output:00log_mel_spectrogram/stft/frame/concat_3:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2*
(log_mel_spectrogram/stft/frame/Reshape_4 
-log_mel_spectrogram/stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-log_mel_spectrogram/stft/hann_window/periodicЦ
)log_mel_spectrogram/stft/hann_window/CastCast6log_mel_spectrogram/stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2+
)log_mel_spectrogram/stft/hann_window/CastЄ
/log_mel_spectrogram/stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :21
/log_mel_spectrogram/stft/hann_window/FloorMod/yѕ
-log_mel_spectrogram/stft/hann_window/FloorModFloorMod.log_mel_spectrogram/stft/frame_length:output:08log_mel_spectrogram/stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2/
-log_mel_spectrogram/stft/hann_window/FloorMod
*log_mel_spectrogram/stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*log_mel_spectrogram/stft/hann_window/sub/xф
(log_mel_spectrogram/stft/hann_window/subSub3log_mel_spectrogram/stft/hann_window/sub/x:output:01log_mel_spectrogram/stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/hann_window/subй
(log_mel_spectrogram/stft/hann_window/mulMul-log_mel_spectrogram/stft/hann_window/Cast:y:0,log_mel_spectrogram/stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/hann_window/mulм
(log_mel_spectrogram/stft/hann_window/addAddV2.log_mel_spectrogram/stft/frame_length:output:0,log_mel_spectrogram/stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/hann_window/add
,log_mel_spectrogram/stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/hann_window/sub_1/yх
*log_mel_spectrogram/stft/hann_window/sub_1Sub,log_mel_spectrogram/stft/hann_window/add:z:05log_mel_spectrogram/stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2,
*log_mel_spectrogram/stft/hann_window/sub_1Т
+log_mel_spectrogram/stft/hann_window/Cast_1Cast.log_mel_spectrogram/stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+log_mel_spectrogram/stft/hann_window/Cast_1І
0log_mel_spectrogram/stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0log_mel_spectrogram/stft/hann_window/range/startІ
0log_mel_spectrogram/stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0log_mel_spectrogram/stft/hann_window/range/deltaЄ
*log_mel_spectrogram/stft/hann_window/rangeRange9log_mel_spectrogram/stft/hann_window/range/start:output:0.log_mel_spectrogram/stft/frame_length:output:09log_mel_spectrogram/stft/hann_window/range/delta:output:0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/rangeЬ
+log_mel_spectrogram/stft/hann_window/Cast_2Cast3log_mel_spectrogram/stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2-
+log_mel_spectrogram/stft/hann_window/Cast_2
*log_mel_spectrogram/stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2,
*log_mel_spectrogram/stft/hann_window/Constы
*log_mel_spectrogram/stft/hann_window/mul_1Mul3log_mel_spectrogram/stft/hann_window/Const:output:0/log_mel_spectrogram/stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/mul_1ю
,log_mel_spectrogram/stft/hann_window/truedivRealDiv.log_mel_spectrogram/stft/hann_window/mul_1:z:0/log_mel_spectrogram/stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2.
,log_mel_spectrogram/stft/hann_window/truedivГ
(log_mel_spectrogram/stft/hann_window/CosCos0log_mel_spectrogram/stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2*
(log_mel_spectrogram/stft/hann_window/CosЁ
,log_mel_spectrogram/stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,log_mel_spectrogram/stft/hann_window/mul_2/xъ
*log_mel_spectrogram/stft/hann_window/mul_2Mul5log_mel_spectrogram/stft/hann_window/mul_2/x:output:0,log_mel_spectrogram/stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/mul_2Ё
,log_mel_spectrogram/stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,log_mel_spectrogram/stft/hann_window/sub_2/xь
*log_mel_spectrogram/stft/hann_window/sub_2Sub5log_mel_spectrogram/stft/hann_window/sub_2/x:output:0.log_mel_spectrogram/stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/sub_2н
log_mel_spectrogram/stft/mulMul1log_mel_spectrogram/stft/frame/Reshape_4:output:0.log_mel_spectrogram/stft/hann_window/sub_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/stft/mulЋ
$log_mel_spectrogram/stft/rfft/packedPack'log_mel_spectrogram/stft/Const:output:0*
N*
T0*
_output_shapes
:2&
$log_mel_spectrogram/stft/rfft/packed
(log_mel_spectrogram/stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2*
(log_mel_spectrogram/stft/rfft/fft_lengthЩ
log_mel_spectrogram/stft/rfftRFFT log_mel_spectrogram/stft/mul:z:01log_mel_spectrogram/stft/rfft/fft_length:output:0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/stft/rfft
log_mel_spectrogram/Abs
ComplexAbs&log_mel_spectrogram/stft/rfft:output:0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Abs
log_mel_spectrogram/SquareSquarelog_mel_spectrogram/Abs:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/SquareН
log_mel_spectrogram/MatMulBatchMatMulV2log_mel_spectrogram/Square:y:0log_mel_spectrogram_matmul_b*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/MatMul
log_mel_spectrogram/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
log_mel_spectrogram/ConstЃ
log_mel_spectrogram/MaxMax#log_mel_spectrogram/MatMul:output:0"log_mel_spectrogram/Const:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Max
log_mel_spectrogram/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2
log_mel_spectrogram/Maximum/xШ
log_mel_spectrogram/MaximumMaximum&log_mel_spectrogram/Maximum/x:output:0#log_mel_spectrogram/MatMul:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Maximum
log_mel_spectrogram/LogLoglog_mel_spectrogram/Maximum:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Log
log_mel_spectrogram/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/Const_1
log_mel_spectrogram/Log_1Log$log_mel_spectrogram/Const_1:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Log_1З
log_mel_spectrogram/truedivRealDivlog_mel_spectrogram/Log:y:0log_mel_spectrogram/Log_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/truediv{
log_mel_spectrogram/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/mul/xД
log_mel_spectrogram/mulMul"log_mel_spectrogram/mul/x:output:0log_mel_spectrogram/truediv:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/mul
log_mel_spectrogram/Maximum_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2!
log_mel_spectrogram/Maximum_1/xЖ
log_mel_spectrogram/Maximum_1Maximum(log_mel_spectrogram/Maximum_1/x:output:0 log_mel_spectrogram/Max:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Maximum_1
log_mel_spectrogram/Log_2Log!log_mel_spectrogram/Maximum_1:z:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Log_2
log_mel_spectrogram/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/Const_2
log_mel_spectrogram/Log_3Log$log_mel_spectrogram/Const_2:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Log_3Ј
log_mel_spectrogram/truediv_1RealDivlog_mel_spectrogram/Log_2:y:0log_mel_spectrogram/Log_3:y:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/truediv_1
log_mel_spectrogram/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/mul_1/xЇ
log_mel_spectrogram/mul_1Mul$log_mel_spectrogram/mul_1/x:output:0!log_mel_spectrogram/truediv_1:z:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/mul_1Ћ
log_mel_spectrogram/subSublog_mel_spectrogram/mul:z:0log_mel_spectrogram/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/sub
log_mel_spectrogram/Const_3Const*
_output_shapes
:*
dtype0*!
valueB"          2
log_mel_spectrogram/Const_3Ё
log_mel_spectrogram/Max_1Maxlog_mel_spectrogram/sub:z:0$log_mel_spectrogram/Const_3:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Max_1
log_mel_spectrogram/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B2
log_mel_spectrogram/sub_1/yЈ
log_mel_spectrogram/sub_1Sub"log_mel_spectrogram/Max_1:output:0$log_mel_spectrogram/sub_1/y:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/sub_1Л
log_mel_spectrogram/Maximum_2Maximumlog_mel_spectrogram/sub:z:0log_mel_spectrogram/sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Maximum_2
"log_mel_spectrogram/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"log_mel_spectrogram/ExpandDims/dimи
log_mel_spectrogram/ExpandDims
ExpandDims!log_mel_spectrogram/Maximum_2:z:0+log_mel_spectrogram/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2 
log_mel_spectrogram/ExpandDimsy
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm
	transpose	Transpose'log_mel_spectrogram/ExpandDims:output:0transpose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
	transpose
)mfccs_from_log_mel_spectrograms/dct/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)mfccs_from_log_mel_spectrograms/dct/Const
*mfccs_from_log_mel_spectrograms/dct/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*mfccs_from_log_mel_spectrograms/dct/Cast/xС
(mfccs_from_log_mel_spectrograms/dct/CastCast3mfccs_from_log_mel_spectrograms/dct/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(mfccs_from_log_mel_spectrograms/dct/CastЄ
/mfccs_from_log_mel_spectrograms/dct/range/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/mfccs_from_log_mel_spectrograms/dct/range/startЄ
/mfccs_from_log_mel_spectrograms/dct/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/mfccs_from_log_mel_spectrograms/dct/range/deltaв
.mfccs_from_log_mel_spectrograms/dct/range/CastCast8mfccs_from_log_mel_spectrograms/dct/range/start:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.mfccs_from_log_mel_spectrograms/dct/range/Castж
0mfccs_from_log_mel_spectrograms/dct/range/Cast_1Cast8mfccs_from_log_mel_spectrograms/dct/range/delta:output:0*

DstT0*

SrcT0*
_output_shapes
: 22
0mfccs_from_log_mel_spectrograms/dct/range/Cast_1
)mfccs_from_log_mel_spectrograms/dct/rangeRange2mfccs_from_log_mel_spectrograms/dct/range/Cast:y:0,mfccs_from_log_mel_spectrograms/dct/Cast:y:04mfccs_from_log_mel_spectrograms/dct/range/Cast_1:y:0*

Tidx0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/rangeВ
'mfccs_from_log_mel_spectrograms/dct/NegNeg2mfccs_from_log_mel_spectrograms/dct/range:output:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/Neg
)mfccs_from_log_mel_spectrograms/dct/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *лI@2+
)mfccs_from_log_mel_spectrograms/dct/mul/yп
'mfccs_from_log_mel_spectrograms/dct/mulMul+mfccs_from_log_mel_spectrograms/dct/Neg:y:02mfccs_from_log_mel_spectrograms/dct/mul/y:output:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/mul
+mfccs_from_log_mel_spectrograms/dct/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2-
+mfccs_from_log_mel_spectrograms/dct/mul_1/yх
)mfccs_from_log_mel_spectrograms/dct/mul_1Mul+mfccs_from_log_mel_spectrograms/dct/mul:z:04mfccs_from_log_mel_spectrograms/dct/mul_1/y:output:0*
T0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/mul_1ч
+mfccs_from_log_mel_spectrograms/dct/truedivRealDiv-mfccs_from_log_mel_spectrograms/dct/mul_1:z:0,mfccs_from_log_mel_spectrograms/dct/Cast:y:0*
T0*
_output_shapes
:2-
+mfccs_from_log_mel_spectrograms/dct/truedivц
+mfccs_from_log_mel_spectrograms/dct/ComplexComplex2mfccs_from_log_mel_spectrograms/dct/Const:output:0/mfccs_from_log_mel_spectrograms/dct/truediv:z:0*
_output_shapes
:2-
+mfccs_from_log_mel_spectrograms/dct/ComplexБ
'mfccs_from_log_mel_spectrograms/dct/ExpExp1mfccs_from_log_mel_spectrograms/dct/Complex:out:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/ExpЃ
+mfccs_from_log_mel_spectrograms/dct/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J   @    2-
+mfccs_from_log_mel_spectrograms/dct/mul_2/xх
)mfccs_from_log_mel_spectrograms/dct/mul_2Mul4mfccs_from_log_mel_spectrograms/dct/mul_2/x:output:0+mfccs_from_log_mel_spectrograms/dct/Exp:y:0*
T0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/mul_2Њ
.mfccs_from_log_mel_spectrograms/dct/rfft/ConstConst*
_output_shapes
:*
dtype0*
valueB:
20
.mfccs_from_log_mel_spectrograms/dct/rfft/Constп
5mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                27
5mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddingsь
,mfccs_from_log_mel_spectrograms/dct/rfft/PadPadtranspose:y:0>mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
2.
,mfccs_from_log_mel_spectrograms/dct/rfft/PadД
3mfccs_from_log_mel_spectrograms/dct/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:
25
3mfccs_from_log_mel_spectrograms/dct/rfft/fft_length
(mfccs_from_log_mel_spectrograms/dct/rfftRFFT5mfccs_from_log_mel_spectrograms/dct/rfft/Pad:output:0<mfccs_from_log_mel_spectrograms/dct/rfft/fft_length:output:0*/
_output_shapes
:џџџџџџџџџ@2*
(mfccs_from_log_mel_spectrograms/dct/rfftУ
7mfccs_from_log_mel_spectrograms/dct/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7mfccs_from_log_mel_spectrograms/dct/strided_slice/stackЧ
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1Ч
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2с
1mfccs_from_log_mel_spectrograms/dct/strided_sliceStridedSlice1mfccs_from_log_mel_spectrograms/dct/rfft:output:0@mfccs_from_log_mel_spectrograms/dct/strided_slice/stack:output:0Bmfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1:output:0Bmfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

begin_mask*
ellipsis_mask23
1mfccs_from_log_mel_spectrograms/dct/strided_slice
)mfccs_from_log_mel_spectrograms/dct/mul_3Mul:mfccs_from_log_mel_spectrograms/dct/strided_slice:output:0-mfccs_from_log_mel_spectrograms/dct/mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2+
)mfccs_from_log_mel_spectrograms/dct/mul_3М
(mfccs_from_log_mel_spectrograms/dct/RealReal-mfccs_from_log_mel_spectrograms/dct/mul_3:z:0*/
_output_shapes
:џџџџџџџџџ@2*
(mfccs_from_log_mel_spectrograms/dct/Real
&mfccs_from_log_mel_spectrograms/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2(
&mfccs_from_log_mel_spectrograms/Cast/xЕ
$mfccs_from_log_mel_spectrograms/CastCast/mfccs_from_log_mel_spectrograms/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$mfccs_from_log_mel_spectrograms/Cast
%mfccs_from_log_mel_spectrograms/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%mfccs_from_log_mel_spectrograms/mul/yЬ
#mfccs_from_log_mel_spectrograms/mulMul(mfccs_from_log_mel_spectrograms/Cast:y:0.mfccs_from_log_mel_spectrograms/mul/y:output:0*
T0*
_output_shapes
: 2%
#mfccs_from_log_mel_spectrograms/mulЁ
%mfccs_from_log_mel_spectrograms/RsqrtRsqrt'mfccs_from_log_mel_spectrograms/mul:z:0*
T0*
_output_shapes
: 2'
%mfccs_from_log_mel_spectrograms/Rsqrtэ
%mfccs_from_log_mel_spectrograms/mul_1Mul1mfccs_from_log_mel_spectrograms/dct/Real:output:0)mfccs_from_log_mel_spectrograms/Rsqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2'
%mfccs_from_log_mel_spectrograms/mul_1c
SquareSquaretranspose:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
	keep_dims(2
Sum\
SqrtSqrtSum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Sqrt
delta/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose/permГ
delta/transpose	Transpose)mfccs_from_log_mel_spectrograms/mul_1:z:0delta/transpose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose
delta/ConstConst*
_output_shapes

:*
dtype0*9
value0B."                               2
delta/ConstЉ
delta/MirrorPad	MirrorPaddelta/transpose:y:0delta/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
mode	SYMMETRIC2
delta/MirrorPads
delta/arange/startConst*
_output_shapes
: *
dtype0*
valueB :
ќџџџџџџџџ2
delta/arange/startj
delta/arange/limitConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange/limitj
delta/arange/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange/delta
delta/arangeRangedelta/arange/start:output:0delta/arange/limit:output:0delta/arange/delta:output:0*
_output_shapes
:	2
delta/arangek

delta/CastCastdelta/arange:output:0*

DstT0*

SrcT0*
_output_shapes
:	2

delta/Cast
delta/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
delta/Reshape/shape
delta/ReshapeReshapedelta/Cast:y:0delta/Reshape/shape:output:0*
T0*&
_output_shapes
:	2
delta/ReshapeХ
delta/convolutionConv2Ddelta/MirrorPad:output:0delta/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
delta/convolutiong
delta/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pB2
delta/truediv/y
delta/truedivRealDivdelta/convolution:output:0delta/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/truediv
delta/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose_1/permЁ
delta/transpose_1	Transposedelta/truediv:z:0delta/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose_1
delta/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose_2/permЅ
delta/transpose_2	Transposedelta/transpose_1:y:0delta/transpose_2/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose_2
delta/Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                               2
delta/Const_1Б
delta/MirrorPad_1	MirrorPaddelta/transpose_2:y:0delta/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
mode	SYMMETRIC2
delta/MirrorPad_1w
delta/arange_1/startConst*
_output_shapes
: *
dtype0*
valueB :
ќџџџџџџџџ2
delta/arange_1/startn
delta/arange_1/limitConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange_1/limitn
delta/arange_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange_1/deltaЂ
delta/arange_1Rangedelta/arange_1/start:output:0delta/arange_1/limit:output:0delta/arange_1/delta:output:0*
_output_shapes
:	2
delta/arange_1q
delta/Cast_1Castdelta/arange_1:output:0*

DstT0*

SrcT0*
_output_shapes
:	2
delta/Cast_1
delta/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
delta/Reshape_1/shape
delta/Reshape_1Reshapedelta/Cast_1:y:0delta/Reshape_1/shape:output:0*
T0*&
_output_shapes
:	2
delta/Reshape_1Э
delta/convolution_1Conv2Ddelta/MirrorPad_1:output:0delta/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
delta/convolution_1k
delta/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pB2
delta/truediv_1/yЁ
delta/truediv_1RealDivdelta/convolution_1:output:0delta/truediv_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/truediv_1
delta/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose_3/permЃ
delta/transpose_3	Transposedelta/truediv_1:z:0delta/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisи
concatConcatV2)mfccs_from_log_mel_spectrograms/mul_1:z:0delta/transpose_1:y:0delta/transpose_3:y:0Sqrt:y:0concat/axis:output:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@2
concat
	Squeeze_1Squeezeconcat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@*
squeeze_dims
2
	Squeeze_1
(resnet_block/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(resnet_block/conv1/conv1d/ExpandDims/dimл
$resnet_block/conv1/conv1d/ExpandDims
ExpandDimsSqueeze_1:output:01resnet_block/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2&
$resnet_block/conv1/conv1d/ExpandDimsё
5resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>resnet_block_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOp
*resnet_block/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*resnet_block/conv1/conv1d/ExpandDims_1/dim
&resnet_block/conv1/conv1d/ExpandDims_1
ExpandDims=resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:03resnet_block/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2(
&resnet_block/conv1/conv1d/ExpandDims_1
resnet_block/conv1/conv1dConv2D-resnet_block/conv1/conv1d/ExpandDims:output:0/resnet_block/conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/conv1/conv1dЫ
!resnet_block/conv1/conv1d/SqueezeSqueeze"resnet_block/conv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2#
!resnet_block/conv1/conv1d/SqueezeХ
)resnet_block/conv1/BiasAdd/ReadVariableOpReadVariableOp2resnet_block_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)resnet_block/conv1/BiasAdd/ReadVariableOpи
resnet_block/conv1/BiasAddBiasAdd*resnet_block/conv1/conv1d/Squeeze:output:01resnet_block/conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/conv1/BiasAdd
resnet_block/relu1/ReluRelu#resnet_block/conv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/relu1/Relu
(resnet_block/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(resnet_block/conv2/conv1d/ExpandDims/dimю
$resnet_block/conv2/conv1d/ExpandDims
ExpandDims%resnet_block/relu1/Relu:activations:01resnet_block/conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2&
$resnet_block/conv2/conv1d/ExpandDimsё
5resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>resnet_block_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype027
5resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOp
*resnet_block/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*resnet_block/conv2/conv1d/ExpandDims_1/dim
&resnet_block/conv2/conv1d/ExpandDims_1
ExpandDims=resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:03resnet_block/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2(
&resnet_block/conv2/conv1d/ExpandDims_1
resnet_block/conv2/conv1dConv2D-resnet_block/conv2/conv1d/ExpandDims:output:0/resnet_block/conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/conv2/conv1dЫ
!resnet_block/conv2/conv1d/SqueezeSqueeze"resnet_block/conv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2#
!resnet_block/conv2/conv1d/SqueezeХ
)resnet_block/conv2/BiasAdd/ReadVariableOpReadVariableOp2resnet_block_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)resnet_block/conv2/BiasAdd/ReadVariableOpи
resnet_block/conv2/BiasAddBiasAdd*resnet_block/conv2/conv1d/Squeeze:output:01resnet_block/conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/conv2/BiasAdd
resnet_block/relu2/ReluRelu#resnet_block/conv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/relu2/Relu
(resnet_block/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(resnet_block/conv3/conv1d/ExpandDims/dimю
$resnet_block/conv3/conv1d/ExpandDims
ExpandDims%resnet_block/relu2/Relu:activations:01resnet_block/conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2&
$resnet_block/conv3/conv1d/ExpandDimsё
5resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>resnet_block_conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype027
5resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOp
*resnet_block/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*resnet_block/conv3/conv1d/ExpandDims_1/dim
&resnet_block/conv3/conv1d/ExpandDims_1
ExpandDims=resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:03resnet_block/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2(
&resnet_block/conv3/conv1d/ExpandDims_1
resnet_block/conv3/conv1dConv2D-resnet_block/conv3/conv1d/ExpandDims:output:0/resnet_block/conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/conv3/conv1dЫ
!resnet_block/conv3/conv1d/SqueezeSqueeze"resnet_block/conv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2#
!resnet_block/conv3/conv1d/SqueezeХ
)resnet_block/conv3/BiasAdd/ReadVariableOpReadVariableOp2resnet_block_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)resnet_block/conv3/BiasAdd/ReadVariableOpи
resnet_block/conv3/BiasAddBiasAdd*resnet_block/conv3/conv1d/Squeeze:output:01resnet_block/conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/conv3/BiasAddЅ
+resnet_block/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2-
+resnet_block/shortcut/conv1d/ExpandDims/dimф
'resnet_block/shortcut/conv1d/ExpandDims
ExpandDimsSqueeze_1:output:04resnet_block/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2)
'resnet_block/shortcut/conv1d/ExpandDimsњ
8resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAresnet_block_shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02:
8resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOp 
-resnet_block/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-resnet_block/shortcut/conv1d/ExpandDims_1/dim
)resnet_block/shortcut/conv1d/ExpandDims_1
ExpandDims@resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:06resnet_block/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2+
)resnet_block/shortcut/conv1d/ExpandDims_1
resnet_block/shortcut/conv1dConv2D0resnet_block/shortcut/conv1d/ExpandDims:output:02resnet_block/shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/shortcut/conv1dд
$resnet_block/shortcut/conv1d/SqueezeSqueeze%resnet_block/shortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2&
$resnet_block/shortcut/conv1d/SqueezeЮ
,resnet_block/shortcut/BiasAdd/ReadVariableOpReadVariableOp5resnet_block_shortcut_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,resnet_block/shortcut/BiasAdd/ReadVariableOpф
resnet_block/shortcut/BiasAddBiasAdd-resnet_block/shortcut/conv1d/Squeeze:output:04resnet_block/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/shortcut/BiasAddИ
resnet_block/add/addAddV2#resnet_block/conv3/BiasAdd:output:0&resnet_block/shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/add/add
resnet_block/out_block/ReluReluresnet_block/add/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/out_block/ReluЃ
*resnet_block_1/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_1/conv1/conv1d/ExpandDims/dimј
&resnet_block_1/conv1/conv1d/ExpandDims
ExpandDims)resnet_block/out_block/Relu:activations:03resnet_block_1/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2(
&resnet_block_1/conv1/conv1d/ExpandDimsї
7resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_1_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype029
7resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_1/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_1/conv1/conv1d/ExpandDims_1/dim
(resnet_block_1/conv1/conv1d/ExpandDims_1
ExpandDims?resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_1/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2*
(resnet_block_1/conv1/conv1d/ExpandDims_1
resnet_block_1/conv1/conv1dConv2D/resnet_block_1/conv1/conv1d/ExpandDims:output:01resnet_block_1/conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
resnet_block_1/conv1/conv1dб
#resnet_block_1/conv1/conv1d/SqueezeSqueeze$resnet_block_1/conv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_1/conv1/conv1d/SqueezeЫ
+resnet_block_1/conv1/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+resnet_block_1/conv1/BiasAdd/ReadVariableOpр
resnet_block_1/conv1/BiasAddBiasAdd,resnet_block_1/conv1/conv1d/Squeeze:output:03resnet_block_1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/conv1/BiasAdd
resnet_block_1/relu1/ReluRelu%resnet_block_1/conv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/relu1/ReluЃ
*resnet_block_1/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_1/conv2/conv1d/ExpandDims/dimі
&resnet_block_1/conv2/conv1d/ExpandDims
ExpandDims'resnet_block_1/relu1/Relu:activations:03resnet_block_1/conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2(
&resnet_block_1/conv2/conv1d/ExpandDimsї
7resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_1_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_1/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_1/conv2/conv1d/ExpandDims_1/dim
(resnet_block_1/conv2/conv1d/ExpandDims_1
ExpandDims?resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_1/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2*
(resnet_block_1/conv2/conv1d/ExpandDims_1
resnet_block_1/conv2/conv1dConv2D/resnet_block_1/conv2/conv1d/ExpandDims:output:01resnet_block_1/conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
resnet_block_1/conv2/conv1dб
#resnet_block_1/conv2/conv1d/SqueezeSqueeze$resnet_block_1/conv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_1/conv2/conv1d/SqueezeЫ
+resnet_block_1/conv2/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+resnet_block_1/conv2/BiasAdd/ReadVariableOpр
resnet_block_1/conv2/BiasAddBiasAdd,resnet_block_1/conv2/conv1d/Squeeze:output:03resnet_block_1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/conv2/BiasAdd
resnet_block_1/relu2/ReluRelu%resnet_block_1/conv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/relu2/ReluЃ
*resnet_block_1/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_1/conv3/conv1d/ExpandDims/dimі
&resnet_block_1/conv3/conv1d/ExpandDims
ExpandDims'resnet_block_1/relu2/Relu:activations:03resnet_block_1/conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2(
&resnet_block_1/conv3/conv1d/ExpandDimsї
7resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_1_conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_1/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_1/conv3/conv1d/ExpandDims_1/dim
(resnet_block_1/conv3/conv1d/ExpandDims_1
ExpandDims?resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_1/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2*
(resnet_block_1/conv3/conv1d/ExpandDims_1
resnet_block_1/conv3/conv1dConv2D/resnet_block_1/conv3/conv1d/ExpandDims:output:01resnet_block_1/conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
resnet_block_1/conv3/conv1dб
#resnet_block_1/conv3/conv1d/SqueezeSqueeze$resnet_block_1/conv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_1/conv3/conv1d/SqueezeЫ
+resnet_block_1/conv3/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_1_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+resnet_block_1/conv3/BiasAdd/ReadVariableOpр
resnet_block_1/conv3/BiasAddBiasAdd,resnet_block_1/conv3/conv1d/Squeeze:output:03resnet_block_1/conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/conv3/BiasAddЉ
-resnet_block_1/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-resnet_block_1/shortcut/conv1d/ExpandDims/dim
)resnet_block_1/shortcut/conv1d/ExpandDims
ExpandDims)resnet_block/out_block/Relu:activations:06resnet_block_1/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2+
)resnet_block_1/shortcut/conv1d/ExpandDims
:resnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCresnet_block_1_shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02<
:resnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOpЄ
/resnet_block_1/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/resnet_block_1/shortcut/conv1d/ExpandDims_1/dim
+resnet_block_1/shortcut/conv1d/ExpandDims_1
ExpandDimsBresnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:08resnet_block_1/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2-
+resnet_block_1/shortcut/conv1d/ExpandDims_1
resnet_block_1/shortcut/conv1dConv2D2resnet_block_1/shortcut/conv1d/ExpandDims:output:04resnet_block_1/shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2 
resnet_block_1/shortcut/conv1dк
&resnet_block_1/shortcut/conv1d/SqueezeSqueeze'resnet_block_1/shortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2(
&resnet_block_1/shortcut/conv1d/Squeezeд
.resnet_block_1/shortcut/BiasAdd/ReadVariableOpReadVariableOp7resnet_block_1_shortcut_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.resnet_block_1/shortcut/BiasAdd/ReadVariableOpь
resnet_block_1/shortcut/BiasAddBiasAdd/resnet_block_1/shortcut/conv1d/Squeeze:output:06resnet_block_1/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2!
resnet_block_1/shortcut/BiasAddФ
resnet_block_1/add_1/addAddV2%resnet_block_1/conv3/BiasAdd:output:0(resnet_block_1/shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/add_1/add
resnet_block_1/out_block/ReluReluresnet_block_1/add_1/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/out_block/ReluЃ
*resnet_block_2/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_2/conv1/conv1d/ExpandDims/dimњ
&resnet_block_2/conv1/conv1d/ExpandDims
ExpandDims+resnet_block_1/out_block/Relu:activations:03resnet_block_2/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2(
&resnet_block_2/conv1/conv1d/ExpandDimsј
7resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_2_conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype029
7resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_2/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_2/conv1/conv1d/ExpandDims_1/dim
(resnet_block_2/conv1/conv1d/ExpandDims_1
ExpandDims?resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_2/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2*
(resnet_block_2/conv1/conv1d/ExpandDims_1
resnet_block_2/conv1/conv1dConv2D/resnet_block_2/conv1/conv1d/ExpandDims:output:01resnet_block_2/conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_2/conv1/conv1dв
#resnet_block_2/conv1/conv1d/SqueezeSqueeze$resnet_block_2/conv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_2/conv1/conv1d/SqueezeЬ
+resnet_block_2/conv1/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_2/conv1/BiasAdd/ReadVariableOpс
resnet_block_2/conv1/BiasAddBiasAdd,resnet_block_2/conv1/conv1d/Squeeze:output:03resnet_block_2/conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/conv1/BiasAdd
resnet_block_2/relu1/ReluRelu%resnet_block_2/conv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/relu1/ReluЃ
*resnet_block_2/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_2/conv2/conv1d/ExpandDims/dimї
&resnet_block_2/conv2/conv1d/ExpandDims
ExpandDims'resnet_block_2/relu1/Relu:activations:03resnet_block_2/conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_2/conv2/conv1d/ExpandDimsљ
7resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_2_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_2/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_2/conv2/conv1d/ExpandDims_1/dim
(resnet_block_2/conv2/conv1d/ExpandDims_1
ExpandDims?resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_2/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_2/conv2/conv1d/ExpandDims_1
resnet_block_2/conv2/conv1dConv2D/resnet_block_2/conv2/conv1d/ExpandDims:output:01resnet_block_2/conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_2/conv2/conv1dв
#resnet_block_2/conv2/conv1d/SqueezeSqueeze$resnet_block_2/conv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_2/conv2/conv1d/SqueezeЬ
+resnet_block_2/conv2/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_2/conv2/BiasAdd/ReadVariableOpс
resnet_block_2/conv2/BiasAddBiasAdd,resnet_block_2/conv2/conv1d/Squeeze:output:03resnet_block_2/conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/conv2/BiasAdd
resnet_block_2/relu2/ReluRelu%resnet_block_2/conv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/relu2/ReluЃ
*resnet_block_2/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_2/conv3/conv1d/ExpandDims/dimї
&resnet_block_2/conv3/conv1d/ExpandDims
ExpandDims'resnet_block_2/relu2/Relu:activations:03resnet_block_2/conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_2/conv3/conv1d/ExpandDimsљ
7resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_2_conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_2/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_2/conv3/conv1d/ExpandDims_1/dim
(resnet_block_2/conv3/conv1d/ExpandDims_1
ExpandDims?resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_2/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_2/conv3/conv1d/ExpandDims_1
resnet_block_2/conv3/conv1dConv2D/resnet_block_2/conv3/conv1d/ExpandDims:output:01resnet_block_2/conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_2/conv3/conv1dв
#resnet_block_2/conv3/conv1d/SqueezeSqueeze$resnet_block_2/conv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_2/conv3/conv1d/SqueezeЬ
+resnet_block_2/conv3/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_2_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_2/conv3/BiasAdd/ReadVariableOpс
resnet_block_2/conv3/BiasAddBiasAdd,resnet_block_2/conv3/conv1d/Squeeze:output:03resnet_block_2/conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/conv3/BiasAddЉ
-resnet_block_2/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-resnet_block_2/shortcut/conv1d/ExpandDims/dim
)resnet_block_2/shortcut/conv1d/ExpandDims
ExpandDims+resnet_block_1/out_block/Relu:activations:06resnet_block_2/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2+
)resnet_block_2/shortcut/conv1d/ExpandDims
:resnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCresnet_block_2_shortcut_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02<
:resnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOpЄ
/resnet_block_2/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/resnet_block_2/shortcut/conv1d/ExpandDims_1/dim
+resnet_block_2/shortcut/conv1d/ExpandDims_1
ExpandDimsBresnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:08resnet_block_2/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2-
+resnet_block_2/shortcut/conv1d/ExpandDims_1
resnet_block_2/shortcut/conv1dConv2D2resnet_block_2/shortcut/conv1d/ExpandDims:output:04resnet_block_2/shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2 
resnet_block_2/shortcut/conv1dл
&resnet_block_2/shortcut/conv1d/SqueezeSqueeze'resnet_block_2/shortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2(
&resnet_block_2/shortcut/conv1d/Squeezeе
.resnet_block_2/shortcut/BiasAdd/ReadVariableOpReadVariableOp7resnet_block_2_shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.resnet_block_2/shortcut/BiasAdd/ReadVariableOpэ
resnet_block_2/shortcut/BiasAddBiasAdd/resnet_block_2/shortcut/conv1d/Squeeze:output:06resnet_block_2/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2!
resnet_block_2/shortcut/BiasAddХ
resnet_block_2/add_2/addAddV2%resnet_block_2/conv3/BiasAdd:output:0(resnet_block_2/shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/add_2/add
resnet_block_2/out_block/ReluReluresnet_block_2/add_2/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/out_block/ReluЃ
*resnet_block_3/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_3/conv1/conv1d/ExpandDims/dimћ
&resnet_block_3/conv1/conv1d/ExpandDims
ExpandDims+resnet_block_2/out_block/Relu:activations:03resnet_block_3/conv1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_3/conv1/conv1d/ExpandDimsљ
7resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_3_conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_3/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_3/conv1/conv1d/ExpandDims_1/dim
(resnet_block_3/conv1/conv1d/ExpandDims_1
ExpandDims?resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_3/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_3/conv1/conv1d/ExpandDims_1
resnet_block_3/conv1/conv1dConv2D/resnet_block_3/conv1/conv1d/ExpandDims:output:01resnet_block_3/conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_3/conv1/conv1dв
#resnet_block_3/conv1/conv1d/SqueezeSqueeze$resnet_block_3/conv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_3/conv1/conv1d/SqueezeЬ
+resnet_block_3/conv1/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_3/conv1/BiasAdd/ReadVariableOpс
resnet_block_3/conv1/BiasAddBiasAdd,resnet_block_3/conv1/conv1d/Squeeze:output:03resnet_block_3/conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/conv1/BiasAdd
resnet_block_3/relu1/ReluRelu%resnet_block_3/conv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/relu1/ReluЃ
*resnet_block_3/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_3/conv2/conv1d/ExpandDims/dimї
&resnet_block_3/conv2/conv1d/ExpandDims
ExpandDims'resnet_block_3/relu1/Relu:activations:03resnet_block_3/conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_3/conv2/conv1d/ExpandDimsљ
7resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_3_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_3/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_3/conv2/conv1d/ExpandDims_1/dim
(resnet_block_3/conv2/conv1d/ExpandDims_1
ExpandDims?resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_3/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_3/conv2/conv1d/ExpandDims_1
resnet_block_3/conv2/conv1dConv2D/resnet_block_3/conv2/conv1d/ExpandDims:output:01resnet_block_3/conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_3/conv2/conv1dв
#resnet_block_3/conv2/conv1d/SqueezeSqueeze$resnet_block_3/conv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_3/conv2/conv1d/SqueezeЬ
+resnet_block_3/conv2/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_3/conv2/BiasAdd/ReadVariableOpс
resnet_block_3/conv2/BiasAddBiasAdd,resnet_block_3/conv2/conv1d/Squeeze:output:03resnet_block_3/conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/conv2/BiasAdd
resnet_block_3/relu2/ReluRelu%resnet_block_3/conv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/relu2/ReluЃ
*resnet_block_3/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_3/conv3/conv1d/ExpandDims/dimї
&resnet_block_3/conv3/conv1d/ExpandDims
ExpandDims'resnet_block_3/relu2/Relu:activations:03resnet_block_3/conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_3/conv3/conv1d/ExpandDimsљ
7resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_3_conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_3/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_3/conv3/conv1d/ExpandDims_1/dim
(resnet_block_3/conv3/conv1d/ExpandDims_1
ExpandDims?resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_3/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_3/conv3/conv1d/ExpandDims_1
resnet_block_3/conv3/conv1dConv2D/resnet_block_3/conv3/conv1d/ExpandDims:output:01resnet_block_3/conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_3/conv3/conv1dв
#resnet_block_3/conv3/conv1d/SqueezeSqueeze$resnet_block_3/conv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_3/conv3/conv1d/SqueezeЬ
+resnet_block_3/conv3/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_3/conv3/BiasAdd/ReadVariableOpс
resnet_block_3/conv3/BiasAddBiasAdd,resnet_block_3/conv3/conv1d/Squeeze:output:03resnet_block_3/conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/conv3/BiasAddЉ
-resnet_block_3/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-resnet_block_3/shortcut/conv1d/ExpandDims/dim
)resnet_block_3/shortcut/conv1d/ExpandDims
ExpandDims+resnet_block_2/out_block/Relu:activations:06resnet_block_3/shortcut/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2+
)resnet_block_3/shortcut/conv1d/ExpandDims
:resnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCresnet_block_3_shortcut_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02<
:resnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOpЄ
/resnet_block_3/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/resnet_block_3/shortcut/conv1d/ExpandDims_1/dim
+resnet_block_3/shortcut/conv1d/ExpandDims_1
ExpandDimsBresnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:08resnet_block_3/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2-
+resnet_block_3/shortcut/conv1d/ExpandDims_1
resnet_block_3/shortcut/conv1dConv2D2resnet_block_3/shortcut/conv1d/ExpandDims:output:04resnet_block_3/shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2 
resnet_block_3/shortcut/conv1dл
&resnet_block_3/shortcut/conv1d/SqueezeSqueeze'resnet_block_3/shortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2(
&resnet_block_3/shortcut/conv1d/Squeezeе
.resnet_block_3/shortcut/BiasAdd/ReadVariableOpReadVariableOp7resnet_block_3_shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.resnet_block_3/shortcut/BiasAdd/ReadVariableOpэ
resnet_block_3/shortcut/BiasAddBiasAdd/resnet_block_3/shortcut/conv1d/Squeeze:output:06resnet_block_3/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2!
resnet_block_3/shortcut/BiasAddХ
resnet_block_3/add_3/addAddV2%resnet_block_3/conv3/BiasAdd:output:0(resnet_block_3/shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/add_3/add
resnet_block_3/out_block/ReluReluresnet_block_3/add_3/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/out_block/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
flatten/ConstЅ
flatten/ReshapeReshape+resnet_block_3/out_block/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ@2
flatten/Reshape
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype02
fc1/MatMul/ReadVariableOp

fc1/MatMulMatMulflatten/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc1/MatMul
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc1/BiasAdd/ReadVariableOp
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
fc1/BiasAdde
fc1/ReluRelufc1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc1/Relu
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
fc2/MatMul/ReadVariableOp

fc2/MatMulMatMulfc1/Relu:activations:0!fc2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc2/MatMul
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc2/BiasAdd/ReadVariableOp
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
fc2/BiasAdde
fc2/ReluRelufc2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc2/Relu
fc3/MatMul/ReadVariableOpReadVariableOp"fc3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
fc3/MatMul/ReadVariableOp

fc3/MatMulMatMulfc2/Relu:activations:0!fc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2

fc3/MatMul
fc3/BiasAdd/ReadVariableOpReadVariableOp#fc3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc3/BiasAdd/ReadVariableOp
fc3/BiasAddBiasAddfc3/MatMul:product:0"fc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fc3/BiasAddm
fc3/SigmoidSigmoidfc3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fc3/Sigmoidc
IdentityIdentityfc3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:џџџџџџџџџ:	:::::::::::::::::::::::::::::::::::::::U Q
,
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:%!

_output_shapes
:	
Т
Ж
A__inference_conv2_layer_call_and_return_conditional_losses_211614

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsК
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЙ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@:::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Љ
Ї
?__inference_fc3_layer_call_and_return_conditional_losses_217281

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц
{
&__inference_conv3_layer_call_fn_217382

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2102792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@ ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Ж
Ж
A__inference_conv3_layer_call_and_return_conditional_losses_217373

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@ :::S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Й
Й
D__inference_shortcut_layer_call_and_return_conditional_losses_217397

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@:::S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ј

J__inference_resnet_block_1_layer_call_and_return_conditional_losses_210942

inputs
conv1_210917
conv1_210919
conv2_210923
conv2_210925
conv3_210929
conv3_210931
shortcut_210934
shortcut_210936
identityЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallЂconv3/StatefulPartitionedCallЂ shortcut/StatefulPartitionedCall
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_210917conv1_210919*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2105942
conv1/StatefulPartitionedCallя
relu1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu1_layer_call_and_return_conditional_losses_2106292
relu1/PartitionedCallЁ
conv2/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0conv2_210923conv2_210925*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2106722
conv2/StatefulPartitionedCallя
relu2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu2_layer_call_and_return_conditional_losses_2107072
relu2/PartitionedCallЁ
conv3/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0conv3_210929conv3_210931*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2107502
conv3/StatefulPartitionedCall
 shortcut/StatefulPartitionedCallStatefulPartitionedCallinputsshortcut_210934shortcut_210936*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_shortcut_layer_call_and_return_conditional_losses_2108052"
 shortcut/StatefulPartitionedCallЄ
add/addAddV2&conv3/StatefulPartitionedCall:output:0)shortcut/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
add/addр
out_block/PartitionedCallPartitionedCalladd/add:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_out_block_layer_call_and_return_conditional_losses_2108412
out_block/PartitionedCall§
IdentityIdentity"out_block/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall!^shortcut/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@ ::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2D
 shortcut/StatefulPartitionedCall shortcut/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Ж
Ж
A__inference_conv1_layer_call_and_return_conditional_losses_217431

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@ :::S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Р
Й
D__inference_shortcut_layer_call_and_return_conditional_losses_217649

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv1d/ExpandDimsЙ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimИ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@@:::S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Т
Ж
A__inference_conv2_layer_call_and_return_conditional_losses_211143

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsК
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЙ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@:::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
П
]
A__inference_relu2_layer_call_and_return_conditional_losses_217479

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџ@@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@@:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
У
]
A__inference_relu2_layer_call_and_return_conditional_losses_211178

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:џџџџџџџџџ@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
­
Ї
?__inference_fc2_layer_call_and_return_conditional_losses_212763

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
]
A__inference_relu2_layer_call_and_return_conditional_losses_210707

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџ@@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@@:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Є
F
*__inference_out_block_layer_call_fn_217416

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_out_block_layer_call_and_return_conditional_losses_2103702
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@ :S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
ж
y
$__inference_fc2_layer_call_fn_217270

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc2_layer_call_and_return_conditional_losses_2127632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
a
E__inference_out_block_layer_call_and_return_conditional_losses_211783

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:џџџџџџџџџ@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
У
a
E__inference_out_block_layer_call_and_return_conditional_losses_210370

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@ :S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs

B
&__inference_relu1_layer_call_fn_217450

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu1_layer_call_and_return_conditional_losses_2106292
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@@:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Т
Ж
A__inference_conv3_layer_call_and_return_conditional_losses_217625

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsК
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЙ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@:::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
П
]
A__inference_relu1_layer_call_and_return_conditional_losses_217445

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџ@@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@@:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ї;
ј
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_216739
input_15
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЉ
conv1/conv1d/ExpandDims
ExpandDimsinput_1$conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv1/conv1d/ExpandDimsЫ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimа
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2
conv1/conv1d/ExpandDims_1Я
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1/conv1dЅ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv1/BiasAdd/ReadVariableOpЅ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv1/BiasAddo

relu1/ReluReluconv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimЛ
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv2/conv1d/ExpandDimsЬ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimб
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv2/conv1d/ExpandDims_1Я
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2/conv1dЅ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2/BiasAdd/ReadVariableOpЅ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv2/BiasAddo

relu2/ReluReluconv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimЛ
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv3/conv1d/ExpandDimsЬ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimб
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv3/conv1d/ExpandDims_1Я
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv3/conv1dЅ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv3/BiasAdd/ReadVariableOpЅ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimВ
shortcut/conv1d/ExpandDims
ExpandDimsinput_1'shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
shortcut/conv1d/ExpandDimsд
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimм
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2
shortcut/conv1d/ExpandDims_1л
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
shortcut/conv1dЎ
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЈ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
shortcut/BiasAdd/ReadVariableOpБ
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
add/addl
out_block/ReluReluadd/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
out_block/Reluu
IdentityIdentityout_block/Relu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@@:::::::::T P
+
_output_shapes
:џџџџџџџџџ@@
!
_user_specified_name	input_1
р­
|
O__inference_log_mel_spectrogram_layer_call_and_return_conditional_losses_216019
	waveforms
matmul_b
identityi
stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2
stft/frame_lengthd
stft/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame_step[

stft/ConstConst*
_output_shapes
: *
dtype0*
value
B :2

stft/Constm
stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
stft/frame/axis]
stft/frame/ShapeShape	waveforms*
T0*
_output_shapes
:2
stft/frame/Shaped
stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Rankr
stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range/startr
stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range/deltaЅ
stft/frame/rangeRangestft/frame/range/start:output:0stft/frame/Rank:output:0stft/frame/range/delta:output:0*
_output_shapes
:2
stft/frame/range
stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2 
stft/frame/strided_slice/stack
 stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 stft/frame/strided_slice/stack_1
 stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 stft/frame/strided_slice/stack_2Є
stft/frame/strided_sliceStridedSlicestft/frame/range:output:0'stft/frame/strided_slice/stack:output:0)stft/frame/strided_slice/stack_1:output:0)stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
stft/frame/strided_slicef
stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/sub/y}
stft/frame/subSubstft/frame/Rank:output:0stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2
stft/frame/sub
stft/frame/sub_1Substft/frame/sub:z:0!stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2
stft/frame/sub_1l
stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/packed/1Г
stft/frame/packedPack!stft/frame/strided_slice:output:0stft/frame/packed/1:output:0stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2
stft/frame/packedz
stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/split/split_dimж
stft/frame/splitSplitVstft/frame/Shape:output:0stft/frame/packed:output:0#stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2
stft/frame/splitw
stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
stft/frame/Reshape/shape{
stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2
stft/frame/Reshape/shape_1
stft/frame/ReshapeReshapestft/frame/split:output:1#stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2
stft/frame/Reshaped
stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Sizeh
stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Size_1i
stft/frame/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
stft/frame/Conste
stft/frame/NegNegstft/frame/Reshape:output:0*
T0*
_output_shapes
: 2
stft/frame/Neg
stft/frame/floordivFloorDivstft/frame/Neg:y:0stft/frame_step:output:0*
T0*
_output_shapes
: 2
stft/frame/floordive
stft/frame/Neg_1Negstft/frame/floordiv:z:0*
T0*
_output_shapes
: 2
stft/frame/Neg_1j
stft/frame/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/sub_2/y
stft/frame/sub_2Substft/frame/Neg_1:y:0stft/frame/sub_2/y:output:0*
T0*
_output_shapes
: 2
stft/frame/sub_2x
stft/frame/mulMulstft/frame_step:output:0stft/frame/sub_2:z:0*
T0*
_output_shapes
: 2
stft/frame/mulz
stft/frame/addAddV2stft/frame_length:output:0stft/frame/mul:z:0*
T0*
_output_shapes
: 2
stft/frame/add}
stft/frame/sub_3Substft/frame/add:z:0stft/frame/Reshape:output:0*
T0*
_output_shapes
: 2
stft/frame/sub_3n
stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Maximum/x
stft/frame/MaximumMaximumstft/frame/Maximum/x:output:0stft/frame/sub_3:z:0*
T0*
_output_shapes
: 2
stft/frame/Maximumr
stft/frame/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/zeros/mul/y
stft/frame/zeros/mulMulstft/frame/Size:output:0stft/frame/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
stft/frame/zeros/mulu
stft/frame/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
stft/frame/zeros/Less/y
stft/frame/zeros/LessLessstft/frame/zeros/mul:z:0 stft/frame/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
stft/frame/zeros/Lessx
stft/frame/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/zeros/packed/1І
stft/frame/zeros/packedPackstft/frame/Size:output:0"stft/frame/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/zeros/packedr
stft/frame/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/zeros/Const
stft/frame/zerosFill stft/frame/zeros/packed:output:0stft/frame/zeros/Const:output:0*
T0*
_output_shapes

:2
stft/frame/zerost
stft/frame/packed_1/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/packed_1/0/0
stft/frame/packed_1/0Pack stft/frame/packed_1/0/0:output:0stft/frame/Maximum:z:0*
N*
T0*
_output_shapes
:2
stft/frame/packed_1/0
stft/frame/packed_1Packstft/frame/packed_1/0:output:0*
N*
T0*
_output_shapes

:2
stft/frame/packed_1v
stft/frame/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/zeros_1/mul/y
stft/frame/zeros_1/mulMulstft/frame/Size_1:output:0!stft/frame/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
stft/frame/zeros_1/muly
stft/frame/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
stft/frame/zeros_1/Less/y
stft/frame/zeros_1/LessLessstft/frame/zeros_1/mul:z:0"stft/frame/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
stft/frame/zeros_1/Less|
stft/frame/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/zeros_1/packed/1Ў
stft/frame/zeros_1/packedPackstft/frame/Size_1:output:0$stft/frame/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/zeros_1/packedv
stft/frame/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/zeros_1/Const
stft/frame/zeros_1Fill"stft/frame/zeros_1/packed:output:0!stft/frame/zeros_1/Const:output:0*
T0*
_output_shapes

: 2
stft/frame/zeros_1r
stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat/axisл
stft/frame/concatConcatV2stft/frame/zeros:output:0stft/frame/packed_1:output:0stft/frame/zeros_1:output:0stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes

:2
stft/frame/concatЊ
stft/frame/PadV2PadV2	waveformsstft/frame/concat:output:0stft/frame/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
stft/frame/PadV2q
stft/frame/Shape_1Shapestft/frame/PadV2:output:0*
T0*
_output_shapes
:2
stft/frame/Shape_1j
stft/frame/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/add_1/y
stft/frame/add_1AddV2!stft/frame/strided_slice:output:0stft/frame/add_1/y:output:0*
T0*
_output_shapes
: 2
stft/frame/add_1
 stft/frame/strided_slice_1/stackPack!stft/frame/strided_slice:output:0*
N*
T0*
_output_shapes
:2"
 stft/frame/strided_slice_1/stack
"stft/frame/strided_slice_1/stack_1Packstft/frame/add_1:z:0*
N*
T0*
_output_shapes
:2$
"stft/frame/strided_slice_1/stack_1
"stft/frame/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"stft/frame/strided_slice_1/stack_2А
stft/frame/strided_slice_1StridedSlicestft/frame/Shape_1:output:0)stft/frame/strided_slice_1/stack:output:0+stft/frame/strided_slice_1/stack_1:output:0+stft/frame/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
stft/frame/strided_slice_1n
stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/gcd/Constt
stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/floordiv_1/y
stft/frame/floordiv_1FloorDivstft/frame_length:output:0 stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_1t
stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/floordiv_2/y
stft/frame/floordiv_2FloorDivstft/frame_step:output:0 stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_2t
stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/floordiv_3/yЂ
stft/frame/floordiv_3FloorDiv#stft/frame/strided_slice_1:output:0 stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_3j
stft/frame/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/mul_1/y
stft/frame/mul_1Mulstft/frame/floordiv_3:z:0stft/frame/mul_1/y:output:0*
T0*
_output_shapes
: 2
stft/frame/mul_1
stft/frame/concat_1/values_1Packstft/frame/mul_1:z:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_1/values_1v
stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat_1/axisф
stft/frame/concat_1ConcatV2stft/frame/split:output:0%stft/frame/concat_1/values_1:output:0stft/frame/split:output:2!stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_1
stft/frame/concat_2/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :2 
stft/frame/concat_2/values_1/1Ж
stft/frame/concat_2/values_1Packstft/frame/floordiv_3:z:0'stft/frame/concat_2/values_1/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_2/values_1v
stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat_2/axisф
stft/frame/concat_2ConcatV2stft/frame/split:output:0%stft/frame/concat_2/values_1:output:0stft/frame/split:output:2!stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_2x
stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2
stft/frame/zeros_like
stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
stft/frame/ones_like/Shapez
stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/ones_like/ConstЃ
stft/frame/ones_likeFill#stft/frame/ones_like/Shape:output:0#stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2
stft/frame/ones_like
stft/frame/StridedSliceStridedSlicestft/frame/PadV2:output:0stft/frame/zeros_like:output:0stft/frame/concat_1:output:0stft/frame/ones_like:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
stft/frame/StridedSliceЖ
stft/frame/Reshape_1Reshape stft/frame/StridedSlice:output:0stft/frame/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
stft/frame/Reshape_1v
stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range_1/startv
stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range_1/deltaВ
stft/frame/range_1Range!stft/frame/range_1/start:output:0stft/frame/Neg_1:y:0!stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2
stft/frame/range_1
stft/frame/mul_2Mulstft/frame/range_1:output:0stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2
stft/frame/mul_2~
stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Reshape_2/shape/1Ћ
stft/frame/Reshape_2/shapePackstft/frame/Neg_1:y:0%stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/Reshape_2/shapeЄ
stft/frame/Reshape_2Reshapestft/frame/mul_2:z:0#stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
stft/frame/Reshape_2v
stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range_2/startv
stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range_2/deltaЎ
stft/frame/range_2Range!stft/frame/range_2/start:output:0stft/frame/floordiv_1:z:0!stft/frame/range_2/delta:output:0*
_output_shapes
: 2
stft/frame/range_2~
stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Reshape_3/shape/0А
stft/frame/Reshape_3/shapePack%stft/frame/Reshape_3/shape/0:output:0stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2
stft/frame/Reshape_3/shapeЂ
stft/frame/Reshape_3Reshapestft/frame/range_2:output:0#stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

: 2
stft/frame/Reshape_3
stft/frame/add_2AddV2stft/frame/Reshape_2:output:0stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
stft/frame/add_2є
stft/frame/GatherV2GatherV2stft/frame/Reshape_1:output:0stft/frame/add_2:z:0!stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 2
stft/frame/GatherV2Є
stft/frame/concat_3/values_1Packstft/frame/Neg_1:y:0stft/frame_length:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_3/values_1v
stft/frame/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat_3/axisф
stft/frame/concat_3ConcatV2stft/frame/split:output:0%stft/frame/concat_3/values_1:output:0stft/frame/split:output:2!stft/frame/concat_3/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_3Њ
stft/frame/Reshape_4Reshapestft/frame/GatherV2:output:0stft/frame/concat_3:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
stft/frame/Reshape_4x
stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
stft/hann_window/periodic
stft/hann_window/CastCast"stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2
stft/hann_window/Cast|
stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/FloorMod/yЅ
stft/hann_window/FloorModFloorModstft/frame_length:output:0$stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2
stft/hann_window/FloorModr
stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/sub/x
stft/hann_window/subSubstft/hann_window/sub/x:output:0stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2
stft/hann_window/sub
stft/hann_window/mulMulstft/hann_window/Cast:y:0stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2
stft/hann_window/mul
stft/hann_window/addAddV2stft/frame_length:output:0stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2
stft/hann_window/addv
stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/sub_1/y
stft/hann_window/sub_1Substft/hann_window/add:z:0!stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2
stft/hann_window/sub_1
stft/hann_window/Cast_1Caststft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
stft/hann_window/Cast_1~
stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/hann_window/range/start~
stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/range/deltaР
stft/hann_window/rangeRange%stft/hann_window/range/start:output:0stft/frame_length:output:0%stft/hann_window/range/delta:output:0*
_output_shapes	
:2
stft/hann_window/range
stft/hann_window/Cast_2Caststft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2
stft/hann_window/Cast_2u
stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2
stft/hann_window/Const
stft/hann_window/mul_1Mulstft/hann_window/Const:output:0stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2
stft/hann_window/mul_1
stft/hann_window/truedivRealDivstft/hann_window/mul_1:z:0stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2
stft/hann_window/truedivw
stft/hann_window/CosCosstft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2
stft/hann_window/Cosy
stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
stft/hann_window/mul_2/x
stft/hann_window/mul_2Mul!stft/hann_window/mul_2/x:output:0stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2
stft/hann_window/mul_2y
stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
stft/hann_window/sub_2/x
stft/hann_window/sub_2Sub!stft/hann_window/sub_2/x:output:0stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2
stft/hann_window/sub_2
stft/mulMulstft/frame/Reshape_4:output:0stft/hann_window/sub_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

stft/mulo
stft/rfft/packedPackstft/Const:output:0*
N*
T0*
_output_shapes
:2
stft/rfft/packedw
stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2
stft/rfft/fft_lengthy
	stft/rfftRFFTstft/mul:z:0stft/rfft/fft_length:output:0*,
_output_shapes
:џџџџџџџџџ@2
	stft/rfftZ
Abs
ComplexAbsstft/rfft:output:0*,
_output_shapes
:џџџџџџџџџ@2
AbsZ
SquareSquareAbs:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
Squarem
MatMulBatchMatMulV2
Square:y:0matmul_b*
T0*+
_output_shapes
:џџџџџџџџџ@2
MatMulc
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
ConstS
MaxMaxMatMul:output:0Const:output:0*
T0*
_output_shapes
: 2
Max[
	Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2
	Maximum/xx
MaximumMaximumMaximum/x:output:0MatMul:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@2	
MaximumT
LogLogMaximum:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
LogW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   A2	
Const_1H
Log_1LogConst_1:output:0*
T0*
_output_shapes
: 2
Log_1g
truedivRealDivLog:y:0	Log_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@2	
truedivS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
mul/xd
mulMulmul/x:output:0truediv:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
mul_
Maximum_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2
Maximum_1/xf
	Maximum_1MaximumMaximum_1/x:output:0Max:output:0*
T0*
_output_shapes
: 2
	Maximum_1E
Log_2LogMaximum_1:z:0*
T0*
_output_shapes
: 2
Log_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *   A2	
Const_2H
Log_3LogConst_2:output:0*
T0*
_output_shapes
: 2
Log_3X
	truediv_1RealDiv	Log_2:y:0	Log_3:y:0*
T0*
_output_shapes
: 2
	truediv_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2	
mul_1/xW
mul_1Mulmul_1/x:output:0truediv_1:z:0*
T0*
_output_shapes
: 2
mul_1[
subSubmul:z:0	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
subg
Const_3Const*
_output_shapes
:*
dtype0*!
valueB"          2	
Const_3Q
Max_1Maxsub:z:0Const_3:output:0*
T0*
_output_shapes
: 2
Max_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B2	
sub_1/yX
sub_1SubMax_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: 2
sub_1k
	Maximum_2Maximumsub:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
	Maximum_2b
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsMaximum_2:z:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ:	:S O
(
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	waveforms:%!

_output_shapes
:	

D
(__inference_flatten_layer_call_fn_217230

inputs
identityТ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2126812
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Й
о
/__inference_resnet_block_1_layer_call_fn_216614

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_2109422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@ ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
ц
{
&__inference_conv2_layer_call_fn_217474

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2106722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ц
{
&__inference_conv3_layer_call_fn_217508

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2107502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Р
п
/__inference_resnet_block_3_layer_call_fn_217052
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_2118842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:џџџџџџџџџ@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_1
­
Ї
?__inference_fc2_layer_call_and_return_conditional_losses_217261

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Б

J__inference_resnet_block_2_layer_call_and_return_conditional_losses_211483

inputs
conv1_211458
conv1_211460
conv2_211464
conv2_211466
conv3_211470
conv3_211472
shortcut_211475
shortcut_211477
identityЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallЂconv3/StatefulPartitionedCallЂ shortcut/StatefulPartitionedCall
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_211458conv1_211460*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2110652
conv1/StatefulPartitionedCall№
relu1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu1_layer_call_and_return_conditional_losses_2111002
relu1/PartitionedCallЂ
conv2/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0conv2_211464conv2_211466*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2111432
conv2/StatefulPartitionedCall№
relu2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu2_layer_call_and_return_conditional_losses_2111782
relu2/PartitionedCallЂ
conv3/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0conv3_211470conv3_211472*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2112212
conv3/StatefulPartitionedCall
 shortcut/StatefulPartitionedCallStatefulPartitionedCallinputsshortcut_211475shortcut_211477*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_shortcut_layer_call_and_return_conditional_losses_2112762"
 shortcut/StatefulPartitionedCallЅ
add/addAddV2&conv3/StatefulPartitionedCall:output:0)shortcut/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
add/addс
out_block/PartitionedCallPartitionedCalladd/add:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_out_block_layer_call_and_return_conditional_losses_2113122
out_block/PartitionedCallў
IdentityIdentity"out_block/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall!^shortcut/StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@@::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2D
 shortcut/StatefulPartitionedCall shortcut/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Т
Ж
A__inference_conv1_layer_call_and_return_conditional_losses_217683

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsК
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЙ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@:::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
а;
ї
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_216593

inputs5
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЈ
conv1/conv1d/ExpandDims
ExpandDimsinputs$conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv1/conv1d/ExpandDimsЪ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimЯ
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1/conv1d/ExpandDims_1Ю
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv1/conv1dЄ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1/BiasAdd/ReadVariableOpЄ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
conv1/BiasAddn

relu1/ReluReluconv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimК
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2/conv1d/ExpandDimsЪ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimЯ
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv2/conv1d/ExpandDims_1Ю
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2/conv1dЄ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2/BiasAdd/ReadVariableOpЄ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
conv2/BiasAddn

relu2/ReluReluconv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimК
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv3/conv1d/ExpandDimsЪ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimЯ
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv3/conv1d/ExpandDims_1Ю
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv3/conv1dЄ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv3/BiasAdd/ReadVariableOpЄ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimБ
shortcut/conv1d/ExpandDims
ExpandDimsinputs'shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
shortcut/conv1d/ExpandDimsг
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimл
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
shortcut/conv1d/ExpandDims_1к
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
shortcut/conv1d­
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЇ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
shortcut/BiasAdd/ReadVariableOpА
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
add/addk
out_block/ReluReluadd/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
out_block/Relut
IdentityIdentityout_block/Relu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@ :::::::::S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
 
B
&__inference_relu2_layer_call_fn_217610

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu2_layer_call_and_return_conditional_losses_2111782
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Р
п
/__inference_resnet_block_3_layer_call_fn_217073
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_2119542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:џџџџџџџџџ@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_1
 
B
&__inference_relu2_layer_call_fn_217736

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu2_layer_call_and_return_conditional_losses_2116492
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ж
Ж
A__inference_conv1_layer_call_and_return_conditional_losses_217305

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@:::S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ј

J__inference_resnet_block_1_layer_call_and_return_conditional_losses_211012

inputs
conv1_210987
conv1_210989
conv2_210993
conv2_210995
conv3_210999
conv3_211001
shortcut_211004
shortcut_211006
identityЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallЂconv3/StatefulPartitionedCallЂ shortcut/StatefulPartitionedCall
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_210987conv1_210989*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2105942
conv1/StatefulPartitionedCallя
relu1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu1_layer_call_and_return_conditional_losses_2106292
relu1/PartitionedCallЁ
conv2/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0conv2_210993conv2_210995*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2106722
conv2/StatefulPartitionedCallя
relu2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu2_layer_call_and_return_conditional_losses_2107072
relu2/PartitionedCallЁ
conv3/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0conv3_210999conv3_211001*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2107502
conv3/StatefulPartitionedCall
 shortcut/StatefulPartitionedCallStatefulPartitionedCallinputsshortcut_211004shortcut_211006*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_shortcut_layer_call_and_return_conditional_losses_2108052"
 shortcut/StatefulPartitionedCallЄ
add/addAddV2&conv3/StatefulPartitionedCall:output:0)shortcut/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
add/addр
out_block/PartitionedCallPartitionedCalladd/add:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_out_block_layer_call_and_return_conditional_losses_2108412
out_block/PartitionedCall§
IdentityIdentity"out_block/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall!^shortcut/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@ ::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2D
 shortcut/StatefulPartitionedCall shortcut/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
ъ
{
&__inference_conv3_layer_call_fn_217760

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2116922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ш
{
&__inference_conv1_layer_call_fn_217566

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2110652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ж
Ж
A__inference_conv3_layer_call_and_return_conditional_losses_210750

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@@:::S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ж
Ж
A__inference_conv1_layer_call_and_return_conditional_losses_210123

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@:::S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Г

J__inference_resnet_block_3_layer_call_and_return_conditional_losses_211954

inputs
conv1_211929
conv1_211931
conv2_211935
conv2_211937
conv3_211941
conv3_211943
shortcut_211946
shortcut_211948
identityЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallЂconv3/StatefulPartitionedCallЂ shortcut/StatefulPartitionedCall
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_211929conv1_211931*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2115362
conv1/StatefulPartitionedCall№
relu1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu1_layer_call_and_return_conditional_losses_2115712
relu1/PartitionedCallЂ
conv2/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0conv2_211935conv2_211937*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2116142
conv2/StatefulPartitionedCall№
relu2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu2_layer_call_and_return_conditional_losses_2116492
relu2/PartitionedCallЂ
conv3/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0conv3_211941conv3_211943*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2116922
conv3/StatefulPartitionedCall
 shortcut/StatefulPartitionedCallStatefulPartitionedCallinputsshortcut_211946shortcut_211948*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_shortcut_layer_call_and_return_conditional_losses_2117472"
 shortcut/StatefulPartitionedCallЅ
add/addAddV2&conv3/StatefulPartitionedCall:output:0)shortcut/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
add/addс
out_block/PartitionedCallPartitionedCalladd/add:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_out_block_layer_call_and_return_conditional_losses_2117832
out_block/PartitionedCallў
IdentityIdentity"out_block/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall!^shortcut/StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:џџџџџџџџџ@::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2D
 shortcut/StatefulPartitionedCall shortcut/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ќ
X
A__inference_delta_layer_call_and_return_conditional_losses_216046
x
identityy
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/permy
	transpose	Transposextranspose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
	transpose
ConstConst*
_output_shapes

:*
dtype0*9
value0B."                               2
Const
	MirrorPad	MirrorPadtranspose:y:0Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
mode	SYMMETRIC2
	MirrorPadg
arange/startConst*
_output_shapes
: *
dtype0*
valueB :
ќџџџџџџџџ2
arange/start^
arange/limitConst*
_output_shapes
: *
dtype0*
value	B :2
arange/limit^
arange/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
arange/deltaz
arangeRangearange/start:output:0arange/limit:output:0arange/delta:output:0*
_output_shapes
:	2
arangeY
CastCastarange:output:0*

DstT0*

SrcT0*
_output_shapes
:	2
Castw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
Reshape/shapep
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*&
_output_shapes
:	2	
Reshape­
convolutionConv2DMirrorPad:output:0Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
convolution[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pB2
	truediv/y
truedivRealDivconvolution:output:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
truediv}
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose_1/perm
transpose_1	Transposetruediv:z:0transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
transpose_1k
IdentityIdentitytranspose_1:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:R N
/
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
ц
{
&__inference_conv1_layer_call_fn_217440

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2105942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@ ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
П
]
A__inference_relu1_layer_call_and_return_conditional_losses_210158

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@ :S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Е
_
C__inference_flatten_layer_call_and_return_conditional_losses_217225

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Е
_
C__inference_flatten_layer_call_and_return_conditional_losses_212681

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Р
Й
D__inference_shortcut_layer_call_and_return_conditional_losses_211276

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv1d/ExpandDimsЙ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimИ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@@:::S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ж
Ж
A__inference_conv2_layer_call_and_return_conditional_losses_217465

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@@:::S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Н
о
/__inference_resnet_block_3_layer_call_fn_217198

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_2118842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:џџџџџџџџџ@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Й
Й
D__inference_shortcut_layer_call_and_return_conditional_losses_217523

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@ :::S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Ж
Ж
A__inference_conv1_layer_call_and_return_conditional_losses_210594

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@ :::S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
д;
ј
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_216395
input_15
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЉ
conv1/conv1d/ExpandDims
ExpandDimsinput_1$conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv1/conv1d/ExpandDimsЪ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimЯ
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1/conv1d/ExpandDims_1Ю
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv1/conv1dЄ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1/BiasAdd/ReadVariableOpЄ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
conv1/BiasAddn

relu1/ReluReluconv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimК
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2/conv1d/ExpandDimsЪ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimЯ
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv2/conv1d/ExpandDims_1Ю
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2/conv1dЄ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2/BiasAdd/ReadVariableOpЄ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
conv2/BiasAddn

relu2/ReluReluconv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimК
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv3/conv1d/ExpandDimsЪ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimЯ
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv3/conv1d/ExpandDims_1Ю
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv3/conv1dЄ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv3/BiasAdd/ReadVariableOpЄ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimВ
shortcut/conv1d/ExpandDims
ExpandDimsinput_1'shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
shortcut/conv1d/ExpandDimsг
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimл
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
shortcut/conv1d/ExpandDims_1к
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
shortcut/conv1d­
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЇ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
shortcut/BiasAdd/ReadVariableOpА
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
add/addk
out_block/ReluReluadd/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
out_block/Relut
IdentityIdentityout_block/Relu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@ :::::::::T P
+
_output_shapes
:џџџџџџџџџ@ 
!
_user_specified_name	input_1
П
]
A__inference_relu1_layer_call_and_return_conditional_losses_217319

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@ :S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
џ;
ј
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_217031
input_15
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЊ
conv1/conv1d/ExpandDims
ExpandDimsinput_1$conv1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1/conv1d/ExpandDimsЬ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimб
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1/conv1d/ExpandDims_1Я
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1/conv1dЅ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv1/BiasAdd/ReadVariableOpЅ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv1/BiasAddo

relu1/ReluReluconv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimЛ
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv2/conv1d/ExpandDimsЬ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimб
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv2/conv1d/ExpandDims_1Я
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2/conv1dЅ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2/BiasAdd/ReadVariableOpЅ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv2/BiasAddo

relu2/ReluReluconv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimЛ
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv3/conv1d/ExpandDimsЬ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimб
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv3/conv1d/ExpandDims_1Я
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv3/conv1dЅ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv3/BiasAdd/ReadVariableOpЅ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimГ
shortcut/conv1d/ExpandDims
ExpandDimsinput_1'shortcut/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
shortcut/conv1d/ExpandDimsе
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimн
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
shortcut/conv1d/ExpandDims_1л
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
shortcut/conv1dЎ
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЈ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
shortcut/BiasAdd/ReadVariableOpБ
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
add/addl
out_block/ReluReluadd/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
out_block/Reluu
IdentityIdentityout_block/Relu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:џџџџџџџџџ@:::::::::U Q
,
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_1
ё

!__inference__wrapped_model_210089
input_1(
$res_net_log_mel_spectrogram_matmul_bJ
Fres_net_resnet_block_conv1_conv1d_expanddims_1_readvariableop_resource>
:res_net_resnet_block_conv1_biasadd_readvariableop_resourceJ
Fres_net_resnet_block_conv2_conv1d_expanddims_1_readvariableop_resource>
:res_net_resnet_block_conv2_biasadd_readvariableop_resourceJ
Fres_net_resnet_block_conv3_conv1d_expanddims_1_readvariableop_resource>
:res_net_resnet_block_conv3_biasadd_readvariableop_resourceM
Ires_net_resnet_block_shortcut_conv1d_expanddims_1_readvariableop_resourceA
=res_net_resnet_block_shortcut_biasadd_readvariableop_resourceL
Hres_net_resnet_block_1_conv1_conv1d_expanddims_1_readvariableop_resource@
<res_net_resnet_block_1_conv1_biasadd_readvariableop_resourceL
Hres_net_resnet_block_1_conv2_conv1d_expanddims_1_readvariableop_resource@
<res_net_resnet_block_1_conv2_biasadd_readvariableop_resourceL
Hres_net_resnet_block_1_conv3_conv1d_expanddims_1_readvariableop_resource@
<res_net_resnet_block_1_conv3_biasadd_readvariableop_resourceO
Kres_net_resnet_block_1_shortcut_conv1d_expanddims_1_readvariableop_resourceC
?res_net_resnet_block_1_shortcut_biasadd_readvariableop_resourceL
Hres_net_resnet_block_2_conv1_conv1d_expanddims_1_readvariableop_resource@
<res_net_resnet_block_2_conv1_biasadd_readvariableop_resourceL
Hres_net_resnet_block_2_conv2_conv1d_expanddims_1_readvariableop_resource@
<res_net_resnet_block_2_conv2_biasadd_readvariableop_resourceL
Hres_net_resnet_block_2_conv3_conv1d_expanddims_1_readvariableop_resource@
<res_net_resnet_block_2_conv3_biasadd_readvariableop_resourceO
Kres_net_resnet_block_2_shortcut_conv1d_expanddims_1_readvariableop_resourceC
?res_net_resnet_block_2_shortcut_biasadd_readvariableop_resourceL
Hres_net_resnet_block_3_conv1_conv1d_expanddims_1_readvariableop_resource@
<res_net_resnet_block_3_conv1_biasadd_readvariableop_resourceL
Hres_net_resnet_block_3_conv2_conv1d_expanddims_1_readvariableop_resource@
<res_net_resnet_block_3_conv2_biasadd_readvariableop_resourceL
Hres_net_resnet_block_3_conv3_conv1d_expanddims_1_readvariableop_resource@
<res_net_resnet_block_3_conv3_biasadd_readvariableop_resourceO
Kres_net_resnet_block_3_shortcut_conv1d_expanddims_1_readvariableop_resourceC
?res_net_resnet_block_3_shortcut_biasadd_readvariableop_resource.
*res_net_fc1_matmul_readvariableop_resource/
+res_net_fc1_biasadd_readvariableop_resource.
*res_net_fc2_matmul_readvariableop_resource/
+res_net_fc2_biasadd_readvariableop_resource.
*res_net_fc3_matmul_readvariableop_resource/
+res_net_fc3_biasadd_readvariableop_resource
identity
res_net/SqueezeSqueezeinput_1*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
res_net/SqueezeЁ
-res_net/log_mel_spectrogram/stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2/
-res_net/log_mel_spectrogram/stft/frame_length
+res_net/log_mel_spectrogram/stft/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2-
+res_net/log_mel_spectrogram/stft/frame_step
&res_net/log_mel_spectrogram/stft/ConstConst*
_output_shapes
: *
dtype0*
value
B :2(
&res_net/log_mel_spectrogram/stft/ConstЅ
+res_net/log_mel_spectrogram/stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+res_net/log_mel_spectrogram/stft/frame/axisЄ
,res_net/log_mel_spectrogram/stft/frame/ShapeShaperes_net/Squeeze:output:0*
T0*
_output_shapes
:2.
,res_net/log_mel_spectrogram/stft/frame/Shape
+res_net/log_mel_spectrogram/stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2-
+res_net/log_mel_spectrogram/stft/frame/RankЊ
2res_net/log_mel_spectrogram/stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2res_net/log_mel_spectrogram/stft/frame/range/startЊ
2res_net/log_mel_spectrogram/stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2res_net/log_mel_spectrogram/stft/frame/range/deltaБ
,res_net/log_mel_spectrogram/stft/frame/rangeRange;res_net/log_mel_spectrogram/stft/frame/range/start:output:04res_net/log_mel_spectrogram/stft/frame/Rank:output:0;res_net/log_mel_spectrogram/stft/frame/range/delta:output:0*
_output_shapes
:2.
,res_net/log_mel_spectrogram/stft/frame/rangeЫ
:res_net/log_mel_spectrogram/stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2<
:res_net/log_mel_spectrogram/stft/frame/strided_slice/stackЦ
<res_net/log_mel_spectrogram/stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<res_net/log_mel_spectrogram/stft/frame/strided_slice/stack_1Ц
<res_net/log_mel_spectrogram/stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<res_net/log_mel_spectrogram/stft/frame/strided_slice/stack_2Ь
4res_net/log_mel_spectrogram/stft/frame/strided_sliceStridedSlice5res_net/log_mel_spectrogram/stft/frame/range:output:0Cres_net/log_mel_spectrogram/stft/frame/strided_slice/stack:output:0Eres_net/log_mel_spectrogram/stft/frame/strided_slice/stack_1:output:0Eres_net/log_mel_spectrogram/stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4res_net/log_mel_spectrogram/stft/frame/strided_slice
,res_net/log_mel_spectrogram/stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,res_net/log_mel_spectrogram/stft/frame/sub/yэ
*res_net/log_mel_spectrogram/stft/frame/subSub4res_net/log_mel_spectrogram/stft/frame/Rank:output:05res_net/log_mel_spectrogram/stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2,
*res_net/log_mel_spectrogram/stft/frame/subѓ
,res_net/log_mel_spectrogram/stft/frame/sub_1Sub.res_net/log_mel_spectrogram/stft/frame/sub:z:0=res_net/log_mel_spectrogram/stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2.
,res_net/log_mel_spectrogram/stft/frame/sub_1Є
/res_net/log_mel_spectrogram/stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :21
/res_net/log_mel_spectrogram/stft/frame/packed/1П
-res_net/log_mel_spectrogram/stft/frame/packedPack=res_net/log_mel_spectrogram/stft/frame/strided_slice:output:08res_net/log_mel_spectrogram/stft/frame/packed/1:output:00res_net/log_mel_spectrogram/stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2/
-res_net/log_mel_spectrogram/stft/frame/packedВ
6res_net/log_mel_spectrogram/stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 28
6res_net/log_mel_spectrogram/stft/frame/split/split_dimт
,res_net/log_mel_spectrogram/stft/frame/splitSplitV5res_net/log_mel_spectrogram/stft/frame/Shape:output:06res_net/log_mel_spectrogram/stft/frame/packed:output:0?res_net/log_mel_spectrogram/stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2.
,res_net/log_mel_spectrogram/stft/frame/splitЏ
4res_net/log_mel_spectrogram/stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 26
4res_net/log_mel_spectrogram/stft/frame/Reshape/shapeГ
6res_net/log_mel_spectrogram/stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 28
6res_net/log_mel_spectrogram/stft/frame/Reshape/shape_1
.res_net/log_mel_spectrogram/stft/frame/ReshapeReshape5res_net/log_mel_spectrogram/stft/frame/split:output:1?res_net/log_mel_spectrogram/stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 20
.res_net/log_mel_spectrogram/stft/frame/Reshape
+res_net/log_mel_spectrogram/stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2-
+res_net/log_mel_spectrogram/stft/frame/Size 
-res_net/log_mel_spectrogram/stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2/
-res_net/log_mel_spectrogram/stft/frame/Size_1Ё
,res_net/log_mel_spectrogram/stft/frame/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,res_net/log_mel_spectrogram/stft/frame/ConstЙ
*res_net/log_mel_spectrogram/stft/frame/NegNeg7res_net/log_mel_spectrogram/stft/frame/Reshape:output:0*
T0*
_output_shapes
: 2,
*res_net/log_mel_spectrogram/stft/frame/Negѕ
/res_net/log_mel_spectrogram/stft/frame/floordivFloorDiv.res_net/log_mel_spectrogram/stft/frame/Neg:y:04res_net/log_mel_spectrogram/stft/frame_step:output:0*
T0*
_output_shapes
: 21
/res_net/log_mel_spectrogram/stft/frame/floordivЙ
,res_net/log_mel_spectrogram/stft/frame/Neg_1Neg3res_net/log_mel_spectrogram/stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2.
,res_net/log_mel_spectrogram/stft/frame/Neg_1Ђ
.res_net/log_mel_spectrogram/stft/frame/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :20
.res_net/log_mel_spectrogram/stft/frame/sub_2/yя
,res_net/log_mel_spectrogram/stft/frame/sub_2Sub0res_net/log_mel_spectrogram/stft/frame/Neg_1:y:07res_net/log_mel_spectrogram/stft/frame/sub_2/y:output:0*
T0*
_output_shapes
: 2.
,res_net/log_mel_spectrogram/stft/frame/sub_2ш
*res_net/log_mel_spectrogram/stft/frame/mulMul4res_net/log_mel_spectrogram/stft/frame_step:output:00res_net/log_mel_spectrogram/stft/frame/sub_2:z:0*
T0*
_output_shapes
: 2,
*res_net/log_mel_spectrogram/stft/frame/mulъ
*res_net/log_mel_spectrogram/stft/frame/addAddV26res_net/log_mel_spectrogram/stft/frame_length:output:0.res_net/log_mel_spectrogram/stft/frame/mul:z:0*
T0*
_output_shapes
: 2,
*res_net/log_mel_spectrogram/stft/frame/addэ
,res_net/log_mel_spectrogram/stft/frame/sub_3Sub.res_net/log_mel_spectrogram/stft/frame/add:z:07res_net/log_mel_spectrogram/stft/frame/Reshape:output:0*
T0*
_output_shapes
: 2.
,res_net/log_mel_spectrogram/stft/frame/sub_3І
0res_net/log_mel_spectrogram/stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 22
0res_net/log_mel_spectrogram/stft/frame/Maximum/xљ
.res_net/log_mel_spectrogram/stft/frame/MaximumMaximum9res_net/log_mel_spectrogram/stft/frame/Maximum/x:output:00res_net/log_mel_spectrogram/stft/frame/sub_3:z:0*
T0*
_output_shapes
: 20
.res_net/log_mel_spectrogram/stft/frame/MaximumЊ
2res_net/log_mel_spectrogram/stft/frame/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :24
2res_net/log_mel_spectrogram/stft/frame/zeros/mul/yџ
0res_net/log_mel_spectrogram/stft/frame/zeros/mulMul4res_net/log_mel_spectrogram/stft/frame/Size:output:0;res_net/log_mel_spectrogram/stft/frame/zeros/mul/y:output:0*
T0*
_output_shapes
: 22
0res_net/log_mel_spectrogram/stft/frame/zeros/mul­
3res_net/log_mel_spectrogram/stft/frame/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш25
3res_net/log_mel_spectrogram/stft/frame/zeros/Less/y
1res_net/log_mel_spectrogram/stft/frame/zeros/LessLess4res_net/log_mel_spectrogram/stft/frame/zeros/mul:z:0<res_net/log_mel_spectrogram/stft/frame/zeros/Less/y:output:0*
T0*
_output_shapes
: 23
1res_net/log_mel_spectrogram/stft/frame/zeros/LessА
5res_net/log_mel_spectrogram/stft/frame/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :27
5res_net/log_mel_spectrogram/stft/frame/zeros/packed/1
3res_net/log_mel_spectrogram/stft/frame/zeros/packedPack4res_net/log_mel_spectrogram/stft/frame/Size:output:0>res_net/log_mel_spectrogram/stft/frame/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:25
3res_net/log_mel_spectrogram/stft/frame/zeros/packedЊ
2res_net/log_mel_spectrogram/stft/frame/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : 24
2res_net/log_mel_spectrogram/stft/frame/zeros/Const
,res_net/log_mel_spectrogram/stft/frame/zerosFill<res_net/log_mel_spectrogram/stft/frame/zeros/packed:output:0;res_net/log_mel_spectrogram/stft/frame/zeros/Const:output:0*
T0*
_output_shapes

:2.
,res_net/log_mel_spectrogram/stft/frame/zerosЌ
3res_net/log_mel_spectrogram/stft/frame/packed_1/0/0Const*
_output_shapes
: *
dtype0*
value	B : 25
3res_net/log_mel_spectrogram/stft/frame/packed_1/0/0
1res_net/log_mel_spectrogram/stft/frame/packed_1/0Pack<res_net/log_mel_spectrogram/stft/frame/packed_1/0/0:output:02res_net/log_mel_spectrogram/stft/frame/Maximum:z:0*
N*
T0*
_output_shapes
:23
1res_net/log_mel_spectrogram/stft/frame/packed_1/0и
/res_net/log_mel_spectrogram/stft/frame/packed_1Pack:res_net/log_mel_spectrogram/stft/frame/packed_1/0:output:0*
N*
T0*
_output_shapes

:21
/res_net/log_mel_spectrogram/stft/frame/packed_1Ў
4res_net/log_mel_spectrogram/stft/frame/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :26
4res_net/log_mel_spectrogram/stft/frame/zeros_1/mul/y
2res_net/log_mel_spectrogram/stft/frame/zeros_1/mulMul6res_net/log_mel_spectrogram/stft/frame/Size_1:output:0=res_net/log_mel_spectrogram/stft/frame/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 24
2res_net/log_mel_spectrogram/stft/frame/zeros_1/mulБ
5res_net/log_mel_spectrogram/stft/frame/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш27
5res_net/log_mel_spectrogram/stft/frame/zeros_1/Less/y
3res_net/log_mel_spectrogram/stft/frame/zeros_1/LessLess6res_net/log_mel_spectrogram/stft/frame/zeros_1/mul:z:0>res_net/log_mel_spectrogram/stft/frame/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 25
3res_net/log_mel_spectrogram/stft/frame/zeros_1/LessД
7res_net/log_mel_spectrogram/stft/frame/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :29
7res_net/log_mel_spectrogram/stft/frame/zeros_1/packed/1
5res_net/log_mel_spectrogram/stft/frame/zeros_1/packedPack6res_net/log_mel_spectrogram/stft/frame/Size_1:output:0@res_net/log_mel_spectrogram/stft/frame/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:27
5res_net/log_mel_spectrogram/stft/frame/zeros_1/packedЎ
4res_net/log_mel_spectrogram/stft/frame/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/log_mel_spectrogram/stft/frame/zeros_1/Const
.res_net/log_mel_spectrogram/stft/frame/zeros_1Fill>res_net/log_mel_spectrogram/stft/frame/zeros_1/packed:output:0=res_net/log_mel_spectrogram/stft/frame/zeros_1/Const:output:0*
T0*
_output_shapes

: 20
.res_net/log_mel_spectrogram/stft/frame/zeros_1Њ
2res_net/log_mel_spectrogram/stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2res_net/log_mel_spectrogram/stft/frame/concat/axis
-res_net/log_mel_spectrogram/stft/frame/concatConcatV25res_net/log_mel_spectrogram/stft/frame/zeros:output:08res_net/log_mel_spectrogram/stft/frame/packed_1:output:07res_net/log_mel_spectrogram/stft/frame/zeros_1:output:0;res_net/log_mel_spectrogram/stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes

:2/
-res_net/log_mel_spectrogram/stft/frame/concatЉ
,res_net/log_mel_spectrogram/stft/frame/PadV2PadV2res_net/Squeeze:output:06res_net/log_mel_spectrogram/stft/frame/concat:output:05res_net/log_mel_spectrogram/stft/frame/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2.
,res_net/log_mel_spectrogram/stft/frame/PadV2Х
.res_net/log_mel_spectrogram/stft/frame/Shape_1Shape5res_net/log_mel_spectrogram/stft/frame/PadV2:output:0*
T0*
_output_shapes
:20
.res_net/log_mel_spectrogram/stft/frame/Shape_1Ђ
.res_net/log_mel_spectrogram/stft/frame/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.res_net/log_mel_spectrogram/stft/frame/add_1/yў
,res_net/log_mel_spectrogram/stft/frame/add_1AddV2=res_net/log_mel_spectrogram/stft/frame/strided_slice:output:07res_net/log_mel_spectrogram/stft/frame/add_1/y:output:0*
T0*
_output_shapes
: 2.
,res_net/log_mel_spectrogram/stft/frame/add_1ё
<res_net/log_mel_spectrogram/stft/frame/strided_slice_1/stackPack=res_net/log_mel_spectrogram/stft/frame/strided_slice:output:0*
N*
T0*
_output_shapes
:2>
<res_net/log_mel_spectrogram/stft/frame/strided_slice_1/stackш
>res_net/log_mel_spectrogram/stft/frame/strided_slice_1/stack_1Pack0res_net/log_mel_spectrogram/stft/frame/add_1:z:0*
N*
T0*
_output_shapes
:2@
>res_net/log_mel_spectrogram/stft/frame/strided_slice_1/stack_1Ъ
>res_net/log_mel_spectrogram/stft/frame/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>res_net/log_mel_spectrogram/stft/frame/strided_slice_1/stack_2и
6res_net/log_mel_spectrogram/stft/frame/strided_slice_1StridedSlice7res_net/log_mel_spectrogram/stft/frame/Shape_1:output:0Eres_net/log_mel_spectrogram/stft/frame/strided_slice_1/stack:output:0Gres_net/log_mel_spectrogram/stft/frame/strided_slice_1/stack_1:output:0Gres_net/log_mel_spectrogram/stft/frame/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6res_net/log_mel_spectrogram/stft/frame/strided_slice_1І
0res_net/log_mel_spectrogram/stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :22
0res_net/log_mel_spectrogram/stft/frame/gcd/ConstЌ
3res_net/log_mel_spectrogram/stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :25
3res_net/log_mel_spectrogram/stft/frame/floordiv_1/y
1res_net/log_mel_spectrogram/stft/frame/floordiv_1FloorDiv6res_net/log_mel_spectrogram/stft/frame_length:output:0<res_net/log_mel_spectrogram/stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 23
1res_net/log_mel_spectrogram/stft/frame/floordiv_1Ќ
3res_net/log_mel_spectrogram/stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :25
3res_net/log_mel_spectrogram/stft/frame/floordiv_2/y
1res_net/log_mel_spectrogram/stft/frame/floordiv_2FloorDiv4res_net/log_mel_spectrogram/stft/frame_step:output:0<res_net/log_mel_spectrogram/stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 23
1res_net/log_mel_spectrogram/stft/frame/floordiv_2Ќ
3res_net/log_mel_spectrogram/stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :25
3res_net/log_mel_spectrogram/stft/frame/floordiv_3/y
1res_net/log_mel_spectrogram/stft/frame/floordiv_3FloorDiv?res_net/log_mel_spectrogram/stft/frame/strided_slice_1:output:0<res_net/log_mel_spectrogram/stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 23
1res_net/log_mel_spectrogram/stft/frame/floordiv_3Ђ
.res_net/log_mel_spectrogram/stft/frame/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.res_net/log_mel_spectrogram/stft/frame/mul_1/yє
,res_net/log_mel_spectrogram/stft/frame/mul_1Mul5res_net/log_mel_spectrogram/stft/frame/floordiv_3:z:07res_net/log_mel_spectrogram/stft/frame/mul_1/y:output:0*
T0*
_output_shapes
: 2.
,res_net/log_mel_spectrogram/stft/frame/mul_1м
8res_net/log_mel_spectrogram/stft/frame/concat_1/values_1Pack0res_net/log_mel_spectrogram/stft/frame/mul_1:z:0*
N*
T0*
_output_shapes
:2:
8res_net/log_mel_spectrogram/stft/frame/concat_1/values_1Ў
4res_net/log_mel_spectrogram/stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/log_mel_spectrogram/stft/frame/concat_1/axis
/res_net/log_mel_spectrogram/stft/frame/concat_1ConcatV25res_net/log_mel_spectrogram/stft/frame/split:output:0Ares_net/log_mel_spectrogram/stft/frame/concat_1/values_1:output:05res_net/log_mel_spectrogram/stft/frame/split:output:2=res_net/log_mel_spectrogram/stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:21
/res_net/log_mel_spectrogram/stft/frame/concat_1К
:res_net/log_mel_spectrogram/stft/frame/concat_2/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :2<
:res_net/log_mel_spectrogram/stft/frame/concat_2/values_1/1І
8res_net/log_mel_spectrogram/stft/frame/concat_2/values_1Pack5res_net/log_mel_spectrogram/stft/frame/floordiv_3:z:0Cres_net/log_mel_spectrogram/stft/frame/concat_2/values_1/1:output:0*
N*
T0*
_output_shapes
:2:
8res_net/log_mel_spectrogram/stft/frame/concat_2/values_1Ў
4res_net/log_mel_spectrogram/stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/log_mel_spectrogram/stft/frame/concat_2/axis
/res_net/log_mel_spectrogram/stft/frame/concat_2ConcatV25res_net/log_mel_spectrogram/stft/frame/split:output:0Ares_net/log_mel_spectrogram/stft/frame/concat_2/values_1:output:05res_net/log_mel_spectrogram/stft/frame/split:output:2=res_net/log_mel_spectrogram/stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:21
/res_net/log_mel_spectrogram/stft/frame/concat_2А
1res_net/log_mel_spectrogram/stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 23
1res_net/log_mel_spectrogram/stft/frame/zeros_likeК
6res_net/log_mel_spectrogram/stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:28
6res_net/log_mel_spectrogram/stft/frame/ones_like/ShapeВ
6res_net/log_mel_spectrogram/stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :28
6res_net/log_mel_spectrogram/stft/frame/ones_like/Const
0res_net/log_mel_spectrogram/stft/frame/ones_likeFill?res_net/log_mel_spectrogram/stft/frame/ones_like/Shape:output:0?res_net/log_mel_spectrogram/stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:22
0res_net/log_mel_spectrogram/stft/frame/ones_likeЊ
3res_net/log_mel_spectrogram/stft/frame/StridedSliceStridedSlice5res_net/log_mel_spectrogram/stft/frame/PadV2:output:0:res_net/log_mel_spectrogram/stft/frame/zeros_like:output:08res_net/log_mel_spectrogram/stft/frame/concat_1:output:09res_net/log_mel_spectrogram/stft/frame/ones_like:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3res_net/log_mel_spectrogram/stft/frame/StridedSliceІ
0res_net/log_mel_spectrogram/stft/frame/Reshape_1Reshape<res_net/log_mel_spectrogram/stft/frame/StridedSlice:output:08res_net/log_mel_spectrogram/stft/frame/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
0res_net/log_mel_spectrogram/stft/frame/Reshape_1Ў
4res_net/log_mel_spectrogram/stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/log_mel_spectrogram/stft/frame/range_1/startЎ
4res_net/log_mel_spectrogram/stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :26
4res_net/log_mel_spectrogram/stft/frame/range_1/deltaО
.res_net/log_mel_spectrogram/stft/frame/range_1Range=res_net/log_mel_spectrogram/stft/frame/range_1/start:output:00res_net/log_mel_spectrogram/stft/frame/Neg_1:y:0=res_net/log_mel_spectrogram/stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ20
.res_net/log_mel_spectrogram/stft/frame/range_1
,res_net/log_mel_spectrogram/stft/frame/mul_2Mul7res_net/log_mel_spectrogram/stft/frame/range_1:output:05res_net/log_mel_spectrogram/stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2.
,res_net/log_mel_spectrogram/stft/frame/mul_2Ж
8res_net/log_mel_spectrogram/stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2:
8res_net/log_mel_spectrogram/stft/frame/Reshape_2/shape/1
6res_net/log_mel_spectrogram/stft/frame/Reshape_2/shapePack0res_net/log_mel_spectrogram/stft/frame/Neg_1:y:0Ares_net/log_mel_spectrogram/stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:28
6res_net/log_mel_spectrogram/stft/frame/Reshape_2/shape
0res_net/log_mel_spectrogram/stft/frame/Reshape_2Reshape0res_net/log_mel_spectrogram/stft/frame/mul_2:z:0?res_net/log_mel_spectrogram/stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
0res_net/log_mel_spectrogram/stft/frame/Reshape_2Ў
4res_net/log_mel_spectrogram/stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/log_mel_spectrogram/stft/frame/range_2/startЎ
4res_net/log_mel_spectrogram/stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :26
4res_net/log_mel_spectrogram/stft/frame/range_2/deltaК
.res_net/log_mel_spectrogram/stft/frame/range_2Range=res_net/log_mel_spectrogram/stft/frame/range_2/start:output:05res_net/log_mel_spectrogram/stft/frame/floordiv_1:z:0=res_net/log_mel_spectrogram/stft/frame/range_2/delta:output:0*
_output_shapes
: 20
.res_net/log_mel_spectrogram/stft/frame/range_2Ж
8res_net/log_mel_spectrogram/stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2:
8res_net/log_mel_spectrogram/stft/frame/Reshape_3/shape/0 
6res_net/log_mel_spectrogram/stft/frame/Reshape_3/shapePackAres_net/log_mel_spectrogram/stft/frame/Reshape_3/shape/0:output:05res_net/log_mel_spectrogram/stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:28
6res_net/log_mel_spectrogram/stft/frame/Reshape_3/shape
0res_net/log_mel_spectrogram/stft/frame/Reshape_3Reshape7res_net/log_mel_spectrogram/stft/frame/range_2:output:0?res_net/log_mel_spectrogram/stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

: 22
0res_net/log_mel_spectrogram/stft/frame/Reshape_3
,res_net/log_mel_spectrogram/stft/frame/add_2AddV29res_net/log_mel_spectrogram/stft/frame/Reshape_2:output:09res_net/log_mel_spectrogram/stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,res_net/log_mel_spectrogram/stft/frame/add_2
/res_net/log_mel_spectrogram/stft/frame/GatherV2GatherV29res_net/log_mel_spectrogram/stft/frame/Reshape_1:output:00res_net/log_mel_spectrogram/stft/frame/add_2:z:0=res_net/log_mel_spectrogram/stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 21
/res_net/log_mel_spectrogram/stft/frame/GatherV2
8res_net/log_mel_spectrogram/stft/frame/concat_3/values_1Pack0res_net/log_mel_spectrogram/stft/frame/Neg_1:y:06res_net/log_mel_spectrogram/stft/frame_length:output:0*
N*
T0*
_output_shapes
:2:
8res_net/log_mel_spectrogram/stft/frame/concat_3/values_1Ў
4res_net/log_mel_spectrogram/stft/frame/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/log_mel_spectrogram/stft/frame/concat_3/axis
/res_net/log_mel_spectrogram/stft/frame/concat_3ConcatV25res_net/log_mel_spectrogram/stft/frame/split:output:0Ares_net/log_mel_spectrogram/stft/frame/concat_3/values_1:output:05res_net/log_mel_spectrogram/stft/frame/split:output:2=res_net/log_mel_spectrogram/stft/frame/concat_3/axis:output:0*
N*
T0*
_output_shapes
:21
/res_net/log_mel_spectrogram/stft/frame/concat_3
0res_net/log_mel_spectrogram/stft/frame/Reshape_4Reshape8res_net/log_mel_spectrogram/stft/frame/GatherV2:output:08res_net/log_mel_spectrogram/stft/frame/concat_3:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@22
0res_net/log_mel_spectrogram/stft/frame/Reshape_4А
5res_net/log_mel_spectrogram/stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z27
5res_net/log_mel_spectrogram/stft/hann_window/periodicо
1res_net/log_mel_spectrogram/stft/hann_window/CastCast>res_net/log_mel_spectrogram/stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 23
1res_net/log_mel_spectrogram/stft/hann_window/CastД
7res_net/log_mel_spectrogram/stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :29
7res_net/log_mel_spectrogram/stft/hann_window/FloorMod/y
5res_net/log_mel_spectrogram/stft/hann_window/FloorModFloorMod6res_net/log_mel_spectrogram/stft/frame_length:output:0@res_net/log_mel_spectrogram/stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 27
5res_net/log_mel_spectrogram/stft/hann_window/FloorModЊ
2res_net/log_mel_spectrogram/stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :24
2res_net/log_mel_spectrogram/stft/hann_window/sub/x
0res_net/log_mel_spectrogram/stft/hann_window/subSub;res_net/log_mel_spectrogram/stft/hann_window/sub/x:output:09res_net/log_mel_spectrogram/stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 22
0res_net/log_mel_spectrogram/stft/hann_window/subљ
0res_net/log_mel_spectrogram/stft/hann_window/mulMul5res_net/log_mel_spectrogram/stft/hann_window/Cast:y:04res_net/log_mel_spectrogram/stft/hann_window/sub:z:0*
T0*
_output_shapes
: 22
0res_net/log_mel_spectrogram/stft/hann_window/mulќ
0res_net/log_mel_spectrogram/stft/hann_window/addAddV26res_net/log_mel_spectrogram/stft/frame_length:output:04res_net/log_mel_spectrogram/stft/hann_window/mul:z:0*
T0*
_output_shapes
: 22
0res_net/log_mel_spectrogram/stft/hann_window/addЎ
4res_net/log_mel_spectrogram/stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :26
4res_net/log_mel_spectrogram/stft/hann_window/sub_1/y
2res_net/log_mel_spectrogram/stft/hann_window/sub_1Sub4res_net/log_mel_spectrogram/stft/hann_window/add:z:0=res_net/log_mel_spectrogram/stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 24
2res_net/log_mel_spectrogram/stft/hann_window/sub_1к
3res_net/log_mel_spectrogram/stft/hann_window/Cast_1Cast6res_net/log_mel_spectrogram/stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 25
3res_net/log_mel_spectrogram/stft/hann_window/Cast_1Ж
8res_net/log_mel_spectrogram/stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2:
8res_net/log_mel_spectrogram/stft/hann_window/range/startЖ
8res_net/log_mel_spectrogram/stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2:
8res_net/log_mel_spectrogram/stft/hann_window/range/deltaЬ
2res_net/log_mel_spectrogram/stft/hann_window/rangeRangeAres_net/log_mel_spectrogram/stft/hann_window/range/start:output:06res_net/log_mel_spectrogram/stft/frame_length:output:0Ares_net/log_mel_spectrogram/stft/hann_window/range/delta:output:0*
_output_shapes	
:24
2res_net/log_mel_spectrogram/stft/hann_window/rangeф
3res_net/log_mel_spectrogram/stft/hann_window/Cast_2Cast;res_net/log_mel_spectrogram/stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:25
3res_net/log_mel_spectrogram/stft/hann_window/Cast_2­
2res_net/log_mel_spectrogram/stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@24
2res_net/log_mel_spectrogram/stft/hann_window/Const
2res_net/log_mel_spectrogram/stft/hann_window/mul_1Mul;res_net/log_mel_spectrogram/stft/hann_window/Const:output:07res_net/log_mel_spectrogram/stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:24
2res_net/log_mel_spectrogram/stft/hann_window/mul_1
4res_net/log_mel_spectrogram/stft/hann_window/truedivRealDiv6res_net/log_mel_spectrogram/stft/hann_window/mul_1:z:07res_net/log_mel_spectrogram/stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:26
4res_net/log_mel_spectrogram/stft/hann_window/truedivЫ
0res_net/log_mel_spectrogram/stft/hann_window/CosCos8res_net/log_mel_spectrogram/stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:22
0res_net/log_mel_spectrogram/stft/hann_window/CosБ
4res_net/log_mel_spectrogram/stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?26
4res_net/log_mel_spectrogram/stft/hann_window/mul_2/x
2res_net/log_mel_spectrogram/stft/hann_window/mul_2Mul=res_net/log_mel_spectrogram/stft/hann_window/mul_2/x:output:04res_net/log_mel_spectrogram/stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:24
2res_net/log_mel_spectrogram/stft/hann_window/mul_2Б
4res_net/log_mel_spectrogram/stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?26
4res_net/log_mel_spectrogram/stft/hann_window/sub_2/x
2res_net/log_mel_spectrogram/stft/hann_window/sub_2Sub=res_net/log_mel_spectrogram/stft/hann_window/sub_2/x:output:06res_net/log_mel_spectrogram/stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:24
2res_net/log_mel_spectrogram/stft/hann_window/sub_2§
$res_net/log_mel_spectrogram/stft/mulMul9res_net/log_mel_spectrogram/stft/frame/Reshape_4:output:06res_net/log_mel_spectrogram/stft/hann_window/sub_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2&
$res_net/log_mel_spectrogram/stft/mulУ
,res_net/log_mel_spectrogram/stft/rfft/packedPack/res_net/log_mel_spectrogram/stft/Const:output:0*
N*
T0*
_output_shapes
:2.
,res_net/log_mel_spectrogram/stft/rfft/packedЏ
0res_net/log_mel_spectrogram/stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:22
0res_net/log_mel_spectrogram/stft/rfft/fft_lengthщ
%res_net/log_mel_spectrogram/stft/rfftRFFT(res_net/log_mel_spectrogram/stft/mul:z:09res_net/log_mel_spectrogram/stft/rfft/fft_length:output:0*,
_output_shapes
:џџџџџџџџџ@2'
%res_net/log_mel_spectrogram/stft/rfftЎ
res_net/log_mel_spectrogram/Abs
ComplexAbs.res_net/log_mel_spectrogram/stft/rfft:output:0*,
_output_shapes
:џџџџџџџџџ@2!
res_net/log_mel_spectrogram/AbsЎ
"res_net/log_mel_spectrogram/SquareSquare#res_net/log_mel_spectrogram/Abs:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@2$
"res_net/log_mel_spectrogram/Squareн
"res_net/log_mel_spectrogram/MatMulBatchMatMulV2&res_net/log_mel_spectrogram/Square:y:0$res_net_log_mel_spectrogram_matmul_b*
T0*+
_output_shapes
:џџџџџџџџџ@2$
"res_net/log_mel_spectrogram/MatMul
!res_net/log_mel_spectrogram/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!res_net/log_mel_spectrogram/ConstУ
res_net/log_mel_spectrogram/MaxMax+res_net/log_mel_spectrogram/MatMul:output:0*res_net/log_mel_spectrogram/Const:output:0*
T0*
_output_shapes
: 2!
res_net/log_mel_spectrogram/Max
%res_net/log_mel_spectrogram/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2'
%res_net/log_mel_spectrogram/Maximum/xш
#res_net/log_mel_spectrogram/MaximumMaximum.res_net/log_mel_spectrogram/Maximum/x:output:0+res_net/log_mel_spectrogram/MatMul:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@2%
#res_net/log_mel_spectrogram/MaximumЈ
res_net/log_mel_spectrogram/LogLog'res_net/log_mel_spectrogram/Maximum:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2!
res_net/log_mel_spectrogram/Log
#res_net/log_mel_spectrogram/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   A2%
#res_net/log_mel_spectrogram/Const_1
!res_net/log_mel_spectrogram/Log_1Log,res_net/log_mel_spectrogram/Const_1:output:0*
T0*
_output_shapes
: 2#
!res_net/log_mel_spectrogram/Log_1з
#res_net/log_mel_spectrogram/truedivRealDiv#res_net/log_mel_spectrogram/Log:y:0%res_net/log_mel_spectrogram/Log_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@2%
#res_net/log_mel_spectrogram/truediv
!res_net/log_mel_spectrogram/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2#
!res_net/log_mel_spectrogram/mul/xд
res_net/log_mel_spectrogram/mulMul*res_net/log_mel_spectrogram/mul/x:output:0'res_net/log_mel_spectrogram/truediv:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2!
res_net/log_mel_spectrogram/mul
'res_net/log_mel_spectrogram/Maximum_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2)
'res_net/log_mel_spectrogram/Maximum_1/xж
%res_net/log_mel_spectrogram/Maximum_1Maximum0res_net/log_mel_spectrogram/Maximum_1/x:output:0(res_net/log_mel_spectrogram/Max:output:0*
T0*
_output_shapes
: 2'
%res_net/log_mel_spectrogram/Maximum_1
!res_net/log_mel_spectrogram/Log_2Log)res_net/log_mel_spectrogram/Maximum_1:z:0*
T0*
_output_shapes
: 2#
!res_net/log_mel_spectrogram/Log_2
#res_net/log_mel_spectrogram/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *   A2%
#res_net/log_mel_spectrogram/Const_2
!res_net/log_mel_spectrogram/Log_3Log,res_net/log_mel_spectrogram/Const_2:output:0*
T0*
_output_shapes
: 2#
!res_net/log_mel_spectrogram/Log_3Ш
%res_net/log_mel_spectrogram/truediv_1RealDiv%res_net/log_mel_spectrogram/Log_2:y:0%res_net/log_mel_spectrogram/Log_3:y:0*
T0*
_output_shapes
: 2'
%res_net/log_mel_spectrogram/truediv_1
#res_net/log_mel_spectrogram/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2%
#res_net/log_mel_spectrogram/mul_1/xЧ
!res_net/log_mel_spectrogram/mul_1Mul,res_net/log_mel_spectrogram/mul_1/x:output:0)res_net/log_mel_spectrogram/truediv_1:z:0*
T0*
_output_shapes
: 2#
!res_net/log_mel_spectrogram/mul_1Ы
res_net/log_mel_spectrogram/subSub#res_net/log_mel_spectrogram/mul:z:0%res_net/log_mel_spectrogram/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2!
res_net/log_mel_spectrogram/sub
#res_net/log_mel_spectrogram/Const_3Const*
_output_shapes
:*
dtype0*!
valueB"          2%
#res_net/log_mel_spectrogram/Const_3С
!res_net/log_mel_spectrogram/Max_1Max#res_net/log_mel_spectrogram/sub:z:0,res_net/log_mel_spectrogram/Const_3:output:0*
T0*
_output_shapes
: 2#
!res_net/log_mel_spectrogram/Max_1
#res_net/log_mel_spectrogram/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B2%
#res_net/log_mel_spectrogram/sub_1/yШ
!res_net/log_mel_spectrogram/sub_1Sub*res_net/log_mel_spectrogram/Max_1:output:0,res_net/log_mel_spectrogram/sub_1/y:output:0*
T0*
_output_shapes
: 2#
!res_net/log_mel_spectrogram/sub_1л
%res_net/log_mel_spectrogram/Maximum_2Maximum#res_net/log_mel_spectrogram/sub:z:0%res_net/log_mel_spectrogram/sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2'
%res_net/log_mel_spectrogram/Maximum_2
*res_net/log_mel_spectrogram/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*res_net/log_mel_spectrogram/ExpandDims/dimј
&res_net/log_mel_spectrogram/ExpandDims
ExpandDims)res_net/log_mel_spectrogram/Maximum_2:z:03res_net/log_mel_spectrogram/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2(
&res_net/log_mel_spectrogram/ExpandDims
res_net/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
res_net/transpose/permП
res_net/transpose	Transpose/res_net/log_mel_spectrogram/ExpandDims:output:0res_net/transpose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
res_net/transposeЋ
1res_net/mfccs_from_log_mel_spectrograms/dct/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1res_net/mfccs_from_log_mel_spectrograms/dct/ConstЊ
2res_net/mfccs_from_log_mel_spectrograms/dct/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :24
2res_net/mfccs_from_log_mel_spectrograms/dct/Cast/xй
0res_net/mfccs_from_log_mel_spectrograms/dct/CastCast;res_net/mfccs_from_log_mel_spectrograms/dct/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 22
0res_net/mfccs_from_log_mel_spectrograms/dct/CastД
7res_net/mfccs_from_log_mel_spectrograms/dct/range/startConst*
_output_shapes
: *
dtype0*
value	B : 29
7res_net/mfccs_from_log_mel_spectrograms/dct/range/startД
7res_net/mfccs_from_log_mel_spectrograms/dct/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :29
7res_net/mfccs_from_log_mel_spectrograms/dct/range/deltaъ
6res_net/mfccs_from_log_mel_spectrograms/dct/range/CastCast@res_net/mfccs_from_log_mel_spectrograms/dct/range/start:output:0*

DstT0*

SrcT0*
_output_shapes
: 28
6res_net/mfccs_from_log_mel_spectrograms/dct/range/Castю
8res_net/mfccs_from_log_mel_spectrograms/dct/range/Cast_1Cast@res_net/mfccs_from_log_mel_spectrograms/dct/range/delta:output:0*

DstT0*

SrcT0*
_output_shapes
: 2:
8res_net/mfccs_from_log_mel_spectrograms/dct/range/Cast_1Ч
1res_net/mfccs_from_log_mel_spectrograms/dct/rangeRange:res_net/mfccs_from_log_mel_spectrograms/dct/range/Cast:y:04res_net/mfccs_from_log_mel_spectrograms/dct/Cast:y:0<res_net/mfccs_from_log_mel_spectrograms/dct/range/Cast_1:y:0*

Tidx0*
_output_shapes
:23
1res_net/mfccs_from_log_mel_spectrograms/dct/rangeЪ
/res_net/mfccs_from_log_mel_spectrograms/dct/NegNeg:res_net/mfccs_from_log_mel_spectrograms/dct/range:output:0*
T0*
_output_shapes
:21
/res_net/mfccs_from_log_mel_spectrograms/dct/NegЋ
1res_net/mfccs_from_log_mel_spectrograms/dct/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *лI@23
1res_net/mfccs_from_log_mel_spectrograms/dct/mul/yџ
/res_net/mfccs_from_log_mel_spectrograms/dct/mulMul3res_net/mfccs_from_log_mel_spectrograms/dct/Neg:y:0:res_net/mfccs_from_log_mel_spectrograms/dct/mul/y:output:0*
T0*
_output_shapes
:21
/res_net/mfccs_from_log_mel_spectrograms/dct/mulЏ
3res_net/mfccs_from_log_mel_spectrograms/dct/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?25
3res_net/mfccs_from_log_mel_spectrograms/dct/mul_1/y
1res_net/mfccs_from_log_mel_spectrograms/dct/mul_1Mul3res_net/mfccs_from_log_mel_spectrograms/dct/mul:z:0<res_net/mfccs_from_log_mel_spectrograms/dct/mul_1/y:output:0*
T0*
_output_shapes
:23
1res_net/mfccs_from_log_mel_spectrograms/dct/mul_1
3res_net/mfccs_from_log_mel_spectrograms/dct/truedivRealDiv5res_net/mfccs_from_log_mel_spectrograms/dct/mul_1:z:04res_net/mfccs_from_log_mel_spectrograms/dct/Cast:y:0*
T0*
_output_shapes
:25
3res_net/mfccs_from_log_mel_spectrograms/dct/truediv
3res_net/mfccs_from_log_mel_spectrograms/dct/ComplexComplex:res_net/mfccs_from_log_mel_spectrograms/dct/Const:output:07res_net/mfccs_from_log_mel_spectrograms/dct/truediv:z:0*
_output_shapes
:25
3res_net/mfccs_from_log_mel_spectrograms/dct/ComplexЩ
/res_net/mfccs_from_log_mel_spectrograms/dct/ExpExp9res_net/mfccs_from_log_mel_spectrograms/dct/Complex:out:0*
T0*
_output_shapes
:21
/res_net/mfccs_from_log_mel_spectrograms/dct/ExpГ
3res_net/mfccs_from_log_mel_spectrograms/dct/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J   @    25
3res_net/mfccs_from_log_mel_spectrograms/dct/mul_2/x
1res_net/mfccs_from_log_mel_spectrograms/dct/mul_2Mul<res_net/mfccs_from_log_mel_spectrograms/dct/mul_2/x:output:03res_net/mfccs_from_log_mel_spectrograms/dct/Exp:y:0*
T0*
_output_shapes
:23
1res_net/mfccs_from_log_mel_spectrograms/dct/mul_2К
6res_net/mfccs_from_log_mel_spectrograms/dct/rfft/ConstConst*
_output_shapes
:*
dtype0*
valueB:
28
6res_net/mfccs_from_log_mel_spectrograms/dct/rfft/Constя
=res_net/mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2?
=res_net/mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddings
4res_net/mfccs_from_log_mel_spectrograms/dct/rfft/PadPadres_net/transpose:y:0Fres_net/mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
26
4res_net/mfccs_from_log_mel_spectrograms/dct/rfft/PadФ
;res_net/mfccs_from_log_mel_spectrograms/dct/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:
2=
;res_net/mfccs_from_log_mel_spectrograms/dct/rfft/fft_lengthЂ
0res_net/mfccs_from_log_mel_spectrograms/dct/rfftRFFT=res_net/mfccs_from_log_mel_spectrograms/dct/rfft/Pad:output:0Dres_net/mfccs_from_log_mel_spectrograms/dct/rfft/fft_length:output:0*/
_output_shapes
:џџџџџџџџџ@22
0res_net/mfccs_from_log_mel_spectrograms/dct/rfftг
?res_net/mfccs_from_log_mel_spectrograms/dct/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2A
?res_net/mfccs_from_log_mel_spectrograms/dct/strided_slice/stackз
Ares_net/mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2C
Ares_net/mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1з
Ares_net/mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2C
Ares_net/mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2
9res_net/mfccs_from_log_mel_spectrograms/dct/strided_sliceStridedSlice9res_net/mfccs_from_log_mel_spectrograms/dct/rfft:output:0Hres_net/mfccs_from_log_mel_spectrograms/dct/strided_slice/stack:output:0Jres_net/mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1:output:0Jres_net/mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

begin_mask*
ellipsis_mask2;
9res_net/mfccs_from_log_mel_spectrograms/dct/strided_sliceЂ
1res_net/mfccs_from_log_mel_spectrograms/dct/mul_3MulBres_net/mfccs_from_log_mel_spectrograms/dct/strided_slice:output:05res_net/mfccs_from_log_mel_spectrograms/dct/mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@23
1res_net/mfccs_from_log_mel_spectrograms/dct/mul_3д
0res_net/mfccs_from_log_mel_spectrograms/dct/RealReal5res_net/mfccs_from_log_mel_spectrograms/dct/mul_3:z:0*/
_output_shapes
:џџџџџџџџџ@22
0res_net/mfccs_from_log_mel_spectrograms/dct/RealЂ
.res_net/mfccs_from_log_mel_spectrograms/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :20
.res_net/mfccs_from_log_mel_spectrograms/Cast/xЭ
,res_net/mfccs_from_log_mel_spectrograms/CastCast7res_net/mfccs_from_log_mel_spectrograms/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,res_net/mfccs_from_log_mel_spectrograms/CastЃ
-res_net/mfccs_from_log_mel_spectrograms/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2/
-res_net/mfccs_from_log_mel_spectrograms/mul/yь
+res_net/mfccs_from_log_mel_spectrograms/mulMul0res_net/mfccs_from_log_mel_spectrograms/Cast:y:06res_net/mfccs_from_log_mel_spectrograms/mul/y:output:0*
T0*
_output_shapes
: 2-
+res_net/mfccs_from_log_mel_spectrograms/mulЙ
-res_net/mfccs_from_log_mel_spectrograms/RsqrtRsqrt/res_net/mfccs_from_log_mel_spectrograms/mul:z:0*
T0*
_output_shapes
: 2/
-res_net/mfccs_from_log_mel_spectrograms/Rsqrt
-res_net/mfccs_from_log_mel_spectrograms/mul_1Mul9res_net/mfccs_from_log_mel_spectrograms/dct/Real:output:01res_net/mfccs_from_log_mel_spectrograms/Rsqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2/
-res_net/mfccs_from_log_mel_spectrograms/mul_1{
res_net/SquareSquareres_net/transpose:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
res_net/Square
res_net/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
res_net/Sum/reduction_indicesЈ
res_net/SumSumres_net/Square:y:0&res_net/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
	keep_dims(2
res_net/Sumt
res_net/SqrtSqrtres_net/Sum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
res_net/Sqrt
res_net/delta/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
res_net/delta/transpose/permг
res_net/delta/transpose	Transpose1res_net/mfccs_from_log_mel_spectrograms/mul_1:z:0%res_net/delta/transpose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
res_net/delta/transpose
res_net/delta/ConstConst*
_output_shapes

:*
dtype0*9
value0B."                               2
res_net/delta/ConstЩ
res_net/delta/MirrorPad	MirrorPadres_net/delta/transpose:y:0res_net/delta/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
mode	SYMMETRIC2
res_net/delta/MirrorPad
res_net/delta/arange/startConst*
_output_shapes
: *
dtype0*
valueB :
ќџџџџџџџџ2
res_net/delta/arange/startz
res_net/delta/arange/limitConst*
_output_shapes
: *
dtype0*
value	B :2
res_net/delta/arange/limitz
res_net/delta/arange/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
res_net/delta/arange/deltaР
res_net/delta/arangeRange#res_net/delta/arange/start:output:0#res_net/delta/arange/limit:output:0#res_net/delta/arange/delta:output:0*
_output_shapes
:	2
res_net/delta/arange
res_net/delta/CastCastres_net/delta/arange:output:0*

DstT0*

SrcT0*
_output_shapes
:	2
res_net/delta/Cast
res_net/delta/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
res_net/delta/Reshape/shapeЈ
res_net/delta/ReshapeReshaperes_net/delta/Cast:y:0$res_net/delta/Reshape/shape:output:0*
T0*&
_output_shapes
:	2
res_net/delta/Reshapeх
res_net/delta/convolutionConv2D res_net/delta/MirrorPad:output:0res_net/delta/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
res_net/delta/convolutionw
res_net/delta/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pB2
res_net/delta/truediv/yЙ
res_net/delta/truedivRealDiv"res_net/delta/convolution:output:0 res_net/delta/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
res_net/delta/truediv
res_net/delta/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2 
res_net/delta/transpose_1/permС
res_net/delta/transpose_1	Transposeres_net/delta/truediv:z:0'res_net/delta/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
res_net/delta/transpose_1
res_net/delta/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2 
res_net/delta/transpose_2/permХ
res_net/delta/transpose_2	Transposeres_net/delta/transpose_1:y:0'res_net/delta/transpose_2/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
res_net/delta/transpose_2
res_net/delta/Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                               2
res_net/delta/Const_1б
res_net/delta/MirrorPad_1	MirrorPadres_net/delta/transpose_2:y:0res_net/delta/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
mode	SYMMETRIC2
res_net/delta/MirrorPad_1
res_net/delta/arange_1/startConst*
_output_shapes
: *
dtype0*
valueB :
ќџџџџџџџџ2
res_net/delta/arange_1/start~
res_net/delta/arange_1/limitConst*
_output_shapes
: *
dtype0*
value	B :2
res_net/delta/arange_1/limit~
res_net/delta/arange_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
res_net/delta/arange_1/deltaЪ
res_net/delta/arange_1Range%res_net/delta/arange_1/start:output:0%res_net/delta/arange_1/limit:output:0%res_net/delta/arange_1/delta:output:0*
_output_shapes
:	2
res_net/delta/arange_1
res_net/delta/Cast_1Castres_net/delta/arange_1:output:0*

DstT0*

SrcT0*
_output_shapes
:	2
res_net/delta/Cast_1
res_net/delta/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
res_net/delta/Reshape_1/shapeА
res_net/delta/Reshape_1Reshaperes_net/delta/Cast_1:y:0&res_net/delta/Reshape_1/shape:output:0*
T0*&
_output_shapes
:	2
res_net/delta/Reshape_1э
res_net/delta/convolution_1Conv2D"res_net/delta/MirrorPad_1:output:0 res_net/delta/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
res_net/delta/convolution_1{
res_net/delta/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pB2
res_net/delta/truediv_1/yС
res_net/delta/truediv_1RealDiv$res_net/delta/convolution_1:output:0"res_net/delta/truediv_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
res_net/delta/truediv_1
res_net/delta/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2 
res_net/delta/transpose_3/permУ
res_net/delta/transpose_3	Transposeres_net/delta/truediv_1:z:0'res_net/delta/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
res_net/delta/transpose_3l
res_net/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
res_net/concat/axis
res_net/concatConcatV21res_net/mfccs_from_log_mel_spectrograms/mul_1:z:0res_net/delta/transpose_1:y:0res_net/delta/transpose_3:y:0res_net/Sqrt:y:0res_net/concat/axis:output:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@2
res_net/concat
res_net/Squeeze_1Squeezeres_net/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@*
squeeze_dims
2
res_net/Squeeze_1Џ
0res_net/resnet_block/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ22
0res_net/resnet_block/conv1/conv1d/ExpandDims/dimћ
,res_net/resnet_block/conv1/conv1d/ExpandDims
ExpandDimsres_net/Squeeze_1:output:09res_net/resnet_block/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2.
,res_net/resnet_block/conv1/conv1d/ExpandDims
=res_net/resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpFres_net_resnet_block_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02?
=res_net/resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOpЊ
2res_net/resnet_block/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2res_net/resnet_block/conv1/conv1d/ExpandDims_1/dimЃ
.res_net/resnet_block/conv1/conv1d/ExpandDims_1
ExpandDimsEres_net/resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0;res_net/resnet_block/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 20
.res_net/resnet_block/conv1/conv1d/ExpandDims_1Ђ
!res_net/resnet_block/conv1/conv1dConv2D5res_net/resnet_block/conv1/conv1d/ExpandDims:output:07res_net/resnet_block/conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2#
!res_net/resnet_block/conv1/conv1dу
)res_net/resnet_block/conv1/conv1d/SqueezeSqueeze*res_net/resnet_block/conv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2+
)res_net/resnet_block/conv1/conv1d/Squeezeн
1res_net/resnet_block/conv1/BiasAdd/ReadVariableOpReadVariableOp:res_net_resnet_block_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1res_net/resnet_block/conv1/BiasAdd/ReadVariableOpј
"res_net/resnet_block/conv1/BiasAddBiasAdd2res_net/resnet_block/conv1/conv1d/Squeeze:output:09res_net/resnet_block/conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2$
"res_net/resnet_block/conv1/BiasAdd­
res_net/resnet_block/relu1/ReluRelu+res_net/resnet_block/conv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2!
res_net/resnet_block/relu1/ReluЏ
0res_net/resnet_block/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ22
0res_net/resnet_block/conv2/conv1d/ExpandDims/dim
,res_net/resnet_block/conv2/conv1d/ExpandDims
ExpandDims-res_net/resnet_block/relu1/Relu:activations:09res_net/resnet_block/conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2.
,res_net/resnet_block/conv2/conv1d/ExpandDims
=res_net/resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpFres_net_resnet_block_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02?
=res_net/resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOpЊ
2res_net/resnet_block/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2res_net/resnet_block/conv2/conv1d/ExpandDims_1/dimЃ
.res_net/resnet_block/conv2/conv1d/ExpandDims_1
ExpandDimsEres_net/resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0;res_net/resnet_block/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  20
.res_net/resnet_block/conv2/conv1d/ExpandDims_1Ђ
!res_net/resnet_block/conv2/conv1dConv2D5res_net/resnet_block/conv2/conv1d/ExpandDims:output:07res_net/resnet_block/conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2#
!res_net/resnet_block/conv2/conv1dу
)res_net/resnet_block/conv2/conv1d/SqueezeSqueeze*res_net/resnet_block/conv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2+
)res_net/resnet_block/conv2/conv1d/Squeezeн
1res_net/resnet_block/conv2/BiasAdd/ReadVariableOpReadVariableOp:res_net_resnet_block_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1res_net/resnet_block/conv2/BiasAdd/ReadVariableOpј
"res_net/resnet_block/conv2/BiasAddBiasAdd2res_net/resnet_block/conv2/conv1d/Squeeze:output:09res_net/resnet_block/conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2$
"res_net/resnet_block/conv2/BiasAdd­
res_net/resnet_block/relu2/ReluRelu+res_net/resnet_block/conv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2!
res_net/resnet_block/relu2/ReluЏ
0res_net/resnet_block/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ22
0res_net/resnet_block/conv3/conv1d/ExpandDims/dim
,res_net/resnet_block/conv3/conv1d/ExpandDims
ExpandDims-res_net/resnet_block/relu2/Relu:activations:09res_net/resnet_block/conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2.
,res_net/resnet_block/conv3/conv1d/ExpandDims
=res_net/resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpFres_net_resnet_block_conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02?
=res_net/resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOpЊ
2res_net/resnet_block/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2res_net/resnet_block/conv3/conv1d/ExpandDims_1/dimЃ
.res_net/resnet_block/conv3/conv1d/ExpandDims_1
ExpandDimsEres_net/resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0;res_net/resnet_block/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  20
.res_net/resnet_block/conv3/conv1d/ExpandDims_1Ђ
!res_net/resnet_block/conv3/conv1dConv2D5res_net/resnet_block/conv3/conv1d/ExpandDims:output:07res_net/resnet_block/conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2#
!res_net/resnet_block/conv3/conv1dу
)res_net/resnet_block/conv3/conv1d/SqueezeSqueeze*res_net/resnet_block/conv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2+
)res_net/resnet_block/conv3/conv1d/Squeezeн
1res_net/resnet_block/conv3/BiasAdd/ReadVariableOpReadVariableOp:res_net_resnet_block_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1res_net/resnet_block/conv3/BiasAdd/ReadVariableOpј
"res_net/resnet_block/conv3/BiasAddBiasAdd2res_net/resnet_block/conv3/conv1d/Squeeze:output:09res_net/resnet_block/conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2$
"res_net/resnet_block/conv3/BiasAddЕ
3res_net/resnet_block/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ25
3res_net/resnet_block/shortcut/conv1d/ExpandDims/dim
/res_net/resnet_block/shortcut/conv1d/ExpandDims
ExpandDimsres_net/Squeeze_1:output:0<res_net/resnet_block/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@21
/res_net/resnet_block/shortcut/conv1d/ExpandDims
@res_net/resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpIres_net_resnet_block_shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02B
@res_net/resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOpА
5res_net/resnet_block/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 27
5res_net/resnet_block/shortcut/conv1d/ExpandDims_1/dimЏ
1res_net/resnet_block/shortcut/conv1d/ExpandDims_1
ExpandDimsHres_net/resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0>res_net/resnet_block/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 23
1res_net/resnet_block/shortcut/conv1d/ExpandDims_1Ў
$res_net/resnet_block/shortcut/conv1dConv2D8res_net/resnet_block/shortcut/conv1d/ExpandDims:output:0:res_net/resnet_block/shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2&
$res_net/resnet_block/shortcut/conv1dь
,res_net/resnet_block/shortcut/conv1d/SqueezeSqueeze-res_net/resnet_block/shortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2.
,res_net/resnet_block/shortcut/conv1d/Squeezeц
4res_net/resnet_block/shortcut/BiasAdd/ReadVariableOpReadVariableOp=res_net_resnet_block_shortcut_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4res_net/resnet_block/shortcut/BiasAdd/ReadVariableOp
%res_net/resnet_block/shortcut/BiasAddBiasAdd5res_net/resnet_block/shortcut/conv1d/Squeeze:output:0<res_net/resnet_block/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2'
%res_net/resnet_block/shortcut/BiasAddи
res_net/resnet_block/add/addAddV2+res_net/resnet_block/conv3/BiasAdd:output:0.res_net/resnet_block/shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
res_net/resnet_block/add/addЊ
#res_net/resnet_block/out_block/ReluRelu res_net/resnet_block/add/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2%
#res_net/resnet_block/out_block/ReluГ
2res_net/resnet_block_1/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ24
2res_net/resnet_block_1/conv1/conv1d/ExpandDims/dim
.res_net/resnet_block_1/conv1/conv1d/ExpandDims
ExpandDims1res_net/resnet_block/out_block/Relu:activations:0;res_net/resnet_block_1/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 20
.res_net/resnet_block_1/conv1/conv1d/ExpandDims
?res_net/resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHres_net_resnet_block_1_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02A
?res_net/resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOpЎ
4res_net/resnet_block_1/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/resnet_block_1/conv1/conv1d/ExpandDims_1/dimЋ
0res_net/resnet_block_1/conv1/conv1d/ExpandDims_1
ExpandDimsGres_net/resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0=res_net/resnet_block_1/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @22
0res_net/resnet_block_1/conv1/conv1d/ExpandDims_1Њ
#res_net/resnet_block_1/conv1/conv1dConv2D7res_net/resnet_block_1/conv1/conv1d/ExpandDims:output:09res_net/resnet_block_1/conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2%
#res_net/resnet_block_1/conv1/conv1dщ
+res_net/resnet_block_1/conv1/conv1d/SqueezeSqueeze,res_net/resnet_block_1/conv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2-
+res_net/resnet_block_1/conv1/conv1d/Squeezeу
3res_net/resnet_block_1/conv1/BiasAdd/ReadVariableOpReadVariableOp<res_net_resnet_block_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3res_net/resnet_block_1/conv1/BiasAdd/ReadVariableOp
$res_net/resnet_block_1/conv1/BiasAddBiasAdd4res_net/resnet_block_1/conv1/conv1d/Squeeze:output:0;res_net/resnet_block_1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2&
$res_net/resnet_block_1/conv1/BiasAddГ
!res_net/resnet_block_1/relu1/ReluRelu-res_net/resnet_block_1/conv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2#
!res_net/resnet_block_1/relu1/ReluГ
2res_net/resnet_block_1/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ24
2res_net/resnet_block_1/conv2/conv1d/ExpandDims/dim
.res_net/resnet_block_1/conv2/conv1d/ExpandDims
ExpandDims/res_net/resnet_block_1/relu1/Relu:activations:0;res_net/resnet_block_1/conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@20
.res_net/resnet_block_1/conv2/conv1d/ExpandDims
?res_net/resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHres_net_resnet_block_1_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02A
?res_net/resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOpЎ
4res_net/resnet_block_1/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/resnet_block_1/conv2/conv1d/ExpandDims_1/dimЋ
0res_net/resnet_block_1/conv2/conv1d/ExpandDims_1
ExpandDimsGres_net/resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0=res_net/resnet_block_1/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@22
0res_net/resnet_block_1/conv2/conv1d/ExpandDims_1Њ
#res_net/resnet_block_1/conv2/conv1dConv2D7res_net/resnet_block_1/conv2/conv1d/ExpandDims:output:09res_net/resnet_block_1/conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2%
#res_net/resnet_block_1/conv2/conv1dщ
+res_net/resnet_block_1/conv2/conv1d/SqueezeSqueeze,res_net/resnet_block_1/conv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2-
+res_net/resnet_block_1/conv2/conv1d/Squeezeу
3res_net/resnet_block_1/conv2/BiasAdd/ReadVariableOpReadVariableOp<res_net_resnet_block_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3res_net/resnet_block_1/conv2/BiasAdd/ReadVariableOp
$res_net/resnet_block_1/conv2/BiasAddBiasAdd4res_net/resnet_block_1/conv2/conv1d/Squeeze:output:0;res_net/resnet_block_1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2&
$res_net/resnet_block_1/conv2/BiasAddГ
!res_net/resnet_block_1/relu2/ReluRelu-res_net/resnet_block_1/conv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2#
!res_net/resnet_block_1/relu2/ReluГ
2res_net/resnet_block_1/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ24
2res_net/resnet_block_1/conv3/conv1d/ExpandDims/dim
.res_net/resnet_block_1/conv3/conv1d/ExpandDims
ExpandDims/res_net/resnet_block_1/relu2/Relu:activations:0;res_net/resnet_block_1/conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@20
.res_net/resnet_block_1/conv3/conv1d/ExpandDims
?res_net/resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHres_net_resnet_block_1_conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02A
?res_net/resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOpЎ
4res_net/resnet_block_1/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/resnet_block_1/conv3/conv1d/ExpandDims_1/dimЋ
0res_net/resnet_block_1/conv3/conv1d/ExpandDims_1
ExpandDimsGres_net/resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0=res_net/resnet_block_1/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@22
0res_net/resnet_block_1/conv3/conv1d/ExpandDims_1Њ
#res_net/resnet_block_1/conv3/conv1dConv2D7res_net/resnet_block_1/conv3/conv1d/ExpandDims:output:09res_net/resnet_block_1/conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2%
#res_net/resnet_block_1/conv3/conv1dщ
+res_net/resnet_block_1/conv3/conv1d/SqueezeSqueeze,res_net/resnet_block_1/conv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2-
+res_net/resnet_block_1/conv3/conv1d/Squeezeу
3res_net/resnet_block_1/conv3/BiasAdd/ReadVariableOpReadVariableOp<res_net_resnet_block_1_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3res_net/resnet_block_1/conv3/BiasAdd/ReadVariableOp
$res_net/resnet_block_1/conv3/BiasAddBiasAdd4res_net/resnet_block_1/conv3/conv1d/Squeeze:output:0;res_net/resnet_block_1/conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2&
$res_net/resnet_block_1/conv3/BiasAddЙ
5res_net/resnet_block_1/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ27
5res_net/resnet_block_1/shortcut/conv1d/ExpandDims/dimЁ
1res_net/resnet_block_1/shortcut/conv1d/ExpandDims
ExpandDims1res_net/resnet_block/out_block/Relu:activations:0>res_net/resnet_block_1/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 23
1res_net/resnet_block_1/shortcut/conv1d/ExpandDims
Bres_net/resnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKres_net_resnet_block_1_shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02D
Bres_net/resnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOpД
7res_net/resnet_block_1/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7res_net/resnet_block_1/shortcut/conv1d/ExpandDims_1/dimЗ
3res_net/resnet_block_1/shortcut/conv1d/ExpandDims_1
ExpandDimsJres_net/resnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0@res_net/resnet_block_1/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @25
3res_net/resnet_block_1/shortcut/conv1d/ExpandDims_1Ж
&res_net/resnet_block_1/shortcut/conv1dConv2D:res_net/resnet_block_1/shortcut/conv1d/ExpandDims:output:0<res_net/resnet_block_1/shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2(
&res_net/resnet_block_1/shortcut/conv1dђ
.res_net/resnet_block_1/shortcut/conv1d/SqueezeSqueeze/res_net/resnet_block_1/shortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ20
.res_net/resnet_block_1/shortcut/conv1d/Squeezeь
6res_net/resnet_block_1/shortcut/BiasAdd/ReadVariableOpReadVariableOp?res_net_resnet_block_1_shortcut_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6res_net/resnet_block_1/shortcut/BiasAdd/ReadVariableOp
'res_net/resnet_block_1/shortcut/BiasAddBiasAdd7res_net/resnet_block_1/shortcut/conv1d/Squeeze:output:0>res_net/resnet_block_1/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2)
'res_net/resnet_block_1/shortcut/BiasAddф
 res_net/resnet_block_1/add_1/addAddV2-res_net/resnet_block_1/conv3/BiasAdd:output:00res_net/resnet_block_1/shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2"
 res_net/resnet_block_1/add_1/addВ
%res_net/resnet_block_1/out_block/ReluRelu$res_net/resnet_block_1/add_1/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2'
%res_net/resnet_block_1/out_block/ReluГ
2res_net/resnet_block_2/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ24
2res_net/resnet_block_2/conv1/conv1d/ExpandDims/dim
.res_net/resnet_block_2/conv1/conv1d/ExpandDims
ExpandDims3res_net/resnet_block_1/out_block/Relu:activations:0;res_net/resnet_block_2/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@20
.res_net/resnet_block_2/conv1/conv1d/ExpandDims
?res_net/resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHres_net_resnet_block_2_conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02A
?res_net/resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOpЎ
4res_net/resnet_block_2/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/resnet_block_2/conv1/conv1d/ExpandDims_1/dimЌ
0res_net/resnet_block_2/conv1/conv1d/ExpandDims_1
ExpandDimsGres_net/resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0=res_net/resnet_block_2/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@22
0res_net/resnet_block_2/conv1/conv1d/ExpandDims_1Ћ
#res_net/resnet_block_2/conv1/conv1dConv2D7res_net/resnet_block_2/conv1/conv1d/ExpandDims:output:09res_net/resnet_block_2/conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2%
#res_net/resnet_block_2/conv1/conv1dъ
+res_net/resnet_block_2/conv1/conv1d/SqueezeSqueeze,res_net/resnet_block_2/conv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2-
+res_net/resnet_block_2/conv1/conv1d/Squeezeф
3res_net/resnet_block_2/conv1/BiasAdd/ReadVariableOpReadVariableOp<res_net_resnet_block_2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3res_net/resnet_block_2/conv1/BiasAdd/ReadVariableOp
$res_net/resnet_block_2/conv1/BiasAddBiasAdd4res_net/resnet_block_2/conv1/conv1d/Squeeze:output:0;res_net/resnet_block_2/conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2&
$res_net/resnet_block_2/conv1/BiasAddД
!res_net/resnet_block_2/relu1/ReluRelu-res_net/resnet_block_2/conv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2#
!res_net/resnet_block_2/relu1/ReluГ
2res_net/resnet_block_2/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ24
2res_net/resnet_block_2/conv2/conv1d/ExpandDims/dim
.res_net/resnet_block_2/conv2/conv1d/ExpandDims
ExpandDims/res_net/resnet_block_2/relu1/Relu:activations:0;res_net/resnet_block_2/conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@20
.res_net/resnet_block_2/conv2/conv1d/ExpandDims
?res_net/resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHres_net_resnet_block_2_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02A
?res_net/resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOpЎ
4res_net/resnet_block_2/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/resnet_block_2/conv2/conv1d/ExpandDims_1/dim­
0res_net/resnet_block_2/conv2/conv1d/ExpandDims_1
ExpandDimsGres_net/resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0=res_net/resnet_block_2/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:22
0res_net/resnet_block_2/conv2/conv1d/ExpandDims_1Ћ
#res_net/resnet_block_2/conv2/conv1dConv2D7res_net/resnet_block_2/conv2/conv1d/ExpandDims:output:09res_net/resnet_block_2/conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2%
#res_net/resnet_block_2/conv2/conv1dъ
+res_net/resnet_block_2/conv2/conv1d/SqueezeSqueeze,res_net/resnet_block_2/conv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2-
+res_net/resnet_block_2/conv2/conv1d/Squeezeф
3res_net/resnet_block_2/conv2/BiasAdd/ReadVariableOpReadVariableOp<res_net_resnet_block_2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3res_net/resnet_block_2/conv2/BiasAdd/ReadVariableOp
$res_net/resnet_block_2/conv2/BiasAddBiasAdd4res_net/resnet_block_2/conv2/conv1d/Squeeze:output:0;res_net/resnet_block_2/conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2&
$res_net/resnet_block_2/conv2/BiasAddД
!res_net/resnet_block_2/relu2/ReluRelu-res_net/resnet_block_2/conv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2#
!res_net/resnet_block_2/relu2/ReluГ
2res_net/resnet_block_2/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ24
2res_net/resnet_block_2/conv3/conv1d/ExpandDims/dim
.res_net/resnet_block_2/conv3/conv1d/ExpandDims
ExpandDims/res_net/resnet_block_2/relu2/Relu:activations:0;res_net/resnet_block_2/conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@20
.res_net/resnet_block_2/conv3/conv1d/ExpandDims
?res_net/resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHres_net_resnet_block_2_conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02A
?res_net/resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOpЎ
4res_net/resnet_block_2/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/resnet_block_2/conv3/conv1d/ExpandDims_1/dim­
0res_net/resnet_block_2/conv3/conv1d/ExpandDims_1
ExpandDimsGres_net/resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0=res_net/resnet_block_2/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:22
0res_net/resnet_block_2/conv3/conv1d/ExpandDims_1Ћ
#res_net/resnet_block_2/conv3/conv1dConv2D7res_net/resnet_block_2/conv3/conv1d/ExpandDims:output:09res_net/resnet_block_2/conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2%
#res_net/resnet_block_2/conv3/conv1dъ
+res_net/resnet_block_2/conv3/conv1d/SqueezeSqueeze,res_net/resnet_block_2/conv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2-
+res_net/resnet_block_2/conv3/conv1d/Squeezeф
3res_net/resnet_block_2/conv3/BiasAdd/ReadVariableOpReadVariableOp<res_net_resnet_block_2_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3res_net/resnet_block_2/conv3/BiasAdd/ReadVariableOp
$res_net/resnet_block_2/conv3/BiasAddBiasAdd4res_net/resnet_block_2/conv3/conv1d/Squeeze:output:0;res_net/resnet_block_2/conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2&
$res_net/resnet_block_2/conv3/BiasAddЙ
5res_net/resnet_block_2/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ27
5res_net/resnet_block_2/shortcut/conv1d/ExpandDims/dimЃ
1res_net/resnet_block_2/shortcut/conv1d/ExpandDims
ExpandDims3res_net/resnet_block_1/out_block/Relu:activations:0>res_net/resnet_block_2/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@23
1res_net/resnet_block_2/shortcut/conv1d/ExpandDims
Bres_net/resnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKres_net_resnet_block_2_shortcut_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02D
Bres_net/resnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOpД
7res_net/resnet_block_2/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7res_net/resnet_block_2/shortcut/conv1d/ExpandDims_1/dimИ
3res_net/resnet_block_2/shortcut/conv1d/ExpandDims_1
ExpandDimsJres_net/resnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0@res_net/resnet_block_2/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@25
3res_net/resnet_block_2/shortcut/conv1d/ExpandDims_1З
&res_net/resnet_block_2/shortcut/conv1dConv2D:res_net/resnet_block_2/shortcut/conv1d/ExpandDims:output:0<res_net/resnet_block_2/shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2(
&res_net/resnet_block_2/shortcut/conv1dѓ
.res_net/resnet_block_2/shortcut/conv1d/SqueezeSqueeze/res_net/resnet_block_2/shortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ20
.res_net/resnet_block_2/shortcut/conv1d/Squeezeэ
6res_net/resnet_block_2/shortcut/BiasAdd/ReadVariableOpReadVariableOp?res_net_resnet_block_2_shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype028
6res_net/resnet_block_2/shortcut/BiasAdd/ReadVariableOp
'res_net/resnet_block_2/shortcut/BiasAddBiasAdd7res_net/resnet_block_2/shortcut/conv1d/Squeeze:output:0>res_net/resnet_block_2/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2)
'res_net/resnet_block_2/shortcut/BiasAddх
 res_net/resnet_block_2/add_2/addAddV2-res_net/resnet_block_2/conv3/BiasAdd:output:00res_net/resnet_block_2/shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2"
 res_net/resnet_block_2/add_2/addГ
%res_net/resnet_block_2/out_block/ReluRelu$res_net/resnet_block_2/add_2/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2'
%res_net/resnet_block_2/out_block/ReluГ
2res_net/resnet_block_3/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ24
2res_net/resnet_block_3/conv1/conv1d/ExpandDims/dim
.res_net/resnet_block_3/conv1/conv1d/ExpandDims
ExpandDims3res_net/resnet_block_2/out_block/Relu:activations:0;res_net/resnet_block_3/conv1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@20
.res_net/resnet_block_3/conv1/conv1d/ExpandDims
?res_net/resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHres_net_resnet_block_3_conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02A
?res_net/resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOpЎ
4res_net/resnet_block_3/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/resnet_block_3/conv1/conv1d/ExpandDims_1/dim­
0res_net/resnet_block_3/conv1/conv1d/ExpandDims_1
ExpandDimsGres_net/resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0=res_net/resnet_block_3/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:22
0res_net/resnet_block_3/conv1/conv1d/ExpandDims_1Ћ
#res_net/resnet_block_3/conv1/conv1dConv2D7res_net/resnet_block_3/conv1/conv1d/ExpandDims:output:09res_net/resnet_block_3/conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2%
#res_net/resnet_block_3/conv1/conv1dъ
+res_net/resnet_block_3/conv1/conv1d/SqueezeSqueeze,res_net/resnet_block_3/conv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2-
+res_net/resnet_block_3/conv1/conv1d/Squeezeф
3res_net/resnet_block_3/conv1/BiasAdd/ReadVariableOpReadVariableOp<res_net_resnet_block_3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3res_net/resnet_block_3/conv1/BiasAdd/ReadVariableOp
$res_net/resnet_block_3/conv1/BiasAddBiasAdd4res_net/resnet_block_3/conv1/conv1d/Squeeze:output:0;res_net/resnet_block_3/conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2&
$res_net/resnet_block_3/conv1/BiasAddД
!res_net/resnet_block_3/relu1/ReluRelu-res_net/resnet_block_3/conv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2#
!res_net/resnet_block_3/relu1/ReluГ
2res_net/resnet_block_3/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ24
2res_net/resnet_block_3/conv2/conv1d/ExpandDims/dim
.res_net/resnet_block_3/conv2/conv1d/ExpandDims
ExpandDims/res_net/resnet_block_3/relu1/Relu:activations:0;res_net/resnet_block_3/conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@20
.res_net/resnet_block_3/conv2/conv1d/ExpandDims
?res_net/resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHres_net_resnet_block_3_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02A
?res_net/resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOpЎ
4res_net/resnet_block_3/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/resnet_block_3/conv2/conv1d/ExpandDims_1/dim­
0res_net/resnet_block_3/conv2/conv1d/ExpandDims_1
ExpandDimsGres_net/resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0=res_net/resnet_block_3/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:22
0res_net/resnet_block_3/conv2/conv1d/ExpandDims_1Ћ
#res_net/resnet_block_3/conv2/conv1dConv2D7res_net/resnet_block_3/conv2/conv1d/ExpandDims:output:09res_net/resnet_block_3/conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2%
#res_net/resnet_block_3/conv2/conv1dъ
+res_net/resnet_block_3/conv2/conv1d/SqueezeSqueeze,res_net/resnet_block_3/conv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2-
+res_net/resnet_block_3/conv2/conv1d/Squeezeф
3res_net/resnet_block_3/conv2/BiasAdd/ReadVariableOpReadVariableOp<res_net_resnet_block_3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3res_net/resnet_block_3/conv2/BiasAdd/ReadVariableOp
$res_net/resnet_block_3/conv2/BiasAddBiasAdd4res_net/resnet_block_3/conv2/conv1d/Squeeze:output:0;res_net/resnet_block_3/conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2&
$res_net/resnet_block_3/conv2/BiasAddД
!res_net/resnet_block_3/relu2/ReluRelu-res_net/resnet_block_3/conv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2#
!res_net/resnet_block_3/relu2/ReluГ
2res_net/resnet_block_3/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ24
2res_net/resnet_block_3/conv3/conv1d/ExpandDims/dim
.res_net/resnet_block_3/conv3/conv1d/ExpandDims
ExpandDims/res_net/resnet_block_3/relu2/Relu:activations:0;res_net/resnet_block_3/conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@20
.res_net/resnet_block_3/conv3/conv1d/ExpandDims
?res_net/resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHres_net_resnet_block_3_conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02A
?res_net/resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOpЎ
4res_net/resnet_block_3/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4res_net/resnet_block_3/conv3/conv1d/ExpandDims_1/dim­
0res_net/resnet_block_3/conv3/conv1d/ExpandDims_1
ExpandDimsGres_net/resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0=res_net/resnet_block_3/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:22
0res_net/resnet_block_3/conv3/conv1d/ExpandDims_1Ћ
#res_net/resnet_block_3/conv3/conv1dConv2D7res_net/resnet_block_3/conv3/conv1d/ExpandDims:output:09res_net/resnet_block_3/conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2%
#res_net/resnet_block_3/conv3/conv1dъ
+res_net/resnet_block_3/conv3/conv1d/SqueezeSqueeze,res_net/resnet_block_3/conv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2-
+res_net/resnet_block_3/conv3/conv1d/Squeezeф
3res_net/resnet_block_3/conv3/BiasAdd/ReadVariableOpReadVariableOp<res_net_resnet_block_3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3res_net/resnet_block_3/conv3/BiasAdd/ReadVariableOp
$res_net/resnet_block_3/conv3/BiasAddBiasAdd4res_net/resnet_block_3/conv3/conv1d/Squeeze:output:0;res_net/resnet_block_3/conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2&
$res_net/resnet_block_3/conv3/BiasAddЙ
5res_net/resnet_block_3/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ27
5res_net/resnet_block_3/shortcut/conv1d/ExpandDims/dimЄ
1res_net/resnet_block_3/shortcut/conv1d/ExpandDims
ExpandDims3res_net/resnet_block_2/out_block/Relu:activations:0>res_net/resnet_block_3/shortcut/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@23
1res_net/resnet_block_3/shortcut/conv1d/ExpandDims
Bres_net/resnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKres_net_resnet_block_3_shortcut_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02D
Bres_net/resnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOpД
7res_net/resnet_block_3/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7res_net/resnet_block_3/shortcut/conv1d/ExpandDims_1/dimЙ
3res_net/resnet_block_3/shortcut/conv1d/ExpandDims_1
ExpandDimsJres_net/resnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0@res_net/resnet_block_3/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:25
3res_net/resnet_block_3/shortcut/conv1d/ExpandDims_1З
&res_net/resnet_block_3/shortcut/conv1dConv2D:res_net/resnet_block_3/shortcut/conv1d/ExpandDims:output:0<res_net/resnet_block_3/shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2(
&res_net/resnet_block_3/shortcut/conv1dѓ
.res_net/resnet_block_3/shortcut/conv1d/SqueezeSqueeze/res_net/resnet_block_3/shortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ20
.res_net/resnet_block_3/shortcut/conv1d/Squeezeэ
6res_net/resnet_block_3/shortcut/BiasAdd/ReadVariableOpReadVariableOp?res_net_resnet_block_3_shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype028
6res_net/resnet_block_3/shortcut/BiasAdd/ReadVariableOp
'res_net/resnet_block_3/shortcut/BiasAddBiasAdd7res_net/resnet_block_3/shortcut/conv1d/Squeeze:output:0>res_net/resnet_block_3/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2)
'res_net/resnet_block_3/shortcut/BiasAddх
 res_net/resnet_block_3/add_3/addAddV2-res_net/resnet_block_3/conv3/BiasAdd:output:00res_net/resnet_block_3/shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2"
 res_net/resnet_block_3/add_3/addГ
%res_net/resnet_block_3/out_block/ReluRelu$res_net/resnet_block_3/add_3/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2'
%res_net/resnet_block_3/out_block/Relu
res_net/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
res_net/flatten/ConstХ
res_net/flatten/ReshapeReshape3res_net/resnet_block_3/out_block/Relu:activations:0res_net/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ@2
res_net/flatten/ReshapeГ
!res_net/fc1/MatMul/ReadVariableOpReadVariableOp*res_net_fc1_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype02#
!res_net/fc1/MatMul/ReadVariableOpВ
res_net/fc1/MatMulMatMul res_net/flatten/Reshape:output:0)res_net/fc1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
res_net/fc1/MatMulБ
"res_net/fc1/BiasAdd/ReadVariableOpReadVariableOp+res_net_fc1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"res_net/fc1/BiasAdd/ReadVariableOpВ
res_net/fc1/BiasAddBiasAddres_net/fc1/MatMul:product:0*res_net/fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
res_net/fc1/BiasAdd}
res_net/fc1/ReluRelures_net/fc1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
res_net/fc1/ReluГ
!res_net/fc2/MatMul/ReadVariableOpReadVariableOp*res_net_fc2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!res_net/fc2/MatMul/ReadVariableOpА
res_net/fc2/MatMulMatMulres_net/fc1/Relu:activations:0)res_net/fc2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
res_net/fc2/MatMulБ
"res_net/fc2/BiasAdd/ReadVariableOpReadVariableOp+res_net_fc2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"res_net/fc2/BiasAdd/ReadVariableOpВ
res_net/fc2/BiasAddBiasAddres_net/fc2/MatMul:product:0*res_net/fc2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
res_net/fc2/BiasAdd}
res_net/fc2/ReluRelures_net/fc2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
res_net/fc2/ReluВ
!res_net/fc3/MatMul/ReadVariableOpReadVariableOp*res_net_fc3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!res_net/fc3/MatMul/ReadVariableOpЏ
res_net/fc3/MatMulMatMulres_net/fc2/Relu:activations:0)res_net/fc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
res_net/fc3/MatMulА
"res_net/fc3/BiasAdd/ReadVariableOpReadVariableOp+res_net_fc3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"res_net/fc3/BiasAdd/ReadVariableOpБ
res_net/fc3/BiasAddBiasAddres_net/fc3/MatMul:product:0*res_net/fc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
res_net/fc3/BiasAdd
res_net/fc3/SigmoidSigmoidres_net/fc3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
res_net/fc3/Sigmoidk
IdentityIdentityres_net/fc3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:џџџџџџџџџ:	:::::::::::::::::::::::::::::::::::::::U Q
,
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:%!

_output_shapes
:	
ц
{
&__inference_conv1_layer_call_fn_217314

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2101232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Л
о
/__inference_resnet_block_2_layer_call_fn_216906

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_2114132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ъ
С
$__inference_signature_wrapper_213674
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37
identityЂStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_2100892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:џџџџџџџџџ:	::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:%!

_output_shapes
:	
Ж
Ж
A__inference_conv3_layer_call_and_return_conditional_losses_210279

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@ :::S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
ћ;
ї
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_217177

inputs5
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЉ
conv1/conv1d/ExpandDims
ExpandDimsinputs$conv1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1/conv1d/ExpandDimsЬ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimб
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1/conv1d/ExpandDims_1Я
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1/conv1dЅ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv1/BiasAdd/ReadVariableOpЅ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv1/BiasAddo

relu1/ReluReluconv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimЛ
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv2/conv1d/ExpandDimsЬ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimб
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv2/conv1d/ExpandDims_1Я
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2/conv1dЅ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2/BiasAdd/ReadVariableOpЅ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv2/BiasAddo

relu2/ReluReluconv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimЛ
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv3/conv1d/ExpandDimsЬ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimб
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv3/conv1d/ExpandDims_1Я
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv3/conv1dЅ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv3/BiasAdd/ReadVariableOpЅ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimВ
shortcut/conv1d/ExpandDims
ExpandDimsinputs'shortcut/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
shortcut/conv1d/ExpandDimsе
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimн
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
shortcut/conv1d/ExpandDims_1л
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
shortcut/conv1dЎ
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЈ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
shortcut/BiasAdd/ReadVariableOpБ
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
add/addl
out_block/ReluReluadd/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
out_block/Reluu
IdentityIdentityout_block/Relu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:џџџџџџџџџ@:::::::::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Є
F
*__inference_out_block_layer_call_fn_217542

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_out_block_layer_call_and_return_conditional_losses_2108412
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@@:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
У
a
E__inference_out_block_layer_call_and_return_conditional_losses_217537

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџ@@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@@:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
џ;
ј
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_216979
input_15
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЊ
conv1/conv1d/ExpandDims
ExpandDimsinput_1$conv1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1/conv1d/ExpandDimsЬ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimб
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1/conv1d/ExpandDims_1Я
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1/conv1dЅ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv1/BiasAdd/ReadVariableOpЅ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv1/BiasAddo

relu1/ReluReluconv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimЛ
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv2/conv1d/ExpandDimsЬ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimб
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv2/conv1d/ExpandDims_1Я
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2/conv1dЅ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2/BiasAdd/ReadVariableOpЅ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv2/BiasAddo

relu2/ReluReluconv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimЛ
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv3/conv1d/ExpandDimsЬ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimб
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv3/conv1d/ExpandDims_1Я
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv3/conv1dЅ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv3/BiasAdd/ReadVariableOpЅ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimГ
shortcut/conv1d/ExpandDims
ExpandDimsinput_1'shortcut/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
shortcut/conv1d/ExpandDimsе
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimн
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
shortcut/conv1d/ExpandDims_1л
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
shortcut/conv1dЎ
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЈ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
shortcut/BiasAdd/ReadVariableOpБ
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
add/addl
out_block/ReluReluadd/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
out_block/Reluu
IdentityIdentityout_block/Relu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:џџџџџџџџџ@:::::::::U Q
,
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_1
У
]
A__inference_relu2_layer_call_and_return_conditional_losses_217605

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:џџџџџџџџџ@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

B
&__inference_relu1_layer_call_fn_217324

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu1_layer_call_and_return_conditional_losses_2101582
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@ :S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
И
н
-__inference_resnet_block_layer_call_fn_216176
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_resnet_block_layer_call_and_return_conditional_losses_2104712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_1
Ж
Ж
A__inference_conv2_layer_call_and_return_conditional_losses_210672

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@@:::S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs

`
4__inference_log_mel_spectrogram_layer_call_fn_216026
	waveforms
unknown
identityт
PartitionedCallPartitionedCall	waveformsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_log_mel_spectrogram_layer_call_and_return_conditional_losses_2123162
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ:	:S O
(
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	waveforms:%!

_output_shapes
:	

B
&__inference_relu2_layer_call_fn_217484

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu2_layer_call_and_return_conditional_losses_2107072
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@@:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
У
]
A__inference_relu2_layer_call_and_return_conditional_losses_211649

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:џџџџџџџџџ@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Н
Ж
A__inference_conv1_layer_call_and_return_conditional_losses_211065

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv1d/ExpandDimsЙ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimИ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@@:::S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ъ
{
&__inference_conv2_layer_call_fn_217726

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2116142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ў
џ
C__inference_res_net_layer_call_and_return_conditional_losses_215684

inputs 
log_mel_spectrogram_matmul_bB
>resnet_block_conv1_conv1d_expanddims_1_readvariableop_resource6
2resnet_block_conv1_biasadd_readvariableop_resourceB
>resnet_block_conv2_conv1d_expanddims_1_readvariableop_resource6
2resnet_block_conv2_biasadd_readvariableop_resourceB
>resnet_block_conv3_conv1d_expanddims_1_readvariableop_resource6
2resnet_block_conv3_biasadd_readvariableop_resourceE
Aresnet_block_shortcut_conv1d_expanddims_1_readvariableop_resource9
5resnet_block_shortcut_biasadd_readvariableop_resourceD
@resnet_block_1_conv1_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_1_conv1_biasadd_readvariableop_resourceD
@resnet_block_1_conv2_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_1_conv2_biasadd_readvariableop_resourceD
@resnet_block_1_conv3_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_1_conv3_biasadd_readvariableop_resourceG
Cresnet_block_1_shortcut_conv1d_expanddims_1_readvariableop_resource;
7resnet_block_1_shortcut_biasadd_readvariableop_resourceD
@resnet_block_2_conv1_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_2_conv1_biasadd_readvariableop_resourceD
@resnet_block_2_conv2_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_2_conv2_biasadd_readvariableop_resourceD
@resnet_block_2_conv3_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_2_conv3_biasadd_readvariableop_resourceG
Cresnet_block_2_shortcut_conv1d_expanddims_1_readvariableop_resource;
7resnet_block_2_shortcut_biasadd_readvariableop_resourceD
@resnet_block_3_conv1_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_3_conv1_biasadd_readvariableop_resourceD
@resnet_block_3_conv2_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_3_conv2_biasadd_readvariableop_resourceD
@resnet_block_3_conv3_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_3_conv3_biasadd_readvariableop_resourceG
Cresnet_block_3_shortcut_conv1d_expanddims_1_readvariableop_resource;
7resnet_block_3_shortcut_biasadd_readvariableop_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource&
"fc2_matmul_readvariableop_resource'
#fc2_biasadd_readvariableop_resource&
"fc3_matmul_readvariableop_resource'
#fc3_biasadd_readvariableop_resource
identityo
SqueezeSqueezeinputs*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2	
Squeeze
%log_mel_spectrogram/stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2'
%log_mel_spectrogram/stft/frame_length
#log_mel_spectrogram/stft/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2%
#log_mel_spectrogram/stft/frame_step
log_mel_spectrogram/stft/ConstConst*
_output_shapes
: *
dtype0*
value
B :2 
log_mel_spectrogram/stft/Const
#log_mel_spectrogram/stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#log_mel_spectrogram/stft/frame/axis
$log_mel_spectrogram/stft/frame/ShapeShapeSqueeze:output:0*
T0*
_output_shapes
:2&
$log_mel_spectrogram/stft/frame/Shape
#log_mel_spectrogram/stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#log_mel_spectrogram/stft/frame/Rank
*log_mel_spectrogram/stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*log_mel_spectrogram/stft/frame/range/start
*log_mel_spectrogram/stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*log_mel_spectrogram/stft/frame/range/delta
$log_mel_spectrogram/stft/frame/rangeRange3log_mel_spectrogram/stft/frame/range/start:output:0,log_mel_spectrogram/stft/frame/Rank:output:03log_mel_spectrogram/stft/frame/range/delta:output:0*
_output_shapes
:2&
$log_mel_spectrogram/stft/frame/rangeЛ
2log_mel_spectrogram/stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ24
2log_mel_spectrogram/stft/frame/strided_slice/stackЖ
4log_mel_spectrogram/stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4log_mel_spectrogram/stft/frame/strided_slice/stack_1Ж
4log_mel_spectrogram/stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4log_mel_spectrogram/stft/frame/strided_slice/stack_2
,log_mel_spectrogram/stft/frame/strided_sliceStridedSlice-log_mel_spectrogram/stft/frame/range:output:0;log_mel_spectrogram/stft/frame/strided_slice/stack:output:0=log_mel_spectrogram/stft/frame/strided_slice/stack_1:output:0=log_mel_spectrogram/stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,log_mel_spectrogram/stft/frame/strided_slice
$log_mel_spectrogram/stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$log_mel_spectrogram/stft/frame/sub/yЭ
"log_mel_spectrogram/stft/frame/subSub,log_mel_spectrogram/stft/frame/Rank:output:0-log_mel_spectrogram/stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/subг
$log_mel_spectrogram/stft/frame/sub_1Sub&log_mel_spectrogram/stft/frame/sub:z:05log_mel_spectrogram/stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/sub_1
'log_mel_spectrogram/stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'log_mel_spectrogram/stft/frame/packed/1
%log_mel_spectrogram/stft/frame/packedPack5log_mel_spectrogram/stft/frame/strided_slice:output:00log_mel_spectrogram/stft/frame/packed/1:output:0(log_mel_spectrogram/stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2'
%log_mel_spectrogram/stft/frame/packedЂ
.log_mel_spectrogram/stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.log_mel_spectrogram/stft/frame/split/split_dimК
$log_mel_spectrogram/stft/frame/splitSplitV-log_mel_spectrogram/stft/frame/Shape:output:0.log_mel_spectrogram/stft/frame/packed:output:07log_mel_spectrogram/stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2&
$log_mel_spectrogram/stft/frame/split
,log_mel_spectrogram/stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2.
,log_mel_spectrogram/stft/frame/Reshape/shapeЃ
.log_mel_spectrogram/stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 20
.log_mel_spectrogram/stft/frame/Reshape/shape_1ф
&log_mel_spectrogram/stft/frame/ReshapeReshape-log_mel_spectrogram/stft/frame/split:output:17log_mel_spectrogram/stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2(
&log_mel_spectrogram/stft/frame/Reshape
#log_mel_spectrogram/stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2%
#log_mel_spectrogram/stft/frame/Size
%log_mel_spectrogram/stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%log_mel_spectrogram/stft/frame/Size_1
$log_mel_spectrogram/stft/frame/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$log_mel_spectrogram/stft/frame/ConstЁ
"log_mel_spectrogram/stft/frame/NegNeg/log_mel_spectrogram/stft/frame/Reshape:output:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/Negе
'log_mel_spectrogram/stft/frame/floordivFloorDiv&log_mel_spectrogram/stft/frame/Neg:y:0,log_mel_spectrogram/stft/frame_step:output:0*
T0*
_output_shapes
: 2)
'log_mel_spectrogram/stft/frame/floordivЁ
$log_mel_spectrogram/stft/frame/Neg_1Neg+log_mel_spectrogram/stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/Neg_1
&log_mel_spectrogram/stft/frame/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_spectrogram/stft/frame/sub_2/yЯ
$log_mel_spectrogram/stft/frame/sub_2Sub(log_mel_spectrogram/stft/frame/Neg_1:y:0/log_mel_spectrogram/stft/frame/sub_2/y:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/sub_2Ш
"log_mel_spectrogram/stft/frame/mulMul,log_mel_spectrogram/stft/frame_step:output:0(log_mel_spectrogram/stft/frame/sub_2:z:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/mulЪ
"log_mel_spectrogram/stft/frame/addAddV2.log_mel_spectrogram/stft/frame_length:output:0&log_mel_spectrogram/stft/frame/mul:z:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/addЭ
$log_mel_spectrogram/stft/frame/sub_3Sub&log_mel_spectrogram/stft/frame/add:z:0/log_mel_spectrogram/stft/frame/Reshape:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/sub_3
(log_mel_spectrogram/stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2*
(log_mel_spectrogram/stft/frame/Maximum/xй
&log_mel_spectrogram/stft/frame/MaximumMaximum1log_mel_spectrogram/stft/frame/Maximum/x:output:0(log_mel_spectrogram/stft/frame/sub_3:z:0*
T0*
_output_shapes
: 2(
&log_mel_spectrogram/stft/frame/Maximum
*log_mel_spectrogram/stft/frame/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*log_mel_spectrogram/stft/frame/zeros/mul/yп
(log_mel_spectrogram/stft/frame/zeros/mulMul,log_mel_spectrogram/stft/frame/Size:output:03log_mel_spectrogram/stft/frame/zeros/mul/y:output:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/frame/zeros/mul
+log_mel_spectrogram/stft/frame/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2-
+log_mel_spectrogram/stft/frame/zeros/Less/yу
)log_mel_spectrogram/stft/frame/zeros/LessLess,log_mel_spectrogram/stft/frame/zeros/mul:z:04log_mel_spectrogram/stft/frame/zeros/Less/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/zeros/Less 
-log_mel_spectrogram/stft/frame/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-log_mel_spectrogram/stft/frame/zeros/packed/1і
+log_mel_spectrogram/stft/frame/zeros/packedPack,log_mel_spectrogram/stft/frame/Size:output:06log_mel_spectrogram/stft/frame/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2-
+log_mel_spectrogram/stft/frame/zeros/packed
*log_mel_spectrogram/stft/frame/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2,
*log_mel_spectrogram/stft/frame/zeros/Constш
$log_mel_spectrogram/stft/frame/zerosFill4log_mel_spectrogram/stft/frame/zeros/packed:output:03log_mel_spectrogram/stft/frame/zeros/Const:output:0*
T0*
_output_shapes

:2&
$log_mel_spectrogram/stft/frame/zeros
+log_mel_spectrogram/stft/frame/packed_1/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2-
+log_mel_spectrogram/stft/frame/packed_1/0/0ю
)log_mel_spectrogram/stft/frame/packed_1/0Pack4log_mel_spectrogram/stft/frame/packed_1/0/0:output:0*log_mel_spectrogram/stft/frame/Maximum:z:0*
N*
T0*
_output_shapes
:2+
)log_mel_spectrogram/stft/frame/packed_1/0Р
'log_mel_spectrogram/stft/frame/packed_1Pack2log_mel_spectrogram/stft/frame/packed_1/0:output:0*
N*
T0*
_output_shapes

:2)
'log_mel_spectrogram/stft/frame/packed_1
,log_mel_spectrogram/stft/frame/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/frame/zeros_1/mul/yч
*log_mel_spectrogram/stft/frame/zeros_1/mulMul.log_mel_spectrogram/stft/frame/Size_1:output:05log_mel_spectrogram/stft/frame/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2,
*log_mel_spectrogram/stft/frame/zeros_1/mulЁ
-log_mel_spectrogram/stft/frame/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2/
-log_mel_spectrogram/stft/frame/zeros_1/Less/yы
+log_mel_spectrogram/stft/frame/zeros_1/LessLess.log_mel_spectrogram/stft/frame/zeros_1/mul:z:06log_mel_spectrogram/stft/frame/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2-
+log_mel_spectrogram/stft/frame/zeros_1/LessЄ
/log_mel_spectrogram/stft/frame/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :21
/log_mel_spectrogram/stft/frame/zeros_1/packed/1ў
-log_mel_spectrogram/stft/frame/zeros_1/packedPack.log_mel_spectrogram/stft/frame/Size_1:output:08log_mel_spectrogram/stft/frame/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2/
-log_mel_spectrogram/stft/frame/zeros_1/packed
,log_mel_spectrogram/stft/frame/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/zeros_1/Constю
&log_mel_spectrogram/stft/frame/zeros_1Fill6log_mel_spectrogram/stft/frame/zeros_1/packed:output:05log_mel_spectrogram/stft/frame/zeros_1/Const:output:0*
T0*
_output_shapes

: 2(
&log_mel_spectrogram/stft/frame/zeros_1
*log_mel_spectrogram/stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*log_mel_spectrogram/stft/frame/concat/axisг
%log_mel_spectrogram/stft/frame/concatConcatV2-log_mel_spectrogram/stft/frame/zeros:output:00log_mel_spectrogram/stft/frame/packed_1:output:0/log_mel_spectrogram/stft/frame/zeros_1:output:03log_mel_spectrogram/stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes

:2'
%log_mel_spectrogram/stft/frame/concat
$log_mel_spectrogram/stft/frame/PadV2PadV2Squeeze:output:0.log_mel_spectrogram/stft/frame/concat:output:0-log_mel_spectrogram/stft/frame/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2&
$log_mel_spectrogram/stft/frame/PadV2­
&log_mel_spectrogram/stft/frame/Shape_1Shape-log_mel_spectrogram/stft/frame/PadV2:output:0*
T0*
_output_shapes
:2(
&log_mel_spectrogram/stft/frame/Shape_1
&log_mel_spectrogram/stft/frame/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_spectrogram/stft/frame/add_1/yо
$log_mel_spectrogram/stft/frame/add_1AddV25log_mel_spectrogram/stft/frame/strided_slice:output:0/log_mel_spectrogram/stft/frame/add_1/y:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/add_1й
4log_mel_spectrogram/stft/frame/strided_slice_1/stackPack5log_mel_spectrogram/stft/frame/strided_slice:output:0*
N*
T0*
_output_shapes
:26
4log_mel_spectrogram/stft/frame/strided_slice_1/stackа
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_1Pack(log_mel_spectrogram/stft/frame/add_1:z:0*
N*
T0*
_output_shapes
:28
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_1К
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_2Ј
.log_mel_spectrogram/stft/frame/strided_slice_1StridedSlice/log_mel_spectrogram/stft/frame/Shape_1:output:0=log_mel_spectrogram/stft/frame/strided_slice_1/stack:output:0?log_mel_spectrogram/stft/frame/strided_slice_1/stack_1:output:0?log_mel_spectrogram/stft/frame/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.log_mel_spectrogram/stft/frame/strided_slice_1
(log_mel_spectrogram/stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(log_mel_spectrogram/stft/frame/gcd/Const
+log_mel_spectrogram/stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_spectrogram/stft/frame/floordiv_1/yщ
)log_mel_spectrogram/stft/frame/floordiv_1FloorDiv.log_mel_spectrogram/stft/frame_length:output:04log_mel_spectrogram/stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/floordiv_1
+log_mel_spectrogram/stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_spectrogram/stft/frame/floordiv_2/yч
)log_mel_spectrogram/stft/frame/floordiv_2FloorDiv,log_mel_spectrogram/stft/frame_step:output:04log_mel_spectrogram/stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/floordiv_2
+log_mel_spectrogram/stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_spectrogram/stft/frame/floordiv_3/yђ
)log_mel_spectrogram/stft/frame/floordiv_3FloorDiv7log_mel_spectrogram/stft/frame/strided_slice_1:output:04log_mel_spectrogram/stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/floordiv_3
&log_mel_spectrogram/stft/frame/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_spectrogram/stft/frame/mul_1/yд
$log_mel_spectrogram/stft/frame/mul_1Mul-log_mel_spectrogram/stft/frame/floordiv_3:z:0/log_mel_spectrogram/stft/frame/mul_1/y:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/mul_1Ф
0log_mel_spectrogram/stft/frame/concat_1/values_1Pack(log_mel_spectrogram/stft/frame/mul_1:z:0*
N*
T0*
_output_shapes
:22
0log_mel_spectrogram/stft/frame/concat_1/values_1
,log_mel_spectrogram/stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/concat_1/axisм
'log_mel_spectrogram/stft/frame/concat_1ConcatV2-log_mel_spectrogram/stft/frame/split:output:09log_mel_spectrogram/stft/frame/concat_1/values_1:output:0-log_mel_spectrogram/stft/frame/split:output:25log_mel_spectrogram/stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'log_mel_spectrogram/stft/frame/concat_1Њ
2log_mel_spectrogram/stft/frame/concat_2/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :24
2log_mel_spectrogram/stft/frame/concat_2/values_1/1
0log_mel_spectrogram/stft/frame/concat_2/values_1Pack-log_mel_spectrogram/stft/frame/floordiv_3:z:0;log_mel_spectrogram/stft/frame/concat_2/values_1/1:output:0*
N*
T0*
_output_shapes
:22
0log_mel_spectrogram/stft/frame/concat_2/values_1
,log_mel_spectrogram/stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/concat_2/axisм
'log_mel_spectrogram/stft/frame/concat_2ConcatV2-log_mel_spectrogram/stft/frame/split:output:09log_mel_spectrogram/stft/frame/concat_2/values_1:output:0-log_mel_spectrogram/stft/frame/split:output:25log_mel_spectrogram/stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2)
'log_mel_spectrogram/stft/frame/concat_2 
)log_mel_spectrogram/stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2+
)log_mel_spectrogram/stft/frame/zeros_likeЊ
.log_mel_spectrogram/stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:20
.log_mel_spectrogram/stft/frame/ones_like/ShapeЂ
.log_mel_spectrogram/stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :20
.log_mel_spectrogram/stft/frame/ones_like/Constѓ
(log_mel_spectrogram/stft/frame/ones_likeFill7log_mel_spectrogram/stft/frame/ones_like/Shape:output:07log_mel_spectrogram/stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2*
(log_mel_spectrogram/stft/frame/ones_likeњ
+log_mel_spectrogram/stft/frame/StridedSliceStridedSlice-log_mel_spectrogram/stft/frame/PadV2:output:02log_mel_spectrogram/stft/frame/zeros_like:output:00log_mel_spectrogram/stft/frame/concat_1:output:01log_mel_spectrogram/stft/frame/ones_like:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2-
+log_mel_spectrogram/stft/frame/StridedSlice
(log_mel_spectrogram/stft/frame/Reshape_1Reshape4log_mel_spectrogram/stft/frame/StridedSlice:output:00log_mel_spectrogram/stft/frame/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
(log_mel_spectrogram/stft/frame/Reshape_1
,log_mel_spectrogram/stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/range_1/start
,log_mel_spectrogram/stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/frame/range_1/delta
&log_mel_spectrogram/stft/frame/range_1Range5log_mel_spectrogram/stft/frame/range_1/start:output:0(log_mel_spectrogram/stft/frame/Neg_1:y:05log_mel_spectrogram/stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2(
&log_mel_spectrogram/stft/frame/range_1с
$log_mel_spectrogram/stft/frame/mul_2Mul/log_mel_spectrogram/stft/frame/range_1:output:0-log_mel_spectrogram/stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2&
$log_mel_spectrogram/stft/frame/mul_2І
0log_mel_spectrogram/stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0log_mel_spectrogram/stft/frame/Reshape_2/shape/1ћ
.log_mel_spectrogram/stft/frame/Reshape_2/shapePack(log_mel_spectrogram/stft/frame/Neg_1:y:09log_mel_spectrogram/stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:20
.log_mel_spectrogram/stft/frame/Reshape_2/shapeє
(log_mel_spectrogram/stft/frame/Reshape_2Reshape(log_mel_spectrogram/stft/frame/mul_2:z:07log_mel_spectrogram/stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(log_mel_spectrogram/stft/frame/Reshape_2
,log_mel_spectrogram/stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/range_2/start
,log_mel_spectrogram/stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/frame/range_2/delta
&log_mel_spectrogram/stft/frame/range_2Range5log_mel_spectrogram/stft/frame/range_2/start:output:0-log_mel_spectrogram/stft/frame/floordiv_1:z:05log_mel_spectrogram/stft/frame/range_2/delta:output:0*
_output_shapes
: 2(
&log_mel_spectrogram/stft/frame/range_2І
0log_mel_spectrogram/stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :22
0log_mel_spectrogram/stft/frame/Reshape_3/shape/0
.log_mel_spectrogram/stft/frame/Reshape_3/shapePack9log_mel_spectrogram/stft/frame/Reshape_3/shape/0:output:0-log_mel_spectrogram/stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:20
.log_mel_spectrogram/stft/frame/Reshape_3/shapeђ
(log_mel_spectrogram/stft/frame/Reshape_3Reshape/log_mel_spectrogram/stft/frame/range_2:output:07log_mel_spectrogram/stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

: 2*
(log_mel_spectrogram/stft/frame/Reshape_3э
$log_mel_spectrogram/stft/frame/add_2AddV21log_mel_spectrogram/stft/frame/Reshape_2:output:01log_mel_spectrogram/stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$log_mel_spectrogram/stft/frame/add_2и
'log_mel_spectrogram/stft/frame/GatherV2GatherV21log_mel_spectrogram/stft/frame/Reshape_1:output:0(log_mel_spectrogram/stft/frame/add_2:z:05log_mel_spectrogram/stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 2)
'log_mel_spectrogram/stft/frame/GatherV2є
0log_mel_spectrogram/stft/frame/concat_3/values_1Pack(log_mel_spectrogram/stft/frame/Neg_1:y:0.log_mel_spectrogram/stft/frame_length:output:0*
N*
T0*
_output_shapes
:22
0log_mel_spectrogram/stft/frame/concat_3/values_1
,log_mel_spectrogram/stft/frame/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/concat_3/axisм
'log_mel_spectrogram/stft/frame/concat_3ConcatV2-log_mel_spectrogram/stft/frame/split:output:09log_mel_spectrogram/stft/frame/concat_3/values_1:output:0-log_mel_spectrogram/stft/frame/split:output:25log_mel_spectrogram/stft/frame/concat_3/axis:output:0*
N*
T0*
_output_shapes
:2)
'log_mel_spectrogram/stft/frame/concat_3њ
(log_mel_spectrogram/stft/frame/Reshape_4Reshape0log_mel_spectrogram/stft/frame/GatherV2:output:00log_mel_spectrogram/stft/frame/concat_3:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2*
(log_mel_spectrogram/stft/frame/Reshape_4 
-log_mel_spectrogram/stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-log_mel_spectrogram/stft/hann_window/periodicЦ
)log_mel_spectrogram/stft/hann_window/CastCast6log_mel_spectrogram/stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2+
)log_mel_spectrogram/stft/hann_window/CastЄ
/log_mel_spectrogram/stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :21
/log_mel_spectrogram/stft/hann_window/FloorMod/yѕ
-log_mel_spectrogram/stft/hann_window/FloorModFloorMod.log_mel_spectrogram/stft/frame_length:output:08log_mel_spectrogram/stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2/
-log_mel_spectrogram/stft/hann_window/FloorMod
*log_mel_spectrogram/stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*log_mel_spectrogram/stft/hann_window/sub/xф
(log_mel_spectrogram/stft/hann_window/subSub3log_mel_spectrogram/stft/hann_window/sub/x:output:01log_mel_spectrogram/stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/hann_window/subй
(log_mel_spectrogram/stft/hann_window/mulMul-log_mel_spectrogram/stft/hann_window/Cast:y:0,log_mel_spectrogram/stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/hann_window/mulм
(log_mel_spectrogram/stft/hann_window/addAddV2.log_mel_spectrogram/stft/frame_length:output:0,log_mel_spectrogram/stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/hann_window/add
,log_mel_spectrogram/stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/hann_window/sub_1/yх
*log_mel_spectrogram/stft/hann_window/sub_1Sub,log_mel_spectrogram/stft/hann_window/add:z:05log_mel_spectrogram/stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2,
*log_mel_spectrogram/stft/hann_window/sub_1Т
+log_mel_spectrogram/stft/hann_window/Cast_1Cast.log_mel_spectrogram/stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+log_mel_spectrogram/stft/hann_window/Cast_1І
0log_mel_spectrogram/stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0log_mel_spectrogram/stft/hann_window/range/startІ
0log_mel_spectrogram/stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0log_mel_spectrogram/stft/hann_window/range/deltaЄ
*log_mel_spectrogram/stft/hann_window/rangeRange9log_mel_spectrogram/stft/hann_window/range/start:output:0.log_mel_spectrogram/stft/frame_length:output:09log_mel_spectrogram/stft/hann_window/range/delta:output:0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/rangeЬ
+log_mel_spectrogram/stft/hann_window/Cast_2Cast3log_mel_spectrogram/stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2-
+log_mel_spectrogram/stft/hann_window/Cast_2
*log_mel_spectrogram/stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2,
*log_mel_spectrogram/stft/hann_window/Constы
*log_mel_spectrogram/stft/hann_window/mul_1Mul3log_mel_spectrogram/stft/hann_window/Const:output:0/log_mel_spectrogram/stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/mul_1ю
,log_mel_spectrogram/stft/hann_window/truedivRealDiv.log_mel_spectrogram/stft/hann_window/mul_1:z:0/log_mel_spectrogram/stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2.
,log_mel_spectrogram/stft/hann_window/truedivГ
(log_mel_spectrogram/stft/hann_window/CosCos0log_mel_spectrogram/stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2*
(log_mel_spectrogram/stft/hann_window/CosЁ
,log_mel_spectrogram/stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,log_mel_spectrogram/stft/hann_window/mul_2/xъ
*log_mel_spectrogram/stft/hann_window/mul_2Mul5log_mel_spectrogram/stft/hann_window/mul_2/x:output:0,log_mel_spectrogram/stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/mul_2Ё
,log_mel_spectrogram/stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,log_mel_spectrogram/stft/hann_window/sub_2/xь
*log_mel_spectrogram/stft/hann_window/sub_2Sub5log_mel_spectrogram/stft/hann_window/sub_2/x:output:0.log_mel_spectrogram/stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/sub_2н
log_mel_spectrogram/stft/mulMul1log_mel_spectrogram/stft/frame/Reshape_4:output:0.log_mel_spectrogram/stft/hann_window/sub_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/stft/mulЋ
$log_mel_spectrogram/stft/rfft/packedPack'log_mel_spectrogram/stft/Const:output:0*
N*
T0*
_output_shapes
:2&
$log_mel_spectrogram/stft/rfft/packed
(log_mel_spectrogram/stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2*
(log_mel_spectrogram/stft/rfft/fft_lengthЩ
log_mel_spectrogram/stft/rfftRFFT log_mel_spectrogram/stft/mul:z:01log_mel_spectrogram/stft/rfft/fft_length:output:0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/stft/rfft
log_mel_spectrogram/Abs
ComplexAbs&log_mel_spectrogram/stft/rfft:output:0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Abs
log_mel_spectrogram/SquareSquarelog_mel_spectrogram/Abs:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/SquareН
log_mel_spectrogram/MatMulBatchMatMulV2log_mel_spectrogram/Square:y:0log_mel_spectrogram_matmul_b*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/MatMul
log_mel_spectrogram/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
log_mel_spectrogram/ConstЃ
log_mel_spectrogram/MaxMax#log_mel_spectrogram/MatMul:output:0"log_mel_spectrogram/Const:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Max
log_mel_spectrogram/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2
log_mel_spectrogram/Maximum/xШ
log_mel_spectrogram/MaximumMaximum&log_mel_spectrogram/Maximum/x:output:0#log_mel_spectrogram/MatMul:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Maximum
log_mel_spectrogram/LogLoglog_mel_spectrogram/Maximum:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Log
log_mel_spectrogram/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/Const_1
log_mel_spectrogram/Log_1Log$log_mel_spectrogram/Const_1:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Log_1З
log_mel_spectrogram/truedivRealDivlog_mel_spectrogram/Log:y:0log_mel_spectrogram/Log_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/truediv{
log_mel_spectrogram/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/mul/xД
log_mel_spectrogram/mulMul"log_mel_spectrogram/mul/x:output:0log_mel_spectrogram/truediv:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/mul
log_mel_spectrogram/Maximum_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2!
log_mel_spectrogram/Maximum_1/xЖ
log_mel_spectrogram/Maximum_1Maximum(log_mel_spectrogram/Maximum_1/x:output:0 log_mel_spectrogram/Max:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Maximum_1
log_mel_spectrogram/Log_2Log!log_mel_spectrogram/Maximum_1:z:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Log_2
log_mel_spectrogram/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/Const_2
log_mel_spectrogram/Log_3Log$log_mel_spectrogram/Const_2:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Log_3Ј
log_mel_spectrogram/truediv_1RealDivlog_mel_spectrogram/Log_2:y:0log_mel_spectrogram/Log_3:y:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/truediv_1
log_mel_spectrogram/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/mul_1/xЇ
log_mel_spectrogram/mul_1Mul$log_mel_spectrogram/mul_1/x:output:0!log_mel_spectrogram/truediv_1:z:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/mul_1Ћ
log_mel_spectrogram/subSublog_mel_spectrogram/mul:z:0log_mel_spectrogram/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/sub
log_mel_spectrogram/Const_3Const*
_output_shapes
:*
dtype0*!
valueB"          2
log_mel_spectrogram/Const_3Ё
log_mel_spectrogram/Max_1Maxlog_mel_spectrogram/sub:z:0$log_mel_spectrogram/Const_3:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Max_1
log_mel_spectrogram/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B2
log_mel_spectrogram/sub_1/yЈ
log_mel_spectrogram/sub_1Sub"log_mel_spectrogram/Max_1:output:0$log_mel_spectrogram/sub_1/y:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/sub_1Л
log_mel_spectrogram/Maximum_2Maximumlog_mel_spectrogram/sub:z:0log_mel_spectrogram/sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Maximum_2
"log_mel_spectrogram/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"log_mel_spectrogram/ExpandDims/dimи
log_mel_spectrogram/ExpandDims
ExpandDims!log_mel_spectrogram/Maximum_2:z:0+log_mel_spectrogram/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2 
log_mel_spectrogram/ExpandDimsy
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm
	transpose	Transpose'log_mel_spectrogram/ExpandDims:output:0transpose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
	transpose
)mfccs_from_log_mel_spectrograms/dct/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)mfccs_from_log_mel_spectrograms/dct/Const
*mfccs_from_log_mel_spectrograms/dct/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*mfccs_from_log_mel_spectrograms/dct/Cast/xС
(mfccs_from_log_mel_spectrograms/dct/CastCast3mfccs_from_log_mel_spectrograms/dct/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(mfccs_from_log_mel_spectrograms/dct/CastЄ
/mfccs_from_log_mel_spectrograms/dct/range/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/mfccs_from_log_mel_spectrograms/dct/range/startЄ
/mfccs_from_log_mel_spectrograms/dct/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/mfccs_from_log_mel_spectrograms/dct/range/deltaв
.mfccs_from_log_mel_spectrograms/dct/range/CastCast8mfccs_from_log_mel_spectrograms/dct/range/start:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.mfccs_from_log_mel_spectrograms/dct/range/Castж
0mfccs_from_log_mel_spectrograms/dct/range/Cast_1Cast8mfccs_from_log_mel_spectrograms/dct/range/delta:output:0*

DstT0*

SrcT0*
_output_shapes
: 22
0mfccs_from_log_mel_spectrograms/dct/range/Cast_1
)mfccs_from_log_mel_spectrograms/dct/rangeRange2mfccs_from_log_mel_spectrograms/dct/range/Cast:y:0,mfccs_from_log_mel_spectrograms/dct/Cast:y:04mfccs_from_log_mel_spectrograms/dct/range/Cast_1:y:0*

Tidx0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/rangeВ
'mfccs_from_log_mel_spectrograms/dct/NegNeg2mfccs_from_log_mel_spectrograms/dct/range:output:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/Neg
)mfccs_from_log_mel_spectrograms/dct/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *лI@2+
)mfccs_from_log_mel_spectrograms/dct/mul/yп
'mfccs_from_log_mel_spectrograms/dct/mulMul+mfccs_from_log_mel_spectrograms/dct/Neg:y:02mfccs_from_log_mel_spectrograms/dct/mul/y:output:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/mul
+mfccs_from_log_mel_spectrograms/dct/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2-
+mfccs_from_log_mel_spectrograms/dct/mul_1/yх
)mfccs_from_log_mel_spectrograms/dct/mul_1Mul+mfccs_from_log_mel_spectrograms/dct/mul:z:04mfccs_from_log_mel_spectrograms/dct/mul_1/y:output:0*
T0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/mul_1ч
+mfccs_from_log_mel_spectrograms/dct/truedivRealDiv-mfccs_from_log_mel_spectrograms/dct/mul_1:z:0,mfccs_from_log_mel_spectrograms/dct/Cast:y:0*
T0*
_output_shapes
:2-
+mfccs_from_log_mel_spectrograms/dct/truedivц
+mfccs_from_log_mel_spectrograms/dct/ComplexComplex2mfccs_from_log_mel_spectrograms/dct/Const:output:0/mfccs_from_log_mel_spectrograms/dct/truediv:z:0*
_output_shapes
:2-
+mfccs_from_log_mel_spectrograms/dct/ComplexБ
'mfccs_from_log_mel_spectrograms/dct/ExpExp1mfccs_from_log_mel_spectrograms/dct/Complex:out:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/ExpЃ
+mfccs_from_log_mel_spectrograms/dct/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J   @    2-
+mfccs_from_log_mel_spectrograms/dct/mul_2/xх
)mfccs_from_log_mel_spectrograms/dct/mul_2Mul4mfccs_from_log_mel_spectrograms/dct/mul_2/x:output:0+mfccs_from_log_mel_spectrograms/dct/Exp:y:0*
T0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/mul_2Њ
.mfccs_from_log_mel_spectrograms/dct/rfft/ConstConst*
_output_shapes
:*
dtype0*
valueB:
20
.mfccs_from_log_mel_spectrograms/dct/rfft/Constп
5mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                27
5mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddingsь
,mfccs_from_log_mel_spectrograms/dct/rfft/PadPadtranspose:y:0>mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
2.
,mfccs_from_log_mel_spectrograms/dct/rfft/PadД
3mfccs_from_log_mel_spectrograms/dct/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:
25
3mfccs_from_log_mel_spectrograms/dct/rfft/fft_length
(mfccs_from_log_mel_spectrograms/dct/rfftRFFT5mfccs_from_log_mel_spectrograms/dct/rfft/Pad:output:0<mfccs_from_log_mel_spectrograms/dct/rfft/fft_length:output:0*/
_output_shapes
:џџџџџџџџџ@2*
(mfccs_from_log_mel_spectrograms/dct/rfftУ
7mfccs_from_log_mel_spectrograms/dct/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7mfccs_from_log_mel_spectrograms/dct/strided_slice/stackЧ
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1Ч
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2с
1mfccs_from_log_mel_spectrograms/dct/strided_sliceStridedSlice1mfccs_from_log_mel_spectrograms/dct/rfft:output:0@mfccs_from_log_mel_spectrograms/dct/strided_slice/stack:output:0Bmfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1:output:0Bmfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

begin_mask*
ellipsis_mask23
1mfccs_from_log_mel_spectrograms/dct/strided_slice
)mfccs_from_log_mel_spectrograms/dct/mul_3Mul:mfccs_from_log_mel_spectrograms/dct/strided_slice:output:0-mfccs_from_log_mel_spectrograms/dct/mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2+
)mfccs_from_log_mel_spectrograms/dct/mul_3М
(mfccs_from_log_mel_spectrograms/dct/RealReal-mfccs_from_log_mel_spectrograms/dct/mul_3:z:0*/
_output_shapes
:џџџџџџџџџ@2*
(mfccs_from_log_mel_spectrograms/dct/Real
&mfccs_from_log_mel_spectrograms/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2(
&mfccs_from_log_mel_spectrograms/Cast/xЕ
$mfccs_from_log_mel_spectrograms/CastCast/mfccs_from_log_mel_spectrograms/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$mfccs_from_log_mel_spectrograms/Cast
%mfccs_from_log_mel_spectrograms/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%mfccs_from_log_mel_spectrograms/mul/yЬ
#mfccs_from_log_mel_spectrograms/mulMul(mfccs_from_log_mel_spectrograms/Cast:y:0.mfccs_from_log_mel_spectrograms/mul/y:output:0*
T0*
_output_shapes
: 2%
#mfccs_from_log_mel_spectrograms/mulЁ
%mfccs_from_log_mel_spectrograms/RsqrtRsqrt'mfccs_from_log_mel_spectrograms/mul:z:0*
T0*
_output_shapes
: 2'
%mfccs_from_log_mel_spectrograms/Rsqrtэ
%mfccs_from_log_mel_spectrograms/mul_1Mul1mfccs_from_log_mel_spectrograms/dct/Real:output:0)mfccs_from_log_mel_spectrograms/Rsqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2'
%mfccs_from_log_mel_spectrograms/mul_1c
SquareSquaretranspose:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
	keep_dims(2
Sum\
SqrtSqrtSum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Sqrt
delta/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose/permГ
delta/transpose	Transpose)mfccs_from_log_mel_spectrograms/mul_1:z:0delta/transpose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose
delta/ConstConst*
_output_shapes

:*
dtype0*9
value0B."                               2
delta/ConstЉ
delta/MirrorPad	MirrorPaddelta/transpose:y:0delta/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
mode	SYMMETRIC2
delta/MirrorPads
delta/arange/startConst*
_output_shapes
: *
dtype0*
valueB :
ќџџџџџџџџ2
delta/arange/startj
delta/arange/limitConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange/limitj
delta/arange/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange/delta
delta/arangeRangedelta/arange/start:output:0delta/arange/limit:output:0delta/arange/delta:output:0*
_output_shapes
:	2
delta/arangek

delta/CastCastdelta/arange:output:0*

DstT0*

SrcT0*
_output_shapes
:	2

delta/Cast
delta/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
delta/Reshape/shape
delta/ReshapeReshapedelta/Cast:y:0delta/Reshape/shape:output:0*
T0*&
_output_shapes
:	2
delta/ReshapeХ
delta/convolutionConv2Ddelta/MirrorPad:output:0delta/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
delta/convolutiong
delta/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pB2
delta/truediv/y
delta/truedivRealDivdelta/convolution:output:0delta/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/truediv
delta/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose_1/permЁ
delta/transpose_1	Transposedelta/truediv:z:0delta/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose_1
delta/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose_2/permЅ
delta/transpose_2	Transposedelta/transpose_1:y:0delta/transpose_2/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose_2
delta/Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                               2
delta/Const_1Б
delta/MirrorPad_1	MirrorPaddelta/transpose_2:y:0delta/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
mode	SYMMETRIC2
delta/MirrorPad_1w
delta/arange_1/startConst*
_output_shapes
: *
dtype0*
valueB :
ќџџџџџџџџ2
delta/arange_1/startn
delta/arange_1/limitConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange_1/limitn
delta/arange_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange_1/deltaЂ
delta/arange_1Rangedelta/arange_1/start:output:0delta/arange_1/limit:output:0delta/arange_1/delta:output:0*
_output_shapes
:	2
delta/arange_1q
delta/Cast_1Castdelta/arange_1:output:0*

DstT0*

SrcT0*
_output_shapes
:	2
delta/Cast_1
delta/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
delta/Reshape_1/shape
delta/Reshape_1Reshapedelta/Cast_1:y:0delta/Reshape_1/shape:output:0*
T0*&
_output_shapes
:	2
delta/Reshape_1Э
delta/convolution_1Conv2Ddelta/MirrorPad_1:output:0delta/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
delta/convolution_1k
delta/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pB2
delta/truediv_1/yЁ
delta/truediv_1RealDivdelta/convolution_1:output:0delta/truediv_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/truediv_1
delta/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose_3/permЃ
delta/transpose_3	Transposedelta/truediv_1:z:0delta/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisи
concatConcatV2)mfccs_from_log_mel_spectrograms/mul_1:z:0delta/transpose_1:y:0delta/transpose_3:y:0Sqrt:y:0concat/axis:output:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@2
concat
	Squeeze_1Squeezeconcat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@*
squeeze_dims
2
	Squeeze_1
(resnet_block/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(resnet_block/conv1/conv1d/ExpandDims/dimл
$resnet_block/conv1/conv1d/ExpandDims
ExpandDimsSqueeze_1:output:01resnet_block/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2&
$resnet_block/conv1/conv1d/ExpandDimsё
5resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>resnet_block_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOp
*resnet_block/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*resnet_block/conv1/conv1d/ExpandDims_1/dim
&resnet_block/conv1/conv1d/ExpandDims_1
ExpandDims=resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:03resnet_block/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2(
&resnet_block/conv1/conv1d/ExpandDims_1
resnet_block/conv1/conv1dConv2D-resnet_block/conv1/conv1d/ExpandDims:output:0/resnet_block/conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/conv1/conv1dЫ
!resnet_block/conv1/conv1d/SqueezeSqueeze"resnet_block/conv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2#
!resnet_block/conv1/conv1d/SqueezeХ
)resnet_block/conv1/BiasAdd/ReadVariableOpReadVariableOp2resnet_block_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)resnet_block/conv1/BiasAdd/ReadVariableOpи
resnet_block/conv1/BiasAddBiasAdd*resnet_block/conv1/conv1d/Squeeze:output:01resnet_block/conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/conv1/BiasAdd
resnet_block/relu1/ReluRelu#resnet_block/conv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/relu1/Relu
(resnet_block/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(resnet_block/conv2/conv1d/ExpandDims/dimю
$resnet_block/conv2/conv1d/ExpandDims
ExpandDims%resnet_block/relu1/Relu:activations:01resnet_block/conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2&
$resnet_block/conv2/conv1d/ExpandDimsё
5resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>resnet_block_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype027
5resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOp
*resnet_block/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*resnet_block/conv2/conv1d/ExpandDims_1/dim
&resnet_block/conv2/conv1d/ExpandDims_1
ExpandDims=resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:03resnet_block/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2(
&resnet_block/conv2/conv1d/ExpandDims_1
resnet_block/conv2/conv1dConv2D-resnet_block/conv2/conv1d/ExpandDims:output:0/resnet_block/conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/conv2/conv1dЫ
!resnet_block/conv2/conv1d/SqueezeSqueeze"resnet_block/conv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2#
!resnet_block/conv2/conv1d/SqueezeХ
)resnet_block/conv2/BiasAdd/ReadVariableOpReadVariableOp2resnet_block_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)resnet_block/conv2/BiasAdd/ReadVariableOpи
resnet_block/conv2/BiasAddBiasAdd*resnet_block/conv2/conv1d/Squeeze:output:01resnet_block/conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/conv2/BiasAdd
resnet_block/relu2/ReluRelu#resnet_block/conv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/relu2/Relu
(resnet_block/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(resnet_block/conv3/conv1d/ExpandDims/dimю
$resnet_block/conv3/conv1d/ExpandDims
ExpandDims%resnet_block/relu2/Relu:activations:01resnet_block/conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2&
$resnet_block/conv3/conv1d/ExpandDimsё
5resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>resnet_block_conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype027
5resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOp
*resnet_block/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*resnet_block/conv3/conv1d/ExpandDims_1/dim
&resnet_block/conv3/conv1d/ExpandDims_1
ExpandDims=resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:03resnet_block/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2(
&resnet_block/conv3/conv1d/ExpandDims_1
resnet_block/conv3/conv1dConv2D-resnet_block/conv3/conv1d/ExpandDims:output:0/resnet_block/conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/conv3/conv1dЫ
!resnet_block/conv3/conv1d/SqueezeSqueeze"resnet_block/conv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2#
!resnet_block/conv3/conv1d/SqueezeХ
)resnet_block/conv3/BiasAdd/ReadVariableOpReadVariableOp2resnet_block_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)resnet_block/conv3/BiasAdd/ReadVariableOpи
resnet_block/conv3/BiasAddBiasAdd*resnet_block/conv3/conv1d/Squeeze:output:01resnet_block/conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/conv3/BiasAddЅ
+resnet_block/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2-
+resnet_block/shortcut/conv1d/ExpandDims/dimф
'resnet_block/shortcut/conv1d/ExpandDims
ExpandDimsSqueeze_1:output:04resnet_block/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2)
'resnet_block/shortcut/conv1d/ExpandDimsњ
8resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAresnet_block_shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02:
8resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOp 
-resnet_block/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-resnet_block/shortcut/conv1d/ExpandDims_1/dim
)resnet_block/shortcut/conv1d/ExpandDims_1
ExpandDims@resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:06resnet_block/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2+
)resnet_block/shortcut/conv1d/ExpandDims_1
resnet_block/shortcut/conv1dConv2D0resnet_block/shortcut/conv1d/ExpandDims:output:02resnet_block/shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/shortcut/conv1dд
$resnet_block/shortcut/conv1d/SqueezeSqueeze%resnet_block/shortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2&
$resnet_block/shortcut/conv1d/SqueezeЮ
,resnet_block/shortcut/BiasAdd/ReadVariableOpReadVariableOp5resnet_block_shortcut_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,resnet_block/shortcut/BiasAdd/ReadVariableOpф
resnet_block/shortcut/BiasAddBiasAdd-resnet_block/shortcut/conv1d/Squeeze:output:04resnet_block/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/shortcut/BiasAddИ
resnet_block/add/addAddV2#resnet_block/conv3/BiasAdd:output:0&resnet_block/shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/add/add
resnet_block/out_block/ReluReluresnet_block/add/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/out_block/ReluЃ
*resnet_block_1/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_1/conv1/conv1d/ExpandDims/dimј
&resnet_block_1/conv1/conv1d/ExpandDims
ExpandDims)resnet_block/out_block/Relu:activations:03resnet_block_1/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2(
&resnet_block_1/conv1/conv1d/ExpandDimsї
7resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_1_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype029
7resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_1/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_1/conv1/conv1d/ExpandDims_1/dim
(resnet_block_1/conv1/conv1d/ExpandDims_1
ExpandDims?resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_1/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2*
(resnet_block_1/conv1/conv1d/ExpandDims_1
resnet_block_1/conv1/conv1dConv2D/resnet_block_1/conv1/conv1d/ExpandDims:output:01resnet_block_1/conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
resnet_block_1/conv1/conv1dб
#resnet_block_1/conv1/conv1d/SqueezeSqueeze$resnet_block_1/conv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_1/conv1/conv1d/SqueezeЫ
+resnet_block_1/conv1/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+resnet_block_1/conv1/BiasAdd/ReadVariableOpр
resnet_block_1/conv1/BiasAddBiasAdd,resnet_block_1/conv1/conv1d/Squeeze:output:03resnet_block_1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/conv1/BiasAdd
resnet_block_1/relu1/ReluRelu%resnet_block_1/conv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/relu1/ReluЃ
*resnet_block_1/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_1/conv2/conv1d/ExpandDims/dimі
&resnet_block_1/conv2/conv1d/ExpandDims
ExpandDims'resnet_block_1/relu1/Relu:activations:03resnet_block_1/conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2(
&resnet_block_1/conv2/conv1d/ExpandDimsї
7resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_1_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_1/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_1/conv2/conv1d/ExpandDims_1/dim
(resnet_block_1/conv2/conv1d/ExpandDims_1
ExpandDims?resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_1/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2*
(resnet_block_1/conv2/conv1d/ExpandDims_1
resnet_block_1/conv2/conv1dConv2D/resnet_block_1/conv2/conv1d/ExpandDims:output:01resnet_block_1/conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
resnet_block_1/conv2/conv1dб
#resnet_block_1/conv2/conv1d/SqueezeSqueeze$resnet_block_1/conv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_1/conv2/conv1d/SqueezeЫ
+resnet_block_1/conv2/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+resnet_block_1/conv2/BiasAdd/ReadVariableOpр
resnet_block_1/conv2/BiasAddBiasAdd,resnet_block_1/conv2/conv1d/Squeeze:output:03resnet_block_1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/conv2/BiasAdd
resnet_block_1/relu2/ReluRelu%resnet_block_1/conv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/relu2/ReluЃ
*resnet_block_1/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_1/conv3/conv1d/ExpandDims/dimі
&resnet_block_1/conv3/conv1d/ExpandDims
ExpandDims'resnet_block_1/relu2/Relu:activations:03resnet_block_1/conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2(
&resnet_block_1/conv3/conv1d/ExpandDimsї
7resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_1_conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_1/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_1/conv3/conv1d/ExpandDims_1/dim
(resnet_block_1/conv3/conv1d/ExpandDims_1
ExpandDims?resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_1/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2*
(resnet_block_1/conv3/conv1d/ExpandDims_1
resnet_block_1/conv3/conv1dConv2D/resnet_block_1/conv3/conv1d/ExpandDims:output:01resnet_block_1/conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
resnet_block_1/conv3/conv1dб
#resnet_block_1/conv3/conv1d/SqueezeSqueeze$resnet_block_1/conv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_1/conv3/conv1d/SqueezeЫ
+resnet_block_1/conv3/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_1_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+resnet_block_1/conv3/BiasAdd/ReadVariableOpр
resnet_block_1/conv3/BiasAddBiasAdd,resnet_block_1/conv3/conv1d/Squeeze:output:03resnet_block_1/conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/conv3/BiasAddЉ
-resnet_block_1/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-resnet_block_1/shortcut/conv1d/ExpandDims/dim
)resnet_block_1/shortcut/conv1d/ExpandDims
ExpandDims)resnet_block/out_block/Relu:activations:06resnet_block_1/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2+
)resnet_block_1/shortcut/conv1d/ExpandDims
:resnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCresnet_block_1_shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02<
:resnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOpЄ
/resnet_block_1/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/resnet_block_1/shortcut/conv1d/ExpandDims_1/dim
+resnet_block_1/shortcut/conv1d/ExpandDims_1
ExpandDimsBresnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:08resnet_block_1/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2-
+resnet_block_1/shortcut/conv1d/ExpandDims_1
resnet_block_1/shortcut/conv1dConv2D2resnet_block_1/shortcut/conv1d/ExpandDims:output:04resnet_block_1/shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2 
resnet_block_1/shortcut/conv1dк
&resnet_block_1/shortcut/conv1d/SqueezeSqueeze'resnet_block_1/shortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2(
&resnet_block_1/shortcut/conv1d/Squeezeд
.resnet_block_1/shortcut/BiasAdd/ReadVariableOpReadVariableOp7resnet_block_1_shortcut_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.resnet_block_1/shortcut/BiasAdd/ReadVariableOpь
resnet_block_1/shortcut/BiasAddBiasAdd/resnet_block_1/shortcut/conv1d/Squeeze:output:06resnet_block_1/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2!
resnet_block_1/shortcut/BiasAddФ
resnet_block_1/add_1/addAddV2%resnet_block_1/conv3/BiasAdd:output:0(resnet_block_1/shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/add_1/add
resnet_block_1/out_block/ReluReluresnet_block_1/add_1/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/out_block/ReluЃ
*resnet_block_2/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_2/conv1/conv1d/ExpandDims/dimњ
&resnet_block_2/conv1/conv1d/ExpandDims
ExpandDims+resnet_block_1/out_block/Relu:activations:03resnet_block_2/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2(
&resnet_block_2/conv1/conv1d/ExpandDimsј
7resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_2_conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype029
7resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_2/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_2/conv1/conv1d/ExpandDims_1/dim
(resnet_block_2/conv1/conv1d/ExpandDims_1
ExpandDims?resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_2/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2*
(resnet_block_2/conv1/conv1d/ExpandDims_1
resnet_block_2/conv1/conv1dConv2D/resnet_block_2/conv1/conv1d/ExpandDims:output:01resnet_block_2/conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_2/conv1/conv1dв
#resnet_block_2/conv1/conv1d/SqueezeSqueeze$resnet_block_2/conv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_2/conv1/conv1d/SqueezeЬ
+resnet_block_2/conv1/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_2/conv1/BiasAdd/ReadVariableOpс
resnet_block_2/conv1/BiasAddBiasAdd,resnet_block_2/conv1/conv1d/Squeeze:output:03resnet_block_2/conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/conv1/BiasAdd
resnet_block_2/relu1/ReluRelu%resnet_block_2/conv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/relu1/ReluЃ
*resnet_block_2/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_2/conv2/conv1d/ExpandDims/dimї
&resnet_block_2/conv2/conv1d/ExpandDims
ExpandDims'resnet_block_2/relu1/Relu:activations:03resnet_block_2/conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_2/conv2/conv1d/ExpandDimsљ
7resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_2_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_2/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_2/conv2/conv1d/ExpandDims_1/dim
(resnet_block_2/conv2/conv1d/ExpandDims_1
ExpandDims?resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_2/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_2/conv2/conv1d/ExpandDims_1
resnet_block_2/conv2/conv1dConv2D/resnet_block_2/conv2/conv1d/ExpandDims:output:01resnet_block_2/conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_2/conv2/conv1dв
#resnet_block_2/conv2/conv1d/SqueezeSqueeze$resnet_block_2/conv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_2/conv2/conv1d/SqueezeЬ
+resnet_block_2/conv2/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_2/conv2/BiasAdd/ReadVariableOpс
resnet_block_2/conv2/BiasAddBiasAdd,resnet_block_2/conv2/conv1d/Squeeze:output:03resnet_block_2/conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/conv2/BiasAdd
resnet_block_2/relu2/ReluRelu%resnet_block_2/conv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/relu2/ReluЃ
*resnet_block_2/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_2/conv3/conv1d/ExpandDims/dimї
&resnet_block_2/conv3/conv1d/ExpandDims
ExpandDims'resnet_block_2/relu2/Relu:activations:03resnet_block_2/conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_2/conv3/conv1d/ExpandDimsљ
7resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_2_conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_2/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_2/conv3/conv1d/ExpandDims_1/dim
(resnet_block_2/conv3/conv1d/ExpandDims_1
ExpandDims?resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_2/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_2/conv3/conv1d/ExpandDims_1
resnet_block_2/conv3/conv1dConv2D/resnet_block_2/conv3/conv1d/ExpandDims:output:01resnet_block_2/conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_2/conv3/conv1dв
#resnet_block_2/conv3/conv1d/SqueezeSqueeze$resnet_block_2/conv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_2/conv3/conv1d/SqueezeЬ
+resnet_block_2/conv3/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_2_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_2/conv3/BiasAdd/ReadVariableOpс
resnet_block_2/conv3/BiasAddBiasAdd,resnet_block_2/conv3/conv1d/Squeeze:output:03resnet_block_2/conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/conv3/BiasAddЉ
-resnet_block_2/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-resnet_block_2/shortcut/conv1d/ExpandDims/dim
)resnet_block_2/shortcut/conv1d/ExpandDims
ExpandDims+resnet_block_1/out_block/Relu:activations:06resnet_block_2/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2+
)resnet_block_2/shortcut/conv1d/ExpandDims
:resnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCresnet_block_2_shortcut_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02<
:resnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOpЄ
/resnet_block_2/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/resnet_block_2/shortcut/conv1d/ExpandDims_1/dim
+resnet_block_2/shortcut/conv1d/ExpandDims_1
ExpandDimsBresnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:08resnet_block_2/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2-
+resnet_block_2/shortcut/conv1d/ExpandDims_1
resnet_block_2/shortcut/conv1dConv2D2resnet_block_2/shortcut/conv1d/ExpandDims:output:04resnet_block_2/shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2 
resnet_block_2/shortcut/conv1dл
&resnet_block_2/shortcut/conv1d/SqueezeSqueeze'resnet_block_2/shortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2(
&resnet_block_2/shortcut/conv1d/Squeezeе
.resnet_block_2/shortcut/BiasAdd/ReadVariableOpReadVariableOp7resnet_block_2_shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.resnet_block_2/shortcut/BiasAdd/ReadVariableOpэ
resnet_block_2/shortcut/BiasAddBiasAdd/resnet_block_2/shortcut/conv1d/Squeeze:output:06resnet_block_2/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2!
resnet_block_2/shortcut/BiasAddХ
resnet_block_2/add_2/addAddV2%resnet_block_2/conv3/BiasAdd:output:0(resnet_block_2/shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/add_2/add
resnet_block_2/out_block/ReluReluresnet_block_2/add_2/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/out_block/ReluЃ
*resnet_block_3/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_3/conv1/conv1d/ExpandDims/dimћ
&resnet_block_3/conv1/conv1d/ExpandDims
ExpandDims+resnet_block_2/out_block/Relu:activations:03resnet_block_3/conv1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_3/conv1/conv1d/ExpandDimsљ
7resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_3_conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_3/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_3/conv1/conv1d/ExpandDims_1/dim
(resnet_block_3/conv1/conv1d/ExpandDims_1
ExpandDims?resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_3/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_3/conv1/conv1d/ExpandDims_1
resnet_block_3/conv1/conv1dConv2D/resnet_block_3/conv1/conv1d/ExpandDims:output:01resnet_block_3/conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_3/conv1/conv1dв
#resnet_block_3/conv1/conv1d/SqueezeSqueeze$resnet_block_3/conv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_3/conv1/conv1d/SqueezeЬ
+resnet_block_3/conv1/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_3/conv1/BiasAdd/ReadVariableOpс
resnet_block_3/conv1/BiasAddBiasAdd,resnet_block_3/conv1/conv1d/Squeeze:output:03resnet_block_3/conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/conv1/BiasAdd
resnet_block_3/relu1/ReluRelu%resnet_block_3/conv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/relu1/ReluЃ
*resnet_block_3/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_3/conv2/conv1d/ExpandDims/dimї
&resnet_block_3/conv2/conv1d/ExpandDims
ExpandDims'resnet_block_3/relu1/Relu:activations:03resnet_block_3/conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_3/conv2/conv1d/ExpandDimsљ
7resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_3_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_3/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_3/conv2/conv1d/ExpandDims_1/dim
(resnet_block_3/conv2/conv1d/ExpandDims_1
ExpandDims?resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_3/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_3/conv2/conv1d/ExpandDims_1
resnet_block_3/conv2/conv1dConv2D/resnet_block_3/conv2/conv1d/ExpandDims:output:01resnet_block_3/conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_3/conv2/conv1dв
#resnet_block_3/conv2/conv1d/SqueezeSqueeze$resnet_block_3/conv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_3/conv2/conv1d/SqueezeЬ
+resnet_block_3/conv2/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_3/conv2/BiasAdd/ReadVariableOpс
resnet_block_3/conv2/BiasAddBiasAdd,resnet_block_3/conv2/conv1d/Squeeze:output:03resnet_block_3/conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/conv2/BiasAdd
resnet_block_3/relu2/ReluRelu%resnet_block_3/conv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/relu2/ReluЃ
*resnet_block_3/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_3/conv3/conv1d/ExpandDims/dimї
&resnet_block_3/conv3/conv1d/ExpandDims
ExpandDims'resnet_block_3/relu2/Relu:activations:03resnet_block_3/conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_3/conv3/conv1d/ExpandDimsљ
7resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_3_conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_3/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_3/conv3/conv1d/ExpandDims_1/dim
(resnet_block_3/conv3/conv1d/ExpandDims_1
ExpandDims?resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_3/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_3/conv3/conv1d/ExpandDims_1
resnet_block_3/conv3/conv1dConv2D/resnet_block_3/conv3/conv1d/ExpandDims:output:01resnet_block_3/conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_3/conv3/conv1dв
#resnet_block_3/conv3/conv1d/SqueezeSqueeze$resnet_block_3/conv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_3/conv3/conv1d/SqueezeЬ
+resnet_block_3/conv3/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_3/conv3/BiasAdd/ReadVariableOpс
resnet_block_3/conv3/BiasAddBiasAdd,resnet_block_3/conv3/conv1d/Squeeze:output:03resnet_block_3/conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/conv3/BiasAddЉ
-resnet_block_3/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-resnet_block_3/shortcut/conv1d/ExpandDims/dim
)resnet_block_3/shortcut/conv1d/ExpandDims
ExpandDims+resnet_block_2/out_block/Relu:activations:06resnet_block_3/shortcut/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2+
)resnet_block_3/shortcut/conv1d/ExpandDims
:resnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCresnet_block_3_shortcut_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02<
:resnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOpЄ
/resnet_block_3/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/resnet_block_3/shortcut/conv1d/ExpandDims_1/dim
+resnet_block_3/shortcut/conv1d/ExpandDims_1
ExpandDimsBresnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:08resnet_block_3/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2-
+resnet_block_3/shortcut/conv1d/ExpandDims_1
resnet_block_3/shortcut/conv1dConv2D2resnet_block_3/shortcut/conv1d/ExpandDims:output:04resnet_block_3/shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2 
resnet_block_3/shortcut/conv1dл
&resnet_block_3/shortcut/conv1d/SqueezeSqueeze'resnet_block_3/shortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2(
&resnet_block_3/shortcut/conv1d/Squeezeе
.resnet_block_3/shortcut/BiasAdd/ReadVariableOpReadVariableOp7resnet_block_3_shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.resnet_block_3/shortcut/BiasAdd/ReadVariableOpэ
resnet_block_3/shortcut/BiasAddBiasAdd/resnet_block_3/shortcut/conv1d/Squeeze:output:06resnet_block_3/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2!
resnet_block_3/shortcut/BiasAddХ
resnet_block_3/add_3/addAddV2%resnet_block_3/conv3/BiasAdd:output:0(resnet_block_3/shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/add_3/add
resnet_block_3/out_block/ReluReluresnet_block_3/add_3/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/out_block/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
flatten/ConstЅ
flatten/ReshapeReshape+resnet_block_3/out_block/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ@2
flatten/Reshape
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype02
fc1/MatMul/ReadVariableOp

fc1/MatMulMatMulflatten/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc1/MatMul
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc1/BiasAdd/ReadVariableOp
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
fc1/BiasAdde
fc1/ReluRelufc1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc1/Relu
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
fc2/MatMul/ReadVariableOp

fc2/MatMulMatMulfc1/Relu:activations:0!fc2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc2/MatMul
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc2/BiasAdd/ReadVariableOp
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
fc2/BiasAdde
fc2/ReluRelufc2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc2/Relu
fc3/MatMul/ReadVariableOpReadVariableOp"fc3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
fc3/MatMul/ReadVariableOp

fc3/MatMulMatMulfc2/Relu:activations:0!fc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2

fc3/MatMul
fc3/BiasAdd/ReadVariableOpReadVariableOp#fc3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc3/BiasAdd/ReadVariableOp
fc3/BiasAddBiasAddfc3/MatMul:product:0"fc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fc3/BiasAddm
fc3/SigmoidSigmoidfc3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fc3/Sigmoidc
IdentityIdentityfc3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:џџџџџџџџџ:	:::::::::::::::::::::::::::::::::::::::T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:%!

_output_shapes
:	
а;
ї
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_216541

inputs5
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЈ
conv1/conv1d/ExpandDims
ExpandDimsinputs$conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv1/conv1d/ExpandDimsЪ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimЯ
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1/conv1d/ExpandDims_1Ю
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv1/conv1dЄ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1/BiasAdd/ReadVariableOpЄ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
conv1/BiasAddn

relu1/ReluReluconv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimК
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2/conv1d/ExpandDimsЪ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimЯ
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv2/conv1d/ExpandDims_1Ю
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2/conv1dЄ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2/BiasAdd/ReadVariableOpЄ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
conv2/BiasAddn

relu2/ReluReluconv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimК
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv3/conv1d/ExpandDimsЪ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimЯ
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv3/conv1d/ExpandDims_1Ю
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv3/conv1dЄ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv3/BiasAdd/ReadVariableOpЄ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimБ
shortcut/conv1d/ExpandDims
ExpandDimsinputs'shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
shortcut/conv1d/ExpandDimsг
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimл
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
shortcut/conv1d/ExpandDims_1к
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
shortcut/conv1d­
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЇ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
shortcut/BiasAdd/ReadVariableOpА
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
add/addk
out_block/ReluReluadd/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
out_block/Relut
IdentityIdentityout_block/Relu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@ :::::::::S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
р­
|
O__inference_log_mel_spectrogram_layer_call_and_return_conditional_losses_212316
	waveforms
matmul_b
identityi
stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2
stft/frame_lengthd
stft/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame_step[

stft/ConstConst*
_output_shapes
: *
dtype0*
value
B :2

stft/Constm
stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
stft/frame/axis]
stft/frame/ShapeShape	waveforms*
T0*
_output_shapes
:2
stft/frame/Shaped
stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Rankr
stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range/startr
stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range/deltaЅ
stft/frame/rangeRangestft/frame/range/start:output:0stft/frame/Rank:output:0stft/frame/range/delta:output:0*
_output_shapes
:2
stft/frame/range
stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2 
stft/frame/strided_slice/stack
 stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 stft/frame/strided_slice/stack_1
 stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 stft/frame/strided_slice/stack_2Є
stft/frame/strided_sliceStridedSlicestft/frame/range:output:0'stft/frame/strided_slice/stack:output:0)stft/frame/strided_slice/stack_1:output:0)stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
stft/frame/strided_slicef
stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/sub/y}
stft/frame/subSubstft/frame/Rank:output:0stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2
stft/frame/sub
stft/frame/sub_1Substft/frame/sub:z:0!stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2
stft/frame/sub_1l
stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/packed/1Г
stft/frame/packedPack!stft/frame/strided_slice:output:0stft/frame/packed/1:output:0stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2
stft/frame/packedz
stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/split/split_dimж
stft/frame/splitSplitVstft/frame/Shape:output:0stft/frame/packed:output:0#stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2
stft/frame/splitw
stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
stft/frame/Reshape/shape{
stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2
stft/frame/Reshape/shape_1
stft/frame/ReshapeReshapestft/frame/split:output:1#stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2
stft/frame/Reshaped
stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Sizeh
stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Size_1i
stft/frame/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
stft/frame/Conste
stft/frame/NegNegstft/frame/Reshape:output:0*
T0*
_output_shapes
: 2
stft/frame/Neg
stft/frame/floordivFloorDivstft/frame/Neg:y:0stft/frame_step:output:0*
T0*
_output_shapes
: 2
stft/frame/floordive
stft/frame/Neg_1Negstft/frame/floordiv:z:0*
T0*
_output_shapes
: 2
stft/frame/Neg_1j
stft/frame/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/sub_2/y
stft/frame/sub_2Substft/frame/Neg_1:y:0stft/frame/sub_2/y:output:0*
T0*
_output_shapes
: 2
stft/frame/sub_2x
stft/frame/mulMulstft/frame_step:output:0stft/frame/sub_2:z:0*
T0*
_output_shapes
: 2
stft/frame/mulz
stft/frame/addAddV2stft/frame_length:output:0stft/frame/mul:z:0*
T0*
_output_shapes
: 2
stft/frame/add}
stft/frame/sub_3Substft/frame/add:z:0stft/frame/Reshape:output:0*
T0*
_output_shapes
: 2
stft/frame/sub_3n
stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Maximum/x
stft/frame/MaximumMaximumstft/frame/Maximum/x:output:0stft/frame/sub_3:z:0*
T0*
_output_shapes
: 2
stft/frame/Maximumr
stft/frame/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/zeros/mul/y
stft/frame/zeros/mulMulstft/frame/Size:output:0stft/frame/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
stft/frame/zeros/mulu
stft/frame/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
stft/frame/zeros/Less/y
stft/frame/zeros/LessLessstft/frame/zeros/mul:z:0 stft/frame/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
stft/frame/zeros/Lessx
stft/frame/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/zeros/packed/1І
stft/frame/zeros/packedPackstft/frame/Size:output:0"stft/frame/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/zeros/packedr
stft/frame/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/zeros/Const
stft/frame/zerosFill stft/frame/zeros/packed:output:0stft/frame/zeros/Const:output:0*
T0*
_output_shapes

:2
stft/frame/zerost
stft/frame/packed_1/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/packed_1/0/0
stft/frame/packed_1/0Pack stft/frame/packed_1/0/0:output:0stft/frame/Maximum:z:0*
N*
T0*
_output_shapes
:2
stft/frame/packed_1/0
stft/frame/packed_1Packstft/frame/packed_1/0:output:0*
N*
T0*
_output_shapes

:2
stft/frame/packed_1v
stft/frame/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/zeros_1/mul/y
stft/frame/zeros_1/mulMulstft/frame/Size_1:output:0!stft/frame/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
stft/frame/zeros_1/muly
stft/frame/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
stft/frame/zeros_1/Less/y
stft/frame/zeros_1/LessLessstft/frame/zeros_1/mul:z:0"stft/frame/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
stft/frame/zeros_1/Less|
stft/frame/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/zeros_1/packed/1Ў
stft/frame/zeros_1/packedPackstft/frame/Size_1:output:0$stft/frame/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/zeros_1/packedv
stft/frame/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/zeros_1/Const
stft/frame/zeros_1Fill"stft/frame/zeros_1/packed:output:0!stft/frame/zeros_1/Const:output:0*
T0*
_output_shapes

: 2
stft/frame/zeros_1r
stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat/axisл
stft/frame/concatConcatV2stft/frame/zeros:output:0stft/frame/packed_1:output:0stft/frame/zeros_1:output:0stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes

:2
stft/frame/concatЊ
stft/frame/PadV2PadV2	waveformsstft/frame/concat:output:0stft/frame/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
stft/frame/PadV2q
stft/frame/Shape_1Shapestft/frame/PadV2:output:0*
T0*
_output_shapes
:2
stft/frame/Shape_1j
stft/frame/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/add_1/y
stft/frame/add_1AddV2!stft/frame/strided_slice:output:0stft/frame/add_1/y:output:0*
T0*
_output_shapes
: 2
stft/frame/add_1
 stft/frame/strided_slice_1/stackPack!stft/frame/strided_slice:output:0*
N*
T0*
_output_shapes
:2"
 stft/frame/strided_slice_1/stack
"stft/frame/strided_slice_1/stack_1Packstft/frame/add_1:z:0*
N*
T0*
_output_shapes
:2$
"stft/frame/strided_slice_1/stack_1
"stft/frame/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"stft/frame/strided_slice_1/stack_2А
stft/frame/strided_slice_1StridedSlicestft/frame/Shape_1:output:0)stft/frame/strided_slice_1/stack:output:0+stft/frame/strided_slice_1/stack_1:output:0+stft/frame/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
stft/frame/strided_slice_1n
stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/gcd/Constt
stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/floordiv_1/y
stft/frame/floordiv_1FloorDivstft/frame_length:output:0 stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_1t
stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/floordiv_2/y
stft/frame/floordiv_2FloorDivstft/frame_step:output:0 stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_2t
stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/floordiv_3/yЂ
stft/frame/floordiv_3FloorDiv#stft/frame/strided_slice_1:output:0 stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_3j
stft/frame/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/mul_1/y
stft/frame/mul_1Mulstft/frame/floordiv_3:z:0stft/frame/mul_1/y:output:0*
T0*
_output_shapes
: 2
stft/frame/mul_1
stft/frame/concat_1/values_1Packstft/frame/mul_1:z:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_1/values_1v
stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat_1/axisф
stft/frame/concat_1ConcatV2stft/frame/split:output:0%stft/frame/concat_1/values_1:output:0stft/frame/split:output:2!stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_1
stft/frame/concat_2/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :2 
stft/frame/concat_2/values_1/1Ж
stft/frame/concat_2/values_1Packstft/frame/floordiv_3:z:0'stft/frame/concat_2/values_1/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_2/values_1v
stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat_2/axisф
stft/frame/concat_2ConcatV2stft/frame/split:output:0%stft/frame/concat_2/values_1:output:0stft/frame/split:output:2!stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_2x
stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2
stft/frame/zeros_like
stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
stft/frame/ones_like/Shapez
stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/ones_like/ConstЃ
stft/frame/ones_likeFill#stft/frame/ones_like/Shape:output:0#stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2
stft/frame/ones_like
stft/frame/StridedSliceStridedSlicestft/frame/PadV2:output:0stft/frame/zeros_like:output:0stft/frame/concat_1:output:0stft/frame/ones_like:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
stft/frame/StridedSliceЖ
stft/frame/Reshape_1Reshape stft/frame/StridedSlice:output:0stft/frame/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
stft/frame/Reshape_1v
stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range_1/startv
stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range_1/deltaВ
stft/frame/range_1Range!stft/frame/range_1/start:output:0stft/frame/Neg_1:y:0!stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2
stft/frame/range_1
stft/frame/mul_2Mulstft/frame/range_1:output:0stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2
stft/frame/mul_2~
stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Reshape_2/shape/1Ћ
stft/frame/Reshape_2/shapePackstft/frame/Neg_1:y:0%stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/Reshape_2/shapeЄ
stft/frame/Reshape_2Reshapestft/frame/mul_2:z:0#stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
stft/frame/Reshape_2v
stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range_2/startv
stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range_2/deltaЎ
stft/frame/range_2Range!stft/frame/range_2/start:output:0stft/frame/floordiv_1:z:0!stft/frame/range_2/delta:output:0*
_output_shapes
: 2
stft/frame/range_2~
stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Reshape_3/shape/0А
stft/frame/Reshape_3/shapePack%stft/frame/Reshape_3/shape/0:output:0stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2
stft/frame/Reshape_3/shapeЂ
stft/frame/Reshape_3Reshapestft/frame/range_2:output:0#stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

: 2
stft/frame/Reshape_3
stft/frame/add_2AddV2stft/frame/Reshape_2:output:0stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
stft/frame/add_2є
stft/frame/GatherV2GatherV2stft/frame/Reshape_1:output:0stft/frame/add_2:z:0!stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 2
stft/frame/GatherV2Є
stft/frame/concat_3/values_1Packstft/frame/Neg_1:y:0stft/frame_length:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_3/values_1v
stft/frame/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat_3/axisф
stft/frame/concat_3ConcatV2stft/frame/split:output:0%stft/frame/concat_3/values_1:output:0stft/frame/split:output:2!stft/frame/concat_3/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_3Њ
stft/frame/Reshape_4Reshapestft/frame/GatherV2:output:0stft/frame/concat_3:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
stft/frame/Reshape_4x
stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
stft/hann_window/periodic
stft/hann_window/CastCast"stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2
stft/hann_window/Cast|
stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/FloorMod/yЅ
stft/hann_window/FloorModFloorModstft/frame_length:output:0$stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2
stft/hann_window/FloorModr
stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/sub/x
stft/hann_window/subSubstft/hann_window/sub/x:output:0stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2
stft/hann_window/sub
stft/hann_window/mulMulstft/hann_window/Cast:y:0stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2
stft/hann_window/mul
stft/hann_window/addAddV2stft/frame_length:output:0stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2
stft/hann_window/addv
stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/sub_1/y
stft/hann_window/sub_1Substft/hann_window/add:z:0!stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2
stft/hann_window/sub_1
stft/hann_window/Cast_1Caststft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
stft/hann_window/Cast_1~
stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/hann_window/range/start~
stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/range/deltaР
stft/hann_window/rangeRange%stft/hann_window/range/start:output:0stft/frame_length:output:0%stft/hann_window/range/delta:output:0*
_output_shapes	
:2
stft/hann_window/range
stft/hann_window/Cast_2Caststft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2
stft/hann_window/Cast_2u
stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2
stft/hann_window/Const
stft/hann_window/mul_1Mulstft/hann_window/Const:output:0stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2
stft/hann_window/mul_1
stft/hann_window/truedivRealDivstft/hann_window/mul_1:z:0stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2
stft/hann_window/truedivw
stft/hann_window/CosCosstft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2
stft/hann_window/Cosy
stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
stft/hann_window/mul_2/x
stft/hann_window/mul_2Mul!stft/hann_window/mul_2/x:output:0stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2
stft/hann_window/mul_2y
stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
stft/hann_window/sub_2/x
stft/hann_window/sub_2Sub!stft/hann_window/sub_2/x:output:0stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2
stft/hann_window/sub_2
stft/mulMulstft/frame/Reshape_4:output:0stft/hann_window/sub_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

stft/mulo
stft/rfft/packedPackstft/Const:output:0*
N*
T0*
_output_shapes
:2
stft/rfft/packedw
stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2
stft/rfft/fft_lengthy
	stft/rfftRFFTstft/mul:z:0stft/rfft/fft_length:output:0*,
_output_shapes
:џџџџџџџџџ@2
	stft/rfftZ
Abs
ComplexAbsstft/rfft:output:0*,
_output_shapes
:џџџџџџџџџ@2
AbsZ
SquareSquareAbs:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
Squarem
MatMulBatchMatMulV2
Square:y:0matmul_b*
T0*+
_output_shapes
:џџџџџџџџџ@2
MatMulc
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
ConstS
MaxMaxMatMul:output:0Const:output:0*
T0*
_output_shapes
: 2
Max[
	Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2
	Maximum/xx
MaximumMaximumMaximum/x:output:0MatMul:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@2	
MaximumT
LogLogMaximum:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
LogW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   A2	
Const_1H
Log_1LogConst_1:output:0*
T0*
_output_shapes
: 2
Log_1g
truedivRealDivLog:y:0	Log_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@2	
truedivS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
mul/xd
mulMulmul/x:output:0truediv:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
mul_
Maximum_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2
Maximum_1/xf
	Maximum_1MaximumMaximum_1/x:output:0Max:output:0*
T0*
_output_shapes
: 2
	Maximum_1E
Log_2LogMaximum_1:z:0*
T0*
_output_shapes
: 2
Log_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *   A2	
Const_2H
Log_3LogConst_2:output:0*
T0*
_output_shapes
: 2
Log_3X
	truediv_1RealDiv	Log_2:y:0	Log_3:y:0*
T0*
_output_shapes
: 2
	truediv_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2	
mul_1/xW
mul_1Mulmul_1/x:output:0truediv_1:z:0*
T0*
_output_shapes
: 2
mul_1[
subSubmul:z:0	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
subg
Const_3Const*
_output_shapes
:*
dtype0*!
valueB"          2	
Const_3Q
Max_1Maxsub:z:0Const_3:output:0*
T0*
_output_shapes
: 2
Max_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B2	
sub_1/yX
sub_1SubMax_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: 2
sub_1k
	Maximum_2Maximumsub:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
	Maximum_2b
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsMaximum_2:z:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ:	:S O
(
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	waveforms:%!

_output_shapes
:	
І

H__inference_resnet_block_layer_call_and_return_conditional_losses_210541

inputs
conv1_210516
conv1_210518
conv2_210522
conv2_210524
conv3_210528
conv3_210530
shortcut_210533
shortcut_210535
identityЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallЂconv3/StatefulPartitionedCallЂ shortcut/StatefulPartitionedCall
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_210516conv1_210518*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2101232
conv1/StatefulPartitionedCallя
relu1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu1_layer_call_and_return_conditional_losses_2101582
relu1/PartitionedCallЁ
conv2/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0conv2_210522conv2_210524*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2102012
conv2/StatefulPartitionedCallя
relu2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu2_layer_call_and_return_conditional_losses_2102362
relu2/PartitionedCallЁ
conv3/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0conv3_210528conv3_210530*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2102792
conv3/StatefulPartitionedCall
 shortcut/StatefulPartitionedCallStatefulPartitionedCallinputsshortcut_210533shortcut_210535*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_shortcut_layer_call_and_return_conditional_losses_2103342"
 shortcut/StatefulPartitionedCallЄ
add/addAddV2&conv3/StatefulPartitionedCall:output:0)shortcut/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
add/addр
out_block/PartitionedCallPartitionedCalladd/add:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_out_block_layer_call_and_return_conditional_losses_2103702
out_block/PartitionedCall§
IdentityIdentity"out_block/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall!^shortcut/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2D
 shortcut/StatefulPartitionedCall shortcut/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

B
&__inference_relu2_layer_call_fn_217358

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu2_layer_call_and_return_conditional_losses_2102362
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@ :S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
д;
ј
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_216447
input_15
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЉ
conv1/conv1d/ExpandDims
ExpandDimsinput_1$conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv1/conv1d/ExpandDimsЪ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimЯ
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1/conv1d/ExpandDims_1Ю
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv1/conv1dЄ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1/BiasAdd/ReadVariableOpЄ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
conv1/BiasAddn

relu1/ReluReluconv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimК
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2/conv1d/ExpandDimsЪ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimЯ
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv2/conv1d/ExpandDims_1Ю
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2/conv1dЄ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2/BiasAdd/ReadVariableOpЄ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
conv2/BiasAddn

relu2/ReluReluconv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimК
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv3/conv1d/ExpandDimsЪ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimЯ
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv3/conv1d/ExpandDims_1Ю
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv3/conv1dЄ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv3/BiasAdd/ReadVariableOpЄ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimВ
shortcut/conv1d/ExpandDims
ExpandDimsinput_1'shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
shortcut/conv1d/ExpandDimsг
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimл
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
shortcut/conv1d/ExpandDims_1к
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
shortcut/conv1d­
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЇ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
shortcut/BiasAdd/ReadVariableOpА
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2	
add/addk
out_block/ReluReluadd/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
out_block/Relut
IdentityIdentityout_block/Relu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@ :::::::::T P
+
_output_shapes
:џџџџџџџџџ@ 
!
_user_specified_name	input_1
П
]
A__inference_relu1_layer_call_and_return_conditional_losses_210629

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџ@@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@@:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs

=
&__inference_delta_layer_call_fn_216051
x
identityТ
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_delta_layer_call_and_return_conditional_losses_2124162
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:R N
/
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
№
Х
(__inference_res_net_layer_call_fn_214762
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_res_net_layer_call_and_return_conditional_losses_2132532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:џџџџџџџџџ:	::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:%!

_output_shapes
:	
Т
Ж
A__inference_conv3_layer_call_and_return_conditional_losses_211692

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsК
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЙ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@:::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѓ;
ї
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_216885

inputs5
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЈ
conv1/conv1d/ExpandDims
ExpandDimsinputs$conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv1/conv1d/ExpandDimsЫ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimа
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2
conv1/conv1d/ExpandDims_1Я
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1/conv1dЅ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv1/BiasAdd/ReadVariableOpЅ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv1/BiasAddo

relu1/ReluReluconv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimЛ
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv2/conv1d/ExpandDimsЬ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimб
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv2/conv1d/ExpandDims_1Я
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2/conv1dЅ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2/BiasAdd/ReadVariableOpЅ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv2/BiasAddo

relu2/ReluReluconv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimЛ
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv3/conv1d/ExpandDimsЬ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimб
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv3/conv1d/ExpandDims_1Я
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv3/conv1dЅ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv3/BiasAdd/ReadVariableOpЅ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimБ
shortcut/conv1d/ExpandDims
ExpandDimsinputs'shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
shortcut/conv1d/ExpandDimsд
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimм
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2
shortcut/conv1d/ExpandDims_1л
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
shortcut/conv1dЎ
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЈ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
shortcut/BiasAdd/ReadVariableOpБ
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
add/addl
out_block/ReluReluadd/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
out_block/Reluu
IdentityIdentityout_block/Relu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@@:::::::::S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ў
џ
C__inference_res_net_layer_call_and_return_conditional_losses_215223

inputs 
log_mel_spectrogram_matmul_bB
>resnet_block_conv1_conv1d_expanddims_1_readvariableop_resource6
2resnet_block_conv1_biasadd_readvariableop_resourceB
>resnet_block_conv2_conv1d_expanddims_1_readvariableop_resource6
2resnet_block_conv2_biasadd_readvariableop_resourceB
>resnet_block_conv3_conv1d_expanddims_1_readvariableop_resource6
2resnet_block_conv3_biasadd_readvariableop_resourceE
Aresnet_block_shortcut_conv1d_expanddims_1_readvariableop_resource9
5resnet_block_shortcut_biasadd_readvariableop_resourceD
@resnet_block_1_conv1_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_1_conv1_biasadd_readvariableop_resourceD
@resnet_block_1_conv2_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_1_conv2_biasadd_readvariableop_resourceD
@resnet_block_1_conv3_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_1_conv3_biasadd_readvariableop_resourceG
Cresnet_block_1_shortcut_conv1d_expanddims_1_readvariableop_resource;
7resnet_block_1_shortcut_biasadd_readvariableop_resourceD
@resnet_block_2_conv1_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_2_conv1_biasadd_readvariableop_resourceD
@resnet_block_2_conv2_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_2_conv2_biasadd_readvariableop_resourceD
@resnet_block_2_conv3_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_2_conv3_biasadd_readvariableop_resourceG
Cresnet_block_2_shortcut_conv1d_expanddims_1_readvariableop_resource;
7resnet_block_2_shortcut_biasadd_readvariableop_resourceD
@resnet_block_3_conv1_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_3_conv1_biasadd_readvariableop_resourceD
@resnet_block_3_conv2_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_3_conv2_biasadd_readvariableop_resourceD
@resnet_block_3_conv3_conv1d_expanddims_1_readvariableop_resource8
4resnet_block_3_conv3_biasadd_readvariableop_resourceG
Cresnet_block_3_shortcut_conv1d_expanddims_1_readvariableop_resource;
7resnet_block_3_shortcut_biasadd_readvariableop_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource&
"fc2_matmul_readvariableop_resource'
#fc2_biasadd_readvariableop_resource&
"fc3_matmul_readvariableop_resource'
#fc3_biasadd_readvariableop_resource
identityo
SqueezeSqueezeinputs*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2	
Squeeze
%log_mel_spectrogram/stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2'
%log_mel_spectrogram/stft/frame_length
#log_mel_spectrogram/stft/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2%
#log_mel_spectrogram/stft/frame_step
log_mel_spectrogram/stft/ConstConst*
_output_shapes
: *
dtype0*
value
B :2 
log_mel_spectrogram/stft/Const
#log_mel_spectrogram/stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#log_mel_spectrogram/stft/frame/axis
$log_mel_spectrogram/stft/frame/ShapeShapeSqueeze:output:0*
T0*
_output_shapes
:2&
$log_mel_spectrogram/stft/frame/Shape
#log_mel_spectrogram/stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#log_mel_spectrogram/stft/frame/Rank
*log_mel_spectrogram/stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*log_mel_spectrogram/stft/frame/range/start
*log_mel_spectrogram/stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*log_mel_spectrogram/stft/frame/range/delta
$log_mel_spectrogram/stft/frame/rangeRange3log_mel_spectrogram/stft/frame/range/start:output:0,log_mel_spectrogram/stft/frame/Rank:output:03log_mel_spectrogram/stft/frame/range/delta:output:0*
_output_shapes
:2&
$log_mel_spectrogram/stft/frame/rangeЛ
2log_mel_spectrogram/stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ24
2log_mel_spectrogram/stft/frame/strided_slice/stackЖ
4log_mel_spectrogram/stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4log_mel_spectrogram/stft/frame/strided_slice/stack_1Ж
4log_mel_spectrogram/stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4log_mel_spectrogram/stft/frame/strided_slice/stack_2
,log_mel_spectrogram/stft/frame/strided_sliceStridedSlice-log_mel_spectrogram/stft/frame/range:output:0;log_mel_spectrogram/stft/frame/strided_slice/stack:output:0=log_mel_spectrogram/stft/frame/strided_slice/stack_1:output:0=log_mel_spectrogram/stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,log_mel_spectrogram/stft/frame/strided_slice
$log_mel_spectrogram/stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$log_mel_spectrogram/stft/frame/sub/yЭ
"log_mel_spectrogram/stft/frame/subSub,log_mel_spectrogram/stft/frame/Rank:output:0-log_mel_spectrogram/stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/subг
$log_mel_spectrogram/stft/frame/sub_1Sub&log_mel_spectrogram/stft/frame/sub:z:05log_mel_spectrogram/stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/sub_1
'log_mel_spectrogram/stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'log_mel_spectrogram/stft/frame/packed/1
%log_mel_spectrogram/stft/frame/packedPack5log_mel_spectrogram/stft/frame/strided_slice:output:00log_mel_spectrogram/stft/frame/packed/1:output:0(log_mel_spectrogram/stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2'
%log_mel_spectrogram/stft/frame/packedЂ
.log_mel_spectrogram/stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.log_mel_spectrogram/stft/frame/split/split_dimК
$log_mel_spectrogram/stft/frame/splitSplitV-log_mel_spectrogram/stft/frame/Shape:output:0.log_mel_spectrogram/stft/frame/packed:output:07log_mel_spectrogram/stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2&
$log_mel_spectrogram/stft/frame/split
,log_mel_spectrogram/stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2.
,log_mel_spectrogram/stft/frame/Reshape/shapeЃ
.log_mel_spectrogram/stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 20
.log_mel_spectrogram/stft/frame/Reshape/shape_1ф
&log_mel_spectrogram/stft/frame/ReshapeReshape-log_mel_spectrogram/stft/frame/split:output:17log_mel_spectrogram/stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2(
&log_mel_spectrogram/stft/frame/Reshape
#log_mel_spectrogram/stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2%
#log_mel_spectrogram/stft/frame/Size
%log_mel_spectrogram/stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%log_mel_spectrogram/stft/frame/Size_1
$log_mel_spectrogram/stft/frame/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$log_mel_spectrogram/stft/frame/ConstЁ
"log_mel_spectrogram/stft/frame/NegNeg/log_mel_spectrogram/stft/frame/Reshape:output:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/Negе
'log_mel_spectrogram/stft/frame/floordivFloorDiv&log_mel_spectrogram/stft/frame/Neg:y:0,log_mel_spectrogram/stft/frame_step:output:0*
T0*
_output_shapes
: 2)
'log_mel_spectrogram/stft/frame/floordivЁ
$log_mel_spectrogram/stft/frame/Neg_1Neg+log_mel_spectrogram/stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/Neg_1
&log_mel_spectrogram/stft/frame/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_spectrogram/stft/frame/sub_2/yЯ
$log_mel_spectrogram/stft/frame/sub_2Sub(log_mel_spectrogram/stft/frame/Neg_1:y:0/log_mel_spectrogram/stft/frame/sub_2/y:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/sub_2Ш
"log_mel_spectrogram/stft/frame/mulMul,log_mel_spectrogram/stft/frame_step:output:0(log_mel_spectrogram/stft/frame/sub_2:z:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/mulЪ
"log_mel_spectrogram/stft/frame/addAddV2.log_mel_spectrogram/stft/frame_length:output:0&log_mel_spectrogram/stft/frame/mul:z:0*
T0*
_output_shapes
: 2$
"log_mel_spectrogram/stft/frame/addЭ
$log_mel_spectrogram/stft/frame/sub_3Sub&log_mel_spectrogram/stft/frame/add:z:0/log_mel_spectrogram/stft/frame/Reshape:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/sub_3
(log_mel_spectrogram/stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2*
(log_mel_spectrogram/stft/frame/Maximum/xй
&log_mel_spectrogram/stft/frame/MaximumMaximum1log_mel_spectrogram/stft/frame/Maximum/x:output:0(log_mel_spectrogram/stft/frame/sub_3:z:0*
T0*
_output_shapes
: 2(
&log_mel_spectrogram/stft/frame/Maximum
*log_mel_spectrogram/stft/frame/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*log_mel_spectrogram/stft/frame/zeros/mul/yп
(log_mel_spectrogram/stft/frame/zeros/mulMul,log_mel_spectrogram/stft/frame/Size:output:03log_mel_spectrogram/stft/frame/zeros/mul/y:output:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/frame/zeros/mul
+log_mel_spectrogram/stft/frame/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2-
+log_mel_spectrogram/stft/frame/zeros/Less/yу
)log_mel_spectrogram/stft/frame/zeros/LessLess,log_mel_spectrogram/stft/frame/zeros/mul:z:04log_mel_spectrogram/stft/frame/zeros/Less/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/zeros/Less 
-log_mel_spectrogram/stft/frame/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-log_mel_spectrogram/stft/frame/zeros/packed/1і
+log_mel_spectrogram/stft/frame/zeros/packedPack,log_mel_spectrogram/stft/frame/Size:output:06log_mel_spectrogram/stft/frame/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2-
+log_mel_spectrogram/stft/frame/zeros/packed
*log_mel_spectrogram/stft/frame/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2,
*log_mel_spectrogram/stft/frame/zeros/Constш
$log_mel_spectrogram/stft/frame/zerosFill4log_mel_spectrogram/stft/frame/zeros/packed:output:03log_mel_spectrogram/stft/frame/zeros/Const:output:0*
T0*
_output_shapes

:2&
$log_mel_spectrogram/stft/frame/zeros
+log_mel_spectrogram/stft/frame/packed_1/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2-
+log_mel_spectrogram/stft/frame/packed_1/0/0ю
)log_mel_spectrogram/stft/frame/packed_1/0Pack4log_mel_spectrogram/stft/frame/packed_1/0/0:output:0*log_mel_spectrogram/stft/frame/Maximum:z:0*
N*
T0*
_output_shapes
:2+
)log_mel_spectrogram/stft/frame/packed_1/0Р
'log_mel_spectrogram/stft/frame/packed_1Pack2log_mel_spectrogram/stft/frame/packed_1/0:output:0*
N*
T0*
_output_shapes

:2)
'log_mel_spectrogram/stft/frame/packed_1
,log_mel_spectrogram/stft/frame/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/frame/zeros_1/mul/yч
*log_mel_spectrogram/stft/frame/zeros_1/mulMul.log_mel_spectrogram/stft/frame/Size_1:output:05log_mel_spectrogram/stft/frame/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2,
*log_mel_spectrogram/stft/frame/zeros_1/mulЁ
-log_mel_spectrogram/stft/frame/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2/
-log_mel_spectrogram/stft/frame/zeros_1/Less/yы
+log_mel_spectrogram/stft/frame/zeros_1/LessLess.log_mel_spectrogram/stft/frame/zeros_1/mul:z:06log_mel_spectrogram/stft/frame/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2-
+log_mel_spectrogram/stft/frame/zeros_1/LessЄ
/log_mel_spectrogram/stft/frame/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :21
/log_mel_spectrogram/stft/frame/zeros_1/packed/1ў
-log_mel_spectrogram/stft/frame/zeros_1/packedPack.log_mel_spectrogram/stft/frame/Size_1:output:08log_mel_spectrogram/stft/frame/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2/
-log_mel_spectrogram/stft/frame/zeros_1/packed
,log_mel_spectrogram/stft/frame/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/zeros_1/Constю
&log_mel_spectrogram/stft/frame/zeros_1Fill6log_mel_spectrogram/stft/frame/zeros_1/packed:output:05log_mel_spectrogram/stft/frame/zeros_1/Const:output:0*
T0*
_output_shapes

: 2(
&log_mel_spectrogram/stft/frame/zeros_1
*log_mel_spectrogram/stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*log_mel_spectrogram/stft/frame/concat/axisг
%log_mel_spectrogram/stft/frame/concatConcatV2-log_mel_spectrogram/stft/frame/zeros:output:00log_mel_spectrogram/stft/frame/packed_1:output:0/log_mel_spectrogram/stft/frame/zeros_1:output:03log_mel_spectrogram/stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes

:2'
%log_mel_spectrogram/stft/frame/concat
$log_mel_spectrogram/stft/frame/PadV2PadV2Squeeze:output:0.log_mel_spectrogram/stft/frame/concat:output:0-log_mel_spectrogram/stft/frame/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2&
$log_mel_spectrogram/stft/frame/PadV2­
&log_mel_spectrogram/stft/frame/Shape_1Shape-log_mel_spectrogram/stft/frame/PadV2:output:0*
T0*
_output_shapes
:2(
&log_mel_spectrogram/stft/frame/Shape_1
&log_mel_spectrogram/stft/frame/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_spectrogram/stft/frame/add_1/yо
$log_mel_spectrogram/stft/frame/add_1AddV25log_mel_spectrogram/stft/frame/strided_slice:output:0/log_mel_spectrogram/stft/frame/add_1/y:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/add_1й
4log_mel_spectrogram/stft/frame/strided_slice_1/stackPack5log_mel_spectrogram/stft/frame/strided_slice:output:0*
N*
T0*
_output_shapes
:26
4log_mel_spectrogram/stft/frame/strided_slice_1/stackа
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_1Pack(log_mel_spectrogram/stft/frame/add_1:z:0*
N*
T0*
_output_shapes
:28
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_1К
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6log_mel_spectrogram/stft/frame/strided_slice_1/stack_2Ј
.log_mel_spectrogram/stft/frame/strided_slice_1StridedSlice/log_mel_spectrogram/stft/frame/Shape_1:output:0=log_mel_spectrogram/stft/frame/strided_slice_1/stack:output:0?log_mel_spectrogram/stft/frame/strided_slice_1/stack_1:output:0?log_mel_spectrogram/stft/frame/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.log_mel_spectrogram/stft/frame/strided_slice_1
(log_mel_spectrogram/stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(log_mel_spectrogram/stft/frame/gcd/Const
+log_mel_spectrogram/stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_spectrogram/stft/frame/floordiv_1/yщ
)log_mel_spectrogram/stft/frame/floordiv_1FloorDiv.log_mel_spectrogram/stft/frame_length:output:04log_mel_spectrogram/stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/floordiv_1
+log_mel_spectrogram/stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_spectrogram/stft/frame/floordiv_2/yч
)log_mel_spectrogram/stft/frame/floordiv_2FloorDiv,log_mel_spectrogram/stft/frame_step:output:04log_mel_spectrogram/stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/floordiv_2
+log_mel_spectrogram/stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_spectrogram/stft/frame/floordiv_3/yђ
)log_mel_spectrogram/stft/frame/floordiv_3FloorDiv7log_mel_spectrogram/stft/frame/strided_slice_1:output:04log_mel_spectrogram/stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2+
)log_mel_spectrogram/stft/frame/floordiv_3
&log_mel_spectrogram/stft/frame/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_spectrogram/stft/frame/mul_1/yд
$log_mel_spectrogram/stft/frame/mul_1Mul-log_mel_spectrogram/stft/frame/floordiv_3:z:0/log_mel_spectrogram/stft/frame/mul_1/y:output:0*
T0*
_output_shapes
: 2&
$log_mel_spectrogram/stft/frame/mul_1Ф
0log_mel_spectrogram/stft/frame/concat_1/values_1Pack(log_mel_spectrogram/stft/frame/mul_1:z:0*
N*
T0*
_output_shapes
:22
0log_mel_spectrogram/stft/frame/concat_1/values_1
,log_mel_spectrogram/stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/concat_1/axisм
'log_mel_spectrogram/stft/frame/concat_1ConcatV2-log_mel_spectrogram/stft/frame/split:output:09log_mel_spectrogram/stft/frame/concat_1/values_1:output:0-log_mel_spectrogram/stft/frame/split:output:25log_mel_spectrogram/stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'log_mel_spectrogram/stft/frame/concat_1Њ
2log_mel_spectrogram/stft/frame/concat_2/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :24
2log_mel_spectrogram/stft/frame/concat_2/values_1/1
0log_mel_spectrogram/stft/frame/concat_2/values_1Pack-log_mel_spectrogram/stft/frame/floordiv_3:z:0;log_mel_spectrogram/stft/frame/concat_2/values_1/1:output:0*
N*
T0*
_output_shapes
:22
0log_mel_spectrogram/stft/frame/concat_2/values_1
,log_mel_spectrogram/stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/concat_2/axisм
'log_mel_spectrogram/stft/frame/concat_2ConcatV2-log_mel_spectrogram/stft/frame/split:output:09log_mel_spectrogram/stft/frame/concat_2/values_1:output:0-log_mel_spectrogram/stft/frame/split:output:25log_mel_spectrogram/stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2)
'log_mel_spectrogram/stft/frame/concat_2 
)log_mel_spectrogram/stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2+
)log_mel_spectrogram/stft/frame/zeros_likeЊ
.log_mel_spectrogram/stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:20
.log_mel_spectrogram/stft/frame/ones_like/ShapeЂ
.log_mel_spectrogram/stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :20
.log_mel_spectrogram/stft/frame/ones_like/Constѓ
(log_mel_spectrogram/stft/frame/ones_likeFill7log_mel_spectrogram/stft/frame/ones_like/Shape:output:07log_mel_spectrogram/stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2*
(log_mel_spectrogram/stft/frame/ones_likeњ
+log_mel_spectrogram/stft/frame/StridedSliceStridedSlice-log_mel_spectrogram/stft/frame/PadV2:output:02log_mel_spectrogram/stft/frame/zeros_like:output:00log_mel_spectrogram/stft/frame/concat_1:output:01log_mel_spectrogram/stft/frame/ones_like:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2-
+log_mel_spectrogram/stft/frame/StridedSlice
(log_mel_spectrogram/stft/frame/Reshape_1Reshape4log_mel_spectrogram/stft/frame/StridedSlice:output:00log_mel_spectrogram/stft/frame/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
(log_mel_spectrogram/stft/frame/Reshape_1
,log_mel_spectrogram/stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/range_1/start
,log_mel_spectrogram/stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/frame/range_1/delta
&log_mel_spectrogram/stft/frame/range_1Range5log_mel_spectrogram/stft/frame/range_1/start:output:0(log_mel_spectrogram/stft/frame/Neg_1:y:05log_mel_spectrogram/stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2(
&log_mel_spectrogram/stft/frame/range_1с
$log_mel_spectrogram/stft/frame/mul_2Mul/log_mel_spectrogram/stft/frame/range_1:output:0-log_mel_spectrogram/stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2&
$log_mel_spectrogram/stft/frame/mul_2І
0log_mel_spectrogram/stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0log_mel_spectrogram/stft/frame/Reshape_2/shape/1ћ
.log_mel_spectrogram/stft/frame/Reshape_2/shapePack(log_mel_spectrogram/stft/frame/Neg_1:y:09log_mel_spectrogram/stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:20
.log_mel_spectrogram/stft/frame/Reshape_2/shapeє
(log_mel_spectrogram/stft/frame/Reshape_2Reshape(log_mel_spectrogram/stft/frame/mul_2:z:07log_mel_spectrogram/stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(log_mel_spectrogram/stft/frame/Reshape_2
,log_mel_spectrogram/stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/range_2/start
,log_mel_spectrogram/stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/frame/range_2/delta
&log_mel_spectrogram/stft/frame/range_2Range5log_mel_spectrogram/stft/frame/range_2/start:output:0-log_mel_spectrogram/stft/frame/floordiv_1:z:05log_mel_spectrogram/stft/frame/range_2/delta:output:0*
_output_shapes
: 2(
&log_mel_spectrogram/stft/frame/range_2І
0log_mel_spectrogram/stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :22
0log_mel_spectrogram/stft/frame/Reshape_3/shape/0
.log_mel_spectrogram/stft/frame/Reshape_3/shapePack9log_mel_spectrogram/stft/frame/Reshape_3/shape/0:output:0-log_mel_spectrogram/stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:20
.log_mel_spectrogram/stft/frame/Reshape_3/shapeђ
(log_mel_spectrogram/stft/frame/Reshape_3Reshape/log_mel_spectrogram/stft/frame/range_2:output:07log_mel_spectrogram/stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

: 2*
(log_mel_spectrogram/stft/frame/Reshape_3э
$log_mel_spectrogram/stft/frame/add_2AddV21log_mel_spectrogram/stft/frame/Reshape_2:output:01log_mel_spectrogram/stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$log_mel_spectrogram/stft/frame/add_2и
'log_mel_spectrogram/stft/frame/GatherV2GatherV21log_mel_spectrogram/stft/frame/Reshape_1:output:0(log_mel_spectrogram/stft/frame/add_2:z:05log_mel_spectrogram/stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 2)
'log_mel_spectrogram/stft/frame/GatherV2є
0log_mel_spectrogram/stft/frame/concat_3/values_1Pack(log_mel_spectrogram/stft/frame/Neg_1:y:0.log_mel_spectrogram/stft/frame_length:output:0*
N*
T0*
_output_shapes
:22
0log_mel_spectrogram/stft/frame/concat_3/values_1
,log_mel_spectrogram/stft/frame/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,log_mel_spectrogram/stft/frame/concat_3/axisм
'log_mel_spectrogram/stft/frame/concat_3ConcatV2-log_mel_spectrogram/stft/frame/split:output:09log_mel_spectrogram/stft/frame/concat_3/values_1:output:0-log_mel_spectrogram/stft/frame/split:output:25log_mel_spectrogram/stft/frame/concat_3/axis:output:0*
N*
T0*
_output_shapes
:2)
'log_mel_spectrogram/stft/frame/concat_3њ
(log_mel_spectrogram/stft/frame/Reshape_4Reshape0log_mel_spectrogram/stft/frame/GatherV2:output:00log_mel_spectrogram/stft/frame/concat_3:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2*
(log_mel_spectrogram/stft/frame/Reshape_4 
-log_mel_spectrogram/stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-log_mel_spectrogram/stft/hann_window/periodicЦ
)log_mel_spectrogram/stft/hann_window/CastCast6log_mel_spectrogram/stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2+
)log_mel_spectrogram/stft/hann_window/CastЄ
/log_mel_spectrogram/stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :21
/log_mel_spectrogram/stft/hann_window/FloorMod/yѕ
-log_mel_spectrogram/stft/hann_window/FloorModFloorMod.log_mel_spectrogram/stft/frame_length:output:08log_mel_spectrogram/stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2/
-log_mel_spectrogram/stft/hann_window/FloorMod
*log_mel_spectrogram/stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*log_mel_spectrogram/stft/hann_window/sub/xф
(log_mel_spectrogram/stft/hann_window/subSub3log_mel_spectrogram/stft/hann_window/sub/x:output:01log_mel_spectrogram/stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/hann_window/subй
(log_mel_spectrogram/stft/hann_window/mulMul-log_mel_spectrogram/stft/hann_window/Cast:y:0,log_mel_spectrogram/stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/hann_window/mulм
(log_mel_spectrogram/stft/hann_window/addAddV2.log_mel_spectrogram/stft/frame_length:output:0,log_mel_spectrogram/stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2*
(log_mel_spectrogram/stft/hann_window/add
,log_mel_spectrogram/stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_spectrogram/stft/hann_window/sub_1/yх
*log_mel_spectrogram/stft/hann_window/sub_1Sub,log_mel_spectrogram/stft/hann_window/add:z:05log_mel_spectrogram/stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2,
*log_mel_spectrogram/stft/hann_window/sub_1Т
+log_mel_spectrogram/stft/hann_window/Cast_1Cast.log_mel_spectrogram/stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+log_mel_spectrogram/stft/hann_window/Cast_1І
0log_mel_spectrogram/stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0log_mel_spectrogram/stft/hann_window/range/startІ
0log_mel_spectrogram/stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0log_mel_spectrogram/stft/hann_window/range/deltaЄ
*log_mel_spectrogram/stft/hann_window/rangeRange9log_mel_spectrogram/stft/hann_window/range/start:output:0.log_mel_spectrogram/stft/frame_length:output:09log_mel_spectrogram/stft/hann_window/range/delta:output:0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/rangeЬ
+log_mel_spectrogram/stft/hann_window/Cast_2Cast3log_mel_spectrogram/stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2-
+log_mel_spectrogram/stft/hann_window/Cast_2
*log_mel_spectrogram/stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2,
*log_mel_spectrogram/stft/hann_window/Constы
*log_mel_spectrogram/stft/hann_window/mul_1Mul3log_mel_spectrogram/stft/hann_window/Const:output:0/log_mel_spectrogram/stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/mul_1ю
,log_mel_spectrogram/stft/hann_window/truedivRealDiv.log_mel_spectrogram/stft/hann_window/mul_1:z:0/log_mel_spectrogram/stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2.
,log_mel_spectrogram/stft/hann_window/truedivГ
(log_mel_spectrogram/stft/hann_window/CosCos0log_mel_spectrogram/stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2*
(log_mel_spectrogram/stft/hann_window/CosЁ
,log_mel_spectrogram/stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,log_mel_spectrogram/stft/hann_window/mul_2/xъ
*log_mel_spectrogram/stft/hann_window/mul_2Mul5log_mel_spectrogram/stft/hann_window/mul_2/x:output:0,log_mel_spectrogram/stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/mul_2Ё
,log_mel_spectrogram/stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,log_mel_spectrogram/stft/hann_window/sub_2/xь
*log_mel_spectrogram/stft/hann_window/sub_2Sub5log_mel_spectrogram/stft/hann_window/sub_2/x:output:0.log_mel_spectrogram/stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2,
*log_mel_spectrogram/stft/hann_window/sub_2н
log_mel_spectrogram/stft/mulMul1log_mel_spectrogram/stft/frame/Reshape_4:output:0.log_mel_spectrogram/stft/hann_window/sub_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/stft/mulЋ
$log_mel_spectrogram/stft/rfft/packedPack'log_mel_spectrogram/stft/Const:output:0*
N*
T0*
_output_shapes
:2&
$log_mel_spectrogram/stft/rfft/packed
(log_mel_spectrogram/stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2*
(log_mel_spectrogram/stft/rfft/fft_lengthЩ
log_mel_spectrogram/stft/rfftRFFT log_mel_spectrogram/stft/mul:z:01log_mel_spectrogram/stft/rfft/fft_length:output:0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/stft/rfft
log_mel_spectrogram/Abs
ComplexAbs&log_mel_spectrogram/stft/rfft:output:0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Abs
log_mel_spectrogram/SquareSquarelog_mel_spectrogram/Abs:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/SquareН
log_mel_spectrogram/MatMulBatchMatMulV2log_mel_spectrogram/Square:y:0log_mel_spectrogram_matmul_b*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/MatMul
log_mel_spectrogram/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
log_mel_spectrogram/ConstЃ
log_mel_spectrogram/MaxMax#log_mel_spectrogram/MatMul:output:0"log_mel_spectrogram/Const:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Max
log_mel_spectrogram/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2
log_mel_spectrogram/Maximum/xШ
log_mel_spectrogram/MaximumMaximum&log_mel_spectrogram/Maximum/x:output:0#log_mel_spectrogram/MatMul:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Maximum
log_mel_spectrogram/LogLoglog_mel_spectrogram/Maximum:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Log
log_mel_spectrogram/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/Const_1
log_mel_spectrogram/Log_1Log$log_mel_spectrogram/Const_1:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Log_1З
log_mel_spectrogram/truedivRealDivlog_mel_spectrogram/Log:y:0log_mel_spectrogram/Log_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/truediv{
log_mel_spectrogram/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/mul/xД
log_mel_spectrogram/mulMul"log_mel_spectrogram/mul/x:output:0log_mel_spectrogram/truediv:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/mul
log_mel_spectrogram/Maximum_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ц$2!
log_mel_spectrogram/Maximum_1/xЖ
log_mel_spectrogram/Maximum_1Maximum(log_mel_spectrogram/Maximum_1/x:output:0 log_mel_spectrogram/Max:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Maximum_1
log_mel_spectrogram/Log_2Log!log_mel_spectrogram/Maximum_1:z:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Log_2
log_mel_spectrogram/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/Const_2
log_mel_spectrogram/Log_3Log$log_mel_spectrogram/Const_2:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Log_3Ј
log_mel_spectrogram/truediv_1RealDivlog_mel_spectrogram/Log_2:y:0log_mel_spectrogram/Log_3:y:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/truediv_1
log_mel_spectrogram/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
log_mel_spectrogram/mul_1/xЇ
log_mel_spectrogram/mul_1Mul$log_mel_spectrogram/mul_1/x:output:0!log_mel_spectrogram/truediv_1:z:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/mul_1Ћ
log_mel_spectrogram/subSublog_mel_spectrogram/mul:z:0log_mel_spectrogram/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/sub
log_mel_spectrogram/Const_3Const*
_output_shapes
:*
dtype0*!
valueB"          2
log_mel_spectrogram/Const_3Ё
log_mel_spectrogram/Max_1Maxlog_mel_spectrogram/sub:z:0$log_mel_spectrogram/Const_3:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/Max_1
log_mel_spectrogram/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B2
log_mel_spectrogram/sub_1/yЈ
log_mel_spectrogram/sub_1Sub"log_mel_spectrogram/Max_1:output:0$log_mel_spectrogram/sub_1/y:output:0*
T0*
_output_shapes
: 2
log_mel_spectrogram/sub_1Л
log_mel_spectrogram/Maximum_2Maximumlog_mel_spectrogram/sub:z:0log_mel_spectrogram/sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@2
log_mel_spectrogram/Maximum_2
"log_mel_spectrogram/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"log_mel_spectrogram/ExpandDims/dimи
log_mel_spectrogram/ExpandDims
ExpandDims!log_mel_spectrogram/Maximum_2:z:0+log_mel_spectrogram/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2 
log_mel_spectrogram/ExpandDimsy
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm
	transpose	Transpose'log_mel_spectrogram/ExpandDims:output:0transpose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
	transpose
)mfccs_from_log_mel_spectrograms/dct/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)mfccs_from_log_mel_spectrograms/dct/Const
*mfccs_from_log_mel_spectrograms/dct/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*mfccs_from_log_mel_spectrograms/dct/Cast/xС
(mfccs_from_log_mel_spectrograms/dct/CastCast3mfccs_from_log_mel_spectrograms/dct/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(mfccs_from_log_mel_spectrograms/dct/CastЄ
/mfccs_from_log_mel_spectrograms/dct/range/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/mfccs_from_log_mel_spectrograms/dct/range/startЄ
/mfccs_from_log_mel_spectrograms/dct/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/mfccs_from_log_mel_spectrograms/dct/range/deltaв
.mfccs_from_log_mel_spectrograms/dct/range/CastCast8mfccs_from_log_mel_spectrograms/dct/range/start:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.mfccs_from_log_mel_spectrograms/dct/range/Castж
0mfccs_from_log_mel_spectrograms/dct/range/Cast_1Cast8mfccs_from_log_mel_spectrograms/dct/range/delta:output:0*

DstT0*

SrcT0*
_output_shapes
: 22
0mfccs_from_log_mel_spectrograms/dct/range/Cast_1
)mfccs_from_log_mel_spectrograms/dct/rangeRange2mfccs_from_log_mel_spectrograms/dct/range/Cast:y:0,mfccs_from_log_mel_spectrograms/dct/Cast:y:04mfccs_from_log_mel_spectrograms/dct/range/Cast_1:y:0*

Tidx0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/rangeВ
'mfccs_from_log_mel_spectrograms/dct/NegNeg2mfccs_from_log_mel_spectrograms/dct/range:output:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/Neg
)mfccs_from_log_mel_spectrograms/dct/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *лI@2+
)mfccs_from_log_mel_spectrograms/dct/mul/yп
'mfccs_from_log_mel_spectrograms/dct/mulMul+mfccs_from_log_mel_spectrograms/dct/Neg:y:02mfccs_from_log_mel_spectrograms/dct/mul/y:output:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/mul
+mfccs_from_log_mel_spectrograms/dct/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2-
+mfccs_from_log_mel_spectrograms/dct/mul_1/yх
)mfccs_from_log_mel_spectrograms/dct/mul_1Mul+mfccs_from_log_mel_spectrograms/dct/mul:z:04mfccs_from_log_mel_spectrograms/dct/mul_1/y:output:0*
T0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/mul_1ч
+mfccs_from_log_mel_spectrograms/dct/truedivRealDiv-mfccs_from_log_mel_spectrograms/dct/mul_1:z:0,mfccs_from_log_mel_spectrograms/dct/Cast:y:0*
T0*
_output_shapes
:2-
+mfccs_from_log_mel_spectrograms/dct/truedivц
+mfccs_from_log_mel_spectrograms/dct/ComplexComplex2mfccs_from_log_mel_spectrograms/dct/Const:output:0/mfccs_from_log_mel_spectrograms/dct/truediv:z:0*
_output_shapes
:2-
+mfccs_from_log_mel_spectrograms/dct/ComplexБ
'mfccs_from_log_mel_spectrograms/dct/ExpExp1mfccs_from_log_mel_spectrograms/dct/Complex:out:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/ExpЃ
+mfccs_from_log_mel_spectrograms/dct/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J   @    2-
+mfccs_from_log_mel_spectrograms/dct/mul_2/xх
)mfccs_from_log_mel_spectrograms/dct/mul_2Mul4mfccs_from_log_mel_spectrograms/dct/mul_2/x:output:0+mfccs_from_log_mel_spectrograms/dct/Exp:y:0*
T0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/mul_2Њ
.mfccs_from_log_mel_spectrograms/dct/rfft/ConstConst*
_output_shapes
:*
dtype0*
valueB:
20
.mfccs_from_log_mel_spectrograms/dct/rfft/Constп
5mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                27
5mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddingsь
,mfccs_from_log_mel_spectrograms/dct/rfft/PadPadtranspose:y:0>mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
2.
,mfccs_from_log_mel_spectrograms/dct/rfft/PadД
3mfccs_from_log_mel_spectrograms/dct/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:
25
3mfccs_from_log_mel_spectrograms/dct/rfft/fft_length
(mfccs_from_log_mel_spectrograms/dct/rfftRFFT5mfccs_from_log_mel_spectrograms/dct/rfft/Pad:output:0<mfccs_from_log_mel_spectrograms/dct/rfft/fft_length:output:0*/
_output_shapes
:џџџџџџџџџ@2*
(mfccs_from_log_mel_spectrograms/dct/rfftУ
7mfccs_from_log_mel_spectrograms/dct/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7mfccs_from_log_mel_spectrograms/dct/strided_slice/stackЧ
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1Ч
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2с
1mfccs_from_log_mel_spectrograms/dct/strided_sliceStridedSlice1mfccs_from_log_mel_spectrograms/dct/rfft:output:0@mfccs_from_log_mel_spectrograms/dct/strided_slice/stack:output:0Bmfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1:output:0Bmfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

begin_mask*
ellipsis_mask23
1mfccs_from_log_mel_spectrograms/dct/strided_slice
)mfccs_from_log_mel_spectrograms/dct/mul_3Mul:mfccs_from_log_mel_spectrograms/dct/strided_slice:output:0-mfccs_from_log_mel_spectrograms/dct/mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2+
)mfccs_from_log_mel_spectrograms/dct/mul_3М
(mfccs_from_log_mel_spectrograms/dct/RealReal-mfccs_from_log_mel_spectrograms/dct/mul_3:z:0*/
_output_shapes
:џџџџџџџџџ@2*
(mfccs_from_log_mel_spectrograms/dct/Real
&mfccs_from_log_mel_spectrograms/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2(
&mfccs_from_log_mel_spectrograms/Cast/xЕ
$mfccs_from_log_mel_spectrograms/CastCast/mfccs_from_log_mel_spectrograms/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$mfccs_from_log_mel_spectrograms/Cast
%mfccs_from_log_mel_spectrograms/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%mfccs_from_log_mel_spectrograms/mul/yЬ
#mfccs_from_log_mel_spectrograms/mulMul(mfccs_from_log_mel_spectrograms/Cast:y:0.mfccs_from_log_mel_spectrograms/mul/y:output:0*
T0*
_output_shapes
: 2%
#mfccs_from_log_mel_spectrograms/mulЁ
%mfccs_from_log_mel_spectrograms/RsqrtRsqrt'mfccs_from_log_mel_spectrograms/mul:z:0*
T0*
_output_shapes
: 2'
%mfccs_from_log_mel_spectrograms/Rsqrtэ
%mfccs_from_log_mel_spectrograms/mul_1Mul1mfccs_from_log_mel_spectrograms/dct/Real:output:0)mfccs_from_log_mel_spectrograms/Rsqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2'
%mfccs_from_log_mel_spectrograms/mul_1c
SquareSquaretranspose:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
	keep_dims(2
Sum\
SqrtSqrtSum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Sqrt
delta/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose/permГ
delta/transpose	Transpose)mfccs_from_log_mel_spectrograms/mul_1:z:0delta/transpose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose
delta/ConstConst*
_output_shapes

:*
dtype0*9
value0B."                               2
delta/ConstЉ
delta/MirrorPad	MirrorPaddelta/transpose:y:0delta/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
mode	SYMMETRIC2
delta/MirrorPads
delta/arange/startConst*
_output_shapes
: *
dtype0*
valueB :
ќџџџџџџџџ2
delta/arange/startj
delta/arange/limitConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange/limitj
delta/arange/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange/delta
delta/arangeRangedelta/arange/start:output:0delta/arange/limit:output:0delta/arange/delta:output:0*
_output_shapes
:	2
delta/arangek

delta/CastCastdelta/arange:output:0*

DstT0*

SrcT0*
_output_shapes
:	2

delta/Cast
delta/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
delta/Reshape/shape
delta/ReshapeReshapedelta/Cast:y:0delta/Reshape/shape:output:0*
T0*&
_output_shapes
:	2
delta/ReshapeХ
delta/convolutionConv2Ddelta/MirrorPad:output:0delta/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
delta/convolutiong
delta/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pB2
delta/truediv/y
delta/truedivRealDivdelta/convolution:output:0delta/truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/truediv
delta/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose_1/permЁ
delta/transpose_1	Transposedelta/truediv:z:0delta/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose_1
delta/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose_2/permЅ
delta/transpose_2	Transposedelta/transpose_1:y:0delta/transpose_2/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose_2
delta/Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                               2
delta/Const_1Б
delta/MirrorPad_1	MirrorPaddelta/transpose_2:y:0delta/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
mode	SYMMETRIC2
delta/MirrorPad_1w
delta/arange_1/startConst*
_output_shapes
: *
dtype0*
valueB :
ќџџџџџџџџ2
delta/arange_1/startn
delta/arange_1/limitConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange_1/limitn
delta/arange_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
delta/arange_1/deltaЂ
delta/arange_1Rangedelta/arange_1/start:output:0delta/arange_1/limit:output:0delta/arange_1/delta:output:0*
_output_shapes
:	2
delta/arange_1q
delta/Cast_1Castdelta/arange_1:output:0*

DstT0*

SrcT0*
_output_shapes
:	2
delta/Cast_1
delta/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
delta/Reshape_1/shape
delta/Reshape_1Reshapedelta/Cast_1:y:0delta/Reshape_1/shape:output:0*
T0*&
_output_shapes
:	2
delta/Reshape_1Э
delta/convolution_1Conv2Ddelta/MirrorPad_1:output:0delta/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
delta/convolution_1k
delta/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pB2
delta/truediv_1/yЁ
delta/truediv_1RealDivdelta/convolution_1:output:0delta/truediv_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/truediv_1
delta/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
delta/transpose_3/permЃ
delta/transpose_3	Transposedelta/truediv_1:z:0delta/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
delta/transpose_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisи
concatConcatV2)mfccs_from_log_mel_spectrograms/mul_1:z:0delta/transpose_1:y:0delta/transpose_3:y:0Sqrt:y:0concat/axis:output:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@2
concat
	Squeeze_1Squeezeconcat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@*
squeeze_dims
2
	Squeeze_1
(resnet_block/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(resnet_block/conv1/conv1d/ExpandDims/dimл
$resnet_block/conv1/conv1d/ExpandDims
ExpandDimsSqueeze_1:output:01resnet_block/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2&
$resnet_block/conv1/conv1d/ExpandDimsё
5resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>resnet_block_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOp
*resnet_block/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*resnet_block/conv1/conv1d/ExpandDims_1/dim
&resnet_block/conv1/conv1d/ExpandDims_1
ExpandDims=resnet_block/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:03resnet_block/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2(
&resnet_block/conv1/conv1d/ExpandDims_1
resnet_block/conv1/conv1dConv2D-resnet_block/conv1/conv1d/ExpandDims:output:0/resnet_block/conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/conv1/conv1dЫ
!resnet_block/conv1/conv1d/SqueezeSqueeze"resnet_block/conv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2#
!resnet_block/conv1/conv1d/SqueezeХ
)resnet_block/conv1/BiasAdd/ReadVariableOpReadVariableOp2resnet_block_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)resnet_block/conv1/BiasAdd/ReadVariableOpи
resnet_block/conv1/BiasAddBiasAdd*resnet_block/conv1/conv1d/Squeeze:output:01resnet_block/conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/conv1/BiasAdd
resnet_block/relu1/ReluRelu#resnet_block/conv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/relu1/Relu
(resnet_block/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(resnet_block/conv2/conv1d/ExpandDims/dimю
$resnet_block/conv2/conv1d/ExpandDims
ExpandDims%resnet_block/relu1/Relu:activations:01resnet_block/conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2&
$resnet_block/conv2/conv1d/ExpandDimsё
5resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>resnet_block_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype027
5resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOp
*resnet_block/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*resnet_block/conv2/conv1d/ExpandDims_1/dim
&resnet_block/conv2/conv1d/ExpandDims_1
ExpandDims=resnet_block/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:03resnet_block/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2(
&resnet_block/conv2/conv1d/ExpandDims_1
resnet_block/conv2/conv1dConv2D-resnet_block/conv2/conv1d/ExpandDims:output:0/resnet_block/conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/conv2/conv1dЫ
!resnet_block/conv2/conv1d/SqueezeSqueeze"resnet_block/conv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2#
!resnet_block/conv2/conv1d/SqueezeХ
)resnet_block/conv2/BiasAdd/ReadVariableOpReadVariableOp2resnet_block_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)resnet_block/conv2/BiasAdd/ReadVariableOpи
resnet_block/conv2/BiasAddBiasAdd*resnet_block/conv2/conv1d/Squeeze:output:01resnet_block/conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/conv2/BiasAdd
resnet_block/relu2/ReluRelu#resnet_block/conv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/relu2/Relu
(resnet_block/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(resnet_block/conv3/conv1d/ExpandDims/dimю
$resnet_block/conv3/conv1d/ExpandDims
ExpandDims%resnet_block/relu2/Relu:activations:01resnet_block/conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2&
$resnet_block/conv3/conv1d/ExpandDimsё
5resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>resnet_block_conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype027
5resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOp
*resnet_block/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*resnet_block/conv3/conv1d/ExpandDims_1/dim
&resnet_block/conv3/conv1d/ExpandDims_1
ExpandDims=resnet_block/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:03resnet_block/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2(
&resnet_block/conv3/conv1d/ExpandDims_1
resnet_block/conv3/conv1dConv2D-resnet_block/conv3/conv1d/ExpandDims:output:0/resnet_block/conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/conv3/conv1dЫ
!resnet_block/conv3/conv1d/SqueezeSqueeze"resnet_block/conv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2#
!resnet_block/conv3/conv1d/SqueezeХ
)resnet_block/conv3/BiasAdd/ReadVariableOpReadVariableOp2resnet_block_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)resnet_block/conv3/BiasAdd/ReadVariableOpи
resnet_block/conv3/BiasAddBiasAdd*resnet_block/conv3/conv1d/Squeeze:output:01resnet_block/conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/conv3/BiasAddЅ
+resnet_block/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2-
+resnet_block/shortcut/conv1d/ExpandDims/dimф
'resnet_block/shortcut/conv1d/ExpandDims
ExpandDimsSqueeze_1:output:04resnet_block/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2)
'resnet_block/shortcut/conv1d/ExpandDimsњ
8resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAresnet_block_shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02:
8resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOp 
-resnet_block/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-resnet_block/shortcut/conv1d/ExpandDims_1/dim
)resnet_block/shortcut/conv1d/ExpandDims_1
ExpandDims@resnet_block/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:06resnet_block/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2+
)resnet_block/shortcut/conv1d/ExpandDims_1
resnet_block/shortcut/conv1dConv2D0resnet_block/shortcut/conv1d/ExpandDims:output:02resnet_block/shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
resnet_block/shortcut/conv1dд
$resnet_block/shortcut/conv1d/SqueezeSqueeze%resnet_block/shortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2&
$resnet_block/shortcut/conv1d/SqueezeЮ
,resnet_block/shortcut/BiasAdd/ReadVariableOpReadVariableOp5resnet_block_shortcut_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,resnet_block/shortcut/BiasAdd/ReadVariableOpф
resnet_block/shortcut/BiasAddBiasAdd-resnet_block/shortcut/conv1d/Squeeze:output:04resnet_block/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/shortcut/BiasAddИ
resnet_block/add/addAddV2#resnet_block/conv3/BiasAdd:output:0&resnet_block/shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/add/add
resnet_block/out_block/ReluReluresnet_block/add/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
resnet_block/out_block/ReluЃ
*resnet_block_1/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_1/conv1/conv1d/ExpandDims/dimј
&resnet_block_1/conv1/conv1d/ExpandDims
ExpandDims)resnet_block/out_block/Relu:activations:03resnet_block_1/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2(
&resnet_block_1/conv1/conv1d/ExpandDimsї
7resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_1_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype029
7resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_1/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_1/conv1/conv1d/ExpandDims_1/dim
(resnet_block_1/conv1/conv1d/ExpandDims_1
ExpandDims?resnet_block_1/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_1/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2*
(resnet_block_1/conv1/conv1d/ExpandDims_1
resnet_block_1/conv1/conv1dConv2D/resnet_block_1/conv1/conv1d/ExpandDims:output:01resnet_block_1/conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
resnet_block_1/conv1/conv1dб
#resnet_block_1/conv1/conv1d/SqueezeSqueeze$resnet_block_1/conv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_1/conv1/conv1d/SqueezeЫ
+resnet_block_1/conv1/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+resnet_block_1/conv1/BiasAdd/ReadVariableOpр
resnet_block_1/conv1/BiasAddBiasAdd,resnet_block_1/conv1/conv1d/Squeeze:output:03resnet_block_1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/conv1/BiasAdd
resnet_block_1/relu1/ReluRelu%resnet_block_1/conv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/relu1/ReluЃ
*resnet_block_1/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_1/conv2/conv1d/ExpandDims/dimі
&resnet_block_1/conv2/conv1d/ExpandDims
ExpandDims'resnet_block_1/relu1/Relu:activations:03resnet_block_1/conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2(
&resnet_block_1/conv2/conv1d/ExpandDimsї
7resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_1_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_1/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_1/conv2/conv1d/ExpandDims_1/dim
(resnet_block_1/conv2/conv1d/ExpandDims_1
ExpandDims?resnet_block_1/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_1/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2*
(resnet_block_1/conv2/conv1d/ExpandDims_1
resnet_block_1/conv2/conv1dConv2D/resnet_block_1/conv2/conv1d/ExpandDims:output:01resnet_block_1/conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
resnet_block_1/conv2/conv1dб
#resnet_block_1/conv2/conv1d/SqueezeSqueeze$resnet_block_1/conv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_1/conv2/conv1d/SqueezeЫ
+resnet_block_1/conv2/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+resnet_block_1/conv2/BiasAdd/ReadVariableOpр
resnet_block_1/conv2/BiasAddBiasAdd,resnet_block_1/conv2/conv1d/Squeeze:output:03resnet_block_1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/conv2/BiasAdd
resnet_block_1/relu2/ReluRelu%resnet_block_1/conv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/relu2/ReluЃ
*resnet_block_1/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_1/conv3/conv1d/ExpandDims/dimі
&resnet_block_1/conv3/conv1d/ExpandDims
ExpandDims'resnet_block_1/relu2/Relu:activations:03resnet_block_1/conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2(
&resnet_block_1/conv3/conv1d/ExpandDimsї
7resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_1_conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_1/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_1/conv3/conv1d/ExpandDims_1/dim
(resnet_block_1/conv3/conv1d/ExpandDims_1
ExpandDims?resnet_block_1/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_1/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2*
(resnet_block_1/conv3/conv1d/ExpandDims_1
resnet_block_1/conv3/conv1dConv2D/resnet_block_1/conv3/conv1d/ExpandDims:output:01resnet_block_1/conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
resnet_block_1/conv3/conv1dб
#resnet_block_1/conv3/conv1d/SqueezeSqueeze$resnet_block_1/conv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_1/conv3/conv1d/SqueezeЫ
+resnet_block_1/conv3/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_1_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+resnet_block_1/conv3/BiasAdd/ReadVariableOpр
resnet_block_1/conv3/BiasAddBiasAdd,resnet_block_1/conv3/conv1d/Squeeze:output:03resnet_block_1/conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/conv3/BiasAddЉ
-resnet_block_1/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-resnet_block_1/shortcut/conv1d/ExpandDims/dim
)resnet_block_1/shortcut/conv1d/ExpandDims
ExpandDims)resnet_block/out_block/Relu:activations:06resnet_block_1/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2+
)resnet_block_1/shortcut/conv1d/ExpandDims
:resnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCresnet_block_1_shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02<
:resnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOpЄ
/resnet_block_1/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/resnet_block_1/shortcut/conv1d/ExpandDims_1/dim
+resnet_block_1/shortcut/conv1d/ExpandDims_1
ExpandDimsBresnet_block_1/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:08resnet_block_1/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2-
+resnet_block_1/shortcut/conv1d/ExpandDims_1
resnet_block_1/shortcut/conv1dConv2D2resnet_block_1/shortcut/conv1d/ExpandDims:output:04resnet_block_1/shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2 
resnet_block_1/shortcut/conv1dк
&resnet_block_1/shortcut/conv1d/SqueezeSqueeze'resnet_block_1/shortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@*
squeeze_dims

§џџџџџџџџ2(
&resnet_block_1/shortcut/conv1d/Squeezeд
.resnet_block_1/shortcut/BiasAdd/ReadVariableOpReadVariableOp7resnet_block_1_shortcut_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.resnet_block_1/shortcut/BiasAdd/ReadVariableOpь
resnet_block_1/shortcut/BiasAddBiasAdd/resnet_block_1/shortcut/conv1d/Squeeze:output:06resnet_block_1/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2!
resnet_block_1/shortcut/BiasAddФ
resnet_block_1/add_1/addAddV2%resnet_block_1/conv3/BiasAdd:output:0(resnet_block_1/shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/add_1/add
resnet_block_1/out_block/ReluReluresnet_block_1/add_1/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2
resnet_block_1/out_block/ReluЃ
*resnet_block_2/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_2/conv1/conv1d/ExpandDims/dimњ
&resnet_block_2/conv1/conv1d/ExpandDims
ExpandDims+resnet_block_1/out_block/Relu:activations:03resnet_block_2/conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2(
&resnet_block_2/conv1/conv1d/ExpandDimsј
7resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_2_conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype029
7resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_2/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_2/conv1/conv1d/ExpandDims_1/dim
(resnet_block_2/conv1/conv1d/ExpandDims_1
ExpandDims?resnet_block_2/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_2/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2*
(resnet_block_2/conv1/conv1d/ExpandDims_1
resnet_block_2/conv1/conv1dConv2D/resnet_block_2/conv1/conv1d/ExpandDims:output:01resnet_block_2/conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_2/conv1/conv1dв
#resnet_block_2/conv1/conv1d/SqueezeSqueeze$resnet_block_2/conv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_2/conv1/conv1d/SqueezeЬ
+resnet_block_2/conv1/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_2/conv1/BiasAdd/ReadVariableOpс
resnet_block_2/conv1/BiasAddBiasAdd,resnet_block_2/conv1/conv1d/Squeeze:output:03resnet_block_2/conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/conv1/BiasAdd
resnet_block_2/relu1/ReluRelu%resnet_block_2/conv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/relu1/ReluЃ
*resnet_block_2/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_2/conv2/conv1d/ExpandDims/dimї
&resnet_block_2/conv2/conv1d/ExpandDims
ExpandDims'resnet_block_2/relu1/Relu:activations:03resnet_block_2/conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_2/conv2/conv1d/ExpandDimsљ
7resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_2_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_2/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_2/conv2/conv1d/ExpandDims_1/dim
(resnet_block_2/conv2/conv1d/ExpandDims_1
ExpandDims?resnet_block_2/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_2/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_2/conv2/conv1d/ExpandDims_1
resnet_block_2/conv2/conv1dConv2D/resnet_block_2/conv2/conv1d/ExpandDims:output:01resnet_block_2/conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_2/conv2/conv1dв
#resnet_block_2/conv2/conv1d/SqueezeSqueeze$resnet_block_2/conv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_2/conv2/conv1d/SqueezeЬ
+resnet_block_2/conv2/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_2/conv2/BiasAdd/ReadVariableOpс
resnet_block_2/conv2/BiasAddBiasAdd,resnet_block_2/conv2/conv1d/Squeeze:output:03resnet_block_2/conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/conv2/BiasAdd
resnet_block_2/relu2/ReluRelu%resnet_block_2/conv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/relu2/ReluЃ
*resnet_block_2/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_2/conv3/conv1d/ExpandDims/dimї
&resnet_block_2/conv3/conv1d/ExpandDims
ExpandDims'resnet_block_2/relu2/Relu:activations:03resnet_block_2/conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_2/conv3/conv1d/ExpandDimsљ
7resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_2_conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_2/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_2/conv3/conv1d/ExpandDims_1/dim
(resnet_block_2/conv3/conv1d/ExpandDims_1
ExpandDims?resnet_block_2/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_2/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_2/conv3/conv1d/ExpandDims_1
resnet_block_2/conv3/conv1dConv2D/resnet_block_2/conv3/conv1d/ExpandDims:output:01resnet_block_2/conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_2/conv3/conv1dв
#resnet_block_2/conv3/conv1d/SqueezeSqueeze$resnet_block_2/conv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_2/conv3/conv1d/SqueezeЬ
+resnet_block_2/conv3/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_2_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_2/conv3/BiasAdd/ReadVariableOpс
resnet_block_2/conv3/BiasAddBiasAdd,resnet_block_2/conv3/conv1d/Squeeze:output:03resnet_block_2/conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/conv3/BiasAddЉ
-resnet_block_2/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-resnet_block_2/shortcut/conv1d/ExpandDims/dim
)resnet_block_2/shortcut/conv1d/ExpandDims
ExpandDims+resnet_block_1/out_block/Relu:activations:06resnet_block_2/shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2+
)resnet_block_2/shortcut/conv1d/ExpandDims
:resnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCresnet_block_2_shortcut_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02<
:resnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOpЄ
/resnet_block_2/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/resnet_block_2/shortcut/conv1d/ExpandDims_1/dim
+resnet_block_2/shortcut/conv1d/ExpandDims_1
ExpandDimsBresnet_block_2/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:08resnet_block_2/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2-
+resnet_block_2/shortcut/conv1d/ExpandDims_1
resnet_block_2/shortcut/conv1dConv2D2resnet_block_2/shortcut/conv1d/ExpandDims:output:04resnet_block_2/shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2 
resnet_block_2/shortcut/conv1dл
&resnet_block_2/shortcut/conv1d/SqueezeSqueeze'resnet_block_2/shortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2(
&resnet_block_2/shortcut/conv1d/Squeezeе
.resnet_block_2/shortcut/BiasAdd/ReadVariableOpReadVariableOp7resnet_block_2_shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.resnet_block_2/shortcut/BiasAdd/ReadVariableOpэ
resnet_block_2/shortcut/BiasAddBiasAdd/resnet_block_2/shortcut/conv1d/Squeeze:output:06resnet_block_2/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2!
resnet_block_2/shortcut/BiasAddХ
resnet_block_2/add_2/addAddV2%resnet_block_2/conv3/BiasAdd:output:0(resnet_block_2/shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/add_2/add
resnet_block_2/out_block/ReluReluresnet_block_2/add_2/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_2/out_block/ReluЃ
*resnet_block_3/conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_3/conv1/conv1d/ExpandDims/dimћ
&resnet_block_3/conv1/conv1d/ExpandDims
ExpandDims+resnet_block_2/out_block/Relu:activations:03resnet_block_3/conv1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_3/conv1/conv1d/ExpandDimsљ
7resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_3_conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_3/conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_3/conv1/conv1d/ExpandDims_1/dim
(resnet_block_3/conv1/conv1d/ExpandDims_1
ExpandDims?resnet_block_3/conv1/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_3/conv1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_3/conv1/conv1d/ExpandDims_1
resnet_block_3/conv1/conv1dConv2D/resnet_block_3/conv1/conv1d/ExpandDims:output:01resnet_block_3/conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_3/conv1/conv1dв
#resnet_block_3/conv1/conv1d/SqueezeSqueeze$resnet_block_3/conv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_3/conv1/conv1d/SqueezeЬ
+resnet_block_3/conv1/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_3/conv1/BiasAdd/ReadVariableOpс
resnet_block_3/conv1/BiasAddBiasAdd,resnet_block_3/conv1/conv1d/Squeeze:output:03resnet_block_3/conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/conv1/BiasAdd
resnet_block_3/relu1/ReluRelu%resnet_block_3/conv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/relu1/ReluЃ
*resnet_block_3/conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_3/conv2/conv1d/ExpandDims/dimї
&resnet_block_3/conv2/conv1d/ExpandDims
ExpandDims'resnet_block_3/relu1/Relu:activations:03resnet_block_3/conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_3/conv2/conv1d/ExpandDimsљ
7resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_3_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_3/conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_3/conv2/conv1d/ExpandDims_1/dim
(resnet_block_3/conv2/conv1d/ExpandDims_1
ExpandDims?resnet_block_3/conv2/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_3/conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_3/conv2/conv1d/ExpandDims_1
resnet_block_3/conv2/conv1dConv2D/resnet_block_3/conv2/conv1d/ExpandDims:output:01resnet_block_3/conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_3/conv2/conv1dв
#resnet_block_3/conv2/conv1d/SqueezeSqueeze$resnet_block_3/conv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_3/conv2/conv1d/SqueezeЬ
+resnet_block_3/conv2/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_3/conv2/BiasAdd/ReadVariableOpс
resnet_block_3/conv2/BiasAddBiasAdd,resnet_block_3/conv2/conv1d/Squeeze:output:03resnet_block_3/conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/conv2/BiasAdd
resnet_block_3/relu2/ReluRelu%resnet_block_3/conv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/relu2/ReluЃ
*resnet_block_3/conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2,
*resnet_block_3/conv3/conv1d/ExpandDims/dimї
&resnet_block_3/conv3/conv1d/ExpandDims
ExpandDims'resnet_block_3/relu2/Relu:activations:03resnet_block_3/conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2(
&resnet_block_3/conv3/conv1d/ExpandDimsљ
7resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@resnet_block_3_conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype029
7resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOp
,resnet_block_3/conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,resnet_block_3/conv3/conv1d/ExpandDims_1/dim
(resnet_block_3/conv3/conv1d/ExpandDims_1
ExpandDims?resnet_block_3/conv3/conv1d/ExpandDims_1/ReadVariableOp:value:05resnet_block_3/conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2*
(resnet_block_3/conv3/conv1d/ExpandDims_1
resnet_block_3/conv3/conv1dConv2D/resnet_block_3/conv3/conv1d/ExpandDims:output:01resnet_block_3/conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
resnet_block_3/conv3/conv1dв
#resnet_block_3/conv3/conv1d/SqueezeSqueeze$resnet_block_3/conv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2%
#resnet_block_3/conv3/conv1d/SqueezeЬ
+resnet_block_3/conv3/BiasAdd/ReadVariableOpReadVariableOp4resnet_block_3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+resnet_block_3/conv3/BiasAdd/ReadVariableOpс
resnet_block_3/conv3/BiasAddBiasAdd,resnet_block_3/conv3/conv1d/Squeeze:output:03resnet_block_3/conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/conv3/BiasAddЉ
-resnet_block_3/shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-resnet_block_3/shortcut/conv1d/ExpandDims/dim
)resnet_block_3/shortcut/conv1d/ExpandDims
ExpandDims+resnet_block_2/out_block/Relu:activations:06resnet_block_3/shortcut/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2+
)resnet_block_3/shortcut/conv1d/ExpandDims
:resnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCresnet_block_3_shortcut_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02<
:resnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOpЄ
/resnet_block_3/shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/resnet_block_3/shortcut/conv1d/ExpandDims_1/dim
+resnet_block_3/shortcut/conv1d/ExpandDims_1
ExpandDimsBresnet_block_3/shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:08resnet_block_3/shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2-
+resnet_block_3/shortcut/conv1d/ExpandDims_1
resnet_block_3/shortcut/conv1dConv2D2resnet_block_3/shortcut/conv1d/ExpandDims:output:04resnet_block_3/shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2 
resnet_block_3/shortcut/conv1dл
&resnet_block_3/shortcut/conv1d/SqueezeSqueeze'resnet_block_3/shortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2(
&resnet_block_3/shortcut/conv1d/Squeezeе
.resnet_block_3/shortcut/BiasAdd/ReadVariableOpReadVariableOp7resnet_block_3_shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.resnet_block_3/shortcut/BiasAdd/ReadVariableOpэ
resnet_block_3/shortcut/BiasAddBiasAdd/resnet_block_3/shortcut/conv1d/Squeeze:output:06resnet_block_3/shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2!
resnet_block_3/shortcut/BiasAddХ
resnet_block_3/add_3/addAddV2%resnet_block_3/conv3/BiasAdd:output:0(resnet_block_3/shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/add_3/add
resnet_block_3/out_block/ReluReluresnet_block_3/add_3/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
resnet_block_3/out_block/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
flatten/ConstЅ
flatten/ReshapeReshape+resnet_block_3/out_block/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ@2
flatten/Reshape
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype02
fc1/MatMul/ReadVariableOp

fc1/MatMulMatMulflatten/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc1/MatMul
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc1/BiasAdd/ReadVariableOp
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
fc1/BiasAdde
fc1/ReluRelufc1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc1/Relu
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
fc2/MatMul/ReadVariableOp

fc2/MatMulMatMulfc1/Relu:activations:0!fc2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc2/MatMul
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc2/BiasAdd/ReadVariableOp
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
fc2/BiasAdde
fc2/ReluRelufc2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

fc2/Relu
fc3/MatMul/ReadVariableOpReadVariableOp"fc3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
fc3/MatMul/ReadVariableOp

fc3/MatMulMatMulfc2/Relu:activations:0!fc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2

fc3/MatMul
fc3/BiasAdd/ReadVariableOpReadVariableOp#fc3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc3/BiasAdd/ReadVariableOp
fc3/BiasAddBiasAddfc3/MatMul:product:0"fc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fc3/BiasAddm
fc3/SigmoidSigmoidfc3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fc3/Sigmoidc
IdentityIdentityfc3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:џџџџџџџџџ:	:::::::::::::::::::::::::::::::::::::::T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:%!

_output_shapes
:	
п
Г
"__inference__traced_restore_218152
file_prefix'
#assignvariableop_res_net_fc1_kernel'
#assignvariableop_1_res_net_fc1_bias)
%assignvariableop_2_res_net_fc2_kernel'
#assignvariableop_3_res_net_fc2_bias)
%assignvariableop_4_res_net_fc3_kernel'
#assignvariableop_5_res_net_fc3_bias
assignvariableop_6_sgd_iter 
assignvariableop_7_sgd_decay(
$assignvariableop_8_sgd_learning_rate#
assignvariableop_9_sgd_momentum9
5assignvariableop_10_res_net_resnet_block_conv1_kernel7
3assignvariableop_11_res_net_resnet_block_conv1_bias9
5assignvariableop_12_res_net_resnet_block_conv2_kernel7
3assignvariableop_13_res_net_resnet_block_conv2_bias9
5assignvariableop_14_res_net_resnet_block_conv3_kernel7
3assignvariableop_15_res_net_resnet_block_conv3_bias<
8assignvariableop_16_res_net_resnet_block_shortcut_kernel:
6assignvariableop_17_res_net_resnet_block_shortcut_bias;
7assignvariableop_18_res_net_resnet_block_1_conv1_kernel9
5assignvariableop_19_res_net_resnet_block_1_conv1_bias;
7assignvariableop_20_res_net_resnet_block_1_conv2_kernel9
5assignvariableop_21_res_net_resnet_block_1_conv2_bias;
7assignvariableop_22_res_net_resnet_block_1_conv3_kernel9
5assignvariableop_23_res_net_resnet_block_1_conv3_bias>
:assignvariableop_24_res_net_resnet_block_1_shortcut_kernel<
8assignvariableop_25_res_net_resnet_block_1_shortcut_bias;
7assignvariableop_26_res_net_resnet_block_2_conv1_kernel9
5assignvariableop_27_res_net_resnet_block_2_conv1_bias;
7assignvariableop_28_res_net_resnet_block_2_conv2_kernel9
5assignvariableop_29_res_net_resnet_block_2_conv2_bias;
7assignvariableop_30_res_net_resnet_block_2_conv3_kernel9
5assignvariableop_31_res_net_resnet_block_2_conv3_bias>
:assignvariableop_32_res_net_resnet_block_2_shortcut_kernel<
8assignvariableop_33_res_net_resnet_block_2_shortcut_bias;
7assignvariableop_34_res_net_resnet_block_3_conv1_kernel9
5assignvariableop_35_res_net_resnet_block_3_conv1_bias;
7assignvariableop_36_res_net_resnet_block_3_conv2_kernel9
5assignvariableop_37_res_net_resnet_block_3_conv2_bias;
7assignvariableop_38_res_net_resnet_block_3_conv3_kernel9
5assignvariableop_39_res_net_resnet_block_3_conv3_bias>
:assignvariableop_40_res_net_resnet_block_3_shortcut_kernel<
8assignvariableop_41_res_net_resnet_block_3_shortcut_bias
assignvariableop_42_total
assignvariableop_43_count
assignvariableop_44_total_1
assignvariableop_45_count_1&
"assignvariableop_46_true_positives'
#assignvariableop_47_false_positives(
$assignvariableop_48_true_positives_1'
#assignvariableop_49_false_negatives#
assignvariableop_50_accumulator%
!assignvariableop_51_accumulator_1%
!assignvariableop_52_accumulator_2%
!assignvariableop_53_accumulator_3
identity_55ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ё
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*­
valueЃB 7B%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc3/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/5/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/6/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/7/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names§
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesС
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ђ
_output_shapesп
м:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЂ
AssignVariableOpAssignVariableOp#assignvariableop_res_net_fc1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ј
AssignVariableOp_1AssignVariableOp#assignvariableop_1_res_net_fc1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Њ
AssignVariableOp_2AssignVariableOp%assignvariableop_2_res_net_fc2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ј
AssignVariableOp_3AssignVariableOp#assignvariableop_3_res_net_fc2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Њ
AssignVariableOp_4AssignVariableOp%assignvariableop_4_res_net_fc3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ј
AssignVariableOp_5AssignVariableOp#assignvariableop_5_res_net_fc3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6 
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ё
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Љ
AssignVariableOp_8AssignVariableOp$assignvariableop_8_sgd_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Є
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Н
AssignVariableOp_10AssignVariableOp5assignvariableop_10_res_net_resnet_block_conv1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Л
AssignVariableOp_11AssignVariableOp3assignvariableop_11_res_net_resnet_block_conv1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Н
AssignVariableOp_12AssignVariableOp5assignvariableop_12_res_net_resnet_block_conv2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Л
AssignVariableOp_13AssignVariableOp3assignvariableop_13_res_net_resnet_block_conv2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Н
AssignVariableOp_14AssignVariableOp5assignvariableop_14_res_net_resnet_block_conv3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Л
AssignVariableOp_15AssignVariableOp3assignvariableop_15_res_net_resnet_block_conv3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Р
AssignVariableOp_16AssignVariableOp8assignvariableop_16_res_net_resnet_block_shortcut_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17О
AssignVariableOp_17AssignVariableOp6assignvariableop_17_res_net_resnet_block_shortcut_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18П
AssignVariableOp_18AssignVariableOp7assignvariableop_18_res_net_resnet_block_1_conv1_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Н
AssignVariableOp_19AssignVariableOp5assignvariableop_19_res_net_resnet_block_1_conv1_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20П
AssignVariableOp_20AssignVariableOp7assignvariableop_20_res_net_resnet_block_1_conv2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Н
AssignVariableOp_21AssignVariableOp5assignvariableop_21_res_net_resnet_block_1_conv2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22П
AssignVariableOp_22AssignVariableOp7assignvariableop_22_res_net_resnet_block_1_conv3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Н
AssignVariableOp_23AssignVariableOp5assignvariableop_23_res_net_resnet_block_1_conv3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Т
AssignVariableOp_24AssignVariableOp:assignvariableop_24_res_net_resnet_block_1_shortcut_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Р
AssignVariableOp_25AssignVariableOp8assignvariableop_25_res_net_resnet_block_1_shortcut_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26П
AssignVariableOp_26AssignVariableOp7assignvariableop_26_res_net_resnet_block_2_conv1_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Н
AssignVariableOp_27AssignVariableOp5assignvariableop_27_res_net_resnet_block_2_conv1_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28П
AssignVariableOp_28AssignVariableOp7assignvariableop_28_res_net_resnet_block_2_conv2_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Н
AssignVariableOp_29AssignVariableOp5assignvariableop_29_res_net_resnet_block_2_conv2_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30П
AssignVariableOp_30AssignVariableOp7assignvariableop_30_res_net_resnet_block_2_conv3_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Н
AssignVariableOp_31AssignVariableOp5assignvariableop_31_res_net_resnet_block_2_conv3_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Т
AssignVariableOp_32AssignVariableOp:assignvariableop_32_res_net_resnet_block_2_shortcut_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Р
AssignVariableOp_33AssignVariableOp8assignvariableop_33_res_net_resnet_block_2_shortcut_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34П
AssignVariableOp_34AssignVariableOp7assignvariableop_34_res_net_resnet_block_3_conv1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Н
AssignVariableOp_35AssignVariableOp5assignvariableop_35_res_net_resnet_block_3_conv1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36П
AssignVariableOp_36AssignVariableOp7assignvariableop_36_res_net_resnet_block_3_conv2_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Н
AssignVariableOp_37AssignVariableOp5assignvariableop_37_res_net_resnet_block_3_conv2_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38П
AssignVariableOp_38AssignVariableOp7assignvariableop_38_res_net_resnet_block_3_conv3_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Н
AssignVariableOp_39AssignVariableOp5assignvariableop_39_res_net_resnet_block_3_conv3_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Т
AssignVariableOp_40AssignVariableOp:assignvariableop_40_res_net_resnet_block_3_shortcut_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Р
AssignVariableOp_41AssignVariableOp8assignvariableop_41_res_net_resnet_block_3_shortcut_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ё
AssignVariableOp_42AssignVariableOpassignvariableop_42_totalIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ё
AssignVariableOp_43AssignVariableOpassignvariableop_43_countIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Ѓ
AssignVariableOp_44AssignVariableOpassignvariableop_44_total_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Ѓ
AssignVariableOp_45AssignVariableOpassignvariableop_45_count_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Њ
AssignVariableOp_46AssignVariableOp"assignvariableop_46_true_positivesIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ћ
AssignVariableOp_47AssignVariableOp#assignvariableop_47_false_positivesIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ќ
AssignVariableOp_48AssignVariableOp$assignvariableop_48_true_positives_1Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ћ
AssignVariableOp_49AssignVariableOp#assignvariableop_49_false_negativesIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ї
AssignVariableOp_50AssignVariableOpassignvariableop_50_accumulatorIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Љ
AssignVariableOp_51AssignVariableOp!assignvariableop_51_accumulator_1Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Љ
AssignVariableOp_52AssignVariableOp!assignvariableop_52_accumulator_2Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Љ
AssignVariableOp_53AssignVariableOp!assignvariableop_53_accumulator_3Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_539
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_54ѕ	
Identity_55IdentityIdentity_54:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_55"#
identity_55Identity_55:output:0*я
_input_shapesн
к: ::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
У
]
A__inference_relu1_layer_call_and_return_conditional_losses_211571

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:џџџџџџџџџ@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ь
~
)__inference_shortcut_layer_call_fn_217406

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_shortcut_layer_call_and_return_conditional_losses_2103342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ч
a
E__inference_out_block_layer_call_and_return_conditional_losses_217663

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:џџџџџџџџџ@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ъ
{
&__inference_conv2_layer_call_fn_217600

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2111432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
В
Ў

C__inference_res_net_layer_call_and_return_conditional_losses_213253

inputs
log_mel_spectrogram_213119
resnet_block_213168
resnet_block_213170
resnet_block_213172
resnet_block_213174
resnet_block_213176
resnet_block_213178
resnet_block_213180
resnet_block_213182
resnet_block_1_213185
resnet_block_1_213187
resnet_block_1_213189
resnet_block_1_213191
resnet_block_1_213193
resnet_block_1_213195
resnet_block_1_213197
resnet_block_1_213199
resnet_block_2_213202
resnet_block_2_213204
resnet_block_2_213206
resnet_block_2_213208
resnet_block_2_213210
resnet_block_2_213212
resnet_block_2_213214
resnet_block_2_213216
resnet_block_3_213219
resnet_block_3_213221
resnet_block_3_213223
resnet_block_3_213225
resnet_block_3_213227
resnet_block_3_213229
resnet_block_3_213231
resnet_block_3_213233

fc1_213237

fc1_213239

fc2_213242

fc2_213244

fc3_213247

fc3_213249
identityЂfc1/StatefulPartitionedCallЂfc2/StatefulPartitionedCallЂfc3/StatefulPartitionedCallЂ$resnet_block/StatefulPartitionedCallЂ&resnet_block_1/StatefulPartitionedCallЂ&resnet_block_2/StatefulPartitionedCallЂ&resnet_block_3/StatefulPartitionedCallo
SqueezeSqueezeinputs*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2	
SqueezeЄ
#log_mel_spectrogram/PartitionedCallPartitionedCallSqueeze:output:0log_mel_spectrogram_213119*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_log_mel_spectrogram_layer_call_and_return_conditional_losses_2123162%
#log_mel_spectrogram/PartitionedCally
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/permЄ
	transpose	Transpose,log_mel_spectrogram/PartitionedCall:output:0transpose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
	transpose
)mfccs_from_log_mel_spectrograms/dct/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)mfccs_from_log_mel_spectrograms/dct/Const
*mfccs_from_log_mel_spectrograms/dct/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*mfccs_from_log_mel_spectrograms/dct/Cast/xС
(mfccs_from_log_mel_spectrograms/dct/CastCast3mfccs_from_log_mel_spectrograms/dct/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(mfccs_from_log_mel_spectrograms/dct/CastЄ
/mfccs_from_log_mel_spectrograms/dct/range/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/mfccs_from_log_mel_spectrograms/dct/range/startЄ
/mfccs_from_log_mel_spectrograms/dct/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/mfccs_from_log_mel_spectrograms/dct/range/deltaв
.mfccs_from_log_mel_spectrograms/dct/range/CastCast8mfccs_from_log_mel_spectrograms/dct/range/start:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.mfccs_from_log_mel_spectrograms/dct/range/Castж
0mfccs_from_log_mel_spectrograms/dct/range/Cast_1Cast8mfccs_from_log_mel_spectrograms/dct/range/delta:output:0*

DstT0*

SrcT0*
_output_shapes
: 22
0mfccs_from_log_mel_spectrograms/dct/range/Cast_1
)mfccs_from_log_mel_spectrograms/dct/rangeRange2mfccs_from_log_mel_spectrograms/dct/range/Cast:y:0,mfccs_from_log_mel_spectrograms/dct/Cast:y:04mfccs_from_log_mel_spectrograms/dct/range/Cast_1:y:0*

Tidx0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/rangeВ
'mfccs_from_log_mel_spectrograms/dct/NegNeg2mfccs_from_log_mel_spectrograms/dct/range:output:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/Neg
)mfccs_from_log_mel_spectrograms/dct/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *лI@2+
)mfccs_from_log_mel_spectrograms/dct/mul/yп
'mfccs_from_log_mel_spectrograms/dct/mulMul+mfccs_from_log_mel_spectrograms/dct/Neg:y:02mfccs_from_log_mel_spectrograms/dct/mul/y:output:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/mul
+mfccs_from_log_mel_spectrograms/dct/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2-
+mfccs_from_log_mel_spectrograms/dct/mul_1/yх
)mfccs_from_log_mel_spectrograms/dct/mul_1Mul+mfccs_from_log_mel_spectrograms/dct/mul:z:04mfccs_from_log_mel_spectrograms/dct/mul_1/y:output:0*
T0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/mul_1ч
+mfccs_from_log_mel_spectrograms/dct/truedivRealDiv-mfccs_from_log_mel_spectrograms/dct/mul_1:z:0,mfccs_from_log_mel_spectrograms/dct/Cast:y:0*
T0*
_output_shapes
:2-
+mfccs_from_log_mel_spectrograms/dct/truedivц
+mfccs_from_log_mel_spectrograms/dct/ComplexComplex2mfccs_from_log_mel_spectrograms/dct/Const:output:0/mfccs_from_log_mel_spectrograms/dct/truediv:z:0*
_output_shapes
:2-
+mfccs_from_log_mel_spectrograms/dct/ComplexБ
'mfccs_from_log_mel_spectrograms/dct/ExpExp1mfccs_from_log_mel_spectrograms/dct/Complex:out:0*
T0*
_output_shapes
:2)
'mfccs_from_log_mel_spectrograms/dct/ExpЃ
+mfccs_from_log_mel_spectrograms/dct/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J   @    2-
+mfccs_from_log_mel_spectrograms/dct/mul_2/xх
)mfccs_from_log_mel_spectrograms/dct/mul_2Mul4mfccs_from_log_mel_spectrograms/dct/mul_2/x:output:0+mfccs_from_log_mel_spectrograms/dct/Exp:y:0*
T0*
_output_shapes
:2+
)mfccs_from_log_mel_spectrograms/dct/mul_2Њ
.mfccs_from_log_mel_spectrograms/dct/rfft/ConstConst*
_output_shapes
:*
dtype0*
valueB:
20
.mfccs_from_log_mel_spectrograms/dct/rfft/Constп
5mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                27
5mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddingsь
,mfccs_from_log_mel_spectrograms/dct/rfft/PadPadtranspose:y:0>mfccs_from_log_mel_spectrograms/dct/rfft/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
2.
,mfccs_from_log_mel_spectrograms/dct/rfft/PadД
3mfccs_from_log_mel_spectrograms/dct/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:
25
3mfccs_from_log_mel_spectrograms/dct/rfft/fft_length
(mfccs_from_log_mel_spectrograms/dct/rfftRFFT5mfccs_from_log_mel_spectrograms/dct/rfft/Pad:output:0<mfccs_from_log_mel_spectrograms/dct/rfft/fft_length:output:0*/
_output_shapes
:џџџџџџџџџ@2*
(mfccs_from_log_mel_spectrograms/dct/rfftУ
7mfccs_from_log_mel_spectrograms/dct/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7mfccs_from_log_mel_spectrograms/dct/strided_slice/stackЧ
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1Ч
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9mfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2с
1mfccs_from_log_mel_spectrograms/dct/strided_sliceStridedSlice1mfccs_from_log_mel_spectrograms/dct/rfft:output:0@mfccs_from_log_mel_spectrograms/dct/strided_slice/stack:output:0Bmfccs_from_log_mel_spectrograms/dct/strided_slice/stack_1:output:0Bmfccs_from_log_mel_spectrograms/dct/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

begin_mask*
ellipsis_mask23
1mfccs_from_log_mel_spectrograms/dct/strided_slice
)mfccs_from_log_mel_spectrograms/dct/mul_3Mul:mfccs_from_log_mel_spectrograms/dct/strided_slice:output:0-mfccs_from_log_mel_spectrograms/dct/mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2+
)mfccs_from_log_mel_spectrograms/dct/mul_3М
(mfccs_from_log_mel_spectrograms/dct/RealReal-mfccs_from_log_mel_spectrograms/dct/mul_3:z:0*/
_output_shapes
:џџџџџџџџџ@2*
(mfccs_from_log_mel_spectrograms/dct/Real
&mfccs_from_log_mel_spectrograms/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2(
&mfccs_from_log_mel_spectrograms/Cast/xЕ
$mfccs_from_log_mel_spectrograms/CastCast/mfccs_from_log_mel_spectrograms/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$mfccs_from_log_mel_spectrograms/Cast
%mfccs_from_log_mel_spectrograms/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%mfccs_from_log_mel_spectrograms/mul/yЬ
#mfccs_from_log_mel_spectrograms/mulMul(mfccs_from_log_mel_spectrograms/Cast:y:0.mfccs_from_log_mel_spectrograms/mul/y:output:0*
T0*
_output_shapes
: 2%
#mfccs_from_log_mel_spectrograms/mulЁ
%mfccs_from_log_mel_spectrograms/RsqrtRsqrt'mfccs_from_log_mel_spectrograms/mul:z:0*
T0*
_output_shapes
: 2'
%mfccs_from_log_mel_spectrograms/Rsqrtэ
%mfccs_from_log_mel_spectrograms/mul_1Mul1mfccs_from_log_mel_spectrograms/dct/Real:output:0)mfccs_from_log_mel_spectrograms/Rsqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2'
%mfccs_from_log_mel_spectrograms/mul_1c
SquareSquaretranspose:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
	keep_dims(2
Sum\
SqrtSqrtSum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Sqrtі
delta/PartitionedCallPartitionedCall)mfccs_from_log_mel_spectrograms/mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_delta_layer_call_and_return_conditional_losses_2124162
delta/PartitionedCallя
delta/PartitionedCall_1PartitionedCalldelta/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_delta_layer_call_and_return_conditional_losses_2124162
delta/PartitionedCall_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisь
concatConcatV2)mfccs_from_log_mel_spectrograms/mul_1:z:0delta/PartitionedCall:output:0 delta/PartitionedCall_1:output:0Sqrt:y:0concat/axis:output:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@2
concat
	Squeeze_1Squeezeconcat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@*
squeeze_dims
2
	Squeeze_1Т
$resnet_block/StatefulPartitionedCallStatefulPartitionedCallSqueeze_1:output:0resnet_block_213168resnet_block_213170resnet_block_213172resnet_block_213174resnet_block_213176resnet_block_213178resnet_block_213180resnet_block_213182*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_resnet_block_layer_call_and_return_conditional_losses_2105412&
$resnet_block/StatefulPartitionedCallѓ
&resnet_block_1/StatefulPartitionedCallStatefulPartitionedCall-resnet_block/StatefulPartitionedCall:output:0resnet_block_1_213185resnet_block_1_213187resnet_block_1_213189resnet_block_1_213191resnet_block_1_213193resnet_block_1_213195resnet_block_1_213197resnet_block_1_213199*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_2110122(
&resnet_block_1/StatefulPartitionedCallі
&resnet_block_2/StatefulPartitionedCallStatefulPartitionedCall/resnet_block_1/StatefulPartitionedCall:output:0resnet_block_2_213202resnet_block_2_213204resnet_block_2_213206resnet_block_2_213208resnet_block_2_213210resnet_block_2_213212resnet_block_2_213214resnet_block_2_213216*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_2114832(
&resnet_block_2/StatefulPartitionedCallі
&resnet_block_3/StatefulPartitionedCallStatefulPartitionedCall/resnet_block_2/StatefulPartitionedCall:output:0resnet_block_3_213219resnet_block_3_213221resnet_block_3_213223resnet_block_3_213225resnet_block_3_213227resnet_block_3_213229resnet_block_3_213231resnet_block_3_213233*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_2119542(
&resnet_block_3/StatefulPartitionedCallћ
flatten/PartitionedCallPartitionedCall/resnet_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2126812
flatten/PartitionedCall
fc1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
fc1_213237
fc1_213239*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_2127162
fc1/StatefulPartitionedCall
fc2/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0
fc2_213242
fc2_213244*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc2_layer_call_and_return_conditional_losses_2127632
fc2/StatefulPartitionedCall
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0
fc3_213247
fc3_213249*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc3_layer_call_and_return_conditional_losses_2128102
fc3/StatefulPartitionedCallє
IdentityIdentity$fc3/StatefulPartitionedCall:output:0^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall%^resnet_block/StatefulPartitionedCall'^resnet_block_1/StatefulPartitionedCall'^resnet_block_2/StatefulPartitionedCall'^resnet_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:џџџџџџџџџ:	::::::::::::::::::::::::::::::::::::::2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall2L
$resnet_block/StatefulPartitionedCall$resnet_block/StatefulPartitionedCall2P
&resnet_block_1/StatefulPartitionedCall&resnet_block_1/StatefulPartitionedCall2P
&resnet_block_2/StatefulPartitionedCall&resnet_block_2/StatefulPartitionedCall2P
&resnet_block_3/StatefulPartitionedCall&resnet_block_3/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:%!

_output_shapes
:	
Ю;
ѕ
H__inference_resnet_block_layer_call_and_return_conditional_losses_216301

inputs5
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЈ
conv1/conv1d/ExpandDims
ExpandDimsinputs$conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv1/conv1d/ExpandDimsЪ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimЯ
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1/conv1d/ExpandDims_1Ю
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv1/conv1dЄ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1/BiasAdd/ReadVariableOpЄ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
conv1/BiasAddn

relu1/ReluReluconv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimК
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv2/conv1d/ExpandDimsЪ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimЯ
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv2/conv1d/ExpandDims_1Ю
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv2/conv1dЄ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOpЄ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
conv2/BiasAddn

relu2/ReluReluconv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimК
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv3/conv1d/ExpandDimsЪ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimЯ
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv3/conv1d/ExpandDims_1Ю
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv3/conv1dЄ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOpЄ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimБ
shortcut/conv1d/ExpandDims
ExpandDimsinputs'shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
shortcut/conv1d/ExpandDimsг
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimл
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
shortcut/conv1d/ExpandDims_1к
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
shortcut/conv1d­
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЇ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
shortcut/BiasAdd/ReadVariableOpА
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
add/addk
out_block/ReluReluadd/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
out_block/Relut
IdentityIdentityout_block/Relu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@:::::::::S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
У
]
A__inference_relu1_layer_call_and_return_conditional_losses_211100

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:џџџџџџџџџ@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
в;
і
H__inference_resnet_block_layer_call_and_return_conditional_losses_216155
input_15
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЉ
conv1/conv1d/ExpandDims
ExpandDimsinput_1$conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv1/conv1d/ExpandDimsЪ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimЯ
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1/conv1d/ExpandDims_1Ю
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv1/conv1dЄ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1/BiasAdd/ReadVariableOpЄ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
conv1/BiasAddn

relu1/ReluReluconv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimК
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv2/conv1d/ExpandDimsЪ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimЯ
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv2/conv1d/ExpandDims_1Ю
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv2/conv1dЄ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOpЄ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
conv2/BiasAddn

relu2/ReluReluconv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimК
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv3/conv1d/ExpandDimsЪ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimЯ
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv3/conv1d/ExpandDims_1Ю
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv3/conv1dЄ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOpЄ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimВ
shortcut/conv1d/ExpandDims
ExpandDimsinput_1'shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
shortcut/conv1d/ExpandDimsг
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimл
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
shortcut/conv1d/ExpandDims_1к
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
shortcut/conv1d­
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЇ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
shortcut/BiasAdd/ReadVariableOpА
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
add/addk
out_block/ReluReluadd/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
out_block/Relut
IdentityIdentityout_block/Relu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@:::::::::T P
+
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_1
Е
м
-__inference_resnet_block_layer_call_fn_216322

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_resnet_block_layer_call_and_return_conditional_losses_2104712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Т
Ж
A__inference_conv1_layer_call_and_return_conditional_losses_211536

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsК
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЙ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@:::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Н
о
/__inference_resnet_block_3_layer_call_fn_217219

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_2119542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:џџџџџџџџџ@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ї;
ј
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_216687
input_15
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЉ
conv1/conv1d/ExpandDims
ExpandDimsinput_1$conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv1/conv1d/ExpandDimsЫ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimа
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2
conv1/conv1d/ExpandDims_1Я
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1/conv1dЅ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv1/BiasAdd/ReadVariableOpЅ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv1/BiasAddo

relu1/ReluReluconv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimЛ
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv2/conv1d/ExpandDimsЬ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimб
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv2/conv1d/ExpandDims_1Я
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2/conv1dЅ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2/BiasAdd/ReadVariableOpЅ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv2/BiasAddo

relu2/ReluReluconv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimЛ
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv3/conv1d/ExpandDimsЬ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimб
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv3/conv1d/ExpandDims_1Я
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv3/conv1dЅ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv3/BiasAdd/ReadVariableOpЅ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimВ
shortcut/conv1d/ExpandDims
ExpandDimsinput_1'shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
shortcut/conv1d/ExpandDimsд
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimм
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2
shortcut/conv1d/ExpandDims_1л
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
shortcut/conv1dЎ
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЈ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
shortcut/BiasAdd/ReadVariableOpБ
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
add/addl
out_block/ReluReluadd/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
out_block/Reluu
IdentityIdentityout_block/Relu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@@:::::::::T P
+
_output_shapes
:џџџџџџџџџ@@
!
_user_specified_name	input_1
Ю;
ѕ
H__inference_resnet_block_layer_call_and_return_conditional_losses_216249

inputs5
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЈ
conv1/conv1d/ExpandDims
ExpandDimsinputs$conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv1/conv1d/ExpandDimsЪ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimЯ
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1/conv1d/ExpandDims_1Ю
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv1/conv1dЄ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1/BiasAdd/ReadVariableOpЄ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
conv1/BiasAddn

relu1/ReluReluconv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimК
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv2/conv1d/ExpandDimsЪ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimЯ
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv2/conv1d/ExpandDims_1Ю
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv2/conv1dЄ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOpЄ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
conv2/BiasAddn

relu2/ReluReluconv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimК
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv3/conv1d/ExpandDimsЪ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimЯ
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv3/conv1d/ExpandDims_1Ю
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv3/conv1dЄ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOpЄ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimБ
shortcut/conv1d/ExpandDims
ExpandDimsinputs'shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
shortcut/conv1d/ExpandDimsг
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimл
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
shortcut/conv1d/ExpandDims_1к
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
shortcut/conv1d­
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЇ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
shortcut/BiasAdd/ReadVariableOpА
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
add/addk
out_block/ReluReluadd/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
out_block/Relut
IdentityIdentityout_block/Relu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@:::::::::S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
э
Ф
(__inference_res_net_layer_call_fn_215767

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_res_net_layer_call_and_return_conditional_losses_2132532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:џџџџџџџџџ:	::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:%!

_output_shapes
:	
Ж
Ж
A__inference_conv2_layer_call_and_return_conditional_losses_210201

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@ :::S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Н
Ж
A__inference_conv1_layer_call_and_return_conditional_losses_217557

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv1d/ExpandDimsЙ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimИ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@@:::S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
І

H__inference_resnet_block_layer_call_and_return_conditional_losses_210471

inputs
conv1_210446
conv1_210448
conv2_210452
conv2_210454
conv3_210458
conv3_210460
shortcut_210463
shortcut_210465
identityЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallЂconv3/StatefulPartitionedCallЂ shortcut/StatefulPartitionedCall
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_210446conv1_210448*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2101232
conv1/StatefulPartitionedCallя
relu1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu1_layer_call_and_return_conditional_losses_2101582
relu1/PartitionedCallЁ
conv2/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0conv2_210452conv2_210454*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2102012
conv2/StatefulPartitionedCallя
relu2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu2_layer_call_and_return_conditional_losses_2102362
relu2/PartitionedCallЁ
conv3/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0conv3_210458conv3_210460*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2102792
conv3/StatefulPartitionedCall
 shortcut/StatefulPartitionedCallStatefulPartitionedCallinputsshortcut_210463shortcut_210465*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_shortcut_layer_call_and_return_conditional_losses_2103342"
 shortcut/StatefulPartitionedCallЄ
add/addAddV2&conv3/StatefulPartitionedCall:output:0)shortcut/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
add/addр
out_block/PartitionedCallPartitionedCalladd/add:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_out_block_layer_call_and_return_conditional_losses_2103702
out_block/PartitionedCall§
IdentityIdentity"out_block/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall!^shortcut/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2D
 shortcut/StatefulPartitionedCall shortcut/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
 
B
&__inference_relu1_layer_call_fn_217702

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu1_layer_call_and_return_conditional_losses_2115712
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
­
Ї
?__inference_fc1_layer_call_and_return_conditional_losses_212716

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ@:::P L
(
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
М
п
/__inference_resnet_block_1_layer_call_fn_216489
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_2110122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@ ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ@ 
!
_user_specified_name	input_1
У
a
E__inference_out_block_layer_call_and_return_conditional_losses_217411

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@ :S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
У
]
A__inference_relu2_layer_call_and_return_conditional_losses_217731

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:џџџџџџџџџ@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Т
Ж
A__inference_conv2_layer_call_and_return_conditional_losses_217717

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsК
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЙ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@:::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
П
]
A__inference_relu2_layer_call_and_return_conditional_losses_210236

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@ :S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
№
Х
(__inference_res_net_layer_call_fn_214679
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_res_net_layer_call_and_return_conditional_losses_2132532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:џџџџџџџџџ:	::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:%!

_output_shapes
:	
Ч
a
E__inference_out_block_layer_call_and_return_conditional_losses_211312

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:џџџџџџџџџ@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
№
~
)__inference_shortcut_layer_call_fn_217784

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_shortcut_layer_call_and_return_conditional_losses_2117472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Й
о
/__inference_resnet_block_1_layer_call_fn_216635

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_2110122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@ ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Т
Ж
A__inference_conv3_layer_call_and_return_conditional_losses_211221

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsК
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЙ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@:::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
э
Ф
(__inference_res_net_layer_call_fn_215850

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_res_net_layer_call_and_return_conditional_losses_2132532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:џџџџџџџџџ:	::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:%!

_output_shapes
:	
Е
м
-__inference_resnet_block_layer_call_fn_216343

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_resnet_block_layer_call_and_return_conditional_losses_2105412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Х
Й
D__inference_shortcut_layer_call_and_return_conditional_losses_211747

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsК
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЙ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@:::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
М
п
/__inference_resnet_block_1_layer_call_fn_216468
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_2109422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@ ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ@ 
!
_user_specified_name	input_1
ќ
X
A__inference_delta_layer_call_and_return_conditional_losses_212416
x
identityy
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/permy
	transpose	Transposextranspose/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
	transpose
ConstConst*
_output_shapes

:*
dtype0*9
value0B."                               2
Const
	MirrorPad	MirrorPadtranspose:y:0Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
mode	SYMMETRIC2
	MirrorPadg
arange/startConst*
_output_shapes
: *
dtype0*
valueB :
ќџџџџџџџџ2
arange/start^
arange/limitConst*
_output_shapes
: *
dtype0*
value	B :2
arange/limit^
arange/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
arange/deltaz
arangeRangearange/start:output:0arange/limit:output:0arange/delta:output:0*
_output_shapes
:	2
arangeY
CastCastarange:output:0*

DstT0*

SrcT0*
_output_shapes
:	2
Castw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
Reshape/shapep
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*&
_output_shapes
:	2	
Reshape­
convolutionConv2DMirrorPad:output:0Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
convolution[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pB2
	truediv/y
truedivRealDivconvolution:output:0truediv/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
truediv}
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose_1/perm
transpose_1	Transposetruediv:z:0transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
transpose_1k
IdentityIdentitytranspose_1:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:R N
/
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
Л
о
/__inference_resnet_block_2_layer_call_fn_216927

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_2114832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Т
Ж
A__inference_conv3_layer_call_and_return_conditional_losses_217751

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsК
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЙ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@:::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ж
y
$__inference_fc1_layer_call_fn_217250

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_2127162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ч
a
E__inference_out_block_layer_call_and_return_conditional_losses_217789

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:џџџџџџџџџ@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ъ
{
&__inference_conv3_layer_call_fn_217634

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2112212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
У
a
E__inference_out_block_layer_call_and_return_conditional_losses_210841

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџ@@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@@:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ю
~
)__inference_shortcut_layer_call_fn_217658

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_shortcut_layer_call_and_return_conditional_losses_2112762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ц
{
&__inference_conv2_layer_call_fn_217348

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2102012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@ ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
ћ;
ї
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_217125

inputs5
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЉ
conv1/conv1d/ExpandDims
ExpandDimsinputs$conv1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1/conv1d/ExpandDimsЬ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimб
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1/conv1d/ExpandDims_1Я
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1/conv1dЅ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv1/BiasAdd/ReadVariableOpЅ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv1/BiasAddo

relu1/ReluReluconv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimЛ
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv2/conv1d/ExpandDimsЬ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimб
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv2/conv1d/ExpandDims_1Я
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2/conv1dЅ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2/BiasAdd/ReadVariableOpЅ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv2/BiasAddo

relu2/ReluReluconv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimЛ
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv3/conv1d/ExpandDimsЬ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimб
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv3/conv1d/ExpandDims_1Я
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv3/conv1dЅ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv3/BiasAdd/ReadVariableOpЅ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimВ
shortcut/conv1d/ExpandDims
ExpandDimsinputs'shortcut/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
shortcut/conv1d/ExpandDimsе
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimн
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
shortcut/conv1d/ExpandDims_1л
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
shortcut/conv1dЎ
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЈ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
shortcut/BiasAdd/ReadVariableOpБ
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
add/addl
out_block/ReluReluadd/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
out_block/Reluu
IdentityIdentityout_block/Relu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:џџџџџџџџџ@:::::::::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Х
Й
D__inference_shortcut_layer_call_and_return_conditional_losses_217775

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsК
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЙ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@:::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
љi
Ц
__inference__traced_save_217980
file_prefix1
-savev2_res_net_fc1_kernel_read_readvariableop/
+savev2_res_net_fc1_bias_read_readvariableop1
-savev2_res_net_fc2_kernel_read_readvariableop/
+savev2_res_net_fc2_bias_read_readvariableop1
-savev2_res_net_fc3_kernel_read_readvariableop/
+savev2_res_net_fc3_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop@
<savev2_res_net_resnet_block_conv1_kernel_read_readvariableop>
:savev2_res_net_resnet_block_conv1_bias_read_readvariableop@
<savev2_res_net_resnet_block_conv2_kernel_read_readvariableop>
:savev2_res_net_resnet_block_conv2_bias_read_readvariableop@
<savev2_res_net_resnet_block_conv3_kernel_read_readvariableop>
:savev2_res_net_resnet_block_conv3_bias_read_readvariableopC
?savev2_res_net_resnet_block_shortcut_kernel_read_readvariableopA
=savev2_res_net_resnet_block_shortcut_bias_read_readvariableopB
>savev2_res_net_resnet_block_1_conv1_kernel_read_readvariableop@
<savev2_res_net_resnet_block_1_conv1_bias_read_readvariableopB
>savev2_res_net_resnet_block_1_conv2_kernel_read_readvariableop@
<savev2_res_net_resnet_block_1_conv2_bias_read_readvariableopB
>savev2_res_net_resnet_block_1_conv3_kernel_read_readvariableop@
<savev2_res_net_resnet_block_1_conv3_bias_read_readvariableopE
Asavev2_res_net_resnet_block_1_shortcut_kernel_read_readvariableopC
?savev2_res_net_resnet_block_1_shortcut_bias_read_readvariableopB
>savev2_res_net_resnet_block_2_conv1_kernel_read_readvariableop@
<savev2_res_net_resnet_block_2_conv1_bias_read_readvariableopB
>savev2_res_net_resnet_block_2_conv2_kernel_read_readvariableop@
<savev2_res_net_resnet_block_2_conv2_bias_read_readvariableopB
>savev2_res_net_resnet_block_2_conv3_kernel_read_readvariableop@
<savev2_res_net_resnet_block_2_conv3_bias_read_readvariableopE
Asavev2_res_net_resnet_block_2_shortcut_kernel_read_readvariableopC
?savev2_res_net_resnet_block_2_shortcut_bias_read_readvariableopB
>savev2_res_net_resnet_block_3_conv1_kernel_read_readvariableop@
<savev2_res_net_resnet_block_3_conv1_bias_read_readvariableopB
>savev2_res_net_resnet_block_3_conv2_kernel_read_readvariableop@
<savev2_res_net_resnet_block_3_conv2_bias_read_readvariableopB
>savev2_res_net_resnet_block_3_conv3_kernel_read_readvariableop@
<savev2_res_net_resnet_block_3_conv3_bias_read_readvariableopE
Asavev2_res_net_resnet_block_3_shortcut_kernel_read_readvariableopC
?savev2_res_net_resnet_block_3_shortcut_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop*
&savev2_accumulator_read_readvariableop,
(savev2_accumulator_1_read_readvariableop,
(savev2_accumulator_2_read_readvariableop,
(savev2_accumulator_3_read_readvariableop
savev2_const_1

identity_1ЂMergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_9f62120cd3194168980666aba553da89/part2	
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*­
valueЃB 7B%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc3/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/5/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/6/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/7/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesї
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesђ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_res_net_fc1_kernel_read_readvariableop+savev2_res_net_fc1_bias_read_readvariableop-savev2_res_net_fc2_kernel_read_readvariableop+savev2_res_net_fc2_bias_read_readvariableop-savev2_res_net_fc3_kernel_read_readvariableop+savev2_res_net_fc3_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop<savev2_res_net_resnet_block_conv1_kernel_read_readvariableop:savev2_res_net_resnet_block_conv1_bias_read_readvariableop<savev2_res_net_resnet_block_conv2_kernel_read_readvariableop:savev2_res_net_resnet_block_conv2_bias_read_readvariableop<savev2_res_net_resnet_block_conv3_kernel_read_readvariableop:savev2_res_net_resnet_block_conv3_bias_read_readvariableop?savev2_res_net_resnet_block_shortcut_kernel_read_readvariableop=savev2_res_net_resnet_block_shortcut_bias_read_readvariableop>savev2_res_net_resnet_block_1_conv1_kernel_read_readvariableop<savev2_res_net_resnet_block_1_conv1_bias_read_readvariableop>savev2_res_net_resnet_block_1_conv2_kernel_read_readvariableop<savev2_res_net_resnet_block_1_conv2_bias_read_readvariableop>savev2_res_net_resnet_block_1_conv3_kernel_read_readvariableop<savev2_res_net_resnet_block_1_conv3_bias_read_readvariableopAsavev2_res_net_resnet_block_1_shortcut_kernel_read_readvariableop?savev2_res_net_resnet_block_1_shortcut_bias_read_readvariableop>savev2_res_net_resnet_block_2_conv1_kernel_read_readvariableop<savev2_res_net_resnet_block_2_conv1_bias_read_readvariableop>savev2_res_net_resnet_block_2_conv2_kernel_read_readvariableop<savev2_res_net_resnet_block_2_conv2_bias_read_readvariableop>savev2_res_net_resnet_block_2_conv3_kernel_read_readvariableop<savev2_res_net_resnet_block_2_conv3_bias_read_readvariableopAsavev2_res_net_resnet_block_2_shortcut_kernel_read_readvariableop?savev2_res_net_resnet_block_2_shortcut_bias_read_readvariableop>savev2_res_net_resnet_block_3_conv1_kernel_read_readvariableop<savev2_res_net_resnet_block_3_conv1_bias_read_readvariableop>savev2_res_net_resnet_block_3_conv2_kernel_read_readvariableop<savev2_res_net_resnet_block_3_conv2_bias_read_readvariableop>savev2_res_net_resnet_block_3_conv3_kernel_read_readvariableop<savev2_res_net_resnet_block_3_conv3_bias_read_readvariableopAsavev2_res_net_resnet_block_3_shortcut_kernel_read_readvariableop?savev2_res_net_resnet_block_3_shortcut_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop&savev2_accumulator_read_readvariableop(savev2_accumulator_1_read_readvariableop(savev2_accumulator_2_read_readvariableop(savev2_accumulator_3_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *E
dtypes;
927	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*ц
_input_shapesд
б: :
@::
::	:: : : : : : :  : :  : : : : @:@:@@:@:@@:@: @:@:@::::::@:::::::::: : : : ::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
@:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
: @: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
: @: 

_output_shapes
:@:)%
#
_output_shapes
:@:!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::! 

_output_shapes	
::)!%
#
_output_shapes
:@:!"

_output_shapes	
::*#&
$
_output_shapes
::!$

_output_shapes	
::*%&
$
_output_shapes
::!&

_output_shapes	
::*'&
$
_output_shapes
::!(

_output_shapes	
::*)&
$
_output_shapes
::!*

_output_shapes	
::+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: : /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
::7

_output_shapes
: 
О
п
/__inference_resnet_block_2_layer_call_fn_216760
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_2114132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ@@
!
_user_specified_name	input_1
ъ
{
&__inference_conv1_layer_call_fn_217692

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2115362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
У
]
A__inference_relu1_layer_call_and_return_conditional_losses_217571

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:џџџџџџџџџ@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Б

J__inference_resnet_block_2_layer_call_and_return_conditional_losses_211413

inputs
conv1_211388
conv1_211390
conv2_211394
conv2_211396
conv3_211400
conv3_211402
shortcut_211405
shortcut_211407
identityЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallЂconv3/StatefulPartitionedCallЂ shortcut/StatefulPartitionedCall
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_211388conv1_211390*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2110652
conv1/StatefulPartitionedCall№
relu1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu1_layer_call_and_return_conditional_losses_2111002
relu1/PartitionedCallЂ
conv2/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0conv2_211394conv2_211396*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2111432
conv2/StatefulPartitionedCall№
relu2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu2_layer_call_and_return_conditional_losses_2111782
relu2/PartitionedCallЂ
conv3/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0conv3_211400conv3_211402*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2112212
conv3/StatefulPartitionedCall
 shortcut/StatefulPartitionedCallStatefulPartitionedCallinputsshortcut_211405shortcut_211407*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_shortcut_layer_call_and_return_conditional_losses_2112762"
 shortcut/StatefulPartitionedCallЅ
add/addAddV2&conv3/StatefulPartitionedCall:output:0)shortcut/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
add/addс
out_block/PartitionedCallPartitionedCalladd/add:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_out_block_layer_call_and_return_conditional_losses_2113122
out_block/PartitionedCallў
IdentityIdentity"out_block/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall!^shortcut/StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@@::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2D
 shortcut/StatefulPartitionedCall shortcut/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
У
]
A__inference_relu1_layer_call_and_return_conditional_losses_217697

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:џџџџџџџџџ@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
д
y
$__inference_fc3_layer_call_fn_217290

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc3_layer_call_and_return_conditional_losses_2128102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ;
ї
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_216833

inputs5
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЈ
conv1/conv1d/ExpandDims
ExpandDimsinputs$conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv1/conv1d/ExpandDimsЫ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimа
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2
conv1/conv1d/ExpandDims_1Я
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1/conv1dЅ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv1/BiasAdd/ReadVariableOpЅ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv1/BiasAddo

relu1/ReluReluconv1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimЛ
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv2/conv1d/ExpandDimsЬ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimб
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv2/conv1d/ExpandDims_1Я
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2/conv1dЅ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2/BiasAdd/ReadVariableOpЅ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv2/BiasAddo

relu2/ReluReluconv2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimЛ
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv3/conv1d/ExpandDimsЬ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimб
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv3/conv1d/ExpandDims_1Я
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv3/conv1dЅ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv3/BiasAdd/ReadVariableOpЅ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimБ
shortcut/conv1d/ExpandDims
ExpandDimsinputs'shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
shortcut/conv1d/ExpandDimsд
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimм
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2
shortcut/conv1d/ExpandDims_1л
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
shortcut/conv1dЎ
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЈ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
shortcut/BiasAdd/ReadVariableOpБ
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
add/addl
out_block/ReluReluadd/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@2
out_block/Reluu
IdentityIdentityout_block/Relu:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@@:::::::::S O
+
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ј
F
*__inference_out_block_layer_call_fn_217668

inputs
identityШ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_out_block_layer_call_and_return_conditional_losses_2113122
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
И
н
-__inference_resnet_block_layer_call_fn_216197
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@ **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_resnet_block_layer_call_and_return_conditional_losses_2105412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_1
­
Ї
?__inference_fc1_layer_call_and_return_conditional_losses_217241

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ@:::P L
(
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ј
F
*__inference_out_block_layer_call_fn_217794

inputs
identityШ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_out_block_layer_call_and_return_conditional_losses_2117832
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
П
]
A__inference_relu2_layer_call_and_return_conditional_losses_217353

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ@ :S O
+
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
 
B
&__inference_relu1_layer_call_fn_217576

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_relu1_layer_call_and_return_conditional_losses_2111002
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Й
Й
D__inference_shortcut_layer_call_and_return_conditional_losses_210334

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ@:::S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
в;
і
H__inference_resnet_block_layer_call_and_return_conditional_losses_216103
input_15
1conv1_conv1d_expanddims_1_readvariableop_resource)
%conv1_biasadd_readvariableop_resource5
1conv2_conv1d_expanddims_1_readvariableop_resource)
%conv2_biasadd_readvariableop_resource5
1conv3_conv1d_expanddims_1_readvariableop_resource)
%conv3_biasadd_readvariableop_resource8
4shortcut_conv1d_expanddims_1_readvariableop_resource,
(shortcut_biasadd_readvariableop_resource
identity
conv1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1/conv1d/ExpandDims/dimЉ
conv1/conv1d/ExpandDims
ExpandDimsinput_1$conv1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv1/conv1d/ExpandDimsЪ
(conv1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02*
(conv1/conv1d/ExpandDims_1/ReadVariableOp
conv1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1/conv1d/ExpandDims_1/dimЯ
conv1/conv1d/ExpandDims_1
ExpandDims0conv1/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1/conv1d/ExpandDims_1Ю
conv1/conv1dConv2D conv1/conv1d/ExpandDims:output:0"conv1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv1/conv1dЄ
conv1/conv1d/SqueezeSqueezeconv1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv1/conv1d/Squeeze
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1/BiasAdd/ReadVariableOpЄ
conv1/BiasAddBiasAddconv1/conv1d/Squeeze:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
conv1/BiasAddn

relu1/ReluReluconv1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

relu1/Relu
conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv2/conv1d/ExpandDims/dimК
conv2/conv1d/ExpandDims
ExpandDimsrelu1/Relu:activations:0$conv2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv2/conv1d/ExpandDimsЪ
(conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02*
(conv2/conv1d/ExpandDims_1/ReadVariableOp
conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2/conv1d/ExpandDims_1/dimЯ
conv2/conv1d/ExpandDims_1
ExpandDims0conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv2/conv1d/ExpandDims_1Ю
conv2/conv1dConv2D conv2/conv1d/ExpandDims:output:0"conv2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv2/conv1dЄ
conv2/conv1d/SqueezeSqueezeconv2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv2/conv1d/Squeeze
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOpЄ
conv2/BiasAddBiasAddconv2/conv1d/Squeeze:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
conv2/BiasAddn

relu2/ReluReluconv2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

relu2/Relu
conv3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv3/conv1d/ExpandDims/dimК
conv3/conv1d/ExpandDims
ExpandDimsrelu2/Relu:activations:0$conv3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 2
conv3/conv1d/ExpandDimsЪ
(conv3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1conv3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02*
(conv3/conv1d/ExpandDims_1/ReadVariableOp
conv3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv3/conv1d/ExpandDims_1/dimЯ
conv3/conv1d/ExpandDims_1
ExpandDims0conv3/conv1d/ExpandDims_1/ReadVariableOp:value:0&conv3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv3/conv1d/ExpandDims_1Ю
conv3/conv1dConv2D conv3/conv1d/ExpandDims:output:0"conv3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
conv3/conv1dЄ
conv3/conv1d/SqueezeSqueezeconv3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
conv3/conv1d/Squeeze
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOpЄ
conv3/BiasAddBiasAddconv3/conv1d/Squeeze:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
conv3/BiasAdd
shortcut/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
shortcut/conv1d/ExpandDims/dimВ
shortcut/conv1d/ExpandDims
ExpandDimsinput_1'shortcut/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
shortcut/conv1d/ExpandDimsг
+shortcut/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4shortcut_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+shortcut/conv1d/ExpandDims_1/ReadVariableOp
 shortcut/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 shortcut/conv1d/ExpandDims_1/dimл
shortcut/conv1d/ExpandDims_1
ExpandDims3shortcut/conv1d/ExpandDims_1/ReadVariableOp:value:0)shortcut/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
shortcut/conv1d/ExpandDims_1к
shortcut/conv1dConv2D#shortcut/conv1d/ExpandDims:output:0%shortcut/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
paddingSAME*
strides
2
shortcut/conv1d­
shortcut/conv1d/SqueezeSqueezeshortcut/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ *
squeeze_dims

§џџџџџџџџ2
shortcut/conv1d/SqueezeЇ
shortcut/BiasAdd/ReadVariableOpReadVariableOp(shortcut_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
shortcut/BiasAdd/ReadVariableOpА
shortcut/BiasAddBiasAdd shortcut/conv1d/Squeeze:output:0'shortcut/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
shortcut/BiasAdd
add/addAddV2conv3/BiasAdd:output:0shortcut/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2	
add/addk
out_block/ReluReluadd/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2
out_block/Relut
IdentityIdentityout_block/Relu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ@ 2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@:::::::::T P
+
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_1
Љ
Ї
?__inference_fc3_layer_call_and_return_conditional_losses_212810

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Т
Ж
A__inference_conv2_layer_call_and_return_conditional_losses_217591

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@2
conv1d/ExpandDimsК
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЙ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ@:::T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
О
п
/__inference_resnet_block_2_layer_call_fn_216781
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_2114832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ@@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ@@
!
_user_specified_name	input_1"їL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*А
serving_default
@
input_15
serving_default_input_1:0џџџџџџџџџ<
output_10
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:бІ
М
	n_filters
	n_kernels
n_fc
mel
	delta

block1

block2

block3

	block4

flatten
fc1
fc2
fc3
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
р_default_save_signature
с__call__
+т&call_and_return_all_conditional_losses"к

_tf_keras_modelР
{"class_name": "ResNet", "name": "res_net", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ResNet"}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy", {"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "TruePositives", "config": {"name": "true_positives", "dtype": "float32", "thresholds": null}}, {"class_name": "TrueNegatives", "config": {"name": "true_negatives", "dtype": "float32", "thresholds": null}}, {"class_name": "FalsePositives", "config": {"name": "false_positives", "dtype": "float32", "thresholds": null}}, {"class_name": "FalseNegatives", "config": {"name": "false_negatives", "dtype": "float32", "thresholds": null}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 9.999999747378752e-06, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

regularization_losses
	variables
trainable_variables
	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"ё
_tf_keras_layerз{"class_name": "LogMelSpectrogram", "name": "log_mel_spectrogram", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"fft_size": 512, "hop_size": 16, "n_mels": 5, "sample_rate": 16000, "f_min": 0.0, "f_max": 8000.0, "name": "log_mel_spectrogram", "trainable": true, "dtype": "float32"}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 1024]}}
ѕ
regularization_losses
	variables
trainable_variables
	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"ф
_tf_keras_layerЪ{"class_name": "Delta", "name": "delta", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "delta", "trainable": true, "dtype": "float32", "win_length": 9, "mode": "symmetric", "data_format": "channels_first"}}
ї
	n_kernels
	conv1
	relu1
	conv2
	relu2
	 conv3
!shortcut
"	out_block
#regularization_losses
$	variables
%trainable_variables
&	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"
_tf_keras_modelщ{"class_name": "ResnetBlock", "name": "resnet_block", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ResnetBlock"}}
љ
	n_kernels
	'conv1
	(relu1
	)conv2
	*relu2
	+conv3
,shortcut
-	out_block
.regularization_losses
/	variables
0trainable_variables
1	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"
_tf_keras_modelы{"class_name": "ResnetBlock", "name": "resnet_block_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ResnetBlock"}}
љ
	n_kernels
	2conv1
	3relu1
	4conv2
	5relu2
	6conv3
7shortcut
8	out_block
9regularization_losses
:	variables
;trainable_variables
<	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"
_tf_keras_modelы{"class_name": "ResnetBlock", "name": "resnet_block_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ResnetBlock"}}
љ
	n_kernels
	=conv1
	>relu1
	?conv2
	@relu2
	Aconv3
Bshortcut
C	out_block
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"
_tf_keras_modelы{"class_name": "ResnetBlock", "name": "resnet_block_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ResnetBlock"}}
ф
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
я__call__
+№&call_and_return_all_conditional_losses"г
_tf_keras_layerЙ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ю

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
ё__call__
+ђ&call_and_return_all_conditional_losses"Ч
_tf_keras_layer­{"class_name": "Dense", "name": "fc1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 2048, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 8192]}}
ю

Rkernel
Sbias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses"Ч
_tf_keras_layer­{"class_name": "Dense", "name": "fc2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 2048, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 2048]}}
ю

Xkernel
Ybias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
ѕ__call__
+і&call_and_return_all_conditional_losses"Ч
_tf_keras_layer­{"class_name": "Dense", "name": "fc3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc3", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 2048]}}
I
^iter
	_decay
`learning_rate
amomentum"
	optimizer
 "
trackable_list_wrapper
Ш
b0
c1
d2
e3
f4
g5
h6
i7
j8
k9
l10
m11
n12
o13
p14
q15
r16
s17
t18
u19
v20
w21
x22
y23
z24
{25
|26
}27
~28
29
30
31
L32
M33
R34
S35
X36
Y37"
trackable_list_wrapper
Ш
b0
c1
d2
e3
f4
g5
h6
i7
j8
k9
l10
m11
n12
o13
p14
q15
r16
s17
t18
u19
v20
w21
x22
y23
z24
{25
|26
}27
~28
29
30
31
L32
M33
R34
S35
X36
Y37"
trackable_list_wrapper
г
regularization_losses
layers
layer_metrics
non_trainable_variables
	variables
metrics
trainable_variables
 layer_regularization_losses
с__call__
р_default_save_signature
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
-
їserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
layers
regularization_losses
non_trainable_variables
	variables
metrics
trainable_variables
 layer_regularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
layers
regularization_losses
non_trainable_variables
	variables
metrics
trainable_variables
 layer_regularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
х	

bkernel
cbias
regularization_losses
	variables
trainable_variables
	keras_api
ј__call__
+љ&call_and_return_all_conditional_losses"К
_tf_keras_layer {"class_name": "Conv1D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 16]}}
Э
regularization_losses
	variables
trainable_variables
	keras_api
њ__call__
+ћ&call_and_return_all_conditional_losses"И
_tf_keras_layer{"class_name": "Activation", "name": "relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu1", "trainable": true, "dtype": "float32", "activation": "relu"}}
х	

dkernel
ebias
regularization_losses
	variables
trainable_variables
	keras_api
ќ__call__
+§&call_and_return_all_conditional_losses"К
_tf_keras_layer {"class_name": "Conv1D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 32]}}
Э
regularization_losses
	variables
trainable_variables
 	keras_api
ў__call__
+џ&call_and_return_all_conditional_losses"И
_tf_keras_layer{"class_name": "Activation", "name": "relu2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu2", "trainable": true, "dtype": "float32", "activation": "relu"}}
х	

fkernel
gbias
Ёregularization_losses
Ђ	variables
Ѓtrainable_variables
Є	keras_api
__call__
+&call_and_return_all_conditional_losses"К
_tf_keras_layer {"class_name": "Conv1D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 32]}}
ы	

hkernel
ibias
Ѕregularization_losses
І	variables
Їtrainable_variables
Ј	keras_api
__call__
+&call_and_return_all_conditional_losses"Р
_tf_keras_layerІ{"class_name": "Conv1D", "name": "shortcut", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "shortcut", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 16]}}
е
Љregularization_losses
Њ	variables
Ћtrainable_variables
Ќ	keras_api
__call__
+&call_and_return_all_conditional_losses"Р
_tf_keras_layerІ{"class_name": "Activation", "name": "out_block", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "out_block", "trainable": true, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
X
b0
c1
d2
e3
f4
g5
h6
i7"
trackable_list_wrapper
X
b0
c1
d2
e3
f4
g5
h6
i7"
trackable_list_wrapper
Е
#regularization_losses
­layers
Ўlayer_metrics
Џnon_trainable_variables
$	variables
Аmetrics
%trainable_variables
 Бlayer_regularization_losses
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
х	

jkernel
kbias
Вregularization_losses
Г	variables
Дtrainable_variables
Е	keras_api
__call__
+&call_and_return_all_conditional_losses"К
_tf_keras_layer {"class_name": "Conv1D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 32]}}
Э
Жregularization_losses
З	variables
Иtrainable_variables
Й	keras_api
__call__
+&call_and_return_all_conditional_losses"И
_tf_keras_layer{"class_name": "Activation", "name": "relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu1", "trainable": true, "dtype": "float32", "activation": "relu"}}
х	

lkernel
mbias
Кregularization_losses
Л	variables
Мtrainable_variables
Н	keras_api
__call__
+&call_and_return_all_conditional_losses"К
_tf_keras_layer {"class_name": "Conv1D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 64]}}
Э
Оregularization_losses
П	variables
Рtrainable_variables
С	keras_api
__call__
+&call_and_return_all_conditional_losses"И
_tf_keras_layer{"class_name": "Activation", "name": "relu2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu2", "trainable": true, "dtype": "float32", "activation": "relu"}}
х	

nkernel
obias
Тregularization_losses
У	variables
Фtrainable_variables
Х	keras_api
__call__
+&call_and_return_all_conditional_losses"К
_tf_keras_layer {"class_name": "Conv1D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 64]}}
ы	

pkernel
qbias
Цregularization_losses
Ч	variables
Шtrainable_variables
Щ	keras_api
__call__
+&call_and_return_all_conditional_losses"Р
_tf_keras_layerІ{"class_name": "Conv1D", "name": "shortcut", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "shortcut", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 32]}}
е
Ъregularization_losses
Ы	variables
Ьtrainable_variables
Э	keras_api
__call__
+&call_and_return_all_conditional_losses"Р
_tf_keras_layerІ{"class_name": "Activation", "name": "out_block", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "out_block", "trainable": true, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
X
j0
k1
l2
m3
n4
o5
p6
q7"
trackable_list_wrapper
X
j0
k1
l2
m3
n4
o5
p6
q7"
trackable_list_wrapper
Е
.regularization_losses
Юlayers
Яlayer_metrics
аnon_trainable_variables
/	variables
бmetrics
0trainable_variables
 вlayer_regularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
ц	

rkernel
sbias
гregularization_losses
д	variables
еtrainable_variables
ж	keras_api
__call__
+&call_and_return_all_conditional_losses"Л
_tf_keras_layerЁ{"class_name": "Conv1D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 64]}}
Э
зregularization_losses
и	variables
йtrainable_variables
к	keras_api
__call__
+&call_and_return_all_conditional_losses"И
_tf_keras_layer{"class_name": "Activation", "name": "relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu1", "trainable": true, "dtype": "float32", "activation": "relu"}}
ш	

tkernel
ubias
лregularization_losses
м	variables
нtrainable_variables
о	keras_api
__call__
+&call_and_return_all_conditional_losses"Н
_tf_keras_layerЃ{"class_name": "Conv1D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 128]}}
Э
пregularization_losses
р	variables
сtrainable_variables
т	keras_api
__call__
+&call_and_return_all_conditional_losses"И
_tf_keras_layer{"class_name": "Activation", "name": "relu2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu2", "trainable": true, "dtype": "float32", "activation": "relu"}}
ш	

vkernel
wbias
уregularization_losses
ф	variables
хtrainable_variables
ц	keras_api
__call__
+&call_and_return_all_conditional_losses"Н
_tf_keras_layerЃ{"class_name": "Conv1D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 128]}}
ь	

xkernel
ybias
чregularization_losses
ш	variables
щtrainable_variables
ъ	keras_api
__call__
+&call_and_return_all_conditional_losses"С
_tf_keras_layerЇ{"class_name": "Conv1D", "name": "shortcut", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "shortcut", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 64]}}
е
ыregularization_losses
ь	variables
эtrainable_variables
ю	keras_api
 __call__
+Ё&call_and_return_all_conditional_losses"Р
_tf_keras_layerІ{"class_name": "Activation", "name": "out_block", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "out_block", "trainable": true, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
X
r0
s1
t2
u3
v4
w5
x6
y7"
trackable_list_wrapper
X
r0
s1
t2
u3
v4
w5
x6
y7"
trackable_list_wrapper
Е
9regularization_losses
яlayers
№layer_metrics
ёnon_trainable_variables
:	variables
ђmetrics
;trainable_variables
 ѓlayer_regularization_losses
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
ш	

zkernel
{bias
єregularization_losses
ѕ	variables
іtrainable_variables
ї	keras_api
Ђ__call__
+Ѓ&call_and_return_all_conditional_losses"Н
_tf_keras_layerЃ{"class_name": "Conv1D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 128]}}
Э
јregularization_losses
љ	variables
њtrainable_variables
ћ	keras_api
Є__call__
+Ѕ&call_and_return_all_conditional_losses"И
_tf_keras_layer{"class_name": "Activation", "name": "relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu1", "trainable": true, "dtype": "float32", "activation": "relu"}}
ш	

|kernel
}bias
ќregularization_losses
§	variables
ўtrainable_variables
џ	keras_api
І__call__
+Ї&call_and_return_all_conditional_losses"Н
_tf_keras_layerЃ{"class_name": "Conv1D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 128]}}
Э
regularization_losses
	variables
trainable_variables
	keras_api
Ј__call__
+Љ&call_and_return_all_conditional_losses"И
_tf_keras_layer{"class_name": "Activation", "name": "relu2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu2", "trainable": true, "dtype": "float32", "activation": "relu"}}
ш	

~kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses"Н
_tf_keras_layerЃ{"class_name": "Conv1D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 128]}}
№	
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
Ќ__call__
+­&call_and_return_all_conditional_losses"У
_tf_keras_layerЉ{"class_name": "Conv1D", "name": "shortcut", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "shortcut", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 128]}}
е
regularization_losses
	variables
trainable_variables
	keras_api
Ў__call__
+Џ&call_and_return_all_conditional_losses"Р
_tf_keras_layerІ{"class_name": "Activation", "name": "out_block", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "out_block", "trainable": true, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
Z
z0
{1
|2
}3
~4
5
6
7"
trackable_list_wrapper
Z
z0
{1
|2
}3
~4
5
6
7"
trackable_list_wrapper
Е
Dregularization_losses
layers
layer_metrics
non_trainable_variables
E	variables
metrics
Ftrainable_variables
 layer_regularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
layers
Hregularization_losses
non_trainable_variables
I	variables
metrics
Jtrainable_variables
 layer_regularization_losses
я__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
&:$
@2res_net/fc1/kernel
:2res_net/fc1/bias
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
Е
layer_metrics
layers
Nregularization_losses
non_trainable_variables
O	variables
metrics
Ptrainable_variables
 layer_regularization_losses
ё__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
&:$
2res_net/fc2/kernel
:2res_net/fc2/bias
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
Е
layer_metrics
 layers
Tregularization_losses
Ёnon_trainable_variables
U	variables
Ђmetrics
Vtrainable_variables
 Ѓlayer_regularization_losses
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
%:#	2res_net/fc3/kernel
:2res_net/fc3/bias
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
Е
Єlayer_metrics
Ѕlayers
Zregularization_losses
Іnon_trainable_variables
[	variables
Їmetrics
\trainable_variables
 Јlayer_regularization_losses
ѕ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
7:5 2!res_net/resnet_block/conv1/kernel
-:+ 2res_net/resnet_block/conv1/bias
7:5  2!res_net/resnet_block/conv2/kernel
-:+ 2res_net/resnet_block/conv2/bias
7:5  2!res_net/resnet_block/conv3/kernel
-:+ 2res_net/resnet_block/conv3/bias
::8 2$res_net/resnet_block/shortcut/kernel
0:. 2"res_net/resnet_block/shortcut/bias
9:7 @2#res_net/resnet_block_1/conv1/kernel
/:-@2!res_net/resnet_block_1/conv1/bias
9:7@@2#res_net/resnet_block_1/conv2/kernel
/:-@2!res_net/resnet_block_1/conv2/bias
9:7@@2#res_net/resnet_block_1/conv3/kernel
/:-@2!res_net/resnet_block_1/conv3/bias
<:: @2&res_net/resnet_block_1/shortcut/kernel
2:0@2$res_net/resnet_block_1/shortcut/bias
::8@2#res_net/resnet_block_2/conv1/kernel
0:.2!res_net/resnet_block_2/conv1/bias
;:92#res_net/resnet_block_2/conv2/kernel
0:.2!res_net/resnet_block_2/conv2/bias
;:92#res_net/resnet_block_2/conv3/kernel
0:.2!res_net/resnet_block_2/conv3/bias
=:;@2&res_net/resnet_block_2/shortcut/kernel
3:12$res_net/resnet_block_2/shortcut/bias
;:92#res_net/resnet_block_3/conv1/kernel
0:.2!res_net/resnet_block_3/conv1/bias
;:92#res_net/resnet_block_3/conv2/kernel
0:.2!res_net/resnet_block_3/conv2/bias
;:92#res_net/resnet_block_3/conv3/kernel
0:.2!res_net/resnet_block_3/conv3/bias
>:<2&res_net/resnet_block_3/shortcut/kernel
3:12$res_net/resnet_block_3/shortcut/bias
f
0
1
2
3
4
	5

6
7
8
9"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
`
Љ0
Њ1
Ћ2
Ќ3
­4
Ў5
Џ6
А7"
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
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
И
Бlayer_metrics
Вlayers
regularization_losses
Гnon_trainable_variables
	variables
Дmetrics
trainable_variables
 Еlayer_regularization_losses
ј__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Жlayer_metrics
Зlayers
regularization_losses
Иnon_trainable_variables
	variables
Йmetrics
trainable_variables
 Кlayer_regularization_losses
њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
И
Лlayer_metrics
Мlayers
regularization_losses
Нnon_trainable_variables
	variables
Оmetrics
trainable_variables
 Пlayer_regularization_losses
ќ__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Рlayer_metrics
Сlayers
regularization_losses
Тnon_trainable_variables
	variables
Уmetrics
trainable_variables
 Фlayer_regularization_losses
ў__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
И
Хlayer_metrics
Цlayers
Ёregularization_losses
Чnon_trainable_variables
Ђ	variables
Шmetrics
Ѓtrainable_variables
 Щlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
И
Ъlayer_metrics
Ыlayers
Ѕregularization_losses
Ьnon_trainable_variables
І	variables
Эmetrics
Їtrainable_variables
 Юlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Яlayer_metrics
аlayers
Љregularization_losses
бnon_trainable_variables
Њ	variables
вmetrics
Ћtrainable_variables
 гlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Q
0
1
2
3
 4
!5
"6"
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
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
И
дlayer_metrics
еlayers
Вregularization_losses
жnon_trainable_variables
Г	variables
зmetrics
Дtrainable_variables
 иlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
йlayer_metrics
кlayers
Жregularization_losses
лnon_trainable_variables
З	variables
мmetrics
Иtrainable_variables
 нlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
И
оlayer_metrics
пlayers
Кregularization_losses
рnon_trainable_variables
Л	variables
сmetrics
Мtrainable_variables
 тlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
уlayer_metrics
фlayers
Оregularization_losses
хnon_trainable_variables
П	variables
цmetrics
Рtrainable_variables
 чlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
И
шlayer_metrics
щlayers
Тregularization_losses
ъnon_trainable_variables
У	variables
ыmetrics
Фtrainable_variables
 ьlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
И
эlayer_metrics
юlayers
Цregularization_losses
яnon_trainable_variables
Ч	variables
№metrics
Шtrainable_variables
 ёlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ђlayer_metrics
ѓlayers
Ъregularization_losses
єnon_trainable_variables
Ы	variables
ѕmetrics
Ьtrainable_variables
 іlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Q
'0
(1
)2
*3
+4
,5
-6"
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
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
И
їlayer_metrics
јlayers
гregularization_losses
љnon_trainable_variables
д	variables
њmetrics
еtrainable_variables
 ћlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ќlayer_metrics
§layers
зregularization_losses
ўnon_trainable_variables
и	variables
џmetrics
йtrainable_variables
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
И
layer_metrics
layers
лregularization_losses
non_trainable_variables
м	variables
metrics
нtrainable_variables
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
пregularization_losses
non_trainable_variables
р	variables
metrics
сtrainable_variables
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
И
layer_metrics
layers
уregularization_losses
non_trainable_variables
ф	variables
metrics
хtrainable_variables
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
И
layer_metrics
layers
чregularization_losses
non_trainable_variables
ш	variables
metrics
щtrainable_variables
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
ыregularization_losses
non_trainable_variables
ь	variables
metrics
эtrainable_variables
 layer_regularization_losses
 __call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
Q
20
31
42
53
64
75
86"
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
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
И
layer_metrics
layers
єregularization_losses
non_trainable_variables
ѕ	variables
metrics
іtrainable_variables
 layer_regularization_losses
Ђ__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
 layers
јregularization_losses
Ёnon_trainable_variables
љ	variables
Ђmetrics
њtrainable_variables
 Ѓlayer_regularization_losses
Є__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
И
Єlayer_metrics
Ѕlayers
ќregularization_losses
Іnon_trainable_variables
§	variables
Їmetrics
ўtrainable_variables
 Јlayer_regularization_losses
І__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Љlayer_metrics
Њlayers
regularization_losses
Ћnon_trainable_variables
	variables
Ќmetrics
trainable_variables
 ­layer_regularization_losses
Ј__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
И
Ўlayer_metrics
Џlayers
regularization_losses
Аnon_trainable_variables
	variables
Бmetrics
trainable_variables
 Вlayer_regularization_losses
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
Гlayer_metrics
Дlayers
regularization_losses
Еnon_trainable_variables
	variables
Жmetrics
trainable_variables
 Зlayer_regularization_losses
Ќ__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Иlayer_metrics
Йlayers
regularization_losses
Кnon_trainable_variables
	variables
Лmetrics
trainable_variables
 Мlayer_regularization_losses
Ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
Q
=0
>1
?2
@3
A4
B5
C6"
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
П

Нtotal

Оcount
П	variables
Р	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
џ

Сtotal

Тcount
У
_fn_kwargs
Ф	variables
Х	keras_api"Г
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
Ј
Ц
thresholds
Чtrue_positives
Шfalse_positives
Щ	variables
Ъ	keras_api"Щ
_tf_keras_metricЎ{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}

Ы
thresholds
Ьtrue_positives
Эfalse_negatives
Ю	variables
Я	keras_api"Р
_tf_keras_metricЅ{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
ќ
а
thresholds
бaccumulator
в	variables
г	keras_api"Ж
_tf_keras_metric{"class_name": "TruePositives", "name": "true_positives", "dtype": "float32", "config": {"name": "true_positives", "dtype": "float32", "thresholds": null}}
ќ
д
thresholds
еaccumulator
ж	variables
з	keras_api"Ж
_tf_keras_metric{"class_name": "TrueNegatives", "name": "true_negatives", "dtype": "float32", "config": {"name": "true_negatives", "dtype": "float32", "thresholds": null}}
џ
и
thresholds
йaccumulator
к	variables
л	keras_api"Й
_tf_keras_metric{"class_name": "FalsePositives", "name": "false_positives", "dtype": "float32", "config": {"name": "false_positives", "dtype": "float32", "thresholds": null}}
џ
м
thresholds
нaccumulator
о	variables
п	keras_api"Й
_tf_keras_metric{"class_name": "FalseNegatives", "name": "false_negatives", "dtype": "float32", "config": {"name": "false_negatives", "dtype": "float32", "thresholds": null}}
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
:  (2total
:  (2count
0
Н0
О1"
trackable_list_wrapper
.
П	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
С0
Т1"
trackable_list_wrapper
.
Ф	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
Ч0
Ш1"
trackable_list_wrapper
.
Щ	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
Ь0
Э1"
trackable_list_wrapper
.
Ю	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
б0"
trackable_list_wrapper
.
в	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
е0"
trackable_list_wrapper
.
ж	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
й0"
trackable_list_wrapper
.
к	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
н0"
trackable_list_wrapper
.
о	variables"
_generic_user_object
ф2с
!__inference__wrapped_model_210089Л
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *+Ђ(
&#
input_1џџџџџџџџџ
с2о
(__inference_res_net_layer_call_fn_214679
(__inference_res_net_layer_call_fn_215850
(__inference_res_net_layer_call_fn_215767
(__inference_res_net_layer_call_fn_214762Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Э2Ъ
C__inference_res_net_layer_call_and_return_conditional_losses_214135
C__inference_res_net_layer_call_and_return_conditional_losses_214596
C__inference_res_net_layer_call_and_return_conditional_losses_215223
C__inference_res_net_layer_call_and_return_conditional_losses_215684Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
с2о
4__inference_log_mel_spectrogram_layer_call_fn_216026Ѕ
В
FullArgSpec 
args
jself
j	waveforms
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќ2љ
O__inference_log_mel_spectrogram_layer_call_and_return_conditional_losses_216019Ѕ
В
FullArgSpec 
args
jself
j	waveforms
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ы2Ш
&__inference_delta_layer_call_fn_216051
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ц2у
A__inference_delta_layer_call_and_return_conditional_losses_216046
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2џ
-__inference_resnet_block_layer_call_fn_216176
-__inference_resnet_block_layer_call_fn_216197
-__inference_resnet_block_layer_call_fn_216343
-__inference_resnet_block_layer_call_fn_216322Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ю2ы
H__inference_resnet_block_layer_call_and_return_conditional_losses_216301
H__inference_resnet_block_layer_call_and_return_conditional_losses_216103
H__inference_resnet_block_layer_call_and_return_conditional_losses_216155
H__inference_resnet_block_layer_call_and_return_conditional_losses_216249Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
/__inference_resnet_block_1_layer_call_fn_216468
/__inference_resnet_block_1_layer_call_fn_216635
/__inference_resnet_block_1_layer_call_fn_216614
/__inference_resnet_block_1_layer_call_fn_216489Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_216395
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_216541
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_216447
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_216593Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
/__inference_resnet_block_2_layer_call_fn_216760
/__inference_resnet_block_2_layer_call_fn_216927
/__inference_resnet_block_2_layer_call_fn_216781
/__inference_resnet_block_2_layer_call_fn_216906Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_216885
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_216739
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_216833
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_216687Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
/__inference_resnet_block_3_layer_call_fn_217198
/__inference_resnet_block_3_layer_call_fn_217052
/__inference_resnet_block_3_layer_call_fn_217219
/__inference_resnet_block_3_layer_call_fn_217073Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_217031
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_217125
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_216979
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_217177Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
в2Я
(__inference_flatten_layer_call_fn_217230Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_flatten_layer_call_and_return_conditional_losses_217225Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ю2Ы
$__inference_fc1_layer_call_fn_217250Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
щ2ц
?__inference_fc1_layer_call_and_return_conditional_losses_217241Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ю2Ы
$__inference_fc2_layer_call_fn_217270Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
щ2ц
?__inference_fc2_layer_call_and_return_conditional_losses_217261Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ю2Ы
$__inference_fc3_layer_call_fn_217290Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
щ2ц
?__inference_fc3_layer_call_and_return_conditional_losses_217281Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
3B1
$__inference_signature_wrapper_213674input_1
а2Э
&__inference_conv1_layer_call_fn_217314Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv1_layer_call_and_return_conditional_losses_217305Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_relu1_layer_call_fn_217324Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_relu1_layer_call_and_return_conditional_losses_217319Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv2_layer_call_fn_217348Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv2_layer_call_and_return_conditional_losses_217339Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_relu2_layer_call_fn_217358Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_relu2_layer_call_and_return_conditional_losses_217353Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv3_layer_call_fn_217382Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv3_layer_call_and_return_conditional_losses_217373Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_shortcut_layer_call_fn_217406Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_shortcut_layer_call_and_return_conditional_losses_217397Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_out_block_layer_call_fn_217416Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_out_block_layer_call_and_return_conditional_losses_217411Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv1_layer_call_fn_217440Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv1_layer_call_and_return_conditional_losses_217431Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_relu1_layer_call_fn_217450Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_relu1_layer_call_and_return_conditional_losses_217445Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv2_layer_call_fn_217474Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv2_layer_call_and_return_conditional_losses_217465Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_relu2_layer_call_fn_217484Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_relu2_layer_call_and_return_conditional_losses_217479Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv3_layer_call_fn_217508Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv3_layer_call_and_return_conditional_losses_217499Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_shortcut_layer_call_fn_217532Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_shortcut_layer_call_and_return_conditional_losses_217523Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_out_block_layer_call_fn_217542Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_out_block_layer_call_and_return_conditional_losses_217537Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv1_layer_call_fn_217566Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv1_layer_call_and_return_conditional_losses_217557Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_relu1_layer_call_fn_217576Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_relu1_layer_call_and_return_conditional_losses_217571Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv2_layer_call_fn_217600Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv2_layer_call_and_return_conditional_losses_217591Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_relu2_layer_call_fn_217610Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_relu2_layer_call_and_return_conditional_losses_217605Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv3_layer_call_fn_217634Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv3_layer_call_and_return_conditional_losses_217625Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_shortcut_layer_call_fn_217658Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_shortcut_layer_call_and_return_conditional_losses_217649Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_out_block_layer_call_fn_217668Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_out_block_layer_call_and_return_conditional_losses_217663Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv1_layer_call_fn_217692Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv1_layer_call_and_return_conditional_losses_217683Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_relu1_layer_call_fn_217702Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_relu1_layer_call_and_return_conditional_losses_217697Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv2_layer_call_fn_217726Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv2_layer_call_and_return_conditional_losses_217717Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_relu2_layer_call_fn_217736Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_relu2_layer_call_and_return_conditional_losses_217731Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv3_layer_call_fn_217760Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv3_layer_call_and_return_conditional_losses_217751Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_shortcut_layer_call_fn_217784Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_shortcut_layer_call_and_return_conditional_losses_217775Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_out_block_layer_call_fn_217794Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_out_block_layer_call_and_return_conditional_losses_217789Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
	J
ConstО
!__inference__wrapped_model_210089*Аbcdefghijklmnopqrstuvwxyz{|}~LMRSXY5Ђ2
+Ђ(
&#
input_1џџџџџџџџџ
Њ "3Њ0
.
output_1"
output_1џџџџџџџџџЉ
A__inference_conv1_layer_call_and_return_conditional_losses_217305dbc3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ ")Ђ&

0џџџџџџџџџ@ 
 Љ
A__inference_conv1_layer_call_and_return_conditional_losses_217431djk3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ ")Ђ&

0џџџџџџџџџ@@
 Њ
A__inference_conv1_layer_call_and_return_conditional_losses_217557ers3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ "*Ђ'
 
0џџџџџџџџџ@
 Ћ
A__inference_conv1_layer_call_and_return_conditional_losses_217683fz{4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџ@
 
&__inference_conv1_layer_call_fn_217314Wbc3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@ 
&__inference_conv1_layer_call_fn_217440Wjk3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ "џџџџџџџџџ@@
&__inference_conv1_layer_call_fn_217566Xrs3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ "џџџџџџџџџ@
&__inference_conv1_layer_call_fn_217692Yz{4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@Љ
A__inference_conv2_layer_call_and_return_conditional_losses_217339dde3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ ")Ђ&

0џџџџџџџџџ@ 
 Љ
A__inference_conv2_layer_call_and_return_conditional_losses_217465dlm3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ ")Ђ&

0џџџџџџџџџ@@
 Ћ
A__inference_conv2_layer_call_and_return_conditional_losses_217591ftu4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџ@
 Ћ
A__inference_conv2_layer_call_and_return_conditional_losses_217717f|}4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџ@
 
&__inference_conv2_layer_call_fn_217348Wde3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ "џџџџџџџџџ@ 
&__inference_conv2_layer_call_fn_217474Wlm3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ "џџџџџџџџџ@@
&__inference_conv2_layer_call_fn_217600Ytu4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@
&__inference_conv2_layer_call_fn_217726Y|}4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@Љ
A__inference_conv3_layer_call_and_return_conditional_losses_217373dfg3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ ")Ђ&

0џџџџџџџџџ@ 
 Љ
A__inference_conv3_layer_call_and_return_conditional_losses_217499dno3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ ")Ђ&

0џџџџџџџџџ@@
 Ћ
A__inference_conv3_layer_call_and_return_conditional_losses_217625fvw4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџ@
 Ћ
A__inference_conv3_layer_call_and_return_conditional_losses_217751f~4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџ@
 
&__inference_conv3_layer_call_fn_217382Wfg3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ "џџџџџџџџџ@ 
&__inference_conv3_layer_call_fn_217508Wno3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ "џџџџџџџџџ@@
&__inference_conv3_layer_call_fn_217634Yvw4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@
&__inference_conv3_layer_call_fn_217760Y~4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@Ј
A__inference_delta_layer_call_and_return_conditional_losses_216046c2Ђ/
(Ђ%
# 
xџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
&__inference_delta_layer_call_fn_216051V2Ђ/
(Ђ%
# 
xџџџџџџџџџ@
Њ " џџџџџџџџџ@Ё
?__inference_fc1_layer_call_and_return_conditional_losses_217241^LM0Ђ-
&Ђ#
!
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџ
 y
$__inference_fc1_layer_call_fn_217250QLM0Ђ-
&Ђ#
!
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЁ
?__inference_fc2_layer_call_and_return_conditional_losses_217261^RS0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 y
$__inference_fc2_layer_call_fn_217270QRS0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ 
?__inference_fc3_layer_call_and_return_conditional_losses_217281]XY0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 x
$__inference_fc3_layer_call_fn_217290PXY0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЅ
C__inference_flatten_layer_call_and_return_conditional_losses_217225^4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџ@
 }
(__inference_flatten_layer_call_fn_217230Q4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@Л
O__inference_log_mel_spectrogram_layer_call_and_return_conditional_losses_216019hА3Ђ0
)Ђ&
$!
	waveformsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
4__inference_log_mel_spectrogram_layer_call_fn_216026[А3Ђ0
)Ђ&
$!
	waveformsџџџџџџџџџ
Њ " џџџџџџџџџ@Љ
E__inference_out_block_layer_call_and_return_conditional_losses_217411`3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ ")Ђ&

0џџџџџџџџџ@ 
 Љ
E__inference_out_block_layer_call_and_return_conditional_losses_217537`3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ ")Ђ&

0џџџџџџџџџ@@
 Ћ
E__inference_out_block_layer_call_and_return_conditional_losses_217663b4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџ@
 Ћ
E__inference_out_block_layer_call_and_return_conditional_losses_217789b4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџ@
 
*__inference_out_block_layer_call_fn_217416S3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ "џџџџџџџџџ@ 
*__inference_out_block_layer_call_fn_217542S3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ "џџџџџџџџџ@@
*__inference_out_block_layer_call_fn_217668U4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@
*__inference_out_block_layer_call_fn_217794U4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@Ѕ
A__inference_relu1_layer_call_and_return_conditional_losses_217319`3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ ")Ђ&

0џџџџџџџџџ@ 
 Ѕ
A__inference_relu1_layer_call_and_return_conditional_losses_217445`3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ ")Ђ&

0џџџџџџџџџ@@
 Ї
A__inference_relu1_layer_call_and_return_conditional_losses_217571b4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџ@
 Ї
A__inference_relu1_layer_call_and_return_conditional_losses_217697b4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџ@
 }
&__inference_relu1_layer_call_fn_217324S3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ "џџџџџџџџџ@ }
&__inference_relu1_layer_call_fn_217450S3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ "џџџџџџџџџ@@
&__inference_relu1_layer_call_fn_217576U4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@
&__inference_relu1_layer_call_fn_217702U4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@Ѕ
A__inference_relu2_layer_call_and_return_conditional_losses_217353`3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ ")Ђ&

0џџџџџџџџџ@ 
 Ѕ
A__inference_relu2_layer_call_and_return_conditional_losses_217479`3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ ")Ђ&

0џџџџџџџџџ@@
 Ї
A__inference_relu2_layer_call_and_return_conditional_losses_217605b4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџ@
 Ї
A__inference_relu2_layer_call_and_return_conditional_losses_217731b4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџ@
 }
&__inference_relu2_layer_call_fn_217358S3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ "џџџџџџџџџ@ }
&__inference_relu2_layer_call_fn_217484S3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ "џџџџџџџџџ@@
&__inference_relu2_layer_call_fn_217610U4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@
&__inference_relu2_layer_call_fn_217736U4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@ж
C__inference_res_net_layer_call_and_return_conditional_losses_214135*Аbcdefghijklmnopqrstuvwxyz{|}~LMRSXY9Ђ6
/Ђ,
&#
input_1џџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 ж
C__inference_res_net_layer_call_and_return_conditional_losses_214596*Аbcdefghijklmnopqrstuvwxyz{|}~LMRSXY9Ђ6
/Ђ,
&#
input_1џџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 е
C__inference_res_net_layer_call_and_return_conditional_losses_215223*Аbcdefghijklmnopqrstuvwxyz{|}~LMRSXY8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 е
C__inference_res_net_layer_call_and_return_conditional_losses_215684*Аbcdefghijklmnopqrstuvwxyz{|}~LMRSXY8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 Ў
(__inference_res_net_layer_call_fn_214679*Аbcdefghijklmnopqrstuvwxyz{|}~LMRSXY9Ђ6
/Ђ,
&#
input_1џџџџџџџџџ
p
Њ "џџџџџџџџџЎ
(__inference_res_net_layer_call_fn_214762*Аbcdefghijklmnopqrstuvwxyz{|}~LMRSXY9Ђ6
/Ђ,
&#
input_1џџџџџџџџџ
p 
Њ "џџџџџџџџџ­
(__inference_res_net_layer_call_fn_215767*Аbcdefghijklmnopqrstuvwxyz{|}~LMRSXY8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ­
(__inference_res_net_layer_call_fn_215850*Аbcdefghijklmnopqrstuvwxyz{|}~LMRSXY8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџС
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_216395sjklmnopq<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ@ 
p

 
Њ ")Ђ&

0џџџџџџџџџ@@
 С
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_216447sjklmnopq<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ@ 
p 

 
Њ ")Ђ&

0џџџџџџџџџ@@
 Р
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_216541rjklmnopq;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ@ 
p

 
Њ ")Ђ&

0џџџџџџџџџ@@
 Р
J__inference_resnet_block_1_layer_call_and_return_conditional_losses_216593rjklmnopq;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ@ 
p 

 
Њ ")Ђ&

0џџџџџџџџџ@@
 
/__inference_resnet_block_1_layer_call_fn_216468fjklmnopq<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ@ 
p

 
Њ "џџџџџџџџџ@@
/__inference_resnet_block_1_layer_call_fn_216489fjklmnopq<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ@ 
p 

 
Њ "џџџџџџџџџ@@
/__inference_resnet_block_1_layer_call_fn_216614ejklmnopq;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ@ 
p

 
Њ "џџџџџџџџџ@@
/__inference_resnet_block_1_layer_call_fn_216635ejklmnopq;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ@ 
p 

 
Њ "џџџџџџџџџ@@Т
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_216687trstuvwxy<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ@@
p

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 Т
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_216739trstuvwxy<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ@@
p 

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 С
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_216833srstuvwxy;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ@@
p

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 С
J__inference_resnet_block_2_layer_call_and_return_conditional_losses_216885srstuvwxy;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ@@
p 

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 
/__inference_resnet_block_2_layer_call_fn_216760grstuvwxy<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ@@
p

 
Њ "џџџџџџџџџ@
/__inference_resnet_block_2_layer_call_fn_216781grstuvwxy<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ@@
p 

 
Њ "џџџџџџџџџ@
/__inference_resnet_block_2_layer_call_fn_216906frstuvwxy;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ@@
p

 
Њ "џџџџџџџџџ@
/__inference_resnet_block_2_layer_call_fn_216927frstuvwxy;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ@@
p 

 
Њ "џџџџџџџџџ@Х
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_216979w
z{|}~=Ђ:
3Ђ0
&#
input_1џџџџџџџџџ@
p

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 Х
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_217031w
z{|}~=Ђ:
3Ђ0
&#
input_1џџџџџџџџџ@
p 

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 Ф
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_217125v
z{|}~<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ@
p

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 Ф
J__inference_resnet_block_3_layer_call_and_return_conditional_losses_217177v
z{|}~<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ@
p 

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 
/__inference_resnet_block_3_layer_call_fn_217052j
z{|}~=Ђ:
3Ђ0
&#
input_1џџџџџџџџџ@
p

 
Њ "џџџџџџџџџ@
/__inference_resnet_block_3_layer_call_fn_217073j
z{|}~=Ђ:
3Ђ0
&#
input_1џџџџџџџџџ@
p 

 
Њ "џџџџџџџџџ@
/__inference_resnet_block_3_layer_call_fn_217198i
z{|}~<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ@
p

 
Њ "џџџџџџџџџ@
/__inference_resnet_block_3_layer_call_fn_217219i
z{|}~<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ@
p 

 
Њ "џџџџџџџџџ@П
H__inference_resnet_block_layer_call_and_return_conditional_losses_216103sbcdefghi<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ@
p

 
Њ ")Ђ&

0џџџџџџџџџ@ 
 П
H__inference_resnet_block_layer_call_and_return_conditional_losses_216155sbcdefghi<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ@
p 

 
Њ ")Ђ&

0џџџџџџџџџ@ 
 О
H__inference_resnet_block_layer_call_and_return_conditional_losses_216249rbcdefghi;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ@
p

 
Њ ")Ђ&

0џџџџџџџџџ@ 
 О
H__inference_resnet_block_layer_call_and_return_conditional_losses_216301rbcdefghi;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ@
p 

 
Њ ")Ђ&

0џџџџџџџџџ@ 
 
-__inference_resnet_block_layer_call_fn_216176fbcdefghi<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ@
p

 
Њ "џџџџџџџџџ@ 
-__inference_resnet_block_layer_call_fn_216197fbcdefghi<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ@
p 

 
Њ "џџџџџџџџџ@ 
-__inference_resnet_block_layer_call_fn_216322ebcdefghi;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ@
p

 
Њ "џџџџџџџџџ@ 
-__inference_resnet_block_layer_call_fn_216343ebcdefghi;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ@
p 

 
Њ "џџџџџџџџџ@ Ќ
D__inference_shortcut_layer_call_and_return_conditional_losses_217397dhi3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ ")Ђ&

0џџџџџџџџџ@ 
 Ќ
D__inference_shortcut_layer_call_and_return_conditional_losses_217523dpq3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ ")Ђ&

0џџџџџџџџџ@@
 ­
D__inference_shortcut_layer_call_and_return_conditional_losses_217649exy3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ "*Ђ'
 
0џџџџџџџџџ@
 А
D__inference_shortcut_layer_call_and_return_conditional_losses_217775h4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџ@
 
)__inference_shortcut_layer_call_fn_217406Whi3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@ 
)__inference_shortcut_layer_call_fn_217532Wpq3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@ 
Њ "џџџџџџџџџ@@
)__inference_shortcut_layer_call_fn_217658Xxy3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@@
Њ "џџџџџџџџџ@
)__inference_shortcut_layer_call_fn_217784[4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@Ь
$__inference_signature_wrapper_213674Ѓ*Аbcdefghijklmnopqrstuvwxyz{|}~LMRSXY@Ђ=
Ђ 
6Њ3
1
input_1&#
input_1џџџџџџџџџ"3Њ0
.
output_1"
output_1џџџџџџџџџ