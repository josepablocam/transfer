i
m
p
o
r
t
 
n
u
m
p
y
 
a
s
 
n
p
 
#
 
N
u
m
P
y
 
i
s
 
t
h
e
 
f
u
n
d
a
m
e
n
t
a
l
 
p
a
c
k
a
g
e
 
f
o
r
 
s
c
i
e
n
t
i
f
i
c
 
c
o
m
p
u
t
i
n
g




i
m
p
o
r
t
 
p
a
n
d
a
s
 
a
s
 
p
d
 
#
 
P
a
n
d
a
s
 
i
s
 
a
n
 
e
a
s
y
-
t
o
-
u
s
e
 
d
a
t
a
 
s
t
r
u
c
t
u
r
e
s
 
a
n
d
 
d
a
t
a
 
a
n
a
l
y
s
i
s
 
t
o
o
l
s


p
d
.
s
e
t
_
o
p
t
i
o
n
(
'
d
i
s
p
l
a
y
.
m
a
x
_
c
o
l
u
m
n
s
'
,
 
N
o
n
e
)
 
#
 
T
o
 
d
i
s
p
l
a
y
 
a
l
l
 
c
o
l
u
m
n
s




i
m
p
o
r
t
 
m
a
t
p
l
o
t
l
i
b
.
p
y
p
l
o
t
 
a
s
 
p
l
t
 
#
 
M
a
t
p
l
o
t
l
i
b
 
i
s
 
a
 
p
y
t
h
o
n
 
2
D
 
p
l
o
t
t
i
n
g
 
l
i
b
r
a
r
y


m
a
t
p
l
o
t
l
i
b
 
i
n
l
i
n
e
 


#
 
A
 
m
a
g
i
c
 
c
o
m
m
a
n
d
 
t
h
a
t
 
t
e
l
l
s
 
m
a
t
p
l
o
t
l
i
b
 
t
o
 
r
e
n
d
e
r
 
f
i
g
u
r
e
s
 
a
s
 
s
t
a
t
i
c
 
i
m
a
g
e
s
 
i
n
 
t
h
e
 
N
o
t
e
b
o
o
k
.




i
m
p
o
r
t
 
s
e
a
b
o
r
n
 
a
s
 
s
n
s
 
#
 
S
e
a
b
o
r
n
 
i
s
 
a
 
v
i
s
u
a
l
i
z
a
t
i
o
n
 
l
i
b
r
a
r
y
 
b
a
s
e
d
 
o
n
 
m
a
t
p
l
o
t
l
i
b
 
(
a
t
t
r
a
c
t
i
v
e
 
s
t
a
t
i
s
t
i
c
a
l
 
g
r
a
p
h
i
c
s
)
.


s
n
s
.
s
e
t
_
s
t
y
l
e
(
'
w
h
i
t
e
g
r
i
d
'
)
 
#
 
O
n
e
 
o
f
 
t
h
e
 
f
i
v
e
 
s
e
a
b
o
r
n
 
t
h
e
m
e
s


i
m
p
o
r
t
 
w
a
r
n
i
n
g
s


w
a
r
n
i
n
g
s
.
f
i
l
t
e
r
w
a
r
n
i
n
g
s
(
'
i
g
n
o
r
e
'
)
 
#
 
T
o
 
i
g
n
o
r
e
 
s
o
m
e
 
o
f
 
s
e
a
b
o
r
n
 
w
a
r
n
i
n
g
 
m
s
g




f
r
o
m
 
s
c
i
p
y
 
i
m
p
o
r
t
 
s
t
a
t
s




f
r
o
m
 
s
k
l
e
a
r
n
 
i
m
p
o
r
t
 
l
i
n
e
a
r
_
m
o
d
e
l
 
#
 
S
c
i
k
i
t
 
l
e
a
r
n
 
l
i
b
r
a
r
y
 
t
h
a
t
 
i
m
p
l
e
m
e
n
t
s
 
g
e
n
e
r
a
l
i
z
e
d
 
l
i
n
e
a
r
 
m
o
d
e
l
s


f
r
o
m
 
s
k
l
e
a
r
n
 
i
m
p
o
r
t
 
n
e
i
g
h
b
o
r
s
 
#
 
p
r
o
v
i
d
e
s
 
f
u
n
c
t
i
o
n
a
l
i
t
y
 
f
o
r
 
u
n
s
u
p
e
r
v
i
s
e
d
 
a
n
d
 
s
u
p
e
r
v
i
s
e
d
 
n
e
i
g
h
b
o
r
s
-
b
a
s
e
d
 
l
e
a
r
n
i
n
g
 
m
e
t
h
o
d
s


f
r
o
m
 
s
k
l
e
a
r
n
.
m
e
t
r
i
c
s
 
i
m
p
o
r
t
 
m
e
a
n
_
s
q
u
a
r
e
d
_
e
r
r
o
r
 
#
 
M
e
a
n
 
s
q
u
a
r
e
d
 
e
r
r
o
r
 
r
e
g
r
e
s
s
i
o
n
 
l
o
s
s


f
r
o
m
 
s
k
l
e
a
r
n
 
i
m
p
o
r
t
 
p
r
e
p
r
o
c
e
s
s
i
n
g
 
#
 
p
r
o
v
i
d
e
s
 
f
u
n
c
t
i
o
n
s
 
a
n
d
 
c
l
a
s
s
e
s
 
t
o
 
c
h
a
n
g
e
 
r
a
w
 
f
e
a
t
u
r
e
 
v
e
c
t
o
r
s




f
r
o
m
 
m
a
t
h
 
i
m
p
o
r
t
 
l
o
g
d
a
t
a
 
=
 
p
d
.
r
e
a
d
_
c
s
v
(
"
.
.
/
i
n
p
u
t
/
k
c
_
h
o
u
s
e
_
d
a
t
a
.
c
s
v
"
,
 
p
a
r
s
e
_
d
a
t
e
s
 
=
 
[
'
d
a
t
e
'
]
)
 
#
 
l
o
a
d
 
t
h
e
 
d
a
t
a
 
i
n
t
o
 
a
 
p
a
n
d
a
s
 
d
a
t
a
f
r
a
m
e


d
a
t
a
.
h
e
a
d
(
2
)
 
#
 
S
h
o
w
 
t
h
e
 
f
i
r
s
t
 
2
 
l
i
n
e
s
d
a
t
a
.
d
r
o
p
(
[
'
i
d
'
,
 
'
d
a
t
e
'
]
,
 
a
x
i
s
 
=
 
1
,
 
i
n
p
l
a
c
e
 
=
 
T
r
u
e
)
d
a
t
a
[
'
b
a
s
e
m
e
n
t
_
p
r
e
s
e
n
t
'
]
 
=
 
d
a
t
a
[
'
s
q
f
t
_
b
a
s
e
m
e
n
t
'
]
.
a
p
p
l
y
(
l
a
m
b
d
a
 
x
:
 
1
 
i
f
 
x
 
>
 
0
 
e
l
s
e
 
0
)
 
#
 
I
n
d
i
c
a
t
e
 
w
h
e
t
h
e
r
 
t
h
e
r
e
 
i
s
 
a
 
b
a
s
e
m
e
n
t
 
o
r
 
n
o
t


d
a
t
a
[
'
r
e
n
o
v
a
t
e
d
'
]
 
=
 
d
a
t
a
[
'
y
r
_
r
e
n
o
v
a
t
e
d
'
]
.
a
p
p
l
y
(
l
a
m
b
d
a
 
x
:
 
1
 
i
f
 
x
 
>
 
0
 
e
l
s
e
 
0
)
 
#
 
1
 
i
f
 
t
h
e
 
h
o
u
s
e
 
h
a
s
 
b
e
e
n
 
r
e
n
o
v
a
t
e
d
c
a
t
e
g
o
r
i
a
l
_
c
o
l
s
 
=
 
[
'
f
l
o
o
r
s
'
,
 
'
v
i
e
w
'
,
 
'
c
o
n
d
i
t
i
o
n
'
,
 
'
g
r
a
d
e
'
]




f
o
r
 
c
c
 
i
n
 
c
a
t
e
g
o
r
i
a
l
_
c
o
l
s
:


 
 
 
 
d
u
m
m
i
e
s
 
=
 
p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
d
a
t
a
[
c
c
]
,
 
d
r
o
p
_
f
i
r
s
t
=
F
a
l
s
e
)


 
 
 
 
d
u
m
m
i
e
s
 
=
 
d
u
m
m
i
e
s
.
a
d
d
_
p
r
e
f
i
x
(
"
{
}
#
"
.
f
o
r
m
a
t
(
c
c
)
)


 
 
 
 
d
a
t
a
.
d
r
o
p
(
c
c
,
 
a
x
i
s
=
1
,
 
i
n
p
l
a
c
e
=
T
r
u
e
)


 
 
 
 
d
a
t
a
 
=
 
d
a
t
a
.
j
o
i
n
(
d
u
m
m
i
e
s
)
d
u
m
m
i
e
s
_
z
i
p
c
o
d
e
s
 
=
 
p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
d
a
t
a
[
'
z
i
p
c
o
d
e
'
]
,
 
d
r
o
p
_
f
i
r
s
t
=
F
a
l
s
e
)


d
u
m
m
i
e
s
_
z
i
p
c
o
d
e
s
.
r
e
s
e
t
_
i
n
d
e
x
(
i
n
p
l
a
c
e
=
T
r
u
e
)


d
u
m
m
i
e
s
_
z
i
p
c
o
d
e
s
 
=
 
d
u
m
m
i
e
s
_
z
i
p
c
o
d
e
s
.
a
d
d
_
p
r
e
f
i
x
(
"
{
}
#
"
.
f
o
r
m
a
t
(
'
z
i
p
c
o
d
e
'
)
)


d
u
m
m
i
e
s
_
z
i
p
c
o
d
e
s
 
=
 
d
u
m
m
i
e
s
_
z
i
p
c
o
d
e
s
[
[
'
z
i
p
c
o
d
e
#
9
8
0
0
4
'
,
'
z
i
p
c
o
d
e
#
9
8
1
0
2
'
,
'
z
i
p
c
o
d
e
#
9
8
1
0
9
'
,
'
z
i
p
c
o
d
e
#
9
8
1
1
2
'
,
'
z
i
p
c
o
d
e
#
9
8
0
3
9
'
,
'
z
i
p
c
o
d
e
#
9
8
0
4
0
'
]
]


d
a
t
a
.
d
r
o
p
(
'
z
i
p
c
o
d
e
'
,
 
a
x
i
s
=
1
,
 
i
n
p
l
a
c
e
=
T
r
u
e
)


d
a
t
a
 
=
 
d
a
t
a
.
j
o
i
n
(
d
u
m
m
i
e
s
_
z
i
p
c
o
d
e
s
)




d
a
t
a
.
d
t
y
p
e
s
f
r
o
m
 
s
k
l
e
a
r
n
.
c
r
o
s
s
_
v
a
l
i
d
a
t
i
o
n
 
i
m
p
o
r
t
 
t
r
a
i
n
_
t
e
s
t
_
s
p
l
i
t


t
r
a
i
n
_
d
a
t
a
,
 
t
e
s
t
_
d
a
t
a
 
=
 
t
r
a
i
n
_
t
e
s
t
_
s
p
l
i
t
(
d
a
t
a
,
 
t
r
a
i
n
_
s
i
z
e
 
=
 
0
.
8
,
 
r
a
n
d
o
m
_
s
t
a
t
e
 
=
 
1
0
)
#
 
A
 
f
u
n
c
t
i
o
n
 
t
h
a
t
 
t
a
k
e
 
o
n
e
 
i
n
p
u
t
 
o
f
 
t
h
e
 
d
a
t
a
s
e
t
 
a
n
d
 
r
e
t
u
r
n
 
t
h
e
 
R
M
S
E
 
(
o
f
 
t
h
e
 
t
e
s
t
 
d
a
t
a
)
,
 
a
n
d
 
t
h
e
 
i
n
t
e
r
c
e
p
t
 
a
n
d
 
c
o
e
f
f
i
c
i
e
n
t


d
e
f
 
s
i
m
p
l
e
_
l
i
n
e
a
r
_
m
o
d
e
l
(
t
r
a
i
n
,
 
t
e
s
t
,
 
i
n
p
u
t
_
f
e
a
t
u
r
e
)
:


 
 
 
 
r
e
g
r
 
=
 
l
i
n
e
a
r
_
m
o
d
e
l
.
L
i
n
e
a
r
R
e
g
r
e
s
s
i
o
n
(
)
 
#
 
C
r
e
a
t
e
 
a
 
l
i
n
e
a
r
 
r
e
g
r
e
s
s
i
o
n
 
o
b
j
e
c
t


 
 
 
 
r
e
g
r
.
f
i
t
(
t
r
a
i
n
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
i
n
p
u
t
_
f
e
a
t
u
r
e
]
)
,
 
t
r
a
i
n
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
)
 
#
 
T
r
a
i
n
 
t
h
e
 
m
o
d
e
l


 
 
 
 
R
M
S
E
 
=
 
m
e
a
n
_
s
q
u
a
r
e
d
_
e
r
r
o
r
(
t
e
s
t
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
r
e
g
r
.
p
r
e
d
i
c
t
(
t
e
s
t
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
i
n
p
u
t
_
f
e
a
t
u
r
e
]
)
)
)
*
*
0
.
5
 
#
 
C
a
l
c
u
l
a
t
e
 
t
h
e
 
R
M
S
E
 
o
n
 
t
e
s
t
 
d
a
t
a


 
 
 
 
r
e
t
u
r
n
 
R
M
S
E
,
 
r
e
g
r
.
i
n
t
e
r
c
e
p
t
_
[
0
]
,
 
r
e
g
r
.
c
o
e
f
_
[
0
]
[
0
]
R
M
S
E
,
 
w
0
,
 
w
1
 
=
 
s
i
m
p
l
e
_
l
i
n
e
a
r
_
m
o
d
e
l
(
t
r
a
i
n
_
d
a
t
a
,
 
t
e
s
t
_
d
a
t
a
,
 
'
s
q
f
t
_
l
i
v
i
n
g
'
)


p
r
i
n
t
 
(
'
R
M
S
E
 
f
o
r
 
s
q
f
t
_
l
i
v
i
n
g
 
i
s
:
 
s
 
'
 
R
M
S
E
)


p
r
i
n
t
 
(
'
i
n
t
e
r
c
e
p
t
 
i
s
:
 
s
'
 
w
0
)


p
r
i
n
t
 
(
'
c
o
e
f
f
i
c
i
e
n
t
 
i
s
:
 
s
'
 
w
1
)
i
n
p
u
t
_
l
i
s
t
 
=
 
d
a
t
a
.
c
o
l
u
m
n
s
.
v
a
l
u
e
s
.
t
o
l
i
s
t
(
)
 
#
 
l
i
s
t
 
o
f
 
c
o
l
u
m
n
 
n
a
m
e


i
n
p
u
t
_
l
i
s
t
.
r
e
m
o
v
e
(
'
p
r
i
c
e
'
)


s
i
m
p
l
e
_
l
i
n
e
a
r
_
r
e
s
u
l
t
 
=
 
p
d
.
D
a
t
a
F
r
a
m
e
(
c
o
l
u
m
n
s
 
=
 
[
'
f
e
a
t
u
r
e
'
,
 
'
R
M
S
E
'
,
 
'
i
n
t
e
r
c
e
p
t
'
,
 
'
c
o
e
f
f
i
c
i
e
n
t
'
]
)




#
 
l
o
o
p
 
t
h
a
t
 
c
a
l
c
u
l
a
t
e
 
t
h
e
 
R
M
S
E
 
o
f
 
t
h
e
 
t
e
s
t
 
d
a
t
a
 
f
o
r
 
e
a
c
h
 
i
n
p
u
t
 


f
o
r
 
p
 
i
n
 
i
n
p
u
t
_
l
i
s
t
:


 
 
 
 
R
M
S
E
,
 
w
1
,
 
w
0
 
=
 
s
i
m
p
l
e
_
l
i
n
e
a
r
_
m
o
d
e
l
(
t
r
a
i
n
_
d
a
t
a
,
 
t
e
s
t
_
d
a
t
a
,
 
p
)


 
 
 
 
s
i
m
p
l
e
_
l
i
n
e
a
r
_
r
e
s
u
l
t
 
=
 
s
i
m
p
l
e
_
l
i
n
e
a
r
_
r
e
s
u
l
t
.
a
p
p
e
n
d
(
{
'
f
e
a
t
u
r
e
'
:
p
,
 
'
R
M
S
E
'
:
R
M
S
E
,
 
'
i
n
t
e
r
c
e
p
t
'
:
w
0
,
 
'
c
o
e
f
f
i
c
i
e
n
t
'
:
 
w
1
}


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
,
i
g
n
o
r
e
_
i
n
d
e
x
=
T
r
u
e
)


s
i
m
p
l
e
_
l
i
n
e
a
r
_
r
e
s
u
l
t
.
s
o
r
t
_
v
a
l
u
e
s
(
'
R
M
S
E
'
)
.
h
e
a
d
(
1
0
)
 
#
 
d
i
s
p
l
a
y
 
t
h
e
 
1
0
 
b
e
s
t
 
e
s
t
i
m
a
t
o
r
s
#
 
A
 
f
u
n
c
t
i
o
n
 
t
h
a
t
 
t
a
k
e
 
m
u
l
t
i
p
l
e
 
f
e
a
t
u
r
e
s
 
a
s
 
i
n
p
u
t
 
a
n
d
 
r
e
t
u
r
n
 
t
h
e
 
R
M
S
E
 
(
o
f
 
t
h
e
 
t
e
s
t
 
d
a
t
a
)
,
 
a
n
d
 
t
h
e
 
 
i
n
t
e
r
c
e
p
t
 
a
n
d
 
c
o
e
f
f
i
c
i
e
n
t
s


d
e
f
 
m
u
l
t
i
p
l
e
_
r
e
g
r
e
s
s
i
o
n
_
m
o
d
e
l
(
t
r
a
i
n
,
 
t
e
s
t
,
 
i
n
p
u
t
_
f
e
a
t
u
r
e
s
)
:


 
 
 
 
r
e
g
r
 
=
 
l
i
n
e
a
r
_
m
o
d
e
l
.
L
i
n
e
a
r
R
e
g
r
e
s
s
i
o
n
(
)
 
#
 
C
r
e
a
t
e
 
a
 
l
i
n
e
a
r
 
r
e
g
r
e
s
s
i
o
n
 
o
b
j
e
c
t


 
 
 
 
r
e
g
r
.
f
i
t
(
t
r
a
i
n
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
i
n
p
u
t
_
f
e
a
t
u
r
e
s
)
,
 
t
r
a
i
n
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
)
 
#
 
T
r
a
i
n
 
t
h
e
 
m
o
d
e
l


 
 
 
 
R
M
S
E
 
=
 
m
e
a
n
_
s
q
u
a
r
e
d
_
e
r
r
o
r
(
t
e
s
t
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
r
e
g
r
.
p
r
e
d
i
c
t
(
t
e
s
t
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
i
n
p
u
t
_
f
e
a
t
u
r
e
s
)
)
)
*
*
0
.
5
 
#
 
C
a
l
c
u
l
a
t
e
 
t
h
e
 
R
M
S
E
 
o
n
 
t
e
s
t
 
d
a
t
a


 
 
 
 
r
e
t
u
r
n
 
R
M
S
E
,
 
r
e
g
r
.
i
n
t
e
r
c
e
p
t
_
[
0
]
,
 
r
e
g
r
.
c
o
e
f
_
 
p
r
i
n
t
 
(
'
R
M
S
E
:
 
s
,
 
i
n
t
e
r
c
e
p
t
:
 
s
,
 
c
o
e
f
f
i
c
i
e
n
t
s
:
 
s
'
 
m
u
l
t
i
p
l
e
_
r
e
g
r
e
s
s
i
o
n
_
m
o
d
e
l
(
t
r
a
i
n
_
d
a
t
a
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
t
e
s
t
_
d
a
t
a
,
 
[
'
s
q
f
t
_
l
i
v
i
n
g
'
,
'
b
a
t
h
r
o
o
m
s
'
,
'
b
e
d
r
o
o
m
s
'
]
)
)


p
r
i
n
t
 
(
'
R
M
S
E
:
 
s
,
 
i
n
t
e
r
c
e
p
t
:
 
s
,
 
c
o
e
f
f
i
c
i
e
n
t
s
:
 
s
'
 
m
u
l
t
i
p
l
e
_
r
e
g
r
e
s
s
i
o
n
_
m
o
d
e
l
(
t
r
a
i
n
_
d
a
t
a
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
t
e
s
t
_
d
a
t
a
,
 
[
'
s
q
f
t
_
a
b
o
v
e
'
,
'
v
i
e
w
#
0
'
,
'
b
a
t
h
r
o
o
m
s
'
]
)
)


p
r
i
n
t
 
(
'
R
M
S
E
:
 
s
,
 
i
n
t
e
r
c
e
p
t
:
 
s
,
 
c
o
e
f
f
i
c
i
e
n
t
s
:
 
s
'
 
m
u
l
t
i
p
l
e
_
r
e
g
r
e
s
s
i
o
n
_
m
o
d
e
l
(
t
r
a
i
n
_
d
a
t
a
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
t
e
s
t
_
d
a
t
a
,
 
[
'
b
a
t
h
r
o
o
m
s
'
,
'
b
e
d
r
o
o
m
s
'
]
)
)


p
r
i
n
t
 
(
'
R
M
S
E
:
 
s
,
 
i
n
t
e
r
c
e
p
t
:
 
s
,
 
c
o
e
f
f
i
c
i
e
n
t
s
:
 
s
'
 
m
u
l
t
i
p
l
e
_
r
e
g
r
e
s
s
i
o
n
_
m
o
d
e
l
(
t
r
a
i
n
_
d
a
t
a
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
t
e
s
t
_
d
a
t
a
,
 
[
'
v
i
e
w
#
0
'
,
'
g
r
a
d
e
#
1
2
'
,
'
b
e
d
r
o
o
m
s
'
,
'
s
q
f
t
_
b
a
s
e
m
e
n
t
'
]
)
)


p
r
i
n
t
 
(
'
R
M
S
E
:
 
s
,
 
i
n
t
e
r
c
e
p
t
:
 
s
,
 
c
o
e
f
f
i
c
i
e
n
t
s
:
 
s
'
 
m
u
l
t
i
p
l
e
_
r
e
g
r
e
s
s
i
o
n
_
m
o
d
e
l
(
t
r
a
i
n
_
d
a
t
a
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
t
e
s
t
_
d
a
t
a
,
 
[
'
s
q
f
t
_
l
i
v
i
n
g
'
,
'
b
a
t
h
r
o
o
m
s
'
,
'
v
i
e
w
#
0
'
]
)
)
t
r
a
i
n
_
d
a
t
a
[
'
s
q
f
t
_
l
i
v
i
n
g
_
s
q
u
a
r
e
d
'
]
 
=
 
t
r
a
i
n
_
d
a
t
a
[
'
s
q
f
t
_
l
i
v
i
n
g
'
]
.
a
p
p
l
y
(
l
a
m
b
d
a
 
x
:
 
x
*
*
2
)
 
#
 
c
r
e
a
t
e
 
a
 
n
e
w
 
c
o
l
u
m
n
 
i
n
 
t
r
a
i
n
_
d
a
t
a


t
e
s
t
_
d
a
t
a
[
'
s
q
f
t
_
l
i
v
i
n
g
_
s
q
u
a
r
e
d
'
]
 
=
 
t
e
s
t
_
d
a
t
a
[
'
s
q
f
t
_
l
i
v
i
n
g
'
]
.
a
p
p
l
y
(
l
a
m
b
d
a
 
x
:
 
x
*
*
2
)
 
#
 
c
r
e
a
t
e
 
a
 
n
e
w
 
c
o
l
u
m
n
 
i
n
 
t
e
s
t
_
d
a
t
a


p
r
i
n
t
 
(
'
R
M
S
E
:
 
s
,
 
i
n
t
e
r
c
e
p
t
:
 
s
,
 
c
o
e
f
f
i
c
i
e
n
t
s
:
 
s
'
 
m
u
l
t
i
p
l
e
_
r
e
g
r
e
s
s
i
o
n
_
m
o
d
e
l
(
t
r
a
i
n
_
d
a
t
a
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
t
e
s
t
_
d
a
t
a
,
 
[
'
s
q
f
t
_
l
i
v
i
n
g
'
,
'
s
q
f
t
_
l
i
v
i
n
g
_
s
q
u
a
r
e
d
'
]
)
)
#
 
w
e
'
r
e
 
f
i
r
s
t
 
g
o
i
n
g
 
t
o
 
a
d
d
 
m
o
r
e
 
f
e
a
t
u
r
e
s
 
i
n
t
o
 
t
h
e
 
d
a
t
a
s
e
t
.




#
 
s
q
f
t
_
l
i
v
i
n
g
 
c
u
b
e
d


t
r
a
i
n
_
d
a
t
a
[
'
s
q
f
t
_
l
i
v
i
n
g
_
c
u
b
e
d
'
]
 
=
 
t
r
a
i
n
_
d
a
t
a
[
'
s
q
f
t
_
l
i
v
i
n
g
'
]
.
a
p
p
l
y
(
l
a
m
b
d
a
 
x
:
 
x
*
*
3
)
 


t
e
s
t
_
d
a
t
a
[
'
s
q
f
t
_
l
i
v
i
n
g
_
c
u
b
e
d
'
]
 
=
 
t
e
s
t
_
d
a
t
a
[
'
s
q
f
t
_
l
i
v
i
n
g
'
]
.
a
p
p
l
y
(
l
a
m
b
d
a
 
x
:
 
x
*
*
3
)
 




#
 
b
e
d
r
o
o
m
s
_
s
q
u
a
r
e
d
:
 
t
h
i
s
 
f
e
a
t
u
r
e
 
w
i
l
l
 
m
o
s
t
l
y
 
a
f
f
e
c
t
 
h
o
u
s
e
s
 
w
i
t
h
 
m
a
n
y
 
b
e
d
r
o
o
m
s
.


t
r
a
i
n
_
d
a
t
a
[
'
b
e
d
r
o
o
m
s
_
s
q
u
a
r
e
d
'
]
 
=
 
t
r
a
i
n
_
d
a
t
a
[
'
b
e
d
r
o
o
m
s
'
]
.
a
p
p
l
y
(
l
a
m
b
d
a
 
x
:
 
x
*
*
2
)
 


t
e
s
t
_
d
a
t
a
[
'
b
e
d
r
o
o
m
s
_
s
q
u
a
r
e
d
'
]
 
=
 
t
e
s
t
_
d
a
t
a
[
'
b
e
d
r
o
o
m
s
'
]
.
a
p
p
l
y
(
l
a
m
b
d
a
 
x
:
 
x
*
*
2
)




#
 
b
e
d
r
o
o
m
s
 
t
i
m
e
s
 
b
a
t
h
r
o
o
m
s
 
g
i
v
e
s
 
w
h
a
t
'
s
 
c
a
l
l
e
d
 
a
n
 
"
i
n
t
e
r
a
c
t
i
o
n
"
 
f
e
a
t
u
r
e
.
 
I
t
 
i
s
 
l
a
r
g
e
 
w
h
e
n
 
b
o
t
h
 
o
f
 
t
h
e
m
 
a
r
e
 
l
a
r
g
e
.


t
r
a
i
n
_
d
a
t
a
[
'
b
e
d
_
b
a
t
h
_
r
o
o
m
s
'
]
 
=
 
t
r
a
i
n
_
d
a
t
a
[
'
b
e
d
r
o
o
m
s
'
]
*
t
r
a
i
n
_
d
a
t
a
[
'
b
a
t
h
r
o
o
m
s
'
]


t
e
s
t
_
d
a
t
a
[
'
b
e
d
_
b
a
t
h
_
r
o
o
m
s
'
]
 
=
 
t
e
s
t
_
d
a
t
a
[
'
b
e
d
r
o
o
m
s
'
]
*
t
e
s
t
_
d
a
t
a
[
'
b
a
t
h
r
o
o
m
s
'
]




#
 
T
a
k
i
n
g
 
t
h
e
 
l
o
g
 
o
f
 
s
q
u
a
r
e
f
e
e
t
 
h
a
s
 
t
h
e
 
e
f
f
e
c
t
 
o
f
 
b
r
i
n
g
i
n
g
 
l
a
r
g
e
 
v
a
l
u
e
s
 
c
l
o
s
e
r
 
t
o
g
e
t
h
e
r
 
a
n
d
 
s
p
r
e
a
d
i
n
g
 
o
u
t
 
s
m
a
l
l
 
v
a
l
u
e
s
.


t
r
a
i
n
_
d
a
t
a
[
'
l
o
g
_
s
q
f
t
_
l
i
v
i
n
g
'
]
 
=
 
t
r
a
i
n
_
d
a
t
a
[
'
s
q
f
t
_
l
i
v
i
n
g
'
]
.
a
p
p
l
y
(
l
a
m
b
d
a
 
x
:
 
l
o
g
(
x
)
)


t
e
s
t
_
d
a
t
a
[
'
l
o
g
_
s
q
f
t
_
l
i
v
i
n
g
'
]
 
=
 
t
e
s
t
_
d
a
t
a
[
'
s
q
f
t
_
l
i
v
i
n
g
'
]
.
a
p
p
l
y
(
l
a
m
b
d
a
 
x
:
 
l
o
g
(
x
)
)




t
r
a
i
n
_
d
a
t
a
.
s
h
a
p
e
#
 
s
p
l
i
t
 
t
h
e
 
t
r
a
i
n
_
d
a
t
a
 
t
o
 
i
n
c
l
u
d
e
 
a
 
v
a
l
i
d
a
t
i
o
n
 
s
e
t
 
(
t
r
a
i
n
_
d
a
t
a
2
 
=
 
6
0
,
 
v
a
l
i
d
a
t
i
o
n
_
d
a
t
a
 
=
 
2
0
,
 
t
e
s
t
_
d
a
t
a
 
=
 
2
0
)


t
r
a
i
n
_
d
a
t
a
_
2
,
 
v
a
l
i
d
a
t
i
o
n
_
d
a
t
a
 
=
 
t
r
a
i
n
_
t
e
s
t
_
s
p
l
i
t
(
t
r
a
i
n
_
d
a
t
a
,
 
t
r
a
i
n
_
s
i
z
e
 
=
 
0
.
7
5
,
 
r
a
n
d
o
m
_
s
t
a
t
e
 
=
 
5
0
)
#
 
A
 
f
u
n
c
t
i
o
n
 
t
h
a
t
 
t
a
k
e
 
m
u
l
t
i
p
l
e
 
f
e
a
t
u
r
e
s
 
a
s
 
i
n
p
u
t
 
a
n
d
 
r
e
t
u
r
n
 
t
h
e
 
R
M
S
E
 
(
o
f
 
t
h
e
 
t
r
a
i
n
 
a
n
d
 
v
a
l
i
d
a
t
i
o
n
 
d
a
t
a
)


d
e
f
 
R
M
S
E
(
t
r
a
i
n
,
 
v
a
l
i
d
a
t
i
o
n
,
 
f
e
a
t
u
r
e
s
,
 
n
e
w
_
i
n
p
u
t
)
:


 
 
 
 
f
e
a
t
u
r
e
s
_
l
i
s
t
 
=
 
l
i
s
t
(
f
e
a
t
u
r
e
s
)


 
 
 
 
f
e
a
t
u
r
e
s
_
l
i
s
t
.
a
p
p
e
n
d
(
n
e
w
_
i
n
p
u
t
)


 
 
 
 
r
e
g
r
 
=
 
l
i
n
e
a
r
_
m
o
d
e
l
.
L
i
n
e
a
r
R
e
g
r
e
s
s
i
o
n
(
)
 
#
 
C
r
e
a
t
e
 
a
 
l
i
n
e
a
r
 
r
e
g
r
e
s
s
i
o
n
 
o
b
j
e
c
t


 
 
 
 
r
e
g
r
.
f
i
t
(
t
r
a
i
n
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
f
e
a
t
u
r
e
s
_
l
i
s
t
)
,
 
t
r
a
i
n
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
)
 
#
 
T
r
a
i
n
 
t
h
e
 
m
o
d
e
l


 
 
 
 
R
M
S
E
_
t
r
a
i
n
 
=
 
m
e
a
n
_
s
q
u
a
r
e
d
_
e
r
r
o
r
(
t
r
a
i
n
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
r
e
g
r
.
p
r
e
d
i
c
t
(
t
r
a
i
n
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
f
e
a
t
u
r
e
s
_
l
i
s
t
)
)
)
*
*
0
.
5
 
#
 
C
a
l
c
u
l
a
t
e
 
t
h
e
 
R
M
S
E
 
o
n
 
t
r
a
i
n
 
d
a
t
a


 
 
 
 
R
M
S
E
_
v
a
l
i
d
a
t
i
o
n
 
=
 
m
e
a
n
_
s
q
u
a
r
e
d
_
e
r
r
o
r
(
v
a
l
i
d
a
t
i
o
n
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
r
e
g
r
.
p
r
e
d
i
c
t
(
v
a
l
i
d
a
t
i
o
n
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
f
e
a
t
u
r
e
s
_
l
i
s
t
)
)
)
*
*
0
.
5
 
#
 
C
a
l
c
u
l
a
t
e
 
t
h
e
 
R
M
S
E
 
o
n
 
t
r
a
i
n
 
d
a
t
a


 
 
 
 
r
e
t
u
r
n
 
R
M
S
E
_
t
r
a
i
n
,
 
R
M
S
E
_
v
a
l
i
d
a
t
i
o
n
 
i
n
p
u
t
_
l
i
s
t
 
=
 
t
r
a
i
n
_
d
a
t
a
_
2
.
c
o
l
u
m
n
s
.
v
a
l
u
e
s
.
t
o
l
i
s
t
(
)
 
#
 
l
i
s
t
 
o
f
 
c
o
l
u
m
n
 
n
a
m
e


i
n
p
u
t
_
l
i
s
t
.
r
e
m
o
v
e
(
'
p
r
i
c
e
'
)




#
 
l
i
s
t
 
o
f
 
f
e
a
t
u
r
e
s
 
i
n
c
l
u
d
e
d
 
i
n
 
t
h
e
 
r
e
g
r
e
s
s
i
o
n
 
m
o
d
e
l
 
a
n
d
 
t
h
e
 
c
a
l
c
u
l
a
t
e
d
 
t
r
a
i
n
 
a
n
d
 
v
a
l
i
d
a
t
i
o
n
 
e
r
r
o
r
s
 
(
R
M
S
E
)


r
e
g
r
e
s
s
i
o
n
_
g
r
e
e
d
y
_
a
l
g
o
r
i
t
h
m
 
=
 
p
d
.
D
a
t
a
F
r
a
m
e
(
c
o
l
u
m
n
s
 
=
 
[
'
f
e
a
t
u
r
e
'
,
 
'
t
r
a
i
n
_
e
r
r
o
r
'
,
 
'
v
a
l
i
d
a
t
i
o
n
_
e
r
r
o
r
'
]
)
 
 


i
 
=
 
0


t
e
m
p
_
l
i
s
t
 
=
 
[
]




#
 
a
 
w
h
i
l
e
 
l
o
o
p
 
g
o
i
n
g
 
t
h
r
o
u
g
h
 
a
l
l
 
t
h
e
 
f
e
a
t
u
r
e
s
 
i
n
 
t
h
e
 
d
a
t
a
f
r
a
m
e


w
h
i
l
e
 
i
 
<
 
l
e
n
(
t
r
a
i
n
_
d
a
t
a
_
2
.
c
o
l
u
m
n
s
)
-
1
:


 
 
 
 


 
 
 
 
#
 
a
 
t
e
m
p
o
r
a
r
y
 
d
a
t
a
f
r
a
m
e
 
t
o
 
s
e
l
e
c
t
 
t
h
e
 
b
e
s
t
 
f
e
a
t
u
r
e
 
a
t
 
e
a
c
h
 
i
t
e
r
a
t
i
o
n


 
 
 
 
t
e
m
p
 
=
 
p
d
.
D
a
t
a
F
r
a
m
e
(
c
o
l
u
m
n
s
 
=
 
[
'
f
e
a
t
u
r
e
'
,
 
'
t
r
a
i
n
_
e
r
r
o
r
'
,
 
'
v
a
l
i
d
a
t
i
o
n
_
e
r
r
o
r
'
]
)


 
 
 
 


 
 
 
 
#
 
a
 
f
o
r
 
l
o
o
p
 
t
o
 
t
e
s
t
 
a
l
l
 
t
h
e
 
r
e
m
a
i
n
i
n
g
 
f
e
a
t
u
r
e
s


 
 
 
 
f
o
r
 
p
 
i
n
 
i
n
p
u
t
_
l
i
s
t
:


 
 
 
 
 
 
 
 
R
M
S
E
_
t
r
a
i
n
,
 
R
M
S
E
_
v
a
l
i
d
a
t
i
o
n
 
=
 
R
M
S
E
(
t
r
a
i
n
_
d
a
t
a
_
2
,
 
v
a
l
i
d
a
t
i
o
n
_
d
a
t
a
,
 
t
e
m
p
_
l
i
s
t
,
 
p
)


 
 
 
 
 
 
 
 
t
e
m
p
 
=
 
t
e
m
p
.
a
p
p
e
n
d
(
{
'
f
e
a
t
u
r
e
'
:
p
,
 
'
t
r
a
i
n
_
e
r
r
o
r
'
:
R
M
S
E
_
t
r
a
i
n
,
 
'
v
a
l
i
d
a
t
i
o
n
_
e
r
r
o
r
'
:
R
M
S
E
_
v
a
l
i
d
a
t
i
o
n
}
,
 
i
g
n
o
r
e
_
i
n
d
e
x
=
T
r
u
e
)


 
 
 
 
 
 
 
 


 
 
 
 
t
e
m
p
 
=
 
t
e
m
p
.
s
o
r
t
_
v
a
l
u
e
s
(
'
t
r
a
i
n
_
e
r
r
o
r
'
)
 
#
 
s
e
l
e
c
t
 
t
h
e
 
b
e
s
t
 
f
e
a
t
u
r
e
 
u
s
i
n
g
 
t
r
a
i
n
 
e
r
r
o
r


 
 
 
 
b
e
s
t
 
=
 
t
e
m
p
.
i
l
o
c
[
0
,
0
]


 
 
 
 
t
e
m
p
_
l
i
s
t
.
a
p
p
e
n
d
(
b
e
s
t
)


 
 
 
 
r
e
g
r
e
s
s
i
o
n
_
g
r
e
e
d
y
_
a
l
g
o
r
i
t
h
m
 
=
 
r
e
g
r
e
s
s
i
o
n
_
g
r
e
e
d
y
_
a
l
g
o
r
i
t
h
m
.
a
p
p
e
n
d
(
{
'
f
e
a
t
u
r
e
'
:
 
b
e
s
t
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
t
r
a
i
n
_
e
r
r
o
r
'
:
 
t
e
m
p
.
i
l
o
c
[
0
,
1
]
,
 
'
v
a
l
i
d
a
t
i
o
n
_
e
r
r
o
r
'
:
 
t
e
m
p
.
i
l
o
c
[
0
,
2
]
}
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
i
g
n
o
r
e
_
i
n
d
e
x
=
T
r
u
e
)
 
#
 
a
d
d
 
t
h
e
 
f
e
a
t
u
r
e
 
t
o
 
t
h
e
 
d
a
t
a
f
r
a
m
e


 
 
 
 
i
n
p
u
t
_
l
i
s
t
.
r
e
m
o
v
e
(
b
e
s
t
)
 
#
 
r
e
m
o
v
e
 
t
h
e
 
b
e
s
t
 
f
e
a
t
u
r
e
 
f
r
o
m
 
t
h
e
 
l
i
s
t
 
o
f
 
a
v
a
i
l
a
b
l
e
 
f
e
a
t
u
r
e
s


 
 
 
 
i
 
+
=
 
1


r
e
g
r
e
s
s
i
o
n
_
g
r
e
e
d
y
_
a
l
g
o
r
i
t
h
m
g
r
e
e
d
y
_
a
l
g
o
_
f
e
a
t
u
r
e
s
_
l
i
s
t
 
=
 
r
e
g
r
e
s
s
i
o
n
_
g
r
e
e
d
y
_
a
l
g
o
r
i
t
h
m
[
'
f
e
a
t
u
r
e
'
]
.
t
o
l
i
s
t
(
)
[
:
2
4
]
 
#
 
s
e
l
e
c
t
 
t
h
e
 
f
i
r
s
t
 
3
0
 
f
e
a
t
u
r
e
s


t
e
s
t
_
e
r
r
o
r
,
 
_
,
 
_
 
=
 
m
u
l
t
i
p
l
e
_
r
e
g
r
e
s
s
i
o
n
_
m
o
d
e
l
(
t
r
a
i
n
_
d
a
t
a
_
2
,
 
t
e
s
t
_
d
a
t
a
,
 
g
r
e
e
d
y
_
a
l
g
o
_
f
e
a
t
u
r
e
s
_
l
i
s
t
)


p
r
i
n
t
 
(
'
t
e
s
t
 
e
r
r
o
r
 
(
R
M
S
E
)
 
i
s
:
 
s
'
 
t
e
s
t
_
e
r
r
o
r
)
i
n
p
u
t
_
f
e
a
t
u
r
e
 
=
 
t
r
a
i
n
_
d
a
t
a
.
c
o
l
u
m
n
s
.
v
a
l
u
e
s
.
t
o
l
i
s
t
(
)
 
#
 
l
i
s
t
 
o
f
 
c
o
l
u
m
n
 
n
a
m
e


i
n
p
u
t
_
f
e
a
t
u
r
e
.
r
e
m
o
v
e
(
'
p
r
i
c
e
'
)




f
o
r
 
i
 
i
n
 
[
1
,
1
0
]
:


 
 
 
 
r
i
d
g
e
 
=
 
l
i
n
e
a
r
_
m
o
d
e
l
.
R
i
d
g
e
(
a
l
p
h
a
 
=
 
i
,
 
n
o
r
m
a
l
i
z
e
 
=
 
T
r
u
e
)
 
#
 
i
n
i
t
i
a
l
i
z
e
 
t
h
e
 
m
o
d
e
l


 
 
 
 
r
i
d
g
e
.
f
i
t
(
t
r
a
i
n
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
i
n
p
u
t
_
f
e
a
t
u
r
e
)
,
 
t
r
a
i
n
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
)
 
#
 
f
i
t
 
t
h
e
 
t
r
a
i
n
 
d
a
t
a


 
 
 
 
p
r
i
n
t
 
(
'
t
e
s
t
 
e
r
r
o
r
 
(
R
M
S
E
)
 
i
s
:
 
s
'
 
m
e
a
n
_
s
q
u
a
r
e
d
_
e
r
r
o
r
(
t
e
s
t
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
r
i
d
g
e
.
p
r
e
d
i
c
t
(
t
e
s
t
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
i
n
p
u
t
_
f
e
a
t
u
r
e
]
)
)
)
*
*
0
.
5
)
 
#
 
p
r
e
d
i
c
t
 
p
r
i
c
e
 
a
n
d
 
t
e
s
t
 
e
r
r
o
r
r
i
d
g
e
C
V
 
=
 
l
i
n
e
a
r
_
m
o
d
e
l
.
R
i
d
g
e
C
V
(
a
l
p
h
a
s
 
=
 
n
p
.
l
i
n
s
p
a
c
e
(
1
.
0
e
-
1
0
,
1
,
n
u
m
 
=
 
1
0
0
)
,
 
n
o
r
m
a
l
i
z
e
 
=
 
T
r
u
e
,
 
s
t
o
r
e
_
c
v
_
v
a
l
u
e
s
 
=
 
T
r
u
e
)
 
#
 
i
n
i
t
i
a
l
i
z
e
 
t
h
e
 
m
o
d
e
l


r
i
d
g
e
C
V
.
f
i
t
(
t
r
a
i
n
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
i
n
p
u
t
_
f
e
a
t
u
r
e
)
,
 
t
r
a
i
n
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
)
 
#
 
f
i
t
 
t
h
e
 
t
r
a
i
n
 
d
a
t
a


p
r
i
n
t
 
(
'
b
e
s
t
 
a
l
p
h
a
 
i
s
:
 
s
'
 
r
i
d
g
e
C
V
.
a
l
p
h
a
_
)
 
#
 
g
e
t
 
t
h
e
 
b
e
s
t
 
a
l
p
h
a


p
r
i
n
t
 
(
'
t
e
s
t
 
e
r
r
o
r
 
(
R
M
S
E
)
 
i
s
:
 
s
'
 
m
e
a
n
_
s
q
u
a
r
e
d
_
e
r
r
o
r
(
t
e
s
t
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
r
i
d
g
e
C
V
.
p
r
e
d
i
c
t
(
t
e
s
t
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
i
n
p
u
t
_
f
e
a
t
u
r
e
]
)
)
)
*
*
0
.
5
)
 
#
 
p
r
e
d
i
c
t
 
p
r
i
c
e
 
a
n
d
 
t
e
s
t
 
e
r
r
o
r
f
o
r
 
i
 
i
n
 
[
0
.
0
1
,
0
.
1
,
1
,
2
5
0
,
5
0
0
,
1
0
0
0
]
:


 
 
 
 
l
a
s
s
o
 
=
 
l
i
n
e
a
r
_
m
o
d
e
l
.
L
a
s
s
o
(
a
l
p
h
a
 
=
 
i
,
 
n
o
r
m
a
l
i
z
e
 
=
 
T
r
u
e
)
 
#
 
i
n
i
t
i
a
l
i
z
e
 
t
h
e
 
m
o
d
e
l


 
 
 
 
l
a
s
s
o
.
f
i
t
(
t
r
a
i
n
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
i
n
p
u
t
_
f
e
a
t
u
r
e
)
,
 
t
r
a
i
n
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
)
 
#
 
f
i
t
 
t
h
e
 
t
r
a
i
n
 
d
a
t
a


 
 
 
 
p
r
i
n
t
 
(
l
a
s
s
o
.
s
p
a
r
s
e
_
c
o
e
f
_
.
g
e
t
n
n
z
)
 
#
 
n
u
m
b
e
r
 
o
f
 
n
o
n
 
z
e
r
o
 
w
e
i
g
h
t
s


 
 
 
 
p
r
i
n
t
 
(
'
t
e
s
t
 
e
r
r
o
r
 
(
R
M
S
E
)
 
i
s
:
 
s
'
 
m
e
a
n
_
s
q
u
a
r
e
d
_
e
r
r
o
r
(
t
e
s
t
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
a
s
s
o
.
p
r
e
d
i
c
t
(
t
e
s
t
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
i
n
p
u
t
_
f
e
a
t
u
r
e
]
)
)
)
*
*
0
.
5
)
 
#
 
p
r
e
d
i
c
t
 
p
r
i
c
e
 
a
n
d
 
t
e
s
t
 
e
r
r
o
r
l
a
s
s
o
C
V
 
=
 
l
i
n
e
a
r
_
m
o
d
e
l
.
L
a
s
s
o
C
V
(
n
o
r
m
a
l
i
z
e
 
=
 
T
r
u
e
)
 
#
 
i
n
i
t
i
a
l
i
z
e
 
t
h
e
 
m
o
d
e
l
 
(
a
l
p
h
a
s
 
a
r
e
 
s
e
t
 
a
u
t
o
m
a
t
i
c
a
l
l
y
)


l
a
s
s
o
C
V
.
f
i
t
(
t
r
a
i
n
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
i
n
p
u
t
_
f
e
a
t
u
r
e
)
,
 
n
p
.
r
a
v
e
l
(
t
r
a
i
n
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
)
)
 
#
 
f
i
t
 
t
h
e
 
t
r
a
i
n
 
d
a
t
a


p
r
i
n
t
 
(
'
b
e
s
t
 
a
l
p
h
a
 
i
s
:
 
s
'
 
l
a
s
s
o
C
V
.
a
l
p
h
a
_
)
 
#
 
g
e
t
 
t
h
e
 
b
e
s
t
 
a
l
p
h
a


p
r
i
n
t
 
(
'
n
u
m
b
e
r
 
o
f
 
n
o
n
 
z
e
r
o
 
w
e
i
g
t
h
s
 
i
s
:
 
s
'
 
n
p
.
c
o
u
n
t
_
n
o
n
z
e
r
o
(
l
a
s
s
o
C
V
.
c
o
e
f
_
)
)
 
#
 
n
u
m
b
e
r
 
o
f
 
n
o
n
 
z
e
r
o
 
w
e
i
g
h
t
s


p
r
i
n
t
 
(
'
t
e
s
t
 
e
r
r
o
r
 
(
R
M
S
E
)
 
i
s
:
 
s
'
 
m
e
a
n
_
s
q
u
a
r
e
d
_
e
r
r
o
r
(
t
e
s
t
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
a
s
s
o
C
V
.
p
r
e
d
i
c
t
(
t
e
s
t
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
i
n
p
u
t
_
f
e
a
t
u
r
e
]
)
)
)
*
*
0
.
5
)
 
#
 
p
r
e
d
i
c
t
 
p
r
i
c
e
 
a
n
d
 
t
e
s
t
 
e
r
r
o
r
#
 
n
o
r
m
a
l
i
z
e
 
t
h
e
 
d
a
t
a


t
r
a
i
n
_
X
 
=
 
t
r
a
i
n
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
i
n
p
u
t
_
f
e
a
t
u
r
e
)


s
c
a
l
e
r
 
=
 
p
r
e
p
r
o
c
e
s
s
i
n
g
.
S
t
a
n
d
a
r
d
S
c
a
l
e
r
(
)
.
f
i
t
(
t
r
a
i
n
_
X
)


t
r
a
i
n
_
X
_
s
c
a
l
e
d
 
=
 
s
c
a
l
e
r
.
t
r
a
n
s
f
o
r
m
(
t
r
a
i
n
_
X
)


t
e
s
t
_
X
 
=
 
t
e
s
t
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
i
n
p
u
t
_
f
e
a
t
u
r
e
]
)


t
e
s
t
_
X
_
s
c
a
l
e
d
 
=
 
s
c
a
l
e
r
.
t
r
a
n
s
f
o
r
m
(
t
e
s
t
_
X
)




k
n
n
 
=
 
n
e
i
g
h
b
o
r
s
.
K
N
e
i
g
h
b
o
r
s
R
e
g
r
e
s
s
o
r
(
n
_
n
e
i
g
h
b
o
r
s
=
1
0
,
 
w
e
i
g
h
t
s
=
'
d
i
s
t
a
n
c
e
'
)
 
#
 
i
n
i
t
i
a
l
i
z
e
 
t
h
e
 
m
o
d
e
l


k
n
n
.
f
i
t
(
t
r
a
i
n
_
X
_
s
c
a
l
e
d
,
 
t
r
a
i
n
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
)
 
#
 
f
i
t
 
t
h
e
 
t
r
a
i
n
 
d
a
t
a


p
r
i
n
t
 
(
'
t
e
s
t
 
e
r
r
o
r
 
(
R
M
S
E
)
 
i
s
:
 
s
'
 
m
e
a
n
_
s
q
u
a
r
e
d
_
e
r
r
o
r
(
t
e
s
t
_
d
a
t
a
.
a
s
_
m
a
t
r
i
x
(
c
o
l
u
m
n
s
 
=
 
[
'
p
r
i
c
e
'
]
)
,
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
k
n
n
.
p
r
e
d
i
c
t
(
t
e
s
t
_
X
_
s
c
a
l
e
d
)
)
*
*
0
.
5
)
 
#
 
p
r
e
d
i
c
t
 
p
r
i
c
e
 
a
n
d
 
t
e
s
t
 
e
r
r
o
r
