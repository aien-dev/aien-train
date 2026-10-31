#!/bin/sh
# $1 lr shift  $2 P1  $3 total  $4 tag
P="taskset -c 10-14 nice -n 19 ionice -c3"; B=/tmp/op45w/b; D=/tmp/op45w/d
W=/tmp/op45w/t-$4; rm -rf $W; mkdir -p $W/ck; cd $W
$P $B/oe45core init ck $D/train.ids 45 $2 $3 7 $1 > init.out; echo init=$? >> log
s=10000
while [ $s -le $3 ]; do
 $P $B/oe45core train ck $D/train.ids $s 10000 >> train.out; echo train$s=$? >> log
 $P $B/oe45core evaldev ck $D/dev.ids >> dev.out
 s=$((s+10000))
done
