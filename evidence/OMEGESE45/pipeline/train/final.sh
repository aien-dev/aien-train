#!/bin/sh
# $1 tag. Full training: seed 45, lr 2^-7, P1 20000, total 400000, order 7, every 20000.
P="taskset -c 10-14 nice -n 19 ionice -c3"; B=/tmp/op45w/b2; D=/tmp/op45w/d
W=/tmp/op45w/final/$1; rm -rf $W; mkdir -p $W/ck; cd $W
$P $B/oe45core init ck $D/train.ids 45 20000 400000 7 7 > init.out 2>&1; echo init=$? >> log
cp ck/ck.a step0.ck
$P $B/oe45core info ck > info0.out; echo info0=$? >> log
$P $B/oe45core evaldev ck $D/dev.ids > dev0.out; echo dev0=$? >> log
/usr/bin/time -v $P $B/oe45core train ck $D/train.ids 400000 20000 > train.out 2> time.txt; echo train=$? >> log
$P $B/oe45core info ck > info.out; echo info=$? >> log
$P $B/oe45core evaldev ck $D/dev.ids > dev.out; echo dev=$? >> log
sha256sum step0.ck ck/ck.a ck/ck.b > sha.txt
