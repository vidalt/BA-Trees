#!/bin/bash
mkdir ../output
for n in COMPAS-ProPublica FICO HTRU2 Pima-Diabetes Seeds Breast-Cancer-Wisconsin
do
mkdir ../output/${n}
done
for o in {0..2}
do
for n in COMPAS-ProPublica FICO HTRU2 Pima-Diabetes Seeds Breast-Cancer-Wisconsin
do
for u in {1..10}
do
for t in {3..10}
do
./bornAgain ../resources/forests/$n/${n}.RF$u.txt ../output/${n}/${n}.BA$u.O$o.T$t -trees $t -obj $o
done
done
done
done

