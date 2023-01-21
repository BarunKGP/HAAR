#!/bin/bash
FILE_DIR=$1
echo "file directory: ${FILE_DIR}"
for f in $FILE_DIR/*.tar; 
do 
bn=basename $f .tar
echo "basename: ${bn}"
mkdir $bn
echo "created directory ${bn}"
tar -xfC $bn "$f"
echo "extracted ${f}"
done