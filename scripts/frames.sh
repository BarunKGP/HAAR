#!/bin/bash
FILE_DIR=/coc/scratch/bdas31/EPIC-Kitchens_100/2g1n6qdydwa9u22shpxqzp0t8m/$1
echo "file directory: ${FILE_DIR}"
for f in $FILE_DIR/*.tar; 
do 
bn=$(basename $f .tar)
echo $bn
(cd $FILE_DIR && mkdir $bn && tar -xfC $bn "$f")
echo "extracted ${f}"
done