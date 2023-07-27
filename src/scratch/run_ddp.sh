#!/bin/bash
for i in {1..10}
do
    echo "Attempt ${i}/10"
    read -p "Enter cfg args " args
    python src/train.py $args && break || sleep 15;
done
