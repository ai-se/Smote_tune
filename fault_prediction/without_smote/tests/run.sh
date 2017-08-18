#!/bin/bash

rm -rf ../dump/new/smote/*
rm -rf ../log/smote/*

for VAR in "xalan" "synapse" "xerces" "camel" "ant" "arc" "poi" "ivy" "velocity" "redaktor" "log4j" "prop"
    do
        echo $VAR
        python test.py _test "$VAR" > ../log/smote/"$VAR".log
    done