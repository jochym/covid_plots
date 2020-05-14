#!/bin/bash


D=`date`
SUM="Bot update $D"

echo $SUM

python Covid_plots.py

for fn in wzrosty_dzienne trajektoria_covid aktywne_wzrost ; do
    pwb.py upload -user:Ptj -family:commons \
        -noverify -ignorewarn -keep \
        -summary:"$SUM" \
        -filename:"${fn}.png" \
        ${fn}.png \
        "Update for $fn"
done
