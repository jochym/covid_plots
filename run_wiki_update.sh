#!/bin/bash
# -*- coding: utf-8 -*-

D=`date`
SUM="Bot update $D"

echo $SUM

declare -A wikifn

wikifn[wzrosty_dzienne]="Dzienne_zmiany_liczby_aktywnych_przypadków_COVID-19"
wikifn[trajektoria_covid]="Trajektoria_epidemii_COVID-19"
wikifn[aktywne_wzrost]="Względne_wzrosty_dzienne_COVID19"

python Covid_plots.py

for fn in wzrosty_dzienne trajektoria_covid aktywne_wzrost ; do
    wfn=${wikifn[$fn]}
    pwb.py upload -user:Ptj -lang:commons -family:commons \
        -noverify -ignorewarn -keep \
        -summary:"$SUM" \
        -filename:"${wfn}.png" \
        ${fn}.png \
        "$SUM"
done
