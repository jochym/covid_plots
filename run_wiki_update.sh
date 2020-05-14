#!/bin/bash
# -*- coding: utf-8 -*-

shap=`cat last_update`

echo "Previous commit: $shap"
echo "Waiting for update of covid data..."

shac=`http GET https://api.github.com/repos/datasets/covid-19/commits/HEAD Accept:application/vnd.github.VERSION.sha`

while [ ${shap}. == ${shac}. ] ; do
    C=`date`
    echo -ne "Last check: $C ; next in 10 min \r"
    sleep 600
    shac=`http GET https://api.github.com/repos/datasets/covid-19/commits/HEAD Accept:application/vnd.github.VERSION.sha`
done

echo -e "Current commit: $shac"
echo ${shac} > last_update
echo "Got updated data"

D=`date`
SUM="Bot update $D"

echo $SUM

declare -A wikifn
declare -A md

wikifn[aktywne_wzrost]="Dzienne_zmiany_liczby_aktywnych_przypadków_COVID-19"
wikifn[trajektoria_covid]="Trajektoria_epidemii_COVID-19"
wikifn[wzrosty_dzienne]="Względne_wzrosty_dzienne_COVID19"

for fn in wzrosty_dzienne trajektoria_covid aktywne_wzrost ; do
    md[$fn]=`md5sum ${fn}.png`
done

python Covid_plots.py

for fn in wzrosty_dzienne trajektoria_covid aktywne_wzrost ; do
    m = `md5sum ${fn}.png`
    if [ ${m}. == ${md[$fn]}. ]; then
        echo "No change in ${fn}. Nothing to do"
    else
        md[$fn]=`md5sum ${fn}.png`
        wfn=${wikifn[$fn]}
        pwb.py upload -user:Ptj -lang:commons -family:commons \
            -noverify -ignorewarn -keep \
            -summary:"$SUM" \
            -filename:"${wfn}.png" \
            ${fn}.png \
            "$SUM"
    fi
done
