#!/bin/bash

sherpa_merge()
{
    olddir=$(pwd)
    cd $1;
    test -d sum || mkdir sum;
    vars=$(ls | sed -n '/MU/ s/.*\(MU.*MU.*\).yoda/\1/1 p' | sort | uniq);
    for i in ${vars}; do test -f sum/${i}.yoda || ${olddir}/yodamerge -o sum/${i}.yoda *.${i}.yoda; done;
    cd sum
    cent=$(echo ${vars} | sed -n 's/.*MUR2_MUF2_\(PDF[^ \t]*\).*/MUR1_MUF1_\1/g p');
    pdfs=$(echo ${vars} | sed -e 's/[^ \t]*MUR[^1_][^ \t]*//g;s/[^ \t]*MUF[^1_][^ \t]*//g');
    test -f pdfvar.yoda || \
	${olddir}/yodaenvelopes -o pdfvar.yoda -c ${cent}.yoda \
		      $(echo ${pdfs} | sed -e 's/\([^ \t]*\)/\1.yoda/g');
    scls=$(echo ${vars} | sed -e 's/MUR1_MUF1_[^ \t]*//g');
    test -f sclvar.yoda || \
	${olddir}/yodaenvelopes -o sclvar.yoda -c ${cent}.yoda ${cent}.yoda \
		      $(echo ${scls} | sed -e 's/\([^ \t]*\)/\1.yoda/g');
    test -f central.yoda || cp ${cent}.yoda central.yoda
    cd ..
    cd ${olddir}
}
