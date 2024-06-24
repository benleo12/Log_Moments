#!/bin/bash

. $(dirname $0)/helpers.sh

print_help(){
  echo "usage: $0 [-o <plotpath>] <type> .. <type>"
  echo -e "\npossible plot types are\n"
  echo "  hw     - Herwig prediction, incl. intrinsic kT & UE variations" 
  echo "  py     - Pythia prediction, incl. shower variations" 
  echo "  sh     - Sherpa prediction, incl. ren/fac scale & PDF variations" 
  echo; exit 0
}

plotpath="plots"
while getopts h:o: OPT
do
  case $OPT in
  g) generator=$OPTARG ;;
  o) plotpath=$OPTARG ;;
  h) print_help && exit 0 ;;
  esac
done
shift `expr $OPTIND - 1`
types=$*

for type in $types; do
  echo "$0: plotting '"$type"'"
  if test "$type" = "hw"; then
    plotstr=$plotstr" herwig/Analysis0.yoda:LegendOrder=1:Title=Herwig:ErrorBars=1:LineColor=green"
    plotstr=$plotstr" herwig/AnalysisKT.yoda:LegendOrder=2:Title=$k_{T}$~variation:LineStyle=none:ErrorBands=1:ErrorBandColor=green:ErrorBandOpacity=0.4"
    plotstr=$plotstr" herwig/AnalysisUE.yoda:LegendOrder=3:Title=UE~variation:LineStyle=none:ErrorBands=1:ErrorBandColor=green:ErrorBandOpacity=0.2"
  elif test "$type" = "py"; then
    plotstr=$plotstr" pythia/Analysis0.yoda:LegendOrder=4:Title=Pythia:ErrorBars=1:LineColor=blue"
    plotstr=$plotstr" pythia/AnalysisFSR.yoda:LegendOrder=5:FSR~variation:LineStyle=none:ErrorBands=1:ErrorBandColor=blue:ErrorBandOpacity=0.4"
    plotstr=$plotstr" pythia/AnalysisISR.yoda:LegendOrder=6:ISR~variation:LineStyle=none:ErrorBands=1:ErrorBandColor=blue:ErrorBandOpacity=0.2"
  elif test "$type" = "sh"; then
    sherpa_merge sherpa/Analysis
    plotstr=$plotstr" sherpa/Analysis/sum/central.yoda:LegendOrder=7:Title=Sherpa:ErrorBars=1:LineColor=red"
    plotstr=$plotstr" sherpa/Analysis/sum/sclvar.yoda:LegendOrder=8:Title=$\mu_{R/F}$~variation:LineStyle=none:ErrorBands=1:ErrorBandColor=red:ErrorBandOpacity=0.5"
    plotstr=$plotstr" sherpa/Analysis/sum/pdfvar.yoda:LegendOrder=9:Title=PDF~variation:LineStyle=none:ErrorBands=1:ErrorBandColor=red:ErrorBandOpacity=0.3"
  else
    echo "$0: unknown plot type '$type'"
    print_help && exit 1
  fi
done

./rivet-mkhtml -s -m .*/MC_TTBar/.* -o $plotpath/ $plotstr

echo -e "\n$0: output generated into '$plotpath'\n"
