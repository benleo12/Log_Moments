#!/bin/bash

###############################################################################
###############################################################################
#
# THIS IS A BASH SCRIPT THAT ALLOWS TO RUN PYTHIA TOGETHER WITH MULTIPLE
# INSTANCES OF RIVET. THE MAIN PROGRAM IS CALLED "run" (SEE BOTTOM OF FILE)


###############################################################################
# FUNCTION TO START MULTIPLE RIVET RUNS
run_rivet () {
  until [ -z "$1" ]
  do
    in="$(echo $1 | sed 's|?| |g')"
    $in > /dev/null 2>&1 &
    shift
  done
}

###############################################################################
# FUNCTION TO RUN GENERATOR AND PRODUCE HISTOGRAM FILES
run_generation () {

echo ""
echo "-------------------------------------------------------------------------"
echo "RUN EVENT GENERATION AND ANALYSIS"

# Setup output files and Rivet command.
rivetExecString=""
hepmcfiles=""; yodafiles=""; yodafiles_fsr=""; yodafiles_isr=""

# Compile helper program to extract variation labels. 
make extract-labels
while read hepmcfile ; do
  if [ -e $hepmcfile ] ; then rm $hepmcfile; fi
  mkfifo $hepmcfile
  hepmcfiles+="$hepmcfile "
  yodafile="$(echo $hepmcfile | sed 's/.hepmc/.yoda/g')"
  yodafiles+="$yodafile "
  if [[ "$(echo $yodafile | grep "fsr" | wc -l)" != "0" ]] ; then yodafiles_fsr+="$yodafile " ; fi
  if [[ "$(echo $yodafile | grep "isr" | wc -l)" != "0" ]] ; then yodafiles_isr+="$yodafile " ; fi
  rivetExecString+="rivet?-v?-a?$RIVET_ANALYSIS_NAME?-H?$yodafile?$hepmcfile "
done < <(./extract-labels mymain-hepmc.cmnd)

# Setup generator command.
local generatorExecString="./mymain-hepmc mymain-hepmc.cmnd"

echo "  Running Pythia8 with command:      $generatorExecString"
echo "  Running RIVET with command:        $rivetExecString"

# Begin generation.
$generatorExecString &
run_rivet $rivetExecString

# wait until both runs are finished
local tmp_jobs="jobs.log"
while true ; do jobs -l > $tmp_jobs
  # at least one job finished with error exit code:
  if [[ "$(cat $tmp_jobs | grep "Exit" | wc -l)" != "0" ]] ; then
    echo "ERROR: fail to run Pythia8 or Rivet"
    rm $tmp_jobs $hepmcfiles
    exit 1
  fi
  if [[ "$(cat $tmp_jobs | wc -l)" == "0" ]] ; then
    echo "INFO: Pythia8 and Rivet finished successfully."
    break;
  fi;
  sleep 3
done

echo "INFO: Finished generation, wait for 30s before removing fifo files"
sleep 5s

rm $tmp_jobs $hepmcfiles

echo ""
echo "-------------------------------------------------------------------------"

}

###############################################################################
# FUNCTION TO PLOT MULTIPLE YODA FILES
make_plots () {

 #yodamerge $(find $dir -name "myname*.yoda" | sed ':a;N;$!ba;s/\n/ /g') -o myname-combined.yoda --assume-normalized

 # Get central line.
 central=$(echo $yodafiles | cut -d " " -f1)
 cp $central ${label}0.yoda

 # Produce overall envelope.
 yodaenvelopes -o ${label}.yoda -c $central $yodafiles
 # Produce FSR envelope.
 yodaenvelopes -o ${label}FSR.yoda -c $central $central $yodafiles_fsr
 # Produce ISR envelope.
 yodaenvelopes -o ${label}ISR.yoda -c $central $central $yodafiles_isr

  # Plot results
  rivet-mkhtml -o ${label} \
    ${label}.yoda:ErrorBars=0:ErrorBands=1:ErrorBandColor=red:ErrorBandOpacity=0.3:LineStyle=none:LegendOrder=2:Title="Envelope"\
    ${label}ISR.yoda:ErrorBars=0:ErrorBands=1:ErrorBandColor=blue:ErrorBandOpacity=0.3:LineStyle=none:LegendOrder=2:Title="FSR variation"\
    ${label}FSR.yoda:ErrorBars=0:ErrorBands=1:ErrorBandColor=green:ErrorBandOpacity=0.3:LineStyle=none:LegendOrder=2:Title="ISR variation"

}

###############################################################################
# MAIN PROGRAM
run () {

echo ""
echo "========================================================================="
echo "               THIS IS A RUN SCRIPT FOR PYTHIA+Rivet                     "
echo ""
echo " Run with ./run.sh myLabel"
echo ""
echo " INPUTS: mymain-hepmc.cmnnd: Please update to your liking and run.       "
echo " OUTPUT: Multiple YODA files, one for each weight variation.             "
echo ""

  label=$1

  RIVET_ANALYSIS_NAME="MC_XS?-a?MC_TTBar"

  # run event generation and analysis
  run_generation

  # plot results
  make_plots

echo ""
echo "========================================================================="
echo ""

}

run "$@"
