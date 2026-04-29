cd /home/insepien/dftime
export PYTHONPATH=$PYTHONPATH:$(pwd)
# generate event posteriors
# python script/gen_events.py --outdir "/home/insepien/dftime_data/events" --Nobs 100
# run hierarchical
figaro-hierarchical -i /home/insepien/dftime_data/events -b "[[0,40],[0,100]]" --se_draws 3 --draws 1 --selfunc script/selfunc_aq.py