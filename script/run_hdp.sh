export PYTHONPATH=$PYTHONPATH:$(pwd)
figaro-hierarchical -i data/events -b "[[0,40],[0,100]]" --se_draws 3 --draws 1 --selfunc selfunc_aq.py