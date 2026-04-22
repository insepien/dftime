export PYTHONPATH=$PYTHONPATH:$(pwd)
figaro-hierarchical -i data/event -b "[[0,40],[0,100]]" --se_draws 3 --draws 3 --selfunc selfunc_aq.py