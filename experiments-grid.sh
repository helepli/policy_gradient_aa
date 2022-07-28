
#chmod +x experiments-grid.sh
#Experiments pg on small very small grid
export OMP_NUM_THREADS=1
# making advisors
taskset 0xAA python3 pg.py --env TransferA-v0 --episodes 2000 --name good-advisor-grid --save good-advisor-grid
taskset 0xAA python3 pg.py --env TransferA-v0 --episodes 2000 --name bad-advisor-grid --bad 1 --save good_advisor-grid
# making divised with good and bad advisors either with or without learnign correction

taskset 0xAA python3 pg.py --env TransferA-v0 --episodes 2000 --name good-advisor-grid-lc-intersection --advisor good_advisor-grid --lc 1 --intersection 1
taskset 0xAA python3 pg.py --env TransferA-v0 --episodes 2000 --name good-advisor-grid-nolc-intersection --advisor good_advisor-grid --lc 0 --intersection 1
taskset 0xAA python3 pg.py --env TransferA-v0 --episodes 2000 --name bad-advisor-grid-lc-intersection --advisor good_advisor-grid --lc 1 --intersection 1
taskset 0xAA python3 pg.py --env TransferA-v0 --episodes 2000 --name bad-advisor-grid-nolc-intersection --advisor good_advisor-grid --lc 0 --intersection 1




