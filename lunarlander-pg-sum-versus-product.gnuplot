set terminal pdf size 6cm,4cm font "Times, 11"
set output 'plot_lunarlander-pg-sum-vs-product.pdf'

set xlabel 'Episodes (discrete Lunar Lander)'
set ylabel 'Cumulative reward'
set format x "%1.1f"
set style fill transparent solid 0.33 noborder
set border back

set key Left reverse bottom left maxrows 3

advisor = "<(python3 avg_stats.py 0 '' out-advisor)"
mul = "<(python3 avg_stats.py 0 '' out-mixed_withLC_fixed*)"
sum = "<(python3 avg_stats.py 0 '' out-lunarlander-sum*)"


plot [0:2000] [-1000:300] \
    advisor using ($1*1000):3:4 with filledcu notitle lc rgb "#EEEEEE", advisor using ($1*1000):2 with lines title 'PG advisor' lc "#9E9E9E", \
    mul using ($1*1000):3:4 with filledcu notitle lc rgb "#E0E0E0", mul using ($1*1000):2 with lines title 'PG/intersection with LC' lc "#616161", \
    sum using ($1*1000+5):3:4 with filledcu notitle lc rgb "#BDBDBD", sum using ($1*1000+5):2 with lines title 'PG/union with LC' lc "#212121"
