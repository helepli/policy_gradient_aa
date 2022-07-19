set terminal pdf size 6cm,4cm font "Times, 11"
set output 'plot_lunarlander-pg-disturb.pdf'

set xlabel 'Episodes (discrete Lunar Lander)'
set ylabel 'Cumulative reward'
set format x "%1.1f"
set style fill transparent solid 0.33 noborder
set border back

set key Left reverse top left maxrows 2

with = "<(python3 avg_stats.py 0 '' out-with_0.01epsilon_action1_lunarlander)"
without = "<(python3 avg_stats.py 0 '' out-without_0.01epsilon_action1_lunarlander)"


plot [0:2000] [-400:200] \
    without using ($1*1000):3:4 with filledcu notitle lc rgb "#EEEEEE", without using ($1*1000):2 with lines title 'PG' lc "#9E9E9E", \
    with using ($1*1000):3:4 with filledcu notitle lc rgb "#BDBDBD", with using ($1*1000):2 with lines title 'PG + disturbances' lc "#212121"
