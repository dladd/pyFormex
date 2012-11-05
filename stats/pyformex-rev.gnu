set terminal png size 640,480
set output "pyformex-rev.png"
set datafile missing '*'
set title "pyFormex history (http://pyformex.org)\nCreated 2012-11-04 20:30:36"
set key top left
#set offsets 0,0.1,0,0
set xdata time
set timefmt "%Y-%m-%d"
set format x "%y\n%m"
set xlabel "Year/Month"
set ylabel "revision number"
#set yrange [0:1.2]
plot 'pyformex-rev.list' using 2:1 title 'revisions' with lines linetype 1
