set terminal png size 640,480
set output "pyformex-rev.png"
set datafile missing '*'
set title "pyFormex history (http://pyformex.org)\nCreated 2010-01-08 11:06:05"
set key top left
#set offsets 0,0.1,0,0
set xdata time
set timefmt "%Y-%m-%d"
set format x "%m\n%y"
set xlabel "Month/Year"
set ylabel "revision number"
#set yrange [0:1.2]
plot 'pyformex.revisions' using 2:1 title 'revisions' with lines linetype 1
