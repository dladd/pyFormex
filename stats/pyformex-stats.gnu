set terminal png size 640,480
set output "pyformex-stats.png"
set datafile missing '*'
set title "pyFormex history (http://pyformex.org)\nCreated 2013-02-27 08:55:55"
set key top left
#set offsets 0,0.1,0,0
set xdata time
set timefmt "%Y-%m-%d"
set format x "%y\n%m"
set xlabel "Year/Month"
#set ylabel "revision number"
#set yrange [0:1.2]
plot \
  'pyformex-stats.dat' using 1:($5/100) title '100 lines of Python code',\
  'pyformex-stats.dat' using 1:($6/10) title '10 lines of C code',\
  'pyformex-stats.dat' using 1:($11/1000) title '1000 dollars',\
  'pyformex-stats.dat' using 1:2 title 'number of revisions',\
  'pyformex-stats.dat' using 1:($9/0.01) title '0.01 man-years'
