/^python:/ { print "python="$2 }
/^ansic:/ { print "ansic="$2 }
/^sh:/ { print "sh="$2 }
/^Total.*(SLOC)/ { sub("[^=]*= ",""); sub(",",""); print "sloc="$0 }
/^Devel.*Months)/ { sub("[^=]*= ",""); sub(",",""); print "manyears="$1 }
/^Schedule.*Months)/ { sub("[^=]*= ",""); sub(",",""); print "years="$1 }
/^Total Estimated Cost/ { sub("[^=]*= ",""); sub(",",""); print "dollars="$2 }
