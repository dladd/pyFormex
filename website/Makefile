# Makefile for the pyformex WEB site
#

HTDOCS= pyformex.html examples.html *.png *.gif favicon.ico docdoc.inc
DOCDOCS= doc/*.html
MANDOCS= manual/*.html manual/*.png manual/*.png images/*.png
RSYNC= rsync -uv -e ssh

tar: web.tar.gz

web.tar.gz: $(HTDOCS) $(DOCDOCS) ${MANDOCS}
	rm -f web.tar.gz
	tar cvf web.tar $(HTDOCS) $(DOCDOCS)
	tar rvfh web.tar ${MANDOCS}
	gzip web.tar

docdoc.inc:
	@for f in $(DOCDOCS); do echo "<li><a href=\"$$f\">$$(basename $$f)</a></li>"; done > docdoc.inc

# tranfer files to berlios.de
public: web.tar.gz
	$(RSYNC) web.tar.gz bverheg@shell.berlios.de:/home/users/bverheg/

pub: docdoc.inc
	$(RSYNC) $(HTDOCS) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs
	$(RSYNC) $(DOCDOCS) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/doc
