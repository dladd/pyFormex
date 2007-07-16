# Makefile for the pyformex WEB site
#

HTDOCS= pyformex.html examples.html surface.html *.png *.gif *.jpg favicon.ico
DOCDOCS= doc/*.html
MANDOCS= manual/*.html manual/*.png manual/*.png
IMAGES= images/*.png
RSYNC= rsync -uv -e ssh

tar: web.tar.gz

web.tar.gz: $(HTDOCS) $(DOCDOCS) ${MANDOCS}
	rm -f web.tar.gz
	tar cvf web.tar $(HTDOCS) $(DOCDOCS)
	tar rvfh web.tar ${MANDOCS}
	gzip web.tar

#docdoc.inc:
#	@for f in $(DOCDOCS); do echo "<li><a href=\"$$f\">$$(basename $$f)</a></li>"; done > docdoc.inc


# tranfer files to berlios.de
# This one packs everything in a tarball: need to iunpack on berlios
public: web.tar.gz
	$(RSYNC) web.tar.gz bverheg@shell.berlios.de:/home/users/bverheg/

# this one transfers directly in the correct place
pub: docdoc.inc
	$(RSYNC) $(HTDOCS) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs
	$(RSYNC) $(DOCDOCS) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/doc
	$(RSYNC) $(MANDOCS) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/manual
	$(RSYNC) $(IMAGES) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/images
