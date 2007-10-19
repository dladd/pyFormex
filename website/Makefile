# Makefile for the pyformex WEB site
#

HTDOCS= developer.html documentation.html download.html examples.html \
  examples_geometry.html examples_surface.html index.html links.html \
  screenshots.html pyformex.css favicon.ico home_header.inc home_footer.inc
IMAGES= images/*.png images/*.gif images/*.jpg
DOCDOCS= ../doc/html/*.html
MANDOCS= ../manual/html/*.html ../manual/html/*.css ../manual/html/*.png
MANIMAGES= ../manual/images/*.png
RSYNC= rsync -uvl -e ssh

tar: web.tar.gz

web.tar.gz: $(HTDOCS) $(DOCDOCS) ${MANDOCS}
	rm -f web.tar.gz
	tar cvf web.tar $(HTDOCS) $(DOCDOCS)
	tar rvfh web.tar ${MANDOCS}
	gzip web.tar

#docdoc.inc:
#	@for f in $(DOCDOCS); do echo "<li><a href=\"$$f\">$$(basename $$f)</a></li>"; done > docdoc.inc


# tranfer files to berlios.de
# This one packs everything in a tarball: need to unpack on berlios
public: web.tar.gz
	$(RSYNC) web.tar.gz bverheg@shell.berlios.de:/home/users/bverheg/

# this one transfers directly in the correct place
pub: docdoc.inc
	$(RSYNC) $(HTDOCS) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs
	$(RSYNC) $(IMAGES) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/images
	#$(RSYNC) $(DOCDOCS) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/doc
	#$(RSYNC) $(MANDOCS) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/manual

homepage: pyformex.html

pyformex.html: home_header.inc projnews.cache home_footer.inc 
	cat home_header.inc projnews.cache home_footer.inc > $@