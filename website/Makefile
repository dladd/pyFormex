# Makefile for the pyformex WEB site
#

HOMEPARTS= home_header.inc home_footer.inc
HTDOCS= developer.html documentation.html download.html links.html screenshots.html $(HOMEPARTS) pyformex.css favicon.ico
EXAMPLES= examples/*.html
IMAGES= images/*.png images/*.gif images/*.jpg
SCRIPTS= scripts/*.py
DOCDOCS= ../doc/html/*.html
MANDOCS= ../manual/html/*.html ../manual/html/*.css ../manual/html/*.png
MANIMAGES= ../manual/images/*.png
RSYNC= rsync -av -e ssh

.PHONY: html pub all

html: 
	r2w

all:	pub

#tar: web.tar.gz

#web.tar.gz: $(HTDOCS) $(DOCDOCS) ${MANDOCS}
#	rm -f web.tar.gz
#	tar cvf web.tar $(HTDOCS) $(DOCDOCS)
#	tar rvfh web.tar ${MANDOCS}
#	gzip web.tar

#docdoc.inc:
#	@for f in $(DOCDOCS); do echo "<li><a href=\"$$f\">$$(basename $$f)</a></li>"; done > docdoc.inc


# tranfer files to berlios.de
# This one packs everything in a tarball: need to unpack on berlios
#public: web.tar.gz
#	$(RSYNC) web.tar.gz bverheg@shell.berlios.de:/home/users/bverheg/


$(HOMEPARTS): pyformex.html
	./split_home

# Transfers the files to BerliOs
pub: $(HOMEPARTS)
	$(RSYNC) $(HTDOCS) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs
	$(RSYNC) $(EXAMPLES) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/examples
	$(RSYNC) $(IMAGES) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/images
	$(RSYNC) $(SCRIPTS) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/scripts
	#$(RSYNC) $(DOCDOCS) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/doc
	#$(RSYNC) $(MANDOCS) bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/manual
	ssh bverheg@shell.berlios.de ./get_projhtml
