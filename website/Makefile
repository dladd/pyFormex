# Makefile for the pyformex WEB site
#

HOMEPARTS= home_header.inc home_footer.inc
HTDOCS= developer.html documentation.html install.html links.html screenshots.html usermeeting.html support.html pyformexinuse.html $(HOMEPARTS) pyformex.css favicon.ico
EXAMPLES= examples/*.html
IMAGES= images/*.png images/*.gif images/*.jpg
SCRIPTS= scripts/*.py
RSYNC= rsync -av -e ssh --delete

SERVER= shell.berlios.de
TARGETDIR= ${SERVER}:/home/groups/pyformex/htdocs

.PHONY: html pub all

html: 
	r2w
	rm index.html
#	ln -s pyformex.html index.html

all:	pub


$(HOMEPARTS): pyformex.html
	./split_home

# Transfers the files to BerliOs
pubtest: $(HOMEPARTS)
	$(RSYNC) --exclude src . ${TARGETDIR} -n

pub: $(HOMEPARTS)
	$(RSYNC) $(HTDOCS) ${TARGETDIR}
	$(RSYNC) $(EXAMPLES) ${TARGETDIR}/examples
	$(RSYNC) $(IMAGES) ${TARGETDIR}/images
	$(RSYNC) $(SCRIPTS) ${TARGETDIR}/scripts
	# fix the front page
	ssh ${SERVER} ./get_projhtml
