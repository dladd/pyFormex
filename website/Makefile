# Makefile for the pyformex WEB site
#

HOMEPARTS= home_header.inc home_footer.inc
HTDOCS= documentation.html install.html links.html screenshots.html support.html pyformexinuse.html pyformex.html pyformex.css favicon.ico
EXAMPLES= examples/
IMAGES= images/
SCRIPTS= scripts/
RSYNC= rsync -av -e ssh --delete --exclude .svn

SERVER= shell.berlios.de
TARGETDIR= ${SERVER}:/home/groups/pyformex/htdocs

.PHONY: html pub all

html: 
	r2w
	./split_home
	./update-news


all:	pub


# Transfers the files to BerliOs
pubtest: $(HOMEPARTS)
	$(RSYNC) --exclude src . ${TARGETDIR} -n

pub: 
	$(RSYNC) $(HTDOCS) ${TARGETDIR}
	$(RSYNC) $(EXAMPLES) ${TARGETDIR}/examples
	$(RSYNC) $(IMAGES) ${TARGETDIR}/images
	$(RSYNC) $(SCRIPTS) ${TARGETDIR}/scripts
