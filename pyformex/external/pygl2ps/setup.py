from distutils.core import setup, Extension
setup(name="pygl2ps",
      version="1.3.3",
      description="Wrapper for GL2PS, an OpenGL to PostScript Printing Library",
      author="Benedict Verhegghe",
      author_email="benedict.verhegghe@ugent.be",
      url="http://pyformex.org",
      long_description="""
Python wrapper for GL2PS library by Christophe Geuzaine.
See http://www.geuz.org/gl2ps/
""",
      license="GNU LGPL (Library General Public License)",
      py_modules=["gl2ps"],
      ext_modules=[Extension("_gl2ps",
                             ["gl2ps.c","gl2ps_wrap.c"],
#                             include_dirs=["/usr/local/include/python2.3"],
                             libraries=["GL"])])
