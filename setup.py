from distutils.core import setup

setup(name='pyformex',
      version='0.5',
      description='A tool to generate and manipulate complex 3D geometries.',
      long_description="""
pyFormex is a program for generating, manipulating and operating on 
large geometrical models of 3D structures by sequences of mathematical
transformations.
""",
      author='Benedict Verhegghe',
      author_email='benedict.verhegghe@ugent.be',
      url='http://pyformex.berlios.de/',
      packages=['pyformex','pyformex.gui','pyformex.plugins','pyformex.examples'],
      package_data={'pyformex': ['pyformexrc', 'icons/*.xpm','examples/*.db','doc/*']},
      scripts=['pyformex/pyformex'],
#      data_files=[('pyformex/doc',['README','COPYING'])],
)
