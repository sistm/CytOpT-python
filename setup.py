from setuptools.command.build_ext import build_ext

try:
    import setuptools
except ImportError:
    print('''
    setuptools not found.
    On linux, the package is often called python-setuptools
    ''')
    from sys import exit

    exit(1)

try:
    long_description = open('./README.md', encoding='utf-8').read()
except:
    long_description = open('./README.md').read()

copt = {
    'msvc': ['/EHsc'],
    'intelw': ['/EHsc']
}


class build_ext_subclass(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in copt:
            for e in self.extensions:
                e.extra_compile_args = copt[c]
        build_ext.build_extensions(self)


import os

undef_macros = []
define_macros = []
if os.environ.get('DEBUG'):
    undef_macros = ['NDEBUG']
    if os.environ.get('DEBUG') == '2':
        define_macros = [('_GLIBCXX_DEBUG', '1')]

package_dir = {
    'CytOpT.tests': 'CytOpT/tests'
}
package_data = {
    'CytOpT.tests': ['data/*']
}
python_requires = '>=3.6',
classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]

about = {}  # type: ignore
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'CytOpT', 'cytoptVersion.py')) as f:
    exec(f.read(), about)

install_requires = open('./requirements.txt').read().strip().split('\n')
packages = setuptools.find_packages()
setuptools.setup(
    name=about['__title__'],
    version=about['__version__'],
    description='CytOpT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    author=about['__author__'],
    author_email=about['__author_email__'],
    license=about['__license__'],
    platforms=['Any'],
    classifiers=classifiers,
    keywords='cytOpT',
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    install_requires=install_requires,
    project_urls={
        'Article': 'https://arxiv.org/abs/2006.09003',
        'CytOpT pypi': 'https://github.com/sistm/CytOpt-python',
    },
    entry_points={
        'console_scripts': [
            'CytOpT-features = CytOpT.cytOpt:main',
        ],
    },
)