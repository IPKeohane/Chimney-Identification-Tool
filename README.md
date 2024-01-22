## Chimney Identification Tool
#### Author: Isaac Keohane
#### Email: isaackeohane95@gmail.com
#### https://github.com/IPKeohane/Chimney-Identification-Tool
<br>  

##### Overview
This repository contains the code for the Chimney Identification Tool, which uses
a selective search and convolutional neural network to identify potential hydrothermal
chimney edifices from 1m resolution bathymetry. The development process and a more detailed description of the goals and performance is contained in the manuscript that this repository is a companion for:  

Keohane I, White S. Chimney Identification Tool for Automated Detection of Hydrothermal Chimneys from High-Resolution Bathymetry Using Machine Learning. *Geosciences*. 2022; 12(4):176. https://doi.org/10.3390/geosciences12040176


##### Steps to produce an output of point locations picked by the CIT
1. Start with a 1m-gridded bathymetry raster. There is an example one located at *data/cit_test_bathy_gsc_ll_1m.tif*
2. Run *rasterFiltering/create_multiband_raster_from_bathymetry.R* to produce
a normalized multiband raster derived from the input bathymetry. Make sure to edit
this file to point to the correct location and filename of the input bathymetry raster.
3. Run *neuralNet/scripts/runProduceCitOutputs.py* to produce a .csv output file of CIT pick
point locations. Make sure to edit the file so the *fp_in* variable points to
the multiband raster generated in step 2.


##### Notes on package requirements

*rasterFiltering/create_multiband_raster_from_bathymetry.R* runs in R and was
last updated to work with R 4.3.1 and the terra package 1.7.46

*neuralNet/scripts/runProduceCitOutputs.py* runs in Python. The python sourcecode
and other scripts used under the hood require several packages that need to be installed
before running. There is a full copy of the most recent anaconda environment used to
run these scripts at the end of this readme document for checking against if you run into errors.  
The primary ones are listed here with the versions used in the most recent working update of
this repository using Python 3.11.6:  
pytorch = 2.1.1, numpy = 1.26.2, scikit-image = 0.22.0, rasterio = 1.3.9,
pandas = 2.1.3, matplotlib = 3.8.2

<br>
<br>
<br>


Full list of anaconda packages used in development environment:  
affine = 2.4.0  
alabaster = 0.7.13  
aom = 3.7.1  
arrow = 1.3.0  
astroid = 3.0.1  
asttokens = 2.4.1  
atomicwrites = 1.4.1  
attrs = 23.1.0  
autopep8 = 2.0.4  
babel = 2.13.1  
bcrypt = 4.1.1  
beautifulsoup4 = 4.12.2  
binaryornot = 0.4.4  
black = 23.11.0  
blas = 2.120  
blas-devel = 3.9.0  
bleach = 6.1.0  
blosc = 1.21.5  
brotli = 1.1.0  
brotli-bin = 1.1.0  
brotli-python = 1.1.0  
bzip2 = 1.0.8  
c-blosc2 = 2.11.2  
ca-certificates = 2023.11.17  
cairo = 1.18.0  
certifi = 2023.11.17  
cffi = 1.16.0  
cfitsio = 4.3.0  
chardet = 5.2.0  
charls = 2.4.2  
charset-normalizer = 3.3.2  
click = 8.1.7  
click-plugins = 1.1.1  
cligj = 0.7.2  
cloudpickle = 3.0.0  
colorama = 0.4.6  
comm = 0.1.4  
contourpy = 1.2.0  
cookiecutter = 2.5.0  
cryptography = 41.0.5  
cuda-cccl = 12.3.101  
cuda-cudart = 11.8.89  
cuda-cudart-dev = 11.8.89  
cuda-cupti = 11.8.87  
cuda-libraries = 11.8.0  
cuda-libraries-dev = 11.8.0  
cuda-nvrtc = 11.8.89  
cuda-nvrtc-dev = 11.8.89  
cuda-nvtx = 11.8.86  
cuda-profiler-api = 12.3.101  
cuda-runtime = 11.8.0  
cycler = 0.12.1  
dav1d = 1.2.1  
debugpy = 1.8.0  
decorator = 5.1.1  
defusedxml = 0.7.1  
diff-match-patch = 20230430  
dill = 0.3.7  
docstring-to-markdown = 0.13  
docutils = 0.20.1  
entrypoints = 0.4  
exceptiongroup = 1.2.0  
executing = 2.0.1  
expat = 2.5.0  
filelock = 3.13.1  
flake8 = 6.1.0  
font-ttf-dejavu-sans-mono = 2.37  
font-ttf-inconsolata = 3.000  
font-ttf-source-code-pro = 2.038  
font-ttf-ubuntu = 0.83  
fontconfig = 2.14.2  
fonts-conda-ecosystem = 1  
fonts-conda-forge = 1  
fonttools = 4.45.1  
freetype = 2.12.1  
freexl = 2.0.0  
geos = 3.12.1  
geotiff = 1.7.1  
gettext = 0.21.1  
giflib = 5.2.1  
glib = 2.78.1  
glib-tools = 2.78.1  
gst-plugins-base = 1.22.7  
gstreamer = 1.22.7  
hdf4 = 4.2.15  
hdf5 = 1.14.2  
icu = 73.2  
idna = 3.6  
imagecodecs = 2023.9.18  
imageio = 2.31.5  
imagesize = 1.4.1  
importlib-metadata = 6.8.0  
importlib_metadata = 6.8.0  
importlib_resources = 6.1.1  
inflection = 0.5.1  
intel-openmp = 2023.2.0  
intervaltree = 3.1.0  
ipykernel = 6.26.0  
ipython = 8.18.1  
isort = 5.12.0  
jaraco.classes = 3.3.0  
jedi = 0.19.1  
jellyfish = 1.0.3  
jinja2 = 3.1.2  
joblib = 1.3.2  
jsonschema = 4.20.0  
jsonschema-specifications = 2023.11.2  
jupyter_client = 8.6.0  
jupyter_core = 5.5.0  
jupyterlab_pygments = 0.3.0  
jxrlib = 1.1  
kealib = 1.5.2  
keyring = 24.3.0  
kiwisolver = 1.4.5  
krb5 = 1.21.2  
lazy_loader = 0.3  
lcms2 = 2.15  
lerc = 4.0.0  
libaec = 1.1.2  
libarchive = 3.7.2  
libavif = 1.0.1  
libblas = 3.9.0  
libboost-headers = 1.83.0  
libbrotlicommon = 1.1.0  
libbrotlidec = 1.1.0  
libbrotlienc = 1.1.0  
libcblas = 3.9.0  
libclang = 15.0.7  
libclang13 = 15.0.7  
libcublas = 11.11.3.6  
libcublas-dev = 11.11.3.6  
libcufft = 10.9.0.58  
libcufft-dev = 10.9.0.58  
libcurand = 10.3.4.101  
libcurand-dev = 10.3.4.101  
libcurl = 8.4.0  
libcusolver = 11.4.1.48  
libcusolver-dev = 11.4.1.48  
libcusparse = 11.7.5.86  
libcusparse-dev = 11.7.5.86  
libdeflate = 1.19  
libexpat = 2.5.0  
libffi = 3.4.2  
libgdal = 3.7.3  
libglib = 2.78.1  
libhwloc = 2.9.3  
libiconv = 1.17  
libjpeg-turbo = 3.0.0  
libkml = 1.3.0  
liblapack = 3.9.0  
liblapacke = 3.9.0  
libnetcdf = 4.9.2  
libnpp = 11.8.0.86  
libnpp-dev = 11.8.0.86  
libnvjpeg = 11.9.0.86  
libnvjpeg-dev = 11.9.0.86  
libogg = 1.3.4  
libpng = 1.6.39  
libpq = 16.1  
librttopo = 1.1.0  
libsodium = 1.0.18  
libspatialindex = 1.9.3  
libspatialite = 5.1.0  
libsqlite = 3.44.2  
libssh2 = 1.11.0  
libtiff = 4.6.0  
libuv = 1.44.2  
libvorbis = 1.3.7  
libwebp = 1.3.2  
libwebp-base = 1.3.2  
libxcb = 1.15  
libxml2 = 2.11.6  
libzip = 1.10.1  
libzlib = 1.2.13  
libzopfli = 1.0.3  
lz4-c = 1.9.4  
lzo = 2.10  
m2w64-gcc-libgfortran = 5.3.0  
m2w64-gcc-libs = 5.3.0  
m2w64-gcc-libs-core = 5.3.0  
m2w64-gmp = 6.1.0  
m2w64-libwinpthread-git = 5.0.0.4634.697f757  
markdown-it-py = 3.0.0  
markupsafe = 2.1.3  
matplotlib = 3.8.2  
matplotlib-base = 3.8.2  
matplotlib-inline = 0.1.6  
mccabe = 0.7.0  
mdurl = 0.1.0  
minizip = 4.0.3  
mistune = 3.0.2  
mkl = 2023.2.0  
mkl-devel = 2023.2.0  
mkl-include = 2023.2.0  
more-itertools = 10.1.0  
mpmath = 1.3.0  
msys2-conda-epoch = 20160418  
munkres = 1.1.4  
mypy_extensions = 1.0.0  
nbclient = 0.8.0  
nbconvert = 7.11.0  
nbconvert-core = 7.11.0  
nbconvert-pandoc = 7.11.0  
nbformat = 5.9.2  
nest-asyncio = 1.5.8  
networkx = 3.2.1  
numpy = 1.26.2  
numpydoc = 1.5.0  
openjpeg = 2.5.0  
openssl = 3.1.4  
packaging = 23.2  
pandas = 2.1.3  
pandoc = 3.1.3  
pandocfilters = 1.5.0  
paramiko = 3.3.1  
parso = 0.8.3  
pathspec = 0.11.2  
pcre2 = 10.42  
pexpect = 4.8.0  
pickleshare = 0.7.5  
pillow = 10.1.0  
pip = 23.3.1  
pixman = 0.42.2  
pkgutil-resolve-name = 1.3.10  
platformdirs = 4.0.0  
pluggy = 1.3.0  
ply = 3.11  
poppler = 23.11.0  
poppler-data = 0.4.12  
postgresql = 16.1  
proj = 9.3.0  
prompt-toolkit = 3.0.41  
psutil = 5.9.5  
pthread-stubs = 0.4  
pthreads-win32 = 2.9.1  
ptyprocess = 0.7.0  
pure_eval = 0.2.2  
pycodestyle = 2.11.1  
pycparser = 2.21  
pydocstyle = 6.3.0  
pyflakes = 3.1.0  
pygments = 2.17.2  
pylint = 3.0.2  
pylint-venv = 3.0.3  
pyls-spyder = 0.4.0  
pynacl = 1.5.0  
pyparsing = 3.1.1  
pyqt = 5.15.9  
pyqt5-sip = 12.12.2  
pyqtwebengine = 5.15.9  
pysocks = 1.7.1  
python = 3.11.6  
python-dateutil = 2.8.2  
python-fastjsonschema = 2.19.0  
python-lsp-black = 1.3.0  
python-lsp-jsonrpc = 1.1.2  
python-lsp-server = 1.9.0  
python-lsp-server-base = 1.9.0  
python-slugify = 8.0.1  
python-tzdata = 2023.3  
python_abi = 3.11  
pytoolconfig = 1.2.5  
pytorch = 2.1.1  
pytorch-cuda = 11.8  
pytorch-mutex = 1.0  
pytz = 2023.3.post1  
pywavelets = 1.4.1  
pywin32 = 306  
pywin32-ctypes = 0.2.2  
pyyaml = 6.0.1  
pyzmq = 25.1.1  
qdarkstyle = 3.2  
qstylizer = 0.2.2  
qt-main = 5.15.8  
qt-webengine = 5.15.8  
qtawesome = 1.2.3  
qtconsole = 5.5.1  
qtconsole-base = 5.5.1  
qtpy = 2.4.1  
rasterio = 1.3.9  
rav1e = 0.6.6  
referencing = 0.31.1  
requests = 2.31.0  
rich = 13.7.0  
rope = 1.11.0  
rpds-py = 0.13.2  
rtree = 1.1.0  
scikit-image = 0.22.0  
scikit-learn = 1.3.2  
scipy = 1.11.4  
setuptools = 68.2.2  
sip = 6.7.12  
six = 1.16.0  
snappy = 1.1.10  
snowballstemmer = 2.2.0  
snuggs = 1.4.7  
sortedcontainers = 2.4.0  
soupsieve = 2.5  
sphinx = 7.2.6  
sphinxcontrib-applehelp = 1.0.7  
sphinxcontrib-devhelp = 1.0.5  
sphinxcontrib-htmlhelp = 2.0.4  
sphinxcontrib-jsmath = 1.0.1  
sphinxcontrib-qthelp = 1.0.6  
sphinxcontrib-serializinghtml = 1.1.9  
spyder = 5.5.0  
spyder-kernels = 2.5.0  
sqlite = 3.44.2  
stack_data = 0.6.2  
svt-av1 = 1.7.0  
sympy = 1.12  
tbb = 2021.10.0  
text-unidecode = 1.3  
textdistance = 4.5.0  
threadpoolctl = 3.2.0  
three-merge = 0.1.1  
tifffile = 2023.9.26  
tiledb = 2.16.3  
tinycss2 = 1.2.1  
tk = 8.6.13  
toml = 0.10.2  
tomli = 2.0.1  
tomlkit = 0.12.3  
torchaudio = 2.1.1  
torchvision = 0.16.1  
tornado = 6.3.3  
traitlets = 5.14.0  
types-python-dateutil = 2.8.19.14  
typing-extensions = 4.8.0  
typing_extensions = 4.8.0  
tzdata = 2023c  
ucrt = 10.0.22621.0  
ujson = 5.8.0  
uriparser = 0.9.7  
urllib3 = 2.1.0  
vc = 14.3  
vc14_runtime = 14.36.32532  
vs2015_runtime = 14.36.32532  
watchdog = 3.0.0  
wcwidth = 0.2.12  
webencodings = 0.5.1  
whatthepatch = 1.0.5  
wheel = 0.42.0  
win_inet_pton = 1.1.0  
xerces-c = 3.2.4  
xorg-libxau = 1.0.11  
xorg-libxdmcp = 1.1.3  
xz = 5.2.6  
yaml = 0.2.5  
yapf = 0.40.1  
zeromq = 4.3.5  
zfp = 1.0.0  
zipp = 3.17.0  
zlib = 1.2.13  
zlib-ng = 2.0.7  
zstd = 1.5.5  
