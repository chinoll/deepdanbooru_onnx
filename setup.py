from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='deepdanbooru_onnx',
    version='0.0.8',
    description='anime image classification',
    author='chinoll',
    author_email='chinoll@chinoll.org',
    url='https://github.com/chinoll/deepdanbooru_onnx',
    packages=find_packages(),
    keywords=['deepdanbooru', 'anime', 'image classification','onnx','deep learning'],
    requires=['onnx', 'onnxruntime', 'tqdm', 'numpy', 'Pillow', 'requests', 'shutil', 'hashlib', 'os'],
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3 :: Only',
        'Operating System :: OS Independent'],
    long_description_content_type="text/markdown",
    long_description=long_description,
    install_requires=['onnx', 'onnxruntime', 'tqdm', 'numpy', 'pillow', 'requests'],
)