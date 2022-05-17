from setuptools import setup, find_packages

setup(
    name='deepdanbooru_onnx',
    version='0.0.1',
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
        'Operating System :: OS Independent']
)