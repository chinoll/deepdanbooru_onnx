# DeepDanbooru
动漫图片分类模型
# 安装
```
pip install deepdanbooru-onnx
```
# 使用
```python
from deepdanbooru_onnx import DeepDanbooru, process_image
from PIL import Image
import numpy as np
danbooru = DeepDanbooru()
print(danbooru('test.jpg'))
img = Image.open('test.jpg')
print(danbooru(img))
img = process_image(img) # iamge to ndarray
print(danbooru(img))
print(list(danbooru(['test1.jpg','test2.jpg'])))
```