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
#usage 1
print(danbooru('test.jpg'))
img = Image.open('test.jpg')
print(danbooru(img))
#usage 2
img = process_image(img) # iamge to ndarray
print(danbooru(img))
#usage 3
print(list(danbooru(['test1.jpg','test2.jpg'])))
```

# TODO

- [ ] 提供轻量化模型
- [x] 结果缓存
- [ ] 提供WEB接口
- [ ] 模型量化
- [ ] 精度更高的模型(slow) 