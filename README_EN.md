# DeepDanbooru
Anime image classification model
# intall
```
pip install deepdanbooru-onnx
```
# usage
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

- [ ] Lightweight model
- [x] results cache
- [ ] Web API
- [ ] Model Quantification
- [ ] High accuracy model