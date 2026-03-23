```python
import torch                    # 深度学习框架
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

from transformers import DPTImageProcessor, DPTForDepthEstimation  # Hugging Face 模型
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator  # SAM 分割
```

**程序功能**：输入一张图片，输出两张分析图
- 深度图（每个像素的距离）
- SAM 分割图（每个物体的轮廓）

**需要准备的文件**：
- `image_d61af3.jpg` - 输入图片
- `sam_vit_h_4b8939.pth` - SAM 模型权重（可选）

---

## 与 interactive_sam_depth.py 的区别

| 特点 | `import torch.py` | `interactive_sam_depth.py` |
|------|-------------------|---------------------------|
| **运行方式** | 自动执行，保存图片 | 交互式，鼠标点击操作 |
| **分割方式** | 全图所有物体自动分割 | 只分割你点击的物体 |
| **输出** | 保存两张图片到文件 | 实时窗口显示 |
| **使用难度** | 简单，放图片就跑 | 需要点击操作 |

**形象比喻**：
- 第一个 = 一次拍快照，自动找出所有东西
- 第二个 = 有个助手，你指哪里他分割哪里

---

## 设备选择

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

| 情况 | device 值 |
|------|----------|
| 有 NVIDIA 显卡 + CUDA | `"cuda"` |
| 没有独显或不支持 | `"cpu"` |

**Q: 为什么不是特殊语法？**
> `mask_or_depth` 就是普通变量名，下划线只是分隔符，让名字更易读。意思是"mask 或者 depth"。

---

## save_visualization 函数

```python
def save_visualization(image, mask_or_depth, mode="sam", output_name="output.png"):
```

### 参数说明

| 参数 | 含义 | 例子 |
|------|------|------|
| `image` | 原图（numpy数组） | `image_np` |
| `mask_or_depth` | 要叠加的数据（Mask或深度图） | `prediction` 或 `masks` |
| `mode` | 模式，默认 `"sam"` | `"sam"` 或 `"depth"` |
| `output_name` | 输出文件名，默认 `"output.png"` | `"result_01_depth.png"` |

---

## plt vs cv2

| 特点 | `plt` (matplotlib) | `cv2` (OpenCV) |
|------|---------------------|----------------|
| **擅长** | 制作图表、多子图、颜色条 | 实时显示、鼠标交互 |
| **输出** | 适合保存图片文件 | 适合弹窗显示 |
| **用途** | 论文图表、报告 | 交互工具、视频 |

**为什么第一个脚本用 plt？**
- 需要多子图并排显示（`plt.subplot`）
- 需要颜色条（`plt.colorbar`）
- 只是保存图片，不需要交互

**为什么第二个脚本用 cv2？**
- 需要鼠标点击（`cv2.setMouseCallback`）
- 需要实时弹窗（`cv2.imshow`）

---

## matplotlib 子图布局

```python
plt.subplot(1, 2, 1)  # 1行2列的第1个位置
plt.subplot(1, 2, 2)  # 1行2列的第2个位置
```

```
plt.subplot(1, 2, 1) → 位置①  ┌────┬────┐
                              │ ①  │ ②  │
plt.subplot(1, 2, 2) → 位置②  └────┴────┘
```

| 参数 | 意思 |
|------|------|
| 第1个 `1` | 1 行 |
| 第2个 `2` | 2 列 |
| 第3个 `1`或`2` | 第几个位置 |

---

## cmap 颜色映射

`cmap` = **colormap（颜色图）**，把数字映射成颜色。

```python
plt.imshow(mask_or_depth, cmap="inferno")
```

**inferno 颜色渐变**：
```
黑 → 紫 → 红 → 橙 → 黄 → 白
远 ──────────────────────── 近
```

**深度值越大（近）** = 越亮的黄色/白色
**深度值越小（远）** = 越暗的紫色/黑色

> 常用 cmap：`"gray"`（灰度）、`"inferno"`（热力图）、`"viridis"`（绿黄渐变）

---

## SAM 分割模式代码解析

```python
elif mode == "sam":
    plt.imshow(image)                     # 1. 先画原图作为背景
    ax = plt.gca()                        # 2. 获取当前坐标轴
    ax.set_autoscale_on(False)            # 3. 禁止自动缩放，保持比例

    # 4. 按面积从大到小排序（大的在下面，小的叠在上面）
    sorted_anns = sorted(mask_or_depth, key=(lambda x: x['area']), reverse=True)

    # 5. 创建透明叠加层 (H × W × 4)
    img_overlay = np.ones((sorted_anns[0]['segmentation'].shape[0],
                           sorted_anns[0]['segmentation'].shape[1], 4))
    img_overlay[:,:,3] = 0  # 透明度通道全设为0（完全透明）

    # 6. 循环给每个物体上随机颜色
    for ann in sorted_anns:
        m = ann['segmentation']           # 当前物体的像素掩码
        color_mask = np.concatenate([np.random.random(3), [0.4]])  # 随机RGB + 0.4透明度
        img_overlay[m] = color_mask      # 给这个物体涂上颜色

    ax.imshow(img_overlay)                # 7. 叠加显示
    plt.title("SAM Segmentation")
    plt.axis('off')
```


### ax 是什么？

```python
ax = plt.gca()  # get current axis
```

`ax` 是 `matplotlib.axes.Axes` 对象，掌管"这张图的坐标系统"。


**Q: 为什么 set_autoscale_on(False)？**

禁止 matplotlib 自动调整坐标轴范围，保持图片原比例，避免被拉伸或添加多余留白。

---

## numpy 语法详解

### shape[0] 和 shape[1]

```python
arr = np.array([[True, False],
                [True, True]])
arr.shape     # (2, 2)
arr.shape[0]  # 2 → 行数 = 高度 H
arr.shape[1]  # 2 → 列数 = 宽度 W
```

### np.ones() 和 np.zeros()

```python
np.ones(5)        # [1, 1, 1, 1, 1]
np.zeros(5)      # [0, 0, 0, 0, 0]
np.ones((3, 4))  # 3行4列的全1矩阵
```

### np.concatenate()

```python
a = [1, 2, 3]
b = [4, 5]
np.concatenate([a, b])  # [1, 2, 3, 4, 5]
```

### 随机颜色生成

```python
np.random.random(3)       # [0.52, 0.19, 0.89] → 3个随机小数（RGB）
[0.4]                     # 透明度

np.concatenate([[0.52, 0.19, 0.89], [0.4]])
# 结果: [0.52, 0.19, 0.89, 0.4] → RGBA 值

# 每次循环生成新颜色，所以每个物体颜色不同
```

### 透明叠加层原理

```python
# 创建 H×W×4 的数组
img_overlay = np.ones((H, W, 4))  # RGBA 全部为 1（不透明）
img_overlay[:,:,3] = 0            # Alpha 通道全设为 0（完全透明）
```

```
[:,:,3] 分解：
   │ │ │
   │ │ └── 第4个通道（Alpha）
   │ └── 所有列
   └── 所有行
```

### sorted + lambda 排序

```python
sorted_anns = sorted(mask_or_depth, key=(lambda x: x['area']), reverse=True)
```

- `sorted()` = Python 内置排序函数
- `key=lambda x: x['area']` = 按字典的 'area' 值排序
- `reverse=True` = 从大到小

```python
# 等价于
def get_area(x):
    return x['area']
sorted_anns = sorted(mask_or_depth, key=get_area, reverse=True)
```

---

### lambda 详解（匿名函数）

**Q: lambda 是什么？**

`lambda` 是 Python 的关键字，用来创建**匿名函数**（没有名字的函数）。

#### 普通函数写法

```python
def 函数名(参数):
    return 返回值

# 例子
def add(a, b):
    return a + b

result = add(1, 2)  # 返回 3
```

#### lambda 写法

```python
lambda 参数: 返回值

# 例子
lambda a, b: a + b
```

#### 两者的区别

```python
# 普通函数
def add(a, b):
    return a + b

add(1, 2)  # 3

# lambda（赋值给变量后用法一样）
my_add = lambda a, b: a + b
my_add(1, 2)  # 3
```

#### 为什么用 lambda？

因为有时候函数只需要用一次，没必要专门写 def：

```python
# 不用 lambda（需要额外定义函数）
def get_area(x):
    return x['area']
sorted(mask_or_depth, key=get_area)

# 用 lambda（一行搞定）
sorted(mask_or_depth, key=lambda x: x['area'])
```

#### lambda 语法拆解

```python
lambda x: x['area']
│       │ │
│       │ └── 冒号后的内容 = 返回值
│       └── 冒号
└── lambda关键字
```

---

### 字典取值 x['area']

**Q: `x['area']` 是什么？**

这是**字典取值语法**。

```

---

### sorted_anns 从哪里知道是 SAM 分割出来的？

**Q: 为什么知道 sorted_anns 是 SAM 分割的结果？**

`sorted_anns` 只是个参数名，**真正的数据从调用处传来**：

```python
# 调用处（第108行）
save_visualization(image_np, masks, mode="sam", output_name="result_02_sam_seg.png")
#                        ↑
#                   传入的第二个参数

# masks 从哪来？
masks = mask_generator.generate(image_np)  # mask_generator = SAM 的自动分割生成器
```

所以 `mask_or_depth` 在 `mode="sam"` 时，就是 **SAM 分割出来的物体列表**。

---

## 深度估计代码解析

```python
try:
    # 1. 加载模型
    depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)

    # 2. 图片预处理
    inputs = depth_processor(images=image_pil, return_tensors="pt").to(device)

    # 3. 模型推理（不计算梯度）
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # 4. 插值还原尺寸
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),      # (H,W) → (1,H,W)
        size=image_pil.size[::-1],         # (宽,高) → (高,宽)
        mode="bicubic",                    # 三次样条插值
        align_corners=False,
    ).squeeze().cpu().numpy()             # 压缩维度 → 转CPU → 转numpy

    # 5. 保存结果
    save_visualization(image_np, prediction, mode="depth", output_name="result_01_depth.png")

except Exception as e:
    print(f"深度估计失败: {e}")
```

### try-except 异常处理

```python
try:
    # 可能出错的代码
    ...
except Exception as e:
    # 出错时执行这里
    print(f"深度估计失败: {e}")
```

防止程序崩溃，例如：
- 模型下载失败
- 图片格式错误
- GPU 内存不足

### with torch.no_grad()

```python
with torch.no_grad():
    outputs = depth_model(**inputs)
```

**作用**：禁止 PyTorch 计算梯度

| 场景 | 是否需要梯度 | 原因 |
|------|------------|------|
| 训练模型 | 需要 | 要更新参数 |
| 推理/预测 | 不需要 | 只看结果，省内存加快速度 |

### 插值 (interpolate)

**为什么需要插值？**
```
原图尺寸：1920 × 1080
模型输出：384 × 216  ← 太小了
    ↓ interpolate 放大
1920 × 1080  ← 和原图一样大
```

**参数解释**：
- `predicted_depth.unsqueeze(1)` → 在第1维插入一个维度
- `size=image_pil.size[::-1]` → `[::-1]` 把 (宽,高) 变成 (高,宽)
- `mode="bicubic"` → 三次样条插值，效果最平滑
- `.squeeze().cpu().numpy()` → 压缩维度 → 转到CPU → 变成numpy数组

---

## 附录：


### RGB vs RGBA

```
RGB = [红, 绿, 蓝]          # 3通道，0-1或0-255
RGBA = [红, 绿, 蓝, 透明度]  # 4通道，多一个Alpha通道
```

### 变量命名 `mask_or_depth`

Q: 为什么叫这个名字？

> 因为这个变量**可以是 mask，也可以是 depth**，取决于调用时传什么。用下划线分隔只是让名字更易读，不是特殊语法。
