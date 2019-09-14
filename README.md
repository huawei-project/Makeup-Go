# Makeup-Go

# Data

将化妆数据解压到`data/`目录
```
data
└─makeuporigin
    ├─multi
    │   └─1 ~ 5
    └─rgb
        └─1 ~ 5
```

# Run

去除化妆后的图像保存在`data/output`，妆容保存在`data/makeup`，目录结构与原始目录结构一致

1. demo
    ``` shell
    python main_demo.py
    ```

2. 可见光数据
    ``` shell
    python main_rgb.py
    ```

3. 多光谱数据
    ``` shell
    python main_multi.py
    ```

# Reference

1. Chen Y C , Shen X , Jia J . [IEEE 2017 IEEE International Conference on Computer Vision (ICCV) - Venice (2017.10.22-2017.10.29)] 2017 IEEE International Conference on Computer Vision (ICCV) - Makeup-Go: Blind Reversion of Portrait Edit[C]// 2017 IEEE International Conference on Computer Vision (ICCV). IEEE Computer Society, 2017:4511-4519.
2. [w-yi/makeup-go - Github](https://github.com/w-yi/makeup-go)