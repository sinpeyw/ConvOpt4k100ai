# 面向多模态大模型的基础卷积算子优化
**基于国产加速卡曙光K100_AI进行优化**

## 环境信息说明
- 硬件加速卡：曙光K100_AI
- 软件：dtk-24.04

## 快速运行
直接运行
``` bash
make clean
make gpu=1
./conv2dfp16demo 16 128 64 64 27 3 3 1 1 1 1
```
在slurm系统中：   
修改job.sh脚本并执行
``` bash
sbatch job.sh
```

## 卷积参数命名说明

| 参数 | 含义                         |
| ---- | ---------------------------- |
| n    | batch size                   |
| c    | channel number               |
| h    | 数据高                       |
| w    | 数据宽                       |
| k    | 卷积核数量                   |
| r    | 卷积核高                     |
| s    | 卷积核宽                     |
| u    | 卷积在高方向上的步长         |
| v    | 卷积在宽方向上的步长         |
| p    | 卷积在高方向上的补边         |
| q    | 卷积在宽方向上的补边         |
| Oh   | 卷积在高方向上的输出大小     |
| Ow   | 卷积在宽方向上的输出大小     |

相关公式：

Oh = (int)((h - r + 2 * p) / u) + 1  
Ow = (int)((w - s + 2 * q) / v) + 1

数据布局：

- **输入数据**：`nchw`
- **权值数据**：`kcrs`
- **输出数据**：`nkOhOw`

## 测试结果

使用数据集信息（每一行为一个配置）：

| n  | c   | h  | w  | k   | r | s | u | v | p | q |
| -- | --- | -- | -- | --- | - | - | - | - | - | - |
| 16 | 128 | 64 | 64 | 27  | 3 | 3 | 1 | 1 | 1 | 1 |
| 16 | 256 | 32 | 32 | 256 | 3 | 3 | 1 | 1 | 1 | 1 |
| 16 | 64  | 128| 128| 64  | 3 | 3 | 1 | 1 | 1 | 1 |
| 2  | 1920| 32 | 32 | 640 | 3 | 3 | 1 | 1 | 1 | 1 |
| 2  | 640 | 64 | 64 | 640 | 3 | 3 | 1 | 1 | 1 | 1 |
| 2  | 320 | 64 | 64 | 4   | 3 | 3 | 1 | 1 | 1 | 1 |

优化所产生的时间对比如下（单位：微秒）：

| 序号 | 优化前耗时 (us)  | 优化后耗时 (us)  | 加速比  |
| ---- | ---------------- | ---------------- | ------- |
| 1    | 3515.627197      | 263.397278       | 13.35   |
| 2    | 14719.093750     | 439.263794       | 33.52   |
| 3    | 14718.625977     | 589.392639       | 24.97   |
| 4    | 43832.644531     | 1433.512207      | 30.57   |
| 5    | 46812.703125     | 1299.222534      | 36.00   |
| 6    | 809.932861       | 183.935257       | 4.40    |

## 优化细节
请查看 [技术报告](TechnicalReport.pdf) 和代码
