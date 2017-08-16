# depiGPU
GPU version of depi

## Bug 日志

### 2017.08.16

今天才想起来把bug写个日志.

在bug_beta2.cu文件中

1. 当改成share memory,在block为1的情况下,可以正确输出结果
2. 当直接用global memory时,都是可以正确输出结果的.

kernel0808.cu已经正确

![](res_0808.png)
