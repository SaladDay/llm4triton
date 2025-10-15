# 案例契约

每个工作负载都位于`cases/<case_name>/`目录下，并保持原始PyTorch模块几乎不变：

- `model.py`包含端到端的`torch.nn.Module`，与源项目中的完全一致。不需要任何执行框架或辅助函数。
- `manifest.yaml`声明了如何实例化模块（model.module、model.class_name、可选的构造函数kwargs）、如何合成输入，以及agent所需的任何元数据（设备、描述信息等）。
- 可以通过在`manifest`中指向`model.py`内的一个函数来声明可选的权重初始化器；否则，框架将缓存模块默认生成的`state_dict()。`

评估框架读取这些元数据，通过`agent_provide_inputs()`实例化缓存的输入，并确保基准版本和优化版本共享完全相同的张量和权重。

## 创建新案例

1. 复制`TEMPLATE`目录，并将其重命名为你的案例名称。
2. 将未经修改的PyTorch模块粘贴到model.py中，并根据需要调整类名。
3. 更新`manifest.yaml`：
   - 设置`model.class_name`（如果重命名了文件，还需设置`model.module`）。
   - 在`inputs.args / inputs.kwargs`下描述输入张量。
   - （可选）如果确定性的权重必须与默认初始化不同，请将`weights.function`指向一个辅助函数。
   - 记录描述和目标设备。
4. 通过评估框架导入该案例以完成注册（无需额外的胶水代码）。.
