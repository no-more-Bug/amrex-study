# AMReX 中面向 LBM 的离散点集遍历方法整理

这份笔记总结了在 **AMReX 单层代码**里，针对 **离散边界点集**（例如 LBM 中需要特殊处理的边界格点）可以采用的几种组织与遍历方式。

---

## 1. 背景：为什么 LBM 会特别关心“离散点集遍历”

对很多传统 CFD 离散格式来说，边界处理往往可以写成：

- 对整个 patch 做 stencil
- 在靠边区域使用 ghost 或物理边界条件

但对 **LBM**，很多时候真正特殊的并不是整块区域，而是：

- 一批边界格点
- 一批固体格点
- 一批需要 bounce-back / Zou-He / 非平衡外推 / 沉浸边界修正 的点

也就是说，边界处理更像是：

**对一批离散点做特殊更新，而不是对整个 box 做统一 stencil。**

因此，如何在 AMReX 中组织“离散点集的遍历”会成为一个核心工程问题。

---

## 2. 方法一：直接在整个 box 上遍历，然后用 `if` 判断边界点

### 基本思路

在 `MFIter + ParallelFor(box, ...)` 中遍历整个 patch 或 tile，然后通过条件判断识别哪些点是边界点。

伪代码：

```cpp
for (MFIter mfi(f); mfi.isValid(); ++mfi)
{
    Box const& bx = mfi.validbox();
    auto const& arr = f.array(mfi);

    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
    {
        if (is_boundary(i,j,k)) {
            // 特殊边界处理
        } else {
            // 常规碰撞-迁移或别的更新
        }
    });
}
```

### 优点

- 最直接
- 最容易写
- 最适合原型验证

### 缺点

- 对 CPU 会带来分支判断
- 对 GPU 会带来线程束发散
- 如果边界点比例很小，会浪费大量计算

### 适用场景

- 早期原型
- 边界比例不小
- 逻辑还在频繁变化

---

## 3. 方法二：提取某个边界面或薄层 box，单独遍历

### 基本思路

不要在整个 patch 上 `if`，而是先把边界层 box 切出来，再只对边界层启动单独 kernel。

AMReX `Box` 工具很适合做这件事，例如：

- `makeSlab(direction, slab_index)`：压成某个方向上的单层 slab
- `bdryLo(box, dir, len)` / `bdryHi(box, dir, len)`：取某个方向 low/high 侧的边界层
- `operator&`：取交集

### 典型用途

- 处理 xlo / xhi / ylo / yhi / zlo / zhi 边界
- 处理靠近边界的一层或几层节点/单元

### 优点

- 比“全域遍历 + if”更高效
- 边界处理语义清楚
- GPU 发散更少

### 缺点

- 只适合规则边界或能用薄层 box 表达的边界
- 对复杂障碍物不够灵活

### 适用场景

- 规则外边界
- 简单矩形内部区域
- 想把边界处理从主更新里拆出来

---

## 4. 方法三：构造一个较小的局部 box，只在局部区域内遍历

### 基本思路

如果你知道需要特殊处理的点集中在一个局部区域，可以先构造一个包围它们的较小 `Box`，然后对当前 patch 的 `validbox()` 与这个局部 box 取交集，只处理交集区域。

伪代码：

```cpp
Box obj_box(lo, hi, mf.boxArray().ixType());

for (MFIter mfi(mf); mfi.isValid(); ++mfi)
{
    Box work = mfi.validbox() & obj_box;
    if (!work.ok()) continue;

    ParallelFor(work, [=] AMREX_GPU_DEVICE (int i, int j, int k)
    {
        if (is_boundary(i,j,k)) {
            ...
        }
    });
}
```

### 优点

- 显著缩小遍历区域
- 很适合“某个障碍物附近的一块区域”
- 仍然保持 AMReX 的 box/patch 风格

### 缺点

- 如果包围盒过大而真实边界点很稀疏，仍然会有额外开销
- 对高度复杂几何仍需配合 `if` 或 mask

### 适用场景

- 某个物体、某个局部区域附近的特殊处理
- 边界点集中在较小局部块内

---

## 5. 方法四：使用 mask

### 基本思路

预先建立一个与主网格同布局的 `mask`（或整型标签场），例如：

- `mask = 0`：固体边界点
- `mask = 1`：流体内部点
- `mask = 2`：入口点
- `mask = 3`：出口点

之后在 kernel 中根据 mask 决定不同处理方式。

### 优点

- 非常灵活
- 适合复杂几何、复杂边界类别
- 边界识别与数值更新解耦
- 对 LBM 特别自然，因为 LBM 常常本来就会区分 solid/fluid/interface nodes

### 缺点

- 仍然会有条件判断
- 需要额外存储一个标签场
- 如果 mask 只为极少数点存在，可能有些“重”

### 适用场景

- 复杂障碍物
- 多种边界类型混合
- 边界类别需要长期复用
- LBM 中 solid / fluid / boundary node 分类

### 备注

对 LBM 来说，**mask 往往是非常值得考虑的中长期方案**，因为很多边界处理本来就是基于节点类别来分支的。

---

## 6. 方法五：自己维护 `IntVect` 点列表

### 基本思路

如果边界点非常稀疏，或者你已经有一套“边界节点列表”，可以直接存一组 `IntVect`：

```cpp
Vector<IntVect> boundary_pts;
```

然后只遍历这些点。

### 最关键的工程建议

不要只存“全局点列表”，更推荐：

**按 patch / FAB 预分组。**

也就是组织成：

```cpp
Vector<Vector<IntVect>> pts_per_fab;
```

其中 `pts_per_fab[mfi.LocalIndex()]` 保存当前 patch 对应的那一批边界点。

### 为什么要按 patch 分组

因为 AMReX 的数据天然是按 FAB/patch 分块存储的：

- 不分组的话，每个 patch 都要扫描整份点表
- 分组后，每个 patch 只处理自己的点，效率和逻辑都会更好

### 遍历方式

#### CPU 上

直接遍历这一 patch 的点列表：

```cpp
for (auto const& iv : pts_per_fab[mfi.LocalIndex()]) {
    arr(iv,0) = ...;
}
```

#### GPU 上

可以把点列表拷到 device 可见内存，然后做一个 1D `ParallelFor(npts, ...)`：

```cpp
ParallelFor(npts, [=] AMREX_GPU_DEVICE (int n)
{
    IntVect iv = pts[n];
    arr(iv,0) = ...;
});
```

### 优点

- 对真正稀疏点集非常高效
- 很贴合 LBM 的“只改边界节点”思路
- 不需要扫完整个 box

### 缺点

- 需要自己维护点集数据结构
- patch 重分布或网格改变时需要更新点列表
- 对规则边界未必比 box 遍历更简单

### 适用场景

- 稀疏边界点
- 复杂但离散化后节点数量不大
- LBM 的 bounce-back / 特殊边界节点集合

---

## 7. 几种方法的使用建议（面向 LBM）

### 第一阶段：最小原型

优先用：

- 方法一：全域遍历 + `if`

理由：

- 写得最快
- 最容易验证边界处理逻辑
- 先把正确性跑通

---

### 第二阶段：边界类别稳定后

根据边界几何特征分流：

#### 规则外边界

优先用：

- 方法二：边界 slab / boundary box

#### 局部障碍物区域

优先用：

- 方法三：局部包围盒 + 局部 `if`

#### 复杂障碍物 / 多类别边界

优先用：

- 方法四：mask

#### 稀疏离散边界节点集合

优先用：

- 方法五：按 patch 分组后的 `IntVect` 点列表

---

## 8. 一个对未来实现 LBM 很实用的判断准则

每次面对“边界点怎么遍历”时，先问自己三件事：

### 1）这些边界点是规则几何层，还是复杂稀疏点集？

- 规则层：优先 box/slab
- 稀疏点集：优先列表或 mask

### 2）这些边界点会不会频繁复用？

- 若会频繁复用：值得预处理成 mask 或 patch-local 点表
- 若只是临时一次性处理：简单 if 可能就够了

### 3）我当前更缺的是开发速度，还是运行效率？

- 缺开发速度：先直接写 if
- 缺运行效率：尽量改成小 box / mask / 点表

---

## 9. 面向未来的建议

对未来在 AMReX 上做 LBM，建议按下面顺序演化：

1. **先跑通全域 + if 的原型**
2. 再把规则边界拆成 **单独的 boundary boxes**
3. 对内部复杂障碍，逐步引入 **mask**
4. 对真正稀疏、长期复用的边界节点，考虑维护 **patch-local IntVect 列表**

这个顺序通常最稳：

- 正确性先行
- 然后再按热点逐步优化

---

## 10. 一句话总结

**在 AMReX 中处理 LBM 的离散边界点，最核心的思路不是“强行找到一个万能遍历器”，而是根据边界点的几何分布特征，在 `Box` 区域遍历、mask 分类和 patch-local 点列表之间做分层组织。**
