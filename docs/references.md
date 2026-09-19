# 参考资料索引

全文检索引擎(nail-seeker)可行性分析过程中收集并经核实的资料。
收集日期:2026-09-19。

---

## 1. 整体架构 / 存储模型

### Pinecone Sparse V3(term-major 磁盘布局)—— 核心参考
- 官方博客(2026-07-09):https://www.pinecone.io/blog/sparse-v3
- 博客要点:term-major 布局替换 document-major;磁盘 I/O 最高降 1428×(BM25)/151×(SPLADE),延迟降 119×;每 term 独占 posting 块序列;块前元数据(min/max docid、块内最高分、posting 数)用于跳过;term 目录用 Elias-Fano(~300KB/100K terms);块内 docid 用相对最小值的定宽位压缩(2–2.5×);每 term 独立分数量化域;MaxScore 剪枝。

### Pinecone sparse-only 索引发布(背景)
- 《Don't be dense: Launching sparse indexes in Pinecone》:https://www.pinecone.io/learn/sparse-retrieval/

### Pinecone 文档
- 索引总览:http://docs.pinecone.io/guides/index-data/indexing-overview
- 混合检索:http://docs.pinecone.io/guides/search/hybrid-search
- 全文检索(full_text_search / BM25 / Lucene 语法):http://docs.pinecone.io/guides/search/full-text-search
- 官方文档主页:https://pinecone.io/docs

### Pinecone 第三方梳理
- dbengines 引擎档案:https://github.com/druce/dbengines/blob/main/engines/pinecone.md

### Pinecone 稀疏向量化工具(pinecone-text)
- https://pinecone-io.github.io/pinecone-text/pinecone_text/sparse.html

---

## 2. 词典(term 字典)

### C²: Cache-Conscious Succinct Tries with Adaptive Unary Path Compression —— 核心参考
- arXiv 论文:https://arxiv.org/abs/2606.16104
- HTML 全文:https://arxiv.org/html/2606.16104v1
- PDF:https://arxiv.org/pdf/2606.16104
- 第三方阅读站:https://papers.cool/arxiv/2606.16104 、 https://arxiv.deeppaper.ai/papers/2606.16104v1
- 要点:C₁=缓存友好的 rank/select 内联布局(块 704/704/1024 bit,≤2 缓存行),C₂=unary path 压缩;对 FST/CoCo-trie/Marisa 三态 succinic trie 做重构;查询加速 1.12–1.58×,空间省 ~1.3×。
- **官方代码(C++,研究级)**:https://github.com/alexztc/C2

### C² 的基线与对比结构
- CoCo-trie:https://github.com/aboffa/CoCo-trie
  - 期刊版(Information Systems 2024,Open Access):https://doi.org/10.1016/j.is.2023.102316
  - 会议版(SPIRE'22):https://doi.org/10.1007/978-3-031-20643-6_17
- MARISA trie:https://github.com/s-yata/marisa-trie 、Python 绑定 https://github.com/pytries/marisa-trie
- xcdat(压缩 double-array trie,含大量 succinic trie 对比表):https://github.com/kampersanda/xcdat

### Rust 生态(移植对照)
- `fst` crate(tantivy 自身使用的 term 词典):https://crates.io/crates/fst

---

## 3. 索引结构 / 内存执行

### Cocoa / VeloSearch(列式倒排,向量化批量查询)—— 核心参考
- 项目主页:https://velosearch.github.io/
- 论文(ICDE 2025,pp.1800–1813,华东师大):DOI https://doi.org/10.1109/ICDE65448.2025.00138 、 IEEE Xplore https://ieeexplore.ieee.org/document/11112974/
- 源码(Rust):https://github.com/Velosearch/velosearch
- 代码文档:https://velosearch.github.io/velosearch/velosearch
- 要点:倒排表以列式(columnar)紧凑格式组织;向量化批量处理避免分支预测失败;子句枚举 + 剪枝;宣称 ~30× vs Lucene/Tantivy;支持 AVX512/AVX/SSE;Rust 1.68。

---

## 4. 索引压缩

### LICO:An SIMD-Aware High-Performance Learned Inverted Index Compression Framework —— 核心参考
- ACM / SIGMOD 2026(官方入口;ACM 对脚本/爬虫返回 403,浏览器打开正常):https://dl.acm.org/doi/10.1145/3802079
- 机器人可访问的替代入口(论文页):https://rmarcus.info/dbscholar/papers/h53afcc0bc4a17249
- **官方代码(C++)**:https://github.com/xianyuzhuruc/LICO
  - 仓库内附技术报告 PDF:《LICO…(Technical Report).pdf》;含 lico_build/decode/query.cpp + include/ + external/(SimSIMD 子模块);无 license 文件。
- 镜像仓库:https://github.com/qpwoeiruty987123/LICO
- 要点:误差有界分段线性模型 + residual 数组,无损还原;SIMD 感知解码;自动适配分布;大小/延迟 Pareto 最优于经典与学习式方案。
- 同组早期项目(salad,SIMD-aware learned data compression):https://github.com/qyliu-hkust/salad

### 学习式压缩相关
- LeCo:Lightweight Compression via Learning Serial Correlations(SIGMOD'24)
  - arXiv:https://arxiv.org/abs/2306.15374 、 作者页 PDF:https://people.iiis.tsinghua.edu.cn/~huanchen/publications/leco-sigmod24.pdf
  - 代码:https://github.com/yhliu918/Learn-to-Compress
- Learned Data Compression 综述(arXiv 2412.10770):https://arxiv.org/abs/2412.10770 、 PDF https://arxiv.org/pdf/2412.10770

### 经典倒排压缩(survey)
- Techniques for Inverted Index Compression(Pibiri & Venturini,ACM Comput Surv 2020):https://arxiv.org/pdf/1908.10598 、 http://pages.di.unipi.it/pibiri/papers/ii_survey.pdf
- Stanford IR Book, Index Compression 章节:https://nlp.stanford.edu/IR-book/html/htmledition/index-compression-1.html
- Lucene/ES store 压缩:https://www.elastic.co/blog/store-compression-in-lucene-and-elasticsearch

---

## 5. 分词器

### tantivy-tokenizer-api(独立稳定 API crate)
- crates.io:https://crates.io/crates/tantivy-tokenizer-api
- lib.rs(对脚本/爬虫返回 403,浏览器打开正常):https://lib.rs/crates/tantivy-tokenizer-api
- docs.rs(tantivy::tokenizer 模块):https://docs.rs/tantivy/latest/tantivy/tokenizer/index.html
- tantivy 源码中的 tokenizer 实现:https://github.com/quickwit-oss/tantivy/tree/main/src/tokenizer
- 要点:Tokenizer / TokenStream / TokenFilter 三 trait;Token 含 offset_from/to、position、text、position_length;内置 SimpleTokenizer、RawTokenizer、LowerCaser、RemoveLongFilter、StopWordFilter、Stemmer 等;tantivy 0.23+ 通过 `tokenizer_api` 复用。

---

## 6. 依赖 crate

- tokio:https://crates.io/crates/tokio 、 https://docs.rs/tokio
- rayon:https://crates.io/crates/rayon
- memmap2:https://crates.io/crates/memmap2 、 https://docs.rs/memmap2
- fearless_simd(早期实验版,无稳定性保证):https://crates.io/crates/fearless_simd 、 https://docs.rs/fearless_simd/latest/fearless_simd/index.html
- 备选:SIMD 层兜底 std::simd(https://doc.rust-lang.org/stable/core/simd/index.html)、wide(https://github.com/lokathor/wide)

---

## 附录:可行性分析的关键结论

1. 六项参考分属不同层(服务/词典/磁盘布局/内存执行/压缩/分词),正交、可干净分层,无原理性冲突。
2. 主要缺口:查询代数与 top-k 执行器(参考 VeloSearch 的布尔/cnf/dnf handler)、BM25 集合统计、位置(phrase)列、写入/段合并/tombstone 与实时索引。
3. 风险排序:LICO 与 C² 均有 C++ 官方实现可对照移植(工作量主导);Pinecone V3 无开源、块格式细节需自定;fearless_simd 需隔离便于替换;多数论文/博客数字为自报,需自建基准(MS MARCO)验证。