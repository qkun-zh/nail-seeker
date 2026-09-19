2025 IEEE 41st International Conference on Data Engineering (ICDE) 

# Columnar Formatted Inverted Index for Highly-paralleled, Vectorized Query Processing 

Weichen Zhao, Minghao Zhao<sup>_∗_</sup> , Huiqi Hu, Weining Qian 

School of Data Science and Engineering, East China Normal University Shanghai Engineering Research Center of Big Data Management weichenzhao@stu.ecnu.edu.cn, {mhzhao, hqhu, wnqian}@dase.ecnu.edu.cn 

**_Abstract_ —Inverted index is a basic tool in many data-intensive applications. Though numerous efforts have been made on efficient inverted index-based query processing, existing schemes do not achieve the expected performance for modern data centers, in which servers are equipped with powerful CPUs and relatively large memory. Through comprehensive measurement studies, we identify the root course is that the data formats for index representation make it unfeasible to design efficient query execution approaches on top of it, which results in poor parallel query support and waste CPU computation. Driven by the findings, we propose to reconcile the in-memory index as columnar structures. To enable this idea, we construct the compact columnar format (** **_i.e.,_ Cocoa) that achieves both desirable space efficiency and maintains the capability for efficient searching support. With Cocoa, we design an efficient query executing scheme that utilizes vectorized batch processing to avoid frequent branch prediction, as well as clause enumeration with pruning to save the overhead of intermediate batch materialization. We build an open-source system VeloSearch to embody our design; experimental results show that VeloSearch achieves** _∼_ **30** _×_ **better performance compared with state-of-the-art search libraries such as Lucene and Tantivy.** 

**_Index Terms_ —indexing, query processing, columnar format** 

## I. INTRODUCTION 

An inverted list index (inverted index for short) stores a list of ordered _document identifiers_ for each vocabulary _term_ that appears in the documents. It has been extensively used in many data-intensive applications for efficiently locating/finding target information. For example, search engines leverage inverted index to find web pages containing a certain keyword; Databases utilize an inverted index to speed up evaluation of SQL queries<sup>1</sup> ; and in graph analytics, an inverted index can be used to indicate the adjacent nodes of a certain node [4, 5]. 

Through the decades, numerous efforts have been made to improve the performance of inverted index-based queries. These works include index structure design [6], index compression [2, 7] and query processing scheme constructions [8, 9]. Currently, _search libraries_ such as Lucene and Tantivy integrate several advanced techniques of aforementioned aspects and facilitate users to build search-related applications. Especially, among them, Lucene has become the _de facto_ platform [10], and a large number of search systems both in industry and academic have been built on top of it [11, 12]. 

> _∗_ Minghao Zhao is the corresponding author. 

> 1For example, to speed up SQL queries in which select predicates contain several filter criteria, many database systems precompute lists of row IDs that satisfy a certain predicate and store them in an index structure [1–3]. 

**TABLE I:** Performance comparison of several searching systems and handed-code on Wikipedia corpus 

|PostgreSQL|Lucene|Tantivy|**VeloSearch**<br>Hand-Coded|
|---|---|---|---|
|154 ms|3.1 ms|3.2 ms|**0.52 ms**<br>0.13 ms|



In an inverted index-based searching system, query statements are typically combinations of logical expressions involving _terms_ . Accordingly, to execute these queries, the most important task for query executors is to perform set operations such as intersection, union, and complement. Typically, executing such set operations in databases and search engines is less efficient than in hand-coded implementations (e.g., C++ programs). However, to our surprise, we find that their performance gap is exceptionally large, which is even larger than that between SQL queries in databases and hand-coded implementations on storage files [13, 14], _i.e.,_ the performance gap between the search libraries and hand-coded implementations is over 20 _×_ , and the gap between PostgreSQL and hand-coded implementations even exceeds 1000 _×_ . 

To investigate this phenomenon from the inside out, we conduct comprehensive measurement studies on two popular search libraries, _i.e.,_ Lucene [15] and Tantivy [16]. We design our benchmarks with real-world workloads of several different applications, including web search, database indexing, log analysis, etc. Our study results reveal a number of design issues that lead to CPU underutilization. Specifically, we find: 

- **Current query processing approaches bring about interpretation overhead and hinder the CPU’s capability to perform parallel execution.** The iterator execution model has been widely adopted in modern search libraries and platforms [17]. Similar to the _Volcano-style_ query processing approach [18] in databases, the iterator execution model processes one document in the posting list in a single iteration. For a query, all posting lists associated with the query are opened for reading and then traversed in an interleaved comparing fashion to produce matching documents. Consequently, this execution model not only introduces interpretation overhead, and prevents loop pipelining but also hinders modern CPUs from performing parallel execution, which results in a low instruction per cycle (IPC) efficiency. It has been reported that the huge interpretation overhead also appears in modern databases either, and recent studies [19–21] advocate that the iterator execution model falls short in efficiently utilizing CPUs. 

2375-026X/25/$31.00 ©2025 IEEE 1800 DOI 10.1109/ICDE65448.2025.00138 Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 

- **The high error rate of branch predictions interrupts micro-instruction pipelines.** Query processing on an inverted index consists of multiple intersection and union operations on sorted posting lists, which will transform to program branches ( _i.e., if-else_ branches) in execution. The large amount of program branches brings more chance of branch mispredictions, which, once occurred, will disrupt the CPU pipeline and cause CPU stalls [22]. Consequently, a significant portion ( _∼_ 40%) of the total execution time is wasted due to the CPU stalls caused by branch mispredictions. Despite several efforts to mitigate branch mispredictions in intersection operations using SIMD instructions [8, 9, 23, 24], the document-at-a-time access interface of the iterator model limits the applicability of these batch-oriented algorithms in modern search libraries. 

- **Significant variability in segment sizes affects the effectiveness of parallel index processing.** For efficiency and scalability concerns, modern search libraries separate their underlying inverted indexes into several self-contained segments. Each segment has its own term dictionary and posting lists. To boost query execution performance, intraquery parallelism is adopted. When executing a query, each involved segment is assigned a thread for parallel processing. However, the sizes of segments exhibit tremendous variability, making “larger” segments require more time for processing and thus become the “long tail” of execution. 

These findings imply that _the design of current search libraries lacks CPU awareness, resulting in CPU underutilization and poor parallel support during query execution._ More importantly, our in-depth analysis suggests that the iterator execution model and the data formats used for index representation may contribute to CPU underutilization. The unaligned, sparse, and scattered structure of these formats makes it difficult, or even infeasible, to design CPU-efficient query execution approaches (§III-B). 

This paper aims to construct an inverted index-based query processing system that leverages advanced features of modern CPUs and achieves effective parallelism. Driven by the above findings, we propose to use a columnar format to organize posting lists and facilitate vectorized query processing. The core challenge is to design a specialized columnar format that simultaneously reconciles search convenience and space efficiency.<sup>2</sup> Accordingly, we design a <u>Compact columnar</u> format (referred as to Cocoa) to achieve this goal. Conceptionally, Cocoa represents inverted lists as a matrix, with _m_ columns representing _m_ terms, and _n_ items in each column indicating the appearance of the terms in a document (§IV). For ease of access, each column is horizontally partitioned into batches, with postings within certain intervals belonging to the same batch. The matrix is hybrid encoded for storage efficiency: sparse batches are encoded as packed-integer lists, whereas dense batches are encoded as bitmaps (§IV-B). 

2Note that fully aligned vectors are computation-friendly but waste storage, whereas large compressed columns negatively affect query processing. 

With Cocoa, batch-at-a-time vectorized query execution is made easy. As Cocoa allows posting lists to be manipulated _in batches_ , the interpretation overhead originally present in the iterator model can be tremendously reduced. Additionally, its aligned structure makes it suitable to accelerate intersection and union operations on posting batches with SIMD instructions, which achieves adorable parallelism. Moreover, Cocoa encodes DocIds associated with each term into bitmaps, transforming logical operations on posting lists into branchless bitwise numerical calculations. This approach effectively avoids the significant CPU overhead caused by branch mispredictions (§V-A). Nevertheless, vectorized query processing necessitates substantial intermediate results. This is due to each operation must be performed _one by one_ to interpret and execute the query plan, and the intermediate results must be stored, leading to increased computation and storage consumption. To address this issue, we designed a pre-compilation mechanism with fused execution primitives that informs the evaluation engine to calculate the vectors _together_ in the above cases (§V-B). 

We develop VeloSearch to implement our designs on top of Apache DataFusion. Our in-depth evaluation is conducted using diverse workloads and settings. The results show that VeloSearch achieves 2.5 _×_ to 30 _×_ performance improvement compared to Lucene and Tantivy. We have made all source codes publicly available at https://velosearch.github.io. 

## II. BACKGROUND 

Inverted indexes are widely used in search engines and databases [25, 26]. This section presents basic descriptions of the1 inverted index (§II-A) and query processing on it (§II-B). 

## _A. Inverted index._ 

An inverted index consists of a term dictionary and a set of posting lists. The term dictionary maps unique terms to their associated posting lists, whereas each posting list contains an integer document identifier ( _i.e.,_ DocId) indicating in which documents this term appears. Normally, relevant metadata, such as term frequency and appearing positions, is also associated with each DocId. For ease of management and supporting updates, the whole inverted index is organized as a collection of segments. Each segment is a self-contained subindex, containing the entire term dictionary and the posting list with certain interval DocIds in it. When inserting and deleting documents, index modification will be organized into a new segment. Updates will be merged until they accumulate to a threshold, _i.e.,_ their corresponding segments will be combined with certain merging policies, such as tiering-based merge in Lucene and log-based merge in Tantivy. Segment merges improve query efficiency and enable deleted documents to be reclaimed, but they introduce CPU computation and disk I/O overhead, which can affect query performance. 

## _B. Query processing with inverted index._ 

We utilize the query execution process in search engines to illustrate the query processing approach of inverted indexes. A given query is first parsed into an abstract syntax tree (AST), 

1801 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 



<!-- Start of picture text -->
&1.::Seek(d):loop{ Collector::Next(): Dynamic Compiled HandCoded<br>  // recursive comparison 1.for doc in child.Next() 1.0<br>2.  d=right.Seek(d) collector 2. emit(doc, child.Score())<br>3.  if(d != d){<br>4.5.   sd=left.Seek(dz)if(d != sd)  & PostingIter::Seek(d):1.cursor=postingList.Seek(d) 0.5<br>6. continue 2.p=postingList[cursor]<br>7.  } 3.emit(p.doc)<br>8.  break & C<br>9. 10.} emit(d) PostingIter::Score():1.p=postingList[cursor] 0.0 AN AR LG SS WP IT<br>&::Score(): A B 2.score=Scorer(p.doc,<br>1.emit(left.Score() +    p.freqs) // BM25/TF-IDF<br>  right.Score()) next 3.emit(score) Fig. 2: Normalized execution time of three query processing ap-<br>&1.::Next():for d in left.Next() seek PostingIter::Next(): proaches on various datasets.<br>2. emit(Seek(d)) 1.emit(postingList[++cursor])<br> Time<br>Normalized Execution<br><!-- End of picture text -->

**Fig. 2:** Normalized execution time of three query processing approaches on various datasets. 

DocId and corresponding term frequency into tuples that contain the matched DocId and relevance score. 

**Fig. 1:** An example of the iterator model adopted in search engines for the query “A AND B AND C”. 

Through this processing approach, the iterator execution model provides flexible and simplified abstracts to interpret and execute arbitrary search queries. This model works well when the disk is the primary bottleneck because the expensive disk I/O hides the interpretation overhead and inefficiency inherent in the iterator execution model. 

where internal nodes (non-leaf nodes) represent Boolean operations that define the matching conditions between query terms. The Boolean operation AND indicates that all terms must match, which is executed through intersection operations on involved posting lists, while OR indicates that at least one term must match, which is executed through union operations. 

## III. MOTIVATION STUDY 

Modern search engines serve as interpreters to execute arbitrary queries and handle various compression mechanisms of posting lists. Therefore, the iterator execution model, which only requires the operators and underlying datasource to implement simple interfaces (e.g., Next, Seek and Score), has been widely adopted to eliminate the engineering complexity. The Next function calls return either the next matched posting ( _i.e.,_ DocId and metadata) or a null marker indicating the end of the iterator. The Seek function advances the iterator forward until reaching the target or the lowest DocId greater than the target and returns the posting. The Score function scores the matched either employing the score algorithms ( _e.g.,_ BM25 and TF-IDF) or combing the scores of the children operators. This processing approach is similar to the _Volcanostyle_ or Pipeline model execution model commonly adopted by databases and makes search engines flexible and simplified to interpret and execute search queries – search engines transform and optimize AST into an operator tree and evaluate the tree by iteratively traversing and invoking the Next or Score functions of each operator to collect results. 

We conducted comprehensive empirical measurements using various datasets to investigate the performance issues of inverted index-based search libraries. This section outlines the experimental setup (§III-A), presents our measurement results (§III-B), and discusses the implications (§III-C) of these findings motivated by these experiments. 

## _A. Experimental Setup_ 

**Testbed.** All experiments are conducted on a server equipped with a dual-socket 16-core Intel<sup>®</sup> Xeon<sup>®</sup> Silver 4314 CPU operating at 2.40GHZ with 48MB of L3 cache and 256GB of DRAM. This Intel 3rd-generation CPU supports SSE4.2, AVX, AVX2, and AVX-512 SIMD instruction set extensions. 

**Targeted Systems.** We chose the latest versions of Apache Lucene (version 9.6.0) and Tantivy (version 0.20) as our investigated systems. Lucene is a high-performance, full-text search engine library written in Java. To mitigate the impact of JVM garbage collection, we disabled the query cache feature of Lucene and selected the best performance from five runs. Similarly, Tantivy is a full-text search library written in Rust that provides an alternative implementation with certain optimizations. Both libraries offer in-memory indexes. Our experiments are conducted in in-memory mode to eliminate the effect of disk I/O. 

To illustrate the execution process, consider a search query “A AND B AND C”. Fig.1 shows the iterator execution processing for this query. The search plan comprises Intersection ( **&** ) and PostingIter operators. The **&** operator performs the intersection on the posting streams produced by the left and right child operators. The **&** ::Next outputs the next matched posting by invoking Seek based on the left child Next output. The scoring strategy adopted by **&** sums the scores of child operators. The PostingIter provides interfaces to access the posting lists. The Next advances the cursor and returns to the posting indicated by the cursor. The Seek searches the DocId in posting lists, often accelerated by auxiliary data structures ( _e.g.,_ the skip list) and search strategies ( _i.e.,_ binary search). Following the stream of the matched postings, the score procedure converts the postings consisting of matched 

**Datasets and Workloads.** As shown in Table II, we utilize six real-world datasets from three different application domains. The datasets for information retrieval vary in scenarios and corpora characteristics. For log analytics tasks, evaluations are performed on the Hadoop File System (HDFS) logs collected by Loghub [30]. Additionally, we construct the inverted index on database tables generated by the Star Schema Benchmark [31] to accelerate the evaluation of conjunctive and disjunctive predicate queries. Each dataset 

1802 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 

**TABLE II:** Corpora Statistics. 

|Corpus<br># Documents|# Terms<br># Words<br>|Memory Usage||Description||
|---|---|---|---|---|---|
|Antique (AN)<br>0.42 M<br>Argsme (AR)<br>0.36 M<br>Logs (LG)<br>11 M<br><br>|0.5 M<br>0.5 M<br>1.3 M<br>130 M<br>0.7 M<br>0.7 M<br><br>|136 MB<br>545 MB<br>1.8 GB<br>|A a non-facto<br>Arguments cr<br>A log set gen<br>|id dataset based on the questions <br>awled from debate and parliament <br>erated by Hadoop File System usi<br>|and answers of Yahoo! [27].<br> discussions [28, 29].<br>ng benchmark workloads [30].<br>|
|Star Schema (SS)<br>60 M<br><br>|23<br>240 M<br><br>|4.5 GB<br>|Star Schema <br>|(factor 10) benchmarks the perfor<br>|mance of databases [31]<br>|
|Wikipedia (WP)<br>4.3 M<br>Istella22 (IT)<br>8.1 M|3.7 M<br>110 M<br>0.6 M<br>190 M|7.3 GB<br>26 GB|A collection <br>A comprehen|of Web documents sourced from E<br>sive dataset of web documents wit|nglish Wikipedia [32].<br>h a set of queries[33].|
|Computation<br>Bad-specul|ation<br>Front-end|Back-end|y|AN<br>AR<br>LG|SS<br>WP<br>IT|
|AN<br>AR<br>LG<br>SS<br>WP<br>IT<br>(a) Lucene<br>0%<br>20%<br>40%<br>60%<br>80%<br>100%<br>Percentage of<br>different components|AN<br>AR<br>LG<br>SS<br>WP<br>IT<br>(b) Tantivy|&<br>cal<br>(c) Baseline|1<br>0.0<br>0.5<br>1.0<br>1.5<br>Normalized Query Latenc|2<br>4<br>8<br>16<br>32<br>(a) Lucene<br><br>0.00<br>0.25<br>0.50<br>0.75<br>1.00|1<br>2<br>4<br>8<br>16<br>32<br>(b) Tantivy|



**Fig. 3:** Execution time breakdown. 

**Fig. 4:** Average latency with varying number of threads. 

includes corresponding search query datasets derived from real queries and workloads, available in our open-source repository. We transform the raw queries of these datasets into various queries. Specifically, we construct keyword-matching queries that search for documents that meet the specific criteria, and scoring queries ( _e.g.,_ phrase queries) require term metadata ( _e.g.,_ term frequency and positions) to match and score the relevance of the matching documents to the query. 

## _B. Evaluation Results_ 

**Issue 1: Inefficient query processing approach.** Search engine libraries like Apache Lucene and Tantivy, and search engine platforms like Apache Solr and Elasticsearch, all employ the iterator execution model. We implement three search engine prototypes to explore the interpretation overhead and CPU underutilization inherent in the iterator execution model: Dynamic, Compiled, and HandCoded. The Dynamic is the native implementation of the iterator execution model in Tantivy, which relies on the dynamic dispatch mechanism powered by Rust language to interpret and execute query plans. In contrast, the Compiled involves reusing the Tantivy iterators but employs manual static implementation of the query plan, which dispatches execution code during compilation time to avoid the interpretation overhead. The HandCoded implements the operation by a simple merge-based set intersection algorithm. 

Fig.2 shows the execution time of three processing approaches. The findings reveal that the average execution time of the Dynamic is 1.21 _×_ to 12.26 _×_ larger than the Compiled. This substantial difference suggests that a significant portion of execution time is spent on interpretation rather than actual processing work. Additionally, the HandCoded approach demonstrates a noteworthy improvement of 3.27% to 42% compared to the Compiled. This improvement indicates that the iterator execution model not only suffers from interpretation overhead but also underutilizes the capabilities of modern CPUs. This is 

because the iterator model scatters the distributions of actual work units within the stream of interpreting functions, which prevents compilers and modern CPUs from fully exploiting deep pipelining and SIMD instructions [21]. 

**Issue 2: Significant branch bad-speculation overhead.** We employ the top-down performance analysis [34] to categorize execution stall causes and identify performance bottlenecks. This methodology systematically categorizes and quantifies bottlenecks in units of cycles or instruction slots. Each category contributes a specific percentage to the overall execution time. The categories encompass Computation (cycles utilized for instruction retirement, indicating the computation without stalls), Bad-speculation (stalls due to branch-instruction mispredictions), Front-end (stalls from instruction fetching/encoding), and Back-end (stalls from data cache misses). 

Fig.3 shows the execution breakdown of Lucene, Tantivy, and baseline tasks. We observe that branch misprediction overhead becomes the primary bottleneck, constituting approximately 20 _∼_ 47% and 25 _∼_ 58% of the total execution time for Lucene and Tantivy, respectively. Surprisingly, this finding differs significantly from other data-intensive systems [35, 36]. The study conducted by Kim _et al._ [9] also observed that the branch misprediction overhead arises from the conditional loop iterations and if statements produced by the set operations on sorted posting lists. The baselines verify this view, where the breakdown result of the set intersection (& in Fig.3-c) is similar to that of Lucene and Tantivy, whereas the result of prime number calculation ( _cal_ in Fig.3-c) is quite different. 

**Issue 3: Mismatched intra-query parallelism with segmented index structure.** To reduce the long-run queries’ latency, search engines employ intra-query parallelism to exploit parallelism within a single query. However, the segmented inverted indexes make intra-query parallelism inefficient: each query must visit all segments in the indexes sequentially to 

1803 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 



<!-- Start of picture text -->
Posting List (a) Placeholder  (b) Compact  Batch Bitmap 1 0 1 0 0 1 1<br>Columnar Layout Columnar Layout<br>Term Metadata Batch<br>DocId Freq Valid? Freq Valid? Freq Offsets 0 8 14 22 28  variable-length metadata<br>1 4 1 4 1 4<br>offsets 0 1 3 5<br>3 2 0 NULL 0 2 Posting 0101 1001 1 3 0011 1000 2 7<br>4 2 1 2 1 2 Batches values 0 2 6 0 3 4<br>6 3 1 2 1 3 Term Batch 0 Batch 1 Batch 2 Batch 3<br>0 NULL 0 Metadata meta-batch meta-batch … meta-batch freqs 1 2 2 1<br>1 3 1<br>Fig. 6: An example of the Cocoa for posting list = {1,, 3,, 7,, 17,, 19,<br><!-- End of picture text -->

**Fig. 6:** An example of the Cocoa for posting list = {1,, 3,, 7,, 17,, 19, 42, 43, 44, 50, 55} and B = 8. 

**Fig. 5:** Examples of representation schemes for a posting list. 

collect all matched documents. To address this challenge, Lucene assigns large segments to one thread slice while small segments coalesce into a single thread slice to eliminate thread-switching overhead. Tantivy takes a more straightforward approach by assigning one index segment per thread. 

Fig.4 illustrates the average latency with an increasing number of threads. We manually split the indexes into as many equal-size segments as the number of threads to enable assigning one or multiple segments per thread<sup>3</sup> . Surprisingly, Lucene performs even worse with increasing threads in AN, AR, and IT datasets. Tantivy performs slightly better scalability but falls short of expectations. Furthermore, dynamic workloads make the multi-thread performance even worse due to the segment merges. The merge policies aim to reduce segment count by merging near-sized segments into bigger segments, thereby reducing the write amplification and improving query efficiency. However, this may lead to a skewed distribution of segment sizes [37]. Consequently, the reduced number of segments with skewed sizes limits the efficiency of this intersegment parallelism model. The largest segment becomes the performance bottleneck, and the overhead from thread switching and data racing can even result in poorer performance than a simple single-thread execution. 

## _C. Implications_ 

We studied how current popular search libraries facilitate query processing on inverted indexes and investigated the root cause of performance issues in conducting set operations. Our measurement studies reveal that the performance deficiencies are mainly due to poor parallel support, both at the microinstruction level and the thread level. Additionally, we found that the compact list-encoding index format may contribute to low parallelism, as it is difficult or infeasible to construct query execution algorithms with high micro-instruction and threadlevel parallelism on top of it. These findings inspire us to design an inverted index-based query processing engine with better parallel support. To achieve this, designing a suitable index format and constructing a CPU-aware query processing scheme might be the solution. 

> 3This experimental setup is uncommon in practice; however, our goal is to explore the parallel behavior of Tantivy and Lucene under "ideal" conditions. 

## IV. COMPACT COLUMNAR FORMAT 

## _A. Maintaining Inverted List with Column Format_ 

Driven by the above findings, we attempt to design a suitable format for maintaining inverted lists, to achieve adorable parallel support. Recent years have witnessed great success with the columnar format used in data-intensive systems, achieving CPU-efficient query execution. We observed that inverted indexes could be viewed as a specialized form of columnar data structures, _i.e.,_ each posting list constitutes one column. This insight inspires us to organize inverted indexes in columnar format. Nevertheless, our primary investigation indicates that existing columnar formats do naturally suit our task. We find that mainstream columnar formats either follow the Placeholder or Compact layout [38]. The former ( _i.e.,_ Placeholder, adopted by Apache Arrow [39]) stores Null values in positions of nonexistent DocIds (Fig.5 (a)) and uses a bitmap to indicate the positions of Null values. This aligned structure makes it suitable for constant-time random access but tremendously wastes storage space. The latter ( _i.e.,_ Compact layout adopted by Apache Parquet [40] and ORC [41]) contiguously keep metadata associating the DocId list, and utilize a valid bitmap indicating the map ( _i.e.,_ DocIds and their associated metadata) (Fig.5 (b)). This unaligned structure is storage-efficient but inefficient for random and parallel access. 

Under such circumstances, this paper proposes the Compact Columnar Format (Cocoa) to achieve both space efficiency and effective CPU utilization simultaneously. Cocoa organizes inverted indexes as the columnar format with a Compact layout, and certain structures and mechanisms are designed to enable query efficiency. In Cocoa, the inverted index is organized as a large, wide columnar table with millions of columns, where each posting list forms a column. The DocId list acts as the alternative to the Null bitmap and employs the partitioned hybrid encoding mechanism, and the associated metadata is compactly organized in the column. The following sections describe in detail the structural design of Cocoa (§IV-B), as well as efficient metadata access approach (§IV-C) and basic operations (§IV-D) on top of it. 

## _B. Structural Design of Cocoa_ 

**Hybrid Encoding in Cocoa.** Bitmap encoding for DocIds enables efficient query processing through SIMD-optimized, branchless bitwise operations. However, it can be storageinefficient in cases with lower data density ( _i.e.,_ the cases when 

1804 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 

a term only appears in a few documents). In real-world corpora, both dense and sparse occurrences coexist—certain terms may appear frequently within some document groups, while showing low frequency in others. Thus, for the Cocoa design, we take both storage and query efficiency into consideration, _using a bitmap to indicate densely distributed and frequently accessed documents, whereas employing integer list to indicate sparse distributed and less-frequently accessed documents._ 

To implement this idea, we divide the postings (including DocIds and term metadata _etc._ ) of each term into batches, and choose the appropriate index representation ( _i.e.,_ bitmap or integer list) for each batch. A posting list is separated with batches of size _B_ , and each DocId _d_ in the list is assigned to the _⌊d/B⌋_ -th batch. Here, the DocId _d_ is represented by its offset of the start of the interval, computed as _d − B ∗⌊d/B⌋_ . 

The representation method is initially determined by storage efficiency. A bitmap representing a posting list of _m_ postings requires _B_ bits, while a sorted list of packed _⌊_ log2 _B⌋_ -bit integers requires _⌊_ log2 _B⌋· m_ bits. We define the selectivity ratio _B/_ log2 _B_ as the encoding _threshold_ . Cocoa adopts bitmap encoding for a batch when the number of postings _m_ exceeds this _threshold_ . Furthermore, the ratio _threshold_ for each batch can be dynamically adjusted according to workload. Cocoa employs a workload-aware layout adaptation mechanism that favors bitmap encoding for frequently accessed posting columns, while prioritizing storage efficiency for infrequently accessed columns. Cocoa tracks term access frequencies to reflect the term popularity. Periodically or when segment merging tasks (detailed in IV-D) is triggered, Cocoa identifies the top-k (determined by the memory budget) “hot” terms based on their access frequency. Adaptation tasks then lower the _threshold_ to prioritize bitmap encoding for “hot” terms and reset the _threshold_ for other “cold” terms. 

**Compact Layout of Cocoa.** Cocoa employs the Compact layout for both batched DocIds and associated term metadata. Posting batches that do not contain valid postings are not physically stored in the posting column. Instead, Cocoa adopts a _batch bitmap_ to denote the empty batch intervals and an offset list to indicate the start offset of each batch. Fig.6 illustrates an example of Cocoa when the _B_ = 8. In this case, the DocId can be bit-packed as 3-bit data (as the relative offsets are less than 8). We encode the batch as the bitmap for memory efficiency when the batch size is larger than 2. Cocoa divides the raw posting list into four valid batches whose interval range can be indicated by the _batch bitmap_ . The _offsets_ record the start positions of posting batches. Furthermore, each posting batch is associated with a term metadata batch. The fixedlength metadata ( _e.g.,_ term frequency) is physically arranged in a contiguous array. The variable-length metadata ( _e.g.,_ term positions) can be represented by the compact position value array. Term positions in a document can be accessed by combining the start offset and frequency. 

**Term index and Column Metadata.** With the aforementioned designs, the posting lists ( _i.e.,_ indicating which documents contain each term) are organized into compact columnar 



<!-- Start of picture text -->
Term Metadata<br>2 8 3 1 0 Batch0 0 0<br>M Write Mask S<br>0 1 1 0 1 0 0 1<br>Valid Mask<br>①PEXT 1 1 0 1 0 0 0 0<br>S Posting Batch0 1 0 0 1 0 1 1 M ②compress<br>Selected Metadata<br>③PEXT 2 8 1 0 0 0 0 0<br>Align Mask<br>1  0 1 1 0 0 0 0 ④expand<br> Valid Metadata<br>2 0 8 1 0 0 0 0<br><!-- End of picture text -->

**Fig. 7:** Extract term frequencies from metadata batch with the _posting batch_ and _write mask_ . The _posting batch_ refers to the bitmap-encoded posting batch, and the _write mask_ indicates the positions of the term frequencies to be extracted. The _s_ represents the source operand, and _m_ indicates the mask operand of PEXT. 

data, forming a wide columnar table. The term index acts as an auxiliary data structure that maps each term to its associated column metadata. For each column, the _column metadata_ records information such as memory address, cardinality ( _i.e.,_ the amount of DocIDs contained in the column), and the selectivity (the ratio of the amount of DocIDs to the column length) of the column, etc. Such information can be utilized by the query optimizer. 

**Determining Batch Size.** The batch size _B_ introduces tradeoffs in both memory usage and query processing efficiency. For storage, a larger batch size requires additional bits to represent DocId indicators in integer lists and stores more zeros in bitmap indexing, while a smaller batch size needs more bits to store batch metadata. For alinement concerns, _B_ must be the multiples of SIMD widths ( _e.g.,_ 512 bits in AVX512, 256 bits in AVX2/AVX, and 128 bits in NEON). We set _B_ as 512 to achieve balanced storage and query processing overhead. Note that other multiples of the number of SIMD lanes are also acceptable for different workloads ( _e.g.,_ set _B_ = 1024 for high-density inverted index). 

## _C. Extract Term Metadata_ 

The term metadata is crucial in inverted index-based systems. For example, in full-text search, term metadata such as term frequencies and positions are used to determine and rank the relevance of documents to a search query. The hybrid encoding scheme presents a significant challenge to extracting term metadata from bitmap-encoded batches. This challenge has led to the widespread replacement of bitmapbased indexes, which had been adopted in search engines, with sorted-list encoding [42]. One reason is that sorted list-based inverted indexes can seamlessly incorporate term metadata by simply iterating the posting lists, whereas bitmap-based indexes suffer from amplified traversing overhead. 

To eliminate the traversal and manipulation overhead in bitmap-based operations, we utilize the BMI (Bit Manipulation Instruction). This extended instruction set of the X86 architecture, supported by Intel<sup>®</sup> and AMD CPUs, enables bit-level parallelism in bit manipulation. The PEXT (parallel bit extract) instruction in BMI is the core operation of our 

1805 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 



<!-- Start of picture text -->
Algorithm 1: PostingColumn::Next( ) Packed-Integer List List 1 5 6 8 11 14<br>Input: Batch bitmap bb; Alignment bitmap ab; Bitmap encoding Primitives 2 3 5 9 10 12<br>Posting column pc. 1 0 0 1 0 1 0 0<br>Bitwise<br>Output: The next selected posting batch. Primitives 1 0 0 1 0 1 0 0<br>1 // Get the position of the next set bit Arbitrary encoded  Hybrid 2 4 6<br>2 n ← ab.next() A B C batches Primitives 0 0 1 1 0 1 1 0<br>3 if n == Null then Interpreted Primitives<br>4 return Null // Indicate the termination Adaptive<br>5 if bb.contain(n) then primitives at selection of  func eval(…) {  for i in 0.. {<br>6 off = pc.offsets[rank(bb, n)] runtime     res[i] = A & B;    if res[i] == 0 {<br>      continue;<br>7 return deserialize(pc[off])     }<br>8 else Aligned  and      res[i] &= C;  } CodeGen<br>9 return EmptyBatch Selected Batches Unified encoded batches }<br>Pre-compiled Fused Primitives<br>(1) Align and Select Batches. (2) Optimize the Execution Plan.  (3) Evaluate the Execution Plan.<br><!-- End of picture text -->

**Fig. 8:** The query processing model of VeloSearch. 

optimized extraction algorithm. This instruction extracts bits selected by a _select mask_ operand _M_ from a source operand _S_ and copies them to the contiguous low-order bits in the destination, with the high-order bits set to 0s. 

Fig.7 depicts the algorithm for extracting fixed-length ( _e.g.,_ term frequency) or variable-length ( _e.g.,_ term positions) term metadata from metadata batches. The _write mask_ , evaluated by the search query, indicates matched DocIds in a batch, guiding the algorithm in determining which term metadata should be extracted. The PEXT instruction ( 1 _⃝_ ) copies all selected bits of the _write mask_ into the _valid mask_ . Here, the _write mask_ serves as the source operand, and the posting batch acts as the mask operand. The _valid mask_ indicates the selected metadata, with a set bit at index _j_ signifying that the _j_ -th metadata is selected. The compress operations ( 2 _⃝_ ) arrange the selected term metadata contiguously in the _selected metadata_ according to the _valid mask_ . In the _⃝_ 3 , an _align mask_ , produced by the PEXT instruction with exchanged operands compared to the first step, serves as the input for the expand operation. This operation ( 4 _⃝_ ) places the compressed metadata into specific positions indicated by the _align mask_ . The compress and expand operations were tedious tasks that often induced significant overheads, such as additional accesses to predefined lookup tables [43]. However, compress and expand instructions in modern SIMD instructions, such as AVX512, enable these operations to be executed on a 512-bit metadata batch in one or several SIMD instructions. 

This extraction method facilitates not only the fixed-length but also the variable-length term metadata ( _e.g.,_ term positions). The variable-length term metadata can be located using a fixed-length start _offset_ and the length of the metadata entry ( _i.e.,_ the _frequency_ ), as illustrated in the right part of the Fig.6. Consequently, we can access the variable-length metadata by extracting associated _offsets_ and _frequencies_ . 

## _D. Basic Operations_ 

Cocoa is a customized columnar format designed for vectorization, providing a batch-at-a-time access interface. We provide basic interfaces to access and modify index data: 

Get. Cocoa develops the PostingBatch operator to provide the batch-at-a-time access interface for Cocoa. Algorithm 1 illustrates the next function of the PostingBatch operator. The PostingBatch operator is initialized with a batch bitmap (bb), which denotes the valid batch intervals, and an alignment bitmap (ab), which indicates the selected batch intervals. The next function begins by identifying the position _n_ of the next set bit in bb. If all the selected batches have been pulled, this function returns Null. If there are no valid batches in the range _n_ for PostingBatch, it returns EmptyBatch (Line 9), which represents a bitmap with all set bits for AND operators and all unset bits for OR operators. The rank primitive (Line 6) can obtain the batch interval range ( _i.e., n_ ) corresponding to which contiguously organized batch. Denoted as _rank_ ( _bb, n_ ), it counts the number of 1’s in the batch bitmap up to position _n_ . The count _b_ obtained from the rank primitive corresponds to the value at offsets[b] (the batch offsets maintained in column metadata), which indicates the start position of the selected batch in the posting column. It then decodes and produces the indicated posting batch (Line 7). 

Delete. When deleting a document from Cocoa, the document data in a segment of Cocoa is not immediately inplace deleted. To handle deletion operations, each segment of Cocoa maintains a bitmap that tracks which DocIds are live and which have been deleted. Thus, delete operations simply unset corresponding bits in the bitmap. The actual reclamation of deleted documents occurs during segment merges. 

Put (Insert/Update). Cocoa employs the out-of-place update mechanism similar to Lucene and Tantivy. Cocoa first indexes and maintains newly updated documents in a memory buffer. This memory buffer is periodically or manually flushed to a new self-contained posting segment. After the flush operation, the newly inserted documents are visible to subsequent read queries. The flush operations may lead to one or more segment merges, as determined by the merge policy, to reduce space amplification and improve query performance. Cocoa employs the LogMergePolicy, which tries to merge segments into levels 

1806 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 



<!-- Start of picture text -->
List B<br>16-bit int 1 2 9 11 16 20 23 34 bitmask result<br>1 1 0 0 0 0 0 0 0 1 1<br>6 0 0 0 0 0 0 0 0 0 20<br>7 0 0 0 0 0 0 0 0 0 34<br>11 0 0 0 0 0 0 0 0 0 0<br>18 0 0 0 0 0 0 0 0 0 0<br>20 0 0 0 0 0 1 0 0 1 0<br>29 0 0 0 0 0 0 0 0 0 0<br>34 0 0 0 0 0 0 0 1 1 0<br>List A ①Vectorized Compare<br>②Compress<br>(compare equal any)<br><!-- End of picture text -->



<!-- Start of picture text -->
Stage #1 - Enumeration Stage #2 - Translation Stage #3 - Compilation<br>0000 00000 00000 ∧ func eval(..) { for i in 0.. { 0 -<br>  res[i] = A & B; 1 -<br>0101 10000 11110 Decode ∨ C D Translator   if res[i] == 0 {     continue;  }  ….  res[i] &= D; Compiler 0x161E Binary Code<br>1111 11111 11111 A B  }} 0xFFFF -<br>num LOUDS HasChild Physical Expression Intermediate Code Lookup Table<br>…<br>…<br><!-- End of picture text -->

**Fig. 10:** Ahead-of-time compilation of short-circuit primitives. 

**Fig. 9:** Vectorized intersection on list-encoded batches. 

of exponentially increasing size, where each level has smaller segments than the value of the merge factor. Besides the insert operation, the update operation first deletes the document and then inserts the new one. 

## V. VECTORIZED QUERY PROCESSING 

We develop the VeloSearch to facilitate vectorized query processing with Cocoa. Benefiting from the batch-at-a-time access method, computational actions can be implemented as primitive functions that operate exclusively on two batches of a specific encoding type in a vectorized style. As shown in Fig.8, VeloSearch employs the precompiled vectorization approach, consisting of two types of primitives. The interpreted primitives (§V-A) are encoding-specialized and dynamically dispatched to execute across two posting batches. A pre-compiled process (§V-B) is proposed to enumerate and compile primitives ahead of time (AOT) for Boolean expressions, fusing multiple primitives to minimize materialization overhead and exploit the short-circuit features of Boolean expressions. Additionally, a query optimizer (§V-C) is designed to optimize the execution plan and exploit the intra-query parallelism. 

## _A. Interpreted Primitives_ 

VeloSearch employs the dynamic dispatching mechanism to select appropriate primitives for vectorized batch evaluations. Cocoa employs a hybrid encoding mechanism, adopting packed-integer encoding for sparse batches and bitmap encoding for dense batches. Consequently, as illustrated in Fig.8, when performing primitives on two batches, three situations should be considered: 

**Both batches are bitmap-encoded.** It is straightforward to implement the primitives using bitwise operations ( _e.g.,_ AND, OR and XOR) for both bitmap-encoded batches. We can manually use SIMD instruction in function codes such as the ANDPS in SSE and the _mm_and in AVX to implement these primitives. Nevertheless, it is sufficient to enable the compilers to auto-vectorize the bitwise operations by compiler option hints. Compilers can identify when instructions within a loop can be rewritten as vectorized operations. 

**Both batches are list-encoded.** Implementing vectorized primitives for list-encoded batches presents greater challenges. Fortunately, modern SIMD instruction sets provide intrinsics that facilitate branchless batched comparisons. We leverage the 

string comparison instruction cmpestrm, supported in SSE 4.2 and widely available in modern Intel<sup>®</sup> and AMD CPUs, to perform the vectorized comparisons: 

```
__m128ibitmask=_mm_cmpestrm(
```

```
a,a.len(),b,b.len(),
```

```
_SIDD_UWORD_OPS|_SIDD_CMP_EQUAL_ANY|
```

```
_SIDD_BIT_MASK);
```

The flags to this intrinsics specify that it compares elements of two lists of 16-bit characters (_SIDD_UWORD_OPS) in an equal-any fashion (_SIDD_CMP_EQUAL_ANY) and produces a bitmask (_SIDD_BIT_MASK) to indicate the comparison result. Fig.9 depicts the vectorized intersection algorithms. The process starts by unpacking the compressed posting lists into 16-bit integer lists. Then, the cmpestrm performs the vectorized comparisons between these two lists and produces an intermediate comparison matrix, which holds the results of the full comparisons and aggregates the values into a bitmask in an equal-any fashion. This process eliminates branch predictions and exploits data-level parallelism through dedicated hardware instructions. The compress operation then contiguously copies the values from posting lists, as the bitmask indicates, into the result list. Additionally, VeloSearch also supports integrating other existing vectorized intersection algorithms such as [8, 23]. 

**One batch is a posting list, while the other is a posting bitmap.** The primitives specialized for hybrid encoding batches traverse posting lists and probe their value to determine their presence in the posting bitmap using bitwise manipulations. Based on the probe results, the hybrid primitives collect the evaluated result. 

## _B. Pre-compiled Fused Primitives_ 

Vectorized query processing efficiently amortizes the interpretation overhead but necessitates additional load and store instructions to materialize evaluated results into intermediate batches [20]. To address this issue, we design an Ahead-ofTime (AOT) enumeration process to compile primitives that fuse multiple operations. The core idea is to encode Boolean expressions as numerical values, enumerate these encoded values to derive potential Boolean expressions, and compile the corresponding fused primitives. 

We use the succinct Level-Ordered Unary Degree Sequence (LOUDS) [44] to encode the expression trees in the level order 

1807 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 

as numerical values. This encoding mechanism consists of two bit-sequences. The set bits in _Louds_ sequence indicate the node boundaries, marking the first node in every sub-tree. The auxiliary bit-sequence _HasChild_ uses one bit for each node to indicate whether a child branch continues or terminates. We encode Boolean expressions as a fixed 32-bit integer to further compact the encoding. The first 4 bits represent the node count of the Boolean expression, followed by two 14bit values for encoding the _Louds_ and _HasChild_ attributes individually. This design accommodates the representation of all possible Boolean expressions with node counts up to 14. 

As shown in Fig.10, the AOT compilation process consists of three stages. The first stage enumerates all possible Boolean expressions with node counts up to 14 by incrementing the value of _Louds_ . The second stage translates these encodings into valid Boolean expression trees and converts them into the Intermediate Representation (IR). To eliminate the exponential enumeration space, we precompile two types of primitives for each enumerated expression that can evaluate batches with the same encoding. At runtime, the evaluation engine chooses between interpreted and precompiled primitives based on the batches’ encodings. The final stage compiles the IR into native code using compiler infrastructures ( _e.g.,_ LLVM [45]) and stores the generated code in the primitive lookup table. When evaluating a Boolean expression using the precompiled fused primitive, the process begins by identifying the _Louds_ encoding corresponding to the Boolean expression and then looking up the primitive in the primitive lookup table. 

## _C. Query Optimizer for Search Queries_ 

The query optimizer attempts to determine the most efficient way to execute a given query. The optimizer of VeloSearch employs heuristic rewrite rules and evaluation optimizations to enhance the execution plan. 

**Simplify and rewrite the search queries.** This rewrite rule focuses on simplifying expressions and gathering relevant statistics, such as the number of children and selectivity of nodes. The operator trees are simplified and normalized to ensure that the parent of each AND node must be an OR node, and vice versa. This interleaving of AND and OR operators across different levels of the operator tree facilitates more efficient query evaluation. The number of children and selectivity of nodes in operator trees are crucial for subsequent query optimization. We assume that the selectivities of terms are independent, which allows us to estimate the combined selectivity for conjunctive and disjunctive expressions through equations such as _sel_ ( _A ∧ B_ ) = _sel_ ( _A_ ) _· sel_ ( _B_ ). Following a similar approach to the work [46], the optimizer reorders expressions according to _weight_ = ( _si −_ 1) _/fi_ where _si_ represents the estimated selectivity and _fi_ denotes the children count of the _i_ -th clause. A higher _weight_ indicates a higher evaluation cost, which suggests that evaluating these clauses later and potentially bypassing them is more efficient. 

**Pruning invalid posting batch intervals.** The query optimizer identifies valid batch intervals for given search queries 



<!-- Start of picture text -->
Queries Collector<br>Scored Batch<br>Query Parser Extract & Scorer Batch Scorer<br>Query Rewrite Extract Meta<br>Posting Batch<br>Branchless<br>OR Operators<br>Batch-at-a-time<br>Query Optimizer(§V-C) ProcessingVectorized C (A AND B)<br> Vectorized Operators<br>Select and Align Valid Batches<br>(§V-A, §V-B)<br>TableProvider<br>Term<br>Index<br> Cocoa (§IV)<br>Compacted Columnar Format<br>Frontend<br>Backend<br>DataSource<br><!-- End of picture text -->

**Fig. 11:** System Architecture of VeloSearch. 

to minimize the number of posting batches accessed. For example, in a search query “A AND B”, batch intervals are considered relevant if they contain valid values for both A and B. As the _batch bitmap_ indicates valid batch intervals of terms, the optimizer initially evaluates the Boolean expression by the _batch bitmaps_ of the terms involved. This evaluation result is an _alignment bitmap_ representing the valid batch ranges for the given search query. The _alignment bitmap_ is then used as input for the algorithm depicted in Fig.7, aligning and selecting the valid batches for each relevant term. 

**Intra-query parallelism.** The optimizer enhances the performance of the given query by executing the query plan in parallel. In addition to the inter-segment parallel model adopted by Tantivy and Lucene, the optimizer of VeloSearch also exploits the intra-segment parallelism. Thanks to Cocoa, the tasks that need to be executed are represented by the _alignment bitmap_ . The optimizer divides these tasks into nonoverlapping sub-tasks based on the _alignment bitmap_ and assigns them to different threads for parallel execution. This parallel model leverages intra-segment parallelism, enabling more fine-grained and load-balancing parallelism. 

**Selection of primitives at runtime.** The pre-compiled fused primitives are designed to evaluate batches with the same encoding. Therefore, VeloSearch performs the adaptive selection of primitives at runtime, based on encodings the batches pulled each time. To eliminate the overhead of the dynamic dispatching mechanism, the optimizer sets hints during the traversal of expressions at positions that may meet the conditions for executing fused primitives ( _e.g.,_ node count less than 14). At runtime, the evaluation engine simply checks the encoding of the batches at the expression nodes marked with hints and selects the appropriate primitives accordingly. 

## VI. IMPLEMENTATION AND EVALUATION 

## _A. Implementation_ 

We developed VeloSearch and Cocoa on top of Apache DataFusion [47] and Apache Arrow [39] with 8k LOC Rust. Apache DataFusion is an extensible query engine written by 

1808 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 

**TABLE III:** Search latency and memory usage over six datasets. 



<!-- Start of picture text -->
AN AR LG SSB WP IT<br>Library<br>CNF DNF Usage CNF DNF Usage CNF DNF Usage CNF DNF Usage CNF DNF Usage CNF DNF Usage<br>Lucene 99 310 38MB 439 4,004 206 MB 4,823 8,462 232 MB 319,786 375,199 486 MB 1,301 7,370 2.5 GB 1,747 18,282 11 GB<br>Tantivy 119 254 44 MB 480 2,062 222 MB 2,078 4,619 253 MB 293,667 233,077 674 MB 913 5,295 2.89 GB 1,359 13,107 13 GB<br>VeloSearch 85 203 60 MB 126 263 178 MB 168 244 276 MB 9,636 10,197 376 MB 415 894 2.79 GB 419 1,753 15.5 GB<br>Velosearch Tantivy Lucene Lucene Tantivy VeloSearch Extraction<br>Lower is better<br>1<br>4000<br>0<br>Higher is better 2000<br>5<br>0<br>AN AR WP IT<br>0<br>0% 10% 20% 30% 40% 50% Datasets<br>Write ratio [%]<br>[ms]<br>Read Latency<br>Execution Time [s]<br>[kops/s]<br>Throughput<br><!-- End of picture text -->

**Fig. 13:** Phrase queries utilize term frequencies and positions. _Extraction_ indicates the extracting metadata overhead. 

**Fig. 12:** Throughput and read latency with increasing write ratios. 



<!-- Start of picture text -->
15<br>Velosearch<br>Tantivy<br>10 Lucene<br>5<br>0<br>2 4 6 8 10<br>clause number<br>Execution Time [ms]<br><!-- End of picture text -->

Rust that uses Apache Arrow as its in-memory format. The customizable feature powered by DataFusion enables us to build domain-specific query engines utilizing well-engineered, reusable components. Apache Arrow has been an end-to-end, in-memory columnar format adopted by many data analysis systems [48, 49]. We implement Cocoa based on Apache Arrow for better compatibility and generality with Apache DataFusion and other data-intensive systems. As illustrated in Fig.11, VeloSearch comprises three key components. The Frontend parses and rewrites search queries into execution plans. It bridges the search queries and the execution representation ( _i.e.,_ logical plan) of DataFusion. The Backend optimizes execution plans and evaluates them through the vectorized operators. Cocoa serves as a customized columnar DataSource by implementing the TableProvider trait. For each Next call, it supplies one selected interval of posting batches. 

**Fig. 14:** Execution time over the increasing number of clauses in each complex nested Boolean query. 

of these terms. VeloSearch achieves 1.53 _×_ to 37.6 _×_ over Lucene and 1.25 _×_ to 22.9 _×_ over Tantivy. Compared to CNF queries, the performance improvement of VeloSearch when executing DNF queries is more significant, DNF queries require traversing almost all relevant posting lists to perform union operations, which benefits from the bitmap encoding layout and vectorized query processing. The space usage of Cocoa is also comparable to that of Lucene and Tantivy. This is partly because Cocoa employs a memory-optimized hybrid encoding mechanism. Cocoa achieves lower memory usage than Lucene and Tantivy in the SSB and AR datasets. For these dense datasets, where the number of terms is significantly lower than the number of words and documents, the bitmapencoding mechanism achieves better space efficiency. 

## _B. Evaluation Methodology_ 

We conduct extensive evaluations on VeloSearch. Experiments are conducted with the same setups as in _§_ III-A. Our evaluation aims to examine (1) the performance of VeloSearch in query processing both at end-to-end and fine-grained levels (§VI-C), and (2) the effectiveness and overhead of each mechanism we designed (§VI-D). 

## _C. Operational Performance_ 

**End-to-end performance.** We compare the query latency and space usage of VeloSearch, Tantivy, and Lucene over six workloads as illustrated in Table.III. Conjunctive Normal Form (CNF) search queries are conjunctions of multiple terms that match the documents that contain all of these terms. VeloSearch achieves a speedup of 1.15 _×_ to 33.2 _×_ over Lucene and 1.4 _×_ to 30.5 _×_ over Tantivy on average. We observe that VeloSearch performs particularly well with large datasets, such as LG, SSB, and IT, because the optimizations in Cocoa and vectorized execution are more effective in handling the increased intersection operations. Disjunctive Normal Form (DNF) search queries match the documents containing any 

**Performance with writes.** We employ the _Wikipedia_ dataset to construct a workload containing 0.3M documents and an additional 0.2M documents as writing workloads. We evaluate the query performance on this mixed workload with increasing write ratios. The ratio among different types of writes is constant (1:1:2 for insert, deletion, and update) to keep the dataset size stable throughout the evaluation. After each write operation, we flush the changes to a new posting segment so that the subsequent read operation will see the changes. As depicted in Fig.12, the query throughput shows the performance of the read-write mixed workload, while the read latency reflects the degradation in read performance impacted 

1809 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 



<!-- Start of picture text -->
AN WP<br>4<br>AR IT<br>LG SM1<br>SS SM2<br>2<br>1 4 16 64 256 1k 4k 8k Max<br>Batch size (#Posting Batches)<br>Execution Time  Relative to 1K.<br><!-- End of picture text -->

**Fig. 15:** Execution time with increasing interval size, which are normalized by interval size of 1k. Besides the six datasets, we simulate two toy datasets with an average posting list length of 1M and 10M, denoted as _SM1_ and _SM2_ . 

by writes. Although this mixed workload maintains a stable indexed dataset size, write operations increase the segment count and may trigger the segment merge tasks, affecting the read performance. Compared to the read-only workload ( _i.e.,_ 0% write ratio), all systems exhibit higher read latencies and lower throughput with the write ratios increasing. However, VeloSearch achieves the lowest latency for all listed write ratios, up to 1.5 _×_ and 2.3 _×_ faster than Lucene and Tantivy. The query throughput of VeloSearch is significantly better than that of Tantivy and is comparable to Lucene. 

**Performance with term metadata.** Phrase queries match documents containing a specific sequence of terms, requiring the term positions and frequencies to both match and score the documents. This type of comprehensive query serves as an effective benchmark for evaluating the metadata extraction capabilities of VeloSearch. The iterator execution model can seamlessly integrate the term metadata alongside the next() call, while VeloSearch requires an additional process ( _§_ IV-C) to extract the term metadata based on evaluated batches. To assess the extraction overhead, we conduct experiments using phrase queries on information retrieval workloads. As depicted in Fig.13, the cost of extracting term metadata accounts for 20% to 66% of the total execution time. Despite the metadata extraction overhead, VeloSearch still achieves significant performance improvement, delivering up to 2.7 _×_ and 4.2 _×_ better performance compared to Tantivy and Lucene, respectively. 

**Performance with complex queries.** We construct complex, nested Boolean queries by disjuncting multiple CNF queries using the _wikipedia_ workload. As depicted in Fig.14, VeloSearch exhibits an increasing performance advantage as the number of CNF clauses in the complex queries increases. This is because the increased query complexity amplifies the interpretation overhead inherent in the iterator model used by Tantivy and Lucene. In contrast, VeloSearch benefits from its vectorization and optimized pruning of accessed posting batches, achieving speeds 2.1 _×_ and 3 _×_ faster than Tantivy and Lucene, respectively. 

**Out-of-memory experiments.** Previous experiments were conducted in an in-memory setting ( _i.e.,_ inverted index was fully resided in memory). However, large-scale inverted indexes often exceed the memory capacity, so we also evaluate performance when Cocoa is stored in secondary storage. To 



<!-- Start of picture text -->
Interpreted Fused<br>0.4<br>0.2<br>0.0<br>AN AR LG SS WP IT AN AR LG SS WP IT<br>(1) CNF (2) DNF<br>Reduction of<br> Execution Time [%]<br><!-- End of picture text -->

**Fig. 16:** Execution time reductions of interpreted primitives and fused primitives compared to the _scalar_ implementation. 

**TABLE IV:** Out-of-memory experiments when Cocoa is stored in the secondary storage 

||AN|AR|LG|SS|WP|IT|
|---|---|---|---|---|---|---|
|Lucene|1,467|3,880|35,046|804,112|3,759|3,251|
|Tantivy|286|1029|24,778|294,079|1,963|2,062|
|**VeloSearch**|351|904|798|12,232|1,922|1,882|



achieve this, we store the Cocoa as files in NVMe SSDs, which offers a read bandwidth of 5GB/s. We utilize the mmap system call to map the stored Cocoa data into virtual memory. Before each experiment, we clear the system page cache in the main memory. The results show that even when the index data is stored on the SSD, VeloSearch still achieves significant performance improvements ranging from 1.13 _×_ to 67 _×_ . 

**Performance under variable selectivity.** To investigate how query selectivity affects vectorization and space usage, We simulate a dataset containing 0.5 million documents with only three terms. We evaluate the performance of conjunctive queries for VeloSearch, Tantivy, Lucene, and HandCoded. The HandCoded adopts an optimized merge-based intersection algorithm on uncompressed DocId lists, serving as a baseline comparison. As depicted in Fig.17, the execution time of VeloSearch is close to that of Tantivy and significantly better than Lucene when query selectivity is below 1%. After that point, the execution time reduces and then stabilizes, which is even better than HandCoded. This is because, at low query selectivity, each processed posting batch is too sparse, causing it to degrade to the iterator model. At high query selectivity, batches are almost entirely bitmap-encoding, leading to more efficient processing with a similar amount of calculation. Similarly, the space usage of VeloSearch is initially slightly larger than Tantivy’s and similar to Lucene’s, but it decreases as selectivity increases, benefiting from the bitmap encoding. 

## _D. Effectiveness of Designs_ 

**Vectorized query processing.** We configure the batch size to demonstrate the benefits of batch-at-a-time query processing. So far, our VeloSearch experiments have used a value of 512 batches. Fig.15 shows normalized query runtime for batch sizes from 1 to the maximum ( _i.e.,_ full materialization like MonetDB [50]). We observe that small batch sizes (< 64) decrease performance significantly. With a batch size of 1, VeloSearch degrades the _Volcano_ model with its large CPU overhead. As the batch size increases, performance improves 

1810 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 



<!-- Start of picture text -->
Velosearch Tantivy Lucene HandCoded<br>1.5 600<br>1.0 400<br>0.5 200<br>0.0 0<br>5% 10% 15% 5% 10% 15%<br>Selectivity Selectivity<br>Execution Time [ms] Space Usage [MB]<br><!-- End of picture text -->

**Fig. 17:** Execution time and space usage with increasing query selectivity for a conjunctive search query. 

for all datasets except SM1 and SM2 but eventually levels off once the batch size exceeds the number of batches in the associated _posting columns_ . The large _posting columns_ of large-scale SM1 and SM2 cannot entirely fit into the CPU caches and therefore decrease performance when batch size is MAX. Generally, the batch size 512 seems to be a good setting for all queries. Fig.16 shows that the _interpreted_ primitives and pre-compiled _fused_ primitives further speeds the process compared to the naive _scalar_ implementation. The _scalar_ approach uses scalar primitives without manual SIMD intrinsics but with compiler auto-vectorization. The results show that our proposed interpreted primitives improve performance by 10% to 30%, while the pre-compiled fused primitives can further enhance execution performance. 

**Multi-threaded Execution.** As discussed in the §III-Issue 3, the parallel models of Lucene and Tantivy are sensitive to the number of segments and the distribution of segment sizes. However, our proposed intra-segment parallel model of VeloSearch can achieve scalable performance without segment distribution and count limits. As illustrated in Fig.18, we conduct experiments on a simulated dataset containing 10M documents to evaluate the effectiveness of the intra-query parallelism. For an even distribution of segment sizes, we observe that VeloSearch benefits from the parallel model even when the number of threads is more significant than the number of segments. In contrast, the parallelism of Lucene and Tantivy is constrained by the segment count. In the log-normal distribution case, Lucene and Tantivy’s parallel capability further diminishes when the threads are less than the segment count. The scalability of intra-query parallelism is limited at the high number of threads, where the overhead from thread switching and data racing can eliminate the improvement. 

## VII. RELATED WORKS 

**Intersection on inverted indexes.** Intersection on sorted lists is a crucial operation in inverted indexes. BMiss [24] employs larger block size to reduce the branch mispredictions when the sizes of lists are relatively similar and the intersection is less selective. _Fast_ [23] partition input sets into smaller subsets and employs hash functions to map the subsets to a reduced domain for comparison. Bitlist [6] uses an encoded number to represent a set of DocIds with lower space cost. These algorithms serve as efficient execution primitives to perform vectorized intersections on inverted lists. Different from these works, VeloSearch employs the columnar format Cocoa for 



<!-- Start of picture text -->
Tantivy Lucene Velosearch<br>10 15<br>10<br>5<br>5<br>0 0<br>1 2 4 8 16 32 64 1 2 4 8 16 32 64<br>Threads Threads<br>(1) Uniform distribution. (2) Log-normal distribution.<br>Execution Time [ms]<br><!-- End of picture text -->

**Fig. 18:** Query performance with increasing threads for different segment sizes distributions: (1) an even distribution with four segments of equal size. (2) four segments with sizes following a logarithmic gradient reduction. 

inverted lists to provide batched posting access interfaces. Cocoa enables performing bitwise operations on bitmap-encoded posting batches and degrades to perform list-based branchless operations when evaluating both list-encoding batches. 

**Query processing approaches of inverted indexes.** The iterator model employed by search engines was introduced by the work [51]. This model then plays a fundamental role in all query processing strategies [17]. Numerous efforts have been made to optimize this iterator model through enhancing rank retrieval capability [52, 53], index compression [54, 55], and document reordering [6, 56]. In contrast, VeloSearch organizes inverted indexes in columnar formats to employ the vectorized query processing, addressing the issues inherent in the iterator execution model adopted by modern search engines. 

**Columnar Storage Formats.** Widely adopted columnar formats, such as Parquet [40] and Apache ORC [41], incorporated insights from previous columnar storage research like the PAX (Partition Attribute Across) [57] model and format optimization in Dremel [58] including record-shredding and assembly algorithm. Unlike these general-purpose columnar formats, The work [59] proposes a columnar layout tailored to document stores for storing unstructured data. The columnar CSR [60] is proposed for contemporary graph database management systems. Our proposed Cocoa is a columnar format specifically designed for inverted indexes, which employs the Compact layout and hybrid encoding mechanism. 

## VIII. CONCLUSION 

This paper investigates the performance issues of existing search libraries in relation to CPU awareness, and proposes to utilize columnar format maintaining inverted lists to achieve both desirable space efficiency and maintain the capability for vectorized searching support. We design the Cocoa data format to achieve this goal, construct efficient index manipulation algorithms, and build a full-text search system, VeloSearch, on top of Cocoa. Extensive evaluations show that VeloSearch outperforms existing systems by an order of magnitude. 

## ACKNOWLEDGMENTS 

We thank the anonymous reviewers for their constructive comments and suggestions. This work is supported by NSFC Project No. 62472173 and No. 62137001, NSF of Shanghai No. 23ZR1418300 and Shanghai Sailing Program No. 23YF1410600. 

1811 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 

REFERENCES 

generator: Extensibility and efficient search,” in _ICDE_ . IEEE, 1993, pp. 209–218. 

- [1] V. Raman, L. Qiao, W. Han, I. Narang, Y.-L. Chen, K.-H. Yang, and F.-L. Ling, “Lazy, adaptive rid-list intersection, and its application to index anding,” in _SIGMOD_ , 2007, pp. 773–784. 

- [2] J. Wang, C. Lin, R. He, M. Chae, Y. Papakonstantinou, and S. Swanson, “Milc: Inverted list compression in memory,” _PVLDB_ , vol. 10, no. 8, pp. 853–864, 2017. 

- [3] M. Widmoser, D. Kocher, and N. Augsten, “Scalable distributed inverted list indexes in disaggregated memory,” _SIGMOD_ , vol. 2, no. 3, pp. 1–27, 2024. 

- [4] M. Curtiss, I. Becker, T. Bosman, S. Doroshenko, L. Grijincu, T. Jackson, S. Kunnatur, S. Lassen, P. Pronin, S. Sankar _et al._ , “Unicorn: A system for searching the social graph,” _PVLDB_ , 2013. 

- [5] X. Zhou, K. Huang, L. Li, M. Zhang, and X. Zhou, “I/oefficient multi-criteria shortest paths query processing on large graphs,” _IEEE Transactions on Knowledge and Data Engineering (TKDE)_ , 2024. 

- [6] W. Rao, L. Chen, P. Hui, and S. Tarkoma, “Bitlist: New full-text index for low space cost and efficient keyword search,” _PVLDB_ , vol. 6, no. 13, pp. 1522–1533, 2013. 

- [7] G. E. Pibiri and R. Venturini, “Techniques for inverted index compression,” _ACM Computing Surveys (CSUR)_ , vol. 53, no. 6, pp. 1–36, 2020. 

- [8] J. S. Culpepper and A. Moffat, “Efficient set intersection for inverted indexing,” _ACM Transactions on Information Systems (TOIS)_ , 2010. 

- [9] S. Kim, T. Lee, S.-w. Hwang, and S. Elnikety, “List intersection for web search: Algorithms, cost models, and optimizations,” _PVLDB_ , vol. 12, no. 1, pp. 1–13, 2018. 

- [10] P. Yang, H. Fang, and J. Lin, “Anserini: Reproducible ranking baselines using lucene,” _Journal of Data and Information Quality (JDIQ)_ , vol. 10, no. 4. 

- [11] J. Xian, T. Teofili, R. Pradeep, and J. Lin, “Vector search with openai embeddings: Lucene is all you need,” in _WSDM_ , 2024. 

- [12] R. Liu, J. Liang, P. Jin, and Y. Wang, “Mmh-index: Enhancing apache lucene with high-performance multimodal indexing and searching,” in _MM_ , 2022. 

- [13] T. Neumann, “Efficiently compiling efficient query plans for modern hardware,” _PVLDB_ , vol. 4, no. 9, pp. 539– 550, 2011. 

- [14] A. Kohn, V. Leis, and T. Neumann, “Adaptive execution of compiled queries,” in _ICDE_ . IEEE, 2018, pp. 197– 208. 

- [15] Apache Lucene, “Apache Lucene,” 2024. [Online]. Available: https://lucene.apache.org/ 

- [16] “Tantivy,” 2023. [Online]. Available: https://github.com /quickwit-oss/tantivy 

- [17] N. Tonellotto, C. Macdonald, I. Ounis _et al._ , “Efficient query processing for scalable web search,” _Foundations and Trends® in Information Retrieval_ , vol. 12, no. 4-5, pp. 319–500, 2018. 

- [18] G. Graefe and W. J. McKenna, “The volcano optimizer 

- [19] B. Wagner, A. Kohn, P. Boncz, and V. Leis, “Incremental fusion: Unifying compiled and vectorized query execution.” ICDE, 2024. 

- [20] T. Kersten, V. Leis, A. Kemper, T. Neumann, A. Pavlo, and P. Boncz, “Everything you always wanted to know about compiled and vectorized queries but were afraid to ask,” _PVLDB_ , vol. 11, no. 13, pp. 2209–2222, 2018. 

- [21] J. Sompolski, M. Zukowski, and P. Boncz, “Vectorization vs. compilation in query execution,” in _Proceedings of the Seventh International Workshop on Data Management on New Hardware_ , 2011, pp. 33–40. 

- [22] A. Ailamaki, D. J. DeWitt, M. D. Hill, and D. A. Wood, “Dbmss on a modern processor: Where does time go?” in _PVLDB_ , 1999, pp. 266–277. 

- [23] B. Ding and A. C. König, “Fast set intersection in memory,” _PVLDB_ , vol. 4, no. 4, 2011. 

- [24] H. Inoue, M. Ohara, and K. Taura, “Faster set intersection with simd instructions by reducing branch mispredictions,” _PVLDB_ , vol. 8, no. 3, pp. 293–304, 2014. 

- [25] I. H. Witten, A. Moffat, and T. C. Bell, _Managing gigabytes: compressing and indexing documents and images_ . Morgan Kaufmann, 1999. 

- [26] J. Zobel and A. Moffat, “Inverted files for text search engines,” _ACM computing surveys (CSUR)_ , vol. 38, no. 2, pp. 6–es, 2006. 

- [27] H. Hashemi, M. Aliannejadi, H. Zamani, and W. B. Croft, “Antique: A non-factoid question answering benchmark,” in _ECIR_ , 2020, pp. 166–173. 

- [28] H. Wachsmuth, M. Potthast, K. Al Khatib, Y. Ajjour, J. Puschmann, J. Qu, J. Dorsch, V. Morari, J. Bevendorff, and B. Stein, “Building an argument search engine for the web,” in _Proceedings of the 4th Workshop on Argument Mining_ , 2017, pp. 49–59. 

- [29] Y. Ajjour, H. Wachsmuth, J. Kiesel, M. Potthast, M. Hagen, and B. Stein, “Data Acquisition for Argument Search: The args.me corpus,” in _KI_ . Berlin Heidelberg New York: Springer, 2019, pp. 48–59. 

- [30] J. Zhu, S. He, P. He, J. Liu, and M. R. Lyu, “Loghub: A large collection of system log datasets for ai-driven log analytics,” in _ISSRE_ , 2023. 

- [31] P. E. O’Neil, E. J. O’Neil, and X. Chen, “The star schema benchmark (ssb),” _Pat_ , vol. 200, no. 0, p. 50, 2007. 

- [32] Wikipedia, “Wikipedia html data dumps.” 2024. [Online]. Available: https://dumps.wikimedia.org/enwiki/ 

- [33] D. Dato, S. MacAvaney, F. M. Nardini, R. Perego, and N. Tonellotto, “The istella22 dataset: Bridging traditional and neural learning to rank evaluation,” in _SIGIR_ , 2022, pp. 3099–3107. 

- [34] A. Yasin, “A top-down method for performance analysis and counters architecture,” in _ISPASS_ . IEEE, 2014. 

- [35] A. J. Awan, M. Brorsson, V. Vlassov, and E. Ayguade, “Performance characterization of in-memory data analytics on a modern cloud server,” in _BDCloud_ . IEEE, 2015. 

1812 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 

- [36] S. Zhang, B. He, D. Dahlmeier, A. C. Zhou, and T. Heinze, “Revisiting the design of data stream processing systems on multi-core processors,” in _ICDE_ . IEEE, 2017, pp. 659–670. 

- [37] M. McCandless. Visualizing lucene’s segment merges. [Online]. Available: https://blog.mikemccandless.com/20 11/02/visualizing-lucenes-segment-merges.html 

- [38] X. Zeng, R. Meng, A. Pavlo, W. McKinney, and H. Zhang, “Nulls!: Revisiting null representation in modern columnar formats,” in _Proceedings of the 20th International Workshop on Data Management on New Hardware_ , 2024, pp. 1–10. 

- [39] Apache Arrow, “A cross-language development platform for in-memory analytics,” 2024. [Online]. Available: https://arrow.apache.org/ 

- [40] Apache Parquet, “Apache Parquet,” 2024. [Online]. Available: https://parquet.apache.org 

- [41] Apache ORC, “the smallest, fastest columnar storage for hadoop workloads.” 2024. [Online]. Available: https://orc.apache.org 

- [42] T. A. Bjørklund, N. Grimsmo, J. Gehrke, and Ø. Torbjørnsen, “Inverted indexes vs. bitmap indexes in decision support systems,” in _CIKM_ , 2009, pp. 1509–1512. 

   - [54] H. Yan, S. Ding, and T. Suel, “Inverted index compression and query processing with optimized document ordering,” in _WWW_ , 2009, pp. 401–410. 

   - [55] A. Mallia, M. Siedlaczek, and T. Suel, “An experimental study of index compression and daat query processing methods,” in _ECIR_ . Springer, 2019, pp. 353–368. 

   - [56] J. Mackenzie, M. Petri, and A. Moffat, “Faster index reordering with bipartite graph partitioning,” in _SIGIR_ , 2021, pp. 1910–1914. 

   - [57] A. Ailamaki, D. J. DeWitt, M. D. Hill, and M. Skounakis, “Weaving relations for cache performance.” in _VLDB_ , vol. 1, 2001, pp. 169–180. 

   - [58] S. Melnik, A. Gubarev, J. J. Long, G. Romer, S. Shivakumar, M. Tolton, and T. Vassilakis, “Dremel: interactive analysis of web-scale datasets,” _PVLDB_ , vol. 3, no. 1-2, pp. 330–339, 2010. 

   - [59] W. Y. Alkowaileet and M. J. Carey, “Columnar formats for schemaless lsm-based document stores,” _PVLDB_ , vol. 15, no. 10, pp. 2085–2097, 2022. 

   - [60] P. Gupta, A. Mhedhbi, and S. Salihoglu, “Columnar storage and list-based processing for graph database management systems,” _PVLDB_ , vol. 14, no. 11, pp. 2491– 2504, 2021. 

- [43] H. Lang, T. Mühlbauer, F. Funke, P. A. Boncz, T. Neumann, and A. Kemper, “Data blocks: Hybrid oltp and olap on compressed storage using both vectorization and compilation,” in _SIGMOD_ , 2016, pp. 311–326. 

- [44] G. Jacobson, “Space-efficient static trees and graphs,” in _FOCS_ . IEEE Computer Society, 1989, pp. 549–554. 

- [45] C. Lattner and V. Adve, “Llvm: A compilation framework for lifelong program analysis & transformation,” in _CGO_ . IEEE, 2004, pp. 75–86. 

- [46] J. M. Hellerstein and M. Stonebraker, “Predicate migration: Optimizing queries with expensive predicates,” in _SIGMOD_ , 1993, pp. 267–276. 

- [47] “Apache DataFusion SQL Query Engine,” 2024. [Online]. Available: https://github.com/apache/arrow-da tafusion 

- [48] P. Pedreira, O. Erling, M. Basmanova, K. Wilfong, L. Sakka, K. Pai, W. He, and B. Chattopadhyay, “Velox: meta’s unified execution engine,” _PVLDB_ , vol. 15, no. 12, pp. 3372–3384, 2022. 

- [49] M. Raasveldt and H. Mühleisen, “Duckdb: an embeddable analytical database,” in _ICDE_ , 2019. 

- [50] P. A. Boncz, M. Zukowski, and N. Nes, “Monetdb/x100: Hyper-pipelining query execution.” in _CIDR_ , vol. 5, 2005, pp. 225–237. 

- [51] A. Z. Broder, D. Carmel, M. Herscovici, A. Soffer, and J. Zien, “Efficient query evaluation using a two-level retrieval process,” in _CIKM_ , 2003, pp. 426–434. 

- [52] C. Dimopoulos, S. Nepomnyachiy, and T. Suel, “Optimizing top-k document retrieval strategies for block-max indexes,” in _WSDM_ , 2013, pp. 113–122. 

- [53] J. Mackenzie and A. Moffat, “Examining the additivity of top-k query processing innovations,” in _CIKM_ , 2020, pp. 1085–1094. 

1813 

Authorized licensed use limited to: SOUTHWEST JIAOTONG UNIVERSITY. Downloaded on September 19,2026 at 08:39:05 UTC from IEEE Xplore.  Restrictions apply. 

