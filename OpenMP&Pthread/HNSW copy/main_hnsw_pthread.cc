// ===========================================================================
// Lab3 — HNSW inter-query batch Pthread (基于 hnswlib 真实框架)
//
// 指导书 §2.3: 只关注底层图, intra-query 并行困难且通常负优化
// 策略: atomic fetch_add 分发 query, searchKnn const → 多线程并发读安全
//
// 编译: g++ main_hnsw_pthread.cc -o main_hnsw_pthread -O2 -lpthread -std=c++17 -I..
// ===========================================================================
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "hnsw_pthread.h"
using namespace hnswlib;

template<typename T>
T* LoadData(std::string p, size_t& n, size_t& d) {
    std::ifstream f(p, std::ios::binary);
    if(!f){ std::cerr<<"open failed: "<<p<<"\n"; exit(1); }
    f.read((char*)&n,4); f.read((char*)&d,4);
    T* data=new T[n*d];
    for(size_t i=0;i<n;++i) f.read(((char*)data+i*d*sizeof(T)), d*sizeof(T));
    return data;
}
static bool fexists(const std::string& p){ std::ifstream t(p); return t.good(); }

int main() {
    size_t tn=0,bn=0,gd=0,dim=0;
    std::string dp="/anndata/";
    float* q=LoadData<float>(dp+"DEEP100K.query.fbin",tn,dim);
    int*   gt=LoadData<int>(dp+"DEEP100K.gt.query.100k.top100.bin",tn,gd);
    float* ba=LoadData<float>(dp+"DEEP100K.base.100k.fbin",bn,dim);
    tn=2000; const size_t k=10; const int M=16, efc=150;

    // ---- 索引: 缓存优先 ----
    const std::string ipath="files/hnsw.index";
    InnerProductSpace ipspace(dim);
    HierarchicalNSW<float>* idx=nullptr;
    if(fexists(ipath)){
        idx=new HierarchicalNSW<float>(&ipspace, ipath);
        std::cerr<<"[HNSW] Loaded index from "<<ipath<<"\n";
    }else{
        idx=new HierarchicalNSW<float>(&ipspace, bn, M, efc);
        idx->addPoint(ba,0);
        #pragma omp parallel for
        for(size_t i=1;i<bn;++i) idx->addPoint(ba+i*dim,i);
        idx->saveIndex(ipath);
        std::cerr<<"[HNSW] Index saved to "<<ipath<<"\n";
    }
    const size_t ef_search = 64;  // 候选队列大小, 越大recall越高越慢
    idx->ef_ = ef_search;

    std::cout<<"\n==================================================\n";
    std::cout<<"HNSW inter-query batch [Pthread] (ef="<<ef_search<<")\n";
    std::cout<<"==================================================\n";
    for(int nth:{1,2,4,8}){
        std::vector<float> lats,recs;
        auto t0=std::chrono::high_resolution_clock::now();
        hnsw_batch_pthread(idx,q,tn,dim,k,nth,lats,recs,gt,gd);
        auto t1=std::chrono::high_resolution_clock::now();
        double wms=std::chrono::duration<double,std::milli>(t1-t0).count();
        float sl=0,sr=0;
        for(size_t i=0;i<tn;++i){sl+=lats[i];sr+=recs[i];}
        std::cout<<"| pthreads="<<nth<<" | wall="<<wms<<" ms | avg_lat="<<sl/tn<<" us | recall="<<sr/tn<<" |\n";
    }
    delete idx; delete[] ba; delete[] q; delete[] gt;
}
/*
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21139.master_ubss1
[1] 16:31:47 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 16:31:49 [SUCCESS] master_ubss8
[HNSW] Loaded index from files/hnsw.index

==================================================
HNSW inter-query batch [Pthread] (ef=64)
==================================================
| pthreads=1 | wall=740.856 ms | avg_lat=368.621 us | recall=0.978702 |
| pthreads=2 | wall=367.671 ms | avg_lat=365.699 us | recall=0.978702 |
| pthreads=4 | wall=203.363 ms | avg_lat=403.349 us | recall=0.978702 |
| pthreads=8 | wall=115.474 ms | avg_lat=440.94 us | recall=0.978702 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21142.master_ubss1
[1] 16:32:27 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 16:32:28 [SUCCESS] master_ubss8
[HNSW] Index saved to files/hnsw.index

==================================================
HNSW inter-query batch [Pthread] (ef=48)
==================================================
| pthreads=1 | wall=608.903 ms | avg_lat=302.737 us | recall=0.968653 |
| pthreads=2 | wall=294.211 ms | avg_lat=292.568 us | recall=0.968653 |
| pthreads=4 | wall=142.535 ms | avg_lat=282.832 us | recall=0.968653 |
| pthreads=8 | wall=96.905 ms | avg_lat=375.706 us | recall=0.968653 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21144.master_ubss1
[1] 16:32:58 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 16:32:59 [SUCCESS] master_ubss8
[HNSW] Loaded index from files/hnsw.index

==================================================
HNSW inter-query batch [Pthread] (ef=48)
==================================================
| pthreads=1 | wall=579.154 ms | avg_lat=287.916 us | recall=0.968653 |
| pthreads=2 | wall=366.376 ms | avg_lat=364.047 us | recall=0.968653 |
| pthreads=4 | wall=158.79 ms | avg_lat=315.23 us | recall=0.968653 |
| pthreads=8 | wall=108.694 ms | avg_lat=411.549 us | recall=0.968653 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21146.master_ubss1
[1] 16:33:27 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 16:33:28 [SUCCESS] master_ubss8
[HNSW] Index saved to files/hnsw.index

==================================================
HNSW inter-query batch [Pthread] (ef=32)
==================================================
| pthreads=1 | wall=417.833 ms | avg_lat=207.359 us | recall=0.943454 |
| pthreads=2 | wall=179.314 ms | avg_lat=177.656 us | recall=0.943454 |
| pthreads=4 | wall=99.6543 ms | avg_lat=197.168 us | recall=0.943454 |
| pthreads=8 | wall=58.6466 ms | avg_lat=230.281 us | recall=0.943454 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21149.master_ubss1
cat: /home/s2412351/ann/test.e: No such file or directory


==================================================
HNSW inter-query batch [Pthread] (ef=32)
==================================================
| pthreads=1 | wall=391.054 ms | avg_lat=193.986 us | recall=0.943454 |
| pthreads=2 | wall=274.587 ms | avg_lat=272.245 us | recall=0.943454 |
| pthreads=4 | wall=132.872 ms | avg_lat=263.036 us | recall=0.943454 |
| pthreads=8 | wall=88.2022 ms | avg_lat=338.605 us | recall=0.943454 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21153.master_ubss1
[1] 16:34:44 [SUCCESS] master_ubss9

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 16:34:45 [SUCCESS] master_ubss9
[HNSW] Loaded index from files/hnsw.index

==================================================
HNSW inter-query batch [Pthread] (ef=32)
==================================================
| pthreads=1 | wall=435.917 ms | avg_lat=216.372 us | recall=0.943454 |
| pthreads=2 | wall=297.508 ms | avg_lat=295.069 us | recall=0.943454 |
| pthreads=4 | wall=99.8501 ms | avg_lat=197.201 us | recall=0.943454 |
| pthreads=8 | wall=58.4615 ms | avg_lat=230.411 us | recall=0.943454 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21154.master_ubss1
[1] 16:35:04 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 16:35:06 [SUCCESS] master_ubss8
[HNSW] Loaded index from files/hnsw.index

==================================================
HNSW inter-query batch [Pthread] (ef=32)
==================================================
| pthreads=1 | wall=405.154 ms | avg_lat=201.057 us | recall=0.943454 |
| pthreads=2 | wall=275.143 ms | avg_lat=272.991 us | recall=0.943454 |
| pthreads=4 | wall=129.428 ms | avg_lat=255.828 us | recall=0.943454 |
| pthreads=8 | wall=110.961 ms | avg_lat=410.454 us | recall=0.943454 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21166.master_ubss1
[1] 16:41:58 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 16:41:59 [SUCCESS] master_ubss8
[HNSW] Index saved to files/hnsw.index

==================================================
HNSW inter-query batch [Pthread] (ef=40)
==================================================
| pthreads=1 | wall=524.293 ms | avg_lat=260.306 us | recall=0.959203 |
| pthreads=2 | wall=331.73 ms | avg_lat=329.654 us | recall=0.959203 |
| pthreads=4 | wall=133.393 ms | avg_lat=264.632 us | recall=0.959203 |
| pthreads=8 | wall=89.4374 ms | avg_lat=334.664 us | recall=0.959203 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21169.master_ubss1
[1] 16:43:16 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 16:43:18 [SUCCESS] master_ubss8
[HNSW] Index saved to files/hnsw.index

==================================================
HNSW inter-query batch [Pthread] (ef=36)
==================================================
| pthreads=1 | wall=466.321 ms | avg_lat=231.587 us | recall=0.952604 |
| pthreads=2 | wall=218.645 ms | avg_lat=216.626 us | recall=0.952604 |
| pthreads=4 | wall=137.456 ms | avg_lat=272.355 us | recall=0.952604 |
| pthreads=8 | wall=77.1436 ms | avg_lat=295.799 us | recall=0.952604 |
*/