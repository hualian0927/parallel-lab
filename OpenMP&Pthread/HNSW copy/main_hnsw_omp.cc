// ===========================================================================
// Lab3 — HNSW inter-query batch OpenMP (基于 hnswlib 真实框架)
//
// 指导书 §2.3: 只关注底层图, 不实现 intra-query 并行(指导书明确说困难)
// 策略: #pragma omp parallel for schedule(dynamic) 分发 query
//
// 编译: g++ main_hnsw_omp.cc -o main_hnsw_omp -O2 -fopenmp -lpthread -std=c++17 -I..
// ===========================================================================
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "hnsw_omp.h"
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
    const size_t ef_search = 64;
    idx->ef_ = ef_search;

    std::cout<<"\n==================================================\n";
    std::cout<<"HNSW inter-query batch [OpenMP] (ef="<<ef_search<<")\n";
    std::cout<<"==================================================\n";
    for(int nth:{1,2,4,8}){
        std::vector<float> lats,recs;
        auto t0=std::chrono::high_resolution_clock::now();
        hnsw_batch_omp(idx,q,tn,dim,k,nth,lats,recs,gt,gd);
        auto t1=std::chrono::high_resolution_clock::now();
        double wms=std::chrono::duration<double,std::milli>(t1-t0).count();
        float sl=0,sr=0;
        for(size_t i=0;i<tn;++i){sl+=lats[i];sr+=recs[i];}
        std::cout<<"| omp="<<nth<<" | wall="<<wms<<" ms | avg_lat="<<sl/tn<<" us | recall="<<sr/tn<<" |\n";
    }
    delete idx; delete[] ba; delete[] q; delete[] gt;
}
/*
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21233.master_ubss1
[1] 17:07:01 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 17:07:02 [SUCCESS] master_ubss8
[HNSW] Index saved to files/hnsw.index

==================================================
HNSW inter-query batch [OpenMP] (ef=64)
==================================================
| omp=1 | wall=607.111 ms | avg_lat=301.65 us | recall=0.980152 |
| omp=2 | wall=309.099 ms | avg_lat=307.052 us | recall=0.980152 |
| omp=4 | wall=194.492 ms | avg_lat=385.884 us | recall=0.980152 |
| omp=8 | wall=103.204 ms | avg_lat=403.97 us | recall=0.980152 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21238.master_ubss1
[1] 17:08:11 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 17:08:13 [SUCCESS] master_ubss8
[HNSW] Loaded index from files/hnsw.index

==================================================
HNSW inter-query batch [OpenMP] (ef=64)
==================================================
| omp=1 | wall=628.179 ms | avg_lat=312.254 us | recall=0.980152 |
| omp=2 | wall=327.435 ms | avg_lat=325.208 us | recall=0.980152 |
| omp=4 | wall=165.41 ms | avg_lat=327.462 us | recall=0.980152 |
| omp=8 | wall=97.5312 ms | avg_lat=383.866 us | recall=0.980152 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21240.master_ubss1
[1] 17:08:45 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 17:08:47 [SUCCESS] master_ubss8
[HNSW] Index saved to files/hnsw.index

==================================================
HNSW inter-query batch [OpenMP] (ef=48)
==================================================
| omp=1 | wall=800.361 ms | avg_lat=398.129 us | recall=0.968003 |
| omp=2 | wall=365.871 ms | avg_lat=362.962 us | recall=0.968003 |
| omp=4 | wall=192.955 ms | avg_lat=382.612 us | recall=0.968003 |
| omp=8 | wall=100.725 ms | avg_lat=391.684 us | recall=0.968003 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21265.master_ubss1
[1] 17:16:51 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 17:16:53 [SUCCESS] master_ubss8
[HNSW] Loaded index from files/hnsw.index

==================================================
HNSW inter-query batch [OpenMP] (ef=48)
==================================================
| omp=1 | wall=728.22 ms | avg_lat=362.155 us | recall=0.968003 |
| omp=2 | wall=337.508 ms | avg_lat=334.708 us | recall=0.968003 |
| omp=4 | wall=158.126 ms | avg_lat=312.913 us | recall=0.968003 |
| omp=8 | wall=84.1287 ms | avg_lat=330.345 us | recall=0.968003 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21272.master_ubss1
[1] 17:17:56 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 17:17:57 [SUCCESS] master_ubss8
[HNSW] Index saved to files/hnsw.index

==================================================
HNSW inter-query batch [OpenMP] (ef=40)
==================================================
| omp=1 | wall=659.144 ms | avg_lat=327.642 us | recall=0.958353 |
| omp=2 | wall=299.379 ms | avg_lat=296.926 us | recall=0.958353 |
| omp=4 | wall=138.521 ms | avg_lat=274.357 us | recall=0.958353 |
| omp=8 | wall=76.7522 ms | avg_lat=302.064 us | recall=0.958353 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21274.master_ubss1
[1] 17:18:28 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 17:18:30 [SUCCESS] master_ubss8
[HNSW] Loaded index from files/hnsw.index

==================================================
HNSW inter-query batch [OpenMP] (ef=40)
==================================================
| omp=1 | wall=409.56 ms | avg_lat=203.184 us | recall=0.958353 |
| omp=2 | wall=295.106 ms | avg_lat=292.526 us | recall=0.958353 |
| omp=4 | wall=153.104 ms | avg_lat=303.146 us | recall=0.958353 |
| omp=8 | wall=79.7047 ms | avg_lat=314.025 us | recall=0.958353 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21278.master_ubss1
[1] 17:18:54 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 17:18:56 [SUCCESS] master_ubss8
[HNSW] Loaded index from files/hnsw.index

==================================================
HNSW inter-query batch [OpenMP] (ef=36)
==================================================
| omp=1 | wall=689.607 ms | avg_lat=343.029 us | recall=0.950504 |
| omp=2 | wall=306.647 ms | avg_lat=304.035 us | recall=0.950504 |
| omp=4 | wall=150.766 ms | avg_lat=297.644 us | recall=0.950504 |
| omp=8 | wall=98.7081 ms | avg_lat=369.813 us | recall=0.950504 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
g++: error: main.cc: No such file or directory
Submitted job with ID: 21280.master_ubss1
^Z
[1]+  Stopped                 bash test.sh 2 1
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21281.master_ubss1
^C[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21283.master_ubss1
[1] 17:20:21 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 17:20:22 [SUCCESS] master_ubss8
[HNSW] Loaded index from files/hnsw.index

==================================================
HNSW inter-query batch [OpenMP] (ef=36)
==================================================
| omp=1 | wall=461.653 ms | avg_lat=229.24 us | recall=0.950504 |
| omp=2 | wall=212.093 ms | avg_lat=210.014 us | recall=0.950504 |
| omp=4 | wall=128.012 ms | avg_lat=252.98 us | recall=0.950504 |
| omp=8 | wall=71.6566 ms | avg_lat=282.026 us | recall=0.950504 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21285.master_ubss1
[1] 17:20:44 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 17:20:46 [SUCCESS] master_ubss8
[HNSW] Loaded index from files/hnsw.index

==================================================
HNSW inter-query batch [OpenMP] (ef=32)
==================================================
| omp=1 | wall=377.718 ms | avg_lat=187.299 us | recall=0.942254 |
| omp=2 | wall=253.89 ms | avg_lat=251.477 us | recall=0.942254 |
| omp=4 | wall=120.016 ms | avg_lat=237.172 us | recall=0.942254 |
| omp=8 | wall=63.0193 ms | avg_lat=248.066 us | recall=0.942254 |

Authorized users only. All activities may be monitored and reported.
[s2412351@master_ubss1 ann]$ bash test.sh 2 1
Submitted job with ID: 21286.master_ubss1
[1] 17:20:56 [SUCCESS] master_ubss8

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
[1] 17:20:58 [SUCCESS] master_ubss8
[HNSW] Loaded index from files/hnsw.index

==================================================
HNSW inter-query batch [OpenMP] (ef=32)
==================================================
| omp=1 | wall=425.676 ms | avg_lat=211.3 us | recall=0.942254 |
| omp=2 | wall=291.278 ms | avg_lat=288.701 us | recall=0.942254 |
| omp=4 | wall=129.994 ms | avg_lat=256.952 us | recall=0.942254 |
| omp=8 | wall=73.4316 ms | avg_lat=289.056 us | recall=0.942254 |

Authorized users only. All activities may be monitored and reported.
*/