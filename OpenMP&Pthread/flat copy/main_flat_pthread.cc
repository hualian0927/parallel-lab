// ===========================================================================
// Lab3 — Flat-SIMD Pthread: 四种并行模式 + p 值 trade-off 实验
//
// p = 每线程局部堆大小 (p >= k). 关键实验:
//   - 精确距离 (Flat): p=k 延迟最低, recall 不变
//   - 近似距离 (PQ/ADC): p 需 > k, p 太小会丢召回
//   本实验展示 p 对 latency 的影响曲线; recall 在精确场景不变
//
// 编译: g++ main_flat_pthread.cc -o main_flat_pthread -O2 -lpthread -std=c++17 -I..
// ===========================================================================
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sys/time.h>
#include <vector>
#include "flat_pthread.h"

template<typename T>
T* LoadData(std::string p, size_t& n, size_t& d) {
    std::ifstream f(p, std::ios::binary);
    if(!f){ std::cerr<<"open failed: "<<p<<"\n"; exit(1); }
    f.read((char*)&n,4); f.read((char*)&d,4);
    T* data=new T[n*d];
    for(size_t i=0;i<n;++i) f.read(((char*)data+i*d*sizeof(T)), d*sizeof(T));
    return data;
}

int main() {
    size_t tn=0,bn=0,gd=0,dim=0;
    std::string dp="/anndata/";
    float* q=LoadData<float>(dp+"DEEP100K.query.fbin",tn,dim);
    int*   gt=LoadData<int>(dp+"DEEP100K.gt.query.100k.top100.bin",tn,gd);
    float* ba=LoadData<float>(dp+"DEEP100K.base.100k.fbin",bn,dim);
    tn=2000; const size_t k=10;

    // p 值序列: k, 2k, 5k, 10k, 20k
    std::vector<size_t> p_values = {k, 2*k, 5*k, 10*k, 20*k};

    // ================================================================
    // IntraStatic: 底库静态均分 + p 值扫描
    // ================================================================
    std::cout<<"\n=================================================================\n";
    std::cout<<"IntraStatic (static base chunking) — p vs latency trade-off\n";
    std::cout<<"  p = local heap size per thread, k="<<k<<" threads=4\n";
    std::cout<<"=================================================================\n";
    std::cout<<"| p      | latency(us) | recall    |\n";
    for (size_t p : p_values) {
        int64_t tl=0; float tr=0;
        for(size_t i=0;i<tn;++i){
            struct timeval tv; gettimeofday(&tv, nullptr);
            auto res=flat_mt::search_intra_static(ba,q+i*dim,bn,dim,k,p,/*nth=*/4);
            struct timeval tv2; gettimeofday(&tv2, nullptr);
            tl+=(tv2.tv_sec*1000000UL+tv2.tv_usec)-(tv.tv_sec*1000000UL+tv.tv_usec);
            std::set<uint32_t> gs;
            for(size_t j=0;j<k;++j) gs.insert((uint32_t)gt[j+i*gd]);
            size_t hits=0;
            while(!res.empty()){if(gs.count((uint32_t)res.top().second))++hits;res.pop();}
            tr+=(float)hits/k;
        }
        std::cout<<"| p="<<std::setw(5)<<std::left<<p<<" | "<<std::setw(12)<<tl/tn<<" | "<<tr/tn<<" |\n";
    }

    // ================================================================
    // IntraStatic: 线程数扫描 (固定 p=k)
    // ================================================================
    std::cout<<"\n=================================================================\n";
    std::cout<<"IntraStatic — thread scaling (p=k="<<k<<")\n";
    std::cout<<"=================================================================\n";
    for(int nth:{1,2,4,8}){
        int64_t tl=0; float tr=0;
        for(size_t i=0;i<tn;++i){
            struct timeval tv; gettimeofday(&tv, nullptr);
            auto res=flat_mt::search_intra_static(ba,q+i*dim,bn,dim,k,k,nth);
            struct timeval tv2; gettimeofday(&tv2, nullptr);
            tl+=(tv2.tv_sec*1000000UL+tv2.tv_usec)-(tv.tv_sec*1000000UL+tv.tv_usec);
            std::set<uint32_t> gs;
            for(size_t j=0;j<k;++j) gs.insert((uint32_t)gt[j+i*gd]);
            size_t hits=0;
            while(!res.empty()){if(gs.count((uint32_t)res.top().second))++hits;res.pop();}
            tr+=(float)hits/k;
        }
        std::cout<<"| nth="<<nth<<" | latency="<<tl/tn<<" us | recall="<<tr/tn<<" |\n";
    }

    // ================================================================
    // IntraDynamic: 底库动态抢占 + 线程扫描
    // ================================================================
    std::cout<<"\n=================================================================\n";
    std::cout<<"IntraDynamic (atomic fetch_add on base) — thread scaling (p=k="<<k<<")\n";
    std::cout<<"=================================================================\n";
    for(int nth:{1,2,4,8}){
        int64_t tl=0; float tr=0;
        for(size_t i=0;i<tn;++i){
            struct timeval tv; gettimeofday(&tv, nullptr);
            auto res=flat_mt::search_intra_dynamic(ba,q+i*dim,bn,dim,k,k,nth);
            struct timeval tv2; gettimeofday(&tv2, nullptr);
            tl+=(tv2.tv_sec*1000000UL+tv2.tv_usec)-(tv.tv_sec*1000000UL+tv.tv_usec);
            std::set<uint32_t> gs;
            for(size_t j=0;j<k;++j) gs.insert((uint32_t)gt[j+i*gd]);
            size_t hits=0;
            while(!res.empty()){if(gs.count((uint32_t)res.top().second))++hits;res.pop();}
            tr+=(float)hits/k;
        }
        std::cout<<"| nth="<<nth<<" | latency="<<tl/tn<<" us | recall="<<tr/tn<<" |\n";
    }

    // ================================================================
    // InterDynamic: query 批量分发
    // ================================================================
    std::cout<<"\n=================================================================\n";
    std::cout<<"InterDynamic (atomic fetch_add on queries)\n";
    std::cout<<"=================================================================\n";
    for(int nth:{1,2,4,8}){
        std::vector<flat_mt::FlatHeap> results;
        auto t0=std::chrono::high_resolution_clock::now();
        flat_mt::search_inter_dynamic(ba,q,bn,tn,dim,k,nth,results);
        auto t1=std::chrono::high_resolution_clock::now();
        double wms=std::chrono::duration<double,std::milli>(t1-t0).count();
        float tr=0;
        for(size_t i=0;i<tn;++i){
            std::set<uint32_t> gs;
            for(size_t j=0;j<k;++j) gs.insert((uint32_t)gt[j+i*gd]);
            size_t hits=0;
            while(!results[i].empty()){if(gs.count((uint32_t)results[i].top().second))++hits;results[i].pop();}
            tr+=(float)hits/k;
        }
        std::cout<<"| nth="<<nth<<" | wall="<<wms<<" ms | avg_lat="<<wms*1000/tn<<" us | recall="<<tr/tn<<" |\n";
    }

    delete[] ba; delete[] q; delete[] gt;
}
/*
============================================================================
报告用分析模板:

1. p 值 trade-off 实验 (IntraStatic, 4 threads):
   | p   | latency | recall |
   | k   |   X     | 0.9999 |
   | 2k  |  X+Δ   | 0.9999 |
   | 5k  |  X+3Δ  | 0.9999 |
   | 10k |  X+6Δ  | 0.9999 |
   | 20k |  X+12Δ | 0.9999 |

   分析: 精确距离场景 recall 不受 p 影响, 但 latency 随 p 增大近似线性增长
   (堆 push/pop O(log p) 开销)。在近似距离场景(PQ/ADC)中, p 过小会导致
   真近邻被 ADC 误差挤出局部堆, 召回下降。因此 p 是 recall-latency 调节旋钮。

2. 线程扩展性:
   IntraStatic: 1→2→4→8 加速比 | 结论: 4 线程近 DRAM 带宽饱和
   IntraDynamic vs IntraStatic: 原子竞争 vs 静态切分 | 哪个更好?

3. 模式对比 (4 threads, p=k):
   IntraStatic  — latency=X, 每 query 创建/销毁 pthread
   IntraDynamic — latency=Y, 原子竞争, 线程数多时 CAS 重试开销
   InterDynamic — wall=Z ms, 无 per-query 线程开销, 吞吐最优
============================================================================
*/

/*
[1] 14:12:03 [SUCCESS] master_ubss7

=================================================================
IntraStatic (static base chunking) — p vs latency trade-off
  p = local heap size per thread, k=10 threads=4
=================================================================
| p      | latency(us) | recall    |
| p=10    | 2295         | 0.99995 |
| p=20    | 2410         | 0.99995 |
| p=50    | 2370         | 0.99995 |
| p=100   | 2510         | 0.99995 |
| p=200   | 2412         | 0.99995 |

=================================================================
IntraStatic — thread scaling (p=k=10)
=================================================================
| nth=1 | latency=8440 us | recall=0.99995 |
| nth=2 | latency=4575 us | recall=0.99995 |
| nth=4 | latency=2348 us | recall=0.99995 |
| nth=8 | latency=2622 us | recall=0.99995 |

=================================================================
IntraDynamic (atomic fetch_add on base) — thread scaling (p=k=10)
=================================================================
| nth=1 | latency=6761 us | recall=0.99995 |
| nth=2 | latency=9233 us | recall=0.99995 |
| nth=4 | latency=16936 us | recall=0.99995 |
| nth=8 | latency=24761 us | recall=0.99995 |

=================================================================
InterDynamic (atomic fetch_add on queries)
=================================================================
| nth=1 | wall=13944.2 ms | avg_lat=6972.11 us | recall=0.99995 |
| nth=2 | wall=5641.65 ms | avg_lat=2820.83 us | recall=0.99995 |
| nth=4 | wall=3431.67 ms | avg_lat=1715.83 us | recall=0.99995 |
| nth=8 | wall=2011.78 ms | avg_lat=1005.89 us | recall=0.99995 |

Authorized users only. All activities may be monitored and reported.

*/