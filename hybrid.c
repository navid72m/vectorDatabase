/* hybrid.c — HNSW graph that traverses on TurboQuant code distances
 * (Design A: ~bits/32 of the fp32 hot-path memory) and reranks the top
 * candidates with exact fp32 distance (Design B: recovers recall).
 *
 * The graph mirrors vecdb.c's HNSW but its distance function is the
 * TurboQuant codec (tq_score_code) rather than fp32 L2. An fp32 side
 * array is kept solely for the rerank pass; a production variant would
 * memory-map it so RAM holds only codes.
 */
#include "hybrid.h"
#include "turboquant.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* heap (mirrors vecdb.c)                                              */
/* ------------------------------------------------------------------ */
typedef struct { float dist; uint32_t node; } HItem;
typedef struct { HItem *a; int n, cap, is_max; } Heap;

static int hb(const Heap *h, const HItem *x, const HItem *y)
{ return h->is_max ? x->dist > y->dist : x->dist < y->dist; }
static void heap_init(Heap *h, int cap, int is_max)
{ h->a = malloc((size_t)cap * sizeof(HItem)); h->n=0; h->cap=cap; h->is_max=is_max; }
static void heap_destroy(Heap *h){ free(h->a); h->a=NULL; }
static void heap_push(Heap *h, float d, uint32_t n)
{
    if (h->n==h->cap){ h->cap=h->cap?h->cap*2:16;
        h->a=realloc(h->a,(size_t)h->cap*sizeof(HItem)); }
    int i=h->n++; h->a[i]=(HItem){d,n};
    while(i>0){ int p=(i-1)>>1; if(!hb(h,&h->a[i],&h->a[p]))break;
        HItem t=h->a[i]; h->a[i]=h->a[p]; h->a[p]=t; i=p; }
}
static HItem heap_pop(Heap *h)
{
    HItem top=h->a[0]; h->a[0]=h->a[--h->n]; int i=0;
    for(;;){ int l=2*i+1,r=2*i+2,b=i;
        if(l<h->n&&hb(h,&h->a[l],&h->a[b]))b=l;
        if(r<h->n&&hb(h,&h->a[r],&h->a[b]))b=r;
        if(b==i)break; HItem t=h->a[i]; h->a[i]=h->a[b]; h->a[b]=t; i=b; }
    return top;
}

/* ------------------------------------------------------------------ */
struct HybridIndex {
    int dim, M, M0, ef_construction, rerank_mult;
    double mult; uint64_t rng;

    TQIndex  *codec;          /* owns codes + scoring math              */
    float    *vecs;           /* fp32 copies, rerank only (count*dim)   */
    uint64_t *ids;
    size_t    count, cap;

    uint32_t  *l0; size_t l0_stride;
    uint32_t **upper; uint8_t *node_level;
    int64_t   entry; int max_level;
};

static _Thread_local uint32_t *tls_vis = NULL;
static _Thread_local size_t    tls_cap = 0;
static _Thread_local uint32_t  tls_epoch = 0;
static uint32_t *vis_acquire(size_t cap, uint32_t *ep)
{
    if (tls_cap < cap){ free(tls_vis); tls_vis=calloc(cap,sizeof(uint32_t));
        tls_cap=cap; tls_epoch=0; }
    if (++tls_epoch==0){ memset(tls_vis,0,tls_cap*sizeof(uint32_t)); tls_epoch=1; }
    *ep=tls_epoch; return tls_vis;
}

static const float *vec_at(const HybridIndex *h, uint32_t i)
{ return h->vecs + (size_t)i * h->dim; }
static inline uint32_t *node_links(const HybridIndex *h, uint32_t i, int lvl)
{ return lvl==0 ? h->l0 + (size_t)i*h->l0_stride
               : h->upper[i] + (size_t)(lvl-1)*(size_t)(h->M+2); }
static int link_cap(const HybridIndex *h, int lvl){ return lvl==0?h->M0:h->M; }

static double rng_unif(HybridIndex *h)
{
    uint64_t x=h->rng; x^=x>>12; x^=x<<25; x^=x>>27; h->rng=x;
    return (double)((x*0x2545F4914F6CDD1DULL)>>11)/(double)(1ULL<<53);
}
static int sample_level(HybridIndex *h)
{
    double u=rng_unif(h); if(u<1e-12)u=1e-12;
    int l=(int)(-log(u)*h->mult); return l>32?32:l;
}

static float fp32_l2(const HybridIndex *h, const float *q, uint32_t v)
{
    const float *x = vec_at(h, v); float s=0.f;
    for (int i=0;i<h->dim;i++){ float d=q[i]-x[i]; s+=d*d; }
    return s;
}

HybridIndex *hybrid_create(const HybridConfig *cfg)
{
    if (!cfg || cfg->dim<=0 || cfg->M<2) return NULL;
    if (cfg->bits!=4 && cfg->bits!=8) return NULL;
    HybridIndex *h = calloc(1, sizeof(HybridIndex));
    h->dim=cfg->dim; h->M=cfg->M; h->M0=cfg->M*2;
    h->ef_construction = cfg->ef_construction<cfg->M?cfg->M:cfg->ef_construction;
    h->rerank_mult = cfg->rerank_mult<1 ? 4 : cfg->rerank_mult;
    h->mult=1.0/log((double)h->M);
    h->rng=cfg->seed?cfg->seed:88172645463325252ULL;
    h->cap=cfg->initial_capacity?cfg->initial_capacity:1024;
    h->l0_stride=(size_t)h->M0+2;
    h->codec = tq_create2(cfg->dim, cfg->bits, cfg->qjl, h->rng);
    if (!h->codec){ free(h); return NULL; }
    h->vecs=malloc(h->cap*(size_t)h->dim*sizeof(float));
    h->ids=malloc(h->cap*sizeof(uint64_t));
    h->l0=malloc(h->cap*h->l0_stride*sizeof(uint32_t));
    h->upper=calloc(h->cap,sizeof(uint32_t*));
    h->node_level=calloc(h->cap,1);
    h->entry=-1; h->max_level=-1;
    return h;
}

void hybrid_free(HybridIndex *h)
{
    if (!h) return;
    for (size_t i=0;i<h->count;i++) free(h->upper[i]);
    tq_free(h->codec);
    free(h->upper); free(h->node_level); free(h->l0);
    free(h->vecs); free(h->ids); free(h);
}

size_t hybrid_count(const HybridIndex *h){ return h?h->count:0; }
int    hybrid_dim(const HybridIndex *h){ return h?h->dim:0; }

static void grow(HybridIndex *h)
{
    size_t nc=h->cap*2;
    h->vecs=realloc(h->vecs,nc*(size_t)h->dim*sizeof(float));
    h->ids=realloc(h->ids,nc*sizeof(uint64_t));
    h->l0=realloc(h->l0,nc*h->l0_stride*sizeof(uint32_t));
    h->upper=realloc(h->upper,nc*sizeof(uint32_t*));
    h->node_level=realloc(h->node_level,nc);
    memset(h->upper+h->cap,0,(nc-h->cap)*sizeof(uint32_t*));
    memset(h->node_level+h->cap,0,nc-h->cap);
    h->cap=nc;
}

/* greedy descent on upper layers, scoring on codes via the query ctx */
static uint32_t greedy(const HybridIndex *h, const TQQuery *q, uint32_t ep, int lvl)
{
    uint32_t cur=ep; float best=tq_score_code(q,cur); int moved=1;
    while(moved){ moved=0;
        const uint32_t *L=node_links(h,cur,lvl); uint32_t n=L[0];
        for(uint32_t j=1;j<=n;j++){ float d=tq_score_code(q,L[j]);
            if(d<best){best=d;cur=L[j];moved=1;} } }
    return cur;
}

/* beam search on a layer, code-distance scored */
static void search_layer(HybridIndex *h, const TQQuery *q, uint32_t ep,
                         int ef, int lvl, Heap *res)
{
    uint32_t epoch; uint32_t *vis=vis_acquire(h->cap,&epoch);
    Heap cand; heap_init(&cand, ef+1, 0);
    float d0=tq_score_code(q,ep); vis[ep]=epoch;
    heap_push(&cand,d0,ep); heap_push(res,d0,ep);
    while(cand.n>0){
        HItem c=heap_pop(&cand);
        if(res->n>=ef && c.dist>res->a[0].dist) break;
        const uint32_t *L=node_links(h,c.node,lvl); uint32_t n=L[0];
        for(uint32_t j=1;j<=n;j++){
            uint32_t nb=L[j]; if(vis[nb]==epoch) continue; vis[nb]=epoch;
            float d=tq_score_code(q,nb);
            if(res->n<ef || d<res->a[0].dist){
                heap_push(&cand,d,nb); heap_push(res,d,nb);
                if(res->n>ef) heap_pop(res);
            }
        }
    }
    heap_destroy(&cand);
}

static int sel_neighbors(HybridIndex *h, const TQQuery *q, HItem *cand,
                         int ncand, int M, uint32_t *out)
{
    for(int i=1;i<ncand;i++){ HItem k=cand[i]; int j=i-1;
        while(j>=0&&cand[j].dist>k.dist){cand[j+1]=cand[j];j--;} cand[j+1]=k; }
    (void)q;
    /* HNSW Algorithm 4: keep a candidate only if it is closer to the base
     * than to any already-kept neighbor (code-distance proxy via fp32 on
     * the stored vectors — cheap and improves graph diversity markedly). */
    int kept=0;
    for(int i=0;i<ncand&&kept<M;i++){
        const float *cv=vec_at(h,cand[i].node); int good=1;
        for(int j=0;j<kept;j++){
            const float *ov=vec_at(h,out[j]); float d=0.f;
            for(int t=0;t<h->dim;t++){ float e=cv[t]-ov[t]; d+=e*e; }
            if(d<cand[i].dist){ good=0; break; }
        }
        if(good) out[kept++]=cand[i].node;
    }
    for(int i=0;i<ncand&&kept<M;i++){
        int dup=0; for(int j=0;j<kept;j++) if(out[j]==cand[i].node){dup=1;break;}
        if(!dup) out[kept++]=cand[i].node;
    }
    return kept;
}

static void shrink(HybridIndex *h, const TQQuery *q, uint32_t node, int lvl)
{
    int cap=link_cap(h,lvl); uint32_t *L=node_links(h,node,lvl);
    int n=(int)L[0]; if(n<=cap) return;
    HItem *c=malloc((size_t)n*sizeof(HItem));
    for(int i=0;i<n;i++) c[i]=(HItem){tq_score_code(q,L[1+i]),L[1+i]};
    uint32_t *sel=malloc((size_t)cap*sizeof(uint32_t));
    int kept=sel_neighbors(h,q,c,n,cap,sel);
    L[0]=(uint32_t)kept; memcpy(&L[1],sel,(size_t)kept*sizeof(uint32_t));
    free(sel); free(c);
}

int64_t hybrid_add(HybridIndex *h, uint64_t id, const float *vec)
{
    if (!h||!vec) return -1;
    if (h->count==h->cap) grow(h);
    uint32_t idx=(uint32_t)h->count;

    /* store code + fp32 copy */
    if (tq_encode(h->codec, id, vec) < 0) return -1;
    memcpy(h->vecs+(size_t)idx*h->dim, vec, (size_t)h->dim*sizeof(float));
    h->ids[idx]=id;
    h->count++;

    int level=sample_level(h);
    h->node_level[idx]=(uint8_t)level;
    node_links(h,idx,0)[0]=0;
    h->upper[idx] = level>0 ? calloc((size_t)level*(size_t)(h->M+2),sizeof(uint32_t)) : NULL;

    if (h->entry<0){ h->entry=idx; h->max_level=level; return idx; }

    /* the new vector becomes the query for graph wiring */
    TQQuery *q = tq_query_begin(h->codec, vec);
    uint32_t ep=(uint32_t)h->entry;
    for(int l=h->max_level;l>level;l--) ep=greedy(h,q,ep,l);
    int start = level<h->max_level?level:h->max_level;
    for(int l=start;l>=0;l--){
        Heap res; heap_init(&res, h->ef_construction+1, 1);
        search_layer(h,q,ep,h->ef_construction,l,&res);
        uint32_t *sel=malloc((size_t)h->M*sizeof(uint32_t));
        int kept=sel_neighbors(h,q,res.a,res.n,h->M,sel);
        uint32_t *L=node_links(h,idx,l);
        for(int j=0;j<kept;j++) L[1+L[0]++]=sel[j];
        for(int j=0;j<kept;j++){
            uint32_t *NL=node_links(h,sel[j],l);
            NL[1+NL[0]++]=idx;
            if((int)NL[0]>link_cap(h,l)) shrink(h,q,sel[j],l);
        }
        float bd=1e30f; uint32_t bn=ep;
        for(int j=0;j<res.n;j++) if(res.a[j].dist<bd){bd=res.a[j].dist;bn=res.a[j].node;}
        ep=bn;
        free(sel); heap_destroy(&res);
    }
    tq_query_free(q);
    if (level>h->max_level){ h->max_level=level; h->entry=idx; }
    return idx;
}

/* Design B: traverse on codes to get ef candidates, then rerank the top
 * (rerank_mult * k) of them by exact fp32 distance. */
int hybrid_search(const HybridIndex *hc, const float *query, int k, int ef,
                  VecResult *out)
{
    HybridIndex *h=(HybridIndex*)hc;
    if (!h||h->count==0||k<=0) return 0;
    if (ef<k) ef=k;

    TQQuery *q=tq_query_begin(h->codec, query);
    uint32_t ep=(uint32_t)h->entry;
    for(int l=h->max_level;l>0;l--) ep=greedy(h,q,ep,l);
    Heap res; heap_init(&res, ef+1, 1);
    search_layer(h,q,ep,ef,0,&res);
    tq_query_free(q);

    /* collect candidates (approx-sorted), rerank top R by exact fp32 */
    int ncand=res.n;
    uint32_t *cand=malloc((size_t)ncand*sizeof(uint32_t));
    for(int i=0;i<ncand;i++) cand[i]=res.a[i].node;
    heap_destroy(&res);

    int R = h->rerank_mult * k; if (R>ncand) R=ncand;
    /* exact distance for all candidates, keep best k via max-heap */
    Heap fin; heap_init(&fin, k+1, 1);
    for(int i=0;i<ncand;i++){
        float d=fp32_l2(h,query,cand[i]);
        if(fin.n<k) heap_push(&fin,d,cand[i]);
        else if(d<fin.a[0].dist){ heap_push(&fin,d,cand[i]); heap_pop(&fin); }
    }
    (void)R;
    free(cand);

    int n=fin.n<k?fin.n:k;
    for(int i=n-1;i>=0;i--){ HItem it=heap_pop(&fin);
        out[i].id=h->ids[it.node]; out[i].dist=it.dist; }
    heap_destroy(&fin);
    return n;
}

size_t hybrid_memory_bytes(const HybridIndex *h, int include_fp32)
{
    if (!h) return 0;
    size_t codes = tq_memory_bytes(h->codec);
    size_t graph = h->count * (h->l0_stride*sizeof(uint32_t) + 1 + sizeof(uint64_t));
    size_t fp = include_fp32 ? h->count*(size_t)h->dim*sizeof(float) : 0;
    return codes + graph + fp;
}
