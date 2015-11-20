/*
 Petar 'PetarV' Velickovic
 Algorithm: Markov Clustering (MCL)
*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <algorithm>
#include <queue>
#include <stack>
#include <set>
#include <map>
#include <complex>

using namespace std;
typedef long long lld;

/*
 The Markov Clustering (MCL) algorithm outputs a clustering on a weighted graph,
 by operating on its adjacency matrix.
 
 The algorithm represents the graph as a Markov chain, and then alternates
 expansion (raising the matrix to a power, e) and inflation (raising each entry of
 the matrix to a power, r) steps until desired convergence is achieved. Overly weak
 links (ones weaker than some threshold, epsilon) are pruned repeatedly.
 
 Currently a naive implementation is provided, using the standard matrix representation
 and multiplication algorithms. Significant benefits may be achieved by taking advantage
 of the (usual) sparsity of the matrices - this will be addressed by a future iteration.
*/

inline double** matrix_multiply(double **a, double **b, int n, int l, int m)
{
    double **c = new double*[n];
    for (int i=0;i<n;i++) c[i] = new double[m];
    
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            c[i][j] = 0;
            for(int k=0;k<l;k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

inline double** expand(double **a, int n, lld e)
{
    double** ret = new double*[n];
    for (int i=0;i<n;i++)
    {
        ret[i] = new double[n];
        for (int j=0;j<n;j++)
        {
            if (i == j) ret[i][j] = 1;
            else ret[i][j] = 0;
        }
    }
    
    while (e)
    {
        if (e & 1) ret = matrix_multiply(ret, a, n, n, n);
        e >>= 1;
        a = matrix_multiply(a, a, n, n, n);
    }
    
    return ret;
}

inline double pow(double num, lld pow)
{
    double ret = 1.0;
    while (pow)
    {
        if (pow & 1) ret *= num;
        pow >>= 1;
        num *= num;
    }
    return ret;
}

inline void inflate(double **a, int n, lld r, double eps)
{
    for (int i=0;i<n;i++)
    {
        double sum = 0.0;
        bool has_nonzero = false;
        for (int j=0;j<n;j++)
        {
            if (a[i][j] > eps)
            {
                double tmp = pow(a[i][j], r);
                if (tmp > eps)
                {
                    a[i][j] = tmp;
                    sum += tmp;
                    has_nonzero = true;
                }
                else a[i][j] = 0.0;
            }
        }
        if (has_nonzero)
        {
            for (int j=0;j<n;j++) a[i][j] /= sum;
        }
    }
}

inline double** normalise(double **a, int n, double eps)
{
    double **ret = new double*[n];
    for (int i=0;i<n;i++)
    {
        ret[i] = new double[n];
        double sum = 0.0;
        bool has_nonzero = false;
        for (int j=0;j<n;j++)
        {
            if (a[i][j] > eps)
            {
                ret[i][j] = a[i][j];
                sum += a[i][j];
                has_nonzero = true;
            }
            else ret[i][j] = 0.0;
        }
        if (has_nonzero)
        {
            for (int j=0;j<n;j++) ret[i][j] /= sum;
        }
    }
    return ret;
}

inline double sq_diff(double **a, double **b, int n)
{
    double ret = 0.0;
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<n;j++)
        {
            ret += (a[i][j] - b[i][j]) * (a[i][j] - b[i][j]);
        }
    }
    return ret;
}

inline vector<int> get_component(int start, vector<vector<int> > &graph, bool *mark, double eps)
{
    vector<int> ret;
    queue<int> q;
    q.push(start);
    
    while (!q.empty())
    {
        int xt = q.front();
        q.pop();
        
        mark[xt] = true;
        ret.push_back(xt);
        
        for (int j=0;j<graph[xt].size();j++)
        {
            int neighbour = graph[xt][j];
            if (mark[neighbour]) continue;
            q.push(neighbour);
            mark[neighbour] = true;
        }
    }
    
    return ret;
}

inline vector<vector<int> > build_clusters(double **a, int n, double eps)
{
    bool *mark = new bool[n];
    for (int i=0;i<n;i++) mark[i] = false;
    
    vector<vector<int> > graph;
    graph.resize(n);
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<i;j++)
        {
            if (a[i][j] > eps || a[j][i] > eps)
            {
                graph[i].push_back(j);
                graph[j].push_back(i);
            }
        }
    }
    
    vector<vector<int> > ret;
    
    for (int i=0;i<n;i++)
    {
        if (!mark[i])
        {
            ret.push_back(get_component(i, graph, mark, eps));
        }
    }
    
    return ret;
}

inline vector<vector<int> > mcl(double **a, int n, lld e, lld r, double eps, double eps2)
{
    double **m = normalise(a, n, eps);
    double **next_m = m;
    
    do
    {
        m = next_m;
        next_m = expand(m, n, e);
        inflate(next_m, n, r, eps);
    } while (sq_diff(m, next_m, n) > eps2);
    
    return build_clusters(m, n, eps);
}

int main()
{
    int n = 19, e = 2, r = 2;
    double eps = 1e-6, eps2 = 1e-3;
    
    double **a = new double*[n];
    for (int i=0;i<n;i++)
    {
        a[i] = new double[n];
        for (int j=0;j<n;j++) a[i][j] = 0.0;
    }
    
    a[0][1] = a[0][2] = 1.0;
    a[1][0] = a[1][2] = a[1][3] = a[1][4] = 1.0;
    a[2][0] = a[2][1] = a[2][3] = a[1][5] = 1.0;
    a[3][1] = a[3][2] = a[3][4] = a[3][5] = 1.0;
    a[4][1] = a[4][3] = a[4][5] = a[4][6] = 1.0;
    a[5][2] = a[5][3] = a[5][4] = a[5][6] = a[5][7] = 1.0;
    a[6][4] = a[6][5] = 1.0;
    a[7][5] = a[7][8] = a[7][9] = a[7][10] = a[7][11] = 1.0;
    a[8][7] = a[8][9] = a[8][10] = a[8][11] = 1.0;
    a[9][7] = a[9][8] = a[9][10] = a[9][12] = a[9][13] = 1.0;
    a[10][7] = a[10][8] = a[10][9] = a[10][11] = a[10][12] = 1.0;
    a[11][7] = a[11][8] = a[11][10] = a[11][12] = 1.0;
    a[12][9] = a[12][10] = a[12][11] = 1.0;
    a[13][9] = a[13][14] = a[13][16] = a[13][18] = 1.0;
    a[14][13] = a[14][15] = a[14][16] = a[14][17] = 1.0;
    a[15][14] = a[15][16] = a[15][17] = 1.0;
    a[16][13] = a[16][14] = a[16][15] = a[16][17] = a[16][18] = 1.0;
    a[17][14] = a[17][15] = a[17][16] = a[17][18] = 1.0;
    a[18][13] = a[18][16] = a[18][17] = 1.0;
    
    vector<vector<int> > result = mcl(a, n, e, r, eps, eps2);
    
    printf("MCL found %d clusters, as follows:\n", result.size());
    for (int i=0;i<result.size();i++)
    {
        printf("{");
        for (int j=0;j<result[i].size();j++)
        {
            printf("%d%s", result[i][j], (j < result[i].size() - 1) ? ", " : "}");
        }
        printf("\n");
    }
    
    return 0;
}
