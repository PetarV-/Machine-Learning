/*
 Petar 'PetarV' Velickovic
 Algorithm: k-means Clustering (Lloyd's Algorithm)
*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
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
 The k-means Clustering algorithm aims to partition a set of observed points in a
 multi-dimensional space into k subsets (clusters). 
 
 In general, finding the optimal means is an NP-hard problem, and therefore a heuristic
 approach (Lloyd's Algorithm) is performed; initially, k means are selected at random,
 and then each observation is assigned into the cluster corresponding to the mean which 
 is closest to it. The means are then updated and the procedure is iterated until 
 convergence is achieved. As this heuristic is very efficient, one may run it multiple times 
 (with different initial conditions) and choose the most-preferred clustering.
*/

double sq_distance(vector<double> &a, vector<double> &b)
{
    assert(a.size() == b.size());
    double ret = 0.0;
    for (int i=0;i<a.size();i++)
    {
        ret += (a[i] - b[i]) * (a[i] - b[i]);
    }
    
    return ret;
}

vector<vector<int> > k_means(vector<vector<double> > &pts, int k)
{
    int n = pts.size();
    int d = pts[0].size();
    assert(n >= k);
    
    /*
     Here, Forgy initialisation is used: initially one randomly chooses k
     of the observations to serve as the initial means.
     
     Another approach is random initialisation: initially assigning each
     observation to a cluster, and then computing means from there.
    */
    vector<vector<double> > means;
    means.resize(k);
    set<int> chosen;
    for (int i=0;i<k;i++)
    {
        int id;
        do
        {
            id = rand() % n;
        } while (chosen.count(id) > 0);
        
        means[i].resize(d);
        for (int j=0;j<d;j++) means[i][j] = pts[id][j];
        
        chosen.insert(id);
    }
    
    vector<int> cluster_assigned;
    cluster_assigned.resize(n);
    for (int i=0;i<n;i++) cluster_assigned[i] = 0;
    
    bool change;
    
    do
    {
        change = false;
        // Assignment step: assign each observation to closest mean's cluster
        for (int i=0;i<n;i++)
        {
            double best_dist = 0.0;
            int best_mean = -1;
            for (int j=0;j<k;j++)
            {
                double curr_dist = sq_distance(pts[i], means[j]);
                if (best_mean == -1 || best_dist > curr_dist)
                {
                    best_dist = curr_dist;
                    best_mean = j;
                }
            }
            if (best_mean != cluster_assigned[i]) change = true;
            cluster_assigned[i] = best_mean;
        }
        
        // Update step: recompute the means for each cluster
        vector<int> counts;
        counts.resize(k);
        for (int i=0;i<k;i++)
        {
            for (int j=0;j<d;j++)
            {
                means[i][j] = 0.0;
            }
        }
        
        for (int i=0;i<n;i++)
        {
            counts[cluster_assigned[i]]++;
            for (int j=0;j<d;j++)
            {
                means[cluster_assigned[i]][j] += pts[i][j];
            }
        }
        
        for (int i=0;i<k;i++)
        {
            assert(counts[i] > 0);
            for (int j=0;j<d;j++)
            {
                means[i][j] /= counts[i];
            }
        }
    } while (change); // Iterate until convergence
    
    // Reconstruct the clustering
    vector<vector<int> > ret;
    ret.resize(k);
    
    for (int i=0;i<n;i++)
    {
        ret[cluster_assigned[i]].push_back(i);
    }
    
    return ret;
}

int main()
{
    // Generate an easily separable test set
    int n = 50;
    int d = 2;
    int k = 4;
    
    vector<vector<double> > points;
    
    int pos[4][2] = {{0, 0}, {5, 0}, {0, 5}, {5, 5}};
    
    for (int i=0;i<k;i++)
    {
        for (int j=0;j<n;j++)
        {
            vector<double> coords;
            coords.resize(d);
            for (int k=0;k<d;k++)
            {
                double rnd = (double)rand() / RAND_MAX;
                double delta = 2 * rnd;
                if (rand() % 2) delta *= -1;
                coords.push_back(pos[i][k] + delta);
            }
            points.push_back(coords);
        }
    }
    
    vector<vector<int> > ret = k_means(points, k);
    
    printf("k-means clustering found the following clusters:\n");
    for (int i=0;i<ret.size();i++)
    {
        printf("{");
        for (int j=0;j<ret[i].size();j++)
        {
            printf("%d%s", ret[i][j], (j != ret[i].size() - 1) ? ", " : "}\n");
        }
    }
    
    return 0;
}
