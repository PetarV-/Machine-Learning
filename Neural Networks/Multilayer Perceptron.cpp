/*
 Petar 'PetarV' Velickovic
 Data Structure: Multilayer Perceptron
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
#include <functional>
#include <tuple>
#include <random>
#include <fstream>

#define EPS 1e-6 // convergence criterion
#define ETA 1e-3 // learning rate

class MLP;

using namespace std;

typedef long long lld;
typedef function<double(vector<double>, vector<double>)> func; // the common type of functions
typedef function<double(MLP)> prior; // P(w)
typedef function<double(vector<vector<double> >, vector<vector<double> >, MLP)> likelihood; // P(y|x,w)

/*
 The multilayer perceptron is a neural network architecture representing a fully
 unrestricted feedforward structure of perceptrons. In the extreme (and most common)
 case, the neurons are fully connected. This implies that, for a network with one
 hidden layer of neurons, each neuron in the hidden layer receives a copy of all the
 input features to compute its function on; after that, all of the output layer neurons
 receive all of the hidden layer neurons' outputs to compute the final output.
 
 The network's weights can once again be trained by gradient descent, as discussed in
 the perceptron implementation; this time it is necessary to first compute the gradients
 for the output layer weights and then go backwards from there (this is the
 backpropagation algorithm).
 
 By the universal approximation theorem, it is possible to construct a multilayer
 perceptron with a single hidden layer that approximates *any* real-valued function,
 as long as sigmoid activation functions are used. However the proof of this theorem
 is nonconstructive, and therefore does not help in devising appropriate training
 methods. Recent research attempts to circumvent this problem by having multiple hidden
 layers (an approach commonly known as ``deep learning'').
*/

// Helper function to compute the dot product of two vectors
double scalar_product(vector<double> a, vector<double> b)
{
    assert(a.size() == b.size());
    double ret = 0.0;
    for (int i=0;i<a.size();i++)
    {
        ret += a[i] * b[i];
    }
    return ret;
}

class Perceptron
{
private:
    int n;
    vector<double> w;
    function<double(vector<double>, vector<double>)> h; // the output function of the perceptron
    function<double(vector<double>, vector<double>)> d; // the derivative of the output function
    
    // stores the currently computed output and derivative
    // which will be needed for (forward/back)propagation
    double z, der, delta;
    
public:
    // empty constructor provided just for convenience
    Perceptron() { }
    // Initialises a perceptron that accepts n (+1) inputs
    Perceptron(int n, func h, func d) : n(n), h(h), d(d)
    {
        default_random_engine gen;
        uniform_real_distribution<double> U(0.0, 1.0);
        w = vector<double>(n + 1);
        // Initialise each weight to a random value between 0 and 1
        for (int i=0;i<=n;i++)
        {
            w[i] = U(gen);
        }
    }
    
    // Computes the output of the perceptron on a given input, storing it in z for later use
    // also computes the derivative
    double val(vector<double> x)
    {
        der = d(w, x);
        return (z = h(w, x));
    }
    
    friend class MLP;
};

class MLP
{
private:
    int n; // the number of neurons in this MLP
    int in_size, out_size; // the number of inputs and outputs for this MLP
    
    /*
     Convention:
     - the first in_size neurons are input neurons
     - the last out_size neurons are output neurons
     - the neurons are given in topological order (enforced also by conn)
     */
    vector<Perceptron> neurons;
    
    // conn[i][j] = k (< i) specifies that the output of the k-th neuron
    // should be used as the j-th input for neuron i
    vector<vector<int> > conn;
    
    // chd[i] is a vector of pairs {x, y} such that the x-th neuron receives the
    // output of the i-th neuron as its y-th input
    vector<vector<pair<int, int> > > chd;
    
public:
    // Creates a fully-connected MLP that has in_size input neurons
    // It is assumed that each individual layer's neurons compute the same activation function
    // Each layer is specified by a <size, sigma, d_sigma> tuple
    // The final layer specifies the output layer
    MLP(int in_size, vector<tuple<int, func, func> > layers)
    {
        assert(!layers.empty());
        this -> in_size = in_size;
        this -> out_size = get<0>(layers[layers.size() - 1]);
        
        n = in_size;
        for (int i=0;i<layers.size();i++) n += get<0>(layers[i]);
        
        neurons.resize(n);
        conn.resize(n);
        chd.resize(n);
        
        // specially, will need the identity function for the input neurons
        auto ident = [] (vector<double> w, vector<double> x) -> double
        {
            return x[0];
        };
        auto d_ident = [] (vector<double> w, vector<double> x) -> double
        {
            return 1.0;
        };
        
        int curr_layer = 0;
        int ind = 0;
        int start_neuron = 0; // the starting neuron in the previous layer
        // initialise the neurons
        for (int i=0;i<n;i++)
        {
            // The structure allows for arbitrary topologies---therefore to keep it general we must
            // initialise input neurons here (although this is redundant if fully connected)
            if (i < in_size)
            {
                neurons[i] = Perceptron(1, ident, d_ident);
            }
            else
            {
                if (curr_layer == 0)
                {
                    neurons[i] = Perceptron(in_size, get<1>(layers[curr_layer]), get<2>(layers[curr_layer]));
                    conn[i].resize(in_size);
                    for (int j=0;j<in_size;j++)
                    {
                        conn[i][j] = j;
                        chd[j].push_back(make_pair(i, j));
                    }
                }
                else
                {
                    int num_inputs = get<0>(layers[curr_layer - 1]);
                    neurons[i] = Perceptron(num_inputs, get<1>(layers[curr_layer]), get<2>(layers[curr_layer]));
                    conn[i].resize(num_inputs);
                    for (int j=0;j<num_inputs;j++)
                    {
                        conn[i][j] = start_neuron + j;
                        chd[start_neuron + j].push_back(make_pair(i, j));
                    }
                }
                ind++;
                if (ind == get<0>(layers[curr_layer]))
                {
                    start_neuron = i - ind + 1;
                    ind = 0;
                    curr_layer++;
                }
            }
        }
    }
    
    double get_W_norm() // gets the squared norm of weights
    {
        double ret = 0;
        for (int i=in_size;i<n;i++)
        {
            for (int j=0;j<neurons[i].w.size();j++)
            {
                ret += neurons[i].w[j] * neurons[i].w[j];
            }
        }
        return ret;
    }
    
    // Computes the output value of this MLP for a given input
    vector<double> val(vector<double> x)
    {
        vector<double> ret(out_size);
        
        // Compute the outputs of neurons, one at a time
        for (int i=0;i<n;i++)
        {
            if (i < in_size)
            {
                neurons[i].val({x[i]});
            }
            else
            {
                vector<double> inps(neurons[i].n);
                for (int j=0;j<conn[i].size();j++)
                {
                    inps[j] = neurons[conn[i][j]].z;
                }
                neurons[i].val(inps);
            }
        }
        
        for (int i=0;i<out_size;i++)
        {
            ret[i] = neurons[n - out_size + i].z;
        }
        
        return ret;
    }
    
    // Trains the perceptron on a given training set by backpropagation
    void backpropagation(vector<vector<double> > x, vector<vector<double> > y, int max_steps)
    {
        assert(x.size() == y.size());
        double diff = 0.0;
        do
        {
            diff = 0.0;
            // batch training - always retrain on all examples sequentially
            for (int p=0;p<x.size();p++)
            {
                vector<double> out = val(x[p]);
                for (int i=n-1;i>=in_size;i--)
                {
                    // directly reestimate the output neurons' weights
                    // we're doing probabilistic classification, so optimising the likelihood
                    // rather than the sum of squared errors this time
                    if (i >= n - out_size)
                    {
                        int ind = out_size - (n - i);
                        neurons[i].delta = (y[p][ind] - neurons[i].z);
                    }
                    else
                    {
                        // sum deltas over all children
                        double sum_chd = 0.0;
                        for (int j=0;j<chd[i].size();j++)
                        {
                            sum_chd += neurons[chd[i][j].first].delta * neurons[chd[i][j].first].w[chd[i][j].second];
                        }
                        neurons[i].delta = neurons[i].der * sum_chd;
                    }
                }
                for (int i=in_size;i<n;i++)
                {
                    for (int j=0;j<neurons[i].w.size();j++)
                    {
                        double z = (j < neurons[i].n) ? neurons[conn[i][j]].z : 1.0;
                        double curr = ETA * neurons[i].delta * z;
                        neurons[i].w[j] += curr;
                        diff += curr * curr;
                    }
                }
            }
        } while (diff > EPS && (--max_steps));
    }
    
    // Trains the perceptron on a given training set by the Metropolis algorithm
    vector<MLP> metropolis(vector<vector<double> > x, vector<vector<double> > y, prior p, likelihood l, int steps)
    {
        default_random_engine gen;
        normal_distribution<double> N(0.0, 0.04);
        uniform_real_distribution<double> U(0.0, 1.0);
        double p_old = p(*this) * l(y, x, *this);
        int accepted = 0;
        double **w_old = new double*[n];
        for (int i=in_size;i<n;i++)
        {
            w_old[i] = new double[neurons[i].w.size()];
        }
        
        vector<MLP> ret;
        
        while (accepted < steps)
        {
            for (int i=in_size;i<n;i++)
            {
                for (int j=0;j<neurons[i].w.size();j++)
                {
                    w_old[i][j] = neurons[i].w[j];
                    neurons[i].w[j] += N(gen);
                }
            }
            double p_new = p(*this) * l(y, x, *this);
            if (U(gen) * p_old < p_new)
            {
                accepted++;
                if (accepted >= (steps - 50)) ret.push_back(*this);
                p_old = p_new;
            }
            else
            {
                for (int i=in_size;i<n;i++)
                {
                    for (int j=0;j<neurons[i].w.size();j++)
                    {
                        neurons[i].w[j] = w_old[i][j];
                    }
                }
            }
        }
        
        return ret;
    }
};

int main()
{
    srand(time(NULL));
    
    default_random_engine gen;
    uniform_real_distribution<double> U(0.0, 1.0);
    normal_distribution<double> N(0.0, 0.05);
    
    // Here the logistic function will be used for the perceptron
    auto logistic = [] (vector<double> w, vector<double> x) -> double
    {
        x.push_back(1.0);
        return 1.0 / (1.0 + exp(-scalar_product(w, x)));
    };
    auto d_logistic = [logistic] (vector<double> w, vector<double> x) -> double
    {
        double lst = logistic(w, x);
        return lst * (1.0 - lst);
    };
    
    int t = 1000;
    
    // Trains the perceptron to classify (x, y) pairs (with added noise) based on their XOR
    vector<vector<double> > trn;
    vector<vector<double> > vals;
    for (int i=0;i<t;i++)
    {
        int x = rand() % 2 + U(gen) * ((rand() % 2) ? 1 : -1);
        int y = rand() % 2 + U(gen) * ((rand() % 2) ? 1 : -1);
        bool A = (x >= 0.5);
        bool B = (y >= 0.5);
        vector<double> set;
        set.push_back(x); set.push_back(y);
        trn.push_back(set);
        if (A == B) vals.push_back({1.0});
        else vals.push_back({0.0});
    }
    
    MLP mlp(2, {make_tuple(5, logistic, d_logistic), make_tuple(1, logistic, d_logistic)});
    //mlp.backpropagation(trn, vals, 1000);
    
    int correct = 0;
    int tot = 1000;
    for (int i=0;i<tot;i++)
    {
        int x = rand() % 2 + U(gen) * ((rand() % 2) ? 1 : -1);
        int y = rand() % 2 + U(gen) * ((rand() % 2) ? 1 : -1);
        vector<double> test = {(double)x, (double)y};
        double prob = mlp.val(test)[0];
        bool expected = (x >= 0.5) == (y >= 0.5);
        bool predicted = (prob > 0.5);
        if (expected == predicted) correct++;
    }
    cout << "accuracy = " << correct * 1.0 / tot << endl;
    
    // Gaussian prior
    auto g_prior = [] (MLP mlp) -> double
    {
        double alpha = 1.0; // hyperparameter
        double nw2 = mlp.get_W_norm(); // weights' norm squared
        // do not include the normalising constant, because it gets cancelled out anyway!
        return exp(-alpha * 0.5 * nw2);
    };
    
    // Gaussian likelihood
    auto g_likelihood = [] (vector<vector<double> > y, vector<vector<double> > x, MLP mlp) -> double
    {
        assert(x.size() == y.size());
        double beta = 30.0; // hyperparameter
        int m = y.size();
        double err = 0.0;
        for (int i=0;i<m;i++)
        {
            vector<double> out = mlp.val(x[i]);
            assert(out.size() == y[i].size());
            for (int j=0;j<y[i].size();j++)
            {
                err += (y[i][j] - out[j]) * (y[i][j] - out[j]);
            }
        }
        // do not include the normalising constant, because it gets cancelled out anyway!
        return exp(-beta * 0.5 * err);
    };
    
    // Now the hyperbolic tangent will be used for the hidden layer
    auto tgh = [] (vector<double> w, vector<double> x) -> double
    {
        x.push_back(1.0);
        return tanh(scalar_product(w, x));
    };
    auto d_tgh = [tgh] (vector<double> w, vector<double> x) -> double
    {
        double tnh = tgh(w, x);
        return 1.0 - tnh * tnh;
    };
    
    // and the identity will be used for the output
    auto ident = [] (vector<double> w, vector<double> x) -> double
    {
        x.push_back(1.0);
        return scalar_product(w, x);
    };
    auto d_ident = [] (vector<double> w, vector<double> x) -> double
    {
        return 1.0;
    };
    
    t = 30;
    trn.clear(); vals.clear();
    for (int i=0;i<t;i++)
    {
        double x = ((rand() % 2) ? 0.25 : 0.75) + N(gen);
        double y = 0.5 + 0.4 * sin(2.0 * M_PI * x) + N(gen);
        trn.push_back({x});
        vals.push_back({y});
    }
    
    MLP mlp2(1, {make_tuple(4, tgh, d_tgh), make_tuple(1, ident, d_ident)});
    vector<MLP> ret = mlp2.metropolis(trn, vals, g_prior, g_likelihood, 100);
    
    // dump the obtained data for further processing/plotting
    ofstream os("/Users/PetarV/Desktop/percep/points.txt");
    for (int i=0;i<t;i++)
    {
        os << trn[i][0] << " " << vals[i][0] << endl;
    }
    os.close();
    
    os = ofstream("/Users/PetarV/Desktop/percep/plots.txt");
    for (int i=0;i<ret.size();i+=10)
    {
        for (double x=0.0;x<=1.0;x+=0.001)
        {
            os << ret[i].val({x})[0] << " ";
        }
        os << endl;
    }
    os.close();
    
    os = ofstream("/Users/PetarV/Desktop/percep/mean_var.txt");
    vector<double> means, vars;
    for (double x=0.0;x<=1.0;x+=0.001)
    {
        double mean = 0.0, var = 0.0;
        for (int i=0;i<ret.size();i++)
        {
            mean += ret[i].val({x})[0];
        }
        mean /= ret.size();
        for (int i=0;i<ret.size();i++)
        {
            double val = ret[i].val({x})[0];
            var += (val - mean) * (val - mean);
        }
        var /= ret.size() - 1;
        
        means.push_back(mean); vars.push_back(var);
    }
    
    for (int i=0;i<means.size();i++)
    {
        os << means[i] << " ";
    }
    os << endl;
    for (int i=0;i<means.size();i++)
    {
        os << means[i] - vars[i] << " ";
    }
    os << endl;
    for (int i=0;i<means.size();i++)
    {
        os << means[i] + vars[i] << " ";
    }
    os << endl;
    
    return 0;
}