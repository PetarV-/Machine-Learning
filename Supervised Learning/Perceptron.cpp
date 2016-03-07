/*
 Petar 'PetarV' Velickovic
 Data Structure: Perceptron
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

#define EPS 1e-6 // convergence criterion
#define ETA 1e-3 // learning rate

using namespace std;
typedef long long lld;
typedef function<double(vector<double>, vector<double>)> func; // the common type of functions

/*
 The perceptron is the simplest neural network---one that contains only a single neuron.
 It can therefore be used as a basic building block for constructing more complicated
 neural network architectures.
 
 A perceptron maintains a vector, w, of (n + 1) weights. Upon receiving an input, x, of
 size n, it first computes the dot product of the two vectors, and then applies an
 activation function, sigma, on the result:
 
    h(w; x) = sigma(w_0 + sum_{i=1}^{n} (w_i * x_i))
 
 There are three standard types of activation functions commonly used:
    - Identity: sigma(z) = z
        (commonly used for regression problems)
    - Step:     sigma(z) = if z > 0 then 1 else 0
        (commonly used for classification problems)
    - Sigmoid; the two primary examples of which are:
        * Logistic:             sigma(z) = 1 / (1 + exp(-z))
        * Hyperbolic tangent:   sigma(z) = tanh(z)
        (commonly used for probabilistic classification)
 
 The main purpose of a perceptron is to solve supervised learning problems; deriving
 optimal weights from a training set of (input, output) pairs. This can be done by
 starting from a randomly initialised weight vector and then updating it using
 gradient descent, optimising e.g. the sum of squared errors.
 
 There are many problems that a single perceptron can't solve---a common example is
 learning the XOR function, where the data is not linearly separable. To solve them,
 combining many perceptrons together in some form is required.
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
    
public:
    // Initialises a perceptron that accepts n (+1) inputs
    Perceptron(int n, func h, func d) : n(n), h(h), d(d)
    {
        w = vector<double>(n + 1);
        // Initialise each weight to a random value between 0 and 1
        for (int i=0;i<=n;i++)
        {
            w[i] = rand() * 1.0 / RAND_MAX;
        }
    }
    
    // Computes the output of the perceptron on a given input
    double val(vector<double> x)
    {
        return h(w, x);
    }
    
    // Trains the perceptron on a given training set by gradient descent
    void train(vector<vector<double> > x, vector<int> y)
    {
        assert(x.size() == y.size());
        vector<double> delta(n + 1, 0.0);
        double diff = 0.0;
        do
        {
            diff = 0.0;
            for (int i=0;i<x.size();i++)
            {
                for (int j=0;j<=n;j++)
                {
                    double curr = ETA * (y[i] - h(w, x[i])) * d(w, x[i]) * x[i][j];
                    diff += curr * curr;
                    w[j] += curr;
                }
            }
        } while (diff > EPS);
    }
};

int main()
{
    srand(time(NULL));
    
    int t = 1000;
    
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
    
    // Trains the perceptron to classify (x, y) pairs based on whether x > y
    vector<vector<double> > trn;
    vector<int> vals;
    for (int i=0;i<t;i++)
    {
        int x = rand() % 5 + 1;
        int y = rand() % 5 + 1;
        while (y == x) y = rand() % 5 + 1;
        vector<double> set;
        set.push_back(x); set.push_back(y);
        trn.push_back(set);
        if (x > y) vals.push_back(1);
        else vals.push_back(0);
    }
    
    Perceptron p(2, logistic, d_logistic);
    p.train(trn, vals);
    
    int correct = 0;
    int tot = 1000;
    for (int i=0;i<tot;i++)
    {
        double x = rand();
        double y = rand();
        vector<double> test = {x, y};
        double prob = p.val(test);
        bool expected = (x > y);
        bool predicted = (prob > 0.5);
        if (expected == predicted) correct++;
    }
    cout << "accuracy = " << correct * 1.0 / tot << endl;
    
    return 0;
}