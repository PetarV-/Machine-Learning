/*
 Petar 'PetarV' Velickovic
 Hidden Markov Model (HMM)
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
#include <tuple>

#define EPS 1e-3

using namespace std;
typedef long long lld;

/*
 The Hidden Markov Model (HMM) is a model primarily designed for handling temporal data
 sets (i.e. observations of how some features change over time---a great usage example is
 speech recognition), however it may also be used to represent any observations which can 
 be sensibly ordered in some way.
 
 An HMM is a Markov Chain in which the sequence of states cannot be directly observed. However,
 a sequence of "outputs" can be observed, with each assumed state in the sequence emitting one
 output before transitioning to the next state in that sequence. The model is fully specified
 by the start-state probability vector, transition probability matrix (as for any Markov Chain)
 and an output probability matrix (specifying the probabilities of outputting y when in state x).
 This assumes a discrete model of states and outputs, but can be generalised to continuous spaces.
 
 An HMM is capable of solving efficiently, through standard algorithms, three fundamental problems:
 
     1. (Evaluation) Given an HMM (H) and a sequence of observations (y), determine the probability
        that the sequence was produced by the HMM (P(y | H)).
        This is solved by using the forward algorithm.
 
     2. (Decoding) Given an HMM (H) and a sequence of observations (y), determine the most-likely
        sequence of states used to produce the observation sequence (argmax_x P(x | H, y)).
        This is solved by using the Viterbi algorithm.
 
     3. (Learning) Given an HMM (H) and a sequence of observation (y), produce a new HMM (H') such
        that it is more likely to have produced the observation sequence (P(y | H') >= P(y | H)).
        This is solved by using the Baum-Welch algorithm.
 
 Thus the typical usage of the HMM involves training it on a given training set of sequences
 believed to be produced by the model (by iterating the Baum-Welch algorithm a sufficient number
 of times), and then using the forward or Viterbi algorithm to make inferences on unseen sequences
 (the exact usage depending on the problem at hand).
*/

class HMM
{
private:
    int n;      // number of nodes
    int m;      // number of observations
    double *pi; // start-state probability vector
    double **T; // transition probability matrix
    double **O; // output probability matrix
    
public:
    // Generate a new HMM with N states and M observations
    HMM(int n, int m) : n(n), m(m)
    {
        // Initialise the start-state probabilities
        this -> pi = new double[n];
        for (int i=0;i<n;i++)
        {
            this -> pi[i] = 1.0 / n;
        }
        
        // Initialise the transition probabilities
        this -> T = new double*[n];
        for (int i=0;i<n;i++)
        {
            this -> T[i] = new double[n];
            for (int j=0;j<n;j++)
            {
                this -> T[i][j] = 1.0 / n;
            }
        }
        
        // Initialise the output probabilities
        this -> O = new double*[n];
        for (int i=0;i<n;i++)
        {
            this -> O[i] = new double[m];
            for (int j=0;j<m;j++)
            {
                this -> O[i][j] = 1.0 / m;
            }
        }
    }
    
    // Generate an HMM from pre-defined parameters
    HMM(int n, int m, double *pi, double **T, double **O) : n(n), m(m)
    {
        double sum = 0.0;
        this -> pi = new double[n];
        for (int i=0;i<n;i++)
        {
            this -> pi[i] = pi[i];
            sum += pi[i];
        }
        assert(fabs(sum - 1.0) < EPS); // Make sure the probabilities sum to unity
        
        this -> T = new double*[n];
        for (int i=0;i<n;i++)
        {
            this -> T[i] = new double[n];
            sum = 0.0;
            for (int j=0;j<n;j++)
            {
                this -> T[i][j] = T[i][j];
                sum += T[i][j];
            }
            assert(fabs(sum - 1.0) < EPS);
        }
        
        this -> O = new double*[n];
        for (int i=0;i<n;i++)
        {
            this -> O[i] = new double[m];
            sum = 0.0;
            for (int j=0;j<m;j++)
            {
                this -> O[i][j] = O[i][j];
                sum += O[i][j];
            }
            assert(fabs(sum - 1.0) < EPS);
        }
    }
    
    // Destructor: free memory
    ~HMM()
    {
        for (int i=0;i<n;i++) delete[] T[i];
        delete[] T;
        
        for (int i=0;i<n;i++) delete[] O[i];
        delete[] O;
        
        delete[] pi;
    }
    
    // Getters for the private fields
    double* get_pi() { return pi; }
    double** get_T() { return  T; }
    double** get_O() { return  O; }
    
    /* 
     Forward algorithm
       
     Input:      An observation sequence Y of length T.
     Output:     A triplet (alpha, c, L), where
                     - alpha(t, x) is the probability of producing the first
                       t elements of Y, and ending up in state x;
                     - c is a vector of scaling coefficients used at each step,
                       such that for any t', sum_x alpha(t', x) = 1 holds;
                     - L is the (log-)likelihood of producing sequence Y.
     Complexity: O(T * n^2) time, O(T * n) memory
    */
    tuple<double**, double*, double> forward(vector<int> &y)
    {
        int len = y.size();
        
        double **alpha = new double*[len];
        for (int t=0;t<len;t++)
        {
            alpha[t] = new double[n];
        }
        
        double *c = new double[len];
        
        // Base case: alpha(0, x) = pi(x) * O(x, Y[0])
        double sum = 0.0;
        for (int i=0;i<n;i++)
        {
            alpha[0][i] = pi[i] * O[i][y[0]];
            sum += alpha[0][i];
        }
        
        // Scaling
        c[0] = 1.0 / sum;
        for (int i=0;i<n;i++)
        {
            alpha[0][i] /= sum;
        }
        
        // Recurrence relation: alpha(t+1, x) = sum_x' alpha(t, x') * T(x', x) * O(x, Y[t+1])
        for (int t=1;t<len;t++)
        {
            sum = 0.0;
            for (int i=0;i<n;i++)
            {
                alpha[t][i] = 0.0;
                for (int j=0;j<n;j++)
                {
                    alpha[t][i] += alpha[t-1][j] * T[j][i];
                }
                alpha[t][i] *= O[i][y[t]];
                sum += alpha[t][i];
            }
            
            // Scaling
            c[t] = 1.0 / sum;
            for (int i=0;i<n;i++)
            {
                alpha[t][i] /= sum;
            }
        }
        
        // Deriving the log-likelihood from the scaling coefficients
        double log_L = 0.0;
        for (int t=0;t<len;t++) log_L -= log(c[t]);
        
        return make_tuple(alpha, c, log_L);
    }
    
    /*
     Backward algorithm
     
     Input:      An observation sequence Y of length T, scaling coefficients c
     Output:     A matrix beta, where beta(t, x) is the likelihood of producing the
                 output elements Y[t+1], Y[t+2], ... Y[T], assuming we start from x.
                 The entries are scaled at each t using the given scaling coefficients.
     Complexity: O(T * n^2) time, O(T * n) memory
    */
    double** backward(std::vector<int> &y, double *c)
    {
        int len = y.size();
        
        double **beta = new double*[len];
        for (int t=0;t<len;t++)
        {
            beta[t] = new double[n];
        }
        
        // Base case: beta(T-1, x) = 1
        for (int i=0;i<n;i++) beta[len-1][i] = 1.0;
        
        // Recurrence relation: beta(t, x) = sum_x' T(x, x') * O(x', Y[t+1]) * beta(t+1, x')
        for (int t=len-2;t>=0;t--)
        {
            for (int i=0;i<n;i++)
            {
                beta[t][i] = 0.0;
                for (int j=0;j<n;j++)
                {
                    beta[t][i] += T[i][j] * O[j][y[t+1]] * beta[t+1][j];
                }
                
                // Scaling
                beta[t][i] *= c[t+1];
            }
        }
        
        return beta;
    }
    
    /*
     Viterbi algorithm
     
     Input:      An observation sequence Y of length T
     Output:     The most likely state sequence that produces this sequence
     Complexity: O(T * n^2) time, O(T * n) memory
    */
    vector<int> viterbi(std::vector<int> &y)
    {
        int len = y.size();
        
        double **v = new double*[len];
        int **ptr = new int*[len];
        for (int t=0;t<len;t++)
        {
            v[t] = new double[n];
            ptr[t] = new int[n];
        }
        
        // Base case: V(0, x) = pi(x) * O(x, Y[0])
        for (int i=0;i<n;i++)
        {
            v[0][i] = log(pi[i]) + log(O[i][y[0]]);
        }
        
        // Recurrence relation: V(t+1, x) = max_x' V(t, x') * T(x', x) * O(x, Y[t+1])
        for (int t=1;t<len;t++)
        {
            for (int i=0;i<n;i++)
            {
                double max_val = -1.0;
                int arg_max = -1;
                for (int j=0;j<n;j++)
                {
                    double curr_val = v[t-1][j] + log(T[j][i]) + log(O[i][y[t]]);
                    if (arg_max == -1 || curr_val > max_val)
                    {
                        max_val = curr_val;
                        arg_max = j;
                    }
                }
                v[t][i] = max_val;
                ptr[t][i] = arg_max;
            }
        }
        
        // Determining the final state in the optimal sequence, as argmax_x V(T-1, x)
        double best = -1.0;
        int arg_best = -1;
        for (int i=0;i<n;i++)
        {
            if (arg_best == -1 || v[len-1][i] > best)
            {
                best = v[len-1][i];
                arg_best = i;
            }
        }
        
        // Backtracking from there, using ptr[][], to obtain the full solution
        vector<int> ret;
        ret.resize(len);
        for (int t=len-1;t>=0;t--)
        {
            ret[t] = arg_best;
            arg_best = ptr[t][arg_best];
        }
        
        for (int i=0;i<len;i++) delete[] v[i];
        delete[] v;
        
        for (int i=0;i<len;i++) delete[] ptr[i];
        delete[] ptr;
        
        return ret;
    }
    
    /*
     Baum-Welch algorithm
     
     Input:      An observation sequence Y of length T, the maximal number of iterations to make,
                 and the tolerance to change (at smaller changes, convergence is assumed).
     Output:     A reevaluation of the model's parameters, such that it is more likely to produce Y.
     Complexity: O(T * (n^2 + nm)) time (per iteration), O(T * n) memory
    */
    void baumwelch(vector<int> y, int iterations, double tolerance)
    {
        int len = y.size();
        
        double lhood = 0.0, old_lhood = 0.0;
        double PP, QQ;
        
        for (int iter=0;iter<iterations;iter++)
        {
            lhood = 0.0;
            
            /*** E Step ***/
            // Run the forward algorithm
            tuple<double**, double*, double> x = forward(y);
            double **alpha = get<0>(x);
            double *c = get<1>(x);
            lhood += get<2>(x);
            
            // Run the backward algorithm, re-using the scaling coefficients
            double **beta = backward(y, c);
            
            /*** M Step ***/
            double **next_O = new double*[n];
            for (int i=0;i<n;i++)
            {
                next_O[i] = new double[m];
            }
            
            for (int i=0;i<n;i++)
            {
                // Reestimating pi(i) as alpha(0, i) * beta(0, i)
                pi[i] = alpha[0][i] * beta[0][i];
                
                QQ = 0.0;
                
                // Reestimating T(i, j) as sum xi(t, i, j) / sum gamma(t, i)
                for (int t=0;t<len-1;t++)
                {
                    QQ += alpha[t][i] * beta[t][i];
                }
                
                for (int j=0;j<n;j++)
                {
                    PP = 0.0;
                    for (int t=0;t<len-1;t++)
                    {
                        PP += alpha[t][i] * O[j][y[t+1]] * beta[t+1][j] * c[t+1];
                    }
                    T[i][j] *= PP / QQ;
                }
                
                // Reestimating O(x, o) as sum I(y[t] = o) * gamma(t, i) / sum gamma(t, i)
                for (int k=0;k<m;k++)
                {
                    next_O[i][k] = 0.0;
                }
                
                for (int t=0;t<len-1;t++)
                {
                    next_O[i][y[t]] += alpha[t][i] * beta[t][i];
                }
                
                double last = alpha[len-1][i] * beta[len-1][i];
                next_O[i][y[len-1]] += last;
                QQ += last;
                
                for (int k=0;k<m;k++)
                {
                    next_O[i][k] /= QQ;
                }
            }
            
            /*** Cleanup ***/
            for (int t=0;t<len;t++)
            {
                delete[] alpha[t];
                delete[] beta[t];
            }
            delete[] alpha;
            delete[] beta;
            delete[] c;
            
            for (int i=0;i<n;i++)
            {
                delete[] O[i];
            }
            delete[] O;
            
            O = next_O;
            
            if (fabs(lhood - old_lhood) < tolerance) break;
            old_lhood = lhood;
        }
    }
};

int main()
{
    int n = 2;
    int m = 2;
    
    double *pi = new double[n];
    for (int i=0;i<n;i++)
    {
        pi[i] = 0.5;
    }
    
    double **T = new double*[n];
    for (int i=0;i<n;i++)
    {
        T[i] = new double[n];
    }
    T[0][0] = 0.7; T[0][1] = 0.3;
    T[1][0] = 0.3; T[1][1] = 0.7;
    
    double **O = new double*[n];
    for (int i=0;i<n;i++)
    {
        O[i] = new double[m];
    }
    O[0][0] = 0.9; O[0][1] = 0.1;
    O[1][0] = 0.2; O[1][1] = 0.8;
    
    double exp_A[5][2];
    exp_A[0][0] = 0.8182; exp_A[0][1] = 0.1818;
    exp_A[1][0] = 0.8834; exp_A[1][1] = 0.1166;
    exp_A[2][0] = 0.1907; exp_A[2][1] = 0.8093;
    exp_A[3][0] = 0.7308; exp_A[3][1] = 0.2692;
    exp_A[4][0] = 0.8673; exp_A[4][1] = 0.1327;
    
    double exp_B[5][2];
    exp_B[0][0] = 0.5923; exp_B[0][1] = 0.4077;
    exp_B[1][0] = 0.3763; exp_B[1][1] = 0.6237;
    exp_B[2][0] = 0.6533; exp_B[2][1] = 0.3467;
    exp_B[3][0] = 0.6273; exp_B[3][1] = 0.3727;
    exp_B[4][0] = 0.5000; exp_B[4][1] = 0.5000;
    
    HMM *x = new HMM(n, m, pi, T, O);
    
    vector<int> observations;
    observations.push_back(0);
    observations.push_back(0);
    observations.push_back(1);
    observations.push_back(0);
    observations.push_back(0);
    
    printf("Running forward-backward algorithm...\n");
    tuple<double**, double*, double> ret = x -> forward(observations);
    double **AA = get<0>(ret);
    double **BB = x -> backward(observations, get<1>(ret));
    
    printf("Forward likelihoods:\n");
    for (int t=0;t<5;t++)
    {
        double sum = 0.0;
        for (int i=0;i<n;i++)
        {
            sum += AA[t][i];
        }
        for (int i=0;i<n;i++)
        {
            assert(fabs(exp_A[t][i] - AA[t][i] / sum) < EPS);
            printf("%lf ", AA[t][i] / sum);
        }
        printf("\n");
    }
    
    printf("Backward likelihoods:\n");
    for (int t=0;t<5;t++)
    {
        double sum = 0.0;
        for (int i=0;i<n;i++)
        {
            sum += BB[t][i];
        }
        for (int i=0;i<n;i++)
        {
            assert(fabs(exp_B[t][i] - BB[t][i] / sum) < EPS);
            printf("%lf ", BB[t][i] / sum);
        }
        printf("\n");
    }
    
    delete x;
    for (int i=0;i<n;i++)
    {
        delete[] T[i];
        delete[] O[i];
    }
    delete[] pi;
    delete[] T;
    delete[] O;
    observations.clear();
    
    n = 2;
    m = 3;
    
    pi = new double[n];
    pi[0] = 0.6; pi[1] = 0.4;
    
    T = new double*[n];
    for (int i=0;i<n;i++)
    {
        T[i] = new double[n];
    }
    T[0][0] = 0.7; T[0][1] = 0.3;
    T[1][0] = 0.4; T[1][1] = 0.6;
    
    O = new double*[n];
    for (int i=0;i<n;i++)
    {
        O[i] = new double[m];
    }
    O[0][0] = 0.5; O[0][1] = 0.4; O[0][2] = 0.1;
    O[1][0] = 0.1; O[1][1] = 0.3; O[1][2] = 0.6;
    
    int exp_seq[3];
    exp_seq[0] = 0; exp_seq[1] = 0; exp_seq[2] = 1;
    
    x = new HMM(n, m, pi, T, O);
    
    observations.push_back(0);
    observations.push_back(1);
    observations.push_back(2);
    
    printf("Running Viterbi algorithm...\n");
    vector<int> seq = x -> viterbi(observations);
    
    printf("Most likely state sequence: ");
    for (uint i=0;i<seq.size();i++)
    {
        assert(seq[i] == exp_seq[i]);
        printf("%d ", seq[i]);
    }
    printf("\n");
    
    delete x;
    for (int i=0;i<n;i++)
    {
        delete[] T[i];
        delete[] O[i];
    }
    delete[] pi;
    delete[] T;
    delete[] O;
    observations.clear();
    
    n = 2;
    m = 3;
    
    pi = new double[n];
    for (int i=0;i<n;i++)
    {
        pi[i] = 0.5;
    }
    
    T = new double*[n];
    for (int i=0;i<n;i++)
    {
        T[i] = new double[n];
    }
    T[0][0] = 0.5; T[0][1] = 0.5;
    T[1][0] = 0.5; T[1][1] = 0.5;
    
    O = new double*[n];
    for (int i=0;i<n;i++)
    {
        O[i] = new double[m];
    }
    O[0][0] = 0.4; O[0][1] = 0.1; O[0][2] = 0.5;
    O[1][0] = 0.1; O[1][1] = 0.5; O[1][2] = 0.4;
    
    double exp_pi[2];
    exp_pi[0] = 1.0000; exp_pi[1] = 0.0000;
    
    double exp_T[2][2];
    exp_T[0][0] = 0.6906; exp_T[0][1] = 0.3091;
    exp_T[1][0] = 0.0934; exp_T[1][1] = 0.9066;
    
    double exp_O[2][3];
    exp_O[0][0] = 0.5807; exp_O[0][1] = 0.0010; exp_O[0][2] = 0.4183;
    exp_O[1][0] = 0.0000; exp_O[1][1] = 0.7621; exp_O[1][2] = 0.2379;
    
    x = new HMM(n, m, pi, T, O);
    
    observations.push_back(2);
    observations.push_back(0);
    observations.push_back(0);
    observations.push_back(2);
    observations.push_back(1);
    observations.push_back(2);
    observations.push_back(1);
    observations.push_back(1);
    observations.push_back(1);
    observations.push_back(2);
    observations.push_back(1);
    observations.push_back(1);
    observations.push_back(1);
    observations.push_back(1);
    observations.push_back(1);
    observations.push_back(2);
    observations.push_back(2);
    observations.push_back(0);
    observations.push_back(0);
    observations.push_back(1);
    
    printf("Running Baum-Welch algorithm...\n");
    x -> baumwelch(observations, 500, 1e-10);
    
    printf("Start-state probability vector:\n");
    pi = x -> get_pi();
    for (int i=0;i<n;i++)
    {
        printf("%lf ", pi[i]);
        assert(fabs(pi[i] - exp_pi[i]) < EPS);
    }
    printf("\n");
    
    printf("Transition probability matrix:\n");
    T = x -> get_T();
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<n;j++)
        {
            printf("%lf ", T[i][j]);
            assert(fabs(T[i][j] - exp_T[i][j]) < EPS);
        }
        printf("\n");
    }
    
    printf("Emission probability matrix:\n");
    O = x -> get_O();
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<m;j++)
        {
            printf("%lf ", O[i][j]);
            assert(fabs(O[i][j] - exp_O[i][j]) < EPS);
        }
        printf("\n");
    }
    
    return 0;
}
