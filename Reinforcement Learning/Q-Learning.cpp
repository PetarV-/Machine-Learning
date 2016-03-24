/*
 Petar 'PetarV' Velickovic
 Algorithm: Q-Learning
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
#include <unordered_map>
#include <functional>
#include <random>

using namespace std;
typedef long long lld;

/*
 The Q-Learning algorithm is a reinforcement learning algorithm that iteratively learns
 a function Q(s, a), where s is a state and a an appropriate action. We assume a standard
 deterministic Markov Decision Process (MDP) environment, with a state-changing function S(s, a),
 and a reward function R(s, a).
 
 The Q-function represents the best (expected value of the) discounted cumulative reward
 attainable by taking action a in state s (thus assuming optimal play thereafter).
 
 After learning the Q-function, it is easy to construct an optimal policy:
 
    p'(s) = argmax_a Q(s, a)
 
 The Q-learning algorithm's iterative update formula can be easily derived as one of the
 three Bellman equations. We want to find the optimal policy with respect to the discounted
 cumulative reward
 
    V(p, s_t) = sum_{i=0}^{inf} gamma^i * r_{t + i}
 
 where r_t is the reward gained at time t as the policy p is followed, i.e.
 
    r_t         = R(s_t, p(s_t))
    s_{t + 1}   = S(s_t, p(s_t))
 
 and 0 <= gamma < 1 is a discount parameter. Therefore the optimal policy is defined as
 
    p'(s) = {argmax_p V(p, s)}(s)
 
 We can push state s into the formula and make this an argmax over all actions:
 
    p'(s) = argmax_a {R(s, a) + gamma * V(p', S(s, a))}
 
 The trick is now to define the whole expression being argmax-ed as Q(s, a):
 
    Q(s, a) = R(s, a) + gamma * V(p', S(s, a))
 
 but as we are once again taking the value of the optimal policy, it is
 exactly defined by the maximal entry of Q for the next state:
 
    Q(s, a) = R(s, a) + gamma * max_{a'} Q(S(s, a), a')
 
 Following this formula we derive the Q-learning algorithm:
 
    1. Set Q(s, a) = 0 for all states s and appropriate actions a;
 
    2. Starting from a particular state, s:
        (a) If state s is terminal (has no available actions), stop the algorithm.
        (b) Choose an action a to perform from s;
        (c) Obtain the next state: S(s, a), and the reward: R(s, a);
        (d) Update Q(s, a) by the formula above;
        (e) Set s = S(s, a), and go back to step a).
 
    3. Repeat step 2 until convergence is achieved.
 
 There are many ways in which an action can be chosen (step 2b), but the general rule of
 thumb is to prefer exploration early on and switch more and more to exploitation later.
 
 Exploration assumes that the actions are chosen more uniformly at random, while exploitation
 takes into account, to an extent, the values of Q when choosing an action.
*/

default_random_engine gen;

template<typename State, typename Action>
class QLearner
{
private:
    map<pair<State, Action>, double> Q; // State and Action should be hashable!
    function<vector<Action>(State)> A; // the actions available in a given state
    function<State(State, Action)> S; // the state-change function
    function<double(State, Action)> R; // the reward function
    double disc; // the discount factor
    
public:
    QLearner(function<vector<Action>(State)> A, function<State(State, Action)> S, function<double(State, Action)> R, double disc) : A(A), S(S), R(R), disc(disc)
    {
        Q.clear();
    }
    
    // Gets the required Q-function value.
    // N.B. in this case it is a simple lookup,
    // but it could well be e.g. a neural network query in general!
    double get_Q(State s, Action a)
    {
        return Q[make_pair(s, a)];
    }
    
    // Reestimate Q based on obtaining some reward and ending up
    // in a new state after performing an action.
    // N.B. in this case it is a direct assignment
    // but it could well be e.g. a neural network training call in general!
    void update_Q(State s, Action a, double reward, State nxt)
    {
        vector<Action> next_as = A(nxt);
        double max_val = next_as.empty() ? 0.0 : get_Q(nxt, next_as[0]);
        for (int i=1;i<next_as.size();i++)
        {
            max_val = max(max_val, get_Q(nxt, next_as[i]));
        }
        
        Q[make_pair(s, a)] = reward + disc * max_val;
    }
    
    // Deterministically follow the policy learnt so far, i.e. choose an action a that maximises Q(s, a).
    Action p(State s)
    {
        vector<Action> acts = A(s);
        assert(!acts.empty());
        
        Action ret = acts[0];
        double best = get_Q(s, acts[0]);
        
        for (Action a : acts)
        {
            double curr = get_Q(s, a);
            if (curr > best)
            {
                ret = a;
                best = curr;
            }
        }
        
        return ret;
    }
    
    // Choose an action to perform in state s
    // based on the exploitation/exploration tradeoff parameter l
    Action select(State s, double l)
    {
        vector<Action> acts = A(s);
        vector<double> probs(acts.size());
        
        for (int j=0;j<acts.size();j++)
        {
            probs[j] = pow(l, get_Q(s, acts[j]));
        }
        
        discrete_distribution<int> D(probs.begin(), probs.end());
        
        return acts[D(gen)];
    }
    
    // Apply the Q-learning algorithm, performing up to t steps, starting from state s0.
    // The l parameter specifies whether to prefer exploration (close to 1) or exploitation (larger)
    // If episode = true, the learner will first perform a series of steps, and then reestimate Q.
    void q_learn(State s0, int t, double l, bool episode)
    {
        stack<tuple<State, Action, double> > steps; // only need to fill this up if episodic.
        
        State s = s0;
        
        while (t--)
        {
            if (A(s).empty()) break; // terminal state
            
            Action a = select(s, l);
            
            State nxt = S(s, a);
            double rew = R(s, a);
            
            if (episode) steps.push(make_tuple(s, a, rew));
            else update_Q(s, a, rew, nxt);
            
            s = nxt;
        }
        
        if (episode)
        {
            State last = s;
            // reestimate backwards for faster propagation!
            while (!steps.empty())
            {
                auto st = steps.top();
                steps.pop();
                
                update_Q(get<0>(st), get<1>(st), get<2>(st), last);
                
                last = get<0>(st);
            }
        }
    }
    
};

int main()
{
    /* Instantiating a basic Q-Learner for a grid problem as such
     *
     * (s - start state, # - blocked, +,- - rewards
     *
     *     0   1   2   3
     *   -----------------
     * 0 |   |   |   | + |
     *   |---|---|---|---|
     * 1 |   | # |   | - |
     *   |---|---|---|---|
     * 2 | s |   |   |   |
     *   -----------------
     */
    
    enum move {L, R, U, D};
    
    // Can do all actions in all nonreward states (although some might be NoOps)
    auto A = [] (pair<int, int> st) -> vector<move>
    {
        if ((st.second != 3 || st.first == 2) && (st.first != 1 || st.second != 1)) return {L, R, U, D};
        return {};
    };
    
    auto Sta = [] (pair<int, int> st, move m) -> pair<int, int>
    {
        pair<int, int> ret = st;
        if (m == D && ret.first < 2) ret.first++;
        if (m == U && ret.first > 0) ret.first--;
        if (m == R && ret.second < 3) ret.second++;
        if (m == L && ret.second > 0) ret.second--;
        if (ret.first == 1 && ret.second == 1) return st;
        return ret;
    };
    
    auto Rew = [&Sta] (pair<int, int> st, move m) -> double
    {
        pair<int, int> S1 = Sta(st, m);
        if (S1.first == 0 && S1.second == 3) return 1.0;
        else if (S1.first == 1 && S1.second == 3) return -1.0;
        return -0.04; // small negative reward to encourage moving
    };
    
    double disc = 0.1;
    
    QLearner<pair<int, int>, move> q_grid(A, Sta, Rew, disc);
    
    pair<int, int> s0 = make_pair(2, 0);
    
    int num_episodes = 10000;
    
    double l = 1.0; // Start with l = 1 for maximal exploration
    
    while (num_episodes--)
    {
        q_grid.q_learn(s0, 10000, l, true);
        l += 0.1; // This might be a too drastic increase of l. Fine-tune as necessary.
        printf("%d\n", num_episodes);
    }
    
    for (int i=0;i<3;i++)
    {
        for (int j=0;j<4;j++)
        {
            for (int m=0;m<4;m++)
            {
                printf("Q((%d, %d), %d) = %.5lf\n", i, j, m, q_grid.get_Q(make_pair(i, j), (move)m));
            }
        }
    }
    
    printf("\nOptimal policy:\n");
    for (int i=0;i<3;i++)
    {
        for (int j=0;j<4;j++)
        {
            char m = '\0';
            
            if (A(make_pair(i, j)).empty()) m = '#';
            else
            {
                move best = q_grid.p(make_pair(i, j));
                if (best == L) m = 'L';
                if (best == R) m = 'R';
                if (best == U) m = 'U';
                if (best == D) m = 'D';
            }
            
            printf("%c", m);
        }
        printf("\n");
    }
    
    return 0;
}