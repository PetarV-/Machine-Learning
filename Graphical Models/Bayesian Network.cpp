/*
 Petar 'PetarV' Velickovic
 Data Structure: Bayesian Network
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
#include <unordered_set>
#include <unordered_map>
#include <sstream>

using namespace std;
typedef long long lld;

/*
 A Bayesian Network is a directed acyclic graph in which nodes correspond to RVs,
 and edges model direct causality; namely, it is assumed that a node's value is
 conditionally independent of its nondescendants given its parents. Therefore the
 full joint distribution can be expressed as
 
     P(N1, N2, ..., Nn) = prod_{i = 1}^{n} P(Ni | parents(Ni))
 
 This has the benefit of improved storage and time complexities compared to naïvely
 computing the full joint distribution.
 
 The BN implemented here supports only discrete RVs, however extending the nodes to 
 handle continuous RVs is possible; one just has to choose a known distribution that
 can be easily parametrised, that is "close enough" to the desired RV.
 
 The variable elimination algorithm for computing P(Q|e), where Q is a set of query RVs
 and e is a set of observed variables, has also been implemented. There are a few obvious
 inefficiencies that will have to be fixed in the future, but otherwise the algorithm
 seems to be working as expected.
 
 The complexity of the inference algorithm is linear for "well-behaved" BN architectures,
 and in the worst case it is #P-Hard, when one usually has to resort to approximate inference.
*/

struct table
{
    vector<int> inputs; // IDs of the inputs
    vector<int> cards; // inputs' cardinalities
    vector<double> tbl; // The actual table
    
    // Encodes a given set of parameters into an index
    int encode(vector<int> &vals, vector<int> &cards)
    {
        int ret = 0;
        int mul = 1;
        for (int i=0;i<vals.size();i++)
        {
            assert(vals[i] >= 0 && vals[i] < cards[i]);
            ret += vals[i] * mul;
            mul *= cards[i];
        }
        return ret;
    }
    
    // Decodes a given index into a set of parameters
    vector<int> decode(int ind, vector<int> &cards)
    {
        assert(ind < tbl.size());
        vector<int> ret(inputs.size());
        for (int i=0;i<inputs.size();i++)
        {
            ret[i] = ind % cards[i];
            ind /= cards[i];
        }
        return ret;
    }
    
    void set_value(vector<int> &vals, double value)
    {
        tbl[encode(vals, cards)] = value;
    }
    
    void set(int in, int val) // set a particular input (ignoring other rows)
    {
        int ii = 0;
        while (inputs[ii] != in)
        {
            assert(ii < inputs.size() - 1);
            ii++;
        }
        vector<double> next_tbl(tbl.size() / cards[ii], 0);
        
        vector<int> next_inputs = inputs;
        vector<int> next_cards = cards;
        next_inputs.erase(next_inputs.begin() + ii);
        next_cards.erase(next_cards.begin() + ii);
        
        for (int i=0;i<tbl.size();i++)
        {
            vector<int> vals = decode(i, cards);
            if (vals[ii] != val) continue;
            vals.erase(vals.begin() + ii);
            next_tbl[encode(vals, next_cards)] += tbl[i];
        }
        
        inputs = next_inputs;
        cards = next_cards;
        tbl = next_tbl;
    }
    
    void sum_out(int in) // sum out a particular input
    {
        int ii = 0;
        while (inputs[ii] != in)
        {
            assert(ii < inputs.size() - 1);
            ii++;
        }
        vector<double> next_tbl(tbl.size() / cards[ii], 0);
        
        vector<int> next_inputs = inputs;
        vector<int> next_cards = cards;
        next_inputs.erase(next_inputs.begin() + ii);
        next_cards.erase(next_cards.begin() + ii);
        
        for (int i=0;i<tbl.size();i++)
        {
            vector<int> vals = decode(i, cards);
            vals.erase(vals.begin() + ii);
            next_tbl[encode(vals, next_cards)] += tbl[i];
        }
        
        inputs = next_inputs;
        cards = next_cards;
        tbl = next_tbl;
    }
    
    void multiply(table t) // multiply with another table
    {
        unordered_set<int> inps; // keep track of which inputs were already taken into account
        vector<int> inp_cards; // keep track of their cardinalities too
        unordered_set<int> common_inps; // keep track of common inputs
        for (int i=0;i<inputs.size();i++)
        {
            inps.insert(inputs[i]);
            inp_cards.push_back(cards[i]);
        }
        for (int i=0;i<t.inputs.size();i++)
        {
            if (inps.count(t.inputs[i])) common_inps.insert(t.inputs[i]);
            else
            {
                inps.insert(t.inputs[i]);
                inp_cards.push_back(t.cards[i]);
            }
        }
        
        int next_size = 1;
        for (int in : inp_cards)
        {
            next_size *= in;
        }
        vector<double> next_tbl(next_size);
        
        bool done_once = false;
        vector<int> next_inputs;
        vector<int> next_cards;
        
        // Disclaimer: This is very inefficient and will check all pairs of fields, even though
        // many of them may be in fact incompatible. I will optimise it later...
        for (int i=0;i<tbl.size();i++)
        {
            vector<int> i_vals = decode(i, cards);
            for (int j=0;j<t.tbl.size();j++)
            {
                vector<int> j_vals = t.decode(j, t.cards);
                if (!done_once)
                {
                    next_inputs = t.inputs;
                    next_cards = t.cards;
                }
                
                // check if the rows and columns match
                bool match = true;
                for (int a=0;a<inputs.size();a++)
                {
                    if (!common_inps.count(inputs[a])) continue;
                    for (int b=0;b<t.inputs.size();b++)
                    {
                        if (inputs[a] != t.inputs[b]) continue;
                        if (i_vals[a] != j_vals[b])
                        {
                            match = false;
                            break;
                        }
                    }
                    if (!match) break;
                }
                if (!match) continue;
                
                for (int a=0;a<inputs.size();a++)
                {
                    if (common_inps.count(inputs[a])) continue;
                    j_vals.push_back(i_vals[a]);
                    if (!done_once)
                    {
                        next_inputs.push_back(inputs[a]);
                        next_cards.push_back(cards[a]);
                    }
                }
                
                next_tbl[encode(j_vals, next_cards)] += tbl[i] * t.tbl[j];
                done_once = true;
            }
        }
        
        inputs = next_inputs;
        cards = next_cards;
        tbl = next_tbl;
    }
    
    void print()
    {
        printf("Table inputs (cardinalities): ");
        for (int i=0;i<inputs.size();i++)
        {
            printf("%d (%d)%s", inputs[i], cards[i], (i < inputs.size() - 1) ? ", " : "\n");
        }
        for (int i=0;i<tbl.size();i++)
        {
            vector<int> vals = decode(i, cards);
            for (int j=0;j<vals.size();j++)
            {
                printf("%d ", vals[j]);
            }
            printf("-> %.5lf\n", tbl[i]);
        }
    }
};

struct node
{
    vector<int> p; // indices of parents
    vector<int> chd; // indices of children
    vector<double> cd; // conditional distribution table
    int out_card; // cardinality of the output domain; 2 for Boolean
};

struct bayes_net
{
    vector<node> nodes;
    
    // Inserts a new node with the given parents, and output domain cardinality
    // Returns its index in the nodes vector.
    // nodes should be inserted in topological order!
    // (always exists because Bayesian networks are DAGs)
    int insert(vector<int> parents, int card)
    {
        int ret = nodes.size();
        
        node x;
        x.p = parents;
        int cd_size = card;
        for (int i=0;i<parents.size();i++)
        {
            cd_size *= nodes[parents[i]].out_card;
        }
        x.cd.resize(cd_size);
        x.out_card = card;
        nodes.push_back(x);
        
        return ret;
    }
    
    // Sets the value of the cond. distribution table for node ID
    // with given parents' output values and own output value.
    void set_value(int id, vector<int> &parent_vals, int own_val, double value)
    {
        assert(nodes[id].p.size() == parent_vals.size());
        int ind = 0;
        int mul = 1;
        for (int i=0;i<nodes[id].p.size();i++)
        {
            assert(parent_vals[i] >= 0 && parent_vals[i] < nodes[nodes[id].p[i]].out_card);
            ind += parent_vals[i] * mul;
            mul *= nodes[nodes[id].p[i]].out_card;
        }
        ind += own_val * mul;
        nodes[id].cd[ind] = value;
    }
    
    // Derive the distribution of the query variables Q
    // assuming the evidence (known assignments) E
    // It uses the Variable Elimination optimisation to the Enumeration-Ask algorithm.
    // N.B. it is assumed that Q and E are disjoint! Otherwise, routine might break.
    unordered_map<int, table> query(unordered_set<int> Q, unordered_map<int, int> E)
    {
        vector<int> outdeg(nodes.size());
        vector<table> tables(nodes.size());
        
        for (int i=0;i<nodes.size();i++)
        {
            for (int j=0;j<nodes[i].p.size();j++)
            {
                outdeg[nodes[i].p[j]]++;
            }
        }
        
        queue<int> q;
        for (int i=0;i<outdeg.size();i++)
        {
            if (outdeg[i] == 0) q.push(i);
        }
        
        while (!q.empty())
        {
            int it = q.front();
            q.pop();
            
            if (E.count(it)) continue;
            
            // Apologies... also very inefficient!
            // I first construct the entire factor (cloning the table)
            // and only then take into account that some inputs may be observed.
            tables[it].inputs = nodes[it].p;
            tables[it].cards.resize(tables[it].inputs.size());
            for (int i=0;i<tables[it].inputs.size();i++)
            {
                tables[it].cards[i] = nodes[nodes[it].p[i]].out_card;
            }
            tables[it].inputs.push_back(it);
            tables[it].cards.push_back(nodes[it].out_card);
            tables[it].tbl = nodes[it].cd;
            
            for (int i=0;i<tables[it].inputs.size();i++)
            {
                if (E.count(tables[it].inputs[i]))
                {
                    tables[it].set(tables[it].inputs[i], E[tables[it].inputs[i]]);
                }
            }
            
            for (int i=0;i<nodes[it].chd.size();i++)
            {
                if (!E.count(nodes[it].chd[i]))
                {
                    tables[it].multiply(tables[nodes[it].chd[i]]);
                }
            }
            
            if (!Q.count(it)) tables[it].sum_out(it);
            
            printf("\nTable for %d is\n", it);
            tables[it].print();
            
            for (int i=0;i<nodes[it].p.size();i++)
            {
                if (--outdeg[nodes[it].p[i]] == 0)
                {
                    q.push(nodes[it].p[i]);
                }
            }
        }
        
        unordered_map<int, table> ret;
        for (int q : Q)
        {
            ret[q] = tables[q];
        }
        
        return ret;
    }
};


int main()
{
    // Define an example Bayesian Network
    bayes_net bn;
    
    vector<int> empty;
    
    int climber = bn.insert(empty, 2);
    bn.set_value(climber, empty, 1, 0.05);
    bn.set_value(climber, empty, 0, 0.95);
    
    int goose = bn.insert(empty, 2);
    bn.set_value(goose, empty, 1, 0.2);
    bn.set_value(goose, empty, 0, 0.8);
    
    
    vector<int> alarm_par;
    alarm_par.push_back(climber);
    alarm_par.push_back(goose);
    
    int alarm = bn.insert(alarm_par, 2);
    
    vector<int> alarm_vals(2);
    
    alarm_vals[0] = 1, alarm_vals[1] = 1;
    bn.set_value(alarm, alarm_vals, 1, 0.98);
    bn.set_value(alarm, alarm_vals, 0, 0.02);
    alarm_vals[0] = 1, alarm_vals[1] = 0;
    bn.set_value(alarm, alarm_vals, 1, 0.96);
    bn.set_value(alarm, alarm_vals, 0, 0.04);
    alarm_vals[0] = 0, alarm_vals[1] = 1;
    bn.set_value(alarm, alarm_vals, 1, 0.2);
    bn.set_value(alarm, alarm_vals, 0, 0.8);
    alarm_vals[0] = 0, alarm_vals[1] = 0;
    bn.set_value(alarm, alarm_vals, 1, 0.08);
    bn.set_value(alarm, alarm_vals, 0, 0.92);
    
    vector<int> lodge_par;
    lodge_par.push_back(alarm);
    
    int lodge1 = bn.insert(lodge_par, 2);
    int lodge2 = bn.insert(lodge_par, 2);
    
    vector<int> lodge_vals(1);
    
    lodge_vals[0] = 1;
    bn.set_value(lodge1, lodge_vals, 1, 0.3);
    bn.set_value(lodge1, lodge_vals, 0, 0.7);
    bn.set_value(lodge2, lodge_vals, 1, 0.6);
    bn.set_value(lodge2, lodge_vals, 0, 0.4);
    lodge_vals[0] = 0;
    bn.set_value(lodge1, lodge_vals, 1, 0.001);
    bn.set_value(lodge1, lodge_vals, 0, 0.999);
    bn.set_value(lodge2, lodge_vals, 1, 0.001);
    bn.set_value(lodge2, lodge_vals, 0, 0.999);
    
    bn.nodes[climber].chd.push_back(alarm);
    bn.nodes[goose].chd.push_back(alarm);
    bn.nodes[alarm].chd.push_back(lodge1);
    bn.nodes[alarm].chd.push_back(lodge2);
    
    // Now run the query: P(lodge1 | climber = true)
    unordered_set<int> Q; Q.insert(lodge1);
    unordered_map<int, int> E; E[climber] = 1;
    
    printf("ID(climber) = %d\n", climber);
    printf("ID(goose) = %d\n", goose);
    printf("ID(alarm) = %d\n", alarm);
    printf("ID(lodge1) = %d\n", lodge1);
    printf("ID(lodge2) = %d\n", lodge2);
    
    bn.query(Q, E);
    
    return 0;
}
