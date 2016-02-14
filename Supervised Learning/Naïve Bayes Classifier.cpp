/*
 Petar 'PetarV' Velickovic
 Algorithm: Naïve Bayes Classifier
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
#include <sstream>

using namespace std;
typedef long long lld;

/*
 The Naïve Bayes Classifier is one of the simplest classifiers to construct. It makes an
 assumption about feature independence that is usually far too strong, but has still been
 shown to be an excellent tool in text classification, for instance.
 
 The learning model makes advantage of Bayes' theorem, with assumptions of full conditional
 independence of all individual features of the observed random variables. Namely:
 
 P(c | e) = P(c) * prod_{i = 1}^{n} P(e_i | c) / P(e)
 
 where c is a class, and e is a vector of observed features (evidence). This then presents us
 with a simple maximum a posteriori (MAP) classifying rule:
 
 C = argmax_{C'} P(C') * prod_{i = 1}^{n} P(e_i | C')
 
 Here the classifier is trained to assign categories to a given text set. It is designed
 to be trainable online, that is, new types of texts/categories may be added on-the-fly.
 A bag-of-words model is used to estimate the (frequentist) probabilities of individual features.
 Unseen input is resolved via Laplace smoothing, i.e. unseen words are counted as seen exactly
 once in the training data.
*/

// A structure representing the feature probabilities of a single class
struct cls_prob
{
    int cnt; // how many times have we encountered this class? (used for priors)
    int total_words; // how many words have appeared for this class?
    unordered_map<string, int> words; // how many times has each word appeared for this class?
};
unordered_map<string, cls_prob> classes;
int total_trainings;

inline void train(string text, string cls)
{
    assert(cls != "");
    total_trainings++;
    
    cls_prob c;
    
    // extract the class struct, if one exists
    if (classes.count(cls))
    {
        c = classes[cls];
    }
    else
    {
        c.cnt = 0;
        c.words.clear();
    }
    
    c.cnt++;
    
    // tokenize the input text and update frequencies
    stringstream ss(text);
    string token;
    while (ss >> token)
    {
        c.words[token]++;
        c.total_words++;
    }
    
    classes[cls] = c;
}

inline string classify(string text)
{
    unordered_map<string, int> word_counts;
    
    stringstream ss(text);
    string token;
    while (ss >> token)
    {
        word_counts[token]++;
    }
    
    double best = 0.0;
    string best_class = "";
    for (auto kv : classes)
    {
        string curr_class = kv.first;
        cls_prob c = kv.second;
        
        int lapl_total_words = c.total_words;
        for (auto toks : word_counts)
        {
            if (!c.words.count(toks.first))
            {
                lapl_total_words++; // Laplace smoothing: add an extra feature for each unseen word!
            }
        }
        
        // Prior...
        double curr_val = log(c.cnt) - log(total_trainings);
        
        for (auto toks : word_counts)
        {
            string word = toks.first;
            int cnt = toks.second;
            // ...times the individual word likelihoods...
            if (c.words.count(word))
            {
                curr_val += cnt * (log(c.words[word]) - log(lapl_total_words));
            }
            else
            {
                curr_val += cnt * (-log(lapl_total_words));
            }
        }
        
        // ...--> posterior!
        
        if (best_class == "" || (curr_val > best))
        {
            best = curr_val;
            best_class = curr_class;
        }
    }
    
    return best_class;
}

int main()
{
    // Example shamelessly stolen from muatik/naive-bayes-classifier
    train("not to eat too much is not enough to lose weight", "health");
    train("russia is trying to invade ukraine", "politics");
    train("do not neglect exercise", "health");
    train("syria is the main issue, obama says", "politics");
    train("eat to lose weight", "health");
    train("you should not eat much", "health");
    
    string cls = classify("even if i eat too much, is it not possible to lose some weight");
    
    printf("The class assigned is: %s\n", cls.c_str());
    
    return 0;
}
