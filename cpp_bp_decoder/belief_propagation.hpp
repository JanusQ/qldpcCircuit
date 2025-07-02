#ifndef BELIEF_PROPAGATION_HPP
#define BELIEF_PROPAGATION_HPP

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace qldpc_bp {

// Forward declarations
class VariableNode;
class FactorNode;
class FactorGraph;

// Message class for binary variables (stores probabilities for {0, 1})
class Message {
public:
    double prob_0;  // P(x = 0)
    double prob_1;  // P(x = 1)
    
    Message(double p0 = 1.0, double p1 = 1.0) : prob_0(p0), prob_1(p1) {}
    
    // Normalize the message
    void normalize() {
        double sum = prob_0 + prob_1;
        if (sum > 1e-15) {
            prob_0 /= sum;
            prob_1 /= sum;
        } else {
            prob_0 = prob_1 = 0.5;
        }
    }
    
    // Get probability for a specific value
    double get_prob(int value) const {
        return (value == 0) ? prob_0 : prob_1;
    }
    
    // Set probability for a specific value
    void set_prob(int value, double prob) {
        if (value == 0) prob_0 = prob;
        else prob_1 = prob;
    }
    
    // Multiply with another message (element-wise)
    Message operator*(const Message& other) const {
        return Message(prob_0 * other.prob_0, prob_1 * other.prob_1);
    }
};

// Base class for nodes
class Node {
public:
    int id;
    std::string name;
    
    Node(int node_id, const std::string& node_name) : id(node_id), name(node_name) {}
    virtual ~Node() = default;
};

// Variable node (represents error variables e_j or logical syndrome variables s_L,b)
class VariableNode : public Node {
public:
    enum Type { ERROR_VAR, LOGICAL_SYNDROME_VAR };
    
    Type type;
    Message prior;  // Prior probability P(variable)
    std::unordered_map<int, Message> incoming_messages;  // Messages from factor nodes
    std::unordered_map<int, Message> outgoing_messages;  // Messages to factor nodes
    std::unordered_set<int> neighbors;  // Connected factor nodes
    
    VariableNode(int node_id, const std::string& node_name, Type var_type)
        : Node(node_id, node_name), type(var_type), prior(1.0, 1.0) {}
    
    // Set prior probability for error variables
    void set_error_prior(double error_prob) {
        if (type == ERROR_VAR) {
            prior.prob_0 = 1.0 - error_prob;
            prior.prob_1 = error_prob;
        }
    }
    
    // Compute message to factor node
    Message compute_message_to_factor(int factor_id);
    
    // Compute belief (marginal probability)
    Message compute_belief();
    
    // Add neighbor factor node
    void add_neighbor(int factor_id) {
        neighbors.insert(factor_id);
    }
};

// Base class for factor nodes
class FactorNode : public Node {
public:
    std::unordered_map<int, Message> incoming_messages;  // Messages from variable nodes
    std::unordered_map<int, Message> outgoing_messages;  // Messages to variable nodes
    std::unordered_set<int> neighbors;  // Connected variable nodes
    
    FactorNode(int node_id, const std::string& node_name)
        : Node(node_id, node_name) {}
    
    // Pure virtual function to compute message to variable node
    virtual Message compute_message_to_variable(int var_id) = 0;
    
    // Add neighbor variable node
    void add_neighbor(int var_id) {
        neighbors.insert(var_id);
    }
};

// Prior factor node (for error variables)
class PriorFactorNode : public FactorNode {
public:
    int connected_var_id;
    Message prior_prob;
    
    PriorFactorNode(int node_id, int var_id, const Message& prior)
        : FactorNode(node_id, "prior_" + std::to_string(var_id))
        , connected_var_id(var_id), prior_prob(prior) {
        add_neighbor(var_id);
    }
    
    Message compute_message_to_variable(int var_id) override {
        if (var_id == connected_var_id) {
            return prior_prob;
        }
        return Message(1.0, 1.0);
    }
};

// Syndrome constraint factor node (enforces D * e = s')
class SyndromeFactorNode : public FactorNode {
public:
    int syndrome_value;  // s'_a (0 or 1)
    std::vector<int> connected_vars;  // Variables in N(a)
    
    SyndromeFactorNode(int node_id, int syndrome_val, const std::vector<int>& vars)
        : FactorNode(node_id, "syndrome_" + std::to_string(node_id))
        , syndrome_value(syndrome_val), connected_vars(vars) {
        for (int var_id : vars) {
            add_neighbor(var_id);
        }
    }
    
    Message compute_message_to_variable(int var_id) override;
};

// Logical constraint factor node (enforces s_L,b = XOR of connected error variables)
class LogicalFactorNode : public FactorNode {
public:
    int logical_syndrome_var;  // s_L,b variable
    std::vector<int> connected_error_vars;  // Error variables in N_L(b)
    
    LogicalFactorNode(int node_id, int logical_var, const std::vector<int>& error_vars)
        : FactorNode(node_id, "logical_" + std::to_string(node_id))
        , logical_syndrome_var(logical_var), connected_error_vars(error_vars) {
        add_neighbor(logical_var);
        for (int var_id : error_vars) {
            add_neighbor(var_id);
        }
    }
    
    Message compute_message_to_variable(int var_id) override;
};

// Factor graph class
class FactorGraph {
public:
    std::unordered_map<int, std::unique_ptr<VariableNode>> variable_nodes;
    std::unordered_map<int, std::unique_ptr<FactorNode>> factor_nodes;
    
    // Add variable node
    VariableNode* add_variable_node(int node_id, const std::string& name, VariableNode::Type type) {
        auto var_node = std::make_unique<VariableNode>(node_id, name, type);
        VariableNode* ptr = var_node.get();
        variable_nodes[node_id] = std::move(var_node);
        return ptr;
    }
    
    // Add factor node
    template<typename FactorType, typename... Args>
    FactorType* add_factor_node(int node_id, Args&&... args) {
        auto factor_node = std::make_unique<FactorType>(node_id, std::forward<Args>(args)...);
        FactorType* ptr = factor_node.get();
        factor_nodes[node_id] = std::move(factor_node);
        return ptr;
    }
    
    // Get variable node
    VariableNode* get_variable_node(int node_id) {
        auto it = variable_nodes.find(node_id);
        return (it != variable_nodes.end()) ? it->second.get() : nullptr;
    }
    
    // Get factor node
    FactorNode* get_factor_node(int node_id) {
        auto it = factor_nodes.find(node_id);
        return (it != factor_nodes.end()) ? it->second.get() : nullptr;
    }
};

// Belief Propagation decoder class
class BeliefPropagationDecoder {
private:
    FactorGraph graph;
    std::vector<std::vector<int>> D_matrix;    // Syndrome constraint matrix D
    std::vector<std::vector<int>> D_L_matrix;  // Logical constraint matrix D_L
    std::vector<double> error_probs;           // Error probabilities p_j
    std::vector<int> observed_syndrome;        // Observed syndrome s'
    
    int num_error_vars;
    int num_logical_vars;
    int num_syndrome_constraints;
    int num_logical_constraints;
    
    // Node ID management
    int next_var_id;
    int next_factor_id;
    
public:
    BeliefPropagationDecoder(const std::vector<std::vector<int>>& D,
                           const std::vector<std::vector<int>>& D_L,
                           const std::vector<double>& error_probabilities);
    
    // Construct the factor graph
    void construct_factor_graph();
    
    // Set observed syndrome
    void set_observed_syndrome(const std::vector<int>& syndrome);
    
    // Run belief propagation algorithm
    std::vector<int> decode(int max_iterations = 50, double convergence_threshold = 1e-6);
    
    // Perform one iteration of message passing
    bool update_messages();
    
    // Check convergence
    bool check_convergence(double threshold);
    
    // Compute marginal probabilities for logical syndrome variables
    std::vector<Message> compute_logical_marginals();
    
    // Find most likely logical syndrome
    std::vector<int> find_most_likely_logical_syndrome();
    
private:
    // Helper functions
    void initialize_messages();
    std::vector<Message> previous_messages;  // For convergence checking
};

} // namespace qldpc_bp

#endif // BELIEF_PROPAGATION_HPP 