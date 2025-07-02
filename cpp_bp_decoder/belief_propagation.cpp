#include "belief_propagation.hpp"
#include <cassert>
#include <limits>

namespace qldpc_bp {

// VariableNode implementation
Message VariableNode::compute_message_to_factor(int factor_id) {
    Message msg = prior;
    
    // Multiply with all incoming messages except from the target factor
    for (const auto& [neighbor_id, incoming_msg] : incoming_messages) {
        if (neighbor_id != factor_id) {
            msg = msg * incoming_msg;
        }
    }
    
    msg.normalize();
    outgoing_messages[factor_id] = msg;
    return msg;
}

Message VariableNode::compute_belief() {
    Message belief = prior;
    
    // Multiply with all incoming messages
    for (const auto& [neighbor_id, incoming_msg] : incoming_messages) {
        belief = belief * incoming_msg;
    }
    
    belief.normalize();
    return belief;
}

// SyndromeFactorNode implementation
Message SyndromeFactorNode::compute_message_to_variable(int var_id) {
    // Find the target variable in connected_vars
    auto it = std::find(connected_vars.begin(), connected_vars.end(), var_id);
    if (it == connected_vars.end()) {
        return Message(1.0, 1.0);  // Uniform if not connected
    }
    
    // Compute the message using the efficient formula for XOR constraints
    // Let q_j = P(e_j = 1) from incoming messages
    std::vector<double> q_values;
    
    for (int connected_var : connected_vars) {
        if (connected_var != var_id) {
            auto msg_it = incoming_messages.find(connected_var);
            if (msg_it != incoming_messages.end()) {
                const Message& msg = msg_it->second;
                double q = msg.prob_1 / (msg.prob_0 + msg.prob_1);
                q_values.push_back(q);
            } else {
                q_values.push_back(0.5);  // Default uniform
            }
        }
    }
    
    // Compute product of (1 - 2*q_j) terms
    double product = 1.0;
    for (double q : q_values) {
        product *= (1.0 - 2.0 * q);
    }
    
    // Apply the efficient formula for XOR constraint
    double msg_prob_0 = 0.5 * (1.0 + (1.0 - 2.0 * syndrome_value) * product);
    double msg_prob_1 = 0.5 * (1.0 - (1.0 - 2.0 * syndrome_value) * product);
    
    Message result(msg_prob_0, msg_prob_1);
    result.normalize();
    outgoing_messages[var_id] = result;
    return result;
}

// LogicalFactorNode implementation
Message LogicalFactorNode::compute_message_to_variable(int var_id) {
    if (var_id == logical_syndrome_var) {
        // Message to logical syndrome variable s_L,b
        // s_L,b = XOR of error variables, so we compute P(s_L,b) based on parity
        
        std::vector<double> q_values;
        for (int error_var : connected_error_vars) {
            auto msg_it = incoming_messages.find(error_var);
            if (msg_it != incoming_messages.end()) {
                const Message& msg = msg_it->second;
                double q = msg.prob_1 / (msg.prob_0 + msg.prob_1);
                q_values.push_back(q);
            } else {
                q_values.push_back(0.5);  // Default uniform
            }
        }
        
        // Compute probability that XOR equals 0 or 1
        double product = 1.0;
        for (double q : q_values) {
            product *= (1.0 - 2.0 * q);
        }
        
        double prob_xor_0 = 0.5 * (1.0 + product);
        double prob_xor_1 = 0.5 * (1.0 - product);
        
        Message result(prob_xor_0, prob_xor_1);
        result.normalize();
        outgoing_messages[var_id] = result;
        return result;
        
    } else {
        // Message to error variable e_k
        // Similar to syndrome constraint but includes logical syndrome variable
        
        auto it = std::find(connected_error_vars.begin(), connected_error_vars.end(), var_id);
        if (it == connected_error_vars.end()) {
            return Message(1.0, 1.0);  // Uniform if not connected
        }
        
        std::vector<double> q_values;
        
        // Include logical syndrome variable
        auto logical_msg_it = incoming_messages.find(logical_syndrome_var);
        double logical_q = 0.5;  // Default uniform
        if (logical_msg_it != incoming_messages.end()) {
            const Message& msg = logical_msg_it->second;
            logical_q = msg.prob_1 / (msg.prob_0 + msg.prob_1);
        }
        
        // Include other error variables (excluding target)
        for (int error_var : connected_error_vars) {
            if (error_var != var_id) {
                auto msg_it = incoming_messages.find(error_var);
                if (msg_it != incoming_messages.end()) {
                    const Message& msg = msg_it->second;
                    double q = msg.prob_1 / (msg.prob_0 + msg.prob_1);
                    q_values.push_back(q);
                } else {
                    q_values.push_back(0.5);  // Default uniform
                }
            }
        }
        
        // Compute product of (1 - 2*q_j) terms for error variables
        double product = 1.0;
        for (double q : q_values) {
            product *= (1.0 - 2.0 * q);
        }
        
        // The constraint is: s_L,b XOR (XOR of error vars) = 0
        // This means s_L,b = XOR of error vars
        // So we want P(e_k | s_L,b = XOR of other error vars)
        
        double msg_prob_0 = 0.5 * (1.0 + (1.0 - 2.0 * logical_q) * product);
        double msg_prob_1 = 0.5 * (1.0 - (1.0 - 2.0 * logical_q) * product);
        
        Message result(msg_prob_0, msg_prob_1);
        result.normalize();
        outgoing_messages[var_id] = result;
        return result;
    }
}

// BeliefPropagationDecoder implementation
BeliefPropagationDecoder::BeliefPropagationDecoder(
    const std::vector<std::vector<int>>& D,
    const std::vector<std::vector<int>>& D_L,
    const std::vector<double>& error_probabilities)
    : D_matrix(D), D_L_matrix(D_L), error_probs(error_probabilities)
    , next_var_id(0), next_factor_id(0) {
    
    // Determine dimensions
    num_syndrome_constraints = D_matrix.size();
    num_error_vars = D_matrix.empty() ? 0 : D_matrix[0].size();
    num_logical_constraints = D_L_matrix.size();
    num_logical_vars = D_L_matrix.size();  // Assuming one logical var per constraint
    
    assert(static_cast<int>(error_probabilities.size()) == num_error_vars);
    
    // Construct the factor graph
    construct_factor_graph();
}

void BeliefPropagationDecoder::construct_factor_graph() {
    // Create error variable nodes (e_1, e_2, ..., e_n)
    for (int i = 0; i < num_error_vars; ++i) {
        auto* var_node = graph.add_variable_node(next_var_id++, "e_" + std::to_string(i), VariableNode::ERROR_VAR);
        var_node->set_error_prior(error_probs[i]);
        
        // Create prior factor for this error variable
        Message prior_msg(1.0 - error_probs[i], error_probs[i]);
        auto* prior_factor = graph.add_factor_node<PriorFactorNode>(next_factor_id++, var_node->id, prior_msg);
        
        // Connect variable and prior factor
        var_node->add_neighbor(prior_factor->id);
    }
    
    // Create logical syndrome variable nodes (s_L,1, s_L,2, ..., s_L,m_L)
    std::vector<int> logical_var_ids;
    for (int i = 0; i < num_logical_vars; ++i) {
        auto* var_node = graph.add_variable_node(next_var_id++, "s_L_" + std::to_string(i), VariableNode::LOGICAL_SYNDROME_VAR);
        logical_var_ids.push_back(var_node->id);
    }
    
    // Create syndrome constraint factors (f_a for each row of D)
    for (int a = 0; a < num_syndrome_constraints; ++a) {
        std::vector<int> connected_vars;
        for (int j = 0; j < num_error_vars; ++j) {
            if (D_matrix[a][j] == 1) {
                connected_vars.push_back(j);  // Use error variable IDs (0 to n-1)
            }
        }
        
        if (!connected_vars.empty()) {
            auto* syndrome_factor = graph.add_factor_node<SyndromeFactorNode>(
                next_factor_id++, 0, connected_vars);  // syndrome_value will be set when decoding
            
            // Connect to error variables
            for (int var_id : connected_vars) {
                auto* var_node = graph.get_variable_node(var_id);
                if (var_node) {
                    var_node->add_neighbor(syndrome_factor->id);
                }
            }
        }
    }
    
    // Create logical constraint factors (f_L,b for each row of D_L)
    for (int b = 0; b < num_logical_constraints; ++b) {
        std::vector<int> connected_error_vars;
        for (int j = 0; j < num_error_vars; ++j) {
            if (D_L_matrix[b][j] == 1) {
                connected_error_vars.push_back(j);  // Use error variable IDs (0 to n-1)
            }
        }
        
        if (!connected_error_vars.empty()) {
            auto* logical_factor = graph.add_factor_node<LogicalFactorNode>(
                next_factor_id++, logical_var_ids[b], connected_error_vars);
            
            // Connect to logical syndrome variable
            auto* logical_var = graph.get_variable_node(logical_var_ids[b]);
            if (logical_var) {
                logical_var->add_neighbor(logical_factor->id);
            }
            
            // Connect to error variables
            for (int var_id : connected_error_vars) {
                auto* var_node = graph.get_variable_node(var_id);
                if (var_node) {
                    var_node->add_neighbor(logical_factor->id);
                }
            }
        }
    }
}

void BeliefPropagationDecoder::set_observed_syndrome(const std::vector<int>& syndrome) {
    observed_syndrome = syndrome;
    assert(static_cast<int>(observed_syndrome.size()) == num_syndrome_constraints);
    
    // Update syndrome values in syndrome constraint factors
    int factor_idx = 0;
    for (auto& [factor_id, factor_node] : graph.factor_nodes) {
        auto* syndrome_factor = dynamic_cast<SyndromeFactorNode*>(factor_node.get());
        if (syndrome_factor && factor_idx < static_cast<int>(observed_syndrome.size())) {
            syndrome_factor->syndrome_value = observed_syndrome[factor_idx++];
        }
    }
}

std::vector<int> BeliefPropagationDecoder::decode(int max_iterations, double convergence_threshold) {
    initialize_messages();
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        bool converged = update_messages();
        
        if (iter > 0 && (converged || check_convergence(convergence_threshold))) {
            std::cout << "BP converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
        
        if (iter == max_iterations - 1) {
            std::cout << "BP reached maximum iterations (" << max_iterations << ")." << std::endl;
        }
    }
    
    return find_most_likely_logical_syndrome();
}

void BeliefPropagationDecoder::initialize_messages() {
    // Initialize all messages to uniform (1.0, 1.0)
    for (auto& [var_id, var_node] : graph.variable_nodes) {
        for (int neighbor_id : var_node->neighbors) {
            var_node->incoming_messages[neighbor_id] = Message(1.0, 1.0);
            var_node->outgoing_messages[neighbor_id] = Message(1.0, 1.0);
        }
    }
    
    for (auto& [factor_id, factor_node] : graph.factor_nodes) {
        for (int neighbor_id : factor_node->neighbors) {
            factor_node->incoming_messages[neighbor_id] = Message(1.0, 1.0);
            factor_node->outgoing_messages[neighbor_id] = Message(1.0, 1.0);
        }
    }
}

bool BeliefPropagationDecoder::update_messages() {
    // Store previous messages for convergence checking
    previous_messages.clear();
    for (auto& [var_id, var_node] : graph.variable_nodes) {
        for (auto& [factor_id, msg] : var_node->outgoing_messages) {
            previous_messages.push_back(msg);
        }
    }
    
    // Update messages from variables to factors
    for (auto& [var_id, var_node] : graph.variable_nodes) {
        for (int factor_id : var_node->neighbors) {
            Message new_msg = var_node->compute_message_to_factor(factor_id);
            
            auto* factor_node = graph.get_factor_node(factor_id);
            if (factor_node) {
                factor_node->incoming_messages[var_id] = new_msg;
            }
        }
    }
    
    // Update messages from factors to variables
    for (auto& [factor_id, factor_node] : graph.factor_nodes) {
        for (int var_id : factor_node->neighbors) {
            Message new_msg = factor_node->compute_message_to_variable(var_id);
            
            auto* var_node = graph.get_variable_node(var_id);
            if (var_node) {
                var_node->incoming_messages[factor_id] = new_msg;
            }
        }
    }
    
    return false;  // Always return false for now, convergence checked separately
}

bool BeliefPropagationDecoder::check_convergence(double threshold) {
    if (previous_messages.empty()) return false;
    
    int msg_idx = 0;
    for (auto& [var_id, var_node] : graph.variable_nodes) {
        for (auto& [factor_id, current_msg] : var_node->outgoing_messages) {
            if (msg_idx >= static_cast<int>(previous_messages.size())) return false;
            
            const Message& prev_msg = previous_messages[msg_idx++];
            
            double diff_0 = std::abs(current_msg.prob_0 - prev_msg.prob_0);
            double diff_1 = std::abs(current_msg.prob_1 - prev_msg.prob_1);
            
            if (diff_0 > threshold || diff_1 > threshold) {
                return false;
            }
        }
    }
    
    return true;
}

std::vector<Message> BeliefPropagationDecoder::compute_logical_marginals() {
    std::vector<Message> marginals;
    
    // Find logical syndrome variables and compute their beliefs
    for (auto& [var_id, var_node] : graph.variable_nodes) {
        if (var_node->type == VariableNode::LOGICAL_SYNDROME_VAR) {
            Message belief = var_node->compute_belief();
            marginals.push_back(belief);
        }
    }
    
    return marginals;
}

std::vector<int> BeliefPropagationDecoder::find_most_likely_logical_syndrome() {
    std::vector<Message> marginals = compute_logical_marginals();
    
    if (num_logical_vars <= 20) {  // Feasible to enumerate all combinations
        double max_prob = -1.0;
        std::vector<int> best_syndrome(num_logical_vars, 0);
        
        // Enumerate all 2^m_L possible logical syndromes
        for (int syndrome_int = 0; syndrome_int < (1 << num_logical_vars); ++syndrome_int) {
            std::vector<int> current_syndrome(num_logical_vars);
            double joint_prob = 1.0;
            
            for (int i = 0; i < num_logical_vars; ++i) {
                current_syndrome[i] = (syndrome_int >> i) & 1;
                joint_prob *= marginals[i].get_prob(current_syndrome[i]);
            }
            
            if (joint_prob > max_prob) {
                max_prob = joint_prob;
                best_syndrome = current_syndrome;
            }
        }
        
        return best_syndrome;
    } else {
        // For large m_L, use independent approximation
        std::vector<int> best_syndrome(num_logical_vars);
        for (int i = 0; i < num_logical_vars; ++i) {
            best_syndrome[i] = (marginals[i].prob_1 > marginals[i].prob_0) ? 1 : 0;
        }
        return best_syndrome;
    }
}

} // namespace qldpc_bp 