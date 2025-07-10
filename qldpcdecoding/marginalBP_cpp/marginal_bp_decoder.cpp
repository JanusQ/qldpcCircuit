#include "marginal_bp_decoder.hpp"
#include <iostream>
#include <limits>

QLDPC_BP_Marginals::QLDPC_BP_Marginals(const std::vector<std::vector<int>>& D_prime_matrix,
                                       const std::vector<std::vector<int>>& D_L_matrix,
                                       const std::vector<int>& s_prime_vec,
                                       const std::vector<double>& weights_vec)
    : D_prime(D_prime_matrix.size(), D_prime_matrix[0].size()),
      D_L(D_L_matrix.size(), D_prime_matrix[0].size()),
      s_prime(s_prime_vec),
      weights(weights_vec),
      n_vars(D_prime_matrix[0].size()),
      n_syndromes(D_prime_matrix.size()),
      n_logical(D_L_matrix.size()) {
    
    // Build sparse matrices
    for (size_t i = 0; i < D_prime_matrix.size(); i++) {
        for (size_t j = 0; j < D_prime_matrix[i].size(); j++) {
            if (D_prime_matrix[i][j] == 1) {
                D_prime.add_entry(i, j);
            }
        }
    }
    
    for (size_t i = 0; i < D_L_matrix.size(); i++) {
        for (size_t j = 0; j < D_L_matrix[i].size(); j++) {
            if (D_L_matrix[i][j] == 1) {
                D_L.add_entry(i, j);
            }
        }
    }
    
    // Build factor graph connectivity
    build_factor_graph();
    
    // Initialize messages
    initialize_messages();
}

void QLDPC_BP_Marginals::build_factor_graph() {
    // Variable to syndrome check connections
    var_to_syn_checks.resize(n_vars);
    syn_check_to_vars.resize(n_syndromes);
    
    for (int i = 0; i < n_syndromes; i++) {
        syn_check_to_vars[i] = D_prime.row_indices[i];
        for (int j : D_prime.row_indices[i]) {
            var_to_syn_checks[j].push_back(i);
        }
    }
    
    // Variable to logical check connections
    var_to_log_checks.resize(n_vars);
    log_check_to_vars.resize(n_logical);
    
    for (int i = 0; i < n_logical; i++) {
        log_check_to_vars[i] = D_L.row_indices[i];
        for (int j : D_L.row_indices[i]) {
            var_to_log_checks[j].push_back(i);
        }
    }
}

void QLDPC_BP_Marginals::initialize_messages() {
    // Initialize message storage
    msg_var_to_syn.resize(n_vars);
    msg_syn_to_var.resize(n_syndromes);
    msg_sL_to_log.resize(n_logical);
    msg_log_to_sL.resize(n_logical);
    msg_var_to_log.resize(n_vars);
    msg_log_to_var.resize(n_logical);
    
    // Initialize all messages to uniform distribution
    for (int j = 0; j < n_vars; j++) {
        // Prior belief for variable j
        Message prior(1.0, std::exp(-weights[j]));
        prior.normalize();
        
        // Initialize messages to syndrome checks
        for (int check_idx : var_to_syn_checks[j]) {
            (void)check_idx;  // Suppress unused variable warning
            msg_var_to_syn[j].push_back(prior);
        }
        
        // Initialize messages to logical checks
        for (int check_idx : var_to_log_checks[j]) {
            (void)check_idx;  // Suppress unused variable warning
            msg_var_to_log[j].push_back(prior);
        }
    }
    
    // Initialize syndrome check messages
    for (int i = 0; i < n_syndromes; i++) {
        msg_syn_to_var[i].resize(n_vars, Message());
    }
    
    // Initialize logical check messages
    for (int i = 0; i < n_logical; i++) {
        msg_log_to_var[i].resize(n_vars, Message());
        msg_log_to_sL[i].resize(n_logical, Message());
        msg_sL_to_log[i].resize(n_logical, Message());
    }
    
    // Initialize logical syndrome variable messages (uniform prior)
    for (int i = 0; i < n_logical; i++) {
        msg_sL_to_log[i][i] = Message(0.5, 0.5);
    }
}

Message QLDPC_BP_Marginals::compute_parity_distribution(const std::vector<int>& var_indices,
                                                       const std::vector<std::vector<Message>>& messages) {
    Message parity_dist(1.0, 0.0);  // Start with even parity
    
    for (int var_idx : var_indices) {
        // Find the message for this variable
        Message var_msg(0.5, 0.5);  // Default uniform distribution
        
        // Get message from the container if available
        if (var_idx < static_cast<int>(messages.size()) && !messages[var_idx].empty()) {
            var_msg = messages[var_idx][0];  // Take first message (simplified)
        }
        
        // Convolution for XOR operation
        Message new_parity_dist;
        new_parity_dist.prob[0] = parity_dist.prob[0] * var_msg.prob[0] + parity_dist.prob[1] * var_msg.prob[1];
        new_parity_dist.prob[1] = parity_dist.prob[0] * var_msg.prob[1] + parity_dist.prob[1] * var_msg.prob[0];
        
        parity_dist = new_parity_dist;
    }
    
    parity_dist.normalize();
    return parity_dist;
}

void QLDPC_BP_Marginals::update_syndrome_check_messages() {
    for (int i = 0; i < n_syndromes; i++) {
        const auto& vars_in_check = syn_check_to_vars[i];
        int target_parity = s_prime[i];
        
        for (int j : vars_in_check) {
            // Find other variables in this check
            std::vector<int> other_vars;
            for (int v : vars_in_check) {
                if (v != j) {
                    other_vars.push_back(v);
                }
            }
            
            if (other_vars.empty()) {
                // Only one variable in check - direct constraint
                if (target_parity == 0) {
                    msg_syn_to_var[i][j] = Message(1.0, 0.0);
                } else {
                    msg_syn_to_var[i][j] = Message(0.0, 1.0);
                }
                continue;
            }
            
            // Compute probability that other variables have even/odd parity
            Message parity_dist(1.0, 0.0);  // Start with even parity
            
            for (int var_idx : other_vars) {
                // Get message from variable to this check
                Message var_prob(0.5, 0.5);  // Default
                
                // Find the actual message (this is a simplified version)
                // In practice, you'd need to maintain proper message indexing
                if (var_idx < static_cast<int>(msg_var_to_syn.size()) && !msg_var_to_syn[var_idx].empty()) {
                    var_prob = msg_var_to_syn[var_idx][0];  // Simplified indexing
                }
                
                // Convolution for XOR operation
                Message new_parity_dist;
                new_parity_dist.prob[0] = parity_dist.prob[0] * var_prob.prob[0] + parity_dist.prob[1] * var_prob.prob[1];
                new_parity_dist.prob[1] = parity_dist.prob[0] * var_prob.prob[1] + parity_dist.prob[1] * var_prob.prob[0];
                
                parity_dist = new_parity_dist;
            }
            
            // Message constrains variable j based on target parity
            if (target_parity == 0) {
                // Need even total parity
                msg_syn_to_var[i][j] = Message(parity_dist.prob[0], parity_dist.prob[1]);
            } else {
                // Need odd total parity
                msg_syn_to_var[i][j] = Message(parity_dist.prob[1], parity_dist.prob[0]);
            }
            
            msg_syn_to_var[i][j].normalize();
        }
    }
}

void QLDPC_BP_Marginals::update_logical_check_messages() {
    for (int i = 0; i < n_logical; i++) {
        const auto& vars_in_check = log_check_to_vars[i];
        
        // Compute parity distribution of all variables in the check
        Message parity_dist(1.0, 0.0);  // Start with even parity
        
        for (int var_idx : vars_in_check) {
            Message var_prob(0.5, 0.5);  // Default
            
            // Find the actual message (simplified)
            if (var_idx < static_cast<int>(msg_var_to_log.size()) && !msg_var_to_log[var_idx].empty()) {
                var_prob = msg_var_to_log[var_idx][0];  // Simplified indexing
            }
            
            // Convolution for XOR operation
            Message new_parity_dist;
            new_parity_dist.prob[0] = parity_dist.prob[0] * var_prob.prob[0] + parity_dist.prob[1] * var_prob.prob[1];
            new_parity_dist.prob[1] = parity_dist.prob[0] * var_prob.prob[1] + parity_dist.prob[1] * var_prob.prob[0];
            
            parity_dist = new_parity_dist;
        }
        
        // This is the marginal distribution for s_L_i
        parity_dist.normalize();
        msg_log_to_sL[i][i] = parity_dist;
        
        // Messages from logical check to variables
        for (int j : vars_in_check) {
            // Find other variables in this check
            std::vector<int> other_vars;
            for (int v : vars_in_check) {
                if (v != j) {
                    other_vars.push_back(v);
                }
            }
            
            if (other_vars.empty()) {
                // Only one variable in check - message depends on s_L_i belief
                msg_log_to_var[i][j] = msg_sL_to_log[i][i];
                continue;
            }
            
            // Get current belief about s_L_i
            Message s_L_belief = msg_sL_to_log[i][i];
            
            // Compute parity distribution of other variables
            Message other_parity_dist(1.0, 0.0);
            
            for (int var_idx : other_vars) {
                Message var_prob(0.5, 0.5);  // Default
                
                // Find the actual message (simplified)
                if (var_idx < static_cast<int>(msg_var_to_log.size()) && !msg_var_to_log[var_idx].empty()) {
                    var_prob = msg_var_to_log[var_idx][0];  // Simplified indexing
                }
                
                // Convolution for XOR operation
                Message new_parity_dist;
                new_parity_dist.prob[0] = other_parity_dist.prob[0] * var_prob.prob[0] + other_parity_dist.prob[1] * var_prob.prob[1];
                new_parity_dist.prob[1] = other_parity_dist.prob[0] * var_prob.prob[1] + other_parity_dist.prob[1] * var_prob.prob[0];
                
                other_parity_dist = new_parity_dist;
            }
            
            // Message to variable j
            Message msg(0.0, 0.0);
            
            // For each possible value of s_L_i
            for (int s_L_val = 0; s_L_val <= 1; s_L_val++) {
                double s_L_prob = s_L_belief.prob[s_L_val];
                
                // For each possible value of variable j
                for (int j_val = 0; j_val <= 1; j_val++) {
                    // Required parity of other variables
                    int required_other_parity = s_L_val ^ j_val;
                    
                    // Probability that other variables have required parity
                    double prob_other = other_parity_dist.prob[required_other_parity];
                    
                    msg.prob[j_val] += s_L_prob * prob_other;
                }
            }
            
            msg.normalize();
            msg_log_to_var[i][j] = msg;
        }
    }
}

void QLDPC_BP_Marginals::update_variable_messages() {
    for (int j = 0; j < n_vars; j++) {
        // Prior belief
        Message prior(1.0, std::exp(-weights[j]));
        prior.normalize();
        
        // Update messages to syndrome checks
        for (int i : var_to_syn_checks[j]) {
            Message belief = prior;
            
            // Multiply by messages from other syndrome checks (simplified)
            for (int other_i : var_to_syn_checks[j]) {
                if (other_i != i) {
                    // Simplified message multiplication
                    belief.prob[0] *= msg_syn_to_var[other_i][j].prob[0];
                    belief.prob[1] *= msg_syn_to_var[other_i][j].prob[1];
                }
            }
            
            // Multiply by messages from logical checks (simplified)
            for (int log_i : var_to_log_checks[j]) {
                belief.prob[0] *= msg_log_to_var[log_i][j].prob[0];
                belief.prob[1] *= msg_log_to_var[log_i][j].prob[1];
            }
            
            belief.normalize();
            
            // Update message (simplified indexing)
            if (!msg_var_to_syn[j].empty()) {
                msg_var_to_syn[j][0] = belief;
            }
        }
        
        // Update messages to logical checks
        for (int i : var_to_log_checks[j]) {
            Message belief = prior;
            
            // Multiply by messages from syndrome checks (simplified)
            for (int syn_i : var_to_syn_checks[j]) {
                belief.prob[0] *= msg_syn_to_var[syn_i][j].prob[0];
                belief.prob[1] *= msg_syn_to_var[syn_i][j].prob[1];
            }
            
            // Multiply by messages from other logical checks (simplified)
            for (int other_i : var_to_log_checks[j]) {
                if (other_i != i) {
                    belief.prob[0] *= msg_log_to_var[other_i][j].prob[0];
                    belief.prob[1] *= msg_log_to_var[other_i][j].prob[1];
                }
            }
            
            belief.normalize();
            
            // Update message (simplified indexing)
            if (!msg_var_to_log[j].empty()) {
                msg_var_to_log[j][0] = belief;
            }
        }
    }
}

void QLDPC_BP_Marginals::update_logical_syndrome_messages() {
    for (int i = 0; i < n_logical; i++) {
        // Uniform prior for logical syndrome bits
        msg_sL_to_log[i][i] = Message(0.5, 0.5);
    }
}

std::pair<bool, int> QLDPC_BP_Marginals::run_belief_propagation(int max_iterations, double tolerance) {
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        // Store old messages for convergence check (simplified)
        std::vector<std::vector<Message>> old_syn_messages = msg_syn_to_var;
        
        // BP update steps
        update_syndrome_check_messages();
        update_logical_check_messages();
        update_variable_messages();
        update_logical_syndrome_messages();
        
        // Check convergence (simplified)
        bool converged = true;
        for (int i = 0; i < n_syndromes && converged; i++) {
            for (int j = 0; j < n_vars; j++) {
                double diff = std::abs(msg_syn_to_var[i][j].prob[0] - old_syn_messages[i][j].prob[0]) +
                             std::abs(msg_syn_to_var[i][j].prob[1] - old_syn_messages[i][j].prob[1]);
                if (diff > tolerance) {
                    converged = false;
                    break;
                }
            }
        }
        
        if (converged) {
            return {true, iteration + 1};
        }
    }
    
    return {false, max_iterations};
}

std::vector<std::vector<double>> QLDPC_BP_Marginals::compute_logical_syndrome_marginals() {
    std::vector<std::vector<double>> marginals(n_logical, std::vector<double>(2));
    
    for (int i = 0; i < n_logical; i++) {
        // The marginal is computed from the logical check message
        marginals[i] = msg_log_to_sL[i][i].prob;
    }
    
    return marginals;
}

std::pair<std::vector<int>, std::vector<std::vector<double>>> QLDPC_BP_Marginals::find_most_likely_logical_syndrome() {
    auto marginals = compute_logical_syndrome_marginals();
    
    // Componentwise MAP estimation
    std::vector<int> most_likely_s_L(n_logical);
    for (int i = 0; i < n_logical; i++) {
        most_likely_s_L[i] = (marginals[i][1] > marginals[i][0]) ? 1 : 0;
    }
    
    return {most_likely_s_L, marginals};
} 