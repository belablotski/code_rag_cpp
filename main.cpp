// main.cpp
// Simple C++ app to run inference with CodeLlama:7B using llama.cpp
// Requires llama.cpp built and a local GGUF model file (e.g., codellama-7b.Q4_K_M.gguf)

#include <iostream>
#include <string>
#include <vector>
#include "llama.h" // llama.cpp C API header

int main() {
    std::string model_path = "codellama-7b.Q4_K_M.gguf"; // Update path if needed
    std::cout << "Enter your prompt: ";
    std::string prompt;
    std::getline(std::cin, prompt);

    // Initialize llama.cpp context
    llama_model_params model_params = llama_model_default_params();
    llama_context_params ctx_params = llama_context_default_params();
    llama_model *model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }
    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create context." << std::endl;
        llama_model_free(model);
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (!vocab) {
        std::cerr << "Failed to get vocabulary from model." << std::endl;
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Tokenize prompt
    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.length()), tokens.data(), static_cast<int>(tokens.size()), true, true); // Use vocab
    if (n_tokens < 0) {
        std::cerr << "Tokenization failed." << std::endl;
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    std::cout << "Tokenized prompt into " << n_tokens << " tokens." << std::endl;
    std::cout << "Tokens: ";
    for (int i = 0; i < n_tokens; ++i) {
        std::cout << tokens[i];
        if (i != n_tokens - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    tokens.resize(n_tokens);

    // Evaluate prompt
    int n_threads = 1; // You can set this to the number of CPU threads you want to use
    // llama_eval API has changed: use llama_decode instead
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
    }
    batch.n_tokens = tokens.size();
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "Failed to evaluate prompt." << std::endl;
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Generate tokens
    int n_predict = 128; // Number of tokens to generate
    std::cout << "\\nModel output:\\n"; 

    int cur_pos = tokens.size(); 
    llama_token nl_token = llama_token_nl(vocab); // Use vocab

    for (int i = 0; i < n_predict; ++i) {
        
        float *logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

        std::vector<llama_token_data> candidates_data;
        candidates_data.reserve(llama_n_vocab(vocab)); // Use vocab
        for (llama_token token_id = 0; token_id < llama_n_vocab(vocab); ++token_id) { // Use vocab
            candidates_data.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates_data.data(), candidates_data.size(), false };

        // Sample the next token - using llama_sample_token as a more general sampler
        // llama_sample_token_greedy might not be available in all versions.
        // You might need to adjust sampling parameters (temp, top_k, top_p) for desired output.
        llama_token new_token_id = llama_sample_token(ctx, &candidates_p, nullptr, 0, 0.8f, 40, 0.95f, 1.0f, 0, nullptr);

        if (new_token_id == llama_token_eos(vocab)) { // Use vocab
            break;
        }

        const char* token_text_cstr = llama_token_to_str(ctx, new_token_id); // llama_token_to_str is often an alias or replacement for get_text
        if (token_text_cstr) {
            std::cout << token_text_cstr;
        }
        std::cout << std::flush;
        
        if (new_token_id == nl_token) { 
            // std::cout << std::endl; 
        }

        batch.n_tokens = 1;
        batch.token[0]    = new_token_id;
        batch.pos[0]      = cur_pos;
        batch.logits[0]   = false; 

        cur_pos++; 

        if (llama_decode(ctx, batch) != 0) { 
            std::cerr << "Failed to decode next token." << std::endl;
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }
    std::cout << std::endl;

    llama_batch_free(batch); 
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
