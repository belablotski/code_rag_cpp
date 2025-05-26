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
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
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

    // // Evaluate prompt
    // if (llama_eval(ctx, tokens.data(), tokens.size(), 0, 1)) {
    //     std::cerr << "Failed to evaluate prompt." << std::endl;
    //     llama_free(ctx);
    //     llama_model_free(model);
    //     return 1;
    // }

    // // Generate tokens
    // int n_predict = 128; // Number of tokens to generate
    // std::cout << "\nModel output:\n";
    // for (int i = 0; i < n_predict; ++i) {
    //     llama_token token = llama_sample_token(ctx, nullptr);
    //     if (token == llama_token_eos()) break;
    //     char piece[8];
    //     llama_token_to_piece(ctx, token, piece, sizeof(piece));
    //     std::cout << piece << std::flush;
    //     llama_eval(ctx, &token, 1, tokens.size() + i, 1);
    // }
    // std::cout << std::endl;

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
