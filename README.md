# code_rag_cpp

## Build llama.cpp

1. https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#building-the-project
2. Build locally https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
3. Or download release zip from https://github.com/ggml-org/llama.cpp/releases

```
cmake -B build
cmake --build build --config Release
```

or

```
make clean
make
```

Update GCC build task:

```
"args": [
                "-g",
				"${file}",
				"-o",
			    "${fileDirname}/${fileBasenameNoExtension}",
				"-I",
				"/home/beloblotskiy/llama.cpp/include",
				"-I",
				"/home/beloblotskiy/llama.cpp/ggml/include",
				"-L",
				"/home/beloblotskiy/llama.cpp/build/bin",
				"-lllama",
				"-ldl",
				"-pthread"
```

## Download LLM

```
wget https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q4_K_M.gguf
```
