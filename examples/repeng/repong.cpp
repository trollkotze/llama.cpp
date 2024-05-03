
#include "common.h"

#include "console.h"
#include "llama.h"
#include <deque> 
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static gpt_params               * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;

static bool file_exists(const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string &path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
      console::cleanup();
      printf("\n");
      llama_print_timings(*g_ctx);
      //write_logfile(*g_ctx, *g_params, *g_model, *g_input_tokens, g_output_ss->str(), *g_output_tokens);
      _exit(130);
    }
}
#endif

static void llama_log_callback_logTee(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    LOG_TEE("%s", text);
}

static std::tuple<struct llama_model *, struct llama_context *> llama_init_from_gpt_params_with_cb_eval(
    gpt_params & params,
    ggml_backend_sched_eval_callback cb_eval,
    void * cb_eval_user_data,
    struct llama_model *preloaded = NULL) {
    auto mparams = llama_model_params_from_gpt_params(params);

    llama_model * model  = preloaded ? preloaded : llama_load_model_from_file(params.model.c_str(), mparams);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return std::make_tuple(nullptr, nullptr);
    }

    auto cparams = llama_context_params_from_gpt_params(params);

    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = cb_eval_user_data;

    llama_context * lctx = llama_new_context_with_model(model, cparams);
    if (lctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
        llama_free_model(model);
        return std::make_tuple(nullptr, nullptr);
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_n_layer(model);

        const auto cvec = llama_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            llama_free(lctx);
            llama_free_model(model);
            return std::make_tuple(nullptr, nullptr);
        }

        int err = llama_control_vector_apply(lctx,
                                             cvec.data.data(),
                                             cvec.data.size(),
                                             cvec.n_embd,
                                             params.control_vector_layer_start,
                                             params.control_vector_layer_end);
        if (err) {
            llama_free(lctx);
            llama_free_model(model);
            return std::make_tuple(nullptr, nullptr);
        }
    }

    for (unsigned int i = 0; i < params.lora_adapter.size(); ++i) {
        const std::string& lora_adapter = std::get<0>(params.lora_adapter[i]);
        float lora_scale = std::get<1>(params.lora_adapter[i]);
        int err = llama_model_apply_lora_from_file(model,
                                             lora_adapter.c_str(),
                                             lora_scale,
                                             ((i > 0) || params.lora_base.empty())
                                                ? NULL
                                                : params.lora_base.c_str(),
                                             params.n_threads);
        if (err != 0) {
            fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
            llama_free(lctx);
            llama_free_model(model);
            return std::make_tuple(nullptr, nullptr);
        }
    }

    if (params.ignore_eos) {
        params.sparams.logit_bias[llama_token_eos(model)] = -INFINITY;
    }

    {
        LOG("warming up the model with an empty run\n");

        std::vector<llama_token> tmp = { llama_token_bos(model), llama_token_eos(model), };
        llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch), 0, 0));
        llama_kv_cache_clear(lctx);
        llama_synchronize(lctx);
        llama_reset_timings(lctx);
    }

    return std::make_tuple(model, lctx);
}

struct eval_callback_state {
    std::vector<ggml_tensor *> tensors;
    // For each hidden state tensor, how many tokens have we seen from the
    // current batch?  When n_ubatch < n_batch, we don't see the hidden states
    // for all tokens at once, but only n_ubatch-sized chunks of tokens.  This
    // keeps track of our progress.
    std::vector<int> tokens_seen;
    int first_prompt_idx;
    std::vector<int> extract_tokens;
};

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    struct eval_callback_state * eval_state = (eval_callback_state *)user_data;
    if (ask) {
        // Report whether we want to observe this tensor.
        if (strncmp(t->name, "l_out-", 6) == 0) {
            return true;
        } else {
            return false;
        }
    } else {
        // Actually observe the tensor data.

        if (eval_state->first_prompt_idx >= 0) {
            // Find the tensor collecting hidden states for the current layer.
            int layer_idx = -1;
            ggml_tensor * output_tensor = nullptr;
            for (size_t i = 0; i < eval_state->tensors.size(); ++i) {
                auto t2 = eval_state->tensors[i];
                if (strcmp(t2->name, t->name) == 0) {
                    output_tensor = t2;
                    layer_idx = i;
                    break;
                }
            }

            if (output_tensor != nullptr) {
                int ubatch_size = t->ne[1];
                int ubatch_start = eval_state->tokens_seen[layer_idx];
                int ubatch_end = ubatch_start + ubatch_size;
                eval_state->tokens_seen[layer_idx] += ubatch_size;

                int output_idx = eval_state->first_prompt_idx;
                for (int token_idx : eval_state->extract_tokens) {
                    if (token_idx < ubatch_start || token_idx >= ubatch_end) {
                        continue;
                    }
                    int input_idx = token_idx - ubatch_start;

                    // Copy the hidden states for the current token into the
                    // output buffer.
                    size_t input_offset = t->nb[1] * input_idx;
                    size_t output_offset = output_tensor->nb[1] * output_idx;
                    assert(t->nb[0] == output_tensor->nb[0]);
                    assert(t->ne[0] == output_tensor->ne[0]);
                    ggml_backend_tensor_get(t,
                            (char *)output_tensor->data + output_offset,
                            input_offset,
                            t->nb[0] * t->ne[0]);
                    //memcpy((char *)output_tensor->data + output_offset,
                    //        (char *)t->data + input_offset,
                    //        t->nb[0] * t->ne[0]);
                    //std::cerr << "saved " << (t->nb[0] * t->ne[0]) << " bytes of tensor data "
                    //    << " for " << t->name << " in slot " << output_idx << "\n";

                    //float * buf = (float *)((char *)t->data + input_offset);
                    //float * buf = (float *)((char *)output_tensor->data + output_offset);
                    //std::cerr << "prompt " << output_idx
                    //    << " tensor contents for " << t->name << ": "
                    //    << buf[0] << ", "
                    //    << buf[1] << ", "
                    //    << buf[2] << " ... "
                    //    << buf[4093] << ", "
                    //    << buf[4094] << ", "
                    //    << buf[4095] << "\n";

                    ++output_idx;
                }
            }
        }

        // Continue running
        return true;
    }
}

int main(int argc, char ** argv) {
    gpt_params params;
    g_params = &params;

    bool mf = false;
    char *farg;
    std::deque<char *> ff;
    int findex = -1;
    for (int i=0; i<argc; i++) {
      if (strcmp(argv[i], "-f") == 0) {
        if (++i >= argc) {
          exit(1337);
          break;
        }
        findex = i;
        farg = argv[i];
        for (int j=0; argv[i][j] != '\0'; j++) {
          if (argv[i][j] == ',') {
            argv[i][j] = '\0';
            mf = true;
            ff.push_back(&argv[i][j+1]);
          }
        }
        break;
      }
    }
    if (!gpt_params_parse(argc, argv, params)) {
      printf("Fuck you");
        return 1;
    }
    llama_sampling_params & sparams = params.sparams;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("main", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
    llama_log_set(llama_log_callback_logTee, nullptr);
#endif // LOG_DISABLE_LOGS

    // TODO: Dump params ?
    //LOG("Params perplexity: %s\n", LOG_TOSTR(params.perplexity));

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_TEE("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_TEE("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_TEE("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_TEE("%s: build = %d (%s)\n",      __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG_TEE("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    LOG_TEE("%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    LOG("%s: llama backend init\n", __func__);
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model;
    llama_model *preloaded = NULL;
    llama_context * ctx;
    g_model = &model;
    g_ctx = &ctx;

    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
//STARTYO
    do
    {
      ggml_context * eval_ctx = nullptr;
      struct eval_callback_state eval_state;

      std::tie(model, ctx) = llama_init_from_gpt_params_with_cb_eval(
          params,
          eval_callback,
          (void *)&eval_state,
          preloaded);
      preloaded = model;

      if (model == NULL) {
          LOG_TEE("%s: error: unable to load model\n", __func__);
          return 1;
      }

      const int n_ctx_train = llama_n_ctx_train(model);
      const int n_ctx = llama_n_ctx(ctx);
      LOG("n_ctx: %d\n", n_ctx);

      if (n_ctx > n_ctx_train) {
          LOG_TEE("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                  __func__, n_ctx_train, n_ctx);
      }

      // print system information
      {
          LOG_TEE("\n");
          LOG_TEE("%s\n", get_system_info(params).c_str());
      }

      const bool add_bos = llama_should_add_bos_token(model);
      LOG("add_bos: %d\n", add_bos);

      std::vector<llama_token> embd_inp;

      LOG("tokenize the prompt\n");
      embd_inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);

      LOG("prompt: \"%s\"\n", log_tostr(params.prompt));
      LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

      // Should not run without any tokens
      if (embd_inp.empty()) {
          embd_inp.push_back(llama_token_bos(model));
          LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
      }

      // number of tokens to keep when resetting context
      if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct || params.chatml) {
          params.n_keep = (int)embd_inp.size();
      } else {
          params.n_keep += add_bos; // always keep the BOS token
      }

      if (params.verbose_prompt) {
          LOG_TEE("\n");
          LOG_TEE("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
          LOG_TEE("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
          for (int i = 0; i < (int) embd_inp.size(); i++) {
              LOG_TEE("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
          }

          if (params.n_keep > add_bos) {
              LOG_TEE("%s: static prompt based on n_keep: '", __func__);
              for (int i = 0; i < params.n_keep; i++) {
                  LOG_TEE("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
              }
              LOG_TEE("'\n");
          }
          LOG_TEE("\n");
      }

      // ctrl+C handling
      {
  #if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
          struct sigaction sigint_action;
          sigint_action.sa_handler = sigint_handler;
          sigemptyset (&sigint_action.sa_mask);
          sigint_action.sa_flags = 0;
          sigaction(SIGINT, &sigint_action, NULL);
  #elif defined (_WIN32)
          auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
              return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
          };
          SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
  #endif
      }

      LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
      LOG_TEE("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
      LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

      // group-attention state
      // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
      int ga_i = 0;

      const int ga_n = params.grp_attn_n;
      const int ga_w = params.grp_attn_w;

      if (ga_n != 1) {
          GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
          GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
        //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
        //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
          LOG_TEE("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
      }
      LOG_TEE("\n\n");

      bool input_echo           = true;
      bool display              = true;

      int n_past             = 0;
      int n_remain           = params.n_predict;

      std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
      std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
      std::ostringstream output_ss;     g_output_ss     = &output_ss;

      // the first thing we will do is to output the prompt, so set color accordingly
      console::set_display(console::prompt);
      display = params.display_prompt;

      std::vector<llama_token> embd;

      struct llama_sampling_context * ctx_sampling = llama_sampling_init(sparams);



      // Tokenized prompt is in embd_inp


      printf("Start training from prompts %s ...\n", farg);
      // Record prompt boundaries
      const int PROMPT_DELIMITER_TOKEN = 128000;

      // Index of each delimiter token in `embd_inp`.  These mark the end of each
      // prompt.
      std::vector<size_t> delim_idxs;

      for (size_t i = 0; i < embd_inp.size(); ++i) {
          if (embd_inp[i] == PROMPT_DELIMITER_TOKEN) {
              delim_idxs.push_back(i);
          }
      }

      // If the last prompt is missing an ending delimiter, add it.
      if (embd_inp.size() > 0 && embd_inp.back() != PROMPT_DELIMITER_TOKEN) {
          delim_idxs.push_back(embd_inp.size());
          embd_inp.push_back(PROMPT_DELIMITER_TOKEN);
      }

      size_t num_prompts = delim_idxs.size();

      if (num_prompts % 2 != 0) {
        printf("Number of prompts is odd. This indicates that you have fucked up somewhere. We need pairs of positive/negative prompts.\n");
        std::string delim_txt = llama_token_to_piece(ctx, PROMPT_DELIMITER_TOKEN);
        printf("Maybe it's a tokenization problem.\nDelimiter token: %d\nText representation '%s'.\n", PROMPT_DELIMITER_TOKEN, delim_txt.c_str());
        printf("Saving prompt tokenization result in 'repong-tokens.json' for analysis.\n");
        FILE *f = fopen("repong-tokens.json", "w");
        fprintf(f, "[\n  [");
        for (auto it = embd_inp.begin(); it != embd_inp.end(); ++it) {
          auto t = *it;
          fprintf(f, "\n    {\n      \"tok\": %d,\n      \"str\": \"%s\"\n    }", t, llama_token_to_piece(ctx, t).c_str());
          if (t == PROMPT_DELIMITER_TOKEN) {
            fprintf(f, "\n  ]");
            if (it != embd_inp.end() - 1) {
              fprintf(f, ",\n  [");
            } else {
              fprintf(f, "]");
            }
          } else {
            fprintf(f, ",");
          }
        }
        fprintf(f, "]");
        fclose(f);
        exit(1);
      }
      // Set up eval_state
      gguf_context * eval_gguf = gguf_init_empty();
      {
          int n_embd = llama_n_embd(model);
          int n_layer = llama_n_layer(model);
          std::cerr << "build eval state: " << embd_inp.size() << " tokens, " << num_prompts << " prompts, "
              << n_embd << " embd, " << n_layer << " layers\n";

          struct ggml_init_params params = {};
          params.mem_size = ((size_t)n_embd * num_prompts * sizeof(float) + 1024) * n_layer;
          eval_ctx = ggml_init(params);

          for (int i = 0; i < n_layer; ++i) {
              ggml_tensor * t = ggml_new_tensor_2d(eval_ctx, GGML_TYPE_F32, n_embd, num_prompts);
              snprintf(t->name, sizeof(t->name), "l_out-%d", i);
              eval_state.tensors.push_back(t);
              gguf_add_tensor(eval_gguf, t);
          }
          eval_state.first_prompt_idx = -1;
      }


      // Max tokens to include in a single batch.
      int batch_max_tokens = llama_n_batch(ctx);
      unsigned batch_max_seq = llama_n_seq_max(ctx);

      struct llama_batch batch = llama_batch_init(batch_max_tokens, 0, batch_max_seq);

      size_t prompt_idx = 0;
      std::vector<size_t> prev_prompt_start_for_sequence;
      auto last = ggml_time_ms();
      while (prompt_idx < num_prompts) {
          //std::cerr << "start batch at " << prompt_idx << "\n";
          eval_state.first_prompt_idx = prompt_idx;
          eval_state.extract_tokens.clear();
          // Reset `tokens_seen` to zero for all layers.
          eval_state.tokens_seen.clear();
          eval_state.tokens_seen.resize(eval_state.tensors.size(), 0);

          // Clear the token batch.
          batch.n_tokens = 0;
          size_t context_used = 0;

          llama_sampling_reset(ctx_sampling);

          // Add prompts to the batch until it's full.
          unsigned next_seq = 0;
          while (prompt_idx < num_prompts && next_seq < batch_max_seq) {
              size_t start = prompt_idx == 0 ? 0 : delim_idxs[prompt_idx - 1] + 1;
              size_t end = delim_idxs[prompt_idx];
              GGML_ASSERT(end > start && "empty prompts are not allowed");

              // Number of initial tokens in common between this prompt and the
              // previous prompt to use this sequence ID.
              size_t common = 0;

              if (next_seq < prev_prompt_start_for_sequence.size()) {
                  size_t prev_start = prev_prompt_start_for_sequence[next_seq];
                  GGML_ASSERT(prev_start <= start);
                  while (start + common < embd_inp.size()
                          && embd_inp[prev_start + common] != PROMPT_DELIMITER_TOKEN
                          && embd_inp[start + common] == embd_inp[prev_start + common]
                          ) {
                      ++common;
                  }
              }
              // If the current prompt is a prefix of the previous one, then it's
              // possible that all tokens are marked as common.  Ensure that the
              // last token of the current prompt is never marked as common so
              // that we get its hidden states.
              if (common >= end - start) {
                  common = end - start - 1;
              }
              GGML_ASSERT(start + common < end);

              //std::cerr << "adding " << start << " .. " << end
              //    << " (" << (end - start) << " tokens); "
              //    << context_used << " / " << batch_max_tokens << " context used\n";

              // FIXME: We artificially reduce the batch size limit here to
              // account for `llama_kv_cache_defrag` not fully defragmenting the
              // cache.  See the comment below.
              if (end - start > (size_t)(batch_max_tokens * 8 / 10 - context_used)) {
                  // Not enough space for this prompt in the batch.
                  std::cerr << "check space for prompt: [" << end << "-" << start << "=" << end - start << "] | " << context_used << "\n";
                  GGML_ASSERT(end - start <= (size_t)context_used);
                  break;
              }

              // Clear the KV cache for this sequence, except for the common
              // prefix.
              llama_kv_cache_seq_rm(ctx, next_seq, common, -1);

              //std::cerr << "still cooking\n";
              for (size_t j = start + common; j < end; ++j) {
                  int id = embd_inp[j];

                  // push the prompt in the sampling context in order to apply
                  // repetition penalties later for the prompt, we don't apply
                  // grammar rules
                  //llama_sampling_accept(ctx_sampling, ctx, id, false);

                  // Add the token to the current batch.  Its position within the
                  // context is relative to the start of the current prompt.
                  llama_batch_add(batch, id, j - start, {(int)next_seq}, false);

                  //const std::string token_str = llama_token_to_piece(ctx, id);
                  //std::cerr << "pos " << (j - start) << ": token "
                  //    << id << " \"" << token_str << "\"\n";
              }

              eval_state.extract_tokens.push_back(batch.n_tokens - 1);
              if (next_seq >= prev_prompt_start_for_sequence.size()) {
                  GGML_ASSERT(next_seq == prev_prompt_start_for_sequence.size());
                  prev_prompt_start_for_sequence.push_back(start);
              } else {
                  prev_prompt_start_for_sequence[next_seq] = start;
              }

              ++prompt_idx;
              ++next_seq;
              context_used += end - start;
          }

          while (prev_prompt_start_for_sequence.size() > next_seq) {
              llama_kv_cache_seq_rm(ctx, prev_prompt_start_for_sequence.size() - 1, -1, -1);
              prev_prompt_start_for_sequence.pop_back();
          }

          // Force defragmentation of the KV cache.  `llama_decode` needs a
          // contiguous block of `batch.n_tokens` cache slots, which it won't be
          // able to find if the cache is too fragmented.  Since we build batches
          // so as to maximize cache/context utilization, any fragmentation at
          // all will usually cause it to fail.
          //
          // FIXME: This sometimes doesn't fully defragment the cache, as shown
          // by `llama_kv_cache_view` debugging stats: if all free space was
          // contiguous, then `max_contiguous` should equal the number of free
          // cells (`n_cells - used_cells`), but often this is not the case.
          std::cerr << "defrag boi\n";
          auto defragstart = ggml_time_ms();
          llama_kv_cache_defrag(ctx);
          llama_kv_cache_update(ctx);

          // Debug prints to check cache usage and fragmentation:
          auto view = llama_kv_cache_view_init(ctx, 1);
          llama_kv_cache_view_update(ctx, &view);
          std::cerr << "kv cache cells: " << view.n_cells << "\n";
          std::cerr << "kv cache tokens: " << view.token_count << "\n";
          std::cerr << "kv cache used: " << view.used_cells << "\n";
          std::cerr << "kv cache max_contiguous: " << view.max_contiguous << "\n";
          std::cerr << "kv cache free cells: " << (view.n_cells - view.used_cells) << "\n";
          std::cerr << "defrag time: " << (ggml_time_ms() - defragstart) << "ms\n";

          GGML_ASSERT(batch.n_tokens > 0 && batch.n_tokens <= view.n_cells - view.used_cells);

          while (view.max_contiguous < batch.n_tokens) {
            defragstart = ggml_time_ms();
            std::cerr << "defrag again boi\n";
            llama_kv_cache_defrag(ctx);
            llama_kv_cache_update(ctx);

            
            // Debug prints to check cache usage and fragmentation:
            llama_kv_cache_view_update(ctx, &view);
            std::cerr << "kv cache cells: " << view.n_cells << "\n";
            std::cerr << "kv cache tokens: " << view.token_count << "\n";
            std::cerr << "kv cache used: " << view.used_cells << "\n";
            std::cerr << "kv cache max_contiguous: " << view.max_contiguous << "\n";
            std::cerr << "kv cache free cells: " << (view.n_cells - view.used_cells) << "\n";
          }

          std::cerr << "batch " << eval_state.first_prompt_idx << ": "
              << (prompt_idx - eval_state.first_prompt_idx) << " prompts, "
              << batch.n_tokens << " new tokens, "
              << context_used << " total tokens\n";

          //std::cerr << "prompt " << eval_state.prompt_idx << ": " << batch.n_tokens << " tokens\n";

          //batch.logits[batch.n_tokens - 1] = true;


          if (llama_decode(ctx, batch)) {
              LOG_TEE("%s : failed to eval\n", __func__);
              return 1;
          }

          auto now = ggml_time_ms();
          auto timedelta = now - last;
          last = now;
          std::cerr << "batch time: " << timedelta / 1000.0 << "s\n";

          //const llama_token id = llama_sampling_sample(ctx_sampling, ctx, nullptr, batch.n_tokens - 1);
          //const std::string token_str = llama_token_to_piece(ctx, id);
          //LOG_TEE("sample token %d: \"%s\"\n", id, token_str.c_str());
      }

      char *fname = farg, *fe = NULL;
      for (char *fn = fname; *fn != '\0'; fn++) {
        if (*fn == '/')
          fname = &fn[1];
        if (*fn == '_')
          fe = fn;
      }
      if (fe > fname) {
        strcpy(fe, "_data.gguf");
      }

      gguf_write_to_file(eval_gguf, fname, false);

      llama_print_timings(ctx);
      //write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);

      llama_free(ctx);
      ggml_free(eval_ctx);
      llama_sampling_free(ctx_sampling);
      if (ff.size()) {
        farg = ff.front();
        std::ifstream file(farg);
        if (!file) {
          fprintf(stderr, "error: failed to open file '%s'\n", farg);
          exit(1337);
        }
        // store the external file name in params
        params.prompt_file = farg;
        params.prompt.clear();
        std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
        if (!params.prompt.empty() && params.prompt.back() == '\n') {
          params.prompt.pop_back();
        }
        ff.pop_front();
      } else break;
    } while (true);
//ENDYO
    llama_free_model(model);

    llama_backend_free();
#ifndef LOG_DISABLE_LOGS
    LOG_TEE("Log end\n");
#endif // LOG_DISABLE_LOGS

    return 0;
}
