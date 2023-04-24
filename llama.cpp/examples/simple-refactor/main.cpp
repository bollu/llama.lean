// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

static console_state con_st;

static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    set_console_color(con_st, CONSOLE_COLOR_DEFAULT);
    printf("\n"); // this also force flush stdout.
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    gpt_params params;
    params.model = "models/llama-7B/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    con_st.use_color = params.use_color;

    if (params.perplexity) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    params.prompt = "A Kahler manifold is ";


    llama_context * ctx;

    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.use_mmap   = params.use_mmap;
        lparams.use_mlock  = params.use_mlock;

        ctx = llama_init_from_file(params.model.c_str(), lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }
    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // tokenize the prompt
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    const int N_CTX = llama_n_ctx(ctx);


    // determine newline token
    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    fprintf(stderr, "sampling: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n",
        params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d", N_CTX, params.n_batch, params.n_predict);
    fprintf(stderr, "generate: prompt: '%s'\n", params.prompt.c_str());
    fprintf(stderr, "\n\n");

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(N_CTX);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    // number of tokens from prior llama_eval calls to be used.
    // this is reset whenver it grows to longer than the context length.
    // reset to zero whenever (past_ctx_len + embd.size() > N_CTX). Tells how many tokens from past calls should be reused.
    int past_ctx_len     = 0;
    int n_remain   = params.n_predict; // number of tokens to predict.

    // the first thing we will do is to output the prompt, so set color accordingly
    set_console_color(con_st, CONSOLE_COLOR_DEFAULT);

    // what is difference between last_n_tokens and 'embed'?
    std::vector<llama_token> embd;

    while (n_remain != 0) {
        // TODO: change loop order so that we always know that 'embd.size() > 0'.
        // That is, first try producing stuff by reading input/`llama_sample`, and then call
        // llama_eval.
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            if (past_ctx_len + (int) embd.size() > N_CTX) {
                embd = {};
                embd.insert(embd.begin(), last_n_tokens.end() - past_ctx_len/2, last_n_tokens.end());
                past_ctx_len = 0;
            }

            // use past_ctx_len tokens from previous llama_eval call.
            // at this point, if we came from the prior if() condition, then
            // 'past_ctx_len = 0', because we have shoved all the data into the embd
            // vector.
            if (llama_eval(ctx, embd.data(), embd.size(), past_ctx_len, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
        }

        past_ctx_len += embd.size();
        embd.clear(); // POST: embd.size() = 0

        if (!embd_inp.size()) {
            // out of user input, sample next token
            const int32_t top_k          = params.top_k;
            const float   top_p          = params.top_p;
            const float   temp           = params.temp;
            const float   repeat_penalty = params.repeat_penalty;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(ctx);

                if (params.ignore_eos) {
                    logits[llama_token_eos()] = 0;
                }

                id = llama_sample_top_p_top_k(ctx,
                        last_n_tokens.data() + N_CTX - params.repeat_last_n,
                        params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // add it to the context
            embd.push_back(id); // POST: embd.size() = 1
            --n_remain; // decrement remaining sampling budget
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while (embd_inp.size()) { // @sid TODO: remove embd_inp_ix and just directly modify embd_inp by erasing values.
                last_n_tokens.erase(last_n_tokens.begin()); // erase first token.
                last_n_tokens.push_back(*embd_inp.begin()); // add new token.
                embd.push_back(*embd_inp.begin()); // TODO: what is embed versus last_n_tokens?
                embd_inp.erase(embd_inp.begin());
                // note that we do not decrement 'n_remain'
                // only consume upto batch size.
                if ((int) embd.size() >= params.n_batch) {
                  // TODO: can I assert that this break will never happen? assert(false && "will never happen");
                  break;
                }
            }
            // case 1: emed.size() = embed_inp.size(); nconsumes = embed_inp.size()
            // case 2: emed.size() = params.n_batch
        }

        // display text (display everything in embd).
        for (auto id : embd) {
            printf("%s", llama_token_to_str(ctx, id));
        }
        fflush(stdout);


        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos()) {
            fprintf(stderr, " [end of text]\n");
            break;
        }

    }

#if defined (_WIN32)
    signal(SIGINT, SIG_DFL);
#endif

    llama_print_timings(ctx);
    llama_free(ctx);

    set_console_color(con_st, CONSOLE_COLOR_DEFAULT);

    return 0;
}
