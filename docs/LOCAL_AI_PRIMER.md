# A Layman’s Guide to Local AI: Hardware, Engines, and Quants

If you are new to running Large Language Models (LLMs) on your own hardware, the terminology can be overwhelming. You don't just "download a model and run it." You have to match your **Hardware** to an **Engine**, pick the right **Model Size**, and download the correct **Format/Quant**.

This page explains how these pieces fit together in plain English.

> **Scope:** This is general orientation for local AI as a whole. The club-3090 stack itself is built and tested specifically on **NVIDIA 24 GB consumer GPUs (the RTX 3090)**. Other vendors (Apple, AMD, Intel) are included here as context to help you understand the landscape — they are not a support commitment of this stack. Vendor-specific viability changes quickly; treat those sections as a map, not a guarantee.

---

### Step 1: Hardware (The Rule of VRAM)
In local AI, processor speed is an afterthought. **VRAM (Video RAM) is the currency that matters.** If a model doesn't fit in your VRAM, it will either crash instantly or run so slowly on your system RAM that it’s unusable. 

*   **NVIDIA (CUDA):** The undisputed king of the local AI server. Because 99% of AI software is optimized for CUDA, "it just works." Used consumer cards with high VRAM (specifically the 24GB RTX 3090) are the gold standard for home labs.
*   **Apple Silicon (Unified Memory):** The "cheat code" for local AI. Macs share memory natively between the CPU and GPU. A Mac Studio with 128GB of RAM effectively has a 128GB graphics card, allowing users to run massive models quietly on a laptop/desktop using Apple's **MLX** framework.
*   **AMD (ROCm):** AMD has closed the gap massively. Modern ROCm works wonderfully as a CUDA alternative. High VRAM cards like the 7900 XTX (24GB) are now highly viable, first-class citizens.
*   **Intel (dGPU / Arc):** Intel's dedicated graphics cards (like the Arc A770 with 16GB VRAM) have entered the chat via the SYCL backend. While budget-friendly, the software ecosystem is still maturing.

*(For a comprehensive breakdown of hardware architectures, GPU generations, and feature support on this stack, see **[HARDWARE.md](./HARDWARE.md)**).*

---

### Step 2: Model Sizes and Architecture
When you search for a model, you will see numbers like `8B`, `32B`, or `8x7B`. "B" stands for Billions of Parameters (the number of "synapses" in the AI's brain).

*   **8B to 14B:** Fits on a basic 8GB-12GB GPU. Fast, great for basic chat and targeted coding, but prone to complex logic errors.
*   **32B to 35B:** The "home lab sweet spot" for a single 24GB RTX 3090 or a Mid-tier Mac. Extremely smart, highly coherent.
*   **70B+:** Nearing ChatGPT-4 levels of intelligence, but requires multiple GPUs or a very high-RAM Mac to run at reasonable speeds.

**Dense vs. MoE (Mixture of Experts)**
You will also see models labeled as "Dense" or "MoE."
*   **Dense:** The model uses its *entire brain* for every single word it generates. (e.g., Llama-3 70B uses 70B parameters per word). Requires high VRAM *and* high processing power.
*   **MoE (e.g., Mixtral 8x7B, DeepSeek V3):** The model is split into "experts." It might have 50B total parameters, but only activates 12B of them to answer your specific question. MoE requires high VRAM to store the whole brain, but *very low compute power* to run, making them incredibly fast.

---

### Step 3: The Engines (The Software that runs the AI)
To make the model talk, you need an "Inference Engine."

*   **llama.cpp (Cross-platform):** *The rugged off-roader.* It runs on Windows, Linux, Macs, and regular CPUs. If you want to split a model between your GPU and standard RAM, you use this. (Apps like Ollama, LM Studio, and Msty are just user-friendly wrappers around `llama.cpp`).
*   **vLLM (NVIDIA/AMD Linux):** *The Formula 1 car.* Built for heavy-duty, dedicated GPU setups. It is strict—if you run out of memory, it crashes—but it is unimaginably fast and can serve dozens of users at exactly the same time.
*   **MLX (Apple Silicon):** Apple's native machine learning framework. If you are on an M-series Mac, using engines built on MLX (like `mlx-lm`) is the fastest, most battery-efficient way to run models.
*   **KTransformers:** A newer hybrid engine built specifically to run massive MoE models (like DeepSeek R1) by keeping the math-heavy parts on the GPU and offloading the rest to System RAM.

*(For a detailed technical comparison of server engine memory models and CLI surfaces, see **[INFERENCE_ENGINES.md](./INFERENCE_ENGINES.md)**).*

---

### Step 4: Quants, Cache, and Context (Shrinking the model)
An uncompressed 70B model is roughly 140 GB. To make it fit your hardware, the community "quantizes" (compresses) it by rounding off the decimal points. **The Engine you chose in Step 3 dictates the Format you download.**

**1. Formats based on your Engine**
*   **GGUF (`llama.cpp`):** You **must** download `.gguf` files. `Q4_K_M` (4-bit) is the community sweet spot for size vs. smartness. `Q8_0` (8-bit) is nearly identical to the original but half the size.
*   **Safetensors (`vLLM` / `sglang`):** You must download a standard Hugging Face repo (a folder full of `.safetensors` files). Look for **AWQ** or **GPTQ** (4-bit VRAM savers) or **FP8** (the 8-bit standard for modern NVIDIA cards).
    > **RTX 3090 caveat:** FP8 *compute* (running FP8-weight models) requires Ada/Hopper (RTX 4090/5090, H100) — the **3090 (Ampere) cannot run FP8 weights** and will fall back or fail. On a 3090, use **AWQ/GPTQ** for 4-bit weights; FP8 is only useful here as a **KV-cache** storage format (`fp8_e5m2`, see Step 4.2), not as a weight format.
*   **MLX format (`mlx-lm`):** Macs use a custom natively optimized format (usually labeled as `-mlx` on Hugging Face).

**2. Tokens, Context, and KV Cache**
Models don't read words; they read **Tokens** (roughly 3/4 of a word).
The **Context Window** is the AI's short-term memory during your conversation. If a model has an 8,000-token context window and you paste a 10,000-token PDF into it, it will literally "forget" the beginning by the time it reaches the end.
*   **The VRAM Cost:** Your conversation history is stored in VRAM as the **KV Cache**. A 32,000-token context window can consume over 10GB of VRAM *on top* of the model weights. The KV Cache can also be quantized (e.g., `fp8_e5m2`) to squeeze massive conversations onto consumer GPUs.

*(For the deep-dive on exact byte-widths, capacity costs, and kernel constraints for all dtypes on this stack, see **[DTYPE_MATRIX.md](./DTYPE_MATRIX.md)** and **[KERNEL_MATRIX.md](./KERNEL_MATRIX.md)**).*

---

### Step 5: Prompt Templates (How to talk to it)
If you boot up a model, say "Hello", and it responds with HTML code or an endless loop of `User: AI: User:`, your model isn't broken—you are using the wrong **Prompt Template**. 

Base AI models are essentially super-powered autocorrects; they just predict the next word. To make them act like chat assistants, they are trained on specific hidden formatting tags (like `ChatML`, `Llama-3`, or `Alpaca`). 
Most modern frontends (like Open WebUI or LM Studio) detect this automatically, but if a model is speaking gibberish, checking the template is your first debugging step.

---

### Putting it together
Figuring out *"Can I run this AWQ Safetensor in vLLM on my two 3090s with a 32,000 context window?"* requires manual math. You must calculate the weight size, the KV cache overhead, the sequence lengths, and the hardware limits. 

*Note: The club-3090 stack does this math for you. Curated models ship heavily-tested, pre-calculated profiles; and the universal `pull` workflow evaluates **any** safetensors HF repo against this stack's KV math before downloading, so you get a fit verdict instead of a guess. See [PULL.md](./PULL.md) and [KV_MATH.md](./KV_MATH.md).*