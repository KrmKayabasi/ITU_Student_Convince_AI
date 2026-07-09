import time
import sys

def benchmark_mlx():
    print("=" * 50)
    print("MLX LENGHT & SPEED BENCHMARK")
    print("=" * 50)
    
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("[Error] mlx-lm kütüphanesi yüklü değil. Kuruluyor...")
        import subprocess
        subprocess.run(["./venv/bin/pip", "install", "mlx-lm"])
        from mlx_lm import load, generate
        
    model_id = "mlx-community/gemma-4-e2b-it-4bit"
    print(f"[MLX] Modeli yükleniyor: {model_id}...")
    from mlx_lm.utils import load_model, load_tokenizer, _download
    start = time.time()
    model_path = _download(model_id)
    model, config = load_model(model_path, strict=False)
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id", None))
    print(f"[MLX] Model {time.time() - start:.2f} saniyede yüklendi.")
    
    # Prompt with no length limits to test full intelligence
    prompt = "İstanbul Teknik Üniversitesi hakkında detaylı bilgi verir misin?"
    messages = [
        {"role": "system", "content": "Sen yararlı bir Türkçe asistansın."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    print("\n[MLX] Yanıt üretiliyor...")
    start = time.time()
    
    # Generate response
    response = generate(model, tokenizer, prompt=formatted_prompt, verbose=False, max_tokens=256)
    
    generation_time = time.time() - start
    print(f"\n[Asistan Yanıtı]:\n{response}\n")
    print(f"[MLX] Süre: {generation_time:.2f} saniye")
    
if __name__ == "__main__":
    benchmark_mlx()
