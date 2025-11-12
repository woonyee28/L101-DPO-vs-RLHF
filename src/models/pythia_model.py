from transformers import GPTNeoXForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class PythiaModel:
    SUPPORTED_MODELS = ["EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-160m-deduped"]
    def __init__(self, name: str, revision: str = "step143000", cache_dir: str = None):
        if name not in self.SUPPORTED_MODELS:
            logger.info("Model not supported in this class!")
            raise ValueError(f"Model '{name}' is not supported. Supported models are: {self.SUPPORTED_MODELS}")
        
        self.name = name
        self.revision = revision
        self.cache_dir = cache_dir

        logger.info(f"Loading model {name} (revision: {revision}) at cache_dir: {cache_dir}")

        try:
            self.model = GPTNeoXForCausalLM.from_pretrained(
                name,
                revision=revision,
                cache_dir=cache_dir,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                name,
                revision=revision,
                cache_dir=cache_dir,
                )
        except Exception as e:
            logger.error(f"Failed to load model {name}")
            raise
    
    def generate(self, prompt: str, max_len: int = 100, temperature: float=0.9) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and Tokenizer are not loaded")

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        output = self.model.generate(input_ids, max_length=max_len, do_sample=True, temperature=temperature)

        generated_text = self.tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
        return generated_text
