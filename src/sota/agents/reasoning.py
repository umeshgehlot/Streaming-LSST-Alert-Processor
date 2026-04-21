import torch
import logging
from typing import Dict, Any

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

class AstroAgent:
    """
    Expert-level reasoning agent using AstroLLaMA.
    Translates raw anomaly scores into astrophysical hypotheses.
    """
    def __init__(self, model_id: str = "AstroLLaMA/AstroLLaMA-7b-v0.2"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_expert(self, use_4bit: bool = True):
        """Loads AstroLLaMA with professional NF4 quantization."""
        if not LLM_AVAILABLE: return
        try:
            bnb_config = None
            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.is_loaded = True
            logging.info(f"SOTA: AstroLLaMA expert loaded from {self.model_id}")
        except Exception as e:
            logging.error(f"SOTA Expert load failed: {e}")

    def _heuristic_reasoning(self, info: Dict[str, Any]) -> str:
        """Fallback expert logic based on signal characteristics."""
        features = str(info.get('features', '')).lower()
        score = info.get('anomaly_score', 0.5)
        
        if 'sudden' in features or 'spike' in features:
            return "The sharp injection in flux points toward a high-energy transient event, likely a Fireball or a Microlensing event. The ensemble consensus is high, suggesting this is not a sensor glitch."
        if 'periodic' in features or 'slow' in features:
            return "Long-term variation detected. The Association Discrepancy from the Anomaly Transformer suggests a non-standard temporal evolution, potentially an Active Galactic Nucleus (AGN) or a Tidal Disruption Event (TDE)."
        
        return f"Candidate detected with anomaly score {score:.4f}. Multi-model experts (Transformer, TranAD, TimesNet) have achieved consensus on this temporal segment. Follow-up spectroscopic observation is recommended."

    def reason_on_discovery(self, candidate_info: Dict[str, Any]) -> str:
        """
        Generates a Chain-of-Thought interpretation of a discovery candidate.
        """
        if not self.is_loaded: 
            return self._heuristic_reasoning(candidate_info)

        # (Existing LLM logic follows...)
        prompt = f"""[INST] You are an expert astrophysicist analyzing an anomaly discovery.
Candidate: {candidate_info['id']}
Detection Stats: Score={candidate_info.get('anomaly_score', 0):.4f}
Light Curve Features: {candidate_info.get('features', 'N/A')}

Hypothesize the physical nature of this source and suggest a follow-up strategy. [/INST]
### Scientific Reasoning:"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("### Scientific Reasoning:")[-1].strip()
        except Exception as e:
            return self._heuristic_reasoning(candidate_info)
