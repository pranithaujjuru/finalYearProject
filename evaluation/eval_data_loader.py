import json
import os

class EvaluationDataLoader:
    def __init__(self, data_dir="./evaluation"):
        self.data_dir = data_dir
        self.vision_path = os.path.join(data_dir, "vision_ground_truth.json")
        self.nlp_path = os.path.join(data_dir, "nlp_ground_truth.jsonl")

    def load_vision_data(self):
        """Loads vision ground truth mapping filename -> results."""
        if not os.path.exists(self.vision_path):
            print(f"Warning: Vision ground truth not found at {self.vision_path}")
            return {}
        with open(self.vision_path, 'r') as f:
            return json.load(f)

    def load_nlp_data(self):
        """Loads NLP ground truth records from JSONL."""
        records = []
        if not os.path.exists(self.nlp_path):
            print(f"Warning: NLP ground truth not found at {self.nlp_path}")
            return []
        with open(self.nlp_path, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

if __name__ == "__main__":
    # Test the loader
    loader = EvaluationDataLoader()
    v_data = loader.load_vision_data()
    n_data = loader.load_nlp_data()
    print(f"Loaded {len(v_data)} vision samples and {len(n_data)} NLP samples.")
