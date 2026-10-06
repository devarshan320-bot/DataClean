import json
import time
import pandas as pd
from ai_interpreter import NaturalLanguageInterpreter
from ai_provider import OllamaProvider
from context_builder import ContextBuilder

def get_test_df():
    return pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "Alice", "David"],
        "Age": [25, 17, 35, 25, 40],
        "Revenue": [1000, 500, 1500, 1000, 2000],
        "OrderDate": pd.to_datetime(["2023-06-01", "2023-06-15", "2023-07-01", "2023-06-01", "2023-08-01"]),
        "CreatedDate": pd.to_datetime(["2023-05-15", "2023-06-01", "2023-06-20", "2023-05-15", "2023-07-25"]),
        "City": ["Bangalore", "Mysore", "Chennai", "Bangalore", "Delhi"]
    })

class TrackingProvider(OllamaProvider):
    def __init__(self, model_name):
        super().__init__(model_name=model_name)
        self.last_raw = None
        self.last_time = 0.0
        
    def generate_structured_json(self, system_prompt, user_prompt, json_schema):
        start = time.time()
        try:
            raw = super().generate_structured_json(system_prompt, user_prompt, json_schema)
            self.last_raw = raw
            self.last_time = time.time() - start
            return raw
        except Exception as e:
            self.last_time = time.time() - start
            raise e

def run_suite(model_name: str, instructions: list, df: pd.DataFrame, context):
    print(f"\n{'#'*80}")
    print(f"### RUNNING SUITE FOR {model_name} ###")
    print(f"{'#'*80}\n")
    
    provider = TrackingProvider(model_name)
    interpreter = NaturalLanguageInterpreter(provider)
    
    for i, inst in enumerate(instructions, 1):
        print(f"\n{'='*60}")
        print(f"Instruction {i}: {inst}")
        
        provider.last_raw = None
        result = interpreter.interpret(inst, context, df)
        
        print(f"Inference Time: {provider.last_time:.2f}s")
        print("Raw Qwen response:")
        print(provider.last_raw)
        
        print("\nStage 1 IR (parsed):")
        try:
            parsed = json.loads(provider.last_raw)
            print(json.dumps(parsed, indent=2))
            pydantic_success = True
        except Exception:
            print("Failed to parse")
            pydantic_success = False
            
        print("\nResult Status:", result.status)
        if result.message:
            print("Message:", result.message)
        if result.errors:
            print("Errors:", result.errors)
            
        if result.contract:
            print("Contract:")
            print(result.contract.model_dump_json(indent=2))

def main():
    df = get_test_df()
    context = ContextBuilder().build_context(df)
    
    instructions = [
        "Remove everyone below 18.",
        "Keep customers whose OrderDate is in June.",
        "Keep customers whose date is in June.",
        "Keep Bangalore customers who spent more than 500.",
        "Remove duplicates.",
        "Sort by Revenue descending.",
        "Convert names to uppercase.",
        "Remove duplicates and then sort by Revenue descending.",
        "Remove the City column and then keep Bangalore customers.",
        "Group customers by City and calculate total Revenue."
    ]
    
    unseen_instructions = [
        "Drop any record where the person is younger than 18 years old.",
        "Retain only the entries if their order occurred during June.",
        "Discard the city information, then filter out anyone not from Bangalore.",
        "Show me only people from Bangalore who have a revenue over 500.",
        "Get rid of identical rows and order the remaining data by Revenue highest to lowest."
    ]
    
    print("\n" + "="*80)
    print("PHASE A: STANDARD SMOKE TESTS")
    print("="*80)
    run_suite("qwen2.5:3b", instructions, df, context)
    run_suite("qwen2.5:7b", instructions, df, context)
    
    print("\n" + "="*80)
    print("PHASE B: UNSEEN PARAPHRASED INSTRUCTIONS")
    print("="*80)
    run_suite("qwen2.5:3b", unseen_instructions, df, context)
    run_suite("qwen2.5:7b", unseen_instructions, df, context)

if __name__ == "__main__":
    main()
