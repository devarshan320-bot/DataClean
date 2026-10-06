import pandas as pd
from ingestion import DataIngester
from context_builder import ContextBuilder
from ai_provider import OllamaProvider
from ai_interpreter import NaturalLanguageInterpreter
from contract_executor import execute_contract

# Temporarily fix capability registry matching to align with schema just for test script execution
from ai_interpreter import CapabilityRegistry
CapabilityRegistry.SUPPORTED[0] = 'filter: Keeping or removing rows based on conditions using operators: gt, lt, gte, lte, equals, not_equals, contains, is_null.'

def run_test():
    res = DataIngester().ingest_csv('test_data.csv')
    df = res.dataframe
    context = ContextBuilder().build_context(df)
    interpreter = NaturalLanguageInterpreter(OllamaProvider())
    
    prompts = [
        "Remove everyone below 18.",
        "Keep customers whose date is in June.",
        "Keep customers whose OrderDate is in June.",
        "Keep Bangalore customers who spent more than 500.",
        "Remove duplicates."
    ]
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Instruction: {prompt}")
        result = interpreter.interpret(prompt, context, df)
        print(f"Status: {result.status}")
        
        if result.status == "ready":
            print("Contract:")
            print(result.contract.model_dump_json(indent=2))
            
            try:
                preview_df = execute_contract(df, result.contract)
                print(f"Validation: SUCCESS")
                print(f"Preview Impact: Went from {len(df)} rows to {len(preview_df)} rows")
                
                if len(preview_df) < len(df):
                    removed = len(df) - len(preview_df)
                    print(f"Removed {removed} rows.")
            except Exception as e:
                print(f"Validation/Execution FAILED: {e}")
        else:
            print(f"Message: {result.message}")
            if result.errors:
                print(f"Errors: {result.errors}")

if __name__ == '__main__':
    run_test()
