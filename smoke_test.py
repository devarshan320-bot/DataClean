import sys
from ingestion import DataIngester
from context_builder import ContextBuilder
from ai_provider import OllamaProvider
from ai_interpreter import NaturalLanguageInterpreter
from contract_executor import execute_contract

def smoke_test():
    print("Ingesting test_data.csv...")
    ingester = DataIngester()
    res = ingester.ingest_csv("test_data.csv")
    if res.status != "success":
        print("Ingestion failed:", res.message)
        return
        
    df = res.dataframe
    print("Columns and types:")
    print(df.dtypes)
    
    print("\nBuilding context...")
    context = ContextBuilder().build_context(df)
    
    print("\nInterpreting instruction: 'Remove everyone below 18.'")
    provider = OllamaProvider()
    interpreter = NaturalLanguageInterpreter(provider)
    
    result = interpreter.interpret("Remove everyone below 18.", context, df)
    print("\nInterpretation Status:", result.status)
    if result.status == "ready":
        print("Contract:")
        print(result.contract.model_dump_json(indent=2))
        
        print("\nExecuting preview...")
        preview_df, impact = execute_contract(df.copy(), result.contract)
        print("Rows before:", len(df))
        print("Rows after:", len(preview_df))
        print("Preview Dataframe:")
        print(preview_df)
    else:
        print("Message:", result.message)
        print("Errors:", result.errors)

if __name__ == "__main__":
    smoke_test()
