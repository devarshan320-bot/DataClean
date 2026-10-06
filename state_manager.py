from contracts import ReadyContract
from contract_validator import validate_contract
from contract_executor import execute_contract

class HistoryManager:
    def __init__(self, original_df):
        self.original_df = original_df.copy()
        self.contracts = []
        
    def apply_contract(self, contract: ReadyContract):
        # Validate against the current dataframe state before accepting
        current_cols = list(self.get_current_df().columns)
        errors = validate_contract(contract, current_cols)
        if errors:
            raise ValueError(f"Contract failed semantic validation against current state: {errors}")
        
        self.contracts.append(contract)
        
    def undo(self):
        if self.contracts:
            self.contracts.pop()
            
    def get_current_df(self):
        df = self.original_df.copy()
        for c in self.contracts:
            df, _ = execute_contract(df, c)
        return df
        
    def get_history(self):
        return self.contracts
