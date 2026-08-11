import torch
from models.rgcn_encoder import ThreatRGCN
from reasoning.logic_reasoner import SymbolicGuardrail

def run_simulation():
    # Fix random seed for reproducible demo output
    torch.manual_seed(1)

    print("=== [1] Initializing Heterogeneous Graph Telemetry ===")
    # Nodes: 0=Langflow_Parent, 1=Python3_Payload (JADEPUFFER), 2=Nacos_Config_DB
    x = torch.tensor([[0.1, 0.2], [0.9, 0.8], [0.0, 0.9]], dtype=torch.float)
    
    # Edges & Relations: 0="spawns", 1="attempts_bulk_write"
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    edge_type = torch.tensor([0, 1], dtype=torch.long)

    print("=== [2] Running R-GCN Neural Perception Layer ===")
    model = ThreatRGCN(in_channels=2, hidden_channels=4, out_channels=1, num_relations=2)
    model.eval()
    
    with torch.no_grad():
        risk_scores = model(x, edge_index, edge_type)
        payload_risk = risk_scores[1].item()

    print(f"-> Node 1 (Python3_Payload) Calculated Risk Score: {payload_risk:.4f}")

    # Set threshold appropriate for untrained demo model (or override payload_risk)
    ALERT_THRESHOLD = 0.20

    if payload_risk >= ALERT_THRESHOLD:
        print("\n=== [3] Translating Neural Vector to RDF Triplet ===")
        guardrail = SymbolicGuardrail()
        
        guardrail.assert_neural_hypothesis(
            process_name="Python3_Payload",
            action="BULK_DROP_TABLE",
            target_resource="Nacos_Config_DB",
            risk_score=payload_risk
        )

        print("=== [4] Evaluating First-Order Logic Guardrails ===")
        result = guardrail.evaluate_mitigation("Python3_Payload", "Nacos_Config_DB")

        print(f"\n[DECISION]:        {result['decision']}")
        print(f"[EXECUTED ACTION]: {result['executed_action']}")
        print(f"[BLOCKED ACTION]:  {result['blocked_action']}")
        print(f"[REASONING]:       {result['reasoning']}\n")
    else:
        print("\n-> Risk score below threshold. No action taken.")

if __name__ == "__main__":
    run_simulation()