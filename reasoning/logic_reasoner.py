from rdflib import Graph, URIRef, Literal, Namespace, RDF

class SymbolicGuardrail:
    """
    First-Order Logic Guardrail using RDFlib to enforce safety constraints 
    over proposed threat mitigation actions.
    """
    def __init__(self):
        self.g = Graph()
        self.SEC = Namespace("http://lackofintent.org/security#")
        self.g.bind("sec", self.SEC)
        self._init_knowledge_base()

    def _init_knowledge_base(self):
        # Define Domain Knowledge / Ontology Constraints
        # Entity: Nacos_Config_DB is tagged as Critical Infrastructure
        nacos_db = URIRef(self.SEC["Nacos_Config_DB"])
        self.g.add((nacos_db, RDF.type, self.SEC["CriticalDatabase"]))
        self.g.add((nacos_db, self.SEC["hasPolicy"], Literal("NO_HOST_ISOLATION")))

    def assert_neural_hypothesis(self, process_name: str, action: str, target_resource: str, risk_score: float):
        """Discretizes continuous neural output into RDF Semantic Triplets."""
        proc = URIRef(self.SEC[process_name])
        target = URIRef(self.SEC[target_resource])

        self.g.add((proc, RDF.type, self.SEC["UnsignedProcess"]))
        self.g.add((proc, self.SEC["targetAction"], Literal(action)))
        self.g.add((proc, self.SEC["targets"], target))
        self.g.add((proc, self.SEC["hasRiskScore"], Literal(risk_score)))

    def evaluate_mitigation(self, process_name: str, target_resource: str) -> dict:
        """
        Evaluates First-Order Logic rules against proposed mitigation:
        IF Target is CriticalDatabase AND Action would cause DoS
        THEN Override Host Isolation -> Freeze PID (SIGSTOP) & Retain Network.
        """
        target_ref = URIRef(self.SEC[target_resource])
        is_critical = (target_ref, RDF.type, self.SEC["CriticalDatabase"]) in self.g

        if is_critical:
            return {
                "decision": "PERMITTED_WITH_GUARDRAIL",
                "executed_action": "SIGSTOP_PROCESS_PID",
                "blocked_action": "ISOLATE_HOST_NODE",
                "reasoning": f"Target '{target_resource}' is Critical. Node isolation blocked to prevent self-inflicted DoS."
            }
        else:
            return {
                "decision": "PERMITTED_DEFAULT",
                "executed_action": "ISOLATE_HOST_NODE",
                "blocked_action": "NONE",
                "reasoning": "Standard node isolation permitted."
            }