class SceneGraphBuilder:

    def build(self, detections):

        graph = []

        for d in detections:
            graph.append({
                "object": d["label"],
                "direction": d["direction"],
                "distance": d["distance"],
                "risk": d["risk_score"]
            })

        return graph