class CaptionGenerator:

    def generate(self, scene_graph, decision):

        if not scene_graph:
            return decision

        descriptions = []
        seen = set()

        for obj in scene_graph:

            key = (obj["object"], obj["direction"])
            if key in seen:
                continue
            seen.add(key)

            if obj["direction"] == "center":
                descriptions.append(
                    f"A {obj['object']} is {obj['distance']} ahead"
                )
            else:
                descriptions.append(
                    f"A {obj['object']} is {obj['distance']} on your {obj['direction']}"
                )

        scene_description = ". ".join(descriptions)

        return scene_description + ". " + decision