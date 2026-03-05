class NavigationAgent:

    def decide(self, scene):

        if not scene:
            return "The path ahead appears clear. You may move forward."

        vehicles = {"car","motorcycle","bus","truck"}

        vehicle_count = sum(1 for o in scene if o["object"] in vehicles)

        if vehicle_count >= 2:
            return "Multiple vehicles detected ahead. Stay away from the road and move along the right side."

        primary = max(scene, key=lambda x: x["risk"])

        obj = primary["object"]
        direction = primary["direction"]
        distance = primary["distance"]

        if direction in ["center","right"]:
            return f"A {obj} is {distance} ahead. Move slightly to the right side."

        if direction in ["left","far left"]:
            return f"A {obj} is {distance} on the left. Move slightly right."

        return "Proceed carefully."
    
    def verify(self, scene, instruction):
        for obj in scene:

            if obj["direction"] == "left" and "left" in instruction:
                return "Left side has an obstacle. Move slightly right."

            if obj["direction"] == "right" and "right" in instruction:
                return "Right side blocked. Move slightly left."

        return instruction