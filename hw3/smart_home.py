"""Deterministic teaching simulator. No real devices or external services."""
import copy

ROOMS = ["kitchen", "bedroom", "living_room"]


def tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required,
                       "additionalProperties": False}}}


ROOM = {"type": "string", "enum": ROOMS}
SMART_HOME_TOOLS = [
    tool("turn_light", "Switch a room light on or off. Ask for the room if unknown.",
         {"room": ROOM, "state": {"type": "string", "enum": ["on", "off"]}}, ["room", "state"]),
    tool("set_temperature", "Set an absolute room temperature in Celsius. For relative changes, obtain the current temperature first.",
         {"room": ROOM, "value": {"type": "integer", "minimum": 16, "maximum": 30}}, ["room", "value"]),
    tool("get_room_state", "Read the current light and temperature of a room.", {"room": ROOM}, ["room"]),
    tool("add_to_shopping_list", "Add an item to the shopping list.",
         {"item": {"type": "string", "minLength": 1}, "quantity": {"type": "integer", "minimum": 1, "default": 1}}, ["item"]),
    tool("set_timer", "Create a labeled timer in minutes.",
         {"minutes": {"type": "integer", "minimum": 1}, "label": {"type": "string", "minLength": 1}}, ["minutes", "label"]),
]
SYSTEM_PROMPT = "You are a smart home assistant. Use only the provided tools. Do not invent missing arguments or claim actions without tool results. Ask for clarification when needed."
BASELINE_PROMPTS = ["Turn on the kitchen light.", "Set the bedroom to 23 degrees.",
    "What is the state of the living room?", "Add two cartons of milk to my shopping list.",
    "Set a 10-minute timer labeled pasta.", "Turn off the bedroom light and set its temperature to 20 degrees.",
    "Turn off all lights.", "Hello!", "Open the garage.", "Turn on the light."]


class SmartHome:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rooms = {room: {"light": "off", "temperature": temperature}
                      for room, temperature in zip(ROOMS, [21, 21, 20])}
        self.shopping_list = []
        self.timers = []

    def snapshot(self):
        return copy.deepcopy({"rooms": self.rooms, "shopping_list": self.shopping_list, "timers": self.timers})

    def call(self, name, arguments):
        try:
            schema = next((t["function"]["parameters"] for t in SMART_HOME_TOOLS if t["function"]["name"] == name), None)
            if schema is None:
                return {"error": f"Unknown tool: {name}"}
            if not isinstance(arguments, dict):
                return {"error": "Arguments must be an object"}
            for key in schema["required"]:
                if key not in arguments:
                    return {"error": f"Missing argument: {key}"}
            for key, value in arguments.items():
                if key not in schema["properties"]:
                    return {"error": f"Unexpected argument: {key}"}
                spec = schema["properties"][key]
                if spec["type"] == "integer" and type(value) is not int:
                    return {"error": f"{key} must be an integer"}
                if spec["type"] == "string" and not isinstance(value, str):
                    return {"error": f"{key} must be a string"}
                if "enum" in spec and value not in spec["enum"]:
                    return {"error": f"Invalid {key}: {value}"}
                if "minimum" in spec and value < spec["minimum"] or "maximum" in spec and value > spec["maximum"]:
                    return {"error": f"{key} is out of range"}
                if spec.get("minLength") and not value.strip():
                    return {"error": f"{key} must not be empty"}
            if name == "turn_light":
                self.rooms[arguments["room"]]["light"] = arguments["state"]
            elif name == "set_temperature":
                self.rooms[arguments["room"]]["temperature"] = arguments["value"]
            elif name == "add_to_shopping_list":
                entry = {"item": arguments["item"], "quantity": arguments.get("quantity", 1)}
                self.shopping_list.append(entry)
                return copy.deepcopy(entry)
            elif name == "set_timer":
                self.timers.append(copy.deepcopy(arguments))
                return copy.deepcopy(arguments)
            return {"room": arguments["room"], **self.rooms[arguments["room"]]}
        except (TypeError, ValueError, KeyError) as error:
            return {"error": str(error)}
