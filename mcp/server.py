"""FastMCP adapter for the Lost City TypeScript Agent API.

This module deliberately contains no game rules. It translates MCP tool calls
into authenticated HTTP requests to the TypeScript API running with the game
world, so the engine remains the single source of truth.
"""

from __future__ import annotations

import asyncio
from collections import deque
import os
from typing import Any, Literal

import httpx
from fastmcp import FastMCP


class LostCityApiError(RuntimeError):
    """An API request failed with a useful response for an agent."""


class LostCityApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("LOSTCITY_API_URL", "http://127.0.0.1:80/api/v1")).rstrip("/")
        self.username = username or os.getenv("LOSTCITY_API_USERNAME", "admin")
        self.password = password or os.getenv("LOSTCITY_API_PASSWORD", "password")
        self.transport = transport

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        for attempt in range(12):
            async with httpx.AsyncClient(
                base_url=self.base_url,
                auth=(self.username, self.password),
                timeout=15.0,
                transport=self.transport,
            ) as client:
                response = await client.request(method, path, json=json, params=params)

            if response.status_code == 409:
                try:
                    detail = response.json()
                except ValueError:
                    detail = None
                if isinstance(detail, dict) and detail.get("error") == "agent_pending" and attempt < 11:
                    await asyncio.sleep(0.25)
                    continue

            if response.is_error:
                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text
                raise LostCityApiError(f"Lost City API returned HTTP {response.status_code}: {detail}")
            return response.json()

        raise LostCityApiError("Lost City API agent activation timed out")


mcp = FastMCP("Lost City Agents")
api = LostCityApiClient()
_keepalive_tasks: dict[str, asyncio.Task[None]] = {}
_keepalive_state: dict[str, dict[str, Any]] = {}
_progression_tasks: dict[str, asyncio.Task[None]] = {}
_progression_state: dict[str, dict[str, Any]] = {}
_tutorial_ranged_attacks: set[str] = set()
_tutorial_magic_attacks: set[str] = set()
_map_cache: dict[str, dict[str, Any]] = {}
_MAX_MAP_CACHE_RADIUS = 48


def _cached_map_route(window: dict[str, Any], target_x: int, target_z: int, target_level: int | None = None) -> dict[str, Any]:
    """Plan a cardinal route from cached geometry without issuing a game action."""

    center = window.get("center") or {}
    level = int(center.get("level", 0))
    if target_level is not None and target_level != level:
        return {
            "status": "blocked",
            "validated": False,
            "validation": "cached_geometry_same_floor_only",
            "reason": "cached map routes do not plan floor transitions",
        }
    bounds = window.get("bounds") or {}
    min_x, max_x = int(bounds.get("minX", 0)), int(bounds.get("maxX", -1))
    min_z, max_z = int(bounds.get("minZ", 0)), int(bounds.get("maxZ", -1))
    if not (min_x <= target_x <= max_x and min_z <= target_z <= max_z):
        return {
            "status": "blocked",
            "validated": False,
            "validation": "cached_geometry_target_outside_window",
            "reason": "target is outside the cached map window",
        }

    rows = {int(row["z"]): row for row in window.get("rows", []) if isinstance(row, dict) and "z" in row}
    start = (int(center.get("x", 0)), int(center.get("z", 0)))
    target = (target_x, target_z)
    queue = deque([start])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    directions = ((1, 0, 1), (-1, 0, 2), (0, 1, 4), (0, -1, 8))
    while queue:
        x, z = queue.popleft()
        if (x, z) == target:
            break
        row = rows.get(z)
        if row is None:
            continue
        index = x - min_x
        exits = row.get("exits", "")
        blocked = row.get("blocked", "")
        if index < 0 or index >= len(exits) or index >= len(blocked) or blocked[index] == "1":
            continue
        mask = int(exits[index], 16)
        for dx, dz, bit in directions:
            neighbor = (x + dx, z + dz)
            nx, nz = neighbor
            if not (min_x <= nx <= max_x and min_z <= nz <= max_z) or neighbor in previous:
                continue
            neighbor_row = rows.get(nz)
            neighbor_index = nx - min_x
            if neighbor_row is None or neighbor_index < 0 or neighbor_index >= len(neighbor_row.get("blocked", "")):
                continue
            if neighbor_row["blocked"][neighbor_index] == "1" or not (mask & bit):
                continue
            previous[neighbor] = (x, z)
            queue.append(neighbor)

    if target not in previous:
        return {
            "status": "blocked",
            "validated": False,
            "validation": "cached_geometry_no_route",
            "reason": "cached exits do not connect the agent to the target",
            "start": {"level": level, "x": start[0], "z": start[1]},
            "target": {"level": level, "x": target_x, "z": target_z},
        }

    path: list[tuple[int, int]] = []
    current: tuple[int, int] | None = target
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    # Compress the tile path into direction-change waypoints; the server still
    # re-plans and validates the eventual move against current collision.
    waypoints: list[dict[str, int]] = []
    last_direction: tuple[int, int] | None = None
    for index in range(1, len(path)):
        direction = (path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1])
        if last_direction is not None and direction != last_direction:
            x, z = path[index - 1]
            waypoints.append({"level": level, "x": x, "z": z})
        last_direction = direction
    if path[-1] != start:
        waypoints.append({"level": level, "x": path[-1][0], "z": path[-1][1]})
    return {
        "status": "complete",
        "validated": True,
        "validation": "cached_static_geometry_route_proposal",
        "authoritative": False,
        "start": {"level": level, "x": start[0], "z": start[1]},
        "target": {"level": level, "x": target_x, "z": target_z},
        "tile_count": max(0, len(path) - 1),
        "waypoints": waypoints,
    }


def _cached_frontier_door(
    window: dict[str, Any],
    destination: dict[str, int],
    rejected: set[tuple[int, int, int]],
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]] | None:
    """Find a live closed door whose approach tile is in the cached component."""

    nearby = (window.get("observation") or {}).get("nearby") or {}
    candidates: list[tuple[int, int, dict[str, Any], dict[str, int], dict[str, Any]]] = []
    for entity in nearby.get("locations", []):
        if "door" not in (entity.get("name") or "").casefold():
            continue
        position = entity.get("position") or {}
        entity_key = (int(entity.get("id", 0)), int(position.get("x", 0)), int(position.get("z", 0)))
        if entity_key in rejected or position.get("level") != destination.get("level"):
            continue
        option = _option_match(entity.get("options") or [], ["open"])
        if option is None:
            continue
        for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)):
            approach = {
                "level": int(position.get("level", destination.get("level", 0))),
                "x": int(position.get("x", 0)) + dx,
                "z": int(position.get("z", 0)) + dz,
            }
            route = _cached_map_route(window, approach["x"], approach["z"], approach["level"])
            if route.get("status") != "complete":
                continue
            distance_to_goal = max(
                abs(int(destination["x"]) - int(position.get("x", 0))),
                abs(int(destination["z"]) - int(position.get("z", 0))),
            )
            candidates.append((int(route.get("tile_count", 0)) + distance_to_goal, option[0], entity, approach, route))
    if not candidates:
        return None
    _, option, entity, approach, route = min(
        candidates,
        key=lambda item: (item[0], int((item[2].get("position") or {}).get("x", 0)), int((item[2].get("position") or {}).get("z", 0))),
    )
    return {**entity, "selected_option": option}, approach, route


async def _keepalive_loop(agent_id: str, username: str, interval_seconds: float) -> None:
    """Keep an attached API session observed without undoing explicit logout."""

    try:
        while True:
            try:
                await api.request("GET", f"/agents/{agent_id}")
                _keepalive_state[agent_id] = {
                    "enabled": True,
                    "username": username,
                    "status": "active",
                }
            except LostCityApiError as error:
                # If the browser reconnects, attach_agent reuses the same
                # engine session. If the user intentionally logged out, this
                # remains offline and does not create a replacement player.
                try:
                    attached = await api.request("POST", "/agents/attach", json={"username": username})
                    _keepalive_state[agent_id] = {
                        "enabled": True,
                        "username": username,
                        "status": "reattached",
                        "agent_id": attached.get("id", agent_id),
                    }
                except LostCityApiError as reattach_error:
                    _keepalive_state[agent_id] = {
                        "enabled": True,
                        "username": username,
                        "status": "waiting_for_client",
                        "last_error": str(reattach_error),
                        "previous_error": str(error),
                    }
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        raise


def _ensure_keepalive(agent_id: str, username: str, interval_seconds: float = 10.0) -> None:
    task = _keepalive_tasks.get(agent_id)
    if task and not task.done():
        return
    _keepalive_tasks[agent_id] = asyncio.create_task(_keepalive_loop(agent_id, username, interval_seconds))
    _keepalive_state[agent_id] = {"enabled": True, "username": username, "status": "starting"}


async def _stop_keepalive(agent_id: str) -> None:
    task = _keepalive_tasks.pop(agent_id, None)
    _keepalive_state.pop(agent_id, None)
    if not task:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# This is deliberately a policy layer, not a second game engine. Each entry
# describes how an expert agent should approach a skill using the actions that
# the TypeScript API already exports. ``interaction`` skills are discoverable
# from live entity options, ``inventory`` skills use live item names/slots, and
# ``context`` skills report the remaining content policy instead of pretending
# that a random interaction will train the requested skill.
SKILL_PLAYBOOK: dict[str, dict[str, Any]] = {
    "attack": {
        "mode": "combat",
        "targets": ["npc"],
        "option_tokens": ["attack"],
        "strategy": "Fight the nearest NPC exposing the live Attack option; keep observing health and disengage through the engine safety guard.",
    },
    "defence": {
        "mode": "combat",
        "targets": ["npc"],
        "option_tokens": ["attack"],
        "strategy": "Fight safe, reachable NPCs while the client combat style is configured for Defence.",
    },
    "strength": {
        "mode": "combat",
        "targets": ["npc"],
        "option_tokens": ["attack"],
        "strategy": "Fight safe, reachable NPCs while the client combat style is configured for Strength.",
    },
    "hitpoints": {
        "mode": "combat",
        "targets": ["npc"],
        "option_tokens": ["attack"],
        "strategy": "Fight safe, reachable NPCs; Hitpoints XP is produced by normal combat.",
    },
    "ranged": {
        "mode": "context",
        "targets": ["npc"],
        "option_tokens": ["attack"],
        "strategy": "Equip a ranged weapon and ammunition and select the ranged combat style, then use combat automation.",
        "missing": ["equipment persistence", "combat-style policy", "ammo restocking"],
    },
    "prayer": {
        "mode": "context",
        "targets": ["loc", "npc"],
        "option_tokens": ["pray", "altar", "bury"],
        "strategy": "Use bones through the inventory or interact with a prayer altar when the content exposes that option.",
        "missing": ["bone sourcing", "banking/restocking policy"],
    },
    "magic": {
        "mode": "context",
        "targets": ["npc"],
        "option_tokens": ["attack"],
        "strategy": "Prepare runes, select a spell, and then fight safe NPCs while observing health and run energy.",
        "missing": ["spell selection policy", "rune provisioning"],
    },
    "cooking": {
        "mode": "context",
        "targets": ["loc", "obj"],
        "option_tokens": ["cook", "range", "fire"],
        "strategy": "Find a cooking station and use raw food on it, banking or restocking when inventory policy requires it.",
        "missing": ["banking/restocking policy"],
    },
    "woodcutting": {
        "mode": "interaction",
        "targets": ["loc"],
        "option_tokens": ["chop", "tree"],
        "strategy": "Select the nearest live tree option, re-observe after each depletion, and walk to the next tree.",
        "requirement_skill": "woodcutting",
        "entity_requirements": {"oak": 15, "willow": 30, "maple": 45, "yew": 60, "magic": 75},
    },
    "fletching": {
        "mode": "context",
        "targets": ["loc", "obj"],
        "option_tokens": ["fletch"],
        "strategy": "Use a knife or fletching tool on logs, then choose the recipe dialog and repeat until materials are exhausted.",
        "missing": ["recipe/dialogue selection", "banking/restocking policy"],
    },
    "fishing": {
        "mode": "interaction",
        "targets": ["npc", "loc"],
        "option_tokens": ["fish", "net", "bait", "lure", "harpoon"],
        "strategy": "Use the nearest live fishing option and reselect a spot when the current target disappears.",
        "requirement_skill": "fishing",
        # The packed runtime renumbers content categories; keep the source and
        # current 274 identifiers until the engine exposes symbolic script
        # subjects directly.
        "category_requirements": {
            "saltfish": 1,
            "category_453": 5,
            "category_632": 5,
            "category_633": 65,
        },
    },
    "firemaking": {
        "mode": "inventory",
        "targets": [],
        "option_tokens": [],
        "strategy": "Use a live tinderbox and logs pair, then re-observe XP and inventory before repeating.",
        "missing": ["log/tinderbox provisioning", "banking/restocking policy"],
    },
    "crafting": {
        "mode": "context",
        "targets": ["loc", "obj"],
        "option_tokens": ["craft", "make"],
        "strategy": "Use a live pair of shears on an unsheared sheep, then use live wool on a spinning wheel; extend the recipe registry for additional crafting methods.",
        "missing": ["additional recipe registry", "banking/restocking policy", "long-distance station routing"],
    },
    "smithing": {
        "mode": "context",
        "targets": ["loc", "obj"],
        "option_tokens": ["smelt", "smith", "anvil", "furnace"],
        "strategy": "Use bars on a furnace or anvil, choose a valid product, and bank or restock when needed.",
        "missing": ["recipe/dialogue selection", "banking/restocking policy"],
    },
    "mining": {
        "mode": "interaction",
        "targets": ["loc"],
        "option_tokens": ["mine", "rock"],
        "strategy": "Select the nearest live mining option, re-observe after depletion, and route around blockers.",
    },
    "herblore": {
        "mode": "context",
        "targets": ["loc", "obj"],
        "option_tokens": ["clean", "mix", "make"],
        "strategy": "Combine valid ingredients from the inventory and resolve the resulting recipe action.",
        "missing": ["ingredient policy", "recipe/dialogue selection", "banking/restocking policy"],
    },
    "agility": {
        "mode": "interaction",
        "targets": ["loc"],
        "option_tokens": ["climb", "cross", "jump", "balance", "obstacle", "squeeze"],
        "strategy": "Choose the nearest reachable obstacle option and re-plan after each traversal.",
    },
    "thieving": {
        "mode": "interaction",
        "targets": ["npc", "loc"],
        "option_tokens": ["pickpocket", "steal"],
        "strategy": "Select a live Pickpocket or Steal option, pause on danger, and retry after the NPC cooldown.",
    },
    "stat18": {
        "mode": "disabled",
        "targets": [],
        "option_tokens": [],
        "strategy": "This server stat is disabled in the TypeScript PlayerStat configuration.",
        "missing": ["server-side skill definition"],
    },
    "stat19": {
        "mode": "disabled",
        "targets": [],
        "option_tokens": [],
        "strategy": "This server stat is disabled in the TypeScript PlayerStat configuration.",
        "missing": ["server-side skill definition"],
    },
    "runecraft": {
        "mode": "context",
        "targets": ["loc"],
        "option_tokens": ["craft", "rune", "altar"],
        "strategy": "Use essence on a matching altar, resolve the recipe action, and bank or restock as needed.",
        "missing": ["essence/altar policy", "banking/restocking policy"],
    },
}


ENGINE_ACTION_TYPES = {
    "move",
    "chat",
    "teleport",
    "item_op",
    "use_item",
    "use_item_on",
    "button",
    "resume_dialogue",
    "resume_count_dialog",
    "click_side_tab",
    "inventory_button",
    "cast_spell",
    "close_interface",
    "interact",
    "stop",
    "set_run",
}

# This is intentionally explicit: it is the contract audit's answer to
# "does the MCP expose every engine primitive?". High-level wrappers may add
# postcondition validation, but they must still map back to one of these
# authoritative TypeScript actions.
MCP_ENGINE_ACTION_TOOLS = {
    "move": "move_agent",
    "chat": "chat_agent",
    "teleport": "teleport_agent",
    "item_op": "item_op_agent",
    "use_item": "use_item",
    "use_item_on": "use_item_on",
    "button": "press_button",
    "resume_dialogue": "resume_dialogue",
    "resume_count_dialog": "resume_count_dialog",
    "click_side_tab": "click_side_tab",
    "inventory_button": "inventory_button",
    "cast_spell": "cast_spell",
    "close_interface": "close_interface",
    "interact": "interact_agent",
    "stop": "stop_agent",
    "set_run": "set_run",
}

MCP_VALIDATED_ACTIONS = {
    "observe",
    "map_window",
    "travel",
    "move",
    "drop_item",
    "equip_item",
    "combat_loop",
    "recover_health",
    "crafting_loop",
    "skill_step",
    "train_skill",
    "progression_step",
    "quest_plan",
    "run_quest",
    "interact",
    "pickup_object",
    "open_bank",
    "open_shop",
    "buy_item",
    "sell_item",
    "close_shop",
    "deposit_item",
    "withdraw_item",
    "close_bank",
    "use_item",
    "combine",
    "use_item_on_validated",
    "press_button",
    "choose_dialogue_option",
    "resume_dialogue",
    "resume_count_dialog",
    "close_interface",
    "set_run",
    "stop",
}


# Quest plans remain declarative, while the TypeScript server owns the
# content-backed quest registry and persistent quest variables. The runner
# reports exactly which step was resolved or blocked, then compares the final
# engine quest state before claiming quest completion.
QUEST_STEP_KINDS = {
    "travel",
    "interact",
    "discover_interact",
    "pickup",
    "item_on",
    "combat",
    "train",
    "chat",
    "dialogue",
    "resume_dialogue",
    "floor_transition",
    "quest_start",
    "quest_dialogue",
    "quest_item_on",
    "quest_search",
    "combine",
    "combat_loop",
    "crafting_loop",
    "quest_turn_in",
}
QUEST_TEMPLATES: dict[str, dict[str, Any]] = {
    "combat_bounty": {
        "name": "Combat bounty",
        "description": "Find and defeat a bounded number of nearby attackable NPCs.",
        "steps": [{"kind": "combat", "count": 3}],
    },
    "goblin_bounty": {
        "name": "Goblin bounty",
    "description": "Find and defeat three nearby Goblins using live Attack options.",
        "steps": [{"kind": "combat", "npc_name": "Goblin", "count": 3}],
    },
    "sheep": {
        "name": "Sheep Shearer",
        "description": "Start Sheep Shearer, gather and spin 20 balls of wool, then turn them in to Fred.",
        "steps": [
            {
                "kind": "quest_start",
                "quest_id": "sheep",
                "target_kind": "npc",
                "target_name": "Fred the Farmer",
                "option_tokens": ["talk"],
                "dialogue_options": ["I'm looking for a quest.", "Yes okay. I can do that."],
                # The live Lumbridge instance places Fred at (3188, 3270).
                # Keep this coordinate aligned with the authoritative NPC
                # spawn so the route planner can approach the interaction
                # frontier instead of timing out on an empty tile.
                "x": 3188,
                "z": 3270,
                "level": 0,
            },
            {
                "kind": "crafting_loop",
                "skill": "crafting",
                "quest_id": "sheep",
                "target_count": 20,
                "source": {"x": 3195, "z": 3260, "level": 0, "via": [{"x": 3213, "z": 3261, "level": 0}]},
                "station": {"x": 3209, "z": 3212, "level": 1, "approach_x": 3209, "approach_z": 3213},
                "staircase": {
                    "x": 3204,
                    "z": 3207,
                    "approach_x": 3205,
                    "approach_z": 3209,
                    "access": [
                        {
                            "x": 3218,
                            "z": 3218,
                            "level": 0,
                            "target_kind": "loc",
                            "target_name": "Large door",
                            "option_tokens": ["open"],
                            "via": [
                                {"x": 3213, "z": 3256, "level": 0},
                                {"x": 3216, "z": 3246, "level": 0},
                                {"x": 3230, "z": 3232, "level": 0},
                                {"x": 3230, "z": 3227, "level": 0},
                            ],
                        },
                        {
                            "x": 3215,
                            "z": 3212,
                            "level": 0,
                            "target_kind": "loc",
                            "target_name": "Door",
                            "option_tokens": ["open"],
                        },
                    ],
                },
            },
            {
                "kind": "quest_turn_in",
                "quest_id": "sheep",
                "target_kind": "npc",
                "target_name": "Fred the Farmer",
                "option_tokens": ["talk"],
                "x": 3188,
                "z": 3270,
                "level": 0,
            },
        ],
    },
    "gobdip": {
        "name": "Goblin Diplomacy",
        "description": "Start Goblin Diplomacy, prepare orange and blue goblin mail from live inventory inputs, then resolve the generals' armour dispute.",
        "steps": [
            {
                "kind": "quest_start",
                "quest_id": "gobdip",
                "target_kind": "npc",
                "target_name": "Bartender",
                "option_tokens": ["talk"],
                "dialogue_options": ["Not very busy in here today, is it?"],
                "x": 3046,
                "z": 3255,
                "level": 0,
            },
            {
                "kind": "quest_dialogue",
                "quest_id": "gobdip",
                "target_kind": "npc",
                "target_name": "General Wartface",
                "option_tokens": ["talk"],
                "dialogue_options": ["Do you want me to pick an armour colour for you?"],
                "expected_state": 2,
                "x": 2958,
                "z": 3510,
                "level": 0,
            },
            {
                "kind": "combine",
                "quest_id": "gobdip",
                "item_tokens": ["goblin mail"],
                "item_exact": True,
                "use_item_tokens": ["orange dye"],
                "use_item_exact": True,
                "output_tokens": ["orange goblin mail"],
            },
            {
                "kind": "quest_dialogue",
                "quest_id": "gobdip",
                "target_kind": "npc",
                "target_name": "General Wartface",
                "option_tokens": ["talk"],
                "dialogue_options": ["I have some orange armour."],
                "expected_state": 3,
                "x": 2958,
                "z": 3510,
                "level": 0,
            },
            {
                "kind": "combine",
                "quest_id": "gobdip",
                "item_tokens": ["goblin mail"],
                "item_exact": True,
                "use_item_tokens": ["blue dye"],
                "use_item_exact": True,
                "output_tokens": ["blue goblin mail"],
            },
            {
                "kind": "quest_dialogue",
                "quest_id": "gobdip",
                "target_kind": "npc",
                "target_name": "General Wartface",
                "option_tokens": ["talk"],
                "dialogue_options": ["I have some blue armour."],
                "expected_state": 4,
                "x": 2958,
                "z": 3510,
                "level": 0,
            },
            {
                "kind": "quest_dialogue",
                "quest_id": "gobdip",
                "target_kind": "npc",
                "target_name": "General Wartface",
                "option_tokens": ["talk"],
                "dialogue_options": ["Ok I've got brown armour."],
                "expected_state": 5,
                "x": 2958,
                "z": 3510,
                "level": 0,
            },
            {
                "kind": "quest_turn_in",
                "quest_id": "gobdip",
                "target_kind": "npc",
                "target_name": "General Wartface",
                "option_tokens": ["talk"],
                "x": 2958,
                "z": 3510,
                "level": 0,
            },
        ],
    },
    "prince": {
        "name": "Prince Ali Rescue",
        "description": "Start Prince Ali Rescue in Al-Kharid, report to Osman, then continue through the live item and rescue graph.",
        "steps": [
            {
                "kind": "quest_start",
                "quest_id": "prince",
                "target_kind": "npc",
                "target_name": "Hassan",
                "option_tokens": ["talk"],
                "dialogue_options": [
                    "Can I help you? You must need some help here in the desert."
                ],
                "x": 3291,
                "z": 3169,
                "level": 0,
            },
            {
                "kind": "quest_dialogue",
                "quest_id": "prince",
                "target_kind": "npc",
                "target_name": "Osman",
                "option_tokens": ["talk"],
                "dialogue_options": [
                    "What is the first thing I must do?",
                    "Okay, I better go find some things.",
                ],
                "expected_state": 20,
                "x": 3290,
                "z": 3176,
                "level": 0,
            },
        ],
    },
    "fluffs": {
        "name": "Gertrude's Cat",
        "description": "Find Fluffs, satisfy the cat, recover her kitten, and complete the quest with Gertrude.",
        "steps": [
            {
                "kind": "quest_start",
                "quest_id": "fluffs",
                "target_kind": "npc",
                "target_name": "Gertrude",
                "option_tokens": ["talk"],
                "dialogue_options": ["Well, I suppose I could."],
                "x": 3151,
                "z": 3411,
                "level": 0,
            },
            {
                "kind": "quest_dialogue",
                "quest_id": "fluffs",
                "target_kind": "npc",
                "target_name": "Shilop",
                "option_tokens": ["talk"],
                "dialogue_options": ["What will make you tell me?", "Okay then, I'll pay."],
                "expected_state": 2,
                "x": 3220,
                "z": 3434,
                "level": 0,
            },
            {
                "kind": "combine",
                "quest_id": "fluffs",
                "item_tokens": ["raw sardine"],
                "use_item_tokens": ["doogle leaves"],
                "output_tokens": ["seasoned sardine"],
            },
            {
                "kind": "travel",
                "x": 3305,
                "z": 3493,
                "level": 0,
                "run": True,
            },
            {
                "kind": "discover_interact",
                "target_kind": "loc",
                "target_name": "Broken fence",
                "option_tokens": ["climb-over"],
                "accept_movement": True,
            },
            {
                "kind": "floor_transition",
                "x": 3310,
                "z": 3509,
                "approach_x": 3310,
                "approach_z": 3509,
                "from_level": 0,
                "to_level": 1,
                "target_kind": "loc",
                "target_name": "Ladder",
                "option_tokens": ["climb-up"],
            },
            {
                "kind": "quest_item_on",
                "quest_id": "fluffs",
                "item_tokens": ["bucket of milk"],
                "target_kind": "npc",
                "target_name": "Gertrudes cat",
                "expected_state": 3,
            },
            {
                "kind": "quest_item_on",
                "quest_id": "fluffs",
                "item_tokens": ["seasoned sardine"],
                "target_kind": "npc",
                "target_name": "Gertrudes cat",
                "expected_state": 4,
            },
            {
                "kind": "floor_transition",
                "x": 3310,
                "z": 3509,
                "approach_x": 3310,
                "approach_z": 3509,
                "from_level": 1,
                "to_level": 0,
                "target_kind": "loc",
                "target_name": "Ladder",
                "option_tokens": ["climb-down"],
            },
            {
                "kind": "quest_search",
                "quest_id": "fluffs",
                "target_kind": "npc",
                "target_name": "Crate",
                "option_tokens": ["search"],
                "item_tokens": ["fluffs' kitten"],
                "x": 3305,
                "z": 3506,
                "level": 0,
            },
            {
                "kind": "floor_transition",
                "x": 3310,
                "z": 3509,
                "approach_x": 3310,
                "approach_z": 3509,
                "from_level": 0,
                "to_level": 1,
                "target_kind": "loc",
                "target_name": "Ladder",
                "option_tokens": ["climb-up"],
            },
            {
                "kind": "quest_item_on",
                "quest_id": "fluffs",
                "item_tokens": ["fluffs' kitten"],
                "target_kind": "npc",
                "target_name": "Gertrudes cat",
                "expected_state": 5,
            },
            {
                "kind": "floor_transition",
                "x": 3310,
                "z": 3509,
                "approach_x": 3310,
                "approach_z": 3509,
                "from_level": 1,
                "to_level": 0,
                "target_kind": "loc",
                "target_name": "Ladder",
                "option_tokens": ["climb-down"],
            },
            {
                "kind": "quest_turn_in",
                "quest_id": "fluffs",
                "target_kind": "npc",
                "target_name": "Gertrude",
                "option_tokens": ["talk"],
                "x": 3151,
                "z": 3411,
                "level": 0,
            },
        ],
    },
}

# The order is adapted from the 2004Scape community route used for this
# revision. It is deliberately a recommendation, not quest state: the engine
# remains authoritative about whether a quest is actually complete.
GUIDE_ORDER = [
    "cook",
    "sheep",
    "fluffs",
    "gobdip",
    "prince",
    "hetty",
    "imp",
    "runemysteries",
    "fishingcompo",
    "waterfall",
    "murder",
    "mcannon",
    "itexam",
    "elena",
    "doric",
    "priestperil",
    "druidspirit",
    "vampire",
    "haunted",
    "arthur",
    "grail",
    "barcrawl",
    "druid",
    "scorpcatcher",
    "elemental_workshop",
    "drunkmonk",
    "seaslug",
    "desertrescue",
    "tree",
    "demon",
    "grandtree",
    "dragon",
    "zanaris",
    "death",
    "troll",
    "crest",
    "hazeelcult",
    "biohazard",
    "totem",
    "arena",
    "itwatchtower",
    "ikov",
    "junglepotion",
    "blackarmgang",
    "hero",
    "legends",
    "romeojuliet",
    "hunt",
    "blackknight",
    "sheepherder",
    "chompybird",
]

GUIDE_MILESTONES: dict[str, dict[str, Any]] = {
    "cook": {"skills": {"cooking": 4}},
    "sheep": {"skills": {"crafting": 2}},
    "fluffs": {"skills": {"cooking": 12}},
    "gobdip": {"skills": {"crafting": 5}},
    "hetty": {"skills": {"magic": 4}},
    "imp": {"skills": {"magic": 7}},
    "fishingcompo": {"skills": {"fishing": 10}},
    "waterfall": {"skills": {"attack": 30, "strength": 30}},
    "murder": {"skills": {"crafting": 12}},
    "mcannon": {"skills": {"crafting": 15}},
    "doric": {"skills": {"mining": 18}},
    "priestperil": {"skills": {"prayer": 14}},
    "druidspirit": {"skills": {"defence": 13, "hitpoints": 17, "crafting": 24}},
    "vampire": {"skills": {"attack": 33}},
    "arthur": {"skills": {"defence": 32, "prayer": 30}},
    "scorpcatcher": {"skills": {"strength": 33}},
    "elemental_workshop": {"skills": {"crafting": 29, "smithing": 32}},
    "drunkmonk": {"skills": {"woodcutting": 13}},
    "desertrescue": {"skills": {"agility": 26}},
    "tree": {"skills": {"attack": 38}},
    "grandtree": {"skills": {"attack": 42}},
    "dragon": {"skills": {"strength": 40, "defence": 40}},
    "itwatchtower": {"skills": {"magic": 30}},
    "ikov": {"skills": {"ranged": 42, "fletching": 26}},
    "junglepotion": {"skills": {"herblore": 18}},
    "hero": {"skills": {"herblore": 25}},
}

TUTORIAL_GUIDE = [
    {"state": 0, "label": "Basics instructor", "objective": "Talk to the RuneScape Guide."},
    {"state": 1, "label": "Guide dialogue", "objective": "Continue through the RuneScape Guide dialogue."},
    {"state": 4, "label": "Scenery", "objective": "Open the indicated door and continue to the Survival Expert."},
    {"state": 10, "label": "Survival instructor", "objective": "Talk to the Survival Expert and continue through her dialogue."},
    {"state": 20, "label": "Survival instructor", "objective": "Open the inventory and follow the tree/firemaking steps."},
    {"state": 30, "label": "Tutorial woodcutting", "objective": "Use the bronze axe on an indicated tutorial tree."},
    {"state": 40, "label": "Tutorial firemaking", "objective": "Use the tinderbox on the logs to light a fire."},
    {"state": 50, "label": "Skill interface", "objective": "Open the flashing skills tab after gaining experience."},
    {"state": 60, "label": "Survival instructor", "objective": "Talk to the Survival Expert to begin fishing."},
    {"state": 70, "label": "Tutorial fishing", "objective": "Use the tutorial fishing spot with the net."},
    {"state": 80, "label": "Tutorial cooking", "objective": "Use shrimp on the tutorial fire and follow the cooking dialogue."},
    {"state": 90, "label": "Tutorial cooking", "objective": "Catch another shrimp and cook it on the fire."},
    {"state": 120, "label": "Cooking instructor", "objective": "Continue through the cooking instructor and run controls."},
    {"state": 130, "label": "Cooking instructor gate", "objective": "Open the gate and follow the path to the cooking instructor."},
    {"state": 140, "label": "Master Chef", "objective": "Talk to the Master Chef and continue the cooking lesson."},
    {"state": 150, "label": "Bread ingredients", "objective": "Use the bucket of water on the pot of flour."},
    {"state": 160, "label": "Bread dough", "objective": "Use the bread dough on the cooking range."},
    {"state": 170, "label": "Music interface", "objective": "Open the flashing music tab after baking bread."},
    {"state": 180, "label": "Music interface", "objective": "Inspect the music player and continue through the door."},
    {"state": 190, "label": "Run controls", "objective": "Open the flashing player-controls tab."},
    {"state": 195, "label": "Run controls", "objective": "Enable running in the player controls."},
    {"state": 200, "label": "Quest route", "objective": "Run to the next instructor and enter the quest guide house."},
    {"state": 220, "label": "Quest instructor", "objective": "Talk to the quest instructor and inspect the quest journal."},
    {"state": 230, "label": "Quest journal", "objective": "Open the flashing quest-journal tab."},
    {"state": 240, "label": "Quest journal", "objective": "Talk to the quest instructor about the journal."},
    {"state": 250, "label": "Quest route", "objective": "Use the ladder to enter the mining area."},
    {"state": 260, "label": "Mining instructor", "objective": "Talk to the Mining Instructor."},
    {"state": 270, "label": "Prospecting", "objective": "Prospect the first tutorial rock."},
    {"state": 274, "label": "Prospecting", "objective": "Prospect the other tutorial rock."},
    {"state": 275, "label": "Prospecting", "objective": "Prospect the other tutorial rock."},
    {"state": 279, "label": "Prospecting", "objective": "Report the two prospected rocks to the Mining Instructor."},
    {"state": 280, "label": "Prospecting", "objective": "Report the two prospected rocks to the Mining Instructor."},
    {"state": 290, "label": "Mining", "objective": "Mine the first tutorial ore."},
    {"state": 294, "label": "Mining", "objective": "Mine the second tutorial ore."},
    {"state": 295, "label": "Mining", "objective": "Mine the second tutorial ore."},
    {"state": 320, "label": "Smelting", "objective": "Use an ore on the tutorial furnace to make a bronze bar."},
    {"state": 330, "label": "Smithing", "objective": "Talk to the Mining Instructor about the bronze bar."},
    {"state": 340, "label": "Smithing", "objective": "Use the bronze bar on the tutorial anvil and make a dagger."},
    {"state": 350, "label": "Mining exit", "objective": "Open the gate to the combat instructor."},
    {"state": 360, "label": "Combat instructor", "objective": "Equip the starter weapons and complete the combat lessons."},
    {"state": 500, "label": "Account instructor", "objective": "Open the bank and continue through the account lesson."},
    {"state": 550, "label": "Prayer instructor", "objective": "Complete the prayer and interface lessons."},
    {"state": 620, "label": "Magic instructor", "objective": "Open the magic tab and cast Wind Strike."},
    {"state": 1000, "label": "Tutorial complete", "objective": "The player has reached the mainland."},
]


def _normalized_skill(skill: str) -> str:
    normalized = skill.strip().casefold()
    if normalized not in SKILL_PLAYBOOK:
        raise ValueError(f"unknown skill '{skill}'; use skill_catalog to list supported skills")
    return normalized


def _skill_state(observation: dict[str, Any], skill: str) -> dict[str, Any]:
    skills = observation.get("agent", {}).get("skills", {})
    state = skills.get(skill)
    if not isinstance(state, dict):
        raise ValueError(f"skill '{skill}' is not present in the engine observation")
    return {
        "level": state.get("level", 0),
        "baseLevel": state.get("baseLevel", 0),
        "experience": state.get("experience", 0),
    }


def _tutorial_step(observation: dict[str, Any]) -> dict[str, Any]:
    tutorial = observation.get("tutorial") or {}
    state = int(tutorial.get("state", 0))
    current = TUTORIAL_GUIDE[0]
    for milestone in TUTORIAL_GUIDE:
        if milestone["state"] <= state:
            current = milestone
        else:
            break
    next_milestone = next((milestone for milestone in TUTORIAL_GUIDE if milestone["state"] > state), None)
    return {
        "state": state,
        "state_var": tutorial.get("stateVar", "tutorial"),
        "state_var_id": tutorial.get("stateVarId", -1),
        "complete": bool(tutorial.get("complete", state >= 1000)),
        "current": current,
        "next": next_milestone,
    }


def _option_match(options: list[Any], tokens: list[str]) -> tuple[int, str] | None:
    normalized_tokens = [token.casefold() for token in tokens]
    for token in normalized_tokens:
        for index, option in enumerate(options):
            if isinstance(option, str) and option.strip().casefold() == token:
                return index + 1, option
    for index, option in enumerate(options):
        if not isinstance(option, str) or not option.strip():
            continue
        label = option.casefold()
        if any(token in label for token in normalized_tokens):
            return index + 1, option
    return None


def _select_dialogue_button(observation: dict[str, Any], option_text: str) -> dict[str, Any] | None:
    """Select a currently visible dialogue choice by its live rendered text."""

    needle = option_text.strip().casefold()
    if not needle:
        return None
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for button in (observation.get("ui") or {}).get("resumeButtons", []):
        component = button.get("component")
        if not isinstance(component, int):
            continue
        label = next(
            (
                value.strip()
                for value in (button.get("text"), button.get("activeText"), button.get("name"))
                if isinstance(value, str) and value.strip()
            ),
            "",
        )
        normalized = label.casefold()
        if normalized == needle:
            score = 0
        elif needle in normalized:
            score = 1
        else:
            continue
        candidates.append((score, len(label), {"component": component, "text": label, "button": button}))
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1], candidate[2]["component"]))[2] if candidates else None


def _tutorial_location_action(
    observation: dict[str, Any],
    location_id: int,
    option: int,
    label: str,
) -> dict[str, Any] | None:
    """Choose an actual live location origin, not an arbitrary footprint tile."""

    agent_position = observation.get("agent", {}).get("position", {})
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for location in observation.get("nearby", {}).get("locations", []):
        if location.get("id") != location_id:
            continue
        options = location.get("options") or []
        if option < 1 or option > len(options) or not options[option - 1]:
            continue
        position = location.get("position") or {}
        distance = max(
            abs(position.get("x", 0) - agent_position.get("x", 0)),
            abs(position.get("z", 0) - agent_position.get("z", 0)),
        )
        target = {
            "kind": "loc",
            "id": location_id,
            "x": position["x"],
            "z": position["z"],
            "level": position.get("level", 0),
        }
        candidates.append((distance, position["x"] * 10000 + position["z"], {"kind": "interact", "target": target, "option": option, "label": label}))
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2] if candidates else None


def _discover_named_tutorial_entity(
    observation: dict[str, Any],
    collection: str,
    name_tokens: tuple[str, ...],
    option_tokens: list[str],
) -> dict[str, Any] | None:
    """Select the nearest live tutorial entity exposing the requested option."""

    agent_position = observation.get("agent", {}).get("position", {})
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for entity in observation.get("nearby", {}).get(collection, []):
        if not any(token in (entity.get("name") or "").casefold() for token in name_tokens):
            continue
        match = _option_match(entity.get("options") or [], option_tokens)
        if match is None:
            continue
        option, label = match
        position = entity.get("position") or {}
        target_kind = "npc" if collection == "npcs" else "loc"
        target: dict[str, Any] = {"kind": target_kind, "id": entity["id"]}
        if target_kind == "loc":
            target.update({"x": position["x"], "z": position["z"], "level": position.get("level", 0)})
        distance = max(
            abs(position.get("x", 0) - agent_position.get("x", 0)),
            abs(position.get("z", 0) - agent_position.get("z", 0)),
        )
        candidates.append((distance, int(entity["id"]), {
            "kind": "interact",
            "target": target,
            "option": option,
            "label": label,
            "entity": entity,
            "distance": distance,
        }))
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2] if candidates else None


def _discover_skill_action(
    observation: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any] | None:
    agent_position = observation["agent"]["position"]
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for kind in spec["targets"]:
        entities = observation.get("nearby", {}).get({"npc": "npcs", "loc": "locations", "obj": "objects"}[kind], [])
        for entity in entities:
            requested_name = spec.get("target_name")
            if requested_name is not None and (entity.get("name") or "").casefold() != requested_name.casefold():
                continue
            category_requirements = spec.get("category_requirements", {})
            required_category_level = category_requirements.get(entity.get("categoryName"), 0)
            requirement_skill = spec.get("requirement_skill")
            current_level = observation["agent"].get("skills", {}).get(requirement_skill, {}).get("baseLevel", 0)
            if current_level < required_category_level:
                continue
            required_level = max(
                (
                    required
                    for token, required in spec.get("entity_requirements", {}).items()
                    if token in (entity.get("name") or "").casefold()
                ),
                default=0,
            )
            if current_level < required_level:
                continue
            options = entity.get("options") or []
            match = _option_match(options, spec["option_tokens"])
            if match is None:
                continue
            option, label = match
            position = entity["position"]
            distance = max(
                abs(position["x"] - agent_position["x"]),
                abs(position["z"] - agent_position["z"]),
            )
            target: dict[str, Any] = {"kind": kind, "id": entity["id"]}
            if kind in {"loc", "obj"}:
                target.update({"x": position["x"], "z": position["z"], "level": position["level"]})
            # Chebyshev distance alone makes equally-close 2x2 trees tie on
            # their origin tile, even when one origin is actually reachable
            # from the player and the other is behind the tree footprint.
            # Keep the coarse distance preference, then use Manhattan distance
            # as a deterministic reachability-friendly tie breaker.
            score = distance * 10 + abs(position["x"] - agent_position["x"]) + abs(position["z"] - agent_position["z"])
            # Prefer an exact-looking action label over a name-only coincidence,
            # then let distance choose among equivalent resources.
            if label.casefold() in spec["option_tokens"]:
                score -= 1000
            candidates.append((score, entity["id"], {"target": target, "option": option, "label": label, "distance": distance, "entity": entity}))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _discover_tutorial_action(
    observation: dict[str, Any],
    *,
    ranged_attack_started: bool = False,
    magic_attack_started: bool = False,
) -> dict[str, Any] | None:
    """Select only tutorial actions whose live content exposes the option."""

    state = int((observation.get("tutorial") or {}).get("state", 0))
    ui = observation.get("ui") or {}
    if ui.get("activeScriptExecution") == 3:
        resume_buttons = ui.get("resumeButtons") or []
        # Packed interface ids are reused by several tutorial dialogues.  The
        # initial guide dialogue also exposes component 2461, but that option
        # skips the tutorial.  Only choose buttons at live-validated choice
        # milestones; all other script pauses use Continue/resume.
        if resume_buttons and state in {500, 670}:
            choice = resume_buttons[0]
            component = choice.get("component")
            if isinstance(component, int):
                return {
                    "kind": "button",
                    "component": component,
                    "label": choice.get("text") or choice.get("name") or "choose dialogue option",
                }
        return {"kind": "resume_dialogue"}
    if ui.get("activeScript"):
        # Content scripts can suspend while the engine is walking, waiting on
        # a world queue, or finishing a delayed tutorial operation.  Expose
        # that state to the MCP caller instead of misclassifying it as an
        # unsupported tutorial milestone.
        return {"kind": "wait_script", "execution": ui.get("activeScriptExecution")}

    if state == 20:
        return {"kind": "side_tab", "tab": 3, "label": "inventory"}
    if state == 50:
        return {"kind": "side_tab", "tab": 1, "label": "skills"}
    if state == 170:
        return {"kind": "side_tab", "tab": 13, "label": "music"}
    if state == 190:
        return {"kind": "side_tab", "tab": 12, "label": "controls"}
    if state == 195:
        # controls:com_5 is the content-backed "run on" button.  The
        # numeric id comes from the packed interface registry; the engine
        # still validates that it exists and has an IF_BUTTON trigger before
        # executing it.
        return {"kind": "button", "component": 153, "label": "enable run"}
    if state == 200:
        # The tutorial hint is 0_48_48_14_54, which resolves to the
        # quest-house door at (3086, 3126) in the running 2004Scape map.
        return {
            "kind": "interact",
            "target": {"kind": "loc", "id": 3019, "x": 3086, "z": 3126, "level": 0},
            "option": 1,
            "label": "Open quest guide door",
        }
    if state == 230:
        return {"kind": "side_tab", "tab": 2, "label": "quest journal"}
    if state == 250:
        return {
            "kind": "interact",
            "target": {"kind": "loc", "id": 3029, "x": 3088, "z": 3119, "level": 0},
            "option": 1,
            "label": "Climb down to mining",
        }
    if state in {260, 279, 280, 330}:
        return {
            "kind": "interact",
            "target": {"kind": "npc", "id": 5278},
            "option": 1,
            "label": "Talk-to Mining Instructor",
        }
    if state in {270, 274}:
        return _tutorial_location_action(observation, 3043, 2, "Prospect tin rock")
    if state == 275:
        return _tutorial_location_action(observation, 3042, 2, "Prospect copper rock")
    if state in {290, 294}:
        return _tutorial_location_action(observation, 3043, 1, "Mine tin rock")
    if state == 295:
        return _tutorial_location_action(observation, 3042, 1, "Mine copper rock")
    if state == 320:
        ore = _inventory_item(observation, ("copper ore", "tin ore"))
        if ore is None:
            return None
        return {
            "kind": "item_on",
            "skill": "smithing",
            "selected": {
                "inventory": ore["inventory"],
                "slot": ore["item"]["slot"],
                "item": ore["item"],
                "target": {"kind": "loc", "id": 3044, "x": 3078, "z": 9495, "level": 0},
                "label": "Use ore on furnace",
            },
        }
    if state == 340:
        smithing = next(
            (
                inventory
                for inventory in observation.get("agent", {}).get("inventories", [])
                if inventory.get("id") == 101 and inventory.get("items")
            ),
            None,
        )
        if smithing:
            # smithing:column1 is component 1119 and its first product is
            # the tutorial bronze dagger.  The engine validates the live
            # transmitted inventory and IF_BUTTON trigger.
            return {
                "kind": "inventory_button",
                "component": 1119,
                "inventory": 101,
                "slot": 0,
                "option": 1,
                "label": "Make bronze dagger",
            }
        bar = _inventory_item(observation, ("bronze bar",))
        if bar is None:
            return None
        return {
            "kind": "item_on",
            "skill": "smithing",
            "selected": {
                "inventory": bar["inventory"],
                "slot": bar["item"]["slot"],
                "item": bar["item"],
                "target": {"kind": "loc", "id": 2783, "x": 3083, "z": 9499, "level": 0},
                "label": "Use bronze bar on anvil",
            },
        }
    if state == 350:
        return {
            "kind": "interact",
            "target": {"kind": "loc", "id": 3020, "x": 3094, "z": 9503, "level": 0},
            "option": 1,
            "label": "Open mining exit gate",
        }
    if state == 370:
        return {"kind": "side_tab", "tab": 4, "label": "worn items"}
    if state == 380:
        worn = _inventory_with_listener(observation, "wornitems:wear")
        selected = _tutorial_item_op(
            observation,
            ("bronze dagger",),
            ["wield", "wear", "equip"],
            worn.get("id") if worn else None,
        )
        return selected
    if state == 400:
        worn = _inventory_with_listener(observation, "wornitems:wear")
        if worn:
            dagger = next(
                (item for item in worn.get("items", []) if "bronze dagger" in (item.get("name") or "").casefold()),
                None,
            )
            remove = next(
                (
                    listener
                    for listener in worn.get("listeners", [])
                    if (listener.get("name") or "").casefold() == "wornitems:wear"
                ),
                None,
            )
            if dagger and remove:
                match = _option_match(remove.get("options") or [], ["remove"])
                if match:
                    option, label = match
                    return {
                        "kind": "inventory_button",
                        "component": remove["component"],
                        "inventory": worn["id"],
                        "slot": dagger["slot"],
                        "option": option,
                        "label": label,
                    }
        selected = _tutorial_item_op(
            observation,
            ("bronze sword", "wooden shield"),
            ["wield", "wear", "equip"],
            worn.get("id") if worn else None,
        )
        return selected
    if state == 410:
        return {"kind": "side_tab", "tab": 0, "label": "combat options"}
    if state == 440:
        safety = observation.get("safety") or {}
        if safety.get("lowHealth") or safety.get("fleeing"):
            food = _tutorial_food_action(observation)
            if food:
                return food
        agent = observation.get("agent") or {}
        if agent.get("target") is None or agent.get("targetOperation") is None:
            for npc in observation.get("nearby", {}).get("npcs", []):
                if "giant rat" not in (npc.get("name") or "").casefold():
                    continue
                match = _option_match(npc.get("options") or [], ["attack"])
                if match is None:
                    continue
                option, label = match
                return {
                    "kind": "interact",
                    "target": {"kind": "npc", "id": npc["id"]},
                    "option": option,
                    "label": label,
                }
        return {"kind": "wait_combat", "label": "wait for the giant rat fight to finish"}
    if state == 460:
        safety = observation.get("safety") or {}
        if safety.get("lowHealth") or safety.get("fleeing"):
            food = _tutorial_food_action(observation)
            if food:
                return food
        worn = _inventory_with_listener(observation, "wornitems:wear")
        worn_items = {
            (item.get("name") or "").casefold()
            for item in (worn or {}).get("items", [])
        }
        for tokens in (("shortbow",), ("bronze arrow", "bronze arrows")):
            if not any(token in name for name in worn_items for token in tokens):
                selected = _tutorial_item_op(
                    observation,
                    tokens,
                    ["wield", "wear", "equip"],
                    worn.get("id") if worn else None,
                )
                if selected:
                    return selected
        if ranged_attack_started:
            return {"kind": "wait_combat", "label": "wait for the ranged giant rat fight to finish"}
        agent = observation.get("agent") or {}
        if agent.get("target") is not None and agent.get("targetOperation") is not None:
            return {"kind": "wait_combat", "label": "wait for the ranged giant rat fight to finish"}
    if state == 450:
        safety = observation.get("safety") or {}
        if safety.get("lowHealth") or safety.get("fleeing"):
            food = _tutorial_food_action(observation)
            if food:
                return food
    if state == 500:
        return _discover_named_tutorial_entity(observation, "locations", ("bank booth",), ["use"])
    if state == 510:
        return _tutorial_location_action(observation, 3024, 1, "Open tutorial bank exit door")
    if state == 520:
        return _discover_named_tutorial_entity(observation, "npcs", ("financial advisor",), ["talk"])
    if state == 530:
        return _tutorial_location_action(observation, 3025, 1, "Open financial advisor room door")
    if state in {540, 560, 590}:
        # The live map can leave the player just outside the chapel's large
        # double door after the preceding tutorial door script.  Open the
        # observed door before attempting Brother Brace; once it is open the
        # normal collision-aware NPC interaction can reach him.
        if state == 540:
            door = _discover_named_tutorial_entity(observation, "locations", ("large door",), ["open"])
            if door is not None:
                return door
        return _discover_named_tutorial_entity(observation, "npcs", ("brother brace",), ["talk"])
    if state == 550:
        return {"kind": "side_tab", "tab": 5, "label": "prayer"}
    if state == 570:
        return {"kind": "side_tab", "tab": 8, "label": "friends"}
    if state == 580:
        return {"kind": "side_tab", "tab": 9, "label": "ignore"}
    if state == 600:
        return _tutorial_location_action(observation, 3026, 1, "Open chapel exit door")
    if state in {610, 660}:
        return _discover_named_tutorial_entity(observation, "npcs", ("magic instructor", "terrova"), ["talk"])
    if state == 620:
        return {"kind": "side_tab", "tab": 6, "label": "magic"}
    if state in {640, 650}:
        if magic_attack_started and state == 640:
            return {"kind": "wait_combat", "label": "wait for the Wind Strike cast to finish"}
        for npc in observation.get("nearby", {}).get("npcs", []):
            if "chicken" not in (npc.get("name") or "").casefold():
                continue
            return {
                "kind": "cast_spell",
                "component_name": "magic:wind_strike",
                "target": {"kind": "npc", "id": npc["id"]},
                "label": "Cast Wind Strike at chicken",
                "entity": npc,
            }
    if state == 30:
        selected = _discover_skill_action(observation, SKILL_PLAYBOOK["woodcutting"])
        return {"kind": "skill", "skill": "woodcutting", "selected": selected} if selected else None
    if state == 40:
        selected = _discover_inventory_skill_action(observation, "firemaking")
        return {"kind": "skill", "skill": "firemaking", "selected": selected} if selected else None
    if state == 70:
        selected = _discover_tutorial_fishing_action(observation)
        return {"kind": "skill", "skill": "fishing", "selected": selected} if selected else None
    if state == 150:
        selected = _discover_tutorial_mix_action(observation)
        return {"kind": "item_pair", "skill": "cooking", "selected": selected} if selected else None
    if state == 160:
        selected = _discover_tutorial_range_action(observation)
        return {"kind": "item_on", "skill": "cooking", "selected": selected} if selected else None
    if state in {80, 90}:
        selected = _discover_tutorial_cooking_action(observation)
        if selected:
            return {"kind": "item_on", "skill": "cooking", "selected": selected}
        selected = _discover_tutorial_fishing_action(observation)
        return {"kind": "skill", "skill": "fishing", "selected": selected} if selected else None

    nearby = observation.get("nearby") or {}
    if state in {360, 390, 450}:
        entities = nearby.get("npcs", [])
        names = ("combat instructor", "vannaka")
        target_kind = "npc"
        option_tokens = ["talk"]
    elif state in {420}:
        entities = nearby.get("locations", [])
        names = ("gate", "door")
        target_kind = "loc"
        option_tokens = ["open"]
    elif state in {430, 460}:
        entities = nearby.get("npcs", [])
        names = ("giant rat",)
        target_kind = "npc"
        option_tokens = ["attack"]
    elif state == 470:
        entities = nearby.get("locations", [])
        names = ("ladder",)
        target_kind = "loc"
        option_tokens = ["climb"]
    elif state in {0, 1}:
        entities = nearby.get("npcs", [])
        names = ("runescape guide",)
        target_kind = "npc"
        option_tokens = ["talk"]
    elif state == 4:
        entities = nearby.get("locations", [])
        names = ("door", "gate")
        target_kind = "loc"
        option_tokens = ["open"]
    elif state == 120:
        entities = nearby.get("locations", [])
        names = ("gate",)
        target_kind = "loc"
        option_tokens = ["open"]
    elif state == 130:
        entities = nearby.get("locations", [])
        names = ("door",)
        target_kind = "loc"
        option_tokens = ["open"]
    elif state == 140:
        entities = nearby.get("npcs", [])
        names = ("master chef", "cooking instructor", "chef")
        target_kind = "npc"
        option_tokens = ["talk"]
    elif state in {220, 240}:
        entities = nearby.get("npcs", [])
        names = ("quest guide",)
        target_kind = "npc"
        option_tokens = ["talk"]
    elif state == 180:
        entities = nearby.get("locations", [])
        names = ("door",)
        target_kind = "loc"
        option_tokens = ["open"]
    elif state in {10, 60}:
        entities = nearby.get("npcs", [])
        names = ("survival expert",)
        target_kind = "npc"
        option_tokens = ["talk"]
    else:
        return None

    position = observation.get("agent", {}).get("position", {})
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for entity in entities:
        if state == 180 and entity.get("id") != 3018:
            continue
        if state == 420 and entity.get("categoryName") != "rat_pit_cage":
            continue
        name = (entity.get("name") or "").casefold()
        if not any(token in name for token in names):
            continue
        match = _option_match(entity.get("options") or [], option_tokens)
        if match is None:
            continue
        option, label = match
        target_position = entity.get("position") or {}
        target: dict[str, Any] = {"kind": target_kind, "id": entity["id"]}
        if target_kind == "loc":
            target.update(
                {
                    "x": target_position["x"],
                    "z": target_position["z"],
                    "level": target_position["level"],
                }
            )
        distance = max(
            abs(target_position.get("x", 0) - position.get("x", 0)),
            abs(target_position.get("z", 0) - position.get("z", 0)),
        )
        candidates.append(
            (
                distance,
                int(entity["id"]),
                {"kind": target_kind, "target": target, "option": option, "label": label, "entity": entity, "distance": distance},
            )
        )
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2] if candidates else None


def _inventory_item(observation: dict[str, Any], tokens: tuple[str, ...]) -> dict[str, Any] | None:
    for inventory in observation.get("agent", {}).get("inventories", []):
        for item in inventory.get("items", []):
            name = (item.get("name") or "").casefold()
            if any(token in name for token in tokens):
                return {"inventory": inventory["id"], "item": item}
    return None


def _inventory_item_exact(observation: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any] | None:
    """Find an inventory item by its rendered live name, without substring ambiguity."""

    wanted = {name.casefold() for name in names}
    for inventory in observation.get("agent", {}).get("inventories", []):
        for item in inventory.get("items", []):
            if (item.get("name") or "").casefold() in wanted:
                return {"inventory": inventory["id"], "item": item}
    return None


def _inventory_count_exact(observation: dict[str, Any], name: str) -> int:
    wanted = name.casefold()
    return sum(
        int(item.get("count", 0))
        for inventory in observation.get("agent", {}).get("inventories", [])
        for item in inventory.get("items", [])
        if (item.get("name") or "").casefold() == wanted
    )


def _inventory_with_listener(observation: dict[str, Any], listener_name: str) -> dict[str, Any] | None:
    expected = listener_name.casefold()
    for inventory in observation.get("agent", {}).get("inventories", []):
        for listener in inventory.get("listeners", []):
            if (listener.get("name") or "").casefold() == expected:
                return inventory
    return None


def _inventory_item_excluding(
    observation: dict[str, Any],
    tokens: tuple[str, ...],
    excluded_inventory: int | None,
) -> dict[str, Any] | None:
    for inventory in observation.get("agent", {}).get("inventories", []):
        if inventory.get("id") == excluded_inventory:
            continue
        for item in inventory.get("items", []):
            name = (item.get("name") or "").casefold()
            if any(token in name for token in tokens):
                return {"inventory": inventory["id"], "item": item}
    return None


def _item_option(item: dict[str, Any], tokens: list[str]) -> tuple[int, str] | None:
    return _option_match(item.get("options") or [], tokens)


def _tutorial_item_op(
    observation: dict[str, Any],
    tokens: tuple[str, ...],
    option_tokens: list[str],
    excluded_inventory: int | None = None,
) -> dict[str, Any] | None:
    selected = _inventory_item_excluding(observation, tokens, excluded_inventory)
    if selected is None:
        return None
    match = _item_option(selected["item"], option_tokens)
    if match is None:
        return None
    option, label = match
    return {
        "kind": "item_op",
        "inventory": selected["inventory"],
        "slot": selected["item"]["slot"],
        "option": option,
        "label": label,
        "item": selected["item"],
    }


def _tutorial_food_action(observation: dict[str, Any]) -> dict[str, Any] | None:
    return _tutorial_item_op(
        observation,
        ("shrimps", "shrimp", "bread", "burnt fish"),
        ["eat", "drink"],
    )


def _discover_tutorial_rat_gate_action(observation: dict[str, Any]) -> dict[str, Any] | None:
    position = observation.get("agent", {}).get("position", {})
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for location in observation.get("nearby", {}).get("locations", []):
        if location.get("categoryName") != "rat_pit_cage":
            continue
        match = _option_match(location.get("options") or [], ["open"])
        if match is None:
            continue
        option, label = match
        target_position = location.get("position") or {}
        distance = max(
            abs(target_position.get("x", 0) - position.get("x", 0)),
            abs(target_position.get("z", 0) - position.get("z", 0)),
        )
        target = {
            "kind": "loc",
            "id": location["id"],
            "x": target_position["x"],
            "z": target_position["z"],
            "level": target_position.get("level", 0),
        }
        candidates.append(
            (
                distance,
                int(location["id"]),
                {"kind": "loc", "target": target, "option": option, "label": label, "entity": location},
            )
        )
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2] if candidates else None


def _discover_inventory_skill_action(
    observation: dict[str, Any],
    skill: str,
) -> dict[str, Any] | None:
    if skill != "firemaking":
        return None
    tinderbox = _inventory_item(observation, ("tinderbox",))
    logs = _inventory_item(observation, ("logs",))
    if tinderbox is None or logs is None:
        return None
    return {
        "skill": skill,
        "action": "use_item",
        "inventory": tinderbox["inventory"],
        "slot": tinderbox["item"]["slot"],
        "use_inventory": logs["inventory"],
        "use_slot": logs["item"]["slot"],
        "tinderbox": tinderbox["item"],
        "logs": logs["item"],
    }


def _discover_context_skill_action(
    observation: dict[str, Any],
    skill: str,
    excluded_target_ids: set[int] | None = None,
) -> dict[str, Any] | None:
    """Discover a bounded context-skill action from live items and entities.

    This intentionally starts with the simplest content-backed crafting chain
    in the 2004Scape data. The returned target and item slots come only from
    the current observation; the engine remains responsible for resolving the
    item-on-target script and the MCP validates the resulting state mutation.
    """

    if skill == "cooking":
        # Keep this registry deliberately data-shaped: the engine remains the
        # authority on whether a particular food can be cooked at a particular
        # station, while the MCP selects only items and stations observed live.
        # The output names include both successful and burnt products because
        # both are authoritative inventory results of a cooking attempt.
        cooking_recipes = (
            ("Raw shrimps", ("Shrimps", "Burnt fish")),
            ("Raw sardine", ("Sardine", "Burnt fish")),
            ("Raw anchovies", ("Anchovies", "Burnt fish")),
            ("Raw herring", ("Herring", "Burnt fish")),
            ("Raw mackerel", ("Mackerel", "Burnt fish")),
            ("Raw trout", ("Trout", "Burnt fish")),
            ("Raw salmon", ("Salmon", "Burnt fish")),
            ("Raw pike", ("Pike", "Burnt fish")),
            ("Raw cod", ("Cod", "Burnt fish")),
            ("Raw tuna", ("Tuna", "Burnt fish")),
            ("Raw lobster", ("Lobster", "Burnt lobster")),
            ("Raw swordfish", ("Swordfish", "Burnt swordfish")),
            ("Raw chicken", ("Cooked chicken", "Burnt chicken")),
            ("Raw beef", ("Cooked meat", "Burnt meat")),
            ("Raw meat", ("Cooked meat", "Burnt meat")),
            ("Raw rat meat", ("Cooked meat", "Burnt meat")),
        )
        carried = next(
            (inventory for inventory in observation.get("agent", {}).get("inventories", []) if inventory.get("id") == 93),
            None,
        )
        raw = None
        if carried is not None:
            for raw_name, _ in cooking_recipes:
                raw_item = next(
                    (
                        item
                        for item in carried.get("items", [])
                        if (item.get("name") or "").casefold() == raw_name.casefold()
                    ),
                    None,
                )
                if raw_item is not None:
                    raw = {"inventory": carried["id"], "item": raw_item}
                    break
        if raw is None:
            return None

        position = observation.get("agent", {}).get("position", {})
        stations = [
            location
            for location in observation.get("nearby", {}).get("locations", [])
            if (location.get("categoryName") or "").casefold() in {"cooking_oven", "cooking_fire"}
            or any(
                token in (location.get("name") or "").casefold()
                for token in ("cooking range", "range", "fireplace", "fire")
            )
        ]
        station = min(
            stations,
            key=lambda location: (
                max(
                    abs(location.get("position", {}).get("x", 0) - position.get("x", 0)),
                    abs(location.get("position", {}).get("z", 0) - position.get("z", 0)),
                ),
                int(location.get("id", 0)),
            ),
            default=None,
        )
        if station is None:
            return None
        station_position = station.get("position") or {}
        recipe = next(
            recipe
            for recipe in cooking_recipes
            if recipe[0].casefold() == (raw["item"].get("name") or "").casefold()
        )
        return {
            "kind": "item_on",
            "skill": skill,
            "recipe": "cook_raw_food",
            "raw_name": recipe[0],
            "output_names": recipe[1],
            "inventory": raw["inventory"],
            "slot": raw["item"]["slot"],
            "item": raw["item"],
            "target": {
                "kind": "loc",
                "id": station["id"],
                "x": station_position["x"],
                "z": station_position["z"],
                "level": station_position.get("level", 0),
            },
            "station": station,
        }

    if skill != "crafting":
        return None

    excluded_target_ids = excluded_target_ids or set()

    position = observation.get("agent", {}).get("position", {})
    locations = observation.get("nearby", {}).get("locations", [])
    wheels = [
        location
        for location in locations
        if (location.get("categoryName") or "").casefold() == "spinning_wheel"
        or "spinning wheel" in (location.get("name") or "").casefold()
    ]
    wheel = min(
        wheels,
        key=lambda location: (
            max(
                abs(location.get("position", {}).get("x", 0) - position.get("x", 0)),
                abs(location.get("position", {}).get("z", 0) - position.get("z", 0)),
            ),
            int(location.get("id", 0)),
        ),
        default=None,
    )
    wool = _inventory_item_exact(observation, ("Wool",))
    if wool is not None and wheel is not None:
        wheel_position = wheel.get("position") or {}
        return {
            "kind": "item_on",
            "skill": skill,
            "recipe": "wool_to_ball_of_wool",
            "inventory": wool["inventory"],
            "slot": wool["item"]["slot"],
            "item": wool["item"],
            "target": {
                "kind": "loc",
                "id": wheel["id"],
                "x": wheel_position["x"],
                "z": wheel_position["z"],
                "level": wheel_position.get("level", 0),
            },
            "station": wheel,
        }

    shears = _inventory_item_exact(observation, ("Shears",))
    if shears is None:
        return None
    sheep = [
        npc
        for npc in observation.get("nearby", {}).get("npcs", [])
        if (
            (npc.get("name") or "").casefold() == "sheep"
            and npc.get("type") == 43
            and int(npc.get("id", 0)) not in excluded_target_ids
        )
    ]
    if not sheep:
        return None
    target = min(
        sheep,
        key=lambda npc: (
            max(
                abs(npc.get("position", {}).get("x", 0) - position.get("x", 0)),
                abs(npc.get("position", {}).get("z", 0) - position.get("z", 0)),
            ),
            int(npc.get("id", 0)),
        ),
    )
    return {
        "kind": "item_on",
        "skill": skill,
        "recipe": "shear_sheep",
        "inventory": shears["inventory"],
        "slot": shears["item"]["slot"],
        "item": shears["item"],
        "target": {"kind": "npc", "id": target["id"]},
        "entity": target,
    }


def _discover_tutorial_fishing_action(observation: dict[str, Any]) -> dict[str, Any] | None:
    """Discover the level-1 fishing option used only by the tutorial island."""

    tutorial_fishing = {**SKILL_PLAYBOOK["fishing"], "category_requirements": {}}
    return _discover_skill_action(observation, tutorial_fishing)


def _discover_tutorial_cooking_action(observation: dict[str, Any]) -> dict[str, Any] | None:
    """Select raw tutorial shrimp and a live cooking-fire location."""

    shrimp = _inventory_item(observation, ("raw shrimps", "raw shrimp"))
    if shrimp is None:
        return None
    position = observation.get("agent", {}).get("position", {})
    fires = [
        location
        for location in observation.get("nearby", {}).get("locations", [])
        if location.get("categoryName") == "cooking_fire"
    ]
    if not fires:
        return None
    fire = min(
        fires,
        key=lambda location: (
            max(
                abs(location.get("position", {}).get("x", 0) - position.get("x", 0)),
                abs(location.get("position", {}).get("z", 0) - position.get("z", 0)),
            ),
            int(location.get("id", 0)),
        ),
    )
    fire_position = fire["position"]
    return {
        "skill": "cooking",
        "action": "use_item_on",
        "inventory": shrimp["inventory"],
        "slot": shrimp["item"]["slot"],
        "item": shrimp["item"],
        "target": {
            "kind": "loc",
            "id": fire["id"],
            "x": fire_position["x"],
            "z": fire_position["z"],
            "level": fire_position["level"],
        },
        "fire": fire,
    }


def _discover_tutorial_mix_action(observation: dict[str, Any]) -> dict[str, Any] | None:
    """Select the tutorial bucket and pot pair for making bread dough."""

    water = _inventory_item(observation, ("bucket of water", "bucket_water"))
    flour = _inventory_item(observation, ("pot of flour", "newbie_pot_flour"))
    if water is None or flour is None:
        return None
    return {
        "skill": "cooking",
        "action": "use_item",
        "inventory": water["inventory"],
        "slot": water["item"]["slot"],
        "use_inventory": flour["inventory"],
        "use_slot": flour["item"]["slot"],
        "water": water["item"],
        "flour": flour["item"],
    }


def _discover_tutorial_range_action(observation: dict[str, Any]) -> dict[str, Any] | None:
    """Select tutorial bread dough and the live cooking range."""

    dough = _inventory_item(observation, ("bread dough",))
    if dough is None:
        return None
    position = observation.get("agent", {}).get("position", {})
    ranges = [
        location
        for location in observation.get("nearby", {}).get("locations", [])
        if "range" in (location.get("name") or "").casefold()
        or location.get("categoryName") == "cooking_oven"
    ]
    if not ranges:
        return None
    cooking_range = min(
        ranges,
        key=lambda location: (
            max(
                abs(location.get("position", {}).get("x", 0) - position.get("x", 0)),
                abs(location.get("position", {}).get("z", 0) - position.get("z", 0)),
            ),
            int(location.get("id", 0)),
        ),
    )
    range_position = cooking_range["position"]
    return {
        "skill": "cooking",
        "action": "use_item_on",
        "inventory": dough["inventory"],
        "slot": dough["item"]["slot"],
        "item": dough["item"],
        "target": {
            "kind": "loc",
            "id": cooking_range["id"],
            "x": range_position["x"],
            "z": range_position["z"],
            "level": range_position["level"],
        },
        "range": cooking_range,
    }


def _training_result(
    *,
    status: str,
    skill: str,
    initial: dict[str, Any],
    final_observation: dict[str, Any],
    steps: list[dict[str, Any]],
    message: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "skill": skill,
        "initial": initial,
        "final": _skill_state(final_observation, skill),
        "safety": final_observation.get("safety"),
        "steps": steps,
    }
    if message:
        result["message"] = message
    return result


def _quest_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(steps, list) or not steps or len(steps) > 100:
        raise ValueError("steps must be a non-empty list of at most 100 quest steps")
    normalized: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict) or raw_step.get("kind") not in QUEST_STEP_KINDS:
            raise ValueError(f"step {index} must have one of: {', '.join(sorted(QUEST_STEP_KINDS))}")
        kind = raw_step["kind"]
        step = dict(raw_step)
        if kind == "travel":
            for field in ("x", "z"):
                if not isinstance(step.get(field), int):
                    raise ValueError(f"step {index}.{field} must be an integer")
            if not isinstance(step.get("level", 0), int) or not 0 <= step.get("level", 0) <= 3:
                raise ValueError(f"step {index}.level must be between 0 and 3")
            if step.get("via") is not None:
                if not isinstance(step["via"], list) or not all(isinstance(point, dict) for point in step["via"]):
                    raise ValueError(f"step {index}.via must be a list of coordinate objects")
                for point_index, point in enumerate(step["via"]):
                    for field in ("x", "z"):
                        if not isinstance(point.get(field), int):
                            raise ValueError(f"step {index}.via[{point_index}].{field} must be an integer")
                    if not isinstance(point.get("level", step.get("level", 0)), int) or not 0 <= point.get("level", step.get("level", 0)) <= 3:
                        raise ValueError(f"step {index}.via[{point_index}].level must be between 0 and 3")
        elif kind == "interact":
            if step.get("target_kind") not in {"npc", "player", "loc", "obj"}:
                raise ValueError(f"step {index}.target_kind is invalid")
            if not isinstance(step.get("option"), int) or not 1 <= step["option"] <= 5:
                raise ValueError(f"step {index}.option must be between 1 and 5")
            if step["target_kind"] == "player":
                if not isinstance(step.get("target_username"), str) or not step["target_username"].strip():
                    raise ValueError(f"step {index}.target_username is required")
            else:
                if not isinstance(step.get("target_id"), int):
                    raise ValueError(f"step {index}.target_id is required")
            if step["target_kind"] in {"loc", "obj"} and (
                not isinstance(step.get("x"), int) or not isinstance(step.get("z"), int)
            ):
                raise ValueError(f"step {index}.x and step {index}.z are required for loc/obj targets")
        elif kind == "discover_interact":
            if not isinstance(step.get("option_tokens"), list) or not step["option_tokens"] or not all(isinstance(token, str) for token in step["option_tokens"]):
                raise ValueError(f"step {index}.option_tokens must be a non-empty list of strings")
            if step.get("target_kind") is not None and step["target_kind"] not in {"npc", "loc", "obj"}:
                raise ValueError(f"step {index}.target_kind is invalid")
            if step.get("target_name") is not None and (
                not isinstance(step["target_name"], str) or not step["target_name"].strip()
            ):
                raise ValueError(f"step {index}.target_name must be a non-empty string")
            if "accept_movement" in step and not isinstance(step["accept_movement"], bool):
                raise ValueError(f"step {index}.accept_movement must be a boolean")
        elif kind == "pickup":
            if step.get("object_id") is None and (not isinstance(step.get("object_name"), str) or not step["object_name"].strip()):
                raise ValueError(f"step {index}.object_id or object_name is required")
            if step.get("object_id") is not None and not isinstance(step["object_id"], int):
                raise ValueError(f"step {index}.object_id must be an integer")
        elif kind == "item_on":
            if not isinstance(step.get("item_tokens"), list) or not step["item_tokens"] or not all(isinstance(token, str) for token in step["item_tokens"]):
                raise ValueError(f"step {index}.item_tokens must be a non-empty list of strings")
            if step.get("target_kind") not in {"npc", "player", "loc", "obj"}:
                raise ValueError(f"step {index}.target_kind is invalid")
            if step["target_kind"] == "player":
                if not isinstance(step.get("target_username"), str) or not step["target_username"].strip():
                    raise ValueError(f"step {index}.target_username is required")
            elif step.get("target_id") is not None and not isinstance(step["target_id"], int):
                raise ValueError(f"step {index}.target_id must be an integer")
            elif step.get("target_name") is None and step.get("target_id") is None:
                raise ValueError(f"step {index}.target_id or target_name is required")
            if step["target_kind"] in {"loc", "obj"} and step.get("target_id") is not None and (
                not isinstance(step.get("x"), int) or not isinstance(step.get("z"), int)
            ):
                raise ValueError(f"step {index}.x and step {index}.z are required for loc/obj targets")
            if step.get("target_name") is not None and not isinstance(step["target_name"], str):
                raise ValueError(f"step {index}.target_name must be a string")
        elif kind == "combat":
            if not isinstance(step.get("count", 1), int) or not 1 <= step.get("count", 1) <= 100:
                raise ValueError(f"step {index}.count must be between 1 and 100")
            if step.get("npc_name") is not None and not isinstance(step["npc_name"], str):
                raise ValueError(f"step {index}.npc_name must be a string")
        elif kind == "train":
            _normalized_skill(str(step.get("skill", "")))
            if not isinstance(step.get("target_level"), int) or not 1 <= step["target_level"] <= 99:
                raise ValueError(f"step {index}.target_level must be between 1 and 99")
        elif kind == "chat":
            if not isinstance(step.get("message"), str) or not step["message"].strip():
                raise ValueError(f"step {index}.message is required")
        elif kind == "dialogue":
            if not isinstance(step.get("text"), str) or not step["text"].strip():
                raise ValueError(f"step {index}.text is required")
        elif kind == "resume_dialogue":
            pass
        elif kind == "floor_transition":
            for field in ("x", "z", "approach_x", "approach_z"):
                if not isinstance(step.get(field), int):
                    raise ValueError(f"step {index}.{field} must be an integer")
            for field in ("from_level", "to_level"):
                if not isinstance(step.get(field), int) or not 0 <= step[field] <= 3:
                    raise ValueError(f"step {index}.{field} must be between 0 and 3")
            if step["from_level"] == step["to_level"]:
                raise ValueError(f"step {index} must change floors")
            if not isinstance(step.get("option_tokens"), list) or not step["option_tokens"] or not all(
                isinstance(token, str) and token.strip() for token in step["option_tokens"]
            ):
                raise ValueError(f"step {index}.option_tokens must be a non-empty list of strings")
            if step.get("target_kind", "loc") not in {"loc", "obj"}:
                raise ValueError(f"step {index}.target_kind must be loc or obj")
            if not isinstance(step.get("target_name", "Staircase"), str) or not step.get("target_name", "Staircase").strip():
                raise ValueError(f"step {index}.target_name must be a non-empty string")
        elif kind == "quest_start":
            if not isinstance(step.get("quest_id"), str) or not step["quest_id"].strip():
                raise ValueError(f"step {index}.quest_id is required")
            if step.get("target_kind") not in {"npc", "loc", "obj"}:
                raise ValueError(f"step {index}.target_kind is invalid")
            if not isinstance(step.get("target_name"), str) or not step["target_name"].strip():
                raise ValueError(f"step {index}.target_name is required")
            if not isinstance(step.get("option_tokens"), list) or not step["option_tokens"] or not all(
                isinstance(token, str) and token.strip() for token in step["option_tokens"]
            ):
                raise ValueError(f"step {index}.option_tokens must be a non-empty list of strings")
            if not isinstance(step.get("dialogue_options"), list) or not step["dialogue_options"] or not all(
                isinstance(option, str) and option.strip() for option in step["dialogue_options"]
            ):
                raise ValueError(f"step {index}.dialogue_options must be a non-empty list of strings")
            if (step.get("x") is None) != (step.get("z") is None) or any(
                field in step and not isinstance(step[field], int) for field in ("x", "z", "level")
            ):
                raise ValueError(f"step {index}.x and step {index}.z must be integer target coordinates when supplied")
            if "level" in step and not 0 <= step["level"] <= 3:
                raise ValueError(f"step {index}.level must be between 0 and 3")
        elif kind == "quest_dialogue":
            if not isinstance(step.get("quest_id"), str) or not step["quest_id"].strip():
                raise ValueError(f"step {index}.quest_id is required")
            if step.get("target_kind") not in {"npc", "loc", "obj"}:
                raise ValueError(f"step {index}.target_kind is invalid")
            if not isinstance(step.get("target_name"), str) or not step["target_name"].strip():
                raise ValueError(f"step {index}.target_name is required")
            if not isinstance(step.get("option_tokens"), list) or not step["option_tokens"] or not all(
                isinstance(token, str) and token.strip() for token in step["option_tokens"]
            ):
                raise ValueError(f"step {index}.option_tokens must be a non-empty list of strings")
            if not isinstance(step.get("dialogue_options"), list) or not step["dialogue_options"] or not all(
                isinstance(option, str) and option.strip() for option in step["dialogue_options"]
            ):
                raise ValueError(f"step {index}.dialogue_options must be a non-empty list of strings")
            if not isinstance(step.get("expected_state"), int) or step["expected_state"] < 1:
                raise ValueError(f"step {index}.expected_state must be a positive integer")
            if (step.get("x") is None) != (step.get("z") is None) or any(
                field in step and not isinstance(step[field], int) for field in ("x", "z", "level")
            ):
                raise ValueError(f"step {index}.x and step {index}.z must be integer target coordinates when supplied")
            if "level" in step and not 0 <= step["level"] <= 3:
                raise ValueError(f"step {index}.level must be between 0 and 3")
        elif kind == "quest_item_on":
            if not isinstance(step.get("quest_id"), str) or not step["quest_id"].strip():
                raise ValueError(f"step {index}.quest_id is required")
            if not isinstance(step.get("item_tokens"), list) or not step["item_tokens"] or not all(
                isinstance(token, str) and token.strip() for token in step["item_tokens"]
            ):
                raise ValueError(f"step {index}.item_tokens must be a non-empty list of strings")
            if step.get("target_kind") not in {"npc", "loc", "obj"}:
                raise ValueError(f"step {index}.target_kind is invalid")
            if not isinstance(step.get("target_name"), str) or not step["target_name"].strip():
                raise ValueError(f"step {index}.target_name is required")
            if not isinstance(step.get("expected_state"), int) or step["expected_state"] < 1:
                raise ValueError(f"step {index}.expected_state must be a positive integer")
        elif kind == "quest_search":
            if not isinstance(step.get("quest_id"), str) or not step["quest_id"].strip():
                raise ValueError(f"step {index}.quest_id is required")
            if step.get("target_kind") not in {"npc", "loc", "obj"}:
                raise ValueError(f"step {index}.target_kind is invalid")
            if not isinstance(step.get("target_name"), str) or not step["target_name"].strip():
                raise ValueError(f"step {index}.target_name is required")
            for field in ("option_tokens", "item_tokens"):
                if not isinstance(step.get(field), list) or not step[field] or not all(
                    isinstance(token, str) and token.strip() for token in step[field]
                ):
                    raise ValueError(f"step {index}.{field} must be a non-empty list of strings")
            if (step.get("x") is None) != (step.get("z") is None) or any(
                field in step and not isinstance(step[field], int) for field in ("x", "z", "level")
            ):
                raise ValueError(f"step {index}.x and step {index}.z must be integer target coordinates when supplied")
            if "level" in step and not 0 <= step["level"] <= 3:
                raise ValueError(f"step {index}.level must be between 0 and 3")
        elif kind == "combine":
            if not isinstance(step.get("item_tokens"), list) or not step["item_tokens"] or not all(
                isinstance(token, str) and token.strip() for token in step["item_tokens"]
            ):
                raise ValueError(f"step {index}.item_tokens must be a non-empty list of strings")
            if not isinstance(step.get("use_item_tokens"), list) or not step["use_item_tokens"] or not all(
                isinstance(token, str) and token.strip() for token in step["use_item_tokens"]
            ):
                raise ValueError(f"step {index}.use_item_tokens must be a non-empty list of strings")
            if not isinstance(step.get("output_tokens"), list) or not step["output_tokens"] or not all(
                isinstance(token, str) and token.strip() for token in step["output_tokens"]
            ):
                raise ValueError(f"step {index}.output_tokens must be a non-empty list of strings")
        elif kind == "crafting_loop":
            if step.get("skill", "crafting") != "crafting":
                raise ValueError(f"step {index}.skill must be crafting")
            if step.get("quest_id") is not None and (not isinstance(step["quest_id"], str) or not step["quest_id"].strip()):
                raise ValueError(f"step {index}.quest_id must be a non-empty string when supplied")
            if not isinstance(step.get("target_count"), int) or not 1 <= step["target_count"] <= 20_000:
                raise ValueError(f"step {index}.target_count must be between 1 and 20000")
            if step.get("count_mode", "inventory") not in {"inventory", "produced"}:
                raise ValueError(f"step {index}.count_mode must be inventory or produced")
            if step.get("discard_tokens") is not None and (
                not isinstance(step["discard_tokens"], list)
                or not step["discard_tokens"]
                or not all(isinstance(token, str) and token.strip() for token in step["discard_tokens"])
            ):
                raise ValueError(f"step {index}.discard_tokens must be a non-empty list of strings when supplied")
            for name in ("source", "station", "staircase"):
                if not isinstance(step.get(name), dict):
                    raise ValueError(f"step {index}.{name} is required")
            for name in ("source", "station"):
                location = step[name]
                for field in ("x", "z", "level"):
                    if not isinstance(location.get(field), int):
                        raise ValueError(f"step {index}.{name}.{field} must be an integer")
                if not 0 <= location["level"] <= 3:
                    raise ValueError(f"step {index}.{name}.level must be between 0 and 3")
                for field in ("approach_x", "approach_z"):
                    if field in location and not isinstance(location[field], int):
                        raise ValueError(f"step {index}.{name}.{field} must be an integer")
                if location.get("via") is not None:
                    if not isinstance(location["via"], list) or not all(isinstance(point, dict) for point in location["via"]):
                        raise ValueError(f"step {index}.{name}.via must be a list of coordinate objects")
                    for point_index, point in enumerate(location["via"]):
                        for field in ("x", "z"):
                            if not isinstance(point.get(field), int):
                                raise ValueError(f"step {index}.{name}.via[{point_index}].{field} must be an integer")
                        point_level = point.get("level", location["level"])
                        if not isinstance(point_level, int) or not 0 <= point_level <= 3:
                            raise ValueError(f"step {index}.{name}.via[{point_index}].level must be between 0 and 3")
            staircase = step["staircase"]
            for field in ("x", "z", "approach_x", "approach_z"):
                if not isinstance(staircase.get(field), int):
                    raise ValueError(f"step {index}.staircase.{field} must be an integer")
            if staircase.get("access") is not None:
                if not isinstance(staircase["access"], list) or not all(isinstance(point, dict) for point in staircase["access"]):
                    raise ValueError(f"step {index}.staircase.access must be a list of access checkpoints")
                for point_index, point in enumerate(staircase["access"]):
                    for field in ("x", "z"):
                        if not isinstance(point.get(field), int):
                            raise ValueError(f"step {index}.staircase.access[{point_index}].{field} must be an integer")
                    if not isinstance(point.get("level", staircase.get("level", 0)), int):
                        raise ValueError(f"step {index}.staircase.access[{point_index}].level must be an integer")
                    if point.get("target_kind", "loc") not in {"loc", "obj"}:
                        raise ValueError(f"step {index}.staircase.access[{point_index}].target_kind must be loc or obj")
                    if not isinstance(point.get("target_name"), str) or not point["target_name"].strip():
                        raise ValueError(f"step {index}.staircase.access[{point_index}].target_name is required")
                    if not isinstance(point.get("option_tokens"), list) or not point["option_tokens"] or not all(
                        isinstance(token, str) and token.strip() for token in point["option_tokens"]
                    ):
                        raise ValueError(f"step {index}.staircase.access[{point_index}].option_tokens must be a non-empty list of strings")
                    if point.get("via") is not None:
                        if not isinstance(point["via"], list) or not all(isinstance(waypoint, dict) for waypoint in point["via"]):
                            raise ValueError(f"step {index}.staircase.access[{point_index}].via must be a list of coordinate objects")
                        for waypoint_index, waypoint in enumerate(point["via"]):
                            for field in ("x", "z"):
                                if not isinstance(waypoint.get(field), int):
                                    raise ValueError(f"step {index}.staircase.access[{point_index}].via[{waypoint_index}].{field} must be an integer")
                            waypoint_level = waypoint.get("level", point.get("level", 0))
                            if not isinstance(waypoint_level, int) or not 0 <= waypoint_level <= 3:
                                raise ValueError(f"step {index}.staircase.access[{point_index}].via[{waypoint_index}].level must be between 0 and 3")
        elif kind == "quest_turn_in":
            if not isinstance(step.get("quest_id"), str) or not step["quest_id"].strip():
                raise ValueError(f"step {index}.quest_id is required")
            if step.get("target_kind") not in {"npc", "loc", "obj"}:
                raise ValueError(f"step {index}.target_kind is invalid")
            if not isinstance(step.get("target_name"), str) or not step["target_name"].strip():
                raise ValueError(f"step {index}.target_name is required")
            if not isinstance(step.get("option_tokens"), list) or not step["option_tokens"] or not all(
                isinstance(token, str) and token.strip() for token in step["option_tokens"]
            ):
                raise ValueError(f"step {index}.option_tokens must be a non-empty list of strings")
            if (step.get("x") is None) != (step.get("z") is None) or any(
                field in step and not isinstance(step[field], int) for field in ("x", "z", "level")
            ):
                raise ValueError(f"step {index}.x and step {index}.z must be integer target coordinates when supplied")
            if "level" in step and not 0 <= step["level"] <= 3:
                raise ValueError(f"step {index}.level must be between 0 and 3")
        normalized.append(step)
    return normalized


async def _observe(agent_id: str, radius: int) -> dict[str, Any]:
    return await api.request(
        "GET",
        f"/agents/{agent_id}/observation",
        params={"radius": radius},
    )


def _bank_is_open(observation: dict[str, Any]) -> bool:
    """Recognize the live bank interface from its authoritative listeners."""

    listeners = [
        listener.get("name", "").casefold()
        for inventory in (observation.get("agent") or {}).get("inventories", [])
        for listener in inventory.get("listeners", [])
    ]
    return "bank_main:inv" in listeners and "bank_side:inv" in listeners


def _shop_is_open(observation: dict[str, Any]) -> bool:
    """Recognize the live shop interface from its authoritative listeners."""

    listeners = [
        listener.get("name", "").casefold()
        for inventory in (observation.get("agent") or {}).get("inventories", [])
        for listener in inventory.get("listeners", [])
    ]
    return "shop_template:inv" in listeners and "shop_template_side:inv" in listeners


def _shop_target(observation: dict[str, Any]) -> dict[str, Any] | None:
    """Select a nearby live shopkeeper option with content-safe precedence.

    The packed content commonly exposes ``Trade`` as a client convenience,
    while the shopkeeper's actual script is on ``Talk-to`` and opens the shop
    after a rendered confirmation. Prefer that script-backed path so the MCP
    does not queue an option that the live target cannot execute.
    """

    position = (observation.get("agent") or {}).get("position") or {}
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for npc in (observation.get("nearby") or {}).get("npcs", []):
        name = (npc.get("name") or "").casefold()
        category = (npc.get("categoryName") or "").casefold()
        if "shop" not in name and category != "shop_keeper":
            continue
        options = npc.get("options") or []
        # Prefer the script-backed Talk-to option when both labels are
        # exposed; _option_match intentionally honors token order, so select
        # this precedence explicitly rather than relying on the token list.
        match = _option_match(options, ["talk-to", "talk"])
        if match is None:
            match = _option_match(options, ["trade"])
        if match is None:
            continue
        option, label = match
        target_position = npc.get("position") or {}
        distance = max(
            abs(target_position.get("x", 0) - position.get("x", 0)),
            abs(target_position.get("z", 0) - position.get("z", 0)),
        )
        direct = 0 if label.casefold() == "talk-to" else 1
        candidates.append((direct * 1000 + distance, int(npc["id"]), {
            "target": {"kind": "npc", "id": npc["id"]},
            "option": option,
            "label": label,
            "entity": npc,
            "distance": distance,
        }))
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2] if candidates else None


def _bank_target(observation: dict[str, Any]) -> dict[str, Any] | None:
    """Select a nearby bank booth or banker using only live options."""

    position = (observation.get("agent") or {}).get("position") or {}
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for collection, kind in (("locations", "loc"), ("npcs", "npc")):
        for entity in (observation.get("nearby") or {}).get(collection, []):
            name = (entity.get("name") or "").casefold()
            category = (entity.get("categoryName") or "").casefold()
            if kind == "loc" and "bank" not in name:
                continue
            if kind == "npc" and "bank" not in name and category != "bank_teller":
                continue
            match = _option_match(entity.get("options") or [], ["bank", "use", "talk"])
            if match is None:
                continue
            option, label = match
            target_position = entity.get("position") or {}
            target: dict[str, Any] = {"kind": kind, "id": entity["id"]}
            if kind == "loc":
                target.update({
                    "x": target_position["x"],
                    "z": target_position["z"],
                    "level": target_position.get("level", 0),
                })
            distance = max(
                abs(target_position.get("x", 0) - position.get("x", 0)),
                abs(target_position.get("z", 0) - position.get("z", 0)),
            )
            # Prefer the explicit Bank script over booth Use, then prefer
            # booth Use over dialogue-only Talk-to. Some servers expose both
            # at the same tile, but the direct Banker action is the more
            # reliable route to the bank interface.
            direct = {"bank": 0, "use": 1, "talk-to": 2}.get(label.casefold(), 3)
            candidates.append((direct * 1000 + distance, int(entity["id"]), {
                "target": target,
                "option": option,
                "label": label,
                "entity": entity,
                "distance": distance,
            }))
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2] if candidates else None


def _bank_amount_option(amount: int | str) -> int:
    options = {1: 1, 5: 2, 10: 3, "all": 4, "x": 5}
    if amount not in options:
        raise ValueError("amount must be 1, 5, 10, 'all', or 'x'")
    return options[amount]


def _inventory_signature(observation: dict[str, Any]) -> str:
    return repr((observation.get("agent") or {}).get("inventories", []))


async def _resolve_inventory_action(
    agent_id: str,
    action: dict[str, Any],
    before_observation: dict[str, Any],
    radius: int = 8,
    max_wait_seconds: float = 8.0,
    require_bank: bool = False,
    require_shop: bool = False,
    validation: str = "inventory_changed_on_live_server",
) -> dict[str, Any]:
    """Validate an inventory action through a live inventory mutation."""

    before_signature = _inventory_signature(before_observation)
    attempts = max(1, int(max_wait_seconds / 0.25))
    latest = action.get("observation") or before_observation
    for _ in range(attempts):
        await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius)
        if (
            (not require_bank or _bank_is_open(latest))
            and (not require_shop or _shop_is_open(latest))
            and _inventory_signature(latest) != before_signature
        ):
            return {
                "status": "complete",
                "validated": True,
                "validation": validation,
                "action": action,
                "observation": latest,
            }
    return {
        "status": "blocked",
        "validated": False,
        "validation": "inventory_action_without_observable_change",
        "reason": "the inventory did not change before the validation timeout",
        "action": action,
        "observation": latest,
    }


async def _resolve_bank_inventory_action(
    agent_id: str,
    action: dict[str, Any],
    before_observation: dict[str, Any],
    radius: int = 8,
    max_wait_seconds: float = 8.0,
) -> dict[str, Any]:
    return await _resolve_inventory_action(
        agent_id,
        action,
        before_observation,
        radius=radius,
        max_wait_seconds=max_wait_seconds,
        require_bank=True,
        validation="bank_inventory_changed_on_live_server",
    )


async def _resolve_shop_inventory_action(
    agent_id: str,
    action: dict[str, Any],
    before_observation: dict[str, Any],
    radius: int = 8,
    max_wait_seconds: float = 8.0,
) -> dict[str, Any]:
    return await _resolve_inventory_action(
        agent_id,
        action,
        before_observation,
        radius=radius,
        max_wait_seconds=max_wait_seconds,
        require_shop=True,
        validation="shop_inventory_changed_on_live_server",
    )


async def _wait_for_count_dialog(agent_id: str, radius: int = 1, max_wait_seconds: float = 4.0) -> dict[str, Any]:
    latest = await _observe(agent_id, radius)
    attempts = max(1, int(max_wait_seconds / 0.25))
    for _ in range(attempts):
        if (latest.get("ui") or {}).get("activeScriptExecution") == 4:
            return latest
        await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius)
    return latest


async def _settle_chat_dialogue(agent_id: str, radius: int, max_steps: int = 12) -> dict[str, Any]:
    """Advance live Continue pages without guessing a dialogue choice.

    The engine keeps a chat script active between pages.  An interaction is
    therefore not settled merely because its first chat box appeared.  This
    helper resumes only pages with no visible choice buttons and stops at a
    real choice so the caller can select by rendered text.
    """

    latest = await _observe(agent_id, radius=radius)
    for _ in range(max_steps):
        ui = latest.get("ui") or {}
        buttons = ui.get("resumeButtons") or []
        if buttons:
            return {
                "status": "blocked",
                "validated": False,
                "validation": "dialogue_choice_requires_explicit_option",
                "reason": "the live dialogue is waiting for an explicit rendered choice",
                "dialogue_options": buttons,
                "observation": latest,
            }
        if not ui.get("activeScript") or ui.get("modalChat") in (None, -1):
            return {
                "status": "complete",
                "validated": True,
                "validation": "dialogue_settled_on_live_server",
                "observation": latest,
            }
        await resume_dialogue(agent_id)
        await asyncio.sleep(0.15)
        latest = await _observe(agent_id, radius=radius)
    return {
        "status": "blocked",
        "validated": False,
        "validation": "dialogue_settle_timeout",
        "reason": "the live chat script did not settle before the bounded continuation limit",
        "observation": latest,
    }


async def _wait_for_agent_active(agent_id: str, max_attempts: int = 20) -> dict[str, Any]:
    """Turn the engine's pending activation window into an MCP-ready session."""

    latest: dict[str, Any] | None = None
    for _ in range(max_attempts):
        latest = await _observe(agent_id, radius=1)
        if latest.get("agent", {}).get("status") == "active":
            return latest
        await asyncio.sleep(0.25)
    raise LostCityApiError(
        f"agent '{agent_id}' did not become active before the MCP activation timeout"
    )


def _interaction_effect_signature(observation: dict[str, Any]) -> tuple[Any, ...]:
    """Return state that an interaction may change, excluding pathing noise.

    ``targetOperation`` being cleared only proves that the engine finished
    walking/dispatching the request.  It does not prove that the requested
    option had an effect.  Keep the signature deliberately focused on state
    that content scripts commonly mutate: inventory, skills, tutorial/quest
    UI, and nearby loc/object definitions.
    """

    agent = observation.get("agent") or {}
    ui = observation.get("ui") or {}
    nearby = observation.get("nearby") or {}
    return (
        repr(agent.get("inventories")),
        repr(agent.get("skills")),
        repr(observation.get("tutorial")),
        repr({
            key: ui.get(key)
            for key in (
                "modalState",
                "modalMain",
                "modalChat",
                "modalSide",
                "modalTutorial",
                "overlay",
                "lastComponent",
                "activeScript",
                "activeScriptExecution",
                "resumeButtons",
            )
        }),
        repr(nearby.get("locations") or []),
        repr(nearby.get("objects") or []),
    )


def _target_effect_signature(observation: dict[str, Any], target: dict[str, Any]) -> tuple[Any, ...] | None:
    """Snapshot only the requested item-on target, excluding unrelated spawns."""

    kind = target.get("kind")
    collection = {"npc": "npcs", "loc": "locations", "obj": "objects"}.get(kind)
    if collection is None:
        return None
    for entity in (observation.get("nearby") or {}).get(collection, []):
        if kind == "player":
            continue
        if entity.get("id") != target.get("id"):
            continue
        if kind == "npc":
            return (
                kind,
                entity.get("id"),
                entity.get("type"),
                entity.get("name"),
                entity.get("category"),
                repr(entity.get("options") or []),
            )
        return (
            kind,
            entity.get("id"),
            entity.get("category"),
            entity.get("categoryName"),
            entity.get("name"),
            repr(entity.get("options") or []),
        )
    return None


async def _resolve_interaction_action(
    agent_id: str,
    action: dict[str, Any],
    radius: int,
    max_wait_seconds: float = 12.0,
    respect_safety: bool = True,
    before_observation: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    allow_pending_observable_effect: bool = False,
) -> dict[str, Any]:
    """Wait for a queued interaction to resolve before reporting the step validated."""

    observation = action.get("observation") or {}
    before_signature = _interaction_effect_signature(before_observation) if before_observation else None
    before_target_signature = _target_effect_signature(before_observation, target) if before_observation and target else None
    agent = observation.get("agent") or {}
    if agent.get("targetOperation") is None and before_signature is None:
        return {
            "status": "complete",
            "validated": False,
            "validation": "accepted_without_observable_baseline",
            "action": action,
            "observation": observation,
        }

    attempts = max(1, int(max_wait_seconds / 0.5))
    latest = observation
    observed_effect = False
    for _ in range(attempts):
        await asyncio.sleep(0.5)
        latest = await _observe(agent_id, radius=radius)
        safety = latest.get("safety") or {}
        if respect_safety and (safety.get("lowHealth") or safety.get("fleeing")):
            return {
                "status": "paused_low_health",
                "validated": False,
                "validation": "interrupted_by_safety_controller",
                "action": action,
                "safety": safety,
                "observation": latest,
            }
        changed = before_signature is not None and _interaction_effect_signature(latest) != before_signature
        target_changed = target is not None and _target_effect_signature(latest, target) != before_target_signature
        observed_effect = observed_effect or changed or target_changed
        operation_pending = (latest.get("agent") or {}).get("targetOperation") is not None
        ui = latest.get("ui") or {}
        if observed_effect and ui.get("activeScript") and ui.get("modalChat") not in (None, -1):
            settled = await _settle_chat_dialogue(agent_id, radius)
            latest = settled.get("observation") or latest
            if settled.get("status") != "complete":
                return {**settled, "action": action}
            operation_pending = (latest.get("agent") or {}).get("targetOperation") is not None
        if observed_effect and (not operation_pending or allow_pending_observable_effect):
            return {
                "status": "complete",
                "validated": True,
                "validation": (
                    "pending_operation_resolved_with_observable_effect"
                    if not operation_pending
                    else "observable_effect_confirmed_while_operation_pending"
                ),
                "action": action,
                "observation": latest,
            }
        if not operation_pending:
            return {
                "status": "blocked",
                "validated": False,
                "validation": "interaction_resolved_without_observable_effect",
                "reason": "the engine finished the interaction, but no inventory, skill, UI, tutorial, location, or object state changed",
                "action": action,
                "observation": latest,
            }
    return {
        "status": "blocked",
        "validated": False,
        "validation": "pending_operation_timeout",
        "reason": "interaction did not resolve before the validation timeout",
        "action": action,
        "observation": latest,
    }


def _is_toll_gate_target(observation: dict[str, Any], target: dict[str, Any]) -> bool:
    if target.get("kind") != "loc":
        return False
    for location in (observation.get("nearby") or {}).get("locations", []):
        position = location.get("position") or {}
        if (
            location.get("id") == target.get("id")
            and position.get("x") == target.get("x")
            and position.get("z") == target.get("z")
            and str(location.get("categoryName") or "").casefold().startswith("border_gate_toll_")
        ):
            return True
    return False


async def _complete_toll_gate_dialogue(
    agent_id: str,
    resolved: dict[str, Any],
    radius: int,
) -> dict[str, Any]:
    """Finish the live toll dialogue, selecting only its named pay option."""

    latest = resolved.get("observation") or {}
    for _ in range(8):
        ui = latest.get("ui") or {}
        buttons = ui.get("resumeButtons") or []
        if buttons:
            selected = _select_dialogue_button(latest, "Yes, ok.")
            if selected is None:
                return resolved
            choice = await choose_dialogue_option(agent_id, "Yes, ok.", radius=radius)
            latest = choice.get("observation") or latest
            if choice.get("status") != "complete":
                return {**resolved, **choice, "action": resolved.get("action"), "observation": latest}
            continue
        if ui.get("activeScript") and ui.get("modalChat") not in (None, -1):
            settled = await _settle_chat_dialogue(agent_id, radius)
            latest = settled.get("observation") or latest
            if settled.get("status") != "complete":
                return {**resolved, **settled, "action": resolved.get("action"), "observation": latest}
            continue
        position = (latest.get("agent") or {}).get("position") or {}
        if "pay the guard" in str((latest.get("agent") or {}).get("lastGameMessage") or "").casefold():
            return {
                **resolved,
                "status": "complete",
                "validated": True,
                "validation": "toll_gate_dialogue_and_crossing_observed_on_live_server",
                "observation": latest,
            }
        if position:
            await asyncio.sleep(0.25)
            latest = await _observe(agent_id, radius=radius)
            continue
        return resolved
    return {
        **resolved,
        "status": "blocked",
        "validated": False,
        "validation": "toll_gate_dialogue_timeout",
        "reason": "the live toll dialogue did not finish within the bounded continuation limit",
        "observation": latest,
    }


def _nearest_open_door(
    observation: dict[str, Any],
    excluded: set[tuple[int, int, int]] | None = None,
    reference: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    """Find the nearest closed door that can be opened from the live view."""

    position = (observation.get("agent") or {}).get("position") or {}
    excluded = excluded or set()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for location in (observation.get("nearby") or {}).get("locations", []):
        category = (location.get("categoryName") or "").casefold()
        name = (location.get("name") or "").casefold()
        if not (
            category.startswith("door_")
            or category.startswith("gate_")
            or name in {"door", "large door", "gate"}
        ):
            continue
        match = _option_match(location.get("options") or [], ["open"])
        if match is None:
            continue
        target_position = location.get("position") or {}
        if target_position.get("level") != position.get("level"):
            continue
        obstacle_key = (
            int(location.get("id", 0)),
            int(target_position.get("x", 0)),
            int(target_position.get("z", 0)),
        )
        if obstacle_key in excluded:
            continue
        distance = max(
            abs(target_position.get("x", 0) - position.get("x", 0)),
            abs(target_position.get("z", 0) - position.get("z", 0)),
        )
        reference_distance = (
            max(
                abs(target_position.get("x", 0) - reference["x"]),
                abs(target_position.get("z", 0) - reference["z"]),
            )
            if reference is not None
            else distance
        )
        obstacle_kind = 0 if category.startswith("gate_") else 1
        target = {
            "kind": "loc",
            "id": location["id"],
            "x": target_position["x"],
            "z": target_position["z"],
            "level": target_position.get("level", 0),
        }
        candidates.append((obstacle_kind * 1000 + reference_distance - (8 if obstacle_kind == 0 else 0), int(location["id"]), {
            "target": target,
            "option": match[0],
            "label": match[1],
            "entity": location,
            "distance": distance,
            "reference_distance": reference_distance,
        }))
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2] if candidates else None


async def _travel_quest_step(agent_id: str, step: dict[str, Any], radius: int) -> dict[str, Any]:
    """Walk to a live tile, opening blocking doors through normal interactions."""

    step = dict(step)
    via_points = list(step.pop("via", []) or [])
    via_index = 0
    destination = {"x": step["x"], "z": step["z"], "level": step.get("level", 0)}
    opened_doors: list[dict[str, Any]] = []
    rejected_obstacles: set[tuple[int, int, int]] = set()
    explored_frontiers: set[tuple[int, int, int]] = set()
    last_route: dict[str, Any] | None = None
    last_action: dict[str, Any] | None = None
    for _ in range(24):
        observation = await _observe(agent_id, radius=min(64, max(32, radius)))
        safety = observation.get("safety") or {}
        if safety.get("lowHealth") or safety.get("fleeing"):
            return {"status": "paused_low_health", "safety": safety, "observation": observation}
        position = observation["agent"]["position"]
        explored_frontiers.add((position["level"], position["x"], position["z"]))
        if position["level"] != destination["level"]:
            return {
                "status": "blocked",
                "reason": "travel_level_mismatch",
                "destination": destination,
                "observation": observation,
            }
        if position["x"] == destination["x"] and position["z"] == destination["z"]:
            return {
                "status": "complete",
                "validated": True,
                "validation": "destination_reached_on_live_server",
                "destination": destination,
                "opened_doors": opened_doors,
                "route": last_route,
                "action": last_action,
                "observation": observation,
            }

        # Ask the live engine for a bounded multi-leg plan. The engine still
        # validates every queued waypoint against current collision, while the
        # route response lets one MCP call cover a sizable cached-map window.
        route = await api.request("POST", f"/agents/{agent_id}/route", json={**destination, "maxLegs": 32})
        last_route = route
        final = route.get("finalPosition") or {}
        current = position
        if not route.get("reached") and via_index < len(via_points):
            waypoint = via_points[via_index]
            via_index += 1
            waypoint_result = await _travel_quest_step(
                agent_id,
                {"kind": "travel", **waypoint, "run": step.get("run", True)},
                radius,
            )
            if waypoint_result.get("status") != "complete":
                return {
                    **waypoint_result,
                    "phase": "travel_via_waypoint",
                    "waypoint_index": via_index - 1,
                    "destination": destination,
                }
            continue
        cached_frontier_door = None
        if not route.get("reached") and via_index >= len(via_points):
            # Prefer a reachable live door frontier over the routefinder's
            # bounded collision endpoint. The latter may be closer in raw
            # distance while still being on the wrong side of a door, which
            # caused long workers to oscillate between two partial frontiers.
            cached_probe = await map_window(
                agent_id,
                radius=min(_MAX_MAP_CACHE_RADIUS, max(32, radius)),
                target_x=destination["x"],
                target_z=destination["z"],
                target_level=destination["level"],
            )
            cached_frontier_door = _cached_frontier_door(cached_probe, destination, rejected_obstacles)
        if route.get("reached"):
            move_destination = destination
        elif cached_frontier_door is not None:
            move_destination = None
        elif route.get("waypoints") and (final.get("x"), final.get("z")) != (current["x"], current["z"]):
            # The route planner can stop either at the near side of a
            # collision boundary or at its bounded search frontier. Only
            # accept the former: moving to an arbitrary frontier tile can
            # make an autonomous worker wander away from its objective.
            current_distance = max(
                abs(destination["x"] - current["x"]),
                abs(destination["z"] - current["z"]),
            )
            final_distance = max(
                abs(destination["x"] - int(final.get("x", current["x"]))),
                abs(destination["z"] - int(final.get("z", current["z"]))),
            )
            if final_distance <= 1 and final_distance < current_distance:
                move_destination = {
                    "x": final["x"],
                    "z": final["z"],
                    "level": destination["level"],
                }
            elif final_distance < current_distance:
                # The live route planner can return a bounded collision
                # frontier rather than the destination when a door or map
                # boundary blocks the next leg. Advance only to a frontier
                # that is strictly closer to the requested destination; the
                # next observation then either reaches the target or exposes
                # the exact live door to open. This also makes long-distance
                # routes resumable instead of rejecting useful progress.
                move_destination = {
                    "x": final["x"],
                    "z": final["z"],
                    "level": destination["level"],
                }
            else:
                move_destination = None
        else:
            move_destination = None

        if move_destination is not None:
            try:
                action = await api.request(
                    "POST",
                    f"/agents/{agent_id}/actions",
                    json={
                        "type": "move",
                        "x": move_destination["x"],
                        "z": move_destination["z"],
                        "run": step.get("run", True),
                        "maxLegs": 32,
                    },
                )
            except LostCityApiError as error:
                return {
                    "status": "blocked",
                    "reason": "movement_rejected_by_live_server",
                    "message": str(error),
                    "route": route,
                    "observation": observation,
                    "opened_doors": opened_doors,
                }
            last_action = action
            timeout = min(120, max(10, int(route.get("estimatedTicks", 1) * 2 + 5)))
            for _ in range(timeout):
                latest = await _observe(agent_id, radius=1)
                safety = latest.get("safety") or {}
                if safety.get("lowHealth") or safety.get("fleeing"):
                    return {"status": "paused_low_health", "safety": safety, "observation": latest}
                latest_position = latest["agent"]["position"]
                if latest_position["x"] == move_destination["x"] and latest_position["z"] == move_destination["z"]:
                    break
                await asyncio.sleep(0.5)
            continue

        # If the live route stops at a dynamic boundary, use the cached map
        # component to select the nearest actually reachable door frontier.
        # This avoids blindly probing coordinates: the approach tile comes
        # from authoritative cached exits, and the subsequent move/interact
        # still goes through the live engine and is postcondition-validated.
        cached_window = await map_window(
            agent_id,
            radius=min(_MAX_MAP_CACHE_RADIUS, max(32, radius)),
            target_x=destination["x"],
            target_z=destination["z"],
            target_level=destination["level"],
        )
        frontier_door = _cached_frontier_door(cached_window, destination, rejected_obstacles)
        if frontier_door is not None:
            door, approach, cached_route = frontier_door
            if (position["x"], position["z"]) != (approach["x"], approach["z"]):
                try:
                    move_action = await api.request(
                        "POST",
                        f"/agents/{agent_id}/actions",
                        json={"type": "move", "x": approach["x"], "z": approach["z"], "run": step.get("run", True), "maxLegs": 32},
                    )
                    settled = await _wait_for_batch_move(
                        agent_id,
                        approach["x"],
                        approach["z"],
                        timeout_seconds=min(120, max(15, int(cached_route.get("tile_count", 1) * 2 + 5))),
                        radius=1,
                    )
                except LostCityApiError:
                    settled = {"status": "blocked", "validated": False}
                if settled.get("status") == "complete" and settled.get("validated"):
                    last_action = move_action
                    try:
                        opened = await interact_agent(
                            agent_id,
                            "loc",
                            option=int(door["selected_option"]),
                            target_id=int(door["id"]),
                            x=int((door.get("position") or {}).get("x", 0)),
                            z=int((door.get("position") or {}).get("z", 0)),
                            level=int((door.get("position") or {}).get("level", destination["level"])),
                        )
                    except (LostCityApiError, ValueError):
                        opened = {"status": "blocked", "validated": False}
                    if opened.get("status") == "complete" and opened.get("validated"):
                        opened_doors.append(door)
                        rejected_obstacles.add((int(door["id"]), int((door.get("position") or {}).get("x", 0)), int((door.get("position") or {}).get("z", 0))))
                        last_action = opened.get("action")
                        continue
                    rejected_obstacles.add((int(door.get("id", 0)), int((door.get("position") or {}).get("x", 0)), int((door.get("position") or {}).get("z", 0))))
                    continue
                rejected_obstacles.add((int(door.get("id", 0)), int((door.get("position") or {}).get("x", 0)), int((door.get("position") or {}).get("z", 0))))
            else:
                try:
                    opened = await interact_agent(
                        agent_id,
                        "loc",
                        option=int(door["selected_option"]),
                        target_id=int(door["id"]),
                        x=int((door.get("position") or {}).get("x", 0)),
                        z=int((door.get("position") or {}).get("z", 0)),
                        level=int((door.get("position") or {}).get("level", destination["level"])),
                    )
                except (LostCityApiError, ValueError):
                    opened = {"status": "blocked", "validated": False}
                if opened.get("status") == "complete" and opened.get("validated"):
                    opened_doors.append(door)
                    rejected_obstacles.add((int(door["id"]), int((door.get("position") or {}).get("x", 0)), int((door.get("position") or {}).get("z", 0))))
                    last_action = opened.get("action")
                    continue
                rejected_obstacles.add((int(door.get("id", 0)), int((door.get("position") or {}).get("x", 0)), int((door.get("position") or {}).get("z", 0))))

        prioritize_nearby_door = False
        if opened_doors:
            # After crossing a door the route endpoint can remain a local
            # frontier even though the long route has no visible waypoints.
            # Ask the live planner about each adjacent tile and advance only
            # when it confirms a reachable, Manhattan-closer step. This lets
            # the worker walk out of a doorway one tile at a time without
            # accepting an arbitrary coordinate.
            nearby_door = _nearest_open_door(observation, rejected_obstacles, reference=current)
            nearby_door_position = (nearby_door or {}).get("target") or {}
            nearby_door_distance = max(
                abs(int(nearby_door_position.get("x", current["x"])) - current["x"]),
                abs(int(nearby_door_position.get("z", current["z"])) - current["z"]),
            )
            prioritize_nearby_door = nearby_door is not None and nearby_door_distance <= 8
            current_manhattan = abs(destination["x"] - current["x"]) + abs(destination["z"] - current["z"])
            probes = sorted(
                (
                    abs(destination["x"] - (current["x"] + dx)) + abs(destination["z"] - (current["z"] + dz)),
                    dx,
                    dz,
                )
                for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
            )
            advanced = False
            for probe_distance, dx, dz in probes:
                if prioritize_nearby_door:
                    continue
                probe_destination = {"x": current["x"] + dx, "z": current["z"] + dz, "level": destination["level"]}
                probe_key = (probe_destination["level"], probe_destination["x"], probe_destination["z"])
                # Prefer forward progress, but permit a bounded, unvisited
                # backtrack when every forward tile is blocked by the same
                # collision boundary.
                if probe_distance >= current_manhattan and probe_key in explored_frontiers:
                    continue
                if len(explored_frontiers) >= 32 and probe_distance >= current_manhattan:
                    continue
                probe_route = await api.request("POST", f"/agents/{agent_id}/route", json=probe_destination)
                if not probe_route.get("reached"):
                    continue
                probe_action = await api.request(
                    "POST",
                    f"/agents/{agent_id}/actions",
                    json={"type": "move", "x": probe_destination["x"], "z": probe_destination["z"], "run": step.get("run", True)},
                )
                for _ in range(min(20, max(5, int(probe_route.get("estimatedTicks", 1) * 2 + 2)))):
                    latest = await _observe(agent_id, radius=1)
                    latest_position = latest["agent"]["position"]
                    if (latest_position["x"], latest_position["z"]) == (probe_destination["x"], probe_destination["z"]):
                        last_action = probe_action
                        explored_frontiers.add(probe_key)
                        advanced = True
                        break
                    await asyncio.sleep(0.25)
                if advanced:
                    break
            if advanced:
                continue

        # A collision boundary can require a short, validated backtrack to
        # reach another door or the far side of a structure. The route API's
        # finalPosition is still authoritative here: it is only considered
        # when the server returned waypoints, the frontier is nearby, and it
        # has not already been explored. This is deliberately bounded so a
        # quest worker cannot wander indefinitely when the map is genuinely
        # disconnected.
        frontier = route.get("finalPosition") or {}
        frontier_key = (
            int(frontier.get("level", destination["level"])),
            int(frontier.get("x", current["x"])),
            int(frontier.get("z", current["z"])),
        )
        frontier_distance = max(
            abs(frontier_key[1] - current["x"]),
            abs(frontier_key[2] - current["z"]),
        )
        if (
            route.get("waypoints")
            and frontier_key != (current["level"], current["x"], current["z"])
            and frontier_key not in explored_frontiers
            and frontier_distance <= 64
            and len(explored_frontiers) < 8
            and not prioritize_nearby_door
        ):
            explored_frontiers.add(frontier_key)
            try:
                frontier_action = await api.request(
                    "POST",
                    f"/agents/{agent_id}/actions",
                    json={
                        "type": "move",
                        "x": frontier_key[1],
                        "z": frontier_key[2],
                        "run": step.get("run", True),
                    },
                )
            except LostCityApiError:
                frontier_action = None
            if frontier_action is not None:
                last_action = frontier_action
                timeout = min(120, max(10, int(route.get("estimatedTicks", 1) * 2 + 5)))
                for _ in range(timeout):
                    latest = await _observe(agent_id, radius=1)
                    safety = latest.get("safety") or {}
                    if safety.get("lowHealth") or safety.get("fleeing"):
                        return {"status": "paused_low_health", "safety": safety, "observation": latest}
                    latest_position = latest["agent"]["position"]
                    if (
                        latest_position["level"],
                        latest_position["x"],
                        latest_position["z"],
                    ) == frontier_key:
                        break
                    await asyncio.sleep(0.5)
                continue

        reference = current if not route.get("waypoints") else destination
        if final and (final.get("x"), final.get("z")) != (current["x"], current["z"]):
            reference = {"x": final["x"], "z": final["z"], "level": destination["level"]}
        door = _nearest_open_door(observation, rejected_obstacles, reference)
        if door is None:
            return {
                "status": "blocked",
                "reason": "destination_unreachable_without_forward_progress",
                "route": route,
                "observation": observation,
                "opened_doors": opened_doors,
            }
        door_position = door["target"]
        door_distance = max(
            abs(door_position["x"] - position["x"]),
            abs(door_position["z"] - position["z"]),
        )
        if door_distance > 2:
            # A gate/door can be visible through a fence while its interaction
            # boundary is not yet reachable. Walk to a collision-aware tile
            # beside it first, then re-observe and dispatch the interaction.
            approached = False
            for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)):
                approach_destination = {
                    "x": door_position["x"] + dx,
                    "z": door_position["z"] + dz,
                    "level": destination["level"],
                }
                approach_route = await api.request(
                    "POST",
                    f"/agents/{agent_id}/route",
                    json=approach_destination,
                )
                if not approach_route.get("reached"):
                    continue
                try:
                    approach_action = await api.request(
                        "POST",
                        f"/agents/{agent_id}/actions",
                        json={
                            "type": "move",
                            "x": approach_destination["x"],
                            "z": approach_destination["z"],
                            "run": step.get("run", True),
                        },
                    )
                except LostCityApiError:
                    continue
                for _ in range(min(60, max(10, int(approach_route.get("estimatedTicks", 1) * 2 + 5)))):
                    latest = await _observe(agent_id, radius=1)
                    latest_position = latest["agent"]["position"]
                    if (latest_position["x"], latest_position["z"]) == (approach_destination["x"], approach_destination["z"]):
                        approached = True
                        last_action = approach_action
                        break
                    await asyncio.sleep(0.25)
                if approached:
                    break
            if approached:
                continue
        try:
            action = await api.request(
                "POST",
                f"/agents/{agent_id}/actions",
                json={"type": "interact", "target": door["target"], "option": door["option"]},
            )
        except LostCityApiError as error:
            rejected_obstacles.add((int(door["target"]["id"]), int(door["target"]["x"]), int(door["target"]["z"])))
            continue
        result = await _resolve_interaction_action(
            agent_id,
            action,
            radius=max(8, radius),
            before_observation=observation,
            target=door["target"],
        )
        if result.get("status") != "complete" or not result.get("validated"):
            rejected_obstacles.add((int(door["target"]["id"]), int(door["target"]["x"]), int(door["target"]["z"])))
            continue
        opened_doors.append(door)
        _mark_map_cache_dirty(agent_id)
        rejected_obstacles.add(
            (
                int(door["target"]["id"]),
                int(door["target"]["x"]),
                int(door["target"]["z"]),
            )
        )
        last_action = action
        # Some live doors resolve the interaction on the door tile itself.
        # The route API may then report no path until the player takes one
        # legal step through the opening. Probe only adjacent tiles and only
        # accept a server-reported reachable tile that reduces destination
        # distance; this is a bounded escape, never a blind coordinate jump.
        after_open = result.get("observation") or await _observe(agent_id, radius=1)
        after_position = after_open.get("agent", {}).get("position") or {}
        after_distance = (
            abs(destination["x"] - after_position.get("x", position["x"]))
            + abs(destination["z"] - after_position.get("z", position["z"]))
        )
        escape_candidates = sorted(
            (
                abs(destination["x"] - (after_position.get("x", 0) + dx))
                + abs(destination["z"] - (after_position.get("z", 0) + dz)),
                dx,
                dz,
            )
            for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
        )
        for escape_distance, dx, dz in escape_candidates:
            if escape_distance >= after_distance:
                continue
            escape_destination = {
                "x": after_position.get("x", 0) + dx,
                "z": after_position.get("z", 0) + dz,
                "level": destination["level"],
            }
            escape_route = await api.request("POST", f"/agents/{agent_id}/route", json=escape_destination)
            if not escape_route.get("reached"):
                continue
            escape_action = await api.request(
                "POST",
                f"/agents/{agent_id}/actions",
                json={"type": "move", "x": escape_destination["x"], "z": escape_destination["z"], "run": step.get("run", True)},
            )
            timeout = min(20, max(5, int(escape_route.get("estimatedTicks", 1) * 2 + 2)))
            for _ in range(timeout):
                escaped = await _observe(agent_id, radius=1)
                escaped_position = escaped.get("agent", {}).get("position") or {}
                if (escaped_position.get("x"), escaped_position.get("z")) == (escape_destination["x"], escape_destination["z"]):
                    last_action = escape_action
                    break
                await asyncio.sleep(0.25)
            else:
                continue
            break

    return {
        "status": "blocked",
        "reason": "movement_timeout",
        "destination": destination,
        "route": last_route,
        "action": last_action,
        "opened_doors": opened_doors,
        "observation": await _observe(agent_id, radius=1),
    }


async def _live_quest_state(agent_id: str, quest_id: str) -> dict[str, Any] | None:
    result = await api.request("GET", f"/agents/{agent_id}/quests")
    return next((quest for quest in result.get("quests", []) if quest.get("id") == quest_id), None)


def _quest_interaction_action(observation: dict[str, Any], step: dict[str, Any]) -> dict[str, Any] | None:
    targets = [step["target_kind"]]
    return _discover_skill_action(
        observation,
        {
            "targets": targets,
            "target_name": step.get("target_name"),
            "option_tokens": [token.casefold() for token in step["option_tokens"]],
        },
    )


def _carried_item_tokens(
    observation: dict[str, Any],
    tokens: list[str],
    *,
    exact: bool = False,
) -> dict[str, Any] | None:
    wanted = [token.casefold() for token in tokens]
    inventory = _carried_inventory(observation)
    for item in inventory.get("items", []):
        name = (item.get("name") or "").casefold()
        if any(token == name if exact else token in name for token in wanted):
            return {"inventory": inventory["id"], "item": item}
    return None


def _carried_count_tokens(observation: dict[str, Any], tokens: list[str]) -> int:
    wanted = [token.casefold() for token in tokens]
    return sum(
        int(item.get("count", 0))
        for item in _carried_inventory(observation).get("items", [])
        if any(token in (item.get("name") or "").casefold() for token in wanted)
    )


def _carried_count_ids(observation: dict[str, Any], item_ids: list[int] | None) -> int:
    """Count carried items by authoritative object id when display names collide."""

    wanted = {int(item_id) for item_id in (item_ids or [])}
    return sum(
        int(item.get("count", 0))
        for item in _carried_inventory(observation).get("items", [])
        if int(item.get("id", -1)) in wanted
    )


async def _quest_dialogue_step(agent_id: str, step: dict[str, Any], radius: int) -> dict[str, Any]:
    """Drive a multi-choice quest conversation to an authoritative var state."""

    expected_state = step["expected_state"]
    quest = await _live_quest_state(agent_id, step["quest_id"])
    if quest and int(quest.get("state", 0)) >= expected_state:
        return {
            "status": "complete",
            "validated": True,
            "validation": "quest_dialogue_state_already_reached",
            "quest": quest,
            "observation": await _observe(agent_id, radius),
        }

    latest = await _observe(agent_id, radius)
    selected: dict[str, Any] | None = None
    action: dict[str, Any] | None = None
    for interaction_attempt in range(4):
        selected = _quest_interaction_action(latest, step)
        if selected is None and isinstance(step.get("x"), int) and isinstance(step.get("z"), int):
            moved = await _travel_quest_step(
                agent_id,
                {"kind": "travel", "x": step["x"], "z": step["z"], "level": step.get("level", 0), "run": True},
                max(16, radius),
            )
            moved_observation = moved.get("observation") or latest
            moved_position = (moved_observation.get("agent") or {}).get("position") or {}
            within_range = (
                moved_position.get("level", 0) == step.get("level", 0)
                and max(abs(moved_position.get("x", 0) - step["x"]), abs(moved_position.get("z", 0) - step["z"])) <= 2
            )
            if moved.get("status") != "complete" and not within_range:
                return {**moved, "phase": "quest_dialogue_target_route", "observation": moved_observation}
            latest = moved_observation
            selected = _quest_interaction_action(latest, step)
        if selected is None:
            return {"status": "blocked", "validated": False, "reason": "quest_dialogue_target_not_found", "observation": latest}

        # A live NPC can be visible through a doorway or behind a table while
        # the interaction action itself is still unreachable.  Treat the
        # observed entity position as another bounded travel target so the
        # normal route/door-frontier logic can open the obstruction before we
        # retry the dialogue.  This is important for the Goblin Village
        # generals, whose room is separated from the approach tile by a live
        # large door.
        entity_position = (selected.get("entity") or {}).get("position") or {}
        if selected.get("distance", 0) > 1 and entity_position:
            moved = await _travel_quest_step(
                agent_id,
                {
                    "kind": "travel",
                    "x": int(entity_position["x"]),
                    "z": int(entity_position["z"]),
                    "level": int(entity_position.get("level", step.get("level", 0))),
                    "run": True,
                },
                max(16, radius),
            )
            moved_observation = moved.get("observation") or latest
            moved_position = (moved_observation.get("agent") or {}).get("position") or {}
            within_interaction_range = (
                moved_position.get("level", 0) == entity_position.get("level", step.get("level", 0))
                and max(
                    abs(moved_position.get("x", 0) - int(entity_position["x"])),
                    abs(moved_position.get("z", 0) - int(entity_position["z"])),
                )
                <= 1
            )
            if moved.get("status") != "complete" and not within_interaction_range:
                return {**moved, "phase": "quest_dialogue_entity_approach", "observation": moved_observation}
            latest = moved_observation
            selected = _quest_interaction_action(latest, step)
            if selected is None:
                continue

        action = await api.request(
            "POST",
            f"/agents/{agent_id}/actions",
            json={"type": "interact", "target": selected["target"], "option": selected["option"]},
        )
        latest = action.get("observation") or latest
        for _ in range(120):
            quest = await _live_quest_state(agent_id, step["quest_id"])
            if quest and int(quest.get("state", 0)) >= expected_state:
                return {
                    "status": "complete",
                    "validated": True,
                    "validation": "quest_dialogue_state_observed_on_live_server",
                    "quest": quest,
                    "selected": selected,
                    "action": action,
                    "interaction_attempt": interaction_attempt + 1,
                    "observation": latest,
                }
            ui = latest.get("ui") or {}
            if ui.get("resumeButtons"):
                choice = next(
                    (text for text in step["dialogue_options"] if _select_dialogue_button(latest, text) is not None),
                    None,
                )
                if choice is None:
                    return {
                        "status": "blocked",
                        "validated": False,
                        "reason": "quest_dialogue_options_unexpected",
                        "dialogue_options": ui.get("resumeButtons"),
                        "selected": selected,
                        "action": action,
                        "observation": latest,
                    }
                choice_result = await choose_dialogue_option(agent_id, choice, radius=radius)
                if choice_result.get("status") != "complete":
                    return {**choice_result, "phase": "quest_dialogue_choice", "selected": selected}
                latest = choice_result.get("observation") or latest
            elif ui.get("activeScript") or (latest.get("agent") or {}).get("targetOperation") is not None:
                try:
                    resumed = await resume_dialogue(agent_id) if ui.get("activeScript") else None
                    latest = resumed.get("observation") if resumed else await _observe(agent_id, radius)
                except LostCityApiError as error:
                    if "no_paused_dialogue" not in str(error):
                        raise
                    latest = await _observe(agent_id, radius)
            else:
                break
            await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius)

    return {
        "status": "blocked",
        "validated": False,
        "validation": "quest_dialogue_timeout",
        "reason": "quest state did not reach the requested dialogue milestone",
        "selected": selected,
        "action": action,
        "observation": latest,
    }


async def _quest_item_on_step(agent_id: str, step: dict[str, Any], radius: int) -> dict[str, Any]:
    """Use a carried quest item and require the content quest var to advance."""

    expected_state = step["expected_state"]
    quest = await _live_quest_state(agent_id, step["quest_id"])
    if quest and int(quest.get("state", 0)) >= expected_state:
        return {
            "status": "complete",
            "validated": True,
            "validation": "quest_item_on_state_already_reached",
            "quest": quest,
            "observation": await _observe(agent_id, radius),
        }
    observation = await _observe(agent_id, radius)
    item = _carried_item_tokens(observation, step["item_tokens"])
    if item is None:
        return {
            "status": "blocked",
            "validated": False,
            "reason": "required_quest_item_not_found_in_carried_inventory",
            "item_tokens": step["item_tokens"],
            "observation": observation,
        }
    collection = {"npc": "npcs", "loc": "locations", "obj": "objects"}[step["target_kind"]]
    target_entity = next(
        (
            entity
            for entity in observation.get("nearby", {}).get(collection, [])
            if (entity.get("name") or "").casefold() == step["target_name"].casefold()
        ),
        None,
    )
    if target_entity is None:
        return {
            "status": "blocked",
            "validated": False,
            "reason": "quest_item_target_not_found_in_live_observation",
            "target_name": step["target_name"],
            "observation": observation,
        }
    target_position = target_entity.get("position") or {}
    result = await use_item_on_validated(
        agent_id,
        item["inventory"],
        item["item"]["slot"],
        step["target_kind"],
        target_id=target_entity.get("id"),
        x=target_position.get("x"),
        z=target_position.get("z"),
        level=target_position.get("level"),
        radius=radius,
    )
    latest = result.get("observation") or observation
    for _ in range(60):
        quest = await _live_quest_state(agent_id, step["quest_id"])
        if quest and int(quest.get("state", 0)) >= expected_state:
            return {
                "status": "complete",
                "validated": True,
                "validation": "quest_item_on_state_observed_on_live_server",
                "quest": quest,
                "item": item,
                "target": target_entity,
                "result": result,
                "observation": latest,
            }
        await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius)
    return {
        "status": "blocked",
        "validated": False,
        "reason": "quest item-on action did not advance the authoritative quest state",
        "expected_state": expected_state,
        "item": item,
        "target": target_entity,
        "result": result,
        "observation": latest,
    }


async def _combine_quest_items_step(agent_id: str, step: dict[str, Any], radius: int) -> dict[str, Any]:
    observation = await _observe(agent_id, radius=1)
    output_item_ids = [int(item_id) for item_id in step.get("output_item_ids", [])]
    first = _carried_item_tokens(observation, step["item_tokens"], exact=bool(step.get("item_exact")))
    second = _carried_item_tokens(observation, step["use_item_tokens"], exact=bool(step.get("use_item_exact")))
    before_output = _carried_count_tokens(observation, step["output_tokens"])
    before_output_ids = _carried_count_ids(observation, output_item_ids)

    # Some transformations preserve the display name while changing the
    # authoritative object id (for example, a blonde wig is still displayed
    # as "Wig").  Do not select that already-transformed item as an input.
    if output_item_ids:
        output_ids = set(output_item_ids)
        if first and int(first["item"].get("id", -1)) in output_ids:
            first = None
        if second and int(second["item"].get("id", -1)) in output_ids:
            second = None

    if (before_output > 0 or before_output_ids > 0) and (first is None or second is None):
        return {
            "status": "complete",
            "validated": True,
            "validation": "quest_item_combination_already_present",
            "observation": observation,
        }
    if first is None or second is None:
        return {
            "status": "blocked",
            "validated": False,
            "reason": "required_quest_combination_inputs_not_found_in_carried_inventory",
            "observation": observation,
        }
    action = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={
            "type": "use_item",
            "inventory": first["inventory"],
            "slot": first["item"]["slot"],
            "useInventory": second["inventory"],
            "useSlot": second["item"]["slot"],
        },
    )
    latest = action.get("observation") or observation
    for _ in range(40):
        output_observed = _carried_count_tokens(latest, step["output_tokens"]) > before_output
        output_id_observed = _carried_count_ids(latest, output_item_ids) > before_output_ids
        if output_observed or output_id_observed:
            return {
                "status": "complete",
                "validated": True,
                "validation": (
                    "quest_item_combination_output_id_observed_on_live_server"
                    if output_id_observed and not output_observed
                    else "quest_item_combination_observed_on_live_server"
                ),
                "selected": {"first": first, "second": second},
                "action": action,
                "observation": latest,
            }
        await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius=1)
    return {
        "status": "blocked",
        "validated": False,
        "reason": "quest item combination did not produce its live output",
        "selected": {"first": first, "second": second},
        "action": action,
        "observation": latest,
    }


async def _wait_for_quest_item_or_content_script(
    agent_id: str,
    observation: dict[str, Any],
    item_tokens: list[str],
    radius: int,
    max_wait_seconds: float = 20.0,
) -> dict[str, Any]:
    """Let delayed quest content finish before a search target is discarded.

    Some quest searches dispatch a second content script (for example the
    Fluffs kitten script) after the interaction itself has been accepted. The
    action response can therefore show a changed UI while the item is not in
    inventory yet. Returning the crate to the candidate pool at that point
    made ``run_quest`` exhaust every crate and report a false blocker. Keep
    polling the authoritative server, resuming only plain message boxes, and
    return as soon as the required item is visible or the script has settled.
    """

    latest = observation
    attempts = max(1, int(max_wait_seconds / 0.25))
    for _ in range(attempts):
        if _carried_item_tokens(latest, item_tokens) is not None:
            return latest
        ui = latest.get("ui") or {}
        if ui.get("activeScript") and ui.get("resumeButtons"):
            # Search follow-ups use message boxes, never a quest choice. A
            # choice is deliberately left for the caller rather than
            # selecting an arbitrary button.
            await asyncio.sleep(0.25)
            latest = await _observe(agent_id, radius=radius)
            continue
        if ui.get("activeScript") and ui.get("activeScriptExecution") == 3:
            try:
                await resume_dialogue(agent_id)
            except LostCityApiError as error:
                if "no_paused_dialogue" not in str(error):
                    raise
        elif ui.get("activeScript") or (latest.get("agent") or {}).get("targetOperation") is not None:
            await asyncio.sleep(0.25)
        else:
            return latest
        await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius=radius)
    return latest


async def _quest_search_step(agent_id: str, step: dict[str, Any], radius: int) -> dict[str, Any]:
    observation = await _observe(agent_id, radius=max(16, radius))
    if _carried_item_tokens(observation, step["item_tokens"]) is not None:
        return {
            "status": "complete",
            "validated": True,
            "validation": "quest_search_item_already_present",
            "observation": observation,
        }
    position = observation["agent"]["position"]
    if isinstance(step.get("x"), int) and (
        position.get("level") != step.get("level", 0)
        or max(abs(position.get("x", 0) - step["x"]), abs(position.get("z", 0) - step["z"])) > 16
    ):
        moved = await _travel_quest_step(
            agent_id,
            {"kind": "travel", "x": step["x"], "z": step["z"], "level": step.get("level", 0), "run": True},
            max(16, radius),
        )
        if moved.get("status") != "complete":
            return {**moved, "phase": "quest_search_area_route"}
        observation = moved.get("observation") or await _observe(agent_id, radius=max(16, radius))

    excluded_ids: set[int] = set()
    last_result: dict[str, Any] | None = None
    for _ in range(20):
        observation = await _observe(agent_id, radius=max(16, radius))
        if _carried_item_tokens(observation, step["item_tokens"]) is not None:
            return {
                "status": "complete",
                "validated": True,
                "validation": "quest_search_item_observed_on_live_server",
                "attempts": len(excluded_ids) + 1,
                "result": last_result,
                "observation": observation,
            }
        collection = {"npc": "npcs", "loc": "locations", "obj": "objects"}[step["target_kind"]]
        candidates = [
            entity
            for entity in observation.get("nearby", {}).get(collection, [])
            if entity.get("id") not in excluded_ids
            and (entity.get("name") or "").casefold() == step["target_name"].casefold()
        ]
        filtered = {**observation, "nearby": {**(observation.get("nearby") or {}), collection: candidates}}
        selected = _discover_skill_action(
            filtered,
            {
                "targets": [step["target_kind"]],
                "target_name": step["target_name"],
                "option_tokens": [token.casefold() for token in step["option_tokens"]],
            },
        )
        if selected is None:
            break
        action = await api.request(
            "POST",
            f"/agents/{agent_id}/actions",
            json={"type": "interact", "target": selected["target"], "option": selected["option"]},
        )
        last_result = await _resolve_interaction_action(
            agent_id,
            action,
            max(16, radius),
            before_observation=observation,
            target=selected["target"],
        )
        settled = await _wait_for_quest_item_or_content_script(
            agent_id,
            last_result.get("observation") or observation,
            step["item_tokens"],
            max(16, radius),
        )
        if _carried_item_tokens(settled, step["item_tokens"]) is not None:
            return {
                "status": "complete",
                "validated": True,
                "validation": "quest_search_item_observed_on_live_server",
                "attempts": len(excluded_ids) + 1,
                "result": last_result,
                "observation": settled,
            }
        excluded_ids.add(int(selected["target"]["id"]))
    latest = await _observe(agent_id, radius=max(16, radius))
    return {
        "status": "blocked",
        "validated": False,
        "reason": "quest search exhausted live crate targets without finding the required item",
        "attempts": len(excluded_ids),
        "result": last_result,
        "observation": latest,
    }


async def _floor_transition_quest_step(agent_id: str, step: dict[str, Any], radius: int) -> dict[str, Any]:
    observation = await _observe(agent_id, radius=max(8, radius))
    position = observation["agent"]["position"]
    if position["level"] == step["to_level"]:
        return {
            "status": "complete",
            "validated": True,
            "validation": "floor_already_at_destination",
            "observation": observation,
        }
    if position["level"] != step["from_level"]:
        return {
            "status": "blocked",
            "validated": False,
            "reason": "floor_transition_started_on_unexpected_level",
            "expected_level": step["from_level"],
            "observation": observation,
        }

    approach = {
        "kind": "travel",
        "x": step["approach_x"],
        "z": step["approach_z"],
        "level": step["from_level"],
        "run": True,
    }
    if (position["x"], position["z"]) != (step["approach_x"], step["approach_z"]):
        moved = await _travel_quest_step(agent_id, approach, radius)
        moved_observation = moved.get("observation") or observation
        moved_position = (moved_observation.get("agent") or {}).get("position") or {}
        # A staircase/ladder target tile is often occupied by the object
        # itself, so the collision-aware route quite correctly ends on the
        # adjacent operable tile. Do not keep re-planning toward the blocked
        # footprint; let the native interaction approach the live target.
        within_interaction_range = (
            moved_position.get("level") == step["from_level"]
            and max(
                abs(moved_position.get("x", 0) - step["x"]),
                abs(moved_position.get("z", 0) - step["z"]),
            ) <= 1
        )
        if moved.get("status") != "complete" and not within_interaction_range:
            return {**moved, "phase": "floor_transition_approach"}
        observation = moved_observation if within_interaction_range else await _observe(agent_id, radius=max(8, radius))

    collection = {"loc": "locations", "obj": "objects"}[step.get("target_kind", "loc")]
    target_name = step.get("target_name", "Staircase").casefold()
    selected: dict[str, Any] | None = None
    for entity in observation.get("nearby", {}).get(collection, []):
        entity_position = entity.get("position") or {}
        if (
            (entity.get("name") or "").casefold() == target_name
            and entity.get("id") is not None
            and entity_position.get("level") == step["from_level"]
            and entity_position.get("x") == step["x"]
            and entity_position.get("z") == step["z"]
        ):
            match = _option_match(entity.get("options") or [], [token.casefold() for token in step["option_tokens"]])
            if match is not None:
                selected = {
                    "target": {
                        "kind": step.get("target_kind", "loc"),
                        "id": entity["id"],
                        "x": step["x"],
                        "z": step["z"],
                        "level": step["from_level"],
                    },
                    "option": match[0],
                    "label": match[1],
                    "entity": entity,
                }
                break
    if selected is None:
        return {
            "status": "blocked",
            "validated": False,
            "reason": "floor_transition_target_not_found_or_option_not_visible",
            "observation": observation,
        }

    action = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "interact", "target": selected["target"], "option": selected["option"]},
    )
    latest = action.get("observation") or observation
    for _ in range(60):
        latest = await _observe(agent_id, radius=1)
        safety = latest.get("safety") or {}
        if safety.get("lowHealth") or safety.get("fleeing"):
            return {"status": "paused_low_health", "safety": safety, "action": action, "observation": latest}
        if latest["agent"]["position"]["level"] == step["to_level"]:
            return {
                "status": "complete",
                "validated": True,
                "validation": "floor_transition_observed_on_live_server",
                "selected": selected,
                "action": action,
                "observation": latest,
            }
        await asyncio.sleep(0.25)
    return {
        "status": "blocked",
        "validated": False,
        "validation": "floor_transition_timeout",
        "reason": "the live server did not report the requested floor change",
        "selected": selected,
        "action": action,
        "observation": latest,
    }


async def _resume_dialogue_quest_step(agent_id: str, radius: int) -> dict[str, Any]:
    before = await _observe(agent_id, radius=radius)
    ui = before.get("ui") or {}
    if not ui.get("activeScript"):
        return {
            "status": "complete",
            "validated": True,
            "validation": "dialogue_already_resolved",
            "observation": before,
        }
    if ui.get("resumeButtons"):
        return {
            "status": "blocked",
            "validated": False,
            "reason": "dialogue_choice_requires_explicit_option_step",
            "observation": before,
        }
    action = await resume_dialogue(agent_id)
    before_ui = {key: ui.get(key) for key in ("activeScript", "activeScriptExecution", "resumeButtons", "lastComponent")}
    latest = action.get("observation") or before
    for _ in range(40):
        await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius=radius)
        latest_ui = latest.get("ui") or {}
        after_ui = {key: latest_ui.get(key) for key in ("activeScript", "activeScriptExecution", "resumeButtons", "lastComponent")}
        if after_ui != before_ui or not latest_ui.get("activeScript"):
            return {
                "status": "complete",
                "validated": True,
                "validation": "dialogue_resume_observed_on_live_server",
                "action": action,
                "observation": latest,
            }
    return {
        "status": "blocked",
        "validated": False,
        "validation": "dialogue_resume_timeout",
        "reason": "the live dialogue did not advance before the validation timeout",
        "action": action,
        "observation": latest,
    }


async def _quest_start_step(agent_id: str, step: dict[str, Any], radius: int) -> dict[str, Any]:
    quest = await _live_quest_state(agent_id, step["quest_id"])
    if quest is None:
        return {
            "status": "blocked",
            "validated": False,
            "reason": "quest_not_registered_by_running_engine",
            "quest_id": step["quest_id"],
            "observation": await _observe(agent_id, radius),
        }
    if quest.get("status") in {"in_progress", "complete"}:
        return {
            "status": "complete",
            "validated": True,
            "validation": "quest_start_already_authoritative",
            "quest": quest,
            "observation": await _observe(agent_id, radius),
        }

    observation = await _observe(agent_id, radius)
    selected = _quest_interaction_action(observation, step)
    if selected is None and isinstance(step.get("x"), int) and isinstance(step.get("z"), int):
        moved = await _travel_quest_step(
            agent_id,
            {"kind": "travel", "x": step["x"], "z": step["z"], "level": step.get("level", 0), "run": True},
            max(16, radius),
        )
        moved_observation = moved.get("observation") or observation
        moved_position = (moved_observation.get("agent") or {}).get("position") or {}
        within_interaction_range = (
            moved_position.get("level", 0) == step.get("level", 0)
            and max(abs(moved_position.get("x", 0) - step["x"]), abs(moved_position.get("z", 0) - step["z"])) <= 1
        )
        if moved.get("status") != "complete" and not within_interaction_range:
            return {**moved, "phase": "quest_start_target_route", "observation": moved.get("observation") or observation}
        observation = moved_observation if within_interaction_range else moved.get("observation") or await _observe(agent_id, radius)
        selected = _quest_interaction_action(observation, step)
    if selected is None:
        return {"status": "blocked", "validated": False, "reason": "quest_start_target_not_found", "observation": observation}
    entity_position = (selected.get("entity") or {}).get("position") or {}
    if selected.get("distance", 0) > 1 and entity_position:
        approached = await _travel_quest_step(
            agent_id,
            {
                "kind": "travel",
                "x": int(entity_position["x"]),
                "z": int(entity_position["z"]),
                "level": int(entity_position.get("level", step.get("level", 0))),
                "run": True,
            },
            max(16, radius),
        )
        approached_observation = approached.get("observation") or observation
        approached_position = (approached_observation.get("agent") or {}).get("position") or {}
        distance = max(
            abs(approached_position.get("x", 0) - int(entity_position["x"])),
            abs(approached_position.get("z", 0) - int(entity_position["z"])),
        )
        if approached.get("status") != "complete" and distance > 1:
            return {**approached, "phase": "quest_start_entity_approach", "observation": approached_observation}
        observation = approached_observation
        selected = _quest_interaction_action(observation, step)
        if selected is None:
            return {
                "status": "blocked",
                "validated": False,
                "reason": "quest_start_target_lost_after_approach",
                "observation": observation,
            }
    action = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "interact", "target": selected["target"], "option": selected["option"]},
    )
    latest = action.get("observation") or observation
    for _ in range(120):
        quest = await _live_quest_state(agent_id, step["quest_id"])
        if quest and quest.get("status") in {"in_progress", "complete"}:
            return {
                "status": "complete",
                "validated": True,
                "validation": "quest_start_observed_on_live_server",
                "quest": quest,
                "selected": selected,
                "action": action,
                "observation": latest,
            }
        ui = latest.get("ui") or {}
        buttons = ui.get("resumeButtons") or []
        if buttons:
            choice = next(
                (text for text in step["dialogue_options"] if _select_dialogue_button(latest, text) is not None),
                None,
            )
            if choice is None:
                return {
                    "status": "blocked",
                    "validated": False,
                    "reason": "quest_start_dialogue_options_unexpected",
                    "dialogue_options": buttons,
                    "selected": selected,
                    "action": action,
                    "observation": latest,
                }
            choice_result = await choose_dialogue_option(agent_id, choice, radius=radius)
            if choice_result.get("status") != "complete":
                return {**choice_result, "phase": "quest_start_dialogue", "selected": selected}
            latest = choice_result.get("observation") or latest
        elif ui.get("activeScript"):
            try:
                resumed = await resume_dialogue(agent_id)
                latest = resumed.get("observation") or latest
            except LostCityApiError as error:
                if "no_paused_dialogue" not in str(error):
                    raise
                await asyncio.sleep(0.25)
                latest = await _observe(agent_id, radius)
        else:
            await asyncio.sleep(0.25)
            latest = await _observe(agent_id, radius)
    return {
        "status": "blocked",
        "validated": False,
        "validation": "quest_start_timeout",
        "reason": "quest state did not become in progress after the live dialogue",
        "selected": selected,
        "action": action,
        "observation": latest,
    }


async def _quest_turn_in_step(agent_id: str, step: dict[str, Any], radius: int) -> dict[str, Any]:
    quest = await _live_quest_state(agent_id, step["quest_id"])
    if quest is None:
        return {"status": "blocked", "validated": False, "reason": "quest_not_registered_by_running_engine"}
    if quest.get("status") == "complete":
        return {"status": "complete", "validated": True, "validation": "quest_already_complete", "quest": quest, "observation": await _observe(agent_id, radius)}
    latest: dict[str, Any] = await _observe(agent_id, radius)
    selected: dict[str, Any] | None = None
    action: dict[str, Any] | None = None
    for interaction_attempt in range(4):
        quest = await _live_quest_state(agent_id, step["quest_id"])
        if quest and quest.get("status") == "complete":
            return {
                "status": "complete",
                "validated": True,
                "validation": "quest_completion_observed_on_live_server",
                "quest": quest,
                "selected": selected,
                "action": action,
                "observation": latest,
            }
        latest = await _observe(agent_id, radius)
        selected = _quest_interaction_action(latest, step)
        if selected is None and isinstance(step.get("x"), int) and isinstance(step.get("z"), int):
            moved = await _travel_quest_step(
                agent_id,
                {"kind": "travel", "x": step["x"], "z": step["z"], "level": step.get("level", 0), "run": True},
                max(16, radius),
            )
            moved_observation = moved.get("observation") or latest
            moved_position = (moved_observation.get("agent") or {}).get("position") or {}
            within_interaction_range = (
                moved_position.get("level", 0) == step.get("level", 0)
                and max(abs(moved_position.get("x", 0) - step["x"]), abs(moved_position.get("z", 0) - step["z"])) <= 1
            )
            if moved.get("status") != "complete" and not within_interaction_range:
                return {**moved, "phase": "quest_turn_in_target_route", "observation": moved_observation}
            latest = moved_observation
            selected = _quest_interaction_action(latest, step)
        if selected is None:
            return {"status": "blocked", "validated": False, "reason": "quest_turn_in_target_not_found", "observation": latest}
        if selected.get("distance", 0) > 2 and selected.get("entity", {}).get("position"):
            target_position = selected["entity"]["position"]
            approached = await _travel_quest_step(
                agent_id,
                {
                    "kind": "travel",
                    "x": target_position["x"],
                    "z": target_position["z"],
                    "level": target_position.get("level", 0),
                    "run": True,
                },
                max(16, radius),
            )
            approached_observation = approached.get("observation") or latest
            approached_position = (approached_observation.get("agent") or {}).get("position") or {}
            distance = max(
                abs(approached_position.get("x", 0) - target_position["x"]),
                abs(approached_position.get("z", 0) - target_position["z"]),
            )
            if approached.get("status") != "complete" and distance > 1:
                return {**approached, "phase": "quest_turn_in_npc_approach"}
            latest = approached_observation
            selected = _quest_interaction_action(latest, step)
            if selected is None:
                continue
        action = await api.request(
            "POST",
            f"/agents/{agent_id}/actions",
            json={"type": "interact", "target": selected["target"], "option": selected["option"]},
        )
        latest = action.get("observation") or latest
        for _ in range(240):
            quest = await _live_quest_state(agent_id, step["quest_id"])
            if quest and quest.get("status") == "complete":
                return {
                    "status": "complete",
                    "validated": True,
                    "validation": "quest_completion_observed_on_live_server",
                    "quest": quest,
                    "selected": selected,
                    "action": action,
                    "interaction_attempt": interaction_attempt + 1,
                    "observation": latest,
                }
            ui = latest.get("ui") or {}
            if ui.get("resumeButtons") or ui.get("activeScript") or (latest.get("agent") or {}).get("targetOperation") is not None:
                try:
                    if ui.get("resumeButtons"):
                        resumed = await resume_dialogue(agent_id)
                        latest = resumed.get("observation") or latest
                    elif ui.get("activeScript"):
                        resumed = await resume_dialogue(agent_id)
                        latest = resumed.get("observation") or latest
                    else:
                        await asyncio.sleep(0.25)
                        latest = await _observe(agent_id, radius)
                except LostCityApiError as error:
                    if "no_paused_dialogue" not in str(error):
                        raise
                    await asyncio.sleep(0.25)
                    latest = await _observe(agent_id, radius)
            else:
                # The live script finished an interaction without completing
                # the quest. Re-observe and dispatch Fred again; Sheep Shearer
                # hands over one or more balls per talk and can require a final
                # confirmation after state 20.
                break
        else:
            continue
    return {
        "status": "blocked",
        "validated": False,
        "validation": "quest_turn_in_timeout",
        "reason": "the engine did not report quest completion before the validation timeout",
        "selected": selected,
        "action": action,
        "observation": latest,
    }


async def _crafting_loop_step(agent_id: str, step: dict[str, Any], radius: int) -> dict[str, Any]:
    target_count = step["target_count"]
    count_mode = step.get("count_mode", "inventory")
    discard_tokens = tuple(token.casefold() for token in step.get("discard_tokens", []))
    source = step["source"]
    station = step["station"]
    staircase = step["staircase"]
    progress: list[dict[str, Any]] = []
    produced_count = 0

    def goal_reached(ball_count: int) -> bool:
        return produced_count >= target_count if count_mode == "produced" else ball_count >= target_count

    if step.get("quest_id"):
        live_quest = await _live_quest_state(agent_id, step["quest_id"])
        if live_quest and int(live_quest.get("state", 0)) >= 20:
            observation = await _observe(agent_id, radius=max(16, radius))
            return {
                "status": "complete",
                "validated": True,
                "validation": "quest_progress_reached_turn_in_state_on_live_server",
                "count": _inventory_count_exact(observation, "Ball of wool"),
                "quest": live_quest,
                "progress": progress,
                "observation": observation,
            }

    async def transition(from_level: int, to_level: int, tokens: list[str]) -> dict[str, Any]:
        return await _floor_transition_quest_step(
            agent_id,
            {
                "kind": "floor_transition",
                "x": staircase["x"],
                "z": staircase["z"],
                "approach_x": staircase["approach_x"],
                "approach_z": staircase["approach_z"],
                "from_level": from_level,
                "to_level": to_level,
                "option_tokens": tokens,
                "target_kind": "loc",
                "target_name": "Staircase",
            },
            radius,
        )

    async def travel_to_station(count: int, phase: str) -> dict[str, Any]:
        """Open the live castle boundaries and reach the upstairs wheel."""

        # A worker that just spun Wool is already inside the castle. Prefer
        # the direct staircase route in that case; the access checkpoints are
        # only needed when entering from the sheep pen.
        direct_stair = await _travel_quest_step(
            agent_id,
            {"kind": "travel", "x": staircase["approach_x"], "z": staircase["approach_z"], "level": source["level"], "run": True},
            max(16, radius),
        )

        # The castle route contains two real collision boundaries. Reach each
        # outside/inside checkpoint and open only the live door exposed there;
        # this keeps the route deterministic and makes door state resumable
        # after a worker restart.
        if direct_stair.get("status") != "complete":
            for access_index, checkpoint in enumerate(staircase.get("access", [])):
                access_destination = {
                    "kind": "travel",
                    "x": checkpoint["x"],
                    "z": checkpoint["z"],
                    "level": checkpoint.get("level", source["level"]),
                    "run": True,
                }
                if checkpoint.get("via"):
                    access_destination["via"] = checkpoint["via"]
                access_route = await _travel_quest_step(agent_id, access_destination, max(16, radius))
                if access_route.get("status") != "complete":
                    return {
                        **access_route,
                        "phase": phase,
                        "access_index": access_index,
                        "count": count,
                        "progress": progress,
                    }
                access_observation = access_route.get("observation") or await _observe(agent_id, radius=max(16, radius))
                # The travel helper observes a wider radius so it can discover
                # collision boundaries. Narrow the interaction discovery back to
                # this checkpoint; otherwise a similarly named closed door farther
                # inside the map can win after the intended door is already open.
                nearby = access_observation.get("nearby") or {}
                collection = {"loc": "locations", "obj": "objects"}[checkpoint.get("target_kind", "loc")]
                checkpoint_entities = [
                    entity
                    for entity in nearby.get(collection, [])
                    if max(
                        abs((entity.get("position") or {}).get("x", 0) - checkpoint["x"]),
                        abs((entity.get("position") or {}).get("z", 0) - checkpoint["z"]),
                    ) <= 4
                ]
                access_observation = {
                    **access_observation,
                    "nearby": {**nearby, collection: checkpoint_entities},
                }
                access_action = _discover_skill_action(
                    access_observation,
                    {
                        "targets": [checkpoint.get("target_kind", "loc")],
                        "target_name": checkpoint["target_name"],
                        "option_tokens": [token.casefold() for token in checkpoint["option_tokens"]],
                    },
                )
                if access_action is None:
                    # An already-open door no longer exposes the Open option. The
                    # checkpoint is still valid because the live route reached it.
                    continue
                access_request = await api.request(
                    "POST",
                    f"/agents/{agent_id}/actions",
                    json={"type": "interact", "target": access_action["target"], "option": access_action["option"]},
                )
                access_result = await _resolve_interaction_action(
                    agent_id,
                    access_request,
                    max(16, radius),
                    before_observation=access_observation,
                    target=access_action["target"],
                )
                if access_result.get("status") != "complete" or not access_result.get("validated"):
                    return {
                        **access_result,
                        "phase": f"{phase}_interaction",
                        "access_index": access_index,
                        "count": count,
                        "progress": progress,
                    }

        ground_to_stair = direct_stair if direct_stair.get("status") == "complete" else await _travel_quest_step(
            agent_id,
            {"kind": "travel", "x": staircase["approach_x"], "z": staircase["approach_z"], "level": source["level"], "run": True},
            max(16, radius),
        )
        if ground_to_stair.get("status") != "complete":
            return {**ground_to_stair, "phase": f"{phase}_staircase", "count": count, "progress": progress}
        up = await transition(source["level"], station["level"], ["climb-up", "climb"])
        if up.get("status") != "complete":
            return {**up, "phase": f"{phase}_floor_transition", "count": count, "progress": progress}

        station_destination = {
            "kind": "travel",
            "x": station.get("approach_x", station["x"]),
            "z": station.get("approach_z", station["z"]),
            "level": station["level"],
            "run": True,
        }
        station_route = await _travel_quest_step(agent_id, station_destination, max(16, radius))
        if station_route.get("status") != "complete":
            return {**station_route, "phase": f"{phase}_route", "count": count, "progress": progress}
        return {"status": "complete", "validated": True, "observation": station_route.get("observation")}

    async def spin_at_station(ball_count: int, phase: str = "spin") -> dict[str, Any]:
        nonlocal produced_count
        station_action = await skill_step(agent_id, "crafting", radius=max(16, radius))
        after_station = station_action.get("observation") or await _observe(agent_id, radius=max(16, radius))
        new_count = _inventory_count_exact(after_station, "Ball of wool")
        produced_count += max(0, new_count - ball_count)
        progress.append({"iteration": iteration, "phase": phase, "count_before": ball_count, "count_after": new_count, "result": station_action})
        if station_action.get("status") != "action_queued" or not station_action.get("validated") or new_count <= ball_count:
            return {"status": "blocked", "validated": False, "reason": "spinning_step_not_validated", "count": new_count, "progress": progress, "observation": after_station}
        return {"status": "complete", "validated": True, "count": new_count, "observation": after_station}

    async def spin_once(ball_count: int, phase: str = "spin") -> dict[str, Any]:
        station_route = await travel_to_station(ball_count, phase)
        if station_route.get("status") != "complete":
            return station_route
        return await spin_at_station(ball_count, phase)

    for iteration in range(target_count + 2):
        observation = await _observe(agent_id, radius=max(16, radius))
        ball_count = _inventory_count_exact(observation, "Ball of wool")
        if goal_reached(ball_count):
            if observation["agent"]["position"]["level"] == source["level"]:
                return {
                    "status": "complete",
                    "validated": True,
                    "validation": "crafting_target_count_observed_on_live_server",
                    "count": ball_count,
                    "produced_count": produced_count,
                    "progress": progress,
                    "observation": observation,
                }
            if observation["agent"]["position"]["level"] == station["level"]:
                down = await transition(station["level"], source["level"], ["climb-down", "climb"])
                if down.get("status") != "complete":
                    return {**down, "phase": "crafting_final_floor_transition", "count": ball_count, "progress": progress}
                observation = down.get("observation") or await _observe(agent_id, radius=max(16, radius))
                return {
                    "status": "complete",
                    "validated": True,
                    "validation": "crafting_target_count_observed_on_live_server",
                    "count": ball_count,
                    "produced_count": produced_count,
                    "progress": progress,
                    "observation": observation,
                }
            return {"status": "blocked", "validated": False, "reason": "crafting_loop_on_unsupported_floor", "observation": observation}

        current_level = observation["agent"]["position"]["level"]
        # Stay on the station floor while there is still live Wool to consume.
        # Descending after every spin forces a reverse route through the castle
        # and can strand a worker at the interior door even though the next
        # validated action is available beside the wheel.
        if current_level == station["level"] and _inventory_count_exact(observation, "Wool") > 0:
            spun = await spin_at_station(ball_count, phase="spin_at_station")
            if spun.get("status") != "complete":
                return spun
            continue
        if current_level == station["level"]:
            down = await transition(station["level"], source["level"], ["climb-down", "climb"])
            if down.get("status") != "complete":
                return {**down, "phase": "crafting_to_source", "count": ball_count, "progress": progress}
            observation = down.get("observation") or await _observe(agent_id, radius=max(16, radius))
        elif current_level != source["level"]:
            return {"status": "blocked", "validated": False, "reason": "crafting_loop_on_unsupported_floor", "observation": observation}

        wool_count = _inventory_count_exact(observation, "Wool")
        inventory = next(
            (current for current in observation.get("agent", {}).get("inventories", []) if current.get("id") == 93),
            {},
        )
        if wool_count == 0 and int(inventory.get("freeSlots", 0)) <= 0 and discard_tokens:
            remaining = target_count - (produced_count if count_mode == "produced" else ball_count)
            reserve_slots = min(5, max(1, remaining))
            discarded = False
            for _ in range(reserve_slots):
                latest = await _observe(agent_id, radius=max(16, radius))
                live_inventory = next(
                    (current for current in latest.get("agent", {}).get("inventories", []) if current.get("id") == 93),
                    {},
                )
                if int(live_inventory.get("freeSlots", 0)) >= reserve_slots:
                    break
                discard = next(
                    (
                        item
                        for item in live_inventory.get("items", [])
                        if (item.get("name") or "").casefold() in discard_tokens
                    ),
                    None,
                )
                if discard is None:
                    break
                dropped = await drop_item(agent_id, int(discard["slot"]), inventory=93)
                latest = dropped.get("observation") or await _observe(agent_id, radius=max(16, radius))
                progress.append({"iteration": iteration, "phase": "discard_batch_output", "item": discard, "result": dropped})
                if dropped.get("status") != "complete" or not dropped.get("validated"):
                    return {
                        "status": "blocked",
                        "validated": False,
                        "reason": "crafting_batch_discard_not_validated",
                        "count": ball_count,
                        "produced_count": produced_count,
                        "progress": progress,
                        "observation": latest,
                    }
                discarded = True
            if discarded:
                continue
        if wool_count == 0 and int(inventory.get("freeSlots", 0)) <= 0:
            return {
                "status": "blocked",
                "validated": False,
                "reason": "crafting_inventory_full_requires_drop_or_bank_policy",
                "count": ball_count,
                "produced_count": produced_count,
                "progress": progress,
                "observation": observation,
            }
        # Prefer consuming live Wool before shearing again when the inventory
        # is full or the remaining Wool can finish the target. This avoids
        # attempting an item-on-NPC action with no slot for its result.
        if wool_count > 0 and (inventory.get("freeSlots", 0) == 0 or ball_count + wool_count >= target_count):
            spun = await spin_once(ball_count, phase="spin_existing_wool")
            if spun.get("status") != "complete":
                return spun
            continue

        source_route = await api.request(
            "POST",
            f"/agents/{agent_id}/route",
            json={"x": source["x"], "z": source["z"], "level": source["level"], "maxLegs": 32},
        )
        if not source_route.get("reached") and source.get("via"):
            for waypoint in source["via"]:
                waypoint_result = await _travel_quest_step(
                    agent_id,
                    {"kind": "travel", **waypoint, "run": True},
                    max(16, radius),
                )
                if waypoint_result.get("status") != "complete":
                    return {**waypoint_result, "phase": "crafting_source_waypoint", "count": ball_count, "progress": progress}
                gate_observation = waypoint_result.get("observation") or await _observe(agent_id, radius=max(16, radius))
                gate_action = _discover_skill_action(
                    gate_observation,
                    {"targets": ["loc"], "target_name": "Gate", "option_tokens": ["open"]},
                )
                if gate_action is not None:
                    gate_request = await api.request(
                        "POST",
                        f"/agents/{agent_id}/actions",
                        json={"type": "interact", "target": gate_action["target"], "option": gate_action["option"]},
                    )
                    gate_result = await _resolve_interaction_action(
                        agent_id,
                        gate_request,
                        max(16, radius),
                        before_observation=gate_observation,
                        target=gate_action["target"],
                    )
                    if gate_result.get("status") != "complete" or not gate_result.get("validated"):
                        return {**gate_result, "phase": "crafting_source_gate", "count": ball_count, "progress": progress}
        source_step = {key: value for key, value in source.items() if key != "via"}
        moved = await _travel_quest_step(
            agent_id,
            {"kind": "travel", **source_step, "run": True},
            max(16, radius),
        )
        if moved.get("status") != "complete":
            return {**moved, "phase": "crafting_source_route", "count": ball_count, "progress": progress}
        source_action = await skill_step(agent_id, "crafting", radius=max(16, radius))
        progress.append({"iteration": iteration, "phase": "shear", "count_before": ball_count, "result": source_action})
        if source_action.get("status") != "action_queued" or not source_action.get("validated"):
            return {"status": "blocked", "validated": False, "reason": "shearing_step_not_validated", "count": ball_count, "progress": progress, "observation": source_action.get("observation")}
        # Keep gathering until the bounded inventory batch is full enough to
        # finish the target, then make one trip to the wheel and process the
        # batch there. The previous one-shear/one-spin cycle was correct but
        # forced a full castle round trip for every Wool item, which made
        # long-running Crafting and Sheep Shearer execution unnecessarily
        # expensive in both server ticks and MCP calls.
        continue

    observation = await _observe(agent_id, radius=max(16, radius))
    return {"status": "blocked", "validated": False, "reason": "crafting_loop_iteration_limit", "count": _inventory_count_exact(observation, "Ball of wool"), "progress": progress, "observation": observation}


async def _combat_loop_step(agent_id: str, step: dict[str, Any], radius: int) -> dict[str, Any]:
    """Run bounded live combat chunks with food and optional loot pickup."""

    npc_name = step.get("npc_name")
    loot_tokens = tuple(token.casefold() for token in step.get("loot_tokens", []))
    pickup_tokens = tuple(token.casefold() for token in step.get("pickup_tokens", loot_tokens))
    loot_count = int(step.get("loot_count", step.get("count", 1))) if loot_tokens else 0
    kill_target = int(step.get("count", 1)) if not loot_tokens else int(step.get("max_kills", 64))
    max_kills = max(kill_target, int(step.get("max_kills", kill_target)))
    max_iterations = int(step.get("max_iterations", max_kills + 8))
    max_ticks = int(step.get("max_ticks", 160))
    food_threshold = float(step.get("food_threshold", 0.8))
    progress: list[dict[str, Any]] = []
    kills = 0
    foods_eaten = 0

    def loot_reached(observation: dict[str, Any]) -> bool:
        return bool(loot_tokens) and _carried_count_tokens(observation, list(loot_tokens)) >= loot_count

    latest = await _observe(agent_id, radius=max(16, radius))
    for iteration in range(max_iterations):
        if loot_reached(latest) or (not loot_tokens and kills >= kill_target):
            return {
                "status": "complete",
                "validated": True,
                "validation": "combat_loop_goal_observed_on_live_server",
                "kills": kills,
                "foods_eaten": foods_eaten,
                "progress": progress,
                "observation": latest,
            }

        safety = latest.get("safety") or {}
        if safety.get("lowHealth") or (
            safety.get("maxHealth", 0) > 0
            and safety.get("health", 0) / safety.get("maxHealth", 1) < food_threshold
        ):
            food = await consume_food(agent_id, minimum_health_ratio=food_threshold)
            progress.append({"iteration": iteration + 1, "phase": "eat", "status": food.get("status")})
            if food.get("status") != "food_queued":
                return {
                    "status": "paused_low_health" if safety.get("lowHealth") else "blocked",
                    "validated": False,
                    "reason": "combat_loop_has_no_live_food_available",
                    "kills": kills,
                    "foods_eaten": foods_eaten,
                    "progress": progress,
                    "observation": food.get("observation") or latest,
                }
            foods_eaten += 1
            await asyncio.sleep(0.5)
            latest = await _observe(agent_id, radius=max(16, radius))
            continue

        before_xp = _skill_state(latest, "attack")["experience"]
        combat = await _combat_quest_step(
            agent_id,
            {"kind": "combat", "npc_name": npc_name, "count": 1},
            max(16, radius),
            max_ticks,
        )
        latest = combat.get("observation") or await _observe(agent_id, radius=max(16, radius))
        progress.append(
            {
                "iteration": iteration + 1,
                "phase": "combat",
                "status": combat.get("status"),
                "validated": combat.get("status") == "complete",
                "kills": combat.get("kills", 0),
            }
        )
        if combat.get("status") == "complete":
            kills += int(combat.get("kills", 1))
            if pickup_tokens:
                for _ in range(4):
                    candidate = next(
                        (
                            obj
                            for obj in (latest.get("nearby") or {}).get("objects", [])
                            if any(token in (obj.get("name") or "").casefold() for token in pickup_tokens)
                        ),
                        None,
                    )
                    if candidate is None:
                        break
                    picked = await pickup_object(
                        agent_id,
                        object_id=int(candidate["id"]),
                        radius=max(16, radius),
                    )
                    progress.append(
                        {
                            "iteration": iteration + 1,
                            "phase": "pickup_loot",
                            "status": picked.get("status"),
                            "validated": picked.get("validated", False),
                            "item": candidate.get("name"),
                        }
                    )
                    latest = picked.get("observation") or await _observe(agent_id, radius=max(16, radius))
                    if picked.get("status") != "complete" or not picked.get("validated"):
                        break
            else:
                latest = await _observe(agent_id, radius=max(16, radius))
            if loot_reached(latest) or (not loot_tokens and kills >= kill_target):
                return {
                    "status": "complete",
                    "validated": True,
                    "validation": "combat_loop_goal_observed_on_live_server",
                    "kills": kills,
                    "foods_eaten": foods_eaten,
                    "progress": progress,
                    "observation": latest,
                }
            continue
        if combat.get("status") == "paused_low_health":
            latest = await _observe(agent_id, radius=max(16, radius))
            continue
        if _skill_state(latest, "attack")["experience"] <= before_xp:
            return {
                "status": "blocked",
                "validated": False,
                "reason": combat.get("reason", "combat chunk produced no live combat progress"),
                "kills": kills,
                "foods_eaten": foods_eaten,
                "progress": progress,
                "observation": latest,
            }

    return {
        "status": "step_limit",
        "validated": kills > 0 or foods_eaten > 0,
        "validation": "bounded_combat_progress_observed_on_live_server" if kills > 0 or foods_eaten > 0 else "combat_loop_limit_without_goal",
        "reason": "combat loop limit reached before its loot or kill goal",
        "kills": kills,
        "foods_eaten": foods_eaten,
        "progress": progress,
        "observation": latest,
    }


async def _recover_health_step(agent_id: str, step: dict[str, Any], radius: int) -> dict[str, Any]:
    """Wait for the live health watchdog to recover an idle player safely."""

    target_ratio = float(step.get("target_ratio", 1.0))
    max_wait_seconds = float(step.get("max_wait_seconds", 120.0))
    if not 0 < target_ratio <= 1:
        raise ValueError("target_ratio must be greater than 0 and at most 1")
    if not 0 < max_wait_seconds <= 600:
        raise ValueError("max_wait_seconds must be greater than 0 and at most 600")

    latest = await _observe(agent_id, radius=max(1, radius))
    initial_health = int((latest.get("safety") or {}).get("health", 0))
    initial_max_health = int((latest.get("safety") or {}).get("maxHealth", 0))
    deadline = asyncio.get_running_loop().time() + max_wait_seconds
    while True:
        safety = latest.get("safety") or {}
        health = int(safety.get("health", 0))
        maximum = int(safety.get("maxHealth", initial_max_health))
        if maximum > 0 and health / maximum >= target_ratio and not (latest.get("agent") or {}).get("targetOperation"):
            return {
                "status": "complete",
                "validated": True,
                "validation": "health_recovery_observed_on_live_server",
                "initial_health": initial_health,
                "health": health,
                "max_health": maximum,
                "observation": latest,
            }
        if asyncio.get_running_loop().time() >= deadline:
            return {
                "status": "step_limit",
                "validated": False,
                "progress_validated": health > initial_health,
                "validation": "partial_health_recovery_observed_on_live_server" if health > initial_health else "health_recovery_timeout",
                "reason": "the live player did not reach the requested recovery ratio before the wait limit",
                "initial_health": initial_health,
                "health": health,
                "max_health": maximum,
                "observation": latest,
            }
        await asyncio.sleep(0.5)
        latest = await _observe(agent_id, radius=max(1, radius))


async def _combat_quest_step(
    agent_id: str,
    step: dict[str, Any],
    radius: int,
    max_ticks: int,
) -> dict[str, Any]:
    wanted = step.get("npc_name")
    kills = 0
    active_target_id: int | None = None
    last_observation = await _observe(agent_id, radius)
    for _ in range(max_ticks):
        safety = last_observation.get("safety") or {}
        if safety.get("lowHealth") or safety.get("fleeing"):
            return {"status": "paused_low_health", "kills": kills, "safety": safety, "observation": last_observation}
        current_target = last_observation["agent"].get("target")
        if current_target and current_target.get("kind") == "npc":
            active_target_id = current_target.get("id")
        if active_target_id is None:
            try:
                attack = await attack_nearest_npc(agent_id, npc_name=wanted, radius=radius)
            except (LostCityApiError, ValueError) as error:
                return {"status": "blocked", "reason": str(error), "kills": kills, "observation": last_observation}
            active_target_id = attack["selected"]["id"]
        await asyncio.sleep(0.75)
        last_observation = await _observe(agent_id, radius)
        nearby_ids = {npc["id"] for npc in last_observation.get("nearby", {}).get("npcs", [])}
        current_target = last_observation["agent"].get("target")
        if active_target_id not in nearby_ids and (not current_target or current_target.get("id") != active_target_id):
            kills += 1
            active_target_id = None
            if kills >= step.get("count", 1):
                return {"status": "complete", "kills": kills, "observation": last_observation}
    return {"status": "step_limit", "kills": kills, "observation": last_observation}


async def _execute_quest_step(agent_id: str, step: dict[str, Any], radius: int) -> dict[str, Any]:
    kind = step["kind"]
    # The loop owns its food/safety recovery policy. Do not short-circuit it
    # before it can consume food and resume bounded combat chunks.
    if kind == "combat_loop":
        return await _combat_loop_step(agent_id, step, radius)
    observation = await _observe(agent_id, radius)
    safety = observation.get("safety") or {}
    if safety.get("lowHealth") or safety.get("fleeing"):
        return {"status": "paused_low_health", "safety": safety, "observation": observation}
    if kind == "travel":
        return await _travel_quest_step(agent_id, step, radius)
    if kind == "pickup":
        return await pickup_object(
            agent_id,
            object_id=step.get("object_id"),
            object_name=step.get("object_name"),
            radius=radius,
        )
    if kind == "item_on":
        item = _inventory_item(observation, tuple(step["item_tokens"]))
        if item is None:
            return {
                "status": "blocked",
                "validated": False,
                "reason": "required_item_not_found_in_live_inventory",
                "item_tokens": step["item_tokens"],
                "observation": observation,
            }
        target_kind = step["target_kind"]
        target_entity: dict[str, Any] | None = None
        collection = {"npc": "npcs", "loc": "locations", "obj": "objects"}.get(target_kind)
        if step.get("target_name") and collection:
            wanted = step["target_name"].casefold()
            target_entity = next(
                (
                    entity
                    for entity in observation.get("nearby", {}).get(collection, [])
                    if (entity.get("name") or "").casefold() == wanted
                ),
                None,
            )
            if target_entity is None:
                return {
                    "status": "blocked",
                    "validated": False,
                    "reason": "required_item_target_not_found_in_live_observation",
                    "target_name": step["target_name"],
                    "observation": observation,
                }
        target_id = target_entity.get("id") if target_entity else step.get("target_id")
        target_position = target_entity.get("position") if target_entity else None
        x = target_position.get("x") if target_position else step.get("x")
        z = target_position.get("z") if target_position else step.get("z")
        level = target_position.get("level") if target_position else step.get("level")
        result = await use_item_on_validated(
            agent_id,
            item["inventory"],
            item["item"]["slot"],
            target_kind,
            target_id=target_id,
            target_username=step.get("target_username"),
            x=x,
            z=z,
            level=level,
            radius=radius,
        )
        result["selected"] = {
            "item": item,
            "target": target_entity or {"id": target_id, "name": step.get("target_name")},
        }
        return result
    if kind == "combat":
        return await _combat_quest_step(agent_id, step, radius, max_ticks=step.get("max_ticks", 240))
    if kind == "train":
        return await train_skill(
            agent_id,
            step["skill"],
            step["target_level"],
            radius=radius,
            max_steps=step.get("max_steps", 100),
            step_delay_seconds=step.get("step_delay_seconds", 1.0),
        )
    if kind == "chat":
        action = await api.request(
            "POST",
            f"/agents/{agent_id}/actions",
            json={"type": "chat", "message": step["message"]},
        )
        return {"status": "complete", "action": action, "observation": action.get("observation")}
    if kind == "dialogue":
        return await choose_dialogue_option(agent_id, step["text"], radius=radius)
    if kind == "resume_dialogue":
        return await _resume_dialogue_quest_step(agent_id, radius)
    if kind == "quest_dialogue":
        return await _quest_dialogue_step(agent_id, step, radius)
    if kind == "quest_item_on":
        return await _quest_item_on_step(agent_id, step, radius)
    if kind == "quest_search":
        return await _quest_search_step(agent_id, step, radius)
    if kind == "combine":
        return await _combine_quest_items_step(agent_id, step, radius)
    if kind == "floor_transition":
        return await _floor_transition_quest_step(agent_id, step, radius)
    if kind == "quest_start":
        return await _quest_start_step(agent_id, step, radius)
    if kind == "crafting_loop":
        return await _crafting_loop_step(agent_id, step, radius)
    if kind == "quest_turn_in":
        return await _quest_turn_in_step(agent_id, step, radius)
    if kind == "interact":
        target_kind = step["target_kind"]
        target: dict[str, Any] = {"kind": target_kind, "id": step.get("target_id")}
        if target_kind == "player":
            target = {"kind": "player", "username": step["target_username"]}
        if target_kind in {"loc", "obj"}:
            target.update({"x": step["x"], "z": step["z"]})
            if "level" in step:
                target["level"] = step["level"]
        action = await api.request(
            "POST",
            f"/agents/{agent_id}/actions",
            json={"type": "interact", "target": target, "option": step["option"]},
        )
        return await _resolve_interaction_action(agent_id, action, radius, before_observation=observation)
    if kind == "discover_interact":
        targets = [step["target_kind"]] if step.get("target_kind") else ["npc", "loc", "obj"]
        selected = _discover_skill_action(
            observation,
            {
                "targets": targets,
                "target_name": step.get("target_name"),
                "option_tokens": [token.casefold() for token in step["option_tokens"]],
            },
        )
        if selected is None:
            return {"status": "blocked", "reason": "no_matching_live_option", "observation": observation}
        action = await api.request(
            "POST",
            f"/agents/{agent_id}/actions",
            json={"type": "interact", "target": selected["target"], "option": selected["option"]},
        )
        result = await _resolve_interaction_action(agent_id, action, radius, before_observation=observation)
        if step.get("accept_movement"):
            before_position = (observation.get("agent") or {}).get("position") or {}
            after_position = (result.get("observation", {}).get("agent") or {}).get("position") or {}
            moved = (
                before_position.get("level") == after_position.get("level")
                and (before_position.get("x"), before_position.get("z"))
                != (after_position.get("x"), after_position.get("z"))
            )
            if moved:
                result = {
                    **result,
                    "status": "complete",
                    "validated": True,
                    "validation": "movement_postcondition_observed_on_live_server",
                }
        result["selected"] = selected
        return result
    raise ValueError(f"unsupported quest step kind '{kind}'")


@mcp.tool
async def world_state() -> dict[str, Any]:
    """Read the current world tick, online player list, and NPC count."""

    return await api.request("GET", "/world")


@mcp.tool
async def list_players() -> list[dict[str, Any]]:
    """List players currently online in the Lost City world."""

    return await api.request("GET", "/players")


@mcp.tool
async def get_player(username: str) -> dict[str, Any]:
    """Read a player's position, skills, run energy, and inventories."""

    return await api.request("GET", f"/players/{username}")


def _mark_map_cache_dirty(agent_id: str) -> None:
    cached = _map_cache.get(agent_id)
    if cached is not None:
        cached["dirty"] = True


@mcp.tool
async def map_window(
    agent_id: str,
    radius: int = 32,
    refresh: bool = False,
    target_x: int | None = None,
    target_z: int | None = None,
    target_level: int | None = None,
) -> dict[str, Any]:
    """Read or reuse an authoritative collision window near an agent.

    The first read asks the engine for compact tile occupancy, cardinal exits,
    and nearby entities in one response. Later reads reuse the static geometry
    while refreshing only the small live observation, provided the agent is
    still inside the cached window. Door/ladder/item interactions mark the
    cache dirty, and callers can always force a full engine refresh.
    """

    if not isinstance(radius, int) or radius < 1 or radius > _MAX_MAP_CACHE_RADIUS:
        raise ValueError(f"radius must be between 1 and {_MAX_MAP_CACHE_RADIUS}")
    if (target_x is None) != (target_z is None):
        raise ValueError("target_x and target_z must be supplied together")
    if target_level is not None and target_level not in range(4):
        raise ValueError("target_level must be between 0 and 3")

    def attach_route(window: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        if target_x is None or target_z is None:
            return result
        return {
            **result,
            "cached_route": _cached_map_route(window, target_x, target_z, target_level),
        }

    cached = _map_cache.get(agent_id)
    if cached and not refresh and not cached.get("dirty", False):
        window = cached.get("window") or {}
        cached_radius = int(window.get("radius", 0))
        cached_center = window.get("center") or {}
        live_observation = await _observe(agent_id, radius=radius)
        live_position = (live_observation.get("agent") or {}).get("position") or {}
        within_cached_window = (
            cached_radius >= radius
            and live_position.get("level") == cached_center.get("level")
            and abs(int(live_position.get("x", 0)) - int(cached_center.get("x", 0))) <= max(1, cached_radius // 2)
            and abs(int(live_position.get("z", 0)) - int(cached_center.get("z", 0))) <= max(1, cached_radius // 2)
        )
        if within_cached_window:
            return attach_route(window, {
                **window,
                "status": "complete",
                "validated": True,
                "validation": "live_position_observation_with_cached_geometry",
                "observation": live_observation,
                "cache_hit": True,
                "cache_age_seconds": max(0.0, asyncio.get_running_loop().time() - cached["fetched_at"]),
                "cache_source": "mcp_static_collision_window",
                "geometry_warning": "dynamic doors and object changes require refresh; every action remains live-validated",
            })

    window = await api.request("GET", f"/agents/{agent_id}/map-window", params={"radius": radius})
    if not isinstance(window, dict):
        raise LostCityApiError("Lost City API returned an invalid map window")
    _map_cache[agent_id] = {
        "window": window,
        "fetched_at": asyncio.get_running_loop().time(),
        "dirty": False,
    }
    return attach_route(window, {
        **window,
        "status": "complete",
        "validated": True,
        "validation": "authoritative_live_engine_map_window",
        "cache_hit": False,
        "cache_age_seconds": 0.0,
        "cache_source": "live_engine_map_window",
        "geometry_warning": "dynamic doors and object changes require refresh; every action remains live-validated",
    })


@mcp.tool
async def skill_catalog() -> dict[str, Any]:
    """Describe every server skill and the MCP automation available for it.

    The catalog distinguishes interaction-driven skills, which can be trained
    automatically from live options, from item/UI-driven skills that need
    additional client primitives or an explicit setup step.
    """

    return {
        "skills": {
            skill: {
                "mode": spec["mode"],
                "targets": spec["targets"],
                "option_tokens": spec["option_tokens"],
                "strategy": spec["strategy"],
                "missing": spec.get("missing", []),
                "requirements": {
                    "entity": spec.get("entity_requirements", {}),
                    "category": spec.get("category_requirements", {}),
                },
            }
            for skill, spec in SKILL_PLAYBOOK.items()
        },
        "safety": "Every training loop observes the engine safety snapshot and pauses while the health watchdog is fleeing.",
    }


@mcp.tool
async def quest_catalog() -> dict[str, Any]:
    """List built-in quest scenarios and the declarative quest step grammar."""

    return {
        "formal_quest_registry": True,
        "note": "Quest definitions and persistent var state are read from the running content registry; completion is never inferred from a queued action.",
        "templates": QUEST_TEMPLATES,
        "step_kinds": {
            "travel": "Move to x/z/level using collision-aware route planning.",
            "interact": "Execute a known NPC/player/location/object option.",
            "discover_interact": "Find an entity exposing one of option_tokens and interact with it.",
            "pickup": "Find a live ground object exposing Take and validate its inventory pickup.",
            "item_on": "Find an item in live inventory, use it on a known or nearby named target, and validate the effect.",
            "combat": "Defeat count attackable NPCs, optionally filtered by npc_name.",
            "train": "Invoke the guarded skill trainer to a target level.",
            "chat": "Send a game chat message when dialogue content requires it.",
            "dialogue": "Choose a visible dialogue button by its live rendered text.",
            "quest_dialogue": "Drive live quest dialogue choices until the engine quest state reaches an expected value.",
            "quest_item_on": "Use a carried quest item on a live target and require the authoritative quest state to advance.",
            "quest_search": "Search live candidate entities until the required quest item appears in carried inventory.",
            "combine": "Use two carried inventory items together and validate the live output item.",
            "floor_transition": "Change floors through a live staircase or ladder interaction.",
            "quest_start": "Start a registered quest through its live NPC dialogue and verify in-progress state.",
            "quest_turn_in": "Complete a registered quest through its live turn-in dialogue and verify complete state.",
            "combat_loop": "Run bounded live combat chunks with food recovery and optional nearby loot pickup.",
        },
        "safety": "Quest execution pauses during a health escape and never marks a step complete by mutating state.",
    }


@mcp.tool
async def capability_audit(agent_id: str) -> dict[str, Any]:
    """Compare live content coverage with the engine/MCP action contract."""

    observation = await _observe(agent_id, radius=1)
    quest_result = await api.request("GET", f"/agents/{agent_id}/quests")
    quests = quest_result.get("quests", [])
    formal_ids = {quest.get("id") for quest in quests if quest.get("id")}
    graph_ids = {quest_id for quest_id in QUEST_TEMPLATES if quest_id in formal_ids}
    unmapped_quests = [
        {"id": quest.get("id"), "name": quest.get("name"), "state": quest.get("state"), "status": quest.get("status")}
        for quest in quests
        if quest.get("id") not in graph_ids
    ]
    skills = observation.get("agent", {}).get("skills", {})
    skill_coverage: dict[str, Any] = {}
    for skill, spec in SKILL_PLAYBOOK.items():
        current = skills.get(skill, {}).get("baseLevel", 0)
        skill_coverage[skill] = {
            "current_level": current,
            "target_level": 99,
            "mode": spec["mode"],
            "automatable_now": spec["mode"] in {"interaction", "combat"},
            "missing": spec.get("missing", []),
        }
    engine_actions_missing_from_mcp = sorted(ENGINE_ACTION_TYPES - set(MCP_ENGINE_ACTION_TOOLS))
    mcp_actions_without_engine_primitive = sorted(
        set(MCP_ENGINE_ACTION_TOOLS) - ENGINE_ACTION_TYPES
    )
    disabled_skills = [skill for skill, spec in SKILL_PLAYBOOK.items() if spec["mode"] == "disabled"]
    context_skills = [skill for skill, spec in SKILL_PLAYBOOK.items() if spec["mode"] == "context"]
    return {
        "agent_id": agent_id,
        "live_server": {
            "username": observation.get("agent", {}).get("username"),
            "quest_count": len(formal_ids),
            "completed_quests": sum(quest.get("status") == "complete" for quest in quests),
            "skills": skills,
        },
        "engine_action_contract": {
            "engine_action_count": len(ENGINE_ACTION_TYPES),
            "engine_actions_missing_from_mcp": engine_actions_missing_from_mcp,
            "mcp_actions_without_engine_primitive": mcp_actions_without_engine_primitive,
            "bindings": MCP_ENGINE_ACTION_TOOLS,
        },
        "validated_batch_actions": sorted(MCP_VALIDATED_ACTIONS),
        "map_cache": {
            "tool": "map_window",
            "max_radius": _MAX_MAP_CACHE_RADIUS,
            "batch_prefetch": "batch_actions(map_radius=N, refresh_map=...)",
            "validation": "map geometry is advisory for planning; submitted actions remain live-engine validated",
        },
        "quest_coverage": {
            "formal_quest_count": len(formal_ids),
            "quest_graph_count": len(graph_ids),
            "mapped_quest_ids": sorted(graph_ids),
            "unmapped_content_quests": unmapped_quests,
        },
        "skill_coverage": skill_coverage,
        "blocking_capabilities": {
            "disabled_skills": disabled_skills,
            "context_skills_without_full_policy": context_skills,
            "quest_graphs_missing": len(formal_ids - graph_ids),
            "max_skill_target": 99,
        },
        "ready_for_full_autonomy": not engine_actions_missing_from_mcp and not disabled_skills and not unmapped_quests and not context_skills,
        "note": "This audit reports capability boundaries from the live engine and never infers completion from queued actions.",
    }


@mcp.tool
async def quest_status(agent_id: str) -> dict[str, Any]:
    """Read quest state exported by the engine's live content registry."""

    result = await api.request("GET", f"/agents/{agent_id}/quests")
    quests = result.get("quests", [])
    counts = {status: sum(1 for quest in quests if quest.get("status") == status) for status in ("complete", "in_progress", "not_started", "unknown")}
    return {"agent_id": agent_id, "username": result.get("username"), "counts": counts, "quests": quests}


@mcp.tool
async def tutorial_status(agent_id: str) -> dict[str, Any]:
    """Read the authoritative tutorial var and the next guide milestone."""

    observation = await _observe(agent_id, radius=1)
    return {
        "agent_id": agent_id,
        "username": observation.get("agent", {}).get("username"),
        "tutorial": _tutorial_step(observation),
        "safety": observation.get("safety"),
        "observation": observation,
    }


@mcp.tool
async def tutorial_step(agent_id: str, radius: int = 16) -> dict[str, Any]:
    """Advance one supported tutorial action from authoritative live state.

    This controller is intentionally bounded. It resumes live dialogue, or
    discovers the next tutorial NPC/scenery option for the currently supported
    early milestones. Unsupported tutorial states are reported explicitly so a
    caller cannot mistake an accepted action for tutorial completion.
    """

    if radius < 1 or radius > 64:
        raise ValueError("radius must be between 1 and 64")
    before_observation = await _observe(agent_id, radius)
    before = _tutorial_step(before_observation)
    if before["state"] == 460 and ((before_observation.get("safety") or {}).get("lowHealth") or (before_observation.get("safety") or {}).get("fleeing")):
        _tutorial_ranged_attacks.discard(agent_id)
    if before["state"] == 650:
        _tutorial_magic_attacks.discard(agent_id)
    if before["complete"]:
        return {
            "status": "complete",
            "validated": True,
            "tutorial": before,
            "observation": before_observation,
        }

    selected = _discover_tutorial_action(
        before_observation,
        ranged_attack_started=agent_id in _tutorial_ranged_attacks,
        magic_attack_started=agent_id in _tutorial_magic_attacks,
    )
    if selected is None:
        return {
            "status": "unsupported_step",
            "validated": False,
            "tutorial": before,
            "supported_states": [0, 1, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 120, 130, 140, 150, 160, 170, 180, 190, 195, 200, 220, 230, 240, 250, 260, 270, 274, 275, 279, 280, 290, 294, 295, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670],
            "reason": "the current tutorial state requires a skill, inventory, interface, or content action not yet modeled by tutorial_step",
            "observation": before_observation,
        }

    if selected["kind"] == "side_tab":
        action = await click_side_tab(agent_id, selected["tab"])
        await asyncio.sleep(0.25)
        after_observation = await _observe(agent_id, radius)
        after_tutorial = _tutorial_step(after_observation)
        progressed = after_tutorial["state"] != before["state"]
        return {
            "status": "action_queued" if progressed else "no_progress",
            "validated": progressed,
            "progressed": progressed,
            "validation": "engine_processed_side_tab" if progressed else "side_tab_without_observable_tutorial_progress",
            "selected": selected,
            "tutorial_before": before,
            "tutorial": after_tutorial,
            "action": action,
            "observation": after_observation,
        }

    if selected["kind"] == "button":
        action = await press_button(agent_id, selected["component"])
        await asyncio.sleep(0.25)
        after_observation = await _observe(agent_id, radius)
        after_tutorial = _tutorial_step(after_observation)
        progressed = after_tutorial["state"] != before["state"]
        return {
            "status": "action_queued" if progressed else "no_progress",
            "validated": progressed,
            "progressed": progressed,
            "validation": "engine_processed_run_controls_button" if progressed else "run_controls_button_without_observable_tutorial_progress",
            "selected": selected,
            "tutorial_before": before,
            "tutorial": after_tutorial,
            "action": action,
            "observation": after_observation,
        }

    if selected["kind"] == "wait_script":
        await asyncio.sleep(0.5)
        after_observation = await _observe(agent_id, radius)
        after_tutorial = _tutorial_step(after_observation)
        active = (after_observation.get("ui") or {}).get("activeScript")
        progressed = after_tutorial["state"] != before["state"] or not active
        return {
            "status": "waiting" if active and not progressed else "action_queued" if progressed else "no_progress",
            "validated": False,
            "progressed": progressed,
            "validation": "engine_script_pending" if active and not progressed else "engine_script_finished_without_tutorial_progress",
            "selected": selected,
            "tutorial_before": before,
            "tutorial": after_tutorial,
            "observation": after_observation,
        }

    if selected["kind"] == "inventory_button":
        action = await inventory_button(
            agent_id,
            selected["component"],
            selected["inventory"],
            selected["slot"],
            selected["option"],
        )
        await asyncio.sleep(0.5)
        after_observation = await _observe(agent_id, radius)
        after_tutorial = _tutorial_step(after_observation)
        inventory_changed = after_observation.get("agent", {}).get("inventories") != before_observation.get("agent", {}).get("inventories")
        progressed = (
            after_tutorial["state"] != before["state"]
            or inventory_changed
            or (after_observation.get("ui") or {}).get("activeScript") is True
        )
        return {
            "status": "action_queued" if progressed else "no_progress",
            "validated": progressed,
            "progressed": progressed,
            "validation": "engine_processed_smithing_button" if progressed else "smithing_button_without_observable_progress",
            "selected": selected,
            "tutorial_before": before,
            "tutorial": after_tutorial,
            "action": action,
            "observation": after_observation,
        }

    if selected["kind"] == "item_op":
        if before["state"] == 460:
            _tutorial_ranged_attacks.discard(agent_id)
        action = await item_op_agent(
            agent_id,
            selected["inventory"],
            selected["slot"],
            selected["option"],
        )
        await asyncio.sleep(0.5)
        after_observation = await _observe(agent_id, radius)
        after_tutorial = _tutorial_step(after_observation)
        inventory_changed = after_observation.get("agent", {}).get("inventories") != before_observation.get("agent", {}).get("inventories")
        progressed = (
            after_tutorial["state"] != before["state"]
            or inventory_changed
            or (after_observation.get("ui") or {}).get("activeScript") is True
        )
        return {
            "status": "action_queued" if progressed else "no_progress",
            "validated": progressed,
            "progressed": progressed,
            "validation": "engine_processed_item_option" if progressed else "item_option_without_observable_progress",
            "selected": selected,
            "tutorial_before": before,
            "tutorial": after_tutorial,
            "action": action,
            "observation": after_observation,
        }

    if selected["kind"] == "cast_spell":
        target = selected["target"]
        action = await cast_spell(
            agent_id,
            component_name=selected["component_name"],
            target_kind=target["kind"],
            target_id=target["id"],
        )
        _tutorial_magic_attacks.add(agent_id)
        await asyncio.sleep(0.5)
        after_observation = await _observe(agent_id, radius)
        after_tutorial = _tutorial_step(after_observation)
        agent = after_observation.get("agent") or {}
        progressed = (
            after_tutorial["state"] != before["state"]
            or (after_observation.get("ui") or {}).get("activeScript") is True
            or agent.get("targetOperation") is not None
            or agent.get("moving") is True
        )
        return {
            "status": "action_queued" if progressed else "no_progress",
            "validated": progressed,
            "progressed": progressed,
            "validation": "engine_processed_named_spell" if progressed else "spell_action_without_observable_progress",
            "selected": selected,
            "tutorial_before": before,
            "tutorial": after_tutorial,
            "action": action,
            "observation": after_observation,
        }

    if selected["kind"] == "wait_combat":
        after_observation = before_observation
        after_tutorial = before
        progressed = False
        for _ in range(24):
            await asyncio.sleep(1.0)
            after_observation = await _observe(agent_id, radius)
            after_tutorial = _tutorial_step(after_observation)
            if after_tutorial["state"] != before["state"]:
                progressed = True
                break
        return {
            "status": "action_queued" if progressed else "waiting",
            "validated": progressed,
            "progressed": progressed,
            "validation": "engine_observed_combat_completion" if progressed else "combat_still_in_progress",
            "selected": selected,
            "tutorial_before": before,
            "tutorial": after_tutorial,
            "observation": after_observation,
        }

    if selected["kind"] == "skill":
        skill_kwargs: dict[str, Any] = {}
        discovered = selected.get("selected")
        if isinstance(discovered, dict) and isinstance(discovered.get("target"), dict):
            target = discovered["target"]
            skill_kwargs.update(
                {
                    "target_kind": target.get("kind"),
                    "target_id": target.get("id"),
                    "x": target.get("x"),
                    "z": target.get("z"),
                    "level": target.get("level"),
                    "option": discovered.get("option"),
                }
            )
        skill_result = await skill_step(agent_id, selected["skill"], radius=radius, **skill_kwargs)
        after_observation = skill_result.get("observation") or await _observe(agent_id, radius)
        after_tutorial = _tutorial_step(after_observation)
        progressed = (
            after_tutorial["state"] != before["state"]
            or (after_observation.get("ui") or {}).get("activeScript") is True
        )
        status = skill_result.get("status")
        if status == "action_queued" and not progressed:
            status = "no_progress"
        return {
            "status": status,
            "validated": bool(skill_result.get("validated", False)) and progressed,
            "progressed": progressed,
            "validation": skill_result.get("validation") if progressed else "skill_action_without_observable_tutorial_progress",
            "selected": selected,
            "tutorial_before": before,
            "tutorial": after_tutorial,
            "result": skill_result,
            "observation": after_observation,
        }

    if selected["kind"] == "item_pair":
        discovered = selected["selected"]
        action = await use_item(
            agent_id,
            discovered["inventory"],
            discovered["slot"],
            discovered["use_inventory"],
            discovered["use_slot"],
        )
        await asyncio.sleep(0.5)
        after_observation = await _observe(agent_id, radius)
        after_tutorial = _tutorial_step(after_observation)
        inventory_changed = after_observation.get("agent", {}).get("inventories") != before_observation.get("agent", {}).get("inventories")
        progressed = (
            after_tutorial["state"] != before["state"]
            or inventory_changed
            or (after_observation.get("ui") or {}).get("activeScript") is True
        )
        return {
            "status": "action_queued" if progressed else "no_progress",
            "validated": progressed,
            "progressed": progressed,
            "validation": "engine_processed_item_pair" if progressed else "item_pair_without_observable_tutorial_progress",
            "selected": selected,
            "tutorial_before": before,
            "tutorial": after_tutorial,
            "action": action,
            "observation": after_observation,
        }

    if selected["kind"] == "item_on":
        discovered = selected["selected"]
        target = discovered["target"]
        action = await use_item_on(
            agent_id,
            discovered["inventory"],
            discovered["slot"],
            target["kind"],
            target_id=target["id"],
            x=target["x"],
            z=target["z"],
            level=target.get("level"),
        )
        resolved = await _resolve_interaction_action(
            agent_id,
            action,
            radius,
            before_observation=before_observation,
        )
        after_observation = resolved.get("observation") or await _observe(agent_id, radius)
        after_tutorial = _tutorial_step(after_observation)
        inventory_changed = after_observation.get("agent", {}).get("inventories") != before_observation.get("agent", {}).get("inventories")
        progressed = (
            after_tutorial["state"] != before["state"]
            or inventory_changed
            or (after_observation.get("ui") or {}).get("activeScript") is True
        )
        return {
            "status": "action_queued" if resolved["status"] == "complete" and progressed else "no_progress" if resolved["status"] == "complete" else resolved["status"],
            "validated": resolved.get("validated", False) and progressed,
            "progressed": progressed,
            "validation": resolved.get("validation") if progressed else "item_action_without_observable_tutorial_progress",
            "selected": selected,
            "tutorial_before": before,
            "tutorial": after_tutorial,
            "action": action,
            "result": resolved,
            "observation": after_observation,
        }

    if selected["kind"] == "resume_dialogue":
        action = await resume_dialogue(agent_id)
        await asyncio.sleep(0.25)
        after_observation = await _observe(agent_id, radius)
        after_tutorial = _tutorial_step(after_observation)
        progressed = (
            after_tutorial["state"] != before["state"]
            or (after_observation.get("ui") or {}).get("modalChat") != (before_observation.get("ui") or {}).get("modalChat")
            or (after_observation.get("ui") or {}).get("activeScript") is False
        )
        return {
            "status": "action_queued" if progressed else "no_progress",
            "validated": progressed,
            "progressed": progressed,
            "validation": "engine_resumed_paused_dialogue" if progressed else "dialogue_resumed_without_observable_progress",
            "selected": selected,
            "tutorial_before": before,
            "tutorial": after_tutorial,
            "action": action,
            "observation": after_observation,
        }

    if before["state"] == 450 and selected.get("target", {}).get("kind") == "npc":
        gate = _discover_tutorial_rat_gate_action(before_observation)
        if gate is not None:
            await stop_agent(agent_id)
            gate_action = await api.request(
                "POST",
                f"/agents/{agent_id}/actions",
                json={"type": "interact", "target": gate["target"], "option": gate["option"]},
            )
            gate_result = await _resolve_interaction_action(
                agent_id,
                gate_action,
                radius,
                respect_safety=False,
                before_observation=before_observation,
            )
            if gate_result["status"] != "complete":
                return {
                    "status": gate_result["status"],
                    "validated": False,
                    "progressed": False,
                    "validation": gate_result.get("validation"),
                    "selected": {"gate": gate},
                    "tutorial_before": before,
                    "tutorial": before,
                    "actions": [gate_action],
                    "results": [gate_result],
                    "observation": gate_result.get("observation") or before_observation,
                }
            gate_observation = gate_result.get("observation") or before_observation
            for _ in range(8):
                if not (gate_observation.get("ui") or {}).get("activeScript"):
                    break
                await asyncio.sleep(0.5)
                gate_observation = await _observe(agent_id, radius)
            await stop_agent(agent_id)
            await asyncio.sleep(0.25)
            talk = _discover_tutorial_action(gate_observation)
            if talk is None or talk.get("target", {}).get("kind") != "npc":
                return {
                    "status": "blocked",
                    "validated": False,
                    "progressed": False,
                    "validation": "rat_gate_opened_without_visible_combat_instructor",
                    "selected": {"gate": gate, "talk": talk},
                    "tutorial_before": before,
                    "tutorial": _tutorial_step(gate_observation),
                    "actions": [gate_action],
                    "results": [gate_result],
                    "observation": gate_observation,
                }
            talk_action = await api.request(
                "POST",
                f"/agents/{agent_id}/actions",
                json={"type": "interact", "target": talk["target"], "option": talk["option"]},
            )
            talk_result = await _resolve_interaction_action(
                agent_id,
                talk_action,
                radius,
                respect_safety=False,
                before_observation=gate_observation,
            )
            after_observation = talk_result.get("observation") or await _observe(agent_id, radius)
            after_tutorial = _tutorial_step(after_observation)
            progressed = (
                after_tutorial["state"] != before["state"]
                or (after_observation.get("ui") or {}).get("activeScript") is True
                or (after_observation.get("agent") or {}).get("moving") is True
            )
            return {
                "status": "action_queued" if talk_result["status"] == "complete" and progressed else talk_result["status"],
                "validated": talk_result.get("validated", False) and progressed,
                "progressed": progressed,
                "validation": talk_result.get("validation") if progressed else "combat_instructor_talk_without_observable_progress",
                "selected": {"gate": gate, "talk": talk},
                "tutorial_before": before,
                "tutorial": after_tutorial,
                "actions": [gate_action, talk_action],
                "results": [gate_result, talk_result],
                "observation": after_observation,
            }

    action = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "interact", "target": selected["target"], "option": selected["option"]},
    )
    # From the banking lesson onward, tutorial interactions are dialogue,
    # interface, or door progression—not combat.  Do not let the combat
    # safety watchdog strand a wounded tutorial player while resolving one of
    # these harmless actions; combat branches remain explicitly guarded above.
    resolved = await _resolve_interaction_action(
        agent_id,
        action,
        radius,
        respect_safety=before["state"] < 500,
        before_observation=before_observation,
    )
    after_observation = resolved.get("observation") or await _observe(agent_id, radius)
    after_tutorial = _tutorial_step(after_observation)
    if (
        before["state"] == 460
        and selected.get("target", {}).get("kind") == "npc"
        and "giant rat" in (selected.get("entity") or {}).get("name", "").casefold()
        and resolved.get("status") == "complete"
    ):
        _tutorial_ranged_attacks.add(agent_id)
    progressed = (
        after_tutorial["state"] != before["state"]
        or (after_observation.get("ui") or {}).get("activeScript") is True
        or (after_observation.get("agent") or {}).get("moving") is True
    )
    resolved_status = resolved["status"]
    state_progress_overrides_pending = resolved_status == "blocked" and after_tutorial["state"] != before["state"]
    return {
        "status": (
            "action_queued"
            if (resolved_status == "complete" or state_progress_overrides_pending) and progressed
            else "no_progress"
            if resolved_status == "complete"
            else resolved_status
        ),
        "validated": (resolved.get("validated", False) or state_progress_overrides_pending) and progressed,
        "progressed": progressed,
        "validation": (
            "tutorial_state_progressed_with_pending_operation" if state_progress_overrides_pending
            else resolved.get("validation")
            if progressed
            else "interaction_resolved_without_observable_tutorial_progress"
        ),
        "selected": selected,
        "tutorial_before": before,
        "tutorial": after_tutorial,
        "action": action,
        "observation": after_observation,
    }


@mcp.tool
async def progression_plan(agent_id: str, phase: str = "all") -> dict[str, Any]:
    """Combine the 2004Scape guide route with current live skills and quests.

    This is a planner and audit tool. It never sets quest vars, grants XP, or
    treats a queued interaction as completion.
    """

    if phase not in {"all", "early", "mid", "endgame"}:
        raise ValueError("phase must be one of: all, early, mid, endgame")
    observation = await _observe(agent_id, radius=1)
    quest_result = await api.request("GET", f"/agents/{agent_id}/quests")
    quests = {quest.get("id"): quest for quest in quest_result.get("quests", [])}
    skills = observation.get("agent", {}).get("skills", {})
    catalog = {skill: spec["mode"] for skill, spec in SKILL_PLAYBOOK.items()}
    max_skill_unmet = {
        skill: {
            "current": skills.get(skill, {}).get("baseLevel", 0),
            "target": 99,
            "remaining": max(0, 99 - skills.get(skill, {}).get("baseLevel", 0)),
        }
        for skill in SKILL_PLAYBOOK
        if skills.get(skill, {}).get("baseLevel", 0) < 99
    }

    phases = {"early": GUIDE_ORDER[:12], "mid": GUIDE_ORDER[12:36], "endgame": GUIDE_ORDER[36:]}
    ids = GUIDE_ORDER if phase == "all" else phases[phase]
    route: list[dict[str, Any]] = []
    for index, quest_id in enumerate(ids):
        quest = quests.get(quest_id)
        requirements = GUIDE_MILESTONES.get(quest_id, {}).get("skills", {})
        unmet = {
            skill: {"current": skills.get(skill, {}).get("baseLevel", 0), "required": level}
            for skill, level in requirements.items()
            if skills.get(skill, {}).get("baseLevel", 0) < level
        }
        route.append(
            {
                "order": index + 1,
                "quest_id": quest_id,
                "status": quest.get("status", "unregistered") if quest else "unregistered",
                "engine_quest": quest,
                "skill_requirements": requirements,
                "unmet_requirements": unmet,
            }
        )

    registry_ids = set(quests)
    remainder = [quest for quest in quest_result.get("quests", []) if quest.get("id") not in set(GUIDE_ORDER)]
    tutorial = _tutorial_step(observation)
    next_quest = next((item for item in route if item["status"] != "complete"), None)
    next_step = (
        {"kind": "tutorial", **tutorial["current"], "tutorial_state": tutorial["state"]}
        if not tutorial["complete"]
        else next_quest
    )
    return {
        "agent_id": agent_id,
        "phase": phase,
        "guide_source": "https://lostcity.rs/t/redkiwis-somewhat-optimal-quest-skilling-guide/13265",
        "quest_registry": {"count": len(registry_ids), "unmapped_content_quests": remainder},
        "skills": skills,
        "max_skill_target": 99,
        "max_skill_unmet": max_skill_unmet,
        "skill_automation_modes": catalog,
        "tutorial": tutorial,
        "next_step": next_step,
        "next_quest": next_quest,
        "route": route,
        "safety": observation.get("safety"),
        "note": "Quest state is read from the running engine; unknown completion markers remain explicit and require content-specific verification.",
    }


async def _progression_step_once(agent_id: str, radius: int = 32) -> dict[str, Any]:
    """Take one resumable progression step from authoritative live state."""

    observation = await _observe(agent_id, radius)
    safety = observation.get("safety") or {}
    if safety.get("lowHealth") or safety.get("fleeing"):
        return {
            "status": "paused_low_health",
            "validated": False,
            "reason": "the engine safety controller is handling a health escape",
            "safety": safety,
            "observation": observation,
        }

    tutorial = _tutorial_step(observation)
    if not tutorial["complete"]:
        result = await tutorial_step(agent_id, radius=radius)
        if result.get("validated"):
            return {
                "status": "progressed",
                "phase": "tutorial",
                "validated": True,
                "validation": result.get("validation"),
                "before": tutorial,
                "result": result,
                "observation": result.get("observation") or observation,
            }
        if result.get("status") in {"unsupported", "unsupported_step", "blocked", "no_progress"}:
            return {
                "status": result.get("status"),
                "phase": "tutorial",
                "validated": False,
                "reason": result.get("reason") or "tutorial action did not produce a validated state transition",
                "result": result,
                "observation": result.get("observation") or observation,
            }
        return {
            "status": "waiting",
            "phase": "tutorial",
            "validated": False,
            "reason": "tutorial content is still resolving on the live server",
            "result": result,
            "observation": result.get("observation") or observation,
        }

    plan = await progression_plan(agent_id, phase="all")
    next_quest = plan.get("next_quest")
    if next_quest is None:
        if plan.get("max_skill_unmet"):
            return {
                "status": "blocked",
                "phase": "skills",
                "validated": False,
                "reason": "all mapped guide quests are complete, but max-skill targets remain",
                "missing": plan["max_skill_unmet"],
                "plan": plan,
                "observation": observation,
            }
        return {
            "status": "complete",
            "phase": "complete",
            "validated": True,
            "validation": "guide_route_and_skill_targets_confirmed_by_live_state",
            "plan": plan,
            "observation": observation,
        }

    unmet = next_quest.get("unmet_requirements") or {}
    if unmet:
        skill, requirement = next(iter(unmet.items()))
        spec = SKILL_PLAYBOOK[skill]
        if spec["mode"] == "disabled":
            return {
                "status": "blocked",
                "phase": "skills",
                "validated": False,
                "reason": "the next guide quest requires a disabled skill",
                "quest_id": next_quest.get("quest_id"),
                "skill": skill,
                "required_level": requirement.get("required"),
                "current_level": requirement.get("current"),
                "missing": spec.get("missing", []),
                "plan": plan,
                "observation": observation,
            }
        skill_plan_result = await skill_plan(agent_id, skill, target_level=requirement.get("required"), radius=radius)
        if skill_plan_result.get("next_action") is None:
            return {
                "status": "blocked",
                "phase": "skills",
                "validated": False,
                "reason": "no live training action is available for the next guide skill requirement",
                "quest_id": next_quest.get("quest_id"),
                "skill": skill,
                "skill_plan": skill_plan_result,
                "plan": plan,
                "observation": observation,
            }
        result = await skill_step(agent_id, skill, radius=radius)
        if result.get("validated"):
            return {
                "status": "progressed",
                "phase": "skills",
                "validated": True,
                "validation": result.get("validation"),
                "quest_id": next_quest.get("quest_id"),
                "skill": skill,
                "result": result,
                "observation": result.get("observation") or observation,
            }
        return {
            "status": "blocked",
            "phase": "skills",
            "validated": False,
            "reason": "the available skill action did not return a validated live postcondition",
            "quest_id": next_quest.get("quest_id"),
            "skill": skill,
            "result": result,
            "observation": result.get("observation") or observation,
        }

    return {
        "status": "blocked",
        "phase": "quests",
        "validated": False,
        "reason": "no quest graph is registered for the next guide quest",
        "capability": "quest_graph",
        "quest_id": next_quest.get("quest_id"),
        "quest": next_quest,
        "plan": plan,
        "observation": observation,
    }


@mcp.tool
async def progression_step(agent_id: str, radius: int = 32) -> dict[str, Any]:
    """Execute one validated, resumable step of the guide progression."""

    if radius < 1 or radius > 64:
        raise ValueError("radius must be between 1 and 64")
    return await _progression_step_once(agent_id, radius)


_BATCH_ACTION_TYPES = {
    "observe",
    "map_window",
    "travel",
    "move",
    "drop_item",
    "equip_item",
    "combat_loop",
    "recover_health",
    "crafting_loop",
    "skill_step",
    "train_skill",
    "progression_step",
    "quest_plan",
    "run_quest",
    "interact",
    "pickup_object",
    "open_bank",
    "open_shop",
    "buy_item",
    "sell_item",
    "close_shop",
    "deposit_item",
    "withdraw_item",
    "close_bank",
    "use_item",
    "combine",
    "use_item_on_validated",
    "press_button",
    "choose_dialogue_option",
    "resume_dialogue",
    "resume_count_dialog",
    "close_interface",
    "set_run",
    "stop",
}


async def _wait_for_batch_move(
    agent_id: str,
    x: int,
    z: int,
    *,
    timeout_seconds: float = 20.0,
    radius: int = 16,
) -> dict[str, Any]:
    """Turn an engine move request into a validated batch postcondition."""

    latest = await _observe(agent_id, radius)
    attempts = max(1, int(timeout_seconds / 0.5))
    stationary_ticks = 0
    for _ in range(attempts):
        position = (latest.get("agent") or {}).get("position") or {}
        if position.get("x") == x and position.get("z") == z:
            return {
                "status": "complete",
                "validated": True,
                "validation": "move_destination_reached_on_live_server",
                "observation": latest,
            }
        if (latest.get("agent") or {}).get("moving") is False and (latest.get("agent") or {}).get("targetOperation") is None:
            stationary_ticks += 1
            if stationary_ticks >= 6:
                return {
                    "status": "blocked",
                    "validated": False,
                    "validation": "move_stopped_before_destination",
                    "reason": "the engine stopped at a partial position before reaching the requested destination",
                    "observation": latest,
                }
        else:
            stationary_ticks = 0
        await asyncio.sleep(0.5)
        latest = await _observe(agent_id, radius)
    return {
        "status": "blocked",
        "validated": False,
        "validation": "move_destination_timeout",
        "reason": "the engine did not reach the requested destination before the batch validation timeout",
        "observation": latest,
    }


async def _execute_batch_action(agent_id: str, action: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one allowlisted declarative action to an existing MCP primitive."""

    kind = action.get("type")
    if kind not in _BATCH_ACTION_TYPES:
        raise ValueError(f"unsupported batch action '{kind}'; choose from {', '.join(sorted(_BATCH_ACTION_TYPES))}")
    payload = {key: value for key, value in action.items() if key not in {"type", "repeat"}}
    batch_map_radius = payload.pop("_batch_map_radius", None)

    if kind == "observe":
        return {
            "status": "complete",
            "validated": True,
            "validation": "authoritative_live_observation",
            "observation": await _observe(agent_id, int(payload.get("radius", 16))),
        }
    if kind == "map_window":
        window = await map_window(
            agent_id,
            radius=int(payload.get("radius", 32)),
            refresh=bool(payload.get("refresh", False)),
            target_x=payload.get("target_x"),
            target_z=payload.get("target_z"),
            target_level=payload.get("target_level"),
        )
        return {
            **window,
            "status": "complete",
            "validated": True,
            "validation": "authoritative_live_or_cached_map_window",
        }
    if kind == "travel":
        step = {"kind": "travel", **payload}
        return await _travel_quest_step(
            agent_id,
            step,
            int(payload.get("radius", 32)),
        )
    if kind == "move":
        x = int(payload["x"])
        z = int(payload["z"])
        cached_route = None
        if batch_map_radius is not None:
            cached_window = await map_window(
                agent_id,
                radius=int(batch_map_radius),
                target_x=x,
                target_z=z,
                target_level=payload.get("level"),
            )
            cached_route = cached_window.get("cached_route")
        request = await move_agent(
            agent_id,
            x=x,
            z=z,
            run=bool(payload.get("run", False)),
            max_legs=int(payload.get("max_legs", 32)),
        )
        result = await _wait_for_batch_move(
            agent_id,
            x,
            z,
            timeout_seconds=float(payload.get("timeout_seconds", 20.0)),
            radius=int(payload.get("radius", 16)),
        )
        return {**result, "request": request, "cached_route": cached_route}
    if kind == "drop_item":
        return await drop_item(agent_id, **payload)
    if kind == "equip_item":
        return await equip_item(agent_id, **payload)
    if kind == "combat_loop":
        radius = int(payload.pop("radius", 32))
        return await _combat_loop_step(agent_id, payload, radius)
    if kind == "recover_health":
        radius = int(payload.pop("radius", 1))
        return await _recover_health_step(agent_id, payload, radius)
    if kind == "crafting_loop":
        radius = int(payload.pop("radius", 32))
        return await _crafting_loop_step(agent_id, payload, radius)
    if kind == "skill_step":
        return await skill_step(agent_id, **payload)
    if kind == "train_skill":
        result = await train_skill(agent_id, **payload)
        # A bounded trainer can legitimately stop at its step limit after
        # several validated live actions. Preserve that progress as a valid
        # batch result so the caller can chain another bounded batch without
        # making the model inspect every individual action.
        initial = result.get("initial") or {}
        final = result.get("final") or {}
        if not result.get("validated") and final.get("experience", 0) > initial.get("experience", 0):
            result = {
                **result,
                "validated": True,
                "validation": "bounded_training_progress_observed_on_live_server",
            }
        return result
    if kind == "progression_step":
        return await _progression_step_once(agent_id, int(payload.get("radius", 32)))
    if kind == "quest_plan":
        result = await quest_plan(agent_id, **payload)
        return {
            **result,
            "status": "complete",
            "validated": True,
            "validation": "quest_plan_derived_from_live_state",
        }
    if kind == "run_quest":
        result = await run_quest(agent_id, **payload)
        return {
            **result,
            "validated": result.get("status") == "complete",
            "validation": (
                "quest_completion_confirmed_by_live_state"
                if result.get("status") == "complete"
                else "quest_run_did_not_confirm_completion"
            ),
        }
    if kind == "interact":
        return await interact_agent(agent_id, **payload)
    if kind == "pickup_object":
        return await pickup_object(agent_id, **payload)
    if kind == "open_bank":
        return await open_bank(agent_id, **payload)
    if kind == "open_shop":
        return await open_shop(agent_id, **payload)
    if kind == "buy_item":
        return await buy_item(agent_id, **payload)
    if kind == "sell_item":
        return await sell_item(agent_id, **payload)
    if kind == "close_shop":
        return await close_shop(agent_id)
    if kind == "deposit_item":
        return await deposit_item(agent_id, **payload)
    if kind == "withdraw_item":
        return await withdraw_item(agent_id, **payload)
    if kind == "close_bank":
        return await close_bank(agent_id)
    if kind == "use_item":
        return await _use_item_validated(agent_id, **payload)
    if kind == "combine":
        radius = int(payload.pop("radius", 8))
        step = {"kind": "combine", **payload}
        return await _combine_quest_items_step(agent_id, step, radius)
    if kind == "use_item_on_validated":
        return await use_item_on_validated(agent_id, **payload)
    if kind == "press_button":
        return await _press_button_validated(agent_id, **payload)
    if kind == "choose_dialogue_option":
        return await choose_dialogue_option(agent_id, **payload)
    if kind == "resume_dialogue":
        return await _resume_dialogue_quest_step(agent_id, int(payload.get("radius", 8)))
    if kind == "resume_count_dialog":
        return await resume_count_dialog(agent_id, **payload)
    if kind == "close_interface":
        return await close_interface(agent_id, **payload)
    if kind == "set_run":
        return await set_run(agent_id, **payload)
    if kind == "stop":
        result = await stop_agent(agent_id)
        return {
            **result,
            "status": "complete",
            "validated": True,
            "validation": "stop_accepted_by_live_server",
        }
    raise AssertionError(f"unhandled batch action '{kind}'")


@mcp.tool
async def batch_actions(
    agent_id: str,
    actions: list[dict[str, Any]],
    stop_on_failure: bool = True,
    max_actions: int = 64,
    map_radius: int | None = 32,
    refresh_map: bool = False,
) -> dict[str, Any]:
    """Execute a bounded sequence of validated actions in one MCP call.

    Each action is a declarative object with a ``type`` from the documented
    allowlist and the arguments accepted by that existing MCP primitive. An
    optional ``repeat`` runs the same action repeatedly, observing and
    validating every iteration. By default one radius-32 map window is
    prefetched/reused for the batch; pass ``map_radius=None`` to omit it or
    ``refresh_map=True`` to force a live geometry refresh. The batch stops on
    safety, an unvalidated mutation, or the caller's limit, and returns the
    exact index to resume.
    """

    if not isinstance(actions, list) or not actions:
        raise ValueError("actions must be a non-empty list")
    if max_actions < 1 or max_actions > 256:
        raise ValueError("max_actions must be between 1 and 256")
    if map_radius is not None and (not isinstance(map_radius, int) or map_radius < 1 or map_radius > _MAX_MAP_CACHE_RADIUS):
        raise ValueError(f"map_radius must be between 1 and {_MAX_MAP_CACHE_RADIUS}")

    prefetched_map = None
    if map_radius is not None:
        prefetched_map = await map_window(agent_id, radius=map_radius, refresh=refresh_map)

    map_geometry_dirty = False
    latest_live_observation: dict[str, Any] | None = None

    async def with_map_cache(result: dict[str, Any]) -> dict[str, Any]:
        nonlocal prefetched_map, map_geometry_dirty, latest_live_observation
        if prefetched_map is None:
            return result
        refreshed = False
        if map_geometry_dirty:
            prefetched_map = await map_window(agent_id, radius=map_radius, refresh=True)
            map_geometry_dirty = False
            refreshed = True
        observation = result.get("observation")
        if not isinstance(observation, dict) and isinstance(result.get("results"), list) and result["results"]:
            last_result = result["results"][-1].get("result")
            if isinstance(last_result, dict):
                observation = last_result.get("observation")
        if isinstance(observation, dict):
            latest_live_observation = observation
        if latest_live_observation is not None and not refreshed:
            # Preserve the cached static rows while carrying forward the most
            # recent live observation from the validated action. This avoids
            # another full map read for every skill step in a batch, while
            # still preventing the attached observation from going stale.
            prefetched_map = {
                **prefetched_map,
                "observation": latest_live_observation,
                "tick": latest_live_observation.get("tick", prefetched_map.get("tick")),
                "cache_hit": True,
                "validation": "cached_static_geometry_with_latest_live_observation",
            }
        return {**result, "map_cache": prefetched_map}

    results: list[dict[str, Any]] = []
    executed = 0
    for action_index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError(f"actions[{action_index}] must be an object")
        repeat = action.get("repeat", 1)
        if not isinstance(repeat, int) or repeat < 1 or repeat > 64:
            raise ValueError(f"actions[{action_index}].repeat must be between 1 and 64")
        for repetition in range(repeat):
            if executed >= max_actions:
                return await with_map_cache({
                    "status": "step_limit",
                    "validated": all(item["validated"] for item in results),
                    "validation": "batch_action_limit_reached",
                    "next_action": action_index,
                    "executed": executed,
                    "results": results,
                })
            try:
                dispatch = dict(action)
                if map_radius is not None:
                    dispatch["_batch_map_radius"] = map_radius
                result = await _execute_batch_action(agent_id, dispatch)
            except (LostCityApiError, TypeError, ValueError) as error:
                result = {
                    "status": "error",
                    "validated": False,
                    "reason": str(error),
                }
            record = {
                "index": action_index,
                "repetition": repetition + 1,
                "type": action.get("type"),
                "status": result.get("status"),
                "validated": bool(result.get("validated", False)),
                "result": result,
            }
            results.append(record)
            executed += 1
            if isinstance(result.get("observation"), dict):
                latest_live_observation = result["observation"]
            if action.get("type") in {"interact", "travel", "crafting_loop", "run_quest", "progression_step"}:
                _mark_map_cache_dirty(agent_id)
                map_geometry_dirty = prefetched_map is not None
            if not record["validated"] and stop_on_failure:
                return await with_map_cache({
                    "status": "blocked",
                    "validated": False,
                    "validation": "batch_stopped_on_unvalidated_action",
                    "next_action": action_index,
                    "executed": executed,
                    "results": results,
                })

    all_validated = all(item["validated"] for item in results)
    return await with_map_cache({
        "status": "complete" if all_validated else "partial",
        "validated": all_validated,
        "validation": (
            "every_batched_action_validated_on_live_state"
            if all_validated
            else "batch_completed_with_unvalidated_actions"
        ),
        "next_action": len(actions),
        "executed": executed,
        "results": results,
    })


async def _progression_worker_loop(agent_id: str, username: str, interval_seconds: float, radius: int) -> None:
    state = _progression_state.setdefault(
        agent_id,
        {"agent_id": agent_id, "username": username, "status": "starting", "steps": 0, "validated_steps": 0, "history": []},
    )
    stalled = 0
    try:
        while True:
            try:
                result = await _progression_step_once(agent_id, radius)
            except (LostCityApiError, ValueError) as error:
                result = {"status": "blocked", "validated": False, "reason": str(error)}
            state["steps"] += 1
            if result.get("validated"):
                state["validated_steps"] += 1
                stalled = 0
            elif result.get("status") == "waiting":
                stalled += 1
            else:
                stalled = 0
            state["status"] = result.get("status", "waiting")
            state["phase"] = result.get("phase")
            state["last_result"] = result
            state["history"] = (state.get("history") or [])[-19:] + [{
                "step": state["steps"],
                "status": result.get("status"),
                "phase": result.get("phase"),
                "validated": result.get("validated", False),
                "reason": result.get("reason"),
            }]
            if result.get("status") in {"blocked", "unsupported", "unsupported_step", "paused_low_health", "complete"}:
                break
            if stalled >= 8:
                state["status"] = "blocked"
                state["last_result"] = {
                    "status": "blocked",
                    "validated": False,
                    "reason": "progression worker observed eight consecutive waiting steps",
                }
                break
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        state["status"] = "stopped"
        raise
    except Exception as error:  # keep the worker's failure observable to the caller
        state["status"] = "error"
        state["last_result"] = {"status": "error", "validated": False, "reason": str(error)}
    finally:
        _progression_tasks.pop(agent_id, None)


@mcp.tool
async def start_progression_worker(
    agent_id: str,
    interval_seconds: float = 1.0,
    radius: int = 32,
) -> dict[str, Any]:
    """Start the resumable guide worker for an authorized live agent."""

    if interval_seconds < 0.25 or interval_seconds > 60:
        raise ValueError("interval_seconds must be between 0.25 and 60")
    if radius < 1 or radius > 64:
        raise ValueError("radius must be between 1 and 64")
    session = await api.request("GET", f"/agents/{agent_id}")
    existing = _progression_tasks.get(agent_id)
    if existing and not existing.done():
        return {"status": "already_running", "worker": _progression_state.get(agent_id)}
    username = session["username"]
    _ensure_keepalive(agent_id, username)
    _progression_state[agent_id] = {
        "agent_id": agent_id,
        "username": username,
        "status": "starting",
        "interval_seconds": interval_seconds,
        "radius": radius,
        "steps": 0,
        "validated_steps": 0,
        "history": [],
        "durability": "engine/player state is the resume checkpoint; call start again after an MCP process restart",
    }
    _progression_tasks[agent_id] = asyncio.create_task(
        _progression_worker_loop(agent_id, username, interval_seconds, radius)
    )
    return {"status": "started", "worker": _progression_state[agent_id]}


@mcp.tool
async def progression_worker_status(agent_id: str) -> dict[str, Any]:
    """Read the last live result and checkpoint of a progression worker."""

    return _progression_state.get(
        agent_id,
        {"agent_id": agent_id, "status": "not_started", "steps": 0, "validated_steps": 0, "history": []},
    )


@mcp.tool
async def stop_progression_worker(agent_id: str) -> dict[str, Any]:
    """Stop a progression worker without changing the player's game state."""

    task = _progression_tasks.pop(agent_id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    state = _progression_state.setdefault(agent_id, {"agent_id": agent_id})
    state["status"] = "stopped"
    return {"status": "stopped", "worker": state}


@mcp.tool
async def quest_plan(
    agent_id: str,
    quest_id: str = "combat_bounty",
    steps: list[dict[str, Any]] | None = None,
    radius: int = 32,
) -> dict[str, Any]:
    """Preflight a built-in or custom quest against current live game state."""

    custom_steps = steps is not None
    if steps is None:
        template = QUEST_TEMPLATES.get(quest_id)
        if template is None:
            raise ValueError(f"unknown quest '{quest_id}'; provide steps for a custom quest")
        steps = template["steps"]
    normalized = _quest_steps(steps)
    observation = await _observe(agent_id, radius)
    previews: list[dict[str, Any]] = []
    for step in normalized:
        preview: dict[str, Any] = {"kind": step["kind"], "ready": True}
        if step["kind"] == "travel":
            preview["route"] = await api.request(
                "POST",
                f"/agents/{agent_id}/route",
                json={"x": step["x"], "z": step["z"], "level": step.get("level", 0)},
            )
            preview["ready"] = preview["route"].get("reached", False)
        elif step["kind"] == "discover_interact":
            targets = [step["target_kind"]] if step.get("target_kind") else ["npc", "loc", "obj"]
            preview["next_action"] = _discover_skill_action(
                observation,
                {
                    "targets": targets,
                    "target_name": step.get("target_name"),
                    "option_tokens": [token.casefold() for token in step["option_tokens"]],
                },
            )
            preview["ready"] = preview["next_action"] is not None
        elif step["kind"] == "train":
            skill = _normalized_skill(step["skill"])
            preview["skill"] = skill
            preview["ready"] = SKILL_PLAYBOOK[skill]["mode"] not in {"context", "disabled"}
            preview["missing"] = SKILL_PLAYBOOK[skill].get("missing", [])
        previews.append(preview)
    return {
        "quest_id": quest_id,
        "custom": custom_steps,
        "steps": normalized,
        "previews": previews,
        "safety": observation.get("safety"),
        "observation": observation,
    }


@mcp.tool
async def run_quest(
    agent_id: str,
    quest_id: str = "combat_bounty",
    steps: list[dict[str, Any]] | None = None,
    radius: int = 32,
    start_step: int = 0,
) -> dict[str, Any]:
    """Execute a bounded quest graph with routing, combat, and safety guards.

    Custom steps let an agent encode the actual quest script discovered from
    the game. The runner is resumable at the step boundary: if a route is
    blocked, health is low, or a missing client primitive is required, the
    response includes the completed step count and the exact blocker.
    """

    if start_step < 0:
        raise ValueError("start_step must be zero or greater")
    template_steps = steps is None
    if steps is None:
        template = QUEST_TEMPLATES.get(quest_id)
        if template is None:
            raise ValueError(f"unknown quest '{quest_id}'; provide steps for a custom quest")
        steps = template["steps"]
    normalized = _quest_steps(steps)
    if start_step >= len(normalized):
        raise ValueError("start_step must be less than the number of quest steps")
    completed: list[dict[str, Any]] = []
    last_observation = await _observe(agent_id, radius)
    for index, step in enumerate(normalized[start_step:], start=start_step):
        result = await _execute_quest_step(agent_id, step, radius)
        result_summary = {"index": index, "kind": step["kind"], **{key: value for key, value in result.items() if key != "observation"}}
        completed.append(result_summary)
        last_observation = result.get("observation") or await _observe(agent_id, radius)
        if result.get("status") != "complete":
            return {
                "status": result.get("status", "blocked"),
                "quest_id": quest_id,
                "custom": not template_steps,
                "completed_steps": index,
                "total_steps": len(normalized),
                "resume_from": index,
                "progress": completed,
                "blocker": result.get("reason") or result.get("message"),
                "safety": last_observation.get("safety"),
                "observation": last_observation,
            }
    quest_verification = None
    quest_state_result = await api.request("GET", f"/agents/{agent_id}/quests")
    for quest in quest_state_result.get("quests", []):
        if quest.get("id") == quest_id:
            quest_verification = quest
            break
    if quest_verification is None:
        # A custom graph resolving successfully is not proof that a real
        # quest completed. Unknown quest ids must remain explicitly
        # unverified so callers cannot turn a missing registry entry into a
        # false completion claim.
        final_status = "steps_complete_unverified"
        verification_note = "quest id is not registered by the running content registry"
    elif quest_verification.get("status") == "complete":
        final_status = "complete"
        verification_note = "quest completion confirmed by the running engine state"
    else:
        final_status = "steps_complete_unverified"
        verification_note = "all declared steps resolved, but the engine quest state is not complete"
    return {
        "status": final_status,
        "quest_id": quest_id,
        "custom": not template_steps,
        "completed_steps": len(normalized),
        "total_steps": len(normalized),
        "resume_from": len(normalized),
        "progress": completed,
        "quest_verification": quest_verification,
        "verification_note": verification_note,
        "safety": last_observation.get("safety"),
        "observation": last_observation,
    }


@mcp.tool
async def create_agent(
    username: str,
    x: int | None = None,
    z: int | None = None,
    level: int = 0,
) -> dict[str, Any]:
    """Create a headless player controlled by an agent.

    If x and z are omitted, the game's normal tutorial-island start tile is
    used. The returned agent id is required by all action tools.
    """

    payload: dict[str, Any] = {"username": username}
    if x is not None or z is not None or level != 0:
        if x is None or z is None:
            raise ValueError("x and z must be supplied together")
        payload["spawn"] = {"x": x, "z": z, "level": level}
    result = await api.request("POST", "/agents", json=payload)
    _tutorial_ranged_attacks.discard(result["id"])
    _tutorial_magic_attacks.discard(result["id"])
    result["observation"] = await _wait_for_agent_active(result["id"])
    _ensure_keepalive(result["id"], result["username"])
    return result


@mcp.tool
async def attach_agent(username: str) -> dict[str, Any]:
    """Attach an MCP control session to an already-online player."""

    result = await api.request("POST", "/agents/attach", json={"username": username})
    result["observation"] = await _wait_for_agent_active(result["id"])
    _ensure_keepalive(result["id"], result["username"])
    return result


@mcp.tool
async def list_agents() -> list[dict[str, Any]]:
    """List agent sessions created through the MCP/API bridge."""

    agents = await api.request("GET", "/agents")
    for agent in agents:
        if agent["id"] not in _keepalive_tasks:
            _ensure_keepalive(agent["id"], agent["username"])
    return agents


@mcp.tool
async def keep_alive_agent(
    agent_id: str,
    enabled: bool = True,
    interval_seconds: float = 10.0,
) -> dict[str, Any]:
    """Start or stop the persistent MCP keepalive supervisor for an agent.

    The TypeScript engine already refreshes attached sessions every second;
    this supervisor adds an MCP-side observation/reconnect loop. It never
    invents game credentials and does not recreate an account after an explicit
    logout.
    """

    if interval_seconds < 2 or interval_seconds > 60:
        raise ValueError("interval_seconds must be between 2 and 60")
    session = await api.request("GET", f"/agents/{agent_id}")
    if enabled:
        _ensure_keepalive(agent_id, session["username"], interval_seconds)
    else:
        await _stop_keepalive(agent_id)
    return {
        "agent_id": agent_id,
        "username": session["username"],
        "keepalive": _keepalive_state.get(agent_id, {"enabled": False, "status": "stopped"}),
    }


@mcp.tool
async def observe_agent(agent_id: str, radius: int = 16) -> dict[str, Any]:
    """Observe an agent and nearby players, NPCs, locations, and ground items."""

    return await api.request("GET", f"/agents/{agent_id}/observation", params={"radius": radius})


@mcp.tool
async def open_bank(agent_id: str, radius: int = 16) -> dict[str, Any]:
    """Open a nearby live bank booth or banker and validate the bank interface."""

    observation = await _observe(agent_id, radius)
    if _bank_is_open(observation):
        return {
            "status": "complete",
            "validated": True,
            "validation": "bank_interface_already_open",
            "observation": observation,
        }

    selected = _bank_target(observation)
    if selected is None:
        return {
            "status": "blocked",
            "validated": False,
            "reason": "no nearby bank booth or banker exposes a live Bank/Use/Talk-to option",
            "observation": observation,
        }

    action = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "interact", "target": selected["target"], "option": selected["option"]},
    )
    latest = action.get("observation") or observation
    # Opening a bank is a long live interaction: the engine can keep the
    # player on the banker target for several ticks before the modal and bank
    # listeners appear. A 12-second window caused valid actions to be
    # reported as failures and stranded subsequent withdrawal batches.
    for _ in range(120):
        if _bank_is_open(latest):
            return {
                "status": "complete",
                "validated": True,
                "validation": "bank_interface_opened_on_live_server",
                "selected": selected,
                "action": action,
                "observation": latest,
            }
        ui = latest.get("ui") or {}
        if ui.get("activeScript"):
            buttons = ui.get("resumeButtons") or []
            yes = _select_dialogue_button(latest, "yes") if buttons else None
            try:
                if yes is not None:
                    await press_button(agent_id, yes["component"])
                else:
                    await resume_dialogue(agent_id)
            except LostCityApiError:
                # A dialogue can finish between observation and the resume
                # request; the next authoritative observation decides.
                pass
        await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius)

    return {
        "status": "blocked",
        "validated": False,
        "validation": "bank_interface_open_timeout",
        "reason": "the live bank interface did not appear before the validation timeout",
        "selected": selected,
        "action": action,
        "observation": latest,
    }


@mcp.tool
async def deposit_item(
    agent_id: str,
    slot: int,
    amount: Literal[1, 5, 10, "all", "x"] = 1,
    quantity: int | None = None,
    inventory: int = 93,
) -> dict[str, Any]:
    """Deposit an inventory item through the live bank interface."""

    if amount == "x" and (quantity is None or quantity < 1):
        raise ValueError("quantity must be a positive integer when amount is 'x'")
    before = await _observe(agent_id, radius=8)
    if not _bank_is_open(before):
        opened = await open_bank(agent_id, radius=16)
        if opened.get("status") != "complete":
            return opened
        before = opened.get("observation") or await _observe(agent_id, radius=8)

    side_listener = next(
        (
            listener
            for current in before.get("agent", {}).get("inventories", [])
            if current.get("id") == inventory
            for listener in current.get("listeners", [])
            if listener.get("name") == "bank_side:inv"
        ),
        None,
    )
    item = next(
        (
            item
            for current in before.get("agent", {}).get("inventories", [])
            if current.get("id") == inventory
            for item in current.get("items", [])
            if item.get("slot") == slot
        ),
        None,
    )
    if side_listener is None:
        raise ValueError("the live bank deposit inventory is not open")
    if item is None:
        raise ValueError(f"no item exists in inventory {inventory} slot {slot}")

    action = await inventory_button(
        agent_id,
        side_listener["component"],
        inventory,
        slot,
        _bank_amount_option(amount),
    )
    count_result = None
    if amount == "x":
        count_observation = await _wait_for_count_dialog(agent_id)
        if (count_observation.get("ui") or {}).get("activeScriptExecution") != 4:
            return {
                "status": "blocked",
                "validated": False,
                "validation": "count_dialog_not_opened",
                "reason": "the live bank did not open a count dialog for the X deposit option",
                "action": action,
                "observation": count_observation,
            }
        count_result = await resume_count_dialog(agent_id, quantity or 0)
        if count_result.get("status") != "complete":
            return {**count_result, "operation": "deposit", "amount": amount, "quantity": quantity}
    result = await _resolve_bank_inventory_action(agent_id, action, before)
    return {
        **result,
        "operation": "deposit",
        "amount": amount,
        "quantity": quantity,
        "count_dialog": count_result,
        "selected": {"inventory": inventory, "slot": slot, "item": item},
    }


@mcp.tool
async def withdraw_item(
    agent_id: str,
    slot: int,
    amount: Literal[1, 5, 10, "all", "x"] = 1,
    quantity: int | None = None,
) -> dict[str, Any]:
    """Withdraw a bank item through the live bank interface."""

    if amount == "x" and (quantity is None or quantity < 1):
        raise ValueError("quantity must be a positive integer when amount is 'x'")
    before = await _observe(agent_id, radius=8)
    if not _bank_is_open(before):
        opened = await open_bank(agent_id, radius=16)
        if opened.get("status") != "complete":
            return opened
        before = opened.get("observation") or await _observe(agent_id, radius=8)

    bank_inventory = next(
        (
            current
            for current in before.get("agent", {}).get("inventories", [])
            if any(listener.get("name") == "bank_main:inv" for listener in current.get("listeners", []))
        ),
        None,
    )
    if bank_inventory is None:
        raise ValueError("the live bank withdrawal inventory is not open")
    main_listener = next(
        listener for listener in bank_inventory.get("listeners", []) if listener.get("name") == "bank_main:inv"
    )
    item = next((item for item in bank_inventory.get("items", []) if item.get("slot") == slot), None)
    if item is None:
        raise ValueError(f"no bank item exists in slot {slot}")

    action = await inventory_button(
        agent_id,
        main_listener["component"],
        bank_inventory["id"],
        slot,
        _bank_amount_option(amount),
    )
    count_result = None
    if amount == "x":
        count_observation = await _wait_for_count_dialog(agent_id)
        if (count_observation.get("ui") or {}).get("activeScriptExecution") != 4:
            return {
                "status": "blocked",
                "validated": False,
                "validation": "count_dialog_not_opened",
                "reason": "the live bank did not open a count dialog for the X withdrawal option",
                "action": action,
                "observation": count_observation,
            }
        count_result = await resume_count_dialog(agent_id, quantity or 0)
        if count_result.get("status") != "complete":
            return {**count_result, "operation": "withdraw", "amount": amount, "quantity": quantity}
    result = await _resolve_bank_inventory_action(agent_id, action, before)
    return {
        **result,
        "operation": "withdraw",
        "amount": amount,
        "quantity": quantity,
        "count_dialog": count_result,
        "selected": {"inventory": bank_inventory["id"], "slot": slot, "item": item},
    }


@mcp.tool
async def close_bank(agent_id: str) -> dict[str, Any]:
    """Close the live bank interface and validate that its listeners cleared."""

    before = await _observe(agent_id, radius=1)
    if not _bank_is_open(before):
        return {
            "status": "complete",
            "validated": True,
            "validation": "bank_interface_already_closed",
            "observation": before,
        }
    action = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "close_interface"},
    )
    latest = action.get("observation") or before
    for _ in range(20):
        await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius=1)
        if not _bank_is_open(latest):
            return {
                "status": "complete",
                "validated": True,
                "validation": "bank_interface_closed_on_live_server",
                "action": action,
                "observation": latest,
            }
    return {
        "status": "blocked",
        "validated": False,
        "validation": "bank_interface_close_timeout",
        "reason": "the live bank listeners did not clear before the validation timeout",
        "action": action,
        "observation": latest,
    }


@mcp.tool
async def open_shop(agent_id: str, radius: int = 16) -> dict[str, Any]:
    """Open a nearby live shopkeeper interface and validate its listeners."""

    observation = await _observe(agent_id, radius)
    if _shop_is_open(observation):
        return {
            "status": "complete",
            "validated": True,
            "validation": "shop_interface_already_open",
            "observation": observation,
        }
    selected = _shop_target(observation)
    if selected is None:
        return {
            "status": "blocked",
            "validated": False,
            "reason": "no nearby shopkeeper exposes a live Trade/Talk-to option",
            "observation": observation,
        }
    action = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "interact", "target": selected["target"], "option": selected["option"]},
    )
    latest = action.get("observation") or observation
    for _ in range(48):
        if _shop_is_open(latest):
            return {
                "status": "complete",
                "validated": True,
                "validation": "shop_interface_opened_on_live_server",
                "selected": selected,
                "action": action,
                "observation": latest,
            }
        ui = latest.get("ui") or {}
        if ui.get("activeScript"):
            buttons = ui.get("resumeButtons") or []
            yes = _select_dialogue_button(latest, "yes") if buttons else None
            try:
                if yes is not None:
                    await press_button(agent_id, yes["component"])
                else:
                    await resume_dialogue(agent_id)
            except LostCityApiError:
                pass
        await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius)
    return {
        "status": "blocked",
        "validated": False,
        "validation": "shop_interface_open_timeout",
        "reason": "the live shop interface did not appear before the validation timeout",
        "selected": selected,
        "action": action,
        "observation": latest,
    }


@mcp.tool
async def buy_item(
    agent_id: str,
    slot: int,
    amount: Literal[1, 5, 10] = 1,
) -> dict[str, Any]:
    """Buy a shop item through the live shop inventory."""

    before = await _observe(agent_id, radius=8)
    if not _shop_is_open(before):
        opened = await open_shop(agent_id, radius=16)
        if opened.get("status") != "complete":
            return opened
        before = opened.get("observation") or await _observe(agent_id, radius=8)
    shop_inventory = next(
        (
            current
            for current in before.get("agent", {}).get("inventories", [])
            if any(listener.get("name") == "shop_template:inv" for listener in current.get("listeners", []))
        ),
        None,
    )
    if shop_inventory is None:
        raise ValueError("the live shop inventory is not open")
    listener = next(listener for listener in shop_inventory.get("listeners", []) if listener.get("name") == "shop_template:inv")
    item = next((item for item in shop_inventory.get("items", []) if item.get("slot") == slot), None)
    if item is None:
        raise ValueError(f"no shop item exists in slot {slot}")
    action = await inventory_button(agent_id, listener["component"], shop_inventory["id"], slot, {1: 2, 5: 3, 10: 4}[amount])
    result = await _resolve_shop_inventory_action(agent_id, action, before)
    return {
        **result,
        "operation": "buy",
        "amount": amount,
        "selected": {"inventory": shop_inventory["id"], "slot": slot, "item": item, "listener": listener},
    }


@mcp.tool
async def sell_item(
    agent_id: str,
    slot: int,
    amount: Literal[1, 5, 10] = 1,
    inventory: int = 93,
) -> dict[str, Any]:
    """Sell an inventory item through the live shop interface."""

    before = await _observe(agent_id, radius=8)
    if not _shop_is_open(before):
        opened = await open_shop(agent_id, radius=16)
        if opened.get("status") != "complete":
            return opened
        before = opened.get("observation") or await _observe(agent_id, radius=8)
    current = next(
        (current for current in before.get("agent", {}).get("inventories", []) if current.get("id") == inventory),
        None,
    )
    if current is None:
        raise ValueError(f"inventory {inventory} is unavailable")
    listener = next((listener for listener in current.get("listeners", []) if listener.get("name") == "shop_template_side:inv"), None)
    item = next((item for item in current.get("items", []) if item.get("slot") == slot), None)
    if listener is None:
        raise ValueError("the live shop sell inventory is not open")
    if item is None:
        raise ValueError(f"no item exists in inventory {inventory} slot {slot}")
    action = await inventory_button(agent_id, listener["component"], inventory, slot, {1: 2, 5: 3, 10: 4}[amount])
    result = await _resolve_shop_inventory_action(agent_id, action, before)
    return {
        **result,
        "operation": "sell",
        "amount": amount,
        "selected": {"inventory": inventory, "slot": slot, "item": item, "listener": listener},
    }


@mcp.tool
async def close_shop(agent_id: str) -> dict[str, Any]:
    """Close the live shop interface and validate that its listeners cleared."""

    before = await _observe(agent_id, radius=1)
    if not _shop_is_open(before):
        return {
            "status": "complete",
            "validated": True,
            "validation": "shop_interface_already_closed",
            "observation": before,
        }
    action = await api.request("POST", f"/agents/{agent_id}/actions", json={"type": "close_interface"})
    latest = action.get("observation") or before
    for _ in range(20):
        await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius=1)
        if not _shop_is_open(latest):
            return {
                "status": "complete",
                "validated": True,
                "validation": "shop_interface_closed_on_live_server",
                "action": action,
                "observation": latest,
            }
    return {
        "status": "blocked",
        "validated": False,
        "validation": "shop_interface_close_timeout",
        "reason": "the live shop listeners did not clear before the validation timeout",
        "action": action,
        "observation": latest,
    }


@mcp.tool
async def equip_item(agent_id: str, slot: int, inventory: int = 93) -> dict[str, Any]:
    """Equip a live inventory item and validate the worn-inventory mutation."""

    before = await _observe(agent_id, radius=1)
    item = next(
        (
            item
            for current in before.get("agent", {}).get("inventories", [])
            if current.get("id") == inventory
            for item in current.get("items", [])
            if item.get("slot") == slot
        ),
        None,
    )
    if item is None:
        raise ValueError(f"no item exists in inventory {inventory} slot {slot}")
    option = next(
        (
            index + 1
            for index, label in enumerate(item.get("options") or [])
            if isinstance(label, str) and label.casefold() in {"wield", "wear", "equip"}
        ),
        None,
    )
    if option is None:
        raise ValueError(f"item '{item.get('name')}' does not expose a live equip option")
    action = await item_op_agent(agent_id, inventory, slot, option)
    result = await _resolve_inventory_action(
        agent_id,
        action,
        before,
        validation="equipment_changed_on_live_server",
    )
    return {
        **result,
        "operation": "equip",
        "selected": {"inventory": inventory, "slot": slot, "option": option, "item": item},
    }


@mcp.tool
async def unequip_item(agent_id: str, slot: int) -> dict[str, Any]:
    """Remove a live worn item and validate the inventory mutation."""

    before = await _observe(agent_id, radius=1)
    worn = next((current for current in before.get("agent", {}).get("inventories", []) if current.get("id") == 94), None)
    if worn is None:
        raise ValueError("the live worn inventory is unavailable")
    item = next((item for item in worn.get("items", []) if item.get("slot") == slot), None)
    listener = next((listener for listener in worn.get("listeners", []) if listener.get("name") == "wornitems:wear"), None)
    if item is None:
        raise ValueError(f"no worn item exists in slot {slot}")
    if listener is None:
        raise ValueError("the live worn inventory removal listener is unavailable")
    action = await inventory_button(agent_id, listener["component"], worn["id"], slot, 1)
    result = await _resolve_inventory_action(
        agent_id,
        action,
        before,
        validation="equipment_removed_on_live_server",
    )
    return {
        **result,
        "operation": "unequip",
        "selected": {"inventory": worn["id"], "slot": slot, "option": 1, "item": item},
    }


@mcp.tool
async def select_production_product(
    agent_id: str,
    inventory: int,
    slot: int,
    option: int = 1,
) -> dict[str, Any]:
    """Select a live production product and validate the resulting state change.

    Production interfaces expose their product columns as authoritative
    inventory listeners. This wrapper discovers that listener from the live
    snapshot and validates the product click; recipe choice dialogs and
    quantity-input dialogs remain explicit follow-up actions.
    """

    if option < 1 or option > 5:
        raise ValueError("option must be between 1 and 5")
    before = await _observe(agent_id, radius=1)
    current = next(
        (current for current in before.get("agent", {}).get("inventories", []) if current.get("id") == inventory),
        None,
    )
    if current is None:
        raise ValueError(f"the live production inventory {inventory} is unavailable")
    listener = next((listener for listener in current.get("listeners", []) if listener.get("component") is not None), None)
    if listener is None:
        raise ValueError(f"inventory {inventory} has no live production listener")
    item = next((item for item in current.get("items", []) if item.get("slot") == slot), None)
    if item is None:
        raise ValueError(f"no production product exists in inventory {inventory} slot {slot}")

    action = await inventory_button(agent_id, listener["component"], inventory, slot, option)
    result = await _resolve_inventory_action(
        agent_id,
        action,
        before,
        validation="production_action_changed_live_state",
    )
    return {
        **result,
        "operation": "select_production_product",
        "selected": {
            "inventory": inventory,
            "slot": slot,
            "option": option,
            "item": item,
            "listener": listener,
        },
    }


@mcp.tool
async def skill_status(agent_id: str, skill: str | None = None) -> dict[str, Any]:
    """Read skill levels/XP together with the current health safety state."""

    observation = await api.request(
        "GET",
        f"/agents/{agent_id}/observation",
        params={"radius": 1},
    )
    skills = observation["agent"].get("skills", {})
    if skill is not None:
        skill = _normalized_skill(skill)
        skills = {skill: skills[skill]}
    return {
        "agent_id": agent_id,
        "username": observation["agent"].get("username"),
        "skills": skills,
        "safety": observation.get("safety"),
        "position": observation["agent"].get("position"),
        "target": observation["agent"].get("target"),
    }


@mcp.tool
async def consume_food(agent_id: str, minimum_health_ratio: float = 0.7) -> dict[str, Any]:
    """Use the first live inventory item exposing Eat or Drink when health is low."""

    if minimum_health_ratio <= 0 or minimum_health_ratio > 1:
        raise ValueError("minimum_health_ratio must be greater than 0 and at most 1")
    observation = await _observe(agent_id, radius=1)
    health = observation["safety"]["health"]
    maximum = observation["safety"]["maxHealth"]
    if maximum > 0 and health / maximum >= minimum_health_ratio:
        return {
            "status": "not_needed",
            "health": observation["safety"],
            "observation": observation,
        }
    for inventory in observation["agent"].get("inventories", []):
        for item in inventory.get("items", []):
            options = item.get("options") or []
            option = next(
                (
                    index + 1
                    for index, label in enumerate(options)
                    if isinstance(label, str) and label.casefold() in {"eat", "drink"}
                ),
                None,
            )
            if option is None:
                continue
            result = await item_op_agent(agent_id, inventory["id"], item["slot"], option)
            return {
                "status": "food_queued",
                "selected": {"inventory": inventory["id"], "item": item, "option": option},
                "result": result,
            }
    return {
        "status": "no_food",
        "message": "no inventory item exposes an Eat or Drink option",
        "health": observation["safety"],
        "observation": observation,
    }


@mcp.tool
async def item_op_agent(agent_id: str, inventory: int, slot: int, option: int) -> dict[str, Any]:
    """Execute one live inventory option after the engine validates its label and script."""

    if option < 1 or option > 5:
        raise ValueError("option must be between 1 and 5")
    return await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "item_op", "inventory": inventory, "slot": slot, "option": option},
    )


@mcp.tool
async def drop_item(agent_id: str, slot: int, inventory: int = 93) -> dict[str, Any]:
    """Drop one live inventory stack after validating its Drop option."""

    before = await _observe(agent_id, radius=1)
    item = next(
        (
            item
            for current in before.get("agent", {}).get("inventories", [])
            if current.get("id") == inventory
            for item in current.get("items", [])
            if item.get("slot") == slot
        ),
        None,
    )
    if item is None:
        raise ValueError(f"no item exists in inventory {inventory} slot {slot}")
    options = item.get("options") or []
    drop_option = next(
        (
            index + 1
            for index, label in enumerate(options)
            if isinstance(label, str) and label.casefold() == "drop"
        ),
        None,
    )
    if drop_option is None:
        raise ValueError(f"item '{item.get('name')}' does not expose a live Drop option")
    action = await item_op_agent(agent_id, inventory, slot, drop_option)
    result = await _resolve_inventory_action(
        agent_id,
        action,
        before,
        radius=4,
        validation="item_dropped_on_live_server",
    )
    return {
        **result,
        "operation": "drop",
        "selected": {"inventory": inventory, "slot": slot, "option": drop_option, "item": item},
    }


@mcp.tool
async def skill_plan(
    agent_id: str,
    skill: str,
    target_level: int | None = None,
    radius: int = 16,
) -> dict[str, Any]:
    """Build a live, content-aware plan for training one skill.

    The plan reports the current level, nearby viable options, safety state,
    and any missing item/UI capability. It never fabricates XP or assumes an
    interaction option that was not exported by the game.
    """

    skill = _normalized_skill(skill)
    spec = SKILL_PLAYBOOK[skill]
    observation = await api.request(
        "GET",
        f"/agents/{agent_id}/observation",
        params={"radius": radius},
    )
    current = _skill_state(observation, skill)
    if target_level is not None and (target_level < 1 or target_level > 99):
        raise ValueError("target_level must be between 1 and 99")
    action = (
        _discover_skill_action(observation, spec)
        if spec["mode"] == "interaction"
        else _discover_inventory_skill_action(observation, skill)
        if spec["mode"] == "inventory"
        else _discover_context_skill_action(observation, skill)
        if spec["mode"] == "context"
        else None
    )
    return {
        "agent_id": agent_id,
        "skill": skill,
        "current": current,
        "target_level": target_level,
        "mode": spec["mode"],
        "strategy": spec["strategy"],
        "missing": spec.get("missing", []),
        "next_action": action,
        "safety": observation.get("safety"),
        "nearby_counts": {
            kind: len(observation.get("nearby", {}).get(kind, []))
            for kind in ("npcs", "locations", "objects")
        },
    }


@mcp.tool
async def attack_nearest_npc(
    agent_id: str,
    npc_name: str | None = None,
    radius: int = 16,
) -> dict[str, Any]:
    """Find the nearest attackable NPC and start its Attack interaction.

    The tool reads the live NPC option list, so it does not assume that Attack
    is always option 1. It returns the selected NPC and the engine observation
    after the interaction is queued.
    """

    wanted = npc_name.casefold() if npc_name else None
    candidates: list[tuple[int, dict[str, Any], int]] = []
    for attempt in range(8):
        observation = await api.request(
            "GET",
            f"/agents/{agent_id}/observation",
            params={"radius": radius},
        )
        agent_position = observation["agent"]["position"]
        candidates = []
        for npc in observation.get("nearby", {}).get("npcs", []):
            name = (npc.get("name") or "").casefold()
            if wanted and name != wanted:
                continue
            options = npc.get("options") or []
            attack_index = next(
                (index for index, option in enumerate(options) if isinstance(option, str) and option.casefold() == "attack"),
                None,
            )
            if attack_index is None:
                continue
            position = npc["position"]
            distance = max(abs(position["x"] - agent_position["x"]), abs(position["z"] - agent_position["z"]))
            candidates.append((distance, npc, attack_index + 1))
        if candidates or attempt == 7:
            break
        await asyncio.sleep(0.25)

    if not candidates:
        label = f" named '{npc_name}'" if npc_name else ""
        raise ValueError(f"no attackable NPC{label} found within radius {radius}")

    distance, target, option = min(candidates, key=lambda item: (item[0], item[1]["id"]))
    result = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={
            "type": "interact",
            "target": {"kind": "npc", "id": target["id"]},
            "option": option,
        },
    )
    return {
        "selected": target,
        "distance": distance,
        "attack_option": option,
        "result": result,
    }


@mcp.tool
async def skill_step(
    agent_id: str,
    skill: str,
    radius: int = 16,
    target_kind: Literal["npc", "loc", "obj"] | None = None,
    target_id: int | None = None,
    x: int | None = None,
    z: int | None = None,
    level: int | None = None,
    option: int | None = None,
) -> dict[str, Any]:
    """Perform one guarded training step for a skill.

    For interaction-driven skills, the MCP discovers a live option and target
    from the observation. Inventory-driven skills choose item pairs from the
    live inventory snapshot. Explicit target fields are accepted for callers
    that already have a known route or recipe. Context-driven skills return a
    structured prerequisite response when recipe, banking, equipment, or UI
    policy is still missing.
    """

    skill = _normalized_skill(skill)
    spec = SKILL_PLAYBOOK[skill]
    observation = await api.request(
        "GET",
        f"/agents/{agent_id}/observation",
        params={"radius": radius},
    )
    safety = observation.get("safety") or {}
    if safety.get("lowHealth") or safety.get("fleeing"):
        return {
            "status": "safety_pause",
            "skill": skill,
            "message": "health is below the configured threshold; the engine safety controller is handling escape",
            "safety": safety,
            "observation": observation,
        }
    if spec["mode"] == "disabled":
        return {
            "status": "disabled",
            "skill": skill,
            "strategy": spec["strategy"],
            "missing": spec.get("missing", []),
            "current": _skill_state(observation, skill),
            "safety": safety,
        }

    if spec["mode"] == "context":
        latest = observation
        last_result: dict[str, Any] | None = None
        selected: dict[str, Any] | None = None
        excluded_target_ids: set[int] = set()
        # Shearing has a content-defined chance to fail and make the sheep
        # flee. Retry only that live, bounded recipe; every attempt still has
        # to prove the Wool inventory delta before this tool reports success.
        max_attempts = 8 if skill == "crafting" else 1
        for attempt in range(max_attempts):
            selected = _discover_context_skill_action(latest, skill, excluded_target_ids)
            if selected is None:
                return {
                    "status": "needs_context" if last_result is None else "blocked",
                    "skill": skill,
                    "strategy": spec["strategy"],
                    "missing": spec.get("missing", []),
                    "current": _skill_state(latest, skill),
                    "safety": (latest.get("safety") or safety),
                    "reason": "no live context action is currently available" if last_result is None else "recipe did not produce its required postcondition",
                    "selected": selected,
                    "result": last_result,
                    "observation": latest,
                }
            target = selected["target"]
            before_wool = _inventory_count_exact(latest, "Wool")
            before_balls = _inventory_count_exact(latest, "Ball of wool")
            before_raw = _inventory_count_exact(latest, selected.get("raw_name", "")) if selected.get("raw_name") else 0
            before_outputs = {
                name: _inventory_count_exact(latest, name)
                for name in selected.get("output_names", ())
            }
            before_xp = _skill_state(latest, skill)["experience"]
            last_result = await use_item_on_validated(
                agent_id,
                selected["inventory"],
                selected["slot"],
                target["kind"],
                target_id=target.get("id"),
                x=target.get("x"),
                z=target.get("z"),
                level=target.get("level"),
                radius=radius,
            )
            latest = last_result.get("observation") or await _observe(agent_id, radius)
            if selected.get("recipe") == "wool_to_ball_of_wool":
                # The spinning script can award the XP and replace the item
                # on adjacent ticks. Do not let a fast repeated batch report
                # a false failure in that interval; keep the same live action
                # bounded and wait for its inventory postcondition or idle
                # operation state before deciding that it failed.
                for _ in range(20):
                    ball_ready = _inventory_count_exact(latest, "Ball of wool") > before_balls
                    xp_ready = _skill_state(latest, skill)["experience"] > before_xp
                    if ball_ready and xp_ready:
                        break
                    if (latest.get("agent") or {}).get("targetOperation") is None and _ > 3:
                        break
                    await asyncio.sleep(0.25)
                    latest = await _observe(agent_id, radius)
            if selected.get("recipe") == "cook_raw_food":
                # Cooking can award XP before the delayed inventory mutation
                # replaces the raw item. Keep observing even when the action
                # operation has already gone idle, then prove the replacement
                # rather than accepting XP alone as the recipe result.
                # The live range script can take several server ticks to
                # finish after the item-on-location request resolves.  A
                # longer bounded window prevents a real cook from being
                # reported as a batch failure while still requiring both
                # raw-item removal and a cooked/burnt output.
                for _ in range(60):
                    raw_ready = _inventory_count_exact(latest, selected["raw_name"]) < before_raw
                    output_ready = any(
                        _inventory_count_exact(latest, name) > count
                        for name, count in before_outputs.items()
                    )
                    if raw_ready and output_ready:
                        break
                    await asyncio.sleep(0.25)
                    latest = await _observe(agent_id, radius)
            wool_delta = _inventory_count_exact(latest, "Wool") > before_wool
            ball_delta = _inventory_count_exact(latest, "Ball of wool") > before_balls
            xp_delta = _skill_state(latest, skill)["experience"] > before_xp
            raw_delta = (
                _inventory_count_exact(latest, selected["raw_name"]) < before_raw
                if selected.get("recipe") == "cook_raw_food"
                else False
            )
            output_delta = any(
                _inventory_count_exact(latest, name) > count
                for name, count in before_outputs.items()
            )
            expected_effect = (
                wool_delta
                if selected.get("recipe") == "shear_sheep"
                else ball_delta and xp_delta
                if selected.get("recipe") == "wool_to_ball_of_wool"
                else raw_delta and output_delta
                if selected.get("recipe") == "cook_raw_food"
                else bool(last_result.get("validated"))
            )
            if expected_effect:
                return {
                    "status": "action_queued",
                    "skill": skill,
                    "selected": selected,
                    "validated": True,
                    "validation": "recipe_postcondition_observed_on_live_server",
                    "attempts": attempt + 1,
                    "current": _skill_state(latest, skill),
                    "safety": latest.get("safety"),
                    "result": last_result,
                    "observation": latest,
                }
            if selected.get("recipe") != "shear_sheep":
                break
            # Item-on-NPC movement can finish after the interaction resolver
            # has observed a target-side change. Let that operation become
            # idle before choosing another sheep, then re-check the inventory
            # delta so a delayed Wool result is not mistaken for failure.
            for _ in range(20):
                if (latest.get("agent") or {}).get("targetOperation") is None:
                    break
                await asyncio.sleep(0.25)
                latest = await _observe(agent_id, radius)
            wool_delta = _inventory_count_exact(latest, "Wool") > before_wool
            if wool_delta:
                return {
                    "status": "action_queued",
                    "skill": skill,
                    "selected": selected,
                    "validated": True,
                    "validation": "recipe_postcondition_observed_after_operation_idle",
                    "attempts": attempt + 1,
                    "current": _skill_state(latest, skill),
                    "safety": latest.get("safety"),
                    "result": last_result,
                    "observation": latest,
                }
            target_id = selected.get("target", {}).get("id")
            if isinstance(target_id, int):
                excluded_target_ids.add(target_id)
        return {
            "status": "blocked",
            "skill": skill,
            "selected": selected,
            "validated": False,
            "validation": "recipe_postcondition_missing",
            "reason": "the live action resolved without producing the expected crafting result",
            "attempts": max_attempts,
            "current": _skill_state(latest, skill),
            "safety": latest.get("safety"),
            "result": last_result,
            "observation": latest,
        }

    if spec["mode"] == "combat":
        current_target = observation["agent"].get("target")
        if current_target and current_target.get("kind") == "npc" and observation["agent"].get("targetOperation") is not None:
            return {
                "status": "already_engaged",
                "skill": skill,
                "current": _skill_state(observation, skill),
                "target": current_target,
                "safety": safety,
                "observation": observation,
            }
        result = await attack_nearest_npc(agent_id, radius=radius)
        return {
            "status": "action_queued",
            "skill": skill,
            "current": _skill_state(result["result"]["observation"], skill),
            "action": result,
            "safety": result["result"]["observation"].get("safety"),
        }

    if spec["mode"] == "inventory":
        selected = _discover_inventory_skill_action(observation, skill)
        if selected is None:
            return {
                "status": "no_action_found",
                "skill": skill,
                "message": "no live inventory recipe was found for this skill",
                "current": _skill_state(observation, skill),
                "safety": safety,
                "observation": observation,
            }
        result = await use_item(
            agent_id,
            selected["inventory"],
            selected["slot"],
            selected["use_inventory"],
            selected["use_slot"],
        )
        await asyncio.sleep(0.5)
        latest = await _observe(agent_id, radius=radius)
        return {
            "status": "action_queued",
            "skill": skill,
            "selected": selected,
            "validated": True,
            "validation": "inventory_action_accepted",
            "current": _skill_state(latest, skill),
            "safety": latest.get("safety"),
            "result": result,
            "observation": latest,
        }

    if target_kind is not None:
        if target_kind not in spec["targets"]:
            raise ValueError(f"skill '{skill}' does not use target kind '{target_kind}'")
        if option is None or option < 1 or option > 5:
            raise ValueError("option must be between 1 and 5 when an explicit target is supplied")
        if target_id is None:
            raise ValueError("target_id is required for an explicit target")
        target: dict[str, Any] = {"kind": target_kind, "id": target_id}
        if target_kind in {"loc", "obj"}:
            if x is None or z is None:
                raise ValueError("x and z are required for location/object targets")
            target.update({"x": x, "z": z})
            if level is not None:
                target["level"] = level
        selected = {"target": target, "option": option, "label": "explicit"}
    else:
        selected = _discover_skill_action(observation, spec)
        if selected is None:
            return {
                "status": "no_action_found",
                "skill": skill,
                "message": "no eligible nearby entity exposed a matching live interaction option; observed options may be gated by content requirements",
                "current": _skill_state(observation, skill),
                "safety": safety,
                "observation": observation,
            }

    fishing_output_names = (
        "Raw shrimps",
        "Raw anchovies",
        "Raw sardine",
        "Raw herring",
        "Raw trout",
        "Raw salmon",
        "Raw pike",
    )
    before_fishing_output = sum(_inventory_count_exact(observation, name) for name in fishing_output_names)
    before_skill_xp = _skill_state(observation, skill)["experience"]
    before_coins = _inventory_count_exact(observation, "Coins") if skill == "thieving" else 0

    result = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={
            "type": "interact",
            "target": selected["target"],
            "option": selected["option"],
        },
    )
    resolved = await _resolve_interaction_action(
        agent_id,
        result,
        radius,
        # Mining swings resolve on the server's resource tick and can take
        # slightly longer than the generic interaction window.  Keep the
        # action live-validated instead of reporting a timeout just before
        # the ore delta arrives.
        # Resource interactions resolve on delayed server ticks.  Fishing is
        # especially prone to crossing the generic interaction timeout
        # because its visible cast, hidden follow-up, and catch result are
        # separate NPC operations; woodcutting can likewise finish its tree
        # swing after the initial target operation has gone idle. Keep the
        # wait bounded, but long enough for a batched trainer to observe the
        # live inventory/XP postcondition instead of reporting a false
        # timeout while the log is still being awarded.
        max_wait_seconds=12.0 if skill in {"mining", "fishing", "woodcutting"} else 6.0,
        before_observation=observation,
        # Fishing content intentionally requeues the spot interaction for the
        # next catch cycle.  A real catch is therefore a valid postcondition
        # even while that repeating operation remains pending; the inventory
        # delta is still required by the resolver before it returns success.
        allow_pending_observable_effect=skill == "fishing",
    )
    if skill == "fishing":
        # Fishing spots deliberately requeue their cast operation, and their
        # generic nearby signature can change for unrelated reasons (for
        # example a ground item spawn). Never treat that generic effect as a
        # catch. Require the authoritative skill-specific postcondition even
        # when the resolver returned an early observable interaction effect.
        latest = resolved.get("observation") or observation
        fishing_output = sum(_inventory_count_exact(latest, name) for name in fishing_output_names)
        fishing_xp = _skill_state(latest, skill)["experience"]
        if fishing_output <= before_fishing_output and fishing_xp <= before_skill_xp:
            for _ in range(40):
                await asyncio.sleep(0.5)
                latest = await _observe(agent_id, radius)
                fishing_output = sum(_inventory_count_exact(latest, name) for name in fishing_output_names)
                fishing_xp = _skill_state(latest, skill)["experience"]
                if fishing_output > before_fishing_output or fishing_xp > before_skill_xp:
                    break
        if fishing_output > before_fishing_output or fishing_xp > before_skill_xp:
            resolved = {
                **resolved,
                "status": "complete",
                "validated": True,
                "validation": "fishing_inventory_or_xp_postcondition_observed",
                "observation": latest,
            }
        else:
            # Fishing has a legitimate no-catch outcome: the live script can
            # complete its cast and roll without awarding an item or XP.  The
            # rendered cast/attempt message proves that the server accepted
            # this attempt, so mark it as a validated retryable action rather
            # than stopping an otherwise safe batch.  Level/equipment/modal
            # failures do not use these messages and remain blocked.
            message = ((latest.get("agent") or {}).get("lastGameMessage") or "").casefold()
            attempt_observed = any(
                token in message
                for token in (
                    "you cast out your net",
                    "you cast out your line",
                    "you attempt to catch a fish",
                )
            )
            resolved = {
                **resolved,
                "status": "action_observed" if attempt_observed else "blocked",
                "validated": attempt_observed,
                "validation": (
                    "fishing_attempt_observed_without_catch"
                    if attempt_observed
                    else "fishing_postcondition_not_observed"
                ),
                "reason": (
                    "the live fishing attempt completed without a catch; retry is safe"
                    if attempt_observed
                    else "the live fishing cast did not add a fish or Fishing XP before the validation timeout"
                ),
                "observation": latest,
            }
    elif skill == "thieving":
        # Pickpocketing has two legitimate live outcomes: a successful loot
        # and an explicitly reported failure/stun. Do not accept the generic
        # interaction resolver's pending-effect signal as proof of a theft.
        latest = resolved.get("observation") or observation
        coins = _inventory_count_exact(latest, "Coins")
        thieving_xp = _skill_state(latest, skill)["experience"]
        message = ((latest.get("agent") or {}).get("lastGameMessage") or "").casefold()
        known_attempt = any(
            token in message
            for token in (
                "fail to pick",
                "you're stunned",
                "you are stunned",
                "successfully pick",
                "you steal",
                "you get some coins",
            )
        )
        if coins > before_coins or thieving_xp > before_skill_xp:
            resolved = {
                **resolved,
                "status": "complete",
                "validated": True,
                "validation": "thieving_coin_or_xp_postcondition_observed",
                "observation": latest,
            }
        elif known_attempt:
            resolved = {
                **resolved,
                "status": "action_observed",
                "validated": True,
                "validation": "thieving_attempt_outcome_observed_on_live_server",
                "observation": latest,
            }
        else:
            resolved = {
                **resolved,
                "status": "blocked",
                "validated": False,
                "validation": "thieving_postcondition_not_observed",
                "reason": "the live pickpocket action produced neither loot/XP nor a recognized failure message",
                "observation": latest,
            }
    return {
        "status": "action_queued" if resolved["status"] == "complete" else resolved["status"],
        "skill": skill,
        "selected": selected,
        "validated": resolved.get("validated", False),
        "validation": resolved.get("validation"),
        "current": _skill_state(resolved["observation"], skill),
        "safety": resolved["observation"].get("safety"),
        "result": resolved,
        "observation": resolved["observation"],
    }


COOKING_RESOURCE_SOURCE = {"x": 3087, "z": 3230, "level": 0}
COOKING_RESOURCE_STATION = {"x": 3101, "z": 3281, "level": 0}
COOKING_RESOURCE_STATION_APPROACH = {"x": 3100, "z": 3281, "level": 0}
COOKING_RESOURCE_GATE = {"x": 3107, "z": 3273, "level": 0}
COOKING_RESOURCE_DOOR = {"x": 3100, "z": 3276, "level": 0}
COOKING_RAW_NAMES = (
    "Raw shrimps",
    "Raw sardine",
    "Raw anchovies",
    "Raw herring",
    "Raw mackerel",
    "Raw trout",
    "Raw salmon",
    "Raw pike",
    "Raw cod",
    "Raw tuna",
    "Raw lobster",
    "Raw swordfish",
    "Raw chicken",
    "Raw beef",
    "Raw meat",
    "Raw rat meat",
)
COOKING_DISCARD_NAMES = {
    "shrimps",
    "anchovies",
    "sardine",
    "herring",
    "mackerel",
    "trout",
    "salmon",
    "pike",
    "cod",
    "tuna",
    "lobster",
    "swordfish",
    "cooked chicken",
    "cooked meat",
    "burnt fish",
    "burnt lobster",
    "burnt chicken",
    "burnt meat",
    "burnt swordfish",
}


def _carried_inventory(observation: dict[str, Any]) -> dict[str, Any]:
    return next(
        (inventory for inventory in observation.get("agent", {}).get("inventories", []) if inventory.get("id") == 93),
        {"id": 93, "freeSlots": 0, "items": []},
    )


def _carried_count_any(observation: dict[str, Any], names: tuple[str, ...]) -> int:
    wanted = {name.casefold() for name in names}
    return sum(
        int(item.get("count", 0))
        for item in _carried_inventory(observation).get("items", [])
        if (item.get("name") or "").casefold() in wanted
    )


async def _train_cooking_with_live_resources(
    agent_id: str,
    target_level: int,
    radius: int,
    max_steps: int,
    step_delay_seconds: float,
    initial_observation: dict[str, Any],
    initial: dict[str, Any],
) -> dict[str, Any]:
    """Train Cooking through a bounded live fishing-and-cooking route.

    This is intentionally a small, explicit route backed by content that was
    observed on the running server: Draynor net spots produce Raw shrimps and
    the Lumbridge castle range accepts them. It does not teleport, grant XP,
    or invent inventory state. If the route or resource policy is insufficient,
    the result is a resumable, explicit blocker.
    """

    steps: list[dict[str, Any]] = []
    latest = initial_observation
    baseline = initial

    async def open_checkpoint_door() -> dict[str, Any]:
        observation = await _observe(agent_id, radius=max(16, radius))
        nearby = observation.get("nearby") or {}
        candidates = [
            entity
            for entity in nearby.get("locations", [])
            if (entity.get("name") or "").casefold() == "door"
            and (entity.get("position") or {}).get("x") == COOKING_RESOURCE_DOOR["x"]
            # The live loc resolver represents this hinged door one tile
            # farther north after it opens.  Accept only that exact adjacent
            # representation; the x-coordinate and one-tile bound still
            # prevent selecting another nearby building door.
            and abs((entity.get("position") or {}).get("z", 0) - COOKING_RESOURCE_DOOR["z"]) <= 1
        ]
        filtered = {**observation, "nearby": {**nearby, "locations": candidates}}
        action = _discover_skill_action(
            filtered,
            {"targets": ["loc"], "target_name": "Door", "option_tokens": ["open"]},
        )
        if action is None:
            already_open = any(
                any(isinstance(option, str) and option.casefold() == "close" for option in entity.get("options") or [])
                for entity in candidates
            )
            if already_open:
                return {"status": "complete", "validated": True, "validation": "checkpoint_door_already_open", "observation": observation}
            return {"status": "blocked", "validated": False, "reason": "cooking_checkpoint_door_not_observed", "observation": observation}
        request = await api.request(
            "POST",
            f"/agents/{agent_id}/actions",
            json={"type": "interact", "target": action["target"], "option": action["option"]},
        )
        return await _resolve_interaction_action(
            agent_id,
            request,
            max(16, radius),
            before_observation=filtered,
            target=action["target"],
        )

    for index in range(max_steps):
        safety = latest.get("safety") or {}
        if safety.get("lowHealth") or safety.get("fleeing"):
            return _training_result(
                status="paused_low_health",
                skill="cooking",
                initial=baseline,
                final_observation=latest,
                steps=steps,
                message="training paused while the engine health watchdog escapes the threat",
            )
        current = _skill_state(latest, "cooking")
        if current["baseLevel"] >= target_level:
            return _training_result(
                status="complete",
                skill="cooking",
                initial=baseline,
                final_observation=latest,
                steps=steps,
            )

        carried = _carried_inventory(latest)
        raw_count = _carried_count_any(latest, COOKING_RAW_NAMES)
        free_slots = int(carried.get("freeSlots", 0))
        at_cooking_station = (
            latest["agent"]["position"]["level"] == COOKING_RESOURCE_STATION["level"]
            and max(
                abs(latest["agent"]["position"]["x"] - COOKING_RESOURCE_STATION_APPROACH["x"]),
                abs(latest["agent"]["position"]["z"] - COOKING_RESOURCE_STATION_APPROACH["z"]),
            ) <= 2
        )
        # Gather a batch before travelling. One slot is retained so a fish
        # catch cannot silently fail because the inventory became full.
        batch_target = min(5, max(1, free_slots - 1))
        # Once at the fire, finish the carried batch before returning to the
        # fishing spot. This avoids a needless gate/door round trip after
        # every single cooking action and keeps each live route checkpoint
        # resumable if the server stops the worker between actions.
        if raw_count == 0 or (raw_count < batch_target and not at_cooking_station):
            # Build a small batch when the carried inventory is saturated.
            # Every drop is still a live, validated mutation, and the allowlist
            # contains only products created by this cooking route.  Keeping up
            # to six free slots lets the next fishing leg gather five catches
            # while retaining one slot for a delayed server-side catch.
            if raw_count == 0 and free_slots < 6:
                discard = next(
                    (
                        item
                        for item in carried.get("items", [])
                        if (item.get("name") or "").casefold() in COOKING_DISCARD_NAMES
                    ),
                    None,
                )
                if discard is None and free_slots == 0:
                    return _training_result(
                        status="needs_context",
                        skill="cooking",
                        initial=baseline,
                        final_observation=latest,
                        steps=steps,
                        message="the live cooking inventory is full and contains no generated cooking output that can be safely discarded",
                    )
                if discard is not None:
                    dropped = await drop_item(agent_id, int(discard["slot"]), inventory=93)
                    latest = dropped.get("observation") or await _observe(agent_id, radius)
                    steps.append(
                        {
                            "index": index + 1,
                            "phase": "discard_cooking_output",
                            "status": dropped.get("status"),
                            "validated": dropped.get("validated", False),
                            "item": discard,
                        }
                    )
                    if dropped.get("status") != "complete" or not dropped.get("validated"):
                        return _training_result(
                            status="blocked",
                            skill="cooking",
                            initial=baseline,
                            final_observation=latest,
                            steps=steps,
                            message="the live cooking output could not be discarded with a validated inventory mutation",
                        )
                    continue
            if not any(
                (item.get("name") or "").casefold() == "small fishing net"
                for item in carried.get("items", [])
            ):
                return _training_result(
                    status="needs_context",
                    skill="cooking",
                    initial=baseline,
                    final_observation=latest,
                    steps=steps,
                    message="the live cooking route requires a Small fishing net in carried inventory",
                )
            position = latest["agent"]["position"]
            if position["level"] != COOKING_RESOURCE_SOURCE["level"] or max(
                abs(position["x"] - COOKING_RESOURCE_SOURCE["x"]),
                abs(position["z"] - COOKING_RESOURCE_SOURCE["z"]),
            ) > 2:
                if latest["agent"].get("targetOperation") is not None:
                    latest = (await stop_agent(agent_id)).get("observation") or await _observe(agent_id, radius)
                travel = await _travel_quest_step(
                    agent_id,
                    {"kind": "travel", **COOKING_RESOURCE_SOURCE, "run": True},
                    max(16, radius),
                )
                steps.append({"index": index + 1, "phase": "travel_to_fishing", "status": travel.get("status"), "validated": travel.get("validated", False)})
                if travel.get("status") != "complete" or not travel.get("validated"):
                    return _training_result(
                        status="blocked",
                        skill="cooking",
                        initial=baseline,
                        final_observation=travel.get("observation", latest),
                        steps=steps,
                        message=travel.get("reason", "the live route to the fishing source was not validated"),
                    )
                latest = travel.get("observation") or await _observe(agent_id, radius)
            fish: dict[str, Any] | None = None
            for fish_attempt in range(6):
                fish = await skill_step(agent_id, "fishing", radius=max(16, radius))
                latest = fish.get("observation") or await _observe(agent_id, radius)
                steps.append(
                    {
                        "index": index + 1,
                        "phase": "fish_raw_food",
                        "attempt": fish_attempt + 1,
                        "status": fish.get("status"),
                        "validated": fish.get("validated", False),
                        "selected": fish.get("selected"),
                    }
                )
                if fish.get("status") == "action_queued" and fish.get("validated"):
                    break
                # Fishing is probabilistic and its content script requeues
                # the spot after each cast. A missing catch is a retryable
                # live outcome, not permission to claim progress or abandon
                # the whole cooking run after one unlucky cycle. Keep the
                # bound finite so a genuinely unavailable spot remains an
                # explicit, resumable blocker.
                if (latest.get("agent") or {}).get("targetOperation") is not None:
                    latest = (await stop_agent(agent_id)).get("observation") or await _observe(agent_id, radius)
                await asyncio.sleep(step_delay_seconds)
            if fish is None or fish.get("status") != "action_queued" or not fish.get("validated"):
                return _training_result(
                    status="blocked",
                    skill="cooking",
                    initial=baseline,
                    final_observation=latest,
                    steps=steps,
                    message=(fish or {}).get("reason", "the live fishing postcondition was not validated after bounded retries"),
                )
            # Fishing requeues the spot action after a catch. Stop it before
            # walking to the cooking station so the next action is explicit.
            if (latest.get("agent") or {}).get("targetOperation") is not None:
                latest = (await stop_agent(agent_id)).get("observation") or await _observe(agent_id, radius)
            await asyncio.sleep(step_delay_seconds)
            continue

        position = latest["agent"]["position"]
        if position["level"] != COOKING_RESOURCE_STATION["level"] or max(
            abs(position["x"] - COOKING_RESOURCE_STATION_APPROACH["x"]),
            abs(position["z"] - COOKING_RESOURCE_STATION_APPROACH["z"]),
        ) > 2:
            gate_route = await _travel_quest_step(
                agent_id,
                {"kind": "travel", **COOKING_RESOURCE_GATE, "run": True},
                max(16, radius),
            )
            steps.append({"index": index + 1, "phase": "travel_to_cooking_gate", "status": gate_route.get("status"), "validated": gate_route.get("validated", False)})
            if gate_route.get("status") != "complete" or not gate_route.get("validated"):
                return _training_result(
                    status="blocked",
                    skill="cooking",
                    initial=baseline,
                    final_observation=gate_route.get("observation", latest),
                    steps=steps,
                    message=gate_route.get("reason", "the live route to the cooking gate was not validated"),
                )
            latest = gate_route.get("observation") or await _observe(agent_id, radius)
            door_route = await _travel_quest_step(
                agent_id,
                {"kind": "travel", **COOKING_RESOURCE_DOOR, "run": True},
                max(16, radius),
            )
            steps.append({"index": index + 1, "phase": "travel_to_cooking_door", "status": door_route.get("status"), "validated": door_route.get("validated", False)})
            if door_route.get("status") != "complete" or not door_route.get("validated"):
                return _training_result(
                    status="blocked",
                    skill="cooking",
                    initial=baseline,
                    final_observation=door_route.get("observation", latest),
                    steps=steps,
                    message=door_route.get("reason", "the live route to the cooking door was not validated"),
                )
            opened = await open_checkpoint_door()
            steps.append({"index": index + 1, "phase": "open_cooking_door", "status": opened.get("status"), "validated": opened.get("validated", False)})
            if opened.get("status") != "complete" or not opened.get("validated"):
                return _training_result(
                    status="blocked",
                    skill="cooking",
                    initial=baseline,
                    final_observation=opened.get("observation", latest),
                    steps=steps,
                    message=opened.get("reason", "the live cooking door interaction was not validated"),
                )
            approach_route = await _travel_quest_step(
                agent_id,
                {"kind": "travel", **COOKING_RESOURCE_STATION_APPROACH, "run": True},
                max(16, radius),
            )
            steps.append({"index": index + 1, "phase": "travel_to_cooking_fire", "status": approach_route.get("status"), "validated": approach_route.get("validated", False)})
            if approach_route.get("status") != "complete" or not approach_route.get("validated"):
                return _training_result(
                    status="blocked",
                    skill="cooking",
                    initial=baseline,
                    final_observation=approach_route.get("observation", latest),
                    steps=steps,
                    message=approach_route.get("reason", "the live route to the cooking fire was not validated"),
                )
            latest = approach_route.get("observation") or await _observe(agent_id, radius)
        cooked = await skill_step(agent_id, "cooking", radius=max(16, radius))
        latest = cooked.get("observation") or await _observe(agent_id, radius)
        steps.append({"index": index + 1, "phase": "cook_raw_food", "status": cooked.get("status"), "validated": cooked.get("validated", False), "selected": cooked.get("selected")})
        if cooked.get("status") != "action_queued" or not cooked.get("validated"):
            return _training_result(
                status="blocked",
                skill="cooking",
                initial=baseline,
                final_observation=latest,
                steps=steps,
                message=cooked.get("reason", "the live cooking postcondition was not validated"),
            )
        await asyncio.sleep(step_delay_seconds)

    return _training_result(
        status="step_limit",
        skill="cooking",
        initial=baseline,
        final_observation=latest,
        steps=steps,
        message="step limit reached before the target level",
    )


@mcp.tool
async def train_skill(
    agent_id: str,
    skill: str,
    target_level: int,
    radius: int = 16,
    max_steps: int = 100,
    step_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    """Run a resumable, safety-aware training loop until a level or stop condition.

    This is intentionally bounded. The loop observes after every action,
    stops when the target level is reached, pauses during a health escape, and
    reports when the content requires inventory/UI primitives not yet exposed
    by the TypeScript API.
    """

    skill = _normalized_skill(skill)
    if target_level < 1 or target_level > 99:
        raise ValueError("target_level must be between 1 and 99")
    if max_steps < 1 or max_steps > 1_000:
        raise ValueError("max_steps must be between 1 and 1000")
    if step_delay_seconds < 0.1 or step_delay_seconds > 30:
        raise ValueError("step_delay_seconds must be between 0.1 and 30")

    initial_observation = await api.request(
        "GET",
        f"/agents/{agent_id}/observation",
        params={"radius": radius},
    )
    initial = _skill_state(initial_observation, skill)
    spec = SKILL_PLAYBOOK[skill]
    if initial["baseLevel"] >= target_level:
        return _training_result(
            status="complete",
            skill=skill,
            initial=initial,
            final_observation=initial_observation,
            steps=[],
            message="target level already reached",
        )
    if spec["mode"] == "disabled":
        return {
            "status": "disabled",
            "skill": skill,
            "initial": initial,
            "target_level": target_level,
            "strategy": spec["strategy"],
            "missing": spec.get("missing", []),
            "safety": initial_observation.get("safety"),
        }

    if skill == "cooking":
        return await _train_cooking_with_live_resources(
            agent_id,
            target_level,
            radius,
            max_steps,
            step_delay_seconds,
            initial_observation,
            initial,
        )

    steps: list[dict[str, Any]] = []
    latest = initial_observation
    baseline = initial
    last_progress = initial
    stalled_steps = 0
    for index in range(max_steps):
        safety = latest.get("safety") or {}
        if safety.get("lowHealth") or safety.get("fleeing"):
            return _training_result(
                status="paused_low_health",
                skill=skill,
                initial=baseline,
                final_observation=latest,
                steps=steps,
                message="training paused while the engine health watchdog escapes the threat",
            )
        current = _skill_state(latest, skill)
        if current["baseLevel"] >= target_level:
            return _training_result(
                status="complete",
                skill=skill,
                initial=baseline,
                final_observation=latest,
                steps=steps,
            )
        try:
            step = await skill_step(agent_id, skill, radius=radius)
        except (LostCityApiError, ValueError) as error:
            return _training_result(
                status="blocked",
                skill=skill,
                initial=baseline,
                final_observation=latest,
                steps=steps,
                message=str(error),
            )
        steps.append(
            {
                "index": index + 1,
                "status": step.get("status"),
                "validated": step.get("validated", False),
                "selected": step.get("selected"),
                "target": step.get("target"),
                "safety": step.get("safety"),
            }
        )
        if step.get("status") in {"safety_pause", "no_action_found", "needs_context", "disabled", "blocked"}:
            return _training_result(
                status=step["status"],
                skill=skill,
                initial=baseline,
                final_observation=step.get("observation", latest),
                steps=steps,
                message=step.get("message"),
            )
        await asyncio.sleep(step_delay_seconds)
        latest = await api.request(
            "GET",
            f"/agents/{agent_id}/observation",
            params={"radius": radius},
        )
        updated = _skill_state(latest, skill)
        if updated["baseLevel"] == last_progress["baseLevel"] and updated["experience"] == last_progress["experience"]:
            stalled_steps += 1
        else:
            stalled_steps = 0
            last_progress = updated
        if stalled_steps >= 8:
            return _training_result(
                status="no_progress",
                skill=skill,
                initial=baseline,
                final_observation=latest,
                steps=steps,
                message="eight validated actions produced no skill XP or level change; inspect inventory, equipment, and content prerequisites",
            )

    return _training_result(
        status="step_limit",
        skill=skill,
        initial=baseline,
        final_observation=latest,
        steps=steps,
        message="step limit reached before the target level",
    )


@mcp.tool
async def move_agent(agent_id: str, x: int, z: int, run: bool = False, max_legs: int = 32) -> dict[str, Any]:
    """Navigate using one or more engine-planned local legs.

    ``max_legs`` lets open-map travel queue several collision-aware legs in
    one validated engine action. Dynamic blockers and health safety still cause
    the engine to stop/replan; it is not a teleport or client-side shortcut.
    """

    if max_legs < 1 or max_legs > 32:
        raise ValueError("max_legs must be between 1 and 32")

    return await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "move", "x": x, "z": z, "run": run, "maxLegs": max_legs},
    )


@mcp.tool
async def plan_route(agent_id: str, x: int, z: int, level: int = 0, max_legs: int = 32) -> dict[str, Any]:
    """Preview a collision-aware route without moving the agent.

    Long routes are planned as bounded local legs so agents can inspect
    waypoint count, tile cost, and whether the destination is currently
    reachable before committing to movement. The returned plan is live-engine
    authoritative; the MCP map window is a planning cache, not a movement
    authority.
    """

    if max_legs < 1 or max_legs > 32:
        raise ValueError("max_legs must be between 1 and 32")

    return await api.request(
        "POST",
        f"/agents/{agent_id}/route",
        json={"x": x, "z": z, "level": level, "maxLegs": max_legs},
    )


@mcp.tool
async def interact_agent(
    agent_id: str,
    target_kind: Literal["npc", "player", "loc", "obj"],
    option: int = 1,
    target_id: int | None = None,
    target_username: str | None = None,
    x: int | None = None,
    z: int | None = None,
    level: int | None = None,
    radius: int = 16,
) -> dict[str, Any]:
    """Walk to and execute an option, then validate an observable game effect."""

    if radius < 1 or radius > 64:
        raise ValueError("radius must be between 1 and 64")

    before_observation = await _observe(agent_id, radius=radius)

    target: dict[str, Any] = {"kind": target_kind}
    if target_kind == "player":
        if not target_username:
            raise ValueError("target_username is required for player targets")
        target["username"] = target_username
    else:
        if target_id is None:
            raise ValueError("target_id is required for this target kind")
        target["id"] = target_id
    if target_kind in {"loc", "obj"}:
        if x is None or z is None:
            raise ValueError("x and z are required for loc and obj targets")
        target["x"] = x
        target["z"] = z
        if level is not None:
            target["level"] = level

    action = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "interact", "target": target, "option": option},
    )
    resolved = await _resolve_interaction_action(
        agent_id,
        action,
        radius=radius,
        before_observation=before_observation,
    )
    if _is_toll_gate_target(before_observation, target):
        resolved = await _complete_toll_gate_dialogue(agent_id, resolved, radius)
    if target_kind in {"loc", "obj"}:
        _mark_map_cache_dirty(agent_id)
    return {
        "status": resolved["status"],
        "validated": resolved.get("validated", False),
        "validation": resolved.get("validation"),
        "target": target,
        "option": option,
        "action": action,
        "observation": resolved.get("observation"),
        "reason": resolved.get("reason"),
        "dialogue_options": resolved.get("dialogue_options")
        or ((resolved.get("observation") or {}).get("ui") or {}).get("resumeButtons"),
    }


@mcp.tool
async def pickup_object(
    agent_id: str,
    object_id: int | None = None,
    object_name: str | None = None,
    radius: int = 16,
) -> dict[str, Any]:
    """Pick up a nearby ground object using its live Take option."""

    if object_id is None and not object_name:
        raise ValueError("object_id or object_name is required")
    observation = await _observe(agent_id, radius)
    wanted_name = object_name.casefold() if object_name else None
    position = observation.get("agent", {}).get("position") or {}
    candidates: list[tuple[int, int, dict[str, Any], int]] = []
    for obj in observation.get("nearby", {}).get("objects", []):
        if object_id is not None and obj.get("id") != object_id:
            continue
        if wanted_name and (obj.get("name") or "").casefold() != wanted_name:
            continue
        match = _option_match(obj.get("options") or [], ["take", "pick-up", "pickup"])
        if match is None:
            continue
        option, _ = match
        target_position = obj.get("position") or {}
        distance = max(
            abs(target_position.get("x", 0) - position.get("x", 0)),
            abs(target_position.get("z", 0) - position.get("z", 0)),
        )
        candidates.append((distance, int(obj.get("id", 0)), obj, option))
    if not candidates:
        return {
            "status": "blocked",
            "validated": False,
            "reason": "no nearby ground object exposes a live Take option",
            "observation": observation,
        }
    distance, _, selected, option = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    target_position = selected["position"]
    target = {
        "kind": "obj",
        "id": selected["id"],
        "x": target_position["x"],
        "z": target_position["z"],
        "level": target_position.get("level", 0),
    }
    before_carried_count = _inventory_count_exact(observation, selected["name"])
    action = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "interact", "target": target, "option": option},
    )
    result = await _resolve_interaction_action(
        agent_id,
        action,
        radius,
        before_observation=observation,
        target=target,
    )
    latest = result.get("observation") or observation
    after_carried_count = _inventory_count_exact(latest, selected["name"])
    if result.get("status") == "complete" and after_carried_count <= before_carried_count:
        return {
            **result,
            "status": "blocked",
            "validated": False,
            "validation": "pickup_postcondition_not_observed",
            "reason": (
                "the live interaction did not increase the carried inventory "
                f"count for {selected['name']}"
            ),
        }
    return {
        **result,
        "operation": "pickup",
        "distance": distance,
        "before_carried_count": before_carried_count,
        "after_carried_count": after_carried_count,
        "selected": {"object": selected, "target": target, "option": option},
    }


@mcp.tool
async def use_item(
    agent_id: str,
    inventory: int,
    slot: int,
    use_inventory: int,
    use_slot: int,
) -> dict[str, Any]:
    """Use one live inventory item on another item through the game scripts."""

    return await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={
            "type": "use_item",
            "inventory": inventory,
            "slot": slot,
            "useInventory": use_inventory,
            "useSlot": use_slot,
        },
    )


async def _use_item_validated(
    agent_id: str,
    inventory: int,
    slot: int,
    use_inventory: int,
    use_slot: int,
    radius: int = 8,
) -> dict[str, Any]:
    """Use one item on another and require a live inventory mutation."""

    before = await _observe(agent_id, radius)
    action = await use_item(agent_id, inventory, slot, use_inventory, use_slot)
    return await _resolve_inventory_action(
        agent_id,
        action,
        before,
        radius=radius,
        validation="item_combination_changed_inventory_on_live_server",
    )


@mcp.tool
async def use_item_on(
    agent_id: str,
    inventory: int,
    slot: int,
    target_kind: Literal["npc", "player", "loc", "obj"],
    target_id: int | None = None,
    target_username: str | None = None,
    x: int | None = None,
    z: int | None = None,
    level: int | None = None,
) -> dict[str, Any]:
    """Use an inventory item on a live NPC, player, location, or object."""

    target: dict[str, Any] = {"kind": target_kind}
    if target_kind == "player":
        if not target_username:
            raise ValueError("target_username is required for player targets")
        target["username"] = target_username
    else:
        if target_id is None:
            raise ValueError("target_id is required for this target kind")
        target["id"] = target_id
    if target_kind in {"loc", "obj"}:
        if x is None or z is None:
            raise ValueError("x and z are required for loc and obj targets")
        target.update({"x": x, "z": z})
        if level is not None:
            target["level"] = level
    return await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "use_item_on", "inventory": inventory, "slot": slot, "target": target},
    )


@mcp.tool
async def use_item_on_validated(
    agent_id: str,
    inventory: int,
    slot: int,
    target_kind: Literal["npc", "player", "loc", "obj"],
    target_id: int | None = None,
    target_username: str | None = None,
    x: int | None = None,
    z: int | None = None,
    level: int | None = None,
    radius: int = 16,
) -> dict[str, Any]:
    """Use an item on a live target and validate the resulting state effect."""

    before = await _observe(agent_id, radius)
    target: dict[str, Any] = {"kind": target_kind}
    if target_kind == "player":
        if not target_username:
            raise ValueError("target_username is required for player targets")
        target["username"] = target_username
    else:
        if target_id is None:
            raise ValueError("target_id is required for this target kind")
        target["id"] = target_id
    if target_kind in {"loc", "obj"}:
        if x is None or z is None:
            raise ValueError("x and z are required for loc and obj targets")
        target.update({"x": x, "z": z})
        if level is not None:
            target["level"] = level
    action = await use_item_on(
        agent_id,
        inventory,
        slot,
        target_kind,
        target_id=target_id,
        target_username=target_username,
        x=x,
        z=z,
        level=level,
    )
    resolved = await _resolve_interaction_action(
        agent_id,
        action,
        radius,
        before_observation=before,
        target=target,
        allow_pending_observable_effect=True,
    )
    resolved_observation = resolved.get("observation") or {}
    operation_pending = (resolved_observation.get("agent") or {}).get("targetOperation") is not None
    if resolved.get("status") == "complete" and resolved.get("validated") and not operation_pending:
        stopped = await stop_agent(agent_id)
        resolved["observation"] = stopped.get("observation") or resolved.get("observation")
        resolved["stopped_after_observable_effect"] = True
    elif operation_pending:
        # Item-on actions such as shearing can expose an early target-side or
        # ground-object mutation while the content script is still running.
        # Stopping here cancels the operation before its inventory result is
        # delivered, so leave the agent active for the caller's postcondition
        # validation and subsequent action sequencing.
        resolved["operation_left_pending"] = True
    return {
        **resolved,
        "operation": "use_item_on",
        "target": target,
        "selected": {"inventory": inventory, "slot": slot},
    }


@mcp.tool
async def press_button(agent_id: str, component: int) -> dict[str, Any]:
    """Activate a validated interface button script."""

    return await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "button", "component": component},
    )


async def _press_button_validated(agent_id: str, component: int, radius: int = 8) -> dict[str, Any]:
    """Press an interface button and require an authoritative UI transition."""

    before = await _observe(agent_id, radius)
    before_ui = before.get("ui") or {}
    before_fingerprint = (
        before_ui.get("activeScriptExecution"),
        before_ui.get("activeScriptPc"),
        before_ui.get("activeScriptName"),
        tuple(button.get("component") for button in before_ui.get("resumeButtons", [])),
        before_ui.get("lastComponent"),
    )
    action = await press_button(agent_id, component)
    latest = action.get("observation") or before
    for _ in range(40):
        await asyncio.sleep(0.1)
        latest = await _observe(agent_id, radius)
        latest_ui = latest.get("ui") or {}
        latest_fingerprint = (
            latest_ui.get("activeScriptExecution"),
            latest_ui.get("activeScriptPc"),
            latest_ui.get("activeScriptName"),
            tuple(button.get("component") for button in latest_ui.get("resumeButtons", [])),
            latest_ui.get("lastComponent"),
        )
        if latest_ui.get("lastComponent") == component and latest_fingerprint != before_fingerprint:
            return {
                "status": "complete",
                "validated": True,
                "validation": "button_transition_observed_on_live_server",
                "action": action,
                "observation": latest,
            }
    return {
        "status": "blocked",
        "validated": False,
        "validation": "button_transition_not_observed",
        "reason": "the live UI did not report a state transition for the button",
        "action": action,
        "observation": latest,
    }


@mcp.tool
async def choose_dialogue_option(agent_id: str, option_text: str, radius: int = 8) -> dict[str, Any]:
    """Choose a live dialogue option by rendered text and validate the click."""

    observation = await _observe(agent_id, radius=radius)
    selected = _select_dialogue_button(observation, option_text)
    if selected is None:
        return {
            "status": "blocked",
            "validated": False,
            "reason": "dialogue_option_not_visible",
            "requested_text": option_text,
            "observation": observation,
        }

    action = await press_button(agent_id, selected["component"])
    before_ui = observation.get("ui") or {}
    before_fingerprint = (
        before_ui.get("activeScriptExecution"),
        before_ui.get("activeScriptPc"),
        before_ui.get("activeScriptName"),
        tuple(button.get("component") for button in before_ui.get("resumeButtons", [])),
    )
    after = action.get("observation") or observation
    validated = False
    # The HTTP action response contains the observation captured before the
    # engine tick executes the button. Poll the authoritative live snapshot so
    # callers do not mistake a stale choice list for the next dialogue page.
    for _ in range(40):
        await asyncio.sleep(0.1)
        after = await _observe(agent_id, radius=radius)
        after_ui = after.get("ui") or {}
        after_fingerprint = (
            after_ui.get("activeScriptExecution"),
            after_ui.get("activeScriptPc"),
            after_ui.get("activeScriptName"),
            tuple(button.get("component") for button in after_ui.get("resumeButtons", [])),
        )
        last_component = after_ui.get("lastComponent")
        if last_component == selected["component"] and after_fingerprint != before_fingerprint:
            validated = True
            break
    return {
        "status": "complete" if validated else "blocked",
        "validated": validated,
        "validation": "dialogue_transition_observed_on_live_server" if validated else "dialogue_transition_not_observed",
        "selected": selected,
        "action": action,
        "observation": after,
    }


@mcp.tool
async def resume_dialogue(agent_id: str) -> dict[str, Any]:
    """Resume a live dialogue waiting for the game's Continue action."""

    return await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "resume_dialogue"},
    )


@mcp.tool
async def resume_count_dialog(agent_id: str, value: int) -> dict[str, Any]:
    """Submit a value to a live count-input dialog and validate its transition."""

    if value < 0:
        raise ValueError("value must be zero or greater")
    before = await _observe(agent_id, radius=1)
    if (before.get("ui") or {}).get("activeScriptExecution") != 4:
        return {
            "status": "blocked",
            "validated": False,
            "validation": "no_count_dialog",
            "reason": "the engine is not waiting for a count value",
            "observation": before,
        }
    action = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "resume_count_dialog", "value": value},
    )
    before_signature = _interaction_effect_signature(before)
    latest = action.get("observation") or before
    for _ in range(20):
        await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius=1)
        if (
            (latest.get("ui") or {}).get("activeScriptExecution") != 4
            or _interaction_effect_signature(latest) != before_signature
        ):
            return {
                "status": "complete",
                "validated": True,
                "validation": "count_dialog_resumed_on_live_server",
                "value": value,
                "action": action,
                "observation": latest,
            }
    return {
        "status": "blocked",
        "validated": False,
        "validation": "count_dialog_resume_timeout",
        "reason": "the live count dialog did not transition before the validation timeout",
        "value": value,
        "action": action,
        "observation": latest,
    }


@mcp.tool
async def close_interface(agent_id: str, radius: int = 1) -> dict[str, Any]:
    """Close the current live interface and verify that its modal state cleared."""

    before = await _observe(agent_id, radius)
    before_ui = before.get("ui") or {}
    if all(before_ui.get(key, -1) == -1 for key in ("modalMain", "modalChat", "modalSide", "modalTutorial")):
        return {
            "status": "complete",
            "validated": True,
            "validation": "interface_already_closed",
            "observation": before,
        }
    action = await api.request("POST", f"/agents/{agent_id}/actions", json={"type": "close_interface"})
    latest = action.get("observation") or before
    for _ in range(20):
        await asyncio.sleep(0.25)
        latest = await _observe(agent_id, radius)
        ui = latest.get("ui") or {}
        if all(ui.get(key, -1) == -1 for key in ("modalMain", "modalChat", "modalSide", "modalTutorial")):
            return {
                "status": "complete",
                "validated": True,
                "validation": "interface_closed_on_live_server",
                "action": action,
                "observation": latest,
            }
    return {
        "status": "blocked",
        "validated": False,
        "validation": "interface_close_timeout",
        "reason": "the live interface modal did not clear before the validation timeout",
        "action": action,
        "observation": latest,
    }


@mcp.tool
async def set_run(agent_id: str, enabled: bool) -> dict[str, Any]:
    """Set run mode and verify the live player snapshot changed accordingly."""

    before = await _observe(agent_id, radius=1)
    if bool((before.get("agent") or {}).get("run")) == enabled:
        return {
            "status": "complete",
            "validated": True,
            "validation": "run_mode_already_at_requested_state",
            "enabled": enabled,
            "observation": before,
        }
    action = await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "set_run", "enabled": enabled},
    )
    latest = action.get("observation") or before
    for _ in range(20):
        await asyncio.sleep(0.1)
        latest = await _observe(agent_id, radius=1)
        if bool((latest.get("agent") or {}).get("run")) == enabled:
            return {
                "status": "complete",
                "validated": True,
                "validation": "run_mode_observed_on_live_server",
                "enabled": enabled,
                "action": action,
                "observation": latest,
            }
    return {
        "status": "blocked",
        "validated": False,
        "validation": "run_mode_transition_timeout",
        "reason": "the live player snapshot did not reflect the requested run mode",
        "enabled": enabled,
        "action": action,
        "observation": latest,
    }


@mcp.tool
async def click_side_tab(agent_id: str, tab: int) -> dict[str, Any]:
    """Activate a live side-tab through the engine's tutorial tab protocol."""

    if tab < 0 or tab > 13:
        raise ValueError("tab must be between 0 and 13")
    return await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "click_side_tab", "tab": tab},
    )


@mcp.tool
async def inventory_button(
    agent_id: str,
    component: int,
    inventory: int,
    slot: int,
    option: int,
) -> dict[str, Any]:
    """Activate an inventory interface option with live item validation."""

    if option < 1 or option > 5:
        raise ValueError("option must be between 1 and 5")
    return await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={
            "type": "inventory_button",
            "component": component,
            "inventory": inventory,
            "slot": slot,
            "option": option,
        },
    )


@mcp.tool
async def cast_spell(
    agent_id: str,
    target_kind: Literal["npc", "player", "loc", "obj"],
    component: int | None = None,
    component_name: str | None = None,
    target_id: int | None = None,
    target_username: str | None = None,
    x: int | None = None,
    z: int | None = None,
    level: int | None = None,
) -> dict[str, Any]:
    """Cast a spell component, optionally resolving it by live component name."""

    if component is None and not component_name:
        raise ValueError("component or component_name is required")

    target: dict[str, Any] = {"kind": target_kind}
    if target_kind == "player":
        if not target_username:
            raise ValueError("target_username is required for player targets")
        target["username"] = target_username
    else:
        if target_id is None:
            raise ValueError("target_id is required for this target kind")
        target["id"] = target_id
    if target_kind in {"loc", "obj"}:
        if x is None or z is None:
            raise ValueError("x and z are required for loc and obj targets")
        target.update({"x": x, "z": z})
        if level is not None:
            target["level"] = level
    payload: dict[str, Any] = {"type": "cast_spell", "target": target}
    if component is not None:
        payload["component"] = component
    if component_name:
        payload["componentName"] = component_name
    return await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json=payload,
    )


@mcp.tool
async def chat_agent(agent_id: str, message: str) -> dict[str, Any]:
    """Send a public chat message from an agent into the game world."""

    return await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "chat", "message": message},
    )


@mcp.tool
async def teleport_agent(agent_id: str, x: int, z: int, level: int = 0) -> dict[str, Any]:
    """Teleport an agent to an allocated game tile."""

    return await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "teleport", "x": x, "z": z, "level": level},
    )


@mcp.tool
async def stop_agent(agent_id: str) -> dict[str, Any]:
    """Cancel an agent's current movement and interaction."""

    return await api.request(
        "POST",
        f"/agents/{agent_id}/actions",
        json={"type": "stop"},
    )


@mcp.tool
async def release_agent(agent_id: str) -> dict[str, Any]:
    """Release a headless agent and remove its player from the world."""

    await _stop_keepalive(agent_id)
    _tutorial_ranged_attacks.discard(agent_id)
    _tutorial_magic_attacks.discard(agent_id)
    return await api.request("DELETE", f"/agents/{agent_id}")


if __name__ == "__main__":
    mcp.run(transport="stdio")
