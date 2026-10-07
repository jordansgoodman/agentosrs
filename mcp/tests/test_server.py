import asyncio

import httpx
import pytest

import server
from server import (
    LostCityApiClient,
    QUEST_TEMPLATES,
    SKILL_PLAYBOOK,
    _discover_inventory_skill_action,
    _discover_context_skill_action,
    _discover_skill_action,
    _discover_tutorial_action,
    _cached_map_route,
    _carried_item_tokens,
    _resolve_interaction_action,
    _settle_chat_dialogue,
    _travel_quest_step,
    _quest_steps,
    _option_match,
    _select_dialogue_button,
)


def test_cached_map_route_uses_live_window_geometry_only() -> None:
    window = {
        "center": {"level": 0, "x": 0, "z": 0},
        "bounds": {"minX": 0, "maxX": 2, "minZ": 0, "maxZ": 2},
        "rows": [
            {"z": z, "blocked": "000", "indoors": "000", "exits": "fff"}
            for z in range(3)
        ],
    }
    route = _cached_map_route(window, 2, 1)
    assert route["status"] == "complete"
    assert route["validated"] is True
    assert route["authoritative"] is False
    assert route["target"] == {"level": 0, "x": 2, "z": 1}
    assert route["tile_count"] == 3
    assert _cached_map_route(window, 3, 1)["status"] == "blocked"


def test_gobdip_graph_uses_exact_mail_inputs() -> None:
    steps = _quest_steps(QUEST_TEMPLATES["gobdip"]["steps"])
    assert [step["kind"] for step in steps] == [
        "quest_start",
        "quest_dialogue",
        "combine",
        "quest_dialogue",
        "combine",
        "quest_dialogue",
        "quest_dialogue",
        "quest_turn_in",
    ]
    assert steps[2]["item_exact"] is True
    assert steps[4]["use_item_exact"] is True
    observation = {
        "agent": {
            "inventories": [
                {
                    "id": 93,
                    "items": [
                        {"slot": 0, "name": "Orange goblin mail", "count": 1},
                        {"slot": 1, "name": "Goblin mail", "count": 1},
                    ],
                }
            ]
        }
    }
    exact = _carried_item_tokens(observation, ["goblin mail"], exact=True)
    assert exact is not None
    assert exact["item"]["name"] == "Goblin mail"


def test_api_client_uses_basic_auth_and_returns_json() -> None:
    async def run() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/health"
            assert request.headers["authorization"] == "Basic YWRtaW46cGFzc3dvcmQ="
            return httpx.Response(200, json={"ok": True})

        client = LostCityApiClient(
            base_url="http://testserver/api/v1",
            username="admin",
            password="password",
            transport=httpx.MockTransport(handler),
        )
        assert await client.request("GET", "/health") == {"ok": True}

    asyncio.run(run())


def test_api_client_surfaces_api_errors() -> None:
    async def run() -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "not_found"})

        client = LostCityApiClient(transport=httpx.MockTransport(handler))
        try:
            await client.request("GET", "/missing")
        except RuntimeError as error:
            assert "404" in str(error)
        else:
            raise AssertionError("expected the API error to be raised")

    asyncio.run(run())


def test_engine_action_contract_has_one_mcp_binding_per_primitive() -> None:
    assert set(server.MCP_ENGINE_ACTION_TOOLS) == server.ENGINE_ACTION_TYPES
    assert len(server.MCP_ENGINE_ACTION_TOOLS) == 16
    for tool_name in server.MCP_ENGINE_ACTION_TOOLS.values():
        assert callable(getattr(server, tool_name, None)), tool_name


def test_interaction_resolution_requires_observable_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    before = {
        "agent": {"inventories": [], "skills": {}},
        "tutorial": {"state": 1000},
        "ui": {"modalState": 0},
        "nearby": {"locations": [], "objects": []},
    }
    after = {
        **before,
        "agent": {"targetOperation": None, "inventories": [], "skills": {}},
    }

    async def fake_observe(_: str, *, radius: int) -> dict[str, object]:
        assert radius == 8
        return after

    monkeypatch.setattr(server, "_observe", fake_observe)

    async def run() -> None:
        result = await _resolve_interaction_action(
            "agent",
            {"observation": {"agent": {"targetOperation": {}}}},
            radius=8,
            max_wait_seconds=0.5,
            before_observation=before,
        )
        assert result["status"] == "blocked"
        assert result["validated"] is False
        assert result["validation"] == "interaction_resolved_without_observable_effect"

    asyncio.run(run())


def test_chat_dialogue_settlement_resumes_continue_but_stops_at_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    observations = [
        {"ui": {"activeScript": True, "modalChat": 968, "resumeButtons": []}},
        {
            "ui": {
                "activeScript": True,
                "modalChat": 2469,
                "resumeButtons": [{"component": 1, "text": "Yes, ok."}],
            }
        },
    ]
    resumed = 0

    async def fake_observe(_: str, *, radius: int) -> dict[str, object]:
        assert radius == 8
        return observations.pop(0)

    async def fake_resume(_: str) -> dict[str, object]:
        nonlocal resumed
        resumed += 1
        return {"status": "accepted"}

    monkeypatch.setattr(server, "_observe", fake_observe)
    monkeypatch.setattr(server, "resume_dialogue", fake_resume)

    result = asyncio.run(_settle_chat_dialogue("agent", radius=8))

    assert result["status"] == "blocked"
    assert result["validation"] == "dialogue_choice_requires_explicit_option"
    assert resumed == 1


def test_interact_forwards_requested_validation_radius(monkeypatch: pytest.MonkeyPatch) -> None:
    before = {"agent": {"position": {"level": 0, "x": 3200, "z": 3200}}}
    captured: dict[str, object] = {}

    async def fake_observe(_: str, *, radius: int) -> dict[str, object]:
        captured["observe_radius"] = radius
        return before

    async def fake_request(
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        captured["request"] = (method, path, json, params)
        return {"action": {"action": json}, "observation": before}

    async def fake_resolve(
        _: str,
        __: dict[str, object],
        *,
        radius: int,
        before_observation: dict[str, object],
    ) -> dict[str, object]:
        captured["resolve_radius"] = radius
        assert before_observation is before
        return {"status": "complete", "validated": True, "observation": before}

    monkeypatch.setattr(server, "_observe", fake_observe)
    monkeypatch.setattr(server.api, "request", fake_request)
    monkeypatch.setattr(server, "_resolve_interaction_action", fake_resolve)

    result = asyncio.run(
        server.interact_agent(
            "agent-1",
            "npc",
            target_id=42,
            option=1,
            radius=24,
        )
    )

    assert result["validated"] is True
    assert captured["observe_radius"] == 24
    assert captured["resolve_radius"] == 24
    assert captured["request"] == (
        "POST",
        "/agents/agent-1/actions",
        {"type": "interact", "target": {"kind": "npc", "id": 42}, "option": 1},
        None,
    )


def test_api_client_waits_for_pending_agent_activation() -> None:
    async def run() -> None:
        attempts = 0

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(409, json={"error": "agent_pending"})
            return httpx.Response(200, json={"active": True})

        client = LostCityApiClient(transport=httpx.MockTransport(handler))
        assert await client.request("GET", "/agents/test") == {"active": True}
        assert attempts == 2

    asyncio.run(run())


def test_mcp_agent_activation_waits_for_active_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    async def fake_request(method: str, path: str, *, json: dict[str, object] | None = None, params: dict[str, object] | None = None) -> dict[str, object]:
        nonlocal attempts
        assert method == "GET"
        assert path == "/agents/test/observation"
        attempts += 1
        return {"agent": {"status": "pending" if attempts == 1 else "active"}}

    monkeypatch.setattr(server.api, "request", fake_request)

    observation = asyncio.run(server._wait_for_agent_active("test", max_attempts=3))

    assert observation["agent"]["status"] == "active"
    assert attempts == 2


def test_skill_playbook_covers_the_server_skill_set() -> None:
    assert set(SKILL_PLAYBOOK) == {
        "attack",
        "defence",
        "strength",
        "hitpoints",
        "ranged",
        "prayer",
        "magic",
        "cooking",
        "woodcutting",
        "fletching",
        "fishing",
        "firemaking",
        "crafting",
        "smithing",
        "mining",
        "herblore",
        "agility",
        "thieving",
        "stat18",
        "stat19",
        "runecraft",
    }


def test_skill_discovery_uses_live_options_and_nearest_target() -> None:
    observation = {
        "agent": {"position": {"level": 0, "x": 100, "z": 100}},
        "nearby": {
            "npcs": [],
            "locations": [
                {
                    "id": 10,
                    "name": "Tree",
                    "options": ["Chop down", None, None, None, None],
                    "position": {"level": 0, "x": 105, "z": 100},
                },
                {
                    "id": 11,
                    "name": "Tree",
                    "options": ["Chop down", None, None, None, None],
                    "position": {"level": 0, "x": 101, "z": 100},
                },
                {
                    "id": 12,
                    "name": "Oak",
                    "options": ["Chop down", None, None, None, None],
                    "position": {"level": 0, "x": 100, "z": 101},
                },
            ],
            "objects": [],
        },
    }
    selected = _discover_skill_action(observation, SKILL_PLAYBOOK["woodcutting"])
    assert selected is not None
    assert selected["target"] == {"kind": "loc", "id": 11, "x": 101, "z": 100, "level": 0}
    assert selected["option"] == 1


def test_inventory_skill_discovery_uses_live_item_slots() -> None:
    observation = {
        "agent": {
            "inventories": [
                {
                    "id": 93,
                    "items": [
                        {"slot": 1, "id": 590, "name": "Tinderbox"},
                        {"slot": 6, "id": 1511, "name": "Logs"},
                    ],
                }
            ]
        }
    }

    selected = _discover_inventory_skill_action(observation, "firemaking")

    assert selected is not None
    assert selected["inventory"] == 93
    assert selected["slot"] == 1
    assert selected["use_inventory"] == 93
    assert selected["use_slot"] == 6


def test_context_crafting_discovery_prefers_live_wool_station_and_falls_back_to_shearing() -> None:
    base = {
        "agent": {
            "position": {"level": 1, "x": 3209, "z": 3214},
            "inventories": [{"id": 93, "items": []}],
        },
        "nearby": {
            "npcs": [],
            "locations": [{
                "id": 2644,
                "name": "Spinning wheel",
                "categoryName": "spinning_wheel",
                "position": {"level": 1, "x": 3209, "z": 3212},
                "options": [None, "Spin"],
            }],
            "objects": [],
        },
    }
    base["agent"]["inventories"][0]["items"] = [{"slot": 3, "id": 1737, "name": "Wool"}]
    spinning = _discover_context_skill_action(base, "crafting")
    assert spinning is not None
    assert spinning["recipe"] == "wool_to_ball_of_wool"
    assert spinning["target"] == {"kind": "loc", "id": 2644, "x": 3209, "z": 3212, "level": 1}

    base["agent"]["inventories"][0]["items"] = [{"slot": 4, "id": 1735, "name": "Shears"}]
    base["nearby"]["locations"] = []
    base["nearby"]["npcs"] = [{
        "id": 1780,
        "type": 43,
        "name": "Sheep",
        "position": {"level": 0, "x": 3193, "z": 3261},
        "options": [],
    }]
    shearing = _discover_context_skill_action(base, "crafting")
    assert shearing is not None
    assert shearing["recipe"] == "shear_sheep"
    assert shearing["target"] == {"kind": "npc", "id": 1780}


def test_context_cooking_discovery_selects_carried_raw_food_and_live_station() -> None:
    observation = {
        "agent": {
            "position": {"level": 0, "x": 3208, "z": 3215},
            "inventories": [
                {
                    "id": 93,
                    "items": [{"slot": 7, "id": 317, "name": "Raw shrimps", "count": 3}],
                },
                {
                    "id": 95,
                    "items": [{"slot": 2, "id": 317, "name": "Raw shrimps", "count": 20}],
                },
            ],
        },
        "nearby": {
            "npcs": [],
            "locations": [
                {
                    "id": 114,
                    "name": "Cooking range",
                    "categoryName": "cooking_oven",
                    "position": {"level": 0, "x": 3212, "z": 3215},
                    "options": [],
                }
            ],
            "objects": [],
        },
    }

    selected = _discover_context_skill_action(observation, "cooking")

    assert selected is not None
    assert selected["recipe"] == "cook_raw_food"
    assert selected["inventory"] == 93
    assert selected["slot"] == 7
    assert selected["target"] == {"kind": "loc", "id": 114, "x": 3212, "z": 3215, "level": 0}


def test_quest_item_steps_require_live_targets() -> None:
    steps = _quest_steps(
        [
            {"kind": "pickup", "object_name": "Egg"},
            {
                "kind": "item_on",
                "item_tokens": ["bucket"],
                "target_kind": "npc",
                "target_name": "Cow",
            },
        ]
    )
    assert steps[0]["kind"] == "pickup"
    assert steps[1]["target_name"] == "Cow"

    with pytest.raises(ValueError, match="item_tokens"):
        _quest_steps(
            [
                {
                    "kind": "item_on",
                    "target_kind": "npc",
                    "target_name": "Cow",
                }
            ]
        )


def test_fluffs_template_covers_stateful_dialogue_items_search_and_floor_transitions() -> None:
    steps = _quest_steps(QUEST_TEMPLATES["fluffs"]["steps"])
    kinds = [step["kind"] for step in steps]
    assert kinds.count("quest_dialogue") == 1
    assert kinds.count("combine") == 1
    assert kinds.count("quest_item_on") == 3
    assert kinds.count("quest_search") == 1
    assert kinds.count("floor_transition") == 4
    assert steps[4]["accept_movement"] is True
    assert steps[6]["expected_state"] == 3
    assert steps[9]["item_tokens"] == ["fluffs' kitten"]


def test_skill_discovery_respects_live_category_requirements() -> None:
    observation = {
        "agent": {
            "position": {"level": 0, "x": 100, "z": 100},
            "skills": {"fishing": {"baseLevel": 1}},
        },
        "nearby": {
            "npcs": [
                {
                    "id": 22,
                    "name": "Fishing spot",
                    "categoryName": "category_632",
                    "options": ["Net", None, "hidden", None, None],
                    "position": {"level": 0, "x": 101, "z": 100},
                }
            ],
            "locations": [],
            "objects": [],
        },
    }

    assert _discover_skill_action(observation, SKILL_PLAYBOOK["fishing"]) is None

    observation["agent"]["skills"]["fishing"]["baseLevel"] = 5
    selected = _discover_skill_action(observation, SKILL_PLAYBOOK["fishing"])
    assert selected is not None
    assert selected["target"] == {"kind": "npc", "id": 22}


def test_tutorial_discovery_prefers_live_resume_and_early_options() -> None:
    resume_observation = {"tutorial": {"state": 1}, "ui": {"activeScriptExecution": 3}, "nearby": {}}
    assert _discover_tutorial_action(resume_observation) == {"kind": "resume_dialogue"}

    initial_choice_observation = {
        "tutorial": {"state": 1},
        "ui": {
            "activeScriptExecution": 3,
            "resumeButtons": [{"component": 2461, "text": "option1", "name": "multi2:com_1"}],
        },
        "nearby": {},
    }
    assert _discover_tutorial_action(initial_choice_observation) == {"kind": "resume_dialogue"}

    choice_observation = {
        "tutorial": {"state": 500},
        "ui": {
            "activeScriptExecution": 3,
            "resumeButtons": [{"component": 2461, "text": "option1", "name": "multi2:com_1"}],
        },
        "nearby": {},
    }
    assert _discover_tutorial_action(choice_observation) == {
        "kind": "button",
        "component": 2461,
        "label": "option1",
    }

    chapel_door_observation = {
        "tutorial": {"state": 540},
        "ui": {"activeScriptExecution": None},
        "agent": {"position": {"level": 0, "x": 3129, "z": 3106}},
        "nearby": {
            "npcs": [{
                "id": 4267,
                "name": "Brother Brace",
                "options": ["Talk-to", None, None, None, None],
                "position": {"level": 0, "x": 3124, "z": 3106},
            }],
            "locations": [{
                "id": 1516,
                "name": "Large door",
                "options": ["Open", None, None, None, None],
                "position": {"level": 0, "x": 3129, "z": 3106},
            }],
        },
    }
    assert _discover_tutorial_action(chapel_door_observation) == {
        "kind": "interact",
        "target": {"kind": "loc", "id": 1516, "x": 3129, "z": 3106, "level": 0},
        "option": 1,
        "label": "Open",
        "entity": chapel_door_observation["nearby"]["locations"][0],
        "distance": 0,
    }

    observation = {
        "agent": {"position": {"level": 0, "x": 100, "z": 100}},
        "tutorial": {"state": 4},
        "ui": {"activeScriptExecution": None},
        "nearby": {
            "npcs": [],
            "locations": [
                {
                    "id": 3014,
                    "name": "Door",
                    "options": ["Open", None, None, None, None],
                    "position": {"level": 0, "x": 101, "z": 100},
                }
            ],
            "objects": [],
        },
    }
    selected = _discover_tutorial_action(observation)
    assert selected is not None
    assert selected["target"] == {"kind": "loc", "id": 3014, "x": 101, "z": 100, "level": 0}
    assert selected["option"] == 1


def test_tutorial_discovery_models_inventory_tabs_and_tutorial_fishing() -> None:
    base = {
        "agent": {
            "position": {"level": 0, "x": 3101, "z": 3097},
            "skills": {"fishing": {"baseLevel": 1}},
        },
        "ui": {"activeScriptExecution": None},
        "nearby": {"npcs": [], "locations": [], "objects": []},
    }

    inventory = {**base, "tutorial": {"state": 20}}
    assert _discover_tutorial_action(inventory) == {"kind": "side_tab", "tab": 3, "label": "inventory"}

    run_controls = {**base, "tutorial": {"state": 195}}
    assert _discover_tutorial_action(run_controls) == {
        "kind": "button",
        "component": 153,
        "label": "enable run",
    }

    quest_house = {**base, "tutorial": {"state": 200}}
    assert _discover_tutorial_action(quest_house) == {
        "kind": "interact",
        "target": {"kind": "loc", "id": 3019, "x": 3086, "z": 3126, "level": 0},
        "option": 1,
        "label": "Open quest guide door",
    }

    quest_journal = {**base, "tutorial": {"state": 230}}
    assert _discover_tutorial_action(quest_journal) == {"kind": "side_tab", "tab": 2, "label": "quest journal"}

    mining_ladder = {**base, "tutorial": {"state": 250}}
    assert _discover_tutorial_action(mining_ladder) == {
        "kind": "interact",
        "target": {"kind": "loc", "id": 3029, "x": 3088, "z": 3119, "level": 0},
        "option": 1,
        "label": "Climb down to mining",
    }

    mining = {**base, "tutorial": {"state": 260}}
    assert _discover_tutorial_action(mining) == {
        "kind": "interact",
        "target": {"kind": "npc", "id": 5278},
        "option": 1,
        "label": "Talk-to Mining Instructor",
    }

    smelting = {
        **base,
        "tutorial": {"state": 320},
        "agent": {
            **base["agent"],
            "inventories": [{"id": 93, "items": [{"slot": 4, "id": 436, "name": "Copper ore"}]}],
        },
    }
    smelting_action = _discover_tutorial_action(smelting)
    assert smelting_action is not None
    assert smelting_action["kind"] == "item_on"
    assert smelting_action["selected"]["target"] == {
        "kind": "loc",
        "id": 3044,
        "x": 3078,
        "z": 9495,
        "level": 0,
    }

    smithing = {
        **base,
        "tutorial": {"state": 340},
        "agent": {
            **base["agent"],
            "inventories": [{"id": 101, "items": [{"slot": 0, "id": 1205, "name": "Bronze dagger"}]}],
        },
    }
    assert _discover_tutorial_action(smithing) == {
        "kind": "inventory_button",
        "component": 1119,
        "inventory": 101,
        "slot": 0,
        "option": 1,
        "label": "Make bronze dagger",
    }

    fishing = {
        **base,
        "tutorial": {"state": 70},
        "nearby": {
            "npcs": [
                {
                    "id": 4258,
                    "name": "Fishing spot",
                    "categoryName": "category_453",
                    "options": ["Net", None, "hidden", None, None],
                    "position": {"level": 0, "x": 3099, "z": 3090},
                }
            ],
            "locations": [],
            "objects": [],
        },
    }
    selected = _discover_tutorial_action(fishing)
    assert selected is not None
    assert selected["kind"] == "skill"
    assert selected["skill"] == "fishing"
    assert selected["selected"]["target"] == {"kind": "npc", "id": 4258}

    combat_food = {
        **base,
        "tutorial": {"state": 440},
        "safety": {"lowHealth": True, "fleeing": True},
        "agent": {
            **base["agent"],
            "target": None,
            "targetOperation": None,
            "inventories": [{"id": 93, "items": [{"slot": 4, "name": "Shrimps", "options": ["Eat"]}]}],
        },
        "nearby": {
            "npcs": [
                {
                    "id": 5290,
                    "name": "Giant rat",
                    "options": [None, "Attack"],
                    "position": {"level": 0, "x": 100, "z": 100},
                }
            ],
            "locations": [],
            "objects": [],
        },
    }
    assert _discover_tutorial_action(combat_food) == {
        "kind": "item_op",
        "inventory": 93,
        "slot": 4,
        "option": 1,
        "label": "Eat",
        "item": {"slot": 4, "name": "Shrimps", "options": ["Eat"]},
    }

    combat_reengage = {
        **combat_food,
        "safety": {"lowHealth": False, "fleeing": False},
    }
    assert _discover_tutorial_action(combat_reengage) == {
        "kind": "interact",
        "target": {"kind": "npc", "id": 5290},
        "option": 2,
        "label": "Attack",
    }

    bank = {
        **base,
        "tutorial": {"state": 500},
        "agent": {**base["agent"], "position": {"level": 0, "x": 3120, "z": 3124}},
        "nearby": {
            "npcs": [],
            "locations": [
                {
                    "id": 3045,
                    "name": "Bank booth",
                    "options": ["Use", None, None, None, None],
                    "position": {"level": 0, "x": 3120, "z": 3124},
                }
            ],
            "objects": [],
        },
    }
    assert _discover_tutorial_action(bank) == {
        "kind": "interact",
        "target": {"kind": "loc", "id": 3045, "x": 3120, "z": 3124, "level": 0},
        "option": 1,
        "label": "Use",
        "entity": bank["nearby"]["locations"][0],
        "distance": 0,
    }

    magic = {
        **base,
        "tutorial": {"state": 640},
        "nearby": {
            "npcs": [
                {
                    "id": 3316,
                    "name": "Chicken",
                    "options": [None, "Attack", None, None, None],
                    "position": {"level": 0, "x": 3130, "z": 3080},
                }
            ],
            "locations": [],
            "objects": [],
        },
    }
    selected_magic = _discover_tutorial_action(magic)
    assert selected_magic is not None
    assert selected_magic["kind"] == "cast_spell"
    assert selected_magic["component_name"] == "magic:wind_strike"
    assert selected_magic["target"] == {"kind": "npc", "id": 3316}


def test_click_side_tab_forwards_the_engine_protocol_action(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_request(method: str, path: str, *, json: dict[str, object] | None = None, params: dict[str, object] | None = None) -> dict[str, object]:
        captured.update({"method": method, "path": path, "json": json})
        return {"accepted": True}

    monkeypatch.setattr(server.api, "request", fake_request)

    result = asyncio.run(server.click_side_tab("agent-1", 3))

    assert result == {"accepted": True}
    assert captured == {
        "method": "POST",
        "path": "/agents/agent-1/actions",
        "json": {"type": "click_side_tab", "tab": 3},
    }


def test_quest_steps_validate_and_preserve_the_declarative_graph() -> None:
    steps = _quest_steps(
        [
            {"kind": "travel", "x": 3200, "z": 3200, "level": 0},
            {"kind": "discover_interact", "target_kind": "npc", "option_tokens": ["talk"]},
            {"kind": "combat", "npc_name": "Goblin", "count": 2},
            {"kind": "train", "skill": "attack", "target_level": 2},
            {"kind": "dialogue", "text": "Yes, I'll help you."},
        ]
    )
    assert [step["kind"] for step in steps] == ["travel", "discover_interact", "combat", "train", "dialogue"]


def test_option_matching_prefers_exact_live_label() -> None:
    assert _option_match(["Climb", "Climb-up", "Climb-down"], ["climb-down", "climb"]) == (3, "Climb-down")


def test_sheep_template_contains_resumable_content_backed_route() -> None:
    steps = _quest_steps(server.QUEST_TEMPLATES["sheep"]["steps"])
    assert [step["kind"] for step in steps] == ["quest_start", "crafting_loop", "quest_turn_in"]
    assert steps[1]["source"]["via"] == [{"x": 3213, "z": 3261, "level": 0}]
    assert [checkpoint["target_name"] for checkpoint in steps[1]["staircase"]["access"]] == ["Large door", "Door"]


def test_travel_does_not_follow_a_route_search_frontier_away_from_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    observation = {
        "agent": {"position": {"level": 0, "x": 3217, "z": 3209}},
        "nearby": {"locations": [], "npcs": [], "objects": []},
        "safety": {"lowHealth": False, "fleeing": False},
    }

    async def fake_observe(_: str, *, radius: int) -> dict[str, object]:
        return observation

    async def fake_request(method: str, path: str, *, json: dict[str, object] | None = None, params: dict[str, object] | None = None) -> dict[str, object]:
        assert method == "POST"
        assert path == "/agents/agent-1/route"
        return {
            "reached": False,
            "estimatedTicks": 145,
            "finalPosition": {"level": 0, "x": 2797, "z": 3485},
            "waypoints": [{"level": 0, "x": 2797, "z": 3485}],
        }

    async def fake_map_window(*_: object, **__: object) -> dict[str, object]:
        return {"center": observation["agent"]["position"], "bounds": {}, "rows": [], "observation": observation}

    monkeypatch.setattr(server, "_observe", fake_observe)
    monkeypatch.setattr(server.api, "request", fake_request)
    monkeypatch.setattr(server, "map_window", fake_map_window)

    result = asyncio.run(
        _travel_quest_step(
            "agent-1",
            {"kind": "travel", "x": 3205, "z": 3209, "level": 0},
            radius=8,
        )
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "destination_unreachable_without_forward_progress"


def test_dialogue_discovery_uses_live_rendered_choice_text() -> None:
    observation = {
        "ui": {
            "resumeButtons": [
                {"component": 2482, "text": "What's wrong?", "activeText": ""},
                {"component": 2461, "text": "Yes, I'll help you.", "activeText": ""},
            ]
        }
    }

    selected = _select_dialogue_button(observation, "yes, i'll help you")

    assert selected is not None
    assert selected["component"] == 2461
    assert selected["text"] == "Yes, I'll help you."


def test_run_quest_does_not_claim_completion_for_unregistered_quest(monkeypatch: pytest.MonkeyPatch) -> None:
    observation = {
        "agent": {
            "position": {"level": 0, "x": 3200, "z": 3200},
            "skills": {},
        },
        "safety": {"lowHealth": False, "fleeing": False},
    }

    async def fake_request(
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if method == "GET" and path.endswith("/observation"):
            return observation
        if method == "POST" and path.endswith("/actions"):
            return {"action": {"action": json}, "observation": observation}
        if method == "GET" and path.endswith("/quests"):
            return {"quests": []}
        raise AssertionError((method, path, json, params))

    monkeypatch.setattr(server.api, "request", fake_request)

    result = asyncio.run(
        server.run_quest(
            "agent-1",
            quest_id="not_a_registered_quest",
            steps=[{"kind": "chat", "message": "test"}],
        )
    )

    assert result["status"] == "steps_complete_unverified"
    assert result["quest_verification"] is None


def test_item_op_agent_forwards_a_validated_inventory_action(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_request(method: str, path: str, *, json: dict[str, object] | None = None, params: dict[str, object] | None = None) -> dict[str, object]:
        captured.update({"method": method, "path": path, "json": json})
        return {"accepted": True}

    monkeypatch.setattr(server.api, "request", fake_request)

    result = asyncio.run(server.item_op_agent("agent-1", 93, 4, 2))

    assert result == {"accepted": True}
    assert captured == {
        "method": "POST",
        "path": "/agents/agent-1/actions",
        "json": {"type": "item_op", "inventory": 93, "slot": 4, "option": 2},
    }


def test_drop_item_validates_live_drop_option_and_inventory_change(monkeypatch: pytest.MonkeyPatch) -> None:
    observations = iter(
        [
            {
                "agent": {
                    "inventories": [
                        {
                            "id": 93,
                            "items": [{"slot": 4, "id": 1944, "name": "Egg", "options": [None, None, None, None, "Drop"]}],
                        }
                    ]
                }
            },
            {"agent": {"inventories": [{"id": 93, "items": []}]}},
            {"agent": {"inventories": [{"id": 93, "items": []}]}},
        ]
    )
    captured: dict[str, object] = {}

    async def fake_request(method: str, path: str, *, json: dict[str, object] | None = None, params: dict[str, object] | None = None) -> dict[str, object]:
        captured.update({"method": method, "path": path, "json": json})
        return {"observation": next(observations)}

    async def fake_observe(agent_id: str, radius: int = 1) -> dict[str, object]:
        return next(observations)

    monkeypatch.setattr(server, "_observe", fake_observe)
    monkeypatch.setattr(server.api, "request", fake_request)

    result = asyncio.run(server.drop_item("agent-1", 4))

    assert result["status"] == "complete"
    assert result["validated"] is True
    assert result["validation"] == "item_dropped_on_live_server"
    assert captured == {
        "method": "POST",
        "path": "/agents/agent-1/actions",
        "json": {"type": "item_op", "inventory": 93, "slot": 4, "option": 5},
    }


def test_resume_dialogue_forwards_the_engine_protocol_action(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_request(method: str, path: str, *, json: dict[str, object] | None = None, params: dict[str, object] | None = None) -> dict[str, object]:
        captured.update({"method": method, "path": path, "json": json})
        return {"accepted": True}

    monkeypatch.setattr(server.api, "request", fake_request)

    result = asyncio.run(server.resume_dialogue("agent-1"))

    assert result == {"accepted": True}
    assert captured == {
        "method": "POST",
        "path": "/agents/agent-1/actions",
        "json": {"type": "resume_dialogue"},
    }


def test_press_button_forwards_the_engine_protocol_action(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_request(method: str, path: str, *, json: dict[str, object] | None = None, params: dict[str, object] | None = None) -> dict[str, object]:
        captured.update({"method": method, "path": path, "json": json})
        return {"accepted": True}

    monkeypatch.setattr(server.api, "request", fake_request)

    result = asyncio.run(server.press_button("agent-1", 153))

    assert result == {"accepted": True}
    assert captured == {
        "method": "POST",
        "path": "/agents/agent-1/actions",
        "json": {"type": "button", "component": 153},
    }


def test_batch_actions_returns_resumable_trace_and_stops_on_unvalidated_step(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_execute(_: str, action: dict[str, object]) -> dict[str, object]:
        kind = str(action["type"])
        calls.append(kind)
        return {
            "status": "complete" if kind == "observe" else "blocked",
            "validated": kind == "observe",
        }

    monkeypatch.setattr(server, "_execute_batch_action", fake_execute)

    result = asyncio.run(
        server.batch_actions(
            "agent",
            [{"type": "observe"}, {"type": "skill_step"}, {"type": "observe"}],
            map_radius=None,
        )
    )

    assert calls == ["observe", "skill_step"]
    assert result["status"] == "blocked"
    assert result["validated"] is False
    assert result["next_action"] == 1
    assert result["executed"] == 2


def test_batch_actions_supports_bounded_repetition(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_execute(_: str, __: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"status": "complete", "validated": True}

    monkeypatch.setattr(server, "_execute_batch_action", fake_execute)

    result = asyncio.run(server.batch_actions("agent", [{"type": "observe", "repeat": 3}], map_radius=None))

    assert calls == 3
    assert result["status"] == "complete"
    assert result["validated"] is True
    assert result["executed"] == 3


def test_batch_actions_reports_partial_when_continue_mode_has_unvalidated_step(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_execute(_: str, action: dict[str, object]) -> dict[str, object]:
        kind = str(action["type"])
        calls.append(kind)
        return {"status": "complete" if kind == "observe" else "blocked", "validated": kind == "observe"}

    monkeypatch.setattr(server, "_execute_batch_action", fake_execute)

    result = asyncio.run(
        server.batch_actions(
            "agent",
            [{"type": "observe"}, {"type": "skill_step"}, {"type": "observe"}],
            stop_on_failure=False,
            map_radius=None,
        )
    )

    assert calls == ["observe", "skill_step", "observe"]
    assert result["status"] == "partial"
    assert result["validated"] is False
    assert result["validation"] == "batch_completed_with_unvalidated_actions"
    assert result["next_action"] == 3
    assert result["executed"] == 3


def test_map_window_reuses_geometry_and_refreshes_live_position(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    observation = {"agent": {"position": {"level": 0, "x": 100, "z": 100}}}
    window = {
        "tick": 10,
        "revision": "0:12:12",
        "center": {"level": 0, "x": 100, "z": 100},
        "radius": 8,
        "bounds": {"minX": 92, "maxX": 108, "minZ": 92, "maxZ": 108},
        "rows": [{"z": 92, "blocked": "0", "indoors": "0", "exits": "f"}],
        "observation": observation,
    }

    async def fake_request(method: str, path: str, *, json: dict[str, object] | None = None, params: dict[str, object] | None = None) -> dict[str, object]:
        del json
        calls.append((method, path))
        if path.endswith("/map-window"):
            assert params == {"radius": 8}
            return window
        assert path.endswith("/observation")
        return observation

    monkeypatch.setattr(server.api, "request", fake_request)
    server._map_cache.clear()

    first = asyncio.run(server.map_window("agent", radius=8))
    second = asyncio.run(server.map_window("agent", radius=8))

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert calls == [("GET", "/agents/agent/map-window"), ("GET", "/agents/agent/observation")]


def test_batch_actions_prefetches_the_default_map_window(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_map_window(agent_id: str, radius: int, refresh: bool) -> dict[str, object]:
        assert agent_id == "agent"
        assert radius == 32
        assert refresh is False
        return {"cache_hit": False, "radius": radius}

    async def fake_execute(_: str, __: dict[str, object]) -> dict[str, object]:
        return {"status": "complete", "validated": True}

    monkeypatch.setattr(server, "map_window", fake_map_window)
    monkeypatch.setattr(server, "_execute_batch_action", fake_execute)

    result = asyncio.run(server.batch_actions("agent", [{"type": "observe"}]))

    assert result["status"] == "complete"
    assert result["map_cache"] == {"cache_hit": False, "radius": 32}


def test_batch_move_attaches_cached_route_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_map_window(
        agent_id: str,
        radius: int,
        refresh: bool = False,
        target_x: int | None = None,
        target_z: int | None = None,
        target_level: int | None = None,
    ) -> dict[str, object]:
        assert agent_id == "agent"
        assert radius == 8
        assert refresh is False
        assert (target_x, target_z, target_level) == (5, 6, None)
        return {"cached_route": {"status": "complete", "authoritative": False}}

    async def fake_move_agent(_: str, **__: object) -> dict[str, object]:
        return {"accepted": True}

    async def fake_wait(_: str, __: int, ___: int, **____: object) -> dict[str, object]:
        return {"status": "complete", "validated": True, "observation": {"tick": 3}}

    monkeypatch.setattr(server, "map_window", fake_map_window)
    monkeypatch.setattr(server, "move_agent", fake_move_agent)
    monkeypatch.setattr(server, "_wait_for_batch_move", fake_wait)

    result = asyncio.run(
        server._execute_batch_action(
            "agent",
            {"type": "move", "x": 5, "z": 6, "_batch_map_radius": 8},
        )
    )

    assert result["cached_route"] == {"status": "complete", "authoritative": False}
    assert result["validated"] is True


def test_batch_actions_dispatches_crafting_loop_as_one_validated_action(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_crafting_loop(agent_id: str, step: dict[str, object], radius: int) -> dict[str, object]:
        assert agent_id == "agent"
        assert step == {"target_count": 5}
        assert radius == 16
        return {"status": "complete", "validated": True, "count": 5}

    monkeypatch.setattr(server, "_crafting_loop_step", fake_crafting_loop)

    result = asyncio.run(
        server._execute_batch_action(
            "agent",
            {"type": "crafting_loop", "target_count": 5, "radius": 16, "_batch_map_radius": 8},
        )
    )

    assert result == {"status": "complete", "validated": True, "count": 5}


def test_batch_actions_dispatches_combat_loop_as_one_validated_action(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_combat_loop(agent_id: str, step: dict[str, object], radius: int) -> dict[str, object]:
        assert agent_id == "agent"
        assert step == {"npc_name": "Goblin", "count": 2}
        assert radius == 16
        return {"status": "complete", "validated": True, "kills": 2}

    monkeypatch.setattr(server, "_combat_loop_step", fake_combat_loop)

    result = asyncio.run(
        server._execute_batch_action(
            "agent",
            {"type": "combat_loop", "npc_name": "Goblin", "count": 2, "radius": 16},
        )
    )

    assert result == {"status": "complete", "validated": True, "kills": 2}


def test_batch_actions_dispatches_interact_radius(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_interact(agent_id: str, **payload: object) -> dict[str, object]:
        assert agent_id == "agent"
        assert payload["radius"] == 24
        assert payload["target_kind"] == "npc"
        return {"status": "complete", "validated": True}

    monkeypatch.setattr(server, "interact_agent", fake_interact)

    result = asyncio.run(
        server._execute_batch_action(
            "agent",
            {
                "type": "interact",
                "target_kind": "npc",
                "target_id": 42,
                "option": 1,
                "radius": 24,
            },
        )
    )

    assert result == {"status": "complete", "validated": True}


def test_batch_actions_dispatches_recover_health(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_recover(agent_id: str, step: dict[str, object], radius: int) -> dict[str, object]:
        assert agent_id == "agent"
        assert step == {"target_ratio": 0.9}
        assert radius == 1
        return {"status": "complete", "validated": True, "health": 10}

    monkeypatch.setattr(server, "_recover_health_step", fake_recover)

    result = asyncio.run(
        server._execute_batch_action(
            "agent",
            {"type": "recover_health", "target_ratio": 0.9},
        )
    )

    assert result == {"status": "complete", "validated": True, "health": 10}


def test_combat_loop_food_support_does_not_consume_kill_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    low = {
        "safety": {"health": 4, "maxHealth": 10, "lowHealth": False, "fleeing": False},
        "agent": {"skills": {"attack": {"experience": 0}}},
    }
    healthy = {
        "safety": {"health": 10, "maxHealth": 10, "lowHealth": False, "fleeing": False},
        "agent": {"skills": {"attack": {"experience": 0}}},
    }
    defeated = {
        "safety": healthy["safety"],
        "agent": {"target": None, "skills": {"attack": {"experience": 40}}},
    }
    observations = iter([low, healthy, defeated])

    async def fake_observe(_: str, radius: int = 1) -> dict[str, object]:
        assert radius == 16
        return next(observations)

    async def fake_consume_food(_: str, minimum_health_ratio: float = 0.7) -> dict[str, object]:
        assert minimum_health_ratio == 0.8
        return {"status": "food_queued"}

    async def fake_combat(_: str, step: dict[str, object], radius: int, max_ticks: int) -> dict[str, object]:
        assert step["count"] == 1
        assert radius == 16
        assert max_ticks == 4
        return {"status": "complete", "kills": 1, "observation": defeated}

    monkeypatch.setattr(server, "_observe", fake_observe)
    monkeypatch.setattr(server, "consume_food", fake_consume_food)
    monkeypatch.setattr(server, "_combat_quest_step", fake_combat)

    result = asyncio.run(
        server._combat_loop_step(
            "agent",
            {"npc_name": "Goblin", "count": 1, "max_kills": 1, "max_iterations": 2, "max_ticks": 4, "food_threshold": 0.8},
            16,
        )
    )

    assert result["status"] == "complete"
    assert result["kills"] == 1
    assert result["foods_eaten"] == 1


def test_batch_actions_carries_latest_live_observation_with_static_map(monkeypatch: pytest.MonkeyPatch) -> None:
    observation = {"tick": 4, "agent": {"position": {"level": 0, "x": 5, "z": 6}}}

    async def fake_map_window(agent_id: str, radius: int, refresh: bool) -> dict[str, object]:
        assert agent_id == "agent"
        assert radius == 8
        assert refresh is False
        return {"cache_hit": False, "radius": radius, "rows": [{"z": 6}]}

    async def fake_execute(_: str, __: dict[str, object]) -> dict[str, object]:
        return {"status": "complete", "validated": True, "observation": observation}

    monkeypatch.setattr(server, "map_window", fake_map_window)
    monkeypatch.setattr(server, "_execute_batch_action", fake_execute)

    result = asyncio.run(server.batch_actions("agent", [{"type": "observe"}], map_radius=8))

    assert result["map_cache"]["rows"] == [{"z": 6}]
    assert result["map_cache"]["observation"] == observation
    assert result["map_cache"]["validation"] == "cached_static_geometry_with_latest_live_observation"
