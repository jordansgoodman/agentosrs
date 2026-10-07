"""Opt-in smoke test for a real Lost City world.

Run with ``LOSTCITY_LIVE_TEST=1 uv run pytest -q`` while the controlled
engine is running. The test creates one disposable headless session and
checks the API's accepted actions, authoritative quest state, and release
path. It never grants XP or mutates quest state.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx
import pytest

import server


pytestmark = pytest.mark.skipif(
    os.getenv("LOSTCITY_LIVE_TEST") != "1",
    reason="set LOSTCITY_LIVE_TEST=1 to run against a real controlled server",
)


def test_live_server_agent_smoke() -> None:
    async def run() -> None:
        base_url = os.getenv("LOSTCITY_API_URL", "http://127.0.0.1:80/api/v1").rstrip("/")
        username = f"live{uuid.uuid4().hex[:8]}"
        async with httpx.AsyncClient(base_url=base_url, auth=("admin", "password"), timeout=15) as client:
            health = await client.get("/health")
            health.raise_for_status()
            assert health.json()["ok"] is True

            created = await client.post("/agents", json={"username": username})
            created.raise_for_status()
            agent_id = created.json()["id"]
            try:
                observation = None
                for _ in range(12):
                    await asyncio.sleep(0.25)
                    response = await client.get(f"/agents/{agent_id}/observation", params={"radius": 16})
                    if response.status_code == 200 and response.json()["agent"]["status"] == "active":
                        observation = response.json()
                        break
                assert observation is not None
                assert observation["tutorial"]["stateVar"] == "tutorial"
                assert isinstance(observation["tutorial"]["state"], int)
                assert "activeScriptExecution" in observation["ui"]

                chat = await client.post(f"/agents/{agent_id}/actions", json={"type": "chat", "message": "live smoke"})
                chat.raise_for_status()
                assert chat.json()["action"]["action"]["type"] == "chat"

                run_toggle = await client.post(f"/agents/{agent_id}/actions", json={"type": "set_run", "enabled": True})
                run_toggle.raise_for_status()
                assert run_toggle.json()["observation"]["agent"]["run"] is True

                route = await client.post(
                    f"/agents/{agent_id}/route",
                    json={"x": observation["agent"]["position"]["x"], "z": observation["agent"]["position"]["z"], "level": 0},
                )
                route.raise_for_status()
                assert route.json()["reached"] is True

                quests = await client.get(f"/agents/{agent_id}/quests")
                quests.raise_for_status()
                assert quests.json()["quests"]
                assert all("category" in npc and "categoryName" in npc for npc in observation["nearby"]["npcs"])

                guarded_actions = [
                    {"type": "item_op", "inventory": 93, "slot": 0, "option": 1},
                    {"type": "use_item", "inventory": 93, "slot": 0, "useInventory": 93, "useSlot": 1},
                    {"type": "use_item_on", "inventory": 93, "slot": 0, "target": {"kind": "npc", "id": 1}},
                    {"type": "button", "component": 65535},
                    {"type": "resume_dialogue"},
                    {"type": "inventory_button", "component": 65535, "inventory": 93, "slot": 0, "option": 1},
                    {"type": "cast_spell", "component": 65535, "target": {"kind": "npc", "id": 1}},
                ]
                for payload in guarded_actions:
                    rejected = await client.post(f"/agents/{agent_id}/actions", json=payload)
                    assert rejected.status_code in {400, 404, 409}, payload
            finally:
                released = await client.delete(f"/agents/{agent_id}")
                released.raise_for_status()
                assert released.json()["released"] is True

    asyncio.run(run())


def test_live_server_all_engine_action_types_are_dispatchable() -> None:
    """Exercise every primitive action discriminator against a fresh server session.

    Some actions intentionally use invalid or unavailable targets because they
    require a particular live content state.  Those calls still prove that the
    engine recognized and dispatched the primitive instead of rejecting an
    unknown action type.  Actions with a safe universal effect are required to
    succeed and expose their authoritative observation.
    """

    async def run() -> None:
        base_url = os.getenv("LOSTCITY_API_URL", "http://127.0.0.1:80/api/v1").rstrip("/")
        username = f"ac{uuid.uuid4().hex[:8]}"
        created = await server.create_agent(username)
        agent_id = created["id"]
        try:
            observation = await server.observe_agent(agent_id, radius=1)
            position = observation["agent"]["position"]
            universal = [
                {"type": "move", "x": position["x"], "z": position["z"], "level": position["level"]},
                {"type": "chat", "message": "core action validation"},
                {"type": "teleport", "x": position["x"], "z": position["z"], "level": position["level"]},
                {"type": "close_interface"},
                {"type": "stop"},
                {"type": "set_run", "enabled": True},
            ]
            guarded = [
                {"type": "item_op", "inventory": 93, "slot": 0, "option": 1},
                {"type": "use_item", "inventory": 93, "slot": 0, "useInventory": 93, "useSlot": 1},
                {"type": "use_item_on", "inventory": 93, "slot": 0, "target": {"kind": "npc", "id": 1}},
                {"type": "button", "component": 65535},
                {"type": "resume_dialogue"},
                {"type": "resume_count_dialog", "value": 0},
                {"type": "click_side_tab", "tab": 0},
                {"type": "inventory_button", "component": 65535, "inventory": 93, "slot": 0, "option": 1},
                {"type": "cast_spell", "component": 65535, "target": {"kind": "npc", "id": 1}},
                {"type": "interact", "target": {"kind": "npc", "id": 1}, "option": 1},
            ]

            async with httpx.AsyncClient(
                base_url=base_url,
                auth=("admin", "password"),
                timeout=15,
            ) as client:
                for payload in universal:
                    response = await client.post(f"/agents/{agent_id}/actions", json=payload)
                    response.raise_for_status()
                    body = response.json()
                    assert body["action"]["action"]["type"] == payload["type"]
                    assert "observation" in body

                for payload in guarded:
                    response = await client.post(f"/agents/{agent_id}/actions", json=payload)
                    if payload["type"] == "interact" and response.status_code == 202:
                        body = response.json()
                        assert body["action"]["action"]["type"] == payload["type"]
                        assert "observation" in body
                        continue
                    assert response.status_code in {400, 404, 409}, payload
                    error = response.json().get("error")
                    assert error != "invalid_action", (payload, response.text)

            audit = await server.capability_audit(agent_id)
            contract = audit["engine_action_contract"]
            assert contract["engine_action_count"] == 16
            assert contract["engine_actions_missing_from_mcp"] == []
            assert contract["mcp_actions_without_engine_primitive"] == []
        finally:
            await server.release_agent(agent_id)

    asyncio.run(run())


def test_live_server_batch_actions_prefetch_and_validate_core_sequence() -> None:
    """Validate the batched control path and its live postconditions."""

    async def run() -> None:
        username = f"bt{uuid.uuid4().hex[:8]}"
        created = await server.create_agent(username)
        agent_id = created["id"]
        try:
            observation = await server.observe_agent(agent_id, radius=1)
            position = observation["agent"]["position"]
            result = await server.batch_actions(
                agent_id,
                [
                    {"type": "observe", "radius": 1},
                    {"type": "map_window", "radius": 8},
                    {"type": "move", "x": position["x"], "z": position["z"], "level": position["level"]},
                    {"type": "set_run", "enabled": True},
                    {"type": "stop"},
                ],
                map_radius=8,
            )
            assert result["status"] == "complete"
            assert result["validated"] is True
            assert len(result["results"]) == 5
            assert all(step["validated"] is True for step in result["results"])
            assert result["map_cache"]["radius"] == 8
            assert result["map_cache"]["cache_hit"] is True
            assert result["map_cache"]["validation"] == "cached_static_geometry_with_latest_live_observation"
        finally:
            await server.release_agent(agent_id)

    asyncio.run(run())


def test_live_server_mcp_bank_cycle() -> None:
    """Open, mutate, and close a real bank interface through the MCP.

    Bank mutation needs a player with a real item. Supply the already-online
    controlled player explicitly so this test never fabricates inventory state
    or changes a disposable blank tutorial account.
    """

    async def run() -> None:
        username = os.getenv("LOSTCITY_LIVE_BANK_USERNAME")
        if not username:
            pytest.skip("set LOSTCITY_LIVE_BANK_USERNAME to a real online player with an item")
        attached = await server.attach_agent(username)
        agent_id = attached["id"]
        original_position = attached["observation"]["agent"]["position"]
        try:
            await server.teleport_agent(agent_id, 3122, 3124, 0)
            opened = await server.open_bank(agent_id, radius=24)
            assert opened["status"] == "complete"
            assert opened["validated"] is True

            observation = opened["observation"]
            inventory = next(inv for inv in observation["agent"]["inventories"] if inv["id"] == 93)
            if not inventory["items"]:
                pytest.skip("the supplied live bank player has no inventory item to round-trip")
            item = inventory["items"][0]
            deposited = await server.deposit_item(agent_id, item["slot"], "x", quantity=1)
            assert deposited["status"] == "complete"
            assert deposited["validated"] is True

            after_deposit = deposited["observation"]
            bank = next(
                inv
                for inv in after_deposit["agent"]["inventories"]
                if any(listener["name"] == "bank_main:inv" for listener in inv["listeners"])
            )
            bank_item = next(bank_item for bank_item in bank["items"] if bank_item["id"] == item["id"])
            withdrawn = await server.withdraw_item(agent_id, bank_item["slot"], "x", quantity=1)
            assert withdrawn["status"] == "complete"
            assert withdrawn["validated"] is True

            closed = await server.close_bank(agent_id)
            assert closed["status"] == "complete"
            assert closed["validated"] is True
        finally:
            await server.teleport_agent(
                agent_id,
                original_position["x"],
                original_position["z"],
                original_position["level"],
            )
            # This test intentionally attaches to the user's controlled live
            # account; preserve that account and its engine session.

    asyncio.run(run())


def test_live_server_mcp_shop_buy_cycle() -> None:
    """Open the real Lumbridge shop and buy a bucket through its listener."""

    async def run() -> None:
        username = os.getenv("LOSTCITY_LIVE_SHOP_USERNAME") or os.getenv("LOSTCITY_LIVE_BANK_USERNAME")
        if not username:
            pytest.skip("set LOSTCITY_LIVE_SHOP_USERNAME to a real online player with shop coins")
        attached = await server.attach_agent(username)
        agent_id = attached["id"]
        original_position = attached["observation"]["agent"]["position"]
        try:
            await server.close_shop(agent_id)
            await server.teleport_agent(agent_id, 3122, 3124, 0)
            await server.close_bank(agent_id)
            observation = await server.observe_agent(agent_id, radius=1)
            inventory = next(current for current in observation["agent"]["inventories"] if current.get("id") == 93)
            if inventory.get("freeSlots", 0) == 0:
                cleanup = next(
                    item
                    for item in inventory.get("items", [])
                    if (item.get("name") or "").casefold() == "egg"
                )
                dropped = await server.drop_item(agent_id, cleanup["slot"])
                assert dropped["status"] == "complete"
                assert dropped["validated"] is True
                observation = dropped["observation"]
            carried_coins = any(
                item.get("name", "").casefold() == "coins"
                for current in observation["agent"]["inventories"]
                if current.get("id") == 93
                for item in current.get("items", [])
            )
            if not carried_coins:
                bank_opened = await server.open_bank(agent_id, radius=24)
                assert bank_opened["status"] == "complete"
                bank_observation = bank_opened["observation"]
                bank = next(
                    current
                    for current in bank_observation["agent"]["inventories"]
                    if any(listener.get("name") == "bank_main:inv" for listener in current.get("listeners", []))
                )
                coins = next(item for item in bank["items"] if item.get("name", "").casefold() == "coins")
                withdrawn_coins = await server.withdraw_item(agent_id, coins["slot"], "all")
                assert withdrawn_coins["status"] == "complete"
                assert withdrawn_coins["validated"] is True
            closed_bank = await server.close_bank(agent_id)
            assert closed_bank["status"] == "complete"
            await server.teleport_agent(agent_id, 3208, 3244, 0)
            opened = await server.open_shop(agent_id, radius=16)
            assert opened["status"] == "complete"
            assert opened["validated"] is True
            observation = opened["observation"]
            shop = next(
                current
                for current in observation["agent"]["inventories"]
                if any(listener.get("name") == "shop_template:inv" for listener in current.get("listeners", []))
            )
            bucket = next(item for item in shop["items"] if item.get("name", "").casefold() == "bucket")
            bought = await server.buy_item(agent_id, bucket["slot"], 1)
            assert bought["status"] == "complete"
            assert bought["validated"] is True
            assert any(
                item.get("name", "").casefold() == "bucket"
                for current in bought["observation"]["agent"]["inventories"]
                if current.get("id") == 93
                for item in current.get("items", [])
            )
            closed = await server.close_shop(agent_id)
            assert closed["status"] == "complete"
            assert closed["validated"] is True
        finally:
            await server.teleport_agent(
                agent_id,
                original_position["x"],
                original_position["z"],
                original_position["level"],
            )
            # Preserve the user's controlled live account and engine session.

    asyncio.run(run())


def test_live_server_mcp_pickup_object() -> None:
    """Pick up a live egg object and validate its inventory effect."""

    async def run() -> None:
        username = os.getenv("LOSTCITY_LIVE_PICKUP_USERNAME") or os.getenv("LOSTCITY_LIVE_BANK_USERNAME")
        if not username:
            pytest.skip("set LOSTCITY_LIVE_PICKUP_USERNAME to a real online player")
        attached = await server.attach_agent(username)
        agent_id = attached["id"]
        original_position = attached["observation"]["agent"]["position"]
        try:
            await server.teleport_agent(agent_id, 3185, 3279, 0)
            picked_up = await server.pickup_object(agent_id, object_name="Egg", radius=8)
            assert picked_up["status"] == "complete"
            assert picked_up["validated"] is True
            assert any(
                item.get("name", "").casefold() == "egg"
                for current in picked_up["observation"]["agent"]["inventories"]
                if current.get("id") == 93
                for item in current.get("items", [])
            )
        finally:
            await server.teleport_agent(
                agent_id,
                original_position["x"],
                original_position["z"],
                original_position["level"],
            )
            # Preserve the user's controlled live account and engine session.

    asyncio.run(run())


def test_live_server_mcp_use_item_on_target() -> None:
    """Use a real bucket on a live cow and validate the milk mutation."""

    async def run() -> None:
        username = os.getenv("LOSTCITY_LIVE_ITEM_TARGET_USERNAME") or os.getenv("LOSTCITY_LIVE_BANK_USERNAME")
        if not username:
            pytest.skip("set LOSTCITY_LIVE_ITEM_TARGET_USERNAME to a real online player")
        attached = await server.attach_agent(username)
        agent_id = attached["id"]
        original_position = attached["observation"]["agent"]["position"]
        try:
            await server.teleport_agent(agent_id, 3253, 3270, 0)
            observation = await server.observe_agent(agent_id, radius=16)
            cow = next(npc for npc in observation["nearby"]["npcs"] if (npc.get("name") or "").casefold() == "cow")
            inventory = next(current for current in observation["agent"]["inventories"] if current.get("id") == 93)
            bucket = next(item for item in inventory["items"] if (item.get("name") or "").casefold() == "bucket")
            result = await server.use_item_on_validated(
                agent_id,
                inventory["id"],
                bucket["slot"],
                "npc",
                target_id=cow["id"],
                radius=16,
            )
            assert result["status"] == "complete"
            assert result["validated"] is True
            assert any(
                item.get("name", "").casefold() == "bucket of milk"
                for current in result["observation"]["agent"]["inventories"]
                if current.get("id") == 93
                for item in current.get("items", [])
            )
        finally:
            await server.teleport_agent(
                agent_id,
                original_position["x"],
                original_position["z"],
                original_position["level"],
            )
            # Preserve the user's controlled live account and engine session.

    asyncio.run(run())


def test_live_server_mcp_crafting_context_cycle() -> None:
    """Discover and validate the live shearing -> spinning crafting chain."""

    async def run() -> None:
        username = os.getenv("LOSTCITY_LIVE_CRAFTING_USERNAME") or os.getenv("LOSTCITY_LIVE_BANK_USERNAME")
        if not username:
            pytest.skip("set LOSTCITY_LIVE_CRAFTING_USERNAME to a real online player with shop coins")
        attached = await server.attach_agent(username)
        agent_id = attached["id"]
        original_position = attached["observation"]["agent"]["position"]
        try:
            await server.close_shop(agent_id)
            await server.close_bank(agent_id)
            observation = await server.observe_agent(agent_id, radius=1)
            has_shears = any(
                (item.get("name") or "").casefold() == "shears"
                for current in observation["agent"]["inventories"]
                if current.get("id") == 93
                for item in current.get("items", [])
            )
            if not has_shears:
                await server.teleport_agent(agent_id, 3208, 3244, 0)
                opened = await server.open_shop(agent_id, radius=16)
                assert opened["status"] == "complete"
                shop = next(
                    current
                    for current in opened["observation"]["agent"]["inventories"]
                    if any(listener.get("name") == "shop_template:inv" for listener in current.get("listeners", []))
                )
                shears = next(
                    item for item in shop["items"] if (item.get("name") or "").casefold() == "shears"
                )
                bought = await server.buy_item(agent_id, shears["slot"], 1)
                assert bought["status"] == "complete"
                assert bought["validated"] is True
                closed = await server.close_shop(agent_id)
                assert closed["status"] == "complete"

            # Use the open tile beside Lumbridge's sheep pen. The farm gate
            # and nearby house make the more generic 3190,3275 staging tile
            # unable to establish line of approach to the moving sheep.
            await server.teleport_agent(agent_id, 3195, 3260, 0)
            current = await server.observe_agent(agent_id, radius=1)
            inventory = next(current for current in current["agent"]["inventories"] if current.get("id") == 93)
            if inventory.get("freeSlots", 0) == 0:
                cleanup = next(
                    item
                    for item in inventory.get("items", [])
                    if (item.get("name") or "").casefold() == "egg"
                )
                dropped = await server.drop_item(agent_id, cleanup["slot"])
                assert dropped["status"] == "complete"
                assert dropped["validated"] is True
            sheared = await server.skill_step(agent_id, "crafting", radius=24)
            assert sheared["status"] == "action_queued"
            assert sheared["validated"] is True
            assert sheared["selected"]["recipe"] == "shear_sheep"
            assert any(
                (item.get("name") or "").casefold() == "wool"
                for current in sheared["observation"]["agent"]["inventories"]
                if current.get("id") == 93
                for item in current.get("items", [])
            )

            await server.teleport_agent(agent_id, 3209, 3212, 1)
            spun = await server.skill_step(agent_id, "crafting", radius=16)
            assert spun["status"] == "action_queued"
            assert spun["validated"] is True
            assert spun["selected"]["recipe"] == "wool_to_ball_of_wool"
            assert any(
                (item.get("name") or "").casefold() == "ball of wool"
                for current in spun["observation"]["agent"]["inventories"]
                if current.get("id") == 93
                for item in current.get("items", [])
            )
            assert spun["current"]["experience"] > sheared["current"]["experience"]
        finally:
            await server.teleport_agent(
                agent_id,
                original_position["x"],
                original_position["z"],
                original_position["level"],
            )
            # Preserve the user's controlled live account and engine session.

    asyncio.run(run())


def test_live_server_mcp_quest_item_step() -> None:
    """Execute the new validated pickup step through the quest runner."""

    async def run() -> None:
        username = os.getenv("LOSTCITY_LIVE_QUEST_STEP_USERNAME") or os.getenv("LOSTCITY_LIVE_BANK_USERNAME")
        if not username:
            pytest.skip("set LOSTCITY_LIVE_QUEST_STEP_USERNAME to a real online player")
        attached = await server.attach_agent(username)
        agent_id = attached["id"]
        original_position = attached["observation"]["agent"]["position"]
        try:
            await server.teleport_agent(agent_id, 3185, 3279, 0)
            result = await server.run_quest(
                agent_id,
                quest_id="live_item_route",
                steps=[{"kind": "pickup", "object_name": "Egg"}],
                radius=8,
            )
            assert result["status"] == "steps_complete_unverified"
            assert result["completed_steps"] == 1
            assert result["progress"][0]["validated"] is True
        finally:
            await server.teleport_agent(
                agent_id,
                original_position["x"],
                original_position["z"],
                original_position["level"],
            )
            # Preserve the user's controlled live account and engine session.

    asyncio.run(run())


def test_live_server_mcp_floor_transition_cycle() -> None:
    """Validate both staircase directions through the real engine state."""

    async def run() -> None:
        username = os.getenv("LOSTCITY_LIVE_FLOOR_USERNAME") or os.getenv("LOSTCITY_LIVE_BANK_USERNAME")
        if not username:
            pytest.skip("set LOSTCITY_LIVE_FLOOR_USERNAME to a real online player")
        attached = await server.attach_agent(username)
        agent_id = attached["id"]
        original_position = attached["observation"]["agent"]["position"]
        staircase = {"x": 3204, "z": 3207, "approach_x": 3205, "approach_z": 3209}
        try:
            await server.teleport_agent(agent_id, staircase["approach_x"], staircase["approach_z"], 0)
            result = await server.run_quest(
                agent_id,
                quest_id="live_floor_route",
                steps=[
                    {
                        "kind": "floor_transition",
                        **staircase,
                        "from_level": 0,
                        "to_level": 1,
                        "option_tokens": ["climb-up", "climb"],
                    },
                    {
                        "kind": "floor_transition",
                        **staircase,
                        "from_level": 1,
                        "to_level": 0,
                        "option_tokens": ["climb-down", "climb"],
                    },
                ],
                radius=8,
            )
            assert result["status"] == "steps_complete_unverified"
            assert [step["validated"] for step in result["progress"]] == [True, True]
            assert result["observation"]["agent"]["position"]["level"] == 0
        finally:
            await server.teleport_agent(
                agent_id,
                original_position["x"],
                original_position["z"],
                original_position["level"],
            )

    asyncio.run(run())


def test_live_server_mcp_equipment_cycle() -> None:
    """Equip and remove a real item through the MCP against the live engine."""

    async def run() -> None:
        username = os.getenv("LOSTCITY_LIVE_EQUIPMENT_USERNAME") or os.getenv("LOSTCITY_LIVE_BANK_USERNAME")
        if not username:
            pytest.skip("set LOSTCITY_LIVE_EQUIPMENT_USERNAME to a real online player with wearable gear")
        attached = await server.attach_agent(username)
        agent_id = attached["id"]
        original_position = attached["observation"]["agent"]["position"]
        equipped_slot: int | None = None
        try:
            await server.close_bank(agent_id)
            observation = await server.observe_agent(agent_id, radius=1)
            inventory = next(
                current for current in observation["agent"]["inventories"] if current.get("id") == 93
            )
            item = next(
                (
                    item
                    for item in inventory.get("items", [])
                    if any(
                        isinstance(option, str) and option.casefold() in {"wield", "wear", "equip"}
                        for option in item.get("options", [])
                    )
                ),
                None,
            )
            if item is None:
                pytest.skip("the supplied live equipment player has no wearable inventory item")

            equipped = await server.equip_item(agent_id, item["slot"])
            assert equipped["status"] == "complete"
            assert equipped["validated"] is True
            worn = next(current for current in equipped["observation"]["agent"]["inventories"] if current.get("id") == 94)
            worn_item = next(
                (worn_item for worn_item in worn.get("items", []) if worn_item.get("id") == item.get("id")),
                None,
            )
            assert worn_item is not None
            equipped_slot = worn_item["slot"]

            removed = await server.unequip_item(agent_id, equipped_slot)
            assert removed["status"] == "complete"
            assert removed["validated"] is True
        finally:
            if equipped_slot is not None:
                try:
                    latest = await server.observe_agent(agent_id, radius=1)
                    worn = next(
                        (current for current in latest["agent"]["inventories"] if current.get("id") == 94),
                        None,
                    )
                    if worn and any(item.get("slot") == equipped_slot for item in worn.get("items", [])):
                        await server.unequip_item(agent_id, equipped_slot)
                except Exception:
                    pass
            await server.teleport_agent(
                agent_id,
                original_position["x"],
                original_position["z"],
                original_position["level"],
            )
            # Preserve the user's controlled live account and engine session.

    asyncio.run(run())


def test_live_server_mcp_production_product() -> None:
    """Select a tutorial smithing product through a live production listener."""

    async def run() -> None:
        username = f"prod{uuid.uuid4().hex[:8]}"
        created = await server.create_agent(username)
        agent_id = created["id"]
        try:
            for _ in range(180):
                observation = await server.observe_agent(agent_id, radius=32)
                state = observation["tutorial"]["state"]
                if state == 340:
                    production = next(
                        (
                            current
                            for current in observation["agent"]["inventories"]
                            if current.get("id") == 101 and current.get("items")
                        ),
                        None,
                    )
                    if production is not None:
                        product = production["items"][0]
                        result = await server.select_production_product(agent_id, production["id"], product["slot"])
                        assert result["status"] == "complete"
                        assert result["validated"] is True
                        assert result["validation"] == "production_action_changed_live_state"
                        return
                step = await server.tutorial_step(agent_id, radius=32)
                assert step.get("status") not in {"unsupported", "unsupported_step"}, step
                await asyncio.sleep(0.1)
            raise AssertionError("tutorial did not expose the live smithing production inventory")
        finally:
            released = await server.release_agent(agent_id)
            assert released["released"] is True

    asyncio.run(run())


def test_live_server_mcp_quest_start_is_authoritatively_verified() -> None:
    """Start Sheep Shearer through the real Fred dialogue without claiming completion."""

    async def run() -> None:
        username = f"qs{uuid.uuid4().hex[:8]}"
        created = await server.create_agent(username, x=3192, z=3274, level=0)
        agent_id = created["id"]
        try:
            result = await server.run_quest(
                agent_id,
                quest_id="sheep",
                steps=[
                    {
                        "kind": "quest_start",
                        "quest_id": "sheep",
                        "target_kind": "npc",
                        "target_name": "Fred the Farmer",
                        "option_tokens": ["talk"],
                        "dialogue_options": ["I'm looking for a quest.", "Yes okay. I can do that."],
                    }
                ],
                radius=8,
            )
            assert result["status"] == "steps_complete_unverified"
            assert result["quest_verification"]["status"] == "in_progress"
            assert result["verification_note"] == "all declared steps resolved, but the engine quest state is not complete"
        finally:
            released = await server.release_agent(agent_id)
            assert released["released"] is True

    asyncio.run(run())


def test_live_server_mcp_progression_worker_reports_real_blocker() -> None:
    """Run the worker control plane against the preserved live account."""

    async def run() -> None:
        username = os.getenv("LOSTCITY_LIVE_WORKER_USERNAME") or os.getenv("LOSTCITY_LIVE_BANK_USERNAME")
        if not username:
            pytest.skip("set LOSTCITY_LIVE_WORKER_USERNAME to a real online player")
        attached = await server.attach_agent(username)
        agent_id = attached["id"]
        try:
            started = await server.start_progression_worker(agent_id, interval_seconds=0.25, radius=16)
            assert started["status"] == "started"
            for _ in range(20):
                status = await server.progression_worker_status(agent_id)
                if status.get("status") == "blocked":
                    break
                await asyncio.sleep(0.25)
            status = await server.progression_worker_status(agent_id)
            assert status["status"] == "blocked"
            assert status["last_result"]["validated"] is False
            assert status["last_result"]["reason"] in {
                "no quest graph is registered for the next guide quest",
                "the next guide quest requires a disabled skill",
                "no live training action is available for the next guide skill requirement",
            }
        finally:
            await server.stop_progression_worker(agent_id)

    asyncio.run(run())


def test_live_server_mcp_tutorial_combat_branch() -> None:
    """Drive the MCP tutorial controller through the real combat lesson."""

    async def run() -> None:
        username = f"cb{uuid.uuid4().hex[:8]}"
        created = await server.create_agent(username)
        agent_id = created["id"]
        try:
            states: list[int] = []
            trace: list[dict[str, object]] = []
            for _ in range(180):
                result = await server.tutorial_step(agent_id, radius=32)
                tutorial = result.get("tutorial") or result.get("tutorial_before") or {}
                state = int(tutorial.get("state", -1))
                states.append(state)
                trace.append({"state": state, "status": result.get("status"), "selected": result.get("selected"), "validation": result.get("validation")})
                if state >= 470:
                    assert {360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460}.issubset(states)
                    return
                assert result.get("status") != "unsupported_step", {
                    "state": state,
                    "states": states[-12:],
                    "result": result,
                }
                await asyncio.sleep(0.1)
            raise AssertionError(f"combat tutorial did not reach ranged lesson: {trace[-12:]}")
        finally:
            released = await server.release_agent(agent_id)
            assert released["released"] is True

    asyncio.run(run())


def test_live_server_mcp_tutorial_completion() -> None:
    """Drive a fresh agent through the supported tutorial to the mainland."""

    async def run() -> None:
        username = f"tc{uuid.uuid4().hex[:8]}"
        created = await server.create_agent(username)
        agent_id = created["id"]
        try:
            states: list[int] = []
            trace: list[dict[str, object]] = []
            for _ in range(320):
                result = await server.tutorial_step(agent_id, radius=32)
                tutorial = result.get("tutorial") or result.get("tutorial_before") or {}
                state = int(tutorial.get("state", -1))
                states.append(state)
                trace.append({"state": state, "status": result.get("status"), "selected": result.get("selected"), "validation": result.get("validation")})
                if state >= 1000:
                    required = {500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 1000}
                    assert required.issubset(states), {"states": states, "trace": trace[-12:]}
                    spell_actions = [
                        entry
                        for entry in trace
                        if (entry.get("selected") or {}).get("kind") == "cast_spell"
                    ]
                    assert spell_actions
                    assert spell_actions[-1]["selected"]["component_name"] == "magic:wind_strike"
                    return
                assert result.get("status") not in {"unsupported", "unsupported_step"}, {
                    "state": state,
                    "states": states[-12:],
                    "result": result,
                }
                await asyncio.sleep(0.1)
            raise AssertionError(f"tutorial did not reach the mainland: {trace[-12:]}")
        finally:
            released = await server.release_agent(agent_id)
            assert released["released"] is True

    asyncio.run(run())
