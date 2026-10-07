# Lost City Agent MCP

This folder contains the Python FastMCP bridge for the TypeScript Agent API.
The TypeScript engine owns all game state and rules; this module only makes
authenticated HTTP calls to that API and exposes them as MCP tools.

The API uses Basic authentication. The defaults are:

- username: `admin`
- password: `password`

Override them with `LOSTCITY_API_USERNAME` and `LOSTCITY_API_PASSWORD` when
running outside the local development setup. Set `LOSTCITY_API_URL` when the
engine is not using its configured web port; the default is
`http://127.0.0.1:80/api/v1`.

## Run with uv

```sh
cd Server/mcp
uv sync
uv run fastmcp run server.py
```

FastMCP uses stdio by default, which is the normal mode for MCP hosts. For a
local HTTP MCP endpoint, use `uv run fastmcp run server.py --transport http`.

## Tools

The bridge exposes:

- `world_state`, `list_players`, and `get_player` for world/player state.
- `create_agent`, `attach_agent`, `list_agents`, and `release_agent` for
  headless or already-online players.
- `keep_alive_agent` for a persistent observation/reconnect supervisor. Agent
  creation, attachment, and discovery start it automatically; explicit release
  stops it.
- `observe_agent` for nearby NPCs, players, locations, objects, target state,
  and live interaction options.
- `skill_catalog`, `skill_status`, and `skill_plan` for the complete server
  skill matrix, current XP/levels, prerequisites, nearby actions, and safety
  state.
- `consume_food` for health-aware use of a live `Eat`/`Drink` inventory option,
  plus `item_op` for any validated inventory option.
- `skill_step` for one guarded, option-aware training action.
- `train_skill` for bounded automatic loops that re-observe after every step,
  stop at a target level, and pause while the health watchdog is escaping.
- `quest_catalog`, `quest_plan`, and `run_quest` for declarative quest graphs
  combining travel, discovery-based interactions, combat objectives, training,
  and chat/dialogue actions. The crafting loop supports `count_mode=produced`
  plus a validated `discard_tokens` policy for multi-inventory production
  batches.
- `quest_status` and `progression_plan` for live quest-var inspection and the
  2004Scape guide-backed next-step planner.
- `tutorial_status` for the authoritative tutorial var, milestone, and next
  tutorial objective before ordinary quest routing.
- `tutorial_step` for one validated tutorial action through the mainland
  transition, including live dialogue continuation, discovery-based doors and
  NPCs, interface lessons, combat recovery, and named Wind Strike casting.
  Unsupported tutorial states are reported explicitly.
- `plan_route` for a collision-aware route preview with reachability, final
  position, waypoint, tile-cost, and local-leg metrics.
- `map_window` for a compact authoritative collision cache around the agent.
  It returns a sizable row-packed tile window with blocked/roof bits, cardinal
  exits, and nearby live entities. Supplying `target_x`/`target_z` also returns
  a local route proposal computed from the cached exits. The MCP reuses static
  geometry while the agent remains inside the window and refreshes it after
  interaction types that can change doors or other collision; the proposal is
  advisory and submitted actions remain live-validated.
- `move_agent` for resumable local-leg navigation with dynamic-blocker
  replanning. Route previews accept bounded multi-leg plans, and movement
  requests can queue up to 32 live-validated legs.
- `batch_actions` for bounded multi-action execution. Pass `map_radius` to
  prefetch one map window in the same MCP call; each submitted action is still
  postcondition-validated against the running server. Batched `travel` actions
  use the cached geometry to select live frontier doors and then re-plan and
  validate each door and movement on the engine. Batched `move` actions
  also return a cached route proposal for the requested tile, while the live
  engine independently plans and validates the movement. The batch response
  carries the latest live observation alongside the reused static geometry;
  door/interior interactions force a geometry refresh before the cache is
  reused. `crafting_loop`, `combat_loop`, `recover_health`, `equip_item`, and `drop_item` are
  available in the batch allowlist for resource, combat, and inventory batches.
  `combat_loop` keeps food checks, bounded live combat chunks, and optional
  nearby loot pickup inside one validated call; use `loot_tokens` and
  `loot_count` when the batch should continue until a drop goal is observed,
  and `pickup_tokens` for auxiliary drops such as Coins.
  `recover_health` waits for the live idle-regeneration postcondition before
  allowing a follow-up route or combat batch to proceed.
- `attack_nearest_npc` for option-aware target selection, plus the generic
  NPC/player/location/object interaction tool.
- `chat_agent`, `teleport_agent`, and `stop_agent` for basic actions.
- `use_item`, `use_item_on`, `inventory_button`, `press_button`,
  `choose_dialogue_option`, `resume_dialogue`, and `cast_spell` for the
  inventory/UI primitives required by production skills and quest scripts.
  Dialogue choices are selected from live rendered text and the engine
  confirms the component that was clicked; observations also include the
  engine script execution state so callers can distinguish a paused dialogue
  from an ordinary modal.

The raw engine action contract is explicit and complete: `move`, `chat`,
`teleport`, `item_op`, `use_item`, `use_item_on`, `button`, `resume_dialogue`,
`resume_count_dialog`, `click_side_tab`, `inventory_button`, `cast_spell`,
`close_interface`, `interact`, `stop`, and `set_run` each have one MCP binding.

The TypeScript engine remains authoritative. MCP agent creation waits through
the normal one-tick `agent_pending` activation window before returning a
ready-to-act observation.
The browser client also retries transient connection loss while the current
login session is still active; an explicit logout clears its in-memory
credentials and is never silently reversed.

## Live validation

With the engine running, execute the real-server smoke test with:

```sh
LOSTCITY_LIVE_TEST=1 uv run pytest -q tests/test_live_server.py
```

The test creates a disposable agent, waits for world activation, exercises
accepted and guarded actions, reads authoritative quest state, and releases
the agent. It never grants XP or changes quest variables.

For the primitive action contract and batched control path specifically, run:

```sh
LOSTCITY_LIVE_TEST=1 uv run pytest -q tests/test_live_server.py \
  -k 'all_engine_action_types or batch_actions_prefetch'
```

Those checks create disposable agents, exercise all 16 engine action
discriminators, verify the MCP capability audit reports no missing bindings,
and validate a batched map-prefetched sequence against live postconditions.
Resource-dependent checks may require a real controlled username with the
needed inventory or bank contents; they should report that prerequisite
explicitly rather than treating an unavailable item as an MCP failure.

## Skill automation model

The MCP playbook covers every skill exposed by the engine. Combat, woodcutting,
fishing, mining, agility, thieving, and firemaking can be driven from live
NPC/location or inventory observations today. The other skills are represented
with their expert strategies and prerequisites, but return `needs_context` when
a strategy still needs recipe discovery, banking, or other content-specific
policy. Inventory-on-item, inventory-on-location, equipment-option, and button
primitives are available through the Agent API. The MCP never claims a level-up
from an action that the game did not actually execute.

Every training action checks the TypeScript safety snapshot. When health is at
or below the configured threshold, the engine cancels combat and searches for
an escape route; the MCP pauses the training loop until the danger is cleared.

Questing uses the same guard. The TypeScript API exports the tutorial state,
content-backed quest registry, and persistent quest variables; declarative plans verify
observable objectives (route reached, interaction resolved, combat target
defeated, or skill target reached) and never write quest state directly.
The runner distinguishes resolved steps from engine-confirmed quest completion
and returns the exact step and blocker when a quest needs a content-specific
interaction or an unsupported completion marker.

`run_quest` returns `resume_from`; pass that value back as `start_step` after
the blocker is resolved to continue without replaying earlier steps.

## Session keepalive

The TypeScript controller refreshes attached players every second so the game
server does not classify the API-controlled session as idle. The MCP
`keep_alive_agent` supervisor additionally polls the agent and waits for the
browser client to reconnect if the socket drops. It does not store or invent
game credentials, and an explicit client logout is not silently reversed.
