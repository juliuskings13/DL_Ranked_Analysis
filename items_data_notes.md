# Item and ability data in match_player (checked 2026-10-30)

Checked against live ranked matches with the Deadlock Community API connection.

## Columns
Seven parallel arrays per player row, one entry per event, in time order:
`items.game_time_s`, `items.item_id`, `items.upgrade_id`, `items.sold_time_s`, `items.flags`,
`items.imbued_ability_id`, `items.upgrade_info`.

- About 31 entries per player row.
- Filled for ~99.9% of ranked player rows, demo-processed or not (unlike bans, which only exist on demo-processed matches).

## How to tell the events apart

| Event | upgrade_id | upgrade_info >> 16 | Other fields |
|---|---|---|---|
| Shop item bought | 1 | 1 | item_id = shop item id |
| Ability unlocked | 0 | 1 | item_id = the hero's ability id |
| Ability point spent, tier 1 | large id | 3 | |
| Ability point spent, tier 2 | large id | 7 | |
| Ability point spent, tier 3 | large id | 15 | |

- `upgrade_info >> 16` is a bitmask of the ability's tiers owned after the event. The low 16 bits are always 0.
- Ability order = the upgrade rows (bitmask 3/7/15) sorted by game_time_s, mapped to ability names with list_items(type=ability).

## When an item leaves the inventory
- `sold_time_s > 0` means the item left the inventory at that second.
- `flags = 1` means it was used as a component: it was consumed when a higher-tier item was bought at the same second.
  - Examples: Extra Spirit → Improved Spirit, Extra Regen → Healing Booster, Mystic Burst → Tankbuster, Rapid Rounds → Swift Striker, Extra Charge → Rapid Recharge.
- `flags = 0` with `sold_time_s > 0` means it was sold for souls. That's rare: about 7–12% of removals.
- An item still owned at the end of the match has `sold_time_s = 0`.
- `imbued_ability_id` is set on imbue items (Quicksilver Reload, Duration Extender, Compress Cooldown, …) and names the ability they were imbued on.

## Ingest plan
Aggregate in DuckDB before loading anything into Postgres. Weekly summary tables:
- hero × item × won (bought at all)
- hero × item × purchase-time bucket × won
- hero × ability tier order × won
- optionally hero × final-inventory item × won

Opt-in with a `--with-items` flag.
