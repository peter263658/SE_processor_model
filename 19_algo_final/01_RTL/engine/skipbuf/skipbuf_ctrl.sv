/***************************************************************************
* Filename: skipbuf_ctrl.sv  -- DEPRECATED / UNUSED (2026-06-07)
*
* This was a stale copy of the legacy wbuf_ctrl DMA-load FSM: it declared a
* wbuf_ctrl-shaped load sequencer, imported `MEM_def` with the wrong case
* (correct package is `Mem_def`), and used the undefined `WBUF2_ADDRW` /
* `WBUF2_LAST_ADDR` params -- it could not elaborate and does NOT match the
* new passive skipbuf role.
*
* The new `skipbuf_top` is a thin TRUE-dual-port pass-through over
* `skipbuf_sram` (presto_dpram), driven directly by the (future)
* `mem_controller`; there is no buffer-local control FSM. Skip address
* sequencing (static top-50% channel gather, groups_skip=2) belongs to
* mem_controller (Step 2).
*
* The old FSM body is preserved in git history. Delete this file once
* mem_controller lands.
***************************************************************************/
// intentionally empty -- no module
