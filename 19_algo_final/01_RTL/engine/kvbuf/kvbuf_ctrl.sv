/***************************************************************************
* Filename: kvbuf_ctrl.sv  -- DEPRECATED / UNUSED (2026-06-07)
*
* This was a stale copy of the legacy ofbuf_ctrl store/output FSM (6-bank AE
* output buffer). It does NOT match the new passive kvbuf role and the old
* `kvbuf_top` never instantiated it (it instantiated the undefined
* `ofbuf_ctrl`/`ofbuf_sram` with undefined `OFBUF*_` params instead).
*
* The new `kvbuf_top` is a thin TRUE-dual-port pass-through over `kvbuf_sram`
* (presto_dpram), driven directly by the (future) `mem_controller`; there is
* no buffer-local control FSM. KV ring address generation (in_proj store,
* QK all-32-frames read, Score x V top-4 read) belongs to mem_controller
* (Step 2).
*
* The old FSM body is preserved in git history. Delete this file once
* mem_controller lands.
***************************************************************************/
// intentionally empty -- no module
