#!/usr/bin/env python3
"""TraceLab (uw-syfi, CC-BY-4.0) -> the boundary schema (one JSON line per idle period).

tool  : per round, the tool batch the model emitted; idle = first emitted_at -> last result_at
        (the next inference waits for every result). Features: tool names, executables,
        command skeleton, input chars.
human : round r ends with text and round r+1 starts with a user_message:
        idle = user_message time - last event time of round r. The last round of a session is
        right-censored at that user's last observed event. Rounds run by the codex-auto-review model
        are an automated reviewer, so their turn ends are kind workflow (the model is known at idle time).
prog is the first executable that is not a shell builtin (cd, export, source and the like).

Usage: extract_tracelab.py tracelab.duckdb > tracelab.bnd.jsonl
"""
import json, sys
import duckdb
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from extract_transcripts import SKIP, PREFIX

NOT_PROG = sorted(SKIP | PREFIX)

db = duckdb.connect(sys.argv[1], read_only=True)

tool_sql = """
select r.provider, r.user, r.project, r.session_id, r.round_index, r.model,
       r.input_tokens_total + coalesce(r.output_tokens,0) as ctx,
       epoch(min(t.emitted_at)) as s, epoch(max(t.result_at)) as e,
       list(t.tool_name order by t.tool_index) as tools,
       list(coalesce(list_filter(t.executables, x -> x not in ({np}))[1], '') order by t.tool_index) as progs,
       list(coalesce(t.command_skeleton, '') order by t.tool_index) as skel,
       sum(t.input_chars) as arg_len, bool_or(t.is_error) as err
from tool_calls t join rounds r using (round_pk)
where t.emitted_at is not null and t.result_at is not null
group by all
"""
tool_sql = tool_sql.replace("{np}", ", ".join("'" + x + "'" for x in NOT_PROG))
for row in db.execute(tool_sql).fetchall():
    prov, user, proj, sess, ri, model, ctx, s, e, tools, progs, skel, arg_len, err = row
    if s is None or e is None or e < s:
        continue
    print(json.dumps(dict(src="tracelab_" + prov, kind="tool", returned=True, gap_s=e - s, idle_start=s,
                          model=model, session=sess, user=user, project=proj, round=ri,
                          tool=tools[0], tools=tools, prog=progs[0] or None, progs=progs,
                          skeleton=skel[0] or None, arg_len=arg_len, ctx_tokens=ctx, is_error=err,
                          file="tracelab", line_a=f"{sess}:{ri}")))

# human waits: last event of a round -> user_message of the next round in the same session
human_sql = """
with ev as (
  select r.provider, r.user, r.project, r.session_id, r.round_index, r.model,
         r.input_tokens_total + coalesce(r.output_tokens,0) as ctx, r.first_input_event_type,
         epoch(max(te.timestamp)) as last_t,
         epoch(min(case when te.event_type = 'user_message' then te.timestamp end)) as um_t
  from rounds r join timing_events te using (round_pk)
  group by all
), seq as (
  select *, lead(um_t) over w as next_um, lead(first_input_event_type) over w as next_first,
            lead(round_index) over w as next_ri,
            max(last_t) over (partition by user) as user_end
  from ev window w as (partition by provider, session_id order by round_index)
)
select * from seq
"""
cols = None
cur = db.execute(human_sql)
cols = [d[0] for d in cur.description]
for vals in cur.fetchall():
    r = dict(zip(cols, vals))
    base = dict(src="tracelab_" + r["provider"], kind="workflow" if r["model"] == "codex-auto-review" else "human", idle_start=r["last_t"], model=r["model"],
                session=r["session_id"], user=r["user"], project=r["project"], round=r["round_index"],
                ctx_tokens=r["ctx"], file="tracelab", line_a=f"{r['session_id']}:{r['round_index']}")
    if r["last_t"] is None:
        continue
    if r["next_ri"] is None:
        base.update(returned=False, gap_s=max(0.0, r["user_end"] - r["last_t"]))
        print(json.dumps(base))
    elif r["next_first"] == "user_message" and r["next_um"] is not None and r["next_um"] >= r["last_t"]:
        base.update(returned=True, gap_s=r["next_um"] - r["last_t"])
        print(json.dumps(base))
