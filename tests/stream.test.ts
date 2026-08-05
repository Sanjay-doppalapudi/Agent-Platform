// SSE assembly — previously untested despite being the most provider-variable
// code in the project. Covers the shapes real providers actually emit.
import { describe, expect, test } from "bun:test";
import { consumeSSE, StreamEmptyError } from "../src/stream.ts";

function body(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(c) {
      for (const s of chunks) c.enqueue(enc.encode(s));
      c.close();
    },
  });
}
const sse = (o: unknown) => `data: ${JSON.stringify(o)}\n\n`;
const run = (chunks: string[]) => {
  let text = "";
  return consumeSSE(body(chunks), (d) => { text += d; }).then((r) => ({ ...r, streamed: text }));
};

describe("text assembly", () => {
  test("concatenates content deltas and streams them", async () => {
    const r = await run([
      sse({ choices: [{ delta: { content: "Hel" } }] }),
      sse({ choices: [{ delta: { content: "lo" } }] }),
      "data: [DONE]\n\n",
    ]);
    expect(r.text).toBe("Hello");
    expect(r.streamed).toBe("Hello");
  });

  test("survives a delta split across chunk boundaries", async () => {
    const full = sse({ choices: [{ delta: { content: "split works" } }] });
    const cut = Math.floor(full.length / 2);
    const r = await run([full.slice(0, cut), full.slice(cut)]);
    expect(r.text).toBe("split works");
  });

  test("multi-byte UTF-8 split across chunks is not corrupted", async () => {
    const enc = new TextEncoder();
    const full = enc.encode(sse({ choices: [{ delta: { content: "héllo — 日本語" } }] }));
    const at = 30; // lands inside a multi-byte sequence
    const r = await consumeSSE(
      new ReadableStream({ start(c) { c.enqueue(full.slice(0, at)); c.enqueue(full.slice(at)); c.close(); } }),
      () => {},
    );
    expect(r.text).toBe("héllo — 日本語");
  });

  test("CRLF line endings and comment lines are tolerated", async () => {
    const r = await run([`: ping\r\ndata: ${JSON.stringify({ choices: [{ delta: { content: "ok" } }] })}\r\n\r\n`]);
    expect(r.text).toBe("ok");
  });

  test("a final event with no trailing newline is not lost", async () => {
    // Previously the residual buffer was dropped, silently losing this event.
    const r = await run([`data: ${JSON.stringify({ choices: [{ delta: { content: "tail" }, finish_reason: "stop" }] })}`]);
    expect(r.text).toBe("tail");
    expect(r.finishReason).toBe("stop");
  });

  test("reasoning is surfaced separately and never enters text", async () => {
    let reasoning = "";
    const r = await consumeSSE(
      body([
        sse({ choices: [{ delta: { reasoning_content: "thinking hard" } }] }),
        sse({ choices: [{ delta: { content: "answer" } }] }),
      ]),
      () => {}, undefined, (d) => { reasoning += d; },
    );
    expect(reasoning).toBe("thinking hard");
    expect(r.text).toBe("answer");
  });
});

describe("tool_call assembly", () => {
  test("indexed deltas assemble into distinct calls", async () => {
    const r = await run([
      sse({ choices: [{ delta: { tool_calls: [{ index: 0, id: "a", function: { name: "read", arguments: '{"pa' } }] } }] }),
      sse({ choices: [{ delta: { tool_calls: [{ index: 1, id: "b", function: { name: "grep", arguments: '{"pat' } }] } }] }),
      sse({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: 'th":"a"}' } }] } }] }),
      sse({ choices: [{ delta: { tool_calls: [{ index: 1, function: { arguments: 'tern":"x"}' } }] } }] }),
    ]);
    expect(r.toolCalls.length).toBe(2);
    expect(r.toolCalls[0]!.function).toEqual({ name: "read", arguments: '{"path":"a"}' });
    expect(r.toolCalls[1]!.function).toEqual({ name: "grep", arguments: '{"pattern":"x"}' });
  });

  // REGRESSION: providers that deliver a complete `message` on the last chunk
  // send tool_calls WITHOUT `index`. All of them used to collapse into one
  // call named "readgrep" with concatenated, unparsable arguments.
  test("index-free calls in a final message stay separate", async () => {
    const r = await run([
      sse({
        choices: [{
          finish_reason: "tool_calls",
          message: {
            tool_calls: [
              { id: "call_a", type: "function", function: { name: "read", arguments: '{"path":"a"}' } },
              { id: "call_b", type: "function", function: { name: "grep", arguments: '{"pattern":"x"}' } },
            ],
          },
        }],
      }),
    ]);
    expect(r.toolCalls.map((c) => c.function.name)).toEqual(["read", "grep"]);
    expect(r.toolCalls.map((c) => c.id)).toEqual(["call_a", "call_b"]);
  });

  // The companion guard: index-free CONTINUATION fragments must NOT start new
  // slots, or one call would shatter into one call per fragment.
  test("index-free fragments continue the current call", async () => {
    const r = await run([
      sse({ choices: [{ delta: { tool_calls: [{ id: "solo", function: { name: "read", arguments: '{"pa' } }] } }] }),
      sse({ choices: [{ delta: { tool_calls: [{ function: { arguments: 'th":' } }] } }] }),
      sse({ choices: [{ delta: { tool_calls: [{ function: { arguments: '"x.ts"}' } }] } }] }),
    ]);
    expect(r.toolCalls.length).toBe(1);
    expect(r.toolCalls[0]!.function.arguments).toBe('{"path":"x.ts"}');
  });

  test("mixing indexed deltas with a later index-free call does not clobber", async () => {
    const r = await run([
      sse({ choices: [{ delta: { tool_calls: [{ index: 0, id: "i0", function: { name: "read", arguments: "{}" } }] } }] }),
      sse({ choices: [{ delta: { tool_calls: [{ index: 1, id: "i1", function: { name: "glob", arguments: "{}" } }] } }] }),
      sse({ choices: [{ message: { tool_calls: [{ id: "m2", function: { name: "grep", arguments: "{}" } }] } }] }),
    ]);
    expect(r.toolCalls.map((c) => c.function.name)).toEqual(["read", "glob", "grep"]);
  });
});

describe("empty / non-SSE bodies", () => {
  // REGRESSION: these used to return a SUCCESSFUL empty turn, writing
  // {"role":"assistant","content":null} into the session and exiting 0.
  test("a body with zero events throws", async () => {
    await expect(run([])).rejects.toBeInstanceOf(StreamEmptyError);
  });

  test("a plain JSON body (proxy ignored stream:true) throws", async () => {
    await expect(run(['{"choices":[{"message":{"content":"hi"}}]}'])).rejects.toBeInstanceOf(StreamEmptyError);
  });

  test("only comments/keepalives throws", async () => {
    await expect(run([": ping\n\n: ping\n\n"])).rejects.toBeInstanceOf(StreamEmptyError);
  });

  test("a [DONE]-only stream is a valid (empty) answer, not an error", async () => {
    const r = await run(["data: [DONE]\n\n"]);
    expect(r.text).toBe("");
    expect(r.toolCalls).toEqual([]);
  });
});
