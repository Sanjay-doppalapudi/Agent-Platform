import { describe, expect, test } from "bun:test";
import { streamChat, type ToolSchema } from "../src/provider.ts";
import type { ResolvedProvider } from "../src/config.ts";

const provider = (name: string): ResolvedProvider => ({
  name,
  baseUrl: `https://${name}.test/v1`,
  apiKey: "key",
  model: "model",
  cacheControl: false,
  headers: {},
});
const tools: ToolSchema[] = [{ type: "function", function: { name: "read", description: "", parameters: {} } }];
const response = (body: string, status = 200) => new Response(body, {
  status,
  headers: { "content-type": "text/event-stream" },
});

describe("provider request safety", () => {
  test("extras cannot override protected request fields", async () => {
    const original = globalThis.fetch;
    let request: any;
    globalThis.fetch = (async (_url, init) => {
      request = JSON.parse(String(init?.body));
      return response("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n");
    }) as typeof fetch;
    try {
      await streamChat(provider("safe"), [{ role: "user", content: "hello" }], tools, () => {}, undefined, {
        model: "attacker-model",
        messages: [],
        stream: false,
        tools: [],
        reasoning_effort: "high",
      });
    } finally {
      globalThis.fetch = original;
    }
    expect(request.model).toBe("model");
    expect(request.messages).toEqual([{ role: "user", content: "hello" }]);
    expect(request.stream).toBe(true);
    expect(request.tools).toEqual(tools);
    expect(request.reasoning_effort).toBe("high");
  });

  test("stream-options compatibility is isolated per provider", async () => {
    const original = globalThis.fetch;
    const seen: { name: string; body: any }[] = [];
    globalThis.fetch = (async (url, init) => {
      const name = new URL(String(url)).hostname.split(".")[0]!;
      const body = JSON.parse(String(init?.body));
      seen.push({ name, body });
      if (name === "reject" && body.stream_options) return response("stream_options unsupported", 400);
      return response("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n");
    }) as typeof fetch;
    try {
      await streamChat(provider("reject"), [{ role: "user", content: "hello" }], [], () => {});
      await streamChat(provider("accept"), [{ role: "user", content: "hello" }], [], () => {});
      await streamChat(provider("reject"), [{ role: "user", content: "hello" }], [], () => {});
    } finally {
      globalThis.fetch = original;
    }
    expect(seen[0]!.body.stream_options).toBeDefined();
    expect(seen[1]!.body.stream_options).toBeUndefined();
    expect(seen[2]!.body.stream_options).toBeDefined();
    expect(seen[3]!.body.stream_options).toBeUndefined();
  });
});
